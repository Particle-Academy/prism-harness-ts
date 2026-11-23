import type { JsonObject } from './json.js';
import type { PrismHarness } from './harness.js';
import type { AgentRuntime } from './runtime.js';
import type { Session } from './session.js';
import { MAX_DEPTH, RunContext, type ModelBinding, type Subagent } from './subagents.js';
import type { HarnessTool } from './tools.js';
import { HarnessError } from './errors.js';
import { markRefused } from './refusals.js';

export type SubagentOutcome =
  /** It ran and returned. */
  | 'completed'
  /** The tree had nothing left to spend. */
  | 'exhausted'
  /** The tree was stopped. */
  | 'cancelled'
  /** Refused before it began — depth, most often. */
  | 'denied'
  /** It ran and threw. */
  | 'failed';

export interface SubagentResult {
  subagent: string;
  runId: string;
  parentRunId: string;
  outcome: SubagentOutcome;
  text: string;
  /** What to hand back to the model as the tool's result. */
  toToolResult(): string;
}

/**
 * Turns a declared `Subagent` into a TOOL the parent run can call.
 *
 * The parent is mid-run and holding its own session lock when this executes.
 * Everything here is arranged so that fact stays harmless:
 *
 *  - the child resolves its OWN session address, so it takes a different lock;
 *  - the child's authority comes from its declared mode, never the parent's;
 *  - the child's budget is drawn from the tree's remaining allowance;
 *  - EVERY way the child can end returns a framed result rather than throwing,
 *    because the parent is a legitimate audience for "that did not work" and
 *    tearing down the parent run would discard work it had already done.
 */
export class SubagentRunner {
  constructor(
    private readonly harness: PrismHarness,
    private readonly runtime: AgentRuntime,
  ) {}

  tool(subagent: Subagent, parent: Session, context: RunContext, parentRunId: string): HarnessTool {
    return {
      name: subagent.name,
      description: subagent.description,
      handle: async (args: JsonObject): Promise<string> => {
        if (parent.agent !== null && Object.keys(args).some(key => key !== 'task')) {
          throw HarnessError.runNotPermitted('Named subagent input may contain only task. Identity changes require an explicit host-owned delegation binding.');
        }
        const task = typeof args.task === 'string' ? args.task : '';

        const result = await this.run(subagent, parent, context, parentRunId, task);

        // The tree refusing to let a child run is a gate, not the child's
        // failure: said here, so a run step reports it as one.
        if (result.outcome === 'exhausted' || result.outcome === 'cancelled' || result.outcome === 'denied') markRefused();

        return result.toToolResult();
      },
    };
  }

  async run(
    subagent: Subagent,
    parent: Session,
    context: RunContext,
    parentRunId: string,
    task: string,
  ): Promise<SubagentResult> {
    const runId = `run_${Math.random().toString(16).slice(2, 14)}`;

    // Checked BEFORE spawning, so an exhausted tree does not pay for a session,
    // a thread write and a provider call to discover it is exhausted.
    const stop = context.ledger.exhaustion(context.budget);

    if (stop !== null) {
      return refused(
        subagent.name,
        runId,
        parentRunId,
        context.ledger.cancelled ? 'cancelled' : 'exhausted',
        stop,
      );
    }

    // The child runs as the agent its declaration names (which the parent mode
    // had to list), else as the parent's.
    const agent = subagent.agent ?? parent.agent;
    let binding: ModelBinding;
    try {
      binding = await this.#binding(subagent, parent, agent);
    } catch (error) {
      return refused(subagent.name, runId, parentRunId, 'failed', error instanceof Error ? error.message : String(error));
    }
    const childContext = context.forChild(subagent, parentRunId, binding);

    // Refused BEFORE the child's address is built. Two modes naming each other
    // as subagents form a cycle budgets would eventually stop — but only after
    // each level had appended `::sub::<name>` to an address that may truncate
    // rather than error, and two children truncated to the same string are one
    // conversation.
    if (childContext.tooDeep()) {
      return refused(
        subagent.name,
        runId,
        parentRunId,
        'denied',
        `subagent nesting reached the maximum depth of ${MAX_DEPTH}`,
      );
    }

    const child = this.harness
      .pending(parent.participant, agent, parent.actor)
      .session(subagent.scopeUnder(parent.scope));

    // The child's authority comes from ITS mode, not the parent's. Its
    // provider and model come from the binding on its context: the parent's
    // session override is NOT copied onto the child session (G-321 E).
    await child.usingMode(subagent.mode);

    try {
      const response = await this.runtime.send(child, task, undefined, childContext);

      return {
        subagent: subagent.name,
        runId: response.runId,
        parentRunId,
        outcome: response.stoppedBecause === null ? 'completed' : 'exhausted',
        text: response.text,
        toToolResult(): string {
          return response.stoppedBecause === null
            ? response.text
            : `The [${subagent.name}] subagent stopped early: ${response.stoppedBecause}. Partial answer: ${response.text}`;
        },
      };
    } catch (error) {
      // Framed, not thrown. Tearing down the parent run would discard work it
      // has already done, and "that subagent failed" is something the parent
      // can act on. A child agent stopped by its own spend ceiling is the tree
      // refusing to let it spend, not the child breaking: reported as such, so
      // the parent does not retry it.
      const ceiling = error instanceof HarnessError
        && (error.code === 'agent_spend_ceiling_exceeded' || error.code === 'budget_unpriced');

      return refused(
        subagent.name,
        runId,
        parentRunId,
        ceiling ? 'exhausted' : 'failed',
        error instanceof Error ? error.message : String(error),
      );
    }
  }

  /** What the child resolves its provider and model from; see `ModelBinding`. */
  async #binding(subagent: Subagent, parent: Session, agent: string | null): Promise<ModelBinding> {
    const inherit = subagent.inheritSessionModel;
    const sameAgent = agent === parent.agent;
    // A caller may run a subagent beneath a session that is not itself in a
    // configured mode (no run of its own). Then there is no parent mode to
    // fall back to; any OTHER failure to resolve it is still a failure.
    const parentMode = sameAgent
      ? await this.runtime.modeOf(parent).catch((error: unknown) => {
        if (error instanceof HarnessError && error.code === 'mode_not_configured') return null;
        throw error;
      })
      : null;

    return {
      session: {
        provider: inherit ? await parent.provider() : null,
        model: inherit ? await parent.model() : null,
      },
      declared: { provider: subagent.provider, model: subagent.model },
      // The parent mode's binding belongs to the parent's agent; it does not
      // cross into a delegated agent, whose own default applies instead.
      parentMode: { provider: parentMode?.provider ?? null, model: parentMode?.model ?? null },
    };
  }
}

function refused(
  subagent: string,
  runId: string,
  parentRunId: string,
  outcome: SubagentOutcome,
  reason: string,
): SubagentResult {
  const text = `The [${subagent}] subagent did not run: ${reason}.`;

  return {
    subagent,
    runId,
    parentRunId,
    outcome,
    text,
    toToolResult: () => text,
  };
}
