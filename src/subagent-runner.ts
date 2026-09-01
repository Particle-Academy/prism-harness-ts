import type { JsonObject } from './json.js';
import type { PrismHarness } from './harness.js';
import type { AgentRuntime } from './runtime.js';
import type { Session } from './session.js';
import { MAX_DEPTH, RunContext, type Subagent } from './subagents.js';
import type { HarnessTool } from './tools.js';

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
        const task = typeof args.task === 'string' ? args.task : '';

        return (await this.run(subagent, parent, context, parentRunId, task)).toToolResult();
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

    const childContext = context.forChild(subagent, parentRunId);

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
      .for(parent.participant)
      .session(subagent.scopeUnder(parent.scope));

    // The child's authority comes from ITS mode, not the parent's.
    await child.usingMode(subagent.mode);
    await child.usingProvider((await parent.provider()) ?? 'unknown');
    await child.usingModel((await parent.model()) ?? 'unknown');

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
      // can act on.
      return refused(
        subagent.name,
        runId,
        parentRunId,
        'failed',
        error instanceof Error ? error.message : String(error),
      );
    }
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
