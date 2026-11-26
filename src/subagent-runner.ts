import type { JsonObject } from './json.js';
import type { PrismHarness } from './harness.js';
import { activeTurn, stopCodeOf, type AgentRuntime } from './runtime.js';
import type { Session } from './session.js';
import { MAX_DEPTH, RunBudget, RunContext, runStopText, type ModelBinding, type Subagent } from './subagents.js';
import type { HarnessTool } from './tools.js';
import { HarnessError } from './errors.js';
import { errorCode, failureName } from './failures.js';
import { markRefused } from './refusals.js';
import { RunTrees } from './run-trees.js';
import { RunStarts, StartRefused, startRefusal, startRefusalText } from './run-starts.js';

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

  /**
   * Called inside a turn on `parent`, the child runs in THAT TURN's run tree
   * and spends from its ledger: the turn's budget and, for a plan step, the
   * plan's shared budget. `context` can then only narrow the turn's limits,
   * by what it has left; it never widens them and its own ledger is not
   * charged. Called outside a turn (the host invoking the tool itself),
   * `context` is the run tree, as before.
   */
  tool(subagent: Subagent, parent: Session, context: RunContext, parentRunId: string): HarnessTool {
    return {
      name: subagent.name,
      description: subagent.description,
      handle: async (args: JsonObject): Promise<string> => {
        if (parent.agent !== null && Object.keys(args).some(key => key !== 'task')) {
          throw HarnessError.runNotPermitted('Named subagent input may contain only task. Identity changes require an explicit host-owned delegation binding.');
        }
        // As it arrived: the start's one gate decides what a task that is
        // not text, or is blank, means (G-387 round 7).
        const task: unknown = args.task;

        const turn = activeTurn(parent.key());
        let result: SubagentResult;
        try {
          // Inside a turn the child spends from the turn, narrowed by what
          // the host context this tool was wired with has left (pure). That
          // host context's cancel, time, steps and cost are facts of the
          // start's one gate, never a separate refusal here.
          result = await this.#run(subagent, parent, turn === null ? context : narrowedTo(turn.run, context), turn?.runId ?? parentRunId, task, turn === null ? null : context);
        } catch (error) {
          result = refused(subagent.name, `run_${Math.random().toString(16).slice(2, 14)}`, turn?.runId ?? parentRunId, 'failed', `it failed (${failureName(error)})`);
        }

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
    return this.#run(subagent, parent, context, parentRunId, task, null);
  }

  /**
   * A runner start. Every way it can be refused is ONE gate (RunStarts.gate,
   * G-387 round 7), asked by the runtime before the start writes anything:
   * the task (not text, or blank: an empty prompt on the child is how a
   * parked child is RESUMED), depth, the cancel, time, steps and cost of the
   * context the child would run under and of the tool's host context, a turn
   * on the child still running, an approval on it a person can still answer,
   * the run tree and the plan. A refusal comes back as StartRefused, framed
   * here with the harness's own words; nothing was written.
   *
   * EVERY other way the child can fail to start or to finish, a store or
   * host failure included, is framed too. The parent's model reads the
   * result, so it carries the failure's class name and never its message.
   */
  async #run(subagent: Subagent, parent: Session, context: RunContext, parentRunId: string, task: unknown, host: RunContext | null): Promise<SubagentResult> {
    const runId = `run_${Math.random().toString(16).slice(2, 14)}`;

    try {
      return await this.#delegate(subagent, parent, context, parentRunId, task, host);
    } catch (error) {
      // Only harness-written words, chosen by code (G-387 round 8): the
      // gate's reasons, the fixed stops below, and for anything else the
      // class name and harness code. Never an error's message, and never an
      // outcome a thrown object states.
      if (error instanceof StartRefused) {
        const refusal = startRefusal(error);
        return refusal === null
          ? refused(subagent.name, runId, parentRunId, 'failed', `it failed (${failureName(error)})`)
          : refused(subagent.name, runId, parentRunId, refusal.outcome, startRefusalText(refusal.code, MAX_DEPTH));
      }
      // A child agent stopped by its own spend ceiling is the tree refusing
      // to let it spend, not the child breaking: reported as such, so the
      // parent does not retry it. The code is read ONCE, guarded.
      const code = errorCode(error);
      if (code === 'agent_spend_ceiling_exceeded') return refused(subagent.name, runId, parentRunId, 'exhausted', "its agent's spend ceiling is reached");
      if (code === 'budget_unpriced') return refused(subagent.name, runId, parentRunId, 'exhausted', "its agent's spend cannot be metered (a cost was not reported)");

      return refused(subagent.name, runId, parentRunId, 'failed', `it failed (${failureName(error, code)})`);
    }
  }

  async #delegate(subagent: Subagent, parent: Session, context: RunContext, parentRunId: string, task: unknown, host: RunContext | null): Promise<SubagentResult> {
    // The child runs as the agent its declaration names (which the parent mode
    // had to list), else as the parent's.
    const agent = subagent.agent ?? parent.agent;
    const binding = await this.#binding(subagent, parent, agent);
    // Marked as the runner's own, with the run whose turn is calling it (never
    // a run id the caller states): the child joins that run's tree only when
    // `context` is that turn's own (its ledger), G-381. The mode it runs in and
    // the host context are read by the start's gate; the mode is written on
    // the child only once the start commits.
    const turn = activeTurn(parent.key());
    const childContext = RunStarts.starting(
      RunTrees.startingChild(context.forChild(subagent, parentRunId, binding), turn?.runId ?? null, parent.key()),
      subagent.mode,
      host,
    );

    // Built, never written: a refused start reads the child and leaves it as it was.
    const child = this.harness
      .pending(parent.participant, agent, parent.actor)
      .session(subagent.scopeUnder(parent.scope));

    const response = await this.runtime.send(child, task as string, undefined, childContext);
    // Why it stopped, as the harness's fixed words for the stop's code:
    // `stoppedBecause` carries a host's own cancel words, for the host.
    const stop = response.stoppedBecause === null ? null : stopCodeOf(response);

    return {
      subagent: subagent.name,
      runId: response.runId,
      parentRunId,
      outcome: response.stoppedBecause === null ? 'completed' : 'exhausted',
      text: response.text,
      toToolResult(): string {
        return response.stoppedBecause === null
          ? response.text
          : `The [${subagent.name}] subagent stopped early: ${stop === null ? 'its run was stopped' : runStopText(stop)}. Partial answer: ${response.text}`;
      },
    };
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

/**
 * The turn's run context, narrowed by what a host-supplied context has left.
 * The turn's ledger is kept: the child spends from the turn, never from the
 * host's tree.
 */
function narrowedTo(turn: RunContext, host: RunContext): RunContext {
  if (host.metadataOnly || host === turn) return turn;
  const lower = (a: number | null, b: number | null): number | null => a === null ? b : b === null ? a : Math.min(a, b);
  const steps = Math.min(turn.budget.maxSteps, turn.ledger.steps + Math.max(0, host.budget.maxSteps - host.ledger.steps));
  const cost = host.budget.maxCostUsd === null ? turn.budget.maxCostUsd
    : lower(turn.budget.maxCostUsd, turn.ledger.costUsd + Math.max(0, host.budget.maxCostUsd - host.ledger.costUsd));

  return turn.withBudget(new RunBudget(steps, cost, lower(turn.budget.maxSeconds, host.budget.maxSeconds),
    lower(turn.budget.maxOutputTokensPerCall, host.budget.maxOutputTokensPerCall)));
}

/** True when `task` is empty or only blank codepoints (RunStarts' one set). */
export { isBlankTask } from './run-starts.js';
