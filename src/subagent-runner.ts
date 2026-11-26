import type { JsonObject } from './json.js';
import type { PrismHarness } from './harness.js';
import { activeTurn, decisionsAnswerable, type AgentRuntime } from './runtime.js';
import type { Session } from './session.js';
import { MAX_DEPTH, RunBudget, RunContext, type ModelBinding, type Subagent } from './subagents.js';
import type { HarnessTool } from './tools.js';
import { HarnessError } from './errors.js';
import { markRefused } from './refusals.js';
import { RunTrees } from './run-trees.js';
import { pendingApprovalIds } from './thread-rows.js';

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
        const task = typeof args.task === 'string' ? args.task : '';

        const turn = activeTurn(parent.key());
        const result = await this.run(subagent, parent, turn === null ? context : narrowedTo(turn.run, context), turn?.runId ?? parentRunId, task);

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

    // A delegation with no task is not one: an empty prompt on the child is
    // how a parked child is RESUMED, so the model could otherwise use it to
    // act on a person's decisions on the child's behalf. A task arrives from
    // the model's tool arguments, so it is checked at runtime: anything that
    // is not text, or is blank by the one codepoint set every port spells
    // out (isBlankTask), is refused before the child session is built.
    const given: unknown = task;
    if (typeof given !== 'string') return refused(subagent.name, runId, parentRunId, 'denied', 'the task is not text');
    if (isBlankTask(given)) return refused(subagent.name, runId, parentRunId, 'denied', 'the task is empty');

    // EVERY way the child can fail to start or to finish, a store or host
    // failure included, is framed here. The parent's model reads the result,
    // so it carries the failure's class name and never its message: a
    // store's or a host's own words do not reach a model.
    try {
      return await this.#delegate(subagent, parent, context, parentRunId, given, runId);
    } catch (error) {
      // A child agent stopped by its own spend ceiling is the tree refusing
      // to let it spend, not the child breaking: reported as such, so the
      // parent does not retry it.
      const ceiling = error instanceof HarnessError
        && (error.code === 'agent_spend_ceiling_exceeded' || error.code === 'budget_unpriced');

      return refused(subagent.name, runId, parentRunId, ceiling ? 'exhausted' : 'failed', `it failed (${failureName(error)})`);
    }
  }

  async #delegate(subagent: Subagent, parent: Session, context: RunContext, parentRunId: string, task: string, runId: string): Promise<SubagentResult> {
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
    const binding = await this.#binding(subagent, parent, agent);
    // Marked as the runner's own, with the run whose turn is calling it (never
    // a run id the caller states): the child joins that run's tree only when
    // `context` is that turn's own (its ledger), G-381.
    const turn = activeTurn(parent.key());
    const childContext = RunTrees.startingChild(context.forChild(subagent, parentRunId, binding), turn?.runId ?? null, parent.key());

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

    // A child a person can still answer is theirs until they do: a new task
    // would end the approvals it is waiting on, so the model could dismiss a
    // person's approval by delegating again. An approval nobody could answer
    // (its binding is gone, it is too deep) blocks nothing: the parent may
    // start the child again. Read only, with nothing written to the child.
    if (pendingApprovalIds((await child.thread().messages()).map((entry) => entry.message)).length > 0
      && await decisionsAnswerable(child)) {
      return refused(subagent.name, runId, parentRunId, 'denied', 'it is waiting on a person to answer its approval');
    }
    // A turn on the child that is still running holds part of its
    // delegation's step budget; starting the child again would reset that
    // count under it.
    if (await child.childTurnLive(Math.floor(Date.now() / 1000), this.runtime.holdTtl(child))) {
      return refused(subagent.name, runId, parentRunId, 'denied', 'a turn on it is still running');
    }

    // The child's authority comes from ITS mode, not the parent's. Its
    // provider and model come from the binding on its context: the parent's
    // session override is NOT copied onto the child session (G-321 E).
    await child.usingMode(subagent.mode);

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

/**
 * The codepoints a blank task consists of, spelled as an explicit set (the
 * G-36 pattern): identical in every port, never a language's own trim(),
 * whose idea of whitespace differs between languages and versions.
 * U+0009-U+000D, U+0020, U+0085, U+00A0, U+1680, U+2000-U+200D, U+2028,
 * U+2029, U+202F, U+205F, U+2060, U+3000, U+FEFF.
 */
const BLANK_CODEPOINTS: ReadonlySet<number> = new Set([
  0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x20, 0x85, 0xa0, 0x1680,
  0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a, 0x200b, 0x200c, 0x200d,
  0x2028, 0x2029, 0x202f, 0x205f, 0x2060, 0x3000, 0xfeff,
]);

/** True when `task` is empty or consists only of BLANK_CODEPOINTS. */
export function isBlankTask(task: string): boolean {
  for (const character of task) {
    if (!BLANK_CODEPOINTS.has(character.codePointAt(0)!)) return false;
  }

  return true;
}

/**
 * What a framed failure says of its cause: the error's class name, and for a
 * HarnessError its code (a fixed identifier the harness defines). Never the
 * message: a store's or a host's own words do not reach a model.
 */
function failureName(error: unknown): string {
  if (error instanceof HarnessError) return `HarnessError ${error.code}`;

  return error instanceof Error ? error.constructor.name : typeof error;
}
