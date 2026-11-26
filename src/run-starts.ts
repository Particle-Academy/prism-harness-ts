import { HarnessError } from './errors.js';
import type { RunContext } from './subagents.js';

/**
 * What a runner start (the subagent runner starting a child) is decided
 * from (G-387 round 7). Every fact is READ before the start writes anything;
 * the gate is asked once; only when it says start does the start commit.
 *
 * - `task`: the task as it arrived (from the model's tool arguments), raw.
 * - `tooDeep`: the child would run at or past MAX_DEPTH.
 * - `cancelled` / `timeSpent`: of the context the child would run under and
 *   of the host context the subagent tool was wired with.
 * - `stepsLeft`: what the start's own allowance would give
 *   (RunBudgets.allowance for kind runner), and no more than the host
 *   context has left.
 * - `cost`: `spent` when any cost limit that would stop the child is spent
 *   (a context's cost budget, the child agent's spend ceiling);
 *   `unenforceable` when either cannot be metered; else `ok`.
 * - `childBusy`: a turn on the child holds part of its delegation's count.
 * - `childPending`: `answerable` when a decision on the child's pending
 *   approval would be accepted, `unanswerable` when it is pending but nobody
 *   could answer it, `none` when nothing is pending.
 * - `treeLeft` / `planLeft`: the grants the tree and plan records would give
 *   this start now, read without a write; null when none applies.
 */
export interface StartFacts {
  task: unknown;
  tooDeep: unknown;
  cancelled: unknown;
  timeSpent: unknown;
  stepsLeft: unknown;
  cost: unknown;
  childBusy: unknown;
  childPending: unknown;
  treeLeft: unknown;
  planLeft: unknown;
}

export type StartCode =
  | 'task_invalid' | 'task_blank' | 'too_deep' | 'cancelled' | 'time' | 'steps' | 'cost' | 'cost_unenforceable'
  | 'child_busy' | 'child_waiting' | 'tree' | 'plan';

export type StartOutcome = 'denied' | 'cancelled' | 'exhausted';

/** The gate's answer, in the corpus's key names (harness-plans start_gate). */
export type StartGate = { start: { steps: number } } | { refused: StartCode; outcome: StartOutcome };

/**
 * The codepoints a blank task consists of, spelled as an explicit set (the
 * G-36 pattern): identical in every port, never a language's own trim().
 * U+0009-U+000D, U+0020, U+0085, U+00A0, U+1680, U+2000-U+200D, U+2028,
 * U+2029, U+202F, U+205F, U+2060, U+3000, U+FEFF.
 */
const BLANK_CODEPOINTS: ReadonlySet<number> = new Set([
  0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x20, 0x85, 0xa0, 0x1680,
  0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a, 0x200b, 0x200c, 0x200d,
  0x2028, 0x2029, 0x202f, 0x205f, 0x2060, 0x3000, 0xfeff,
]);

/** True when `task` is empty or consists only of the blank codepoints. */
export function isBlankTask(task: string): boolean {
  for (const character of task) {
    if (!BLANK_CODEPOINTS.has(character.codePointAt(0)!)) return false;
  }

  return true;
}

const COST: readonly unknown[] = ['ok', 'spent', 'unenforceable'];
const PENDING: readonly unknown[] = ['none', 'answerable', 'unanswerable'];

/** A positive whole count, else 0: a count that cannot be read never opens the gate. */
function count(value: unknown): number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value > 0 ? value : 0;
}

/** What the start is decided by, before it writes, for the code that refuses it. */
const START_MARKS = new WeakMap<RunContext, { mode: string | null; host: RunContext | null }>();

/**
 * One gate for a runner start (G-387 round 7), pure, and the marks the
 * subagent runner puts on the context it starts a child under.
 *
 * @internal Used by the runtime and the subagent runner; pinned in every language by the harness-plans corpus.
 */
export const RunStarts = {
  /**
   * The one decision a runner start makes before it writes anything, asked
   * in a fixed order so every language gives the same reason: the task,
   * depth, cancel, time, steps, cost, an unmeterable cost cap, a busy child,
   * a waiting child, the tree, the plan. On start, the steps are the least
   * of what the start's allowance, the tree and the plan grant.
   *
   * @throws RangeError for a cost or pending state it does not know
   */
  gate(facts: StartFacts): StartGate {
    if (!COST.includes(facts.cost) || !PENDING.includes(facts.childPending)) throw new RangeError('Unknown cost or pending state.');
    const refuse = (refused: StartCode, outcome: StartOutcome): StartGate => ({ refused, outcome });
    if (typeof facts.task !== 'string') return refuse('task_invalid', 'denied');
    if (isBlankTask(facts.task)) return refuse('task_blank', 'denied');
    if (facts.tooDeep === true) return refuse('too_deep', 'denied');
    if (facts.cancelled === true) return refuse('cancelled', 'cancelled');
    if (facts.timeSpent === true) return refuse('time', 'exhausted');
    if (count(facts.stepsLeft) < 1) return refuse('steps', 'exhausted');
    if (facts.cost === 'spent') return refuse('cost', 'exhausted');
    if (facts.cost === 'unenforceable') return refuse('cost_unenforceable', 'exhausted');
    if (facts.childBusy === true) return refuse('child_busy', 'denied');
    if (facts.childPending === 'answerable') return refuse('child_waiting', 'denied');
    if (facts.treeLeft !== null && count(facts.treeLeft) < 1) return refuse('tree', 'exhausted');
    if (facts.planLeft !== null && count(facts.planLeft) < 1) return refuse('plan', 'exhausted');
    let steps = count(facts.stepsLeft);
    if (facts.treeLeft !== null) steps = Math.min(steps, count(facts.treeLeft));
    if (facts.planLeft !== null) steps = Math.min(steps, count(facts.planLeft));

    return { start: { steps } };
  },

  /**
   * The runner is starting a child under `context`: the mode it runs in
   * (written only once the start commits) and the host context the subagent
   * tool was wired with, whose stops the gate also reads.
   */
  starting(context: RunContext, mode: string | null, host: RunContext | null): RunContext {
    START_MARKS.set(context, { mode, host });

    return context;
  },

  /** What the runner marked on `context`, or a mark with no mode and no host. */
  startOf(context: RunContext): { mode: string | null; host: RunContext | null } {
    return START_MARKS.get(context) ?? { mode: null, host: null };
  },
};

/**
 * A runner start the gate (or a commit race) refused, with nothing written.
 * The subagent runner frames it for the parent's model; its text is the
 * harness's own.
 *
 * @internal Thrown by the runtime, caught by the subagent runner.
 */
export class StartRefused extends HarnessError {
  constructor(readonly refused: StartCode, readonly outcome: StartOutcome) {
    super('run_not_permitted', `This subagent start was refused (${refused}).`);
  }
}

/** What the parent's model is told of a refused start: harness-written, never a host's or a store's words. */
export function startRefusalText(code: StartCode, maxDepth: number): string {
  switch (code) {
    case 'task_invalid': return 'the task is not text';
    case 'task_blank': return 'the task is empty';
    case 'too_deep': return `subagent nesting reached the maximum depth of ${maxDepth}`;
    case 'cancelled': return 'its run was cancelled';
    case 'time': return 'its run has no time left';
    case 'steps': return 'its run has no steps left';
    case 'cost': return "its run's cost budget is spent";
    case 'cost_unenforceable': return "its run's cost budget cannot be enforced (a cost was not reported)";
    case 'child_busy': return 'a turn on it is still running';
    case 'child_waiting': return 'it is waiting on a person to answer its approval';
    case 'tree': return "its run tree's budget is spent";
    case 'plan': return "its plan's shared budget is spent";
  }
}
