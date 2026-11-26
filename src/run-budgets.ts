/**
 * What decides a run's step allowance (G-387), computed ONCE when the run
 * begins, beside its run-tree membership (RunTrees.membership), and recorded
 * with it. Every consumer reads that record or the budget the run was
 * narrowed to with it: the loop's step gate, the row the run parks on, the
 * subagent runner's pre-check, the tree and plan holds and the decision
 * checks. None of them computes "steps left" from a source of its own.
 *
 * - `kind`: the context the turn was handed (RunTrees.contextKind).
 * - `modeSteps`: the resolved mode's max_steps, for a run with neither a
 *   saved ceiling nor a context.
 * - `savedCeiling`: the ceiling saved with a subagent child's binding when
 *   the runner started it, re-applied on every turn the runner did not
 *   start; null for a session the runner never started.
 * - `contextLeft`: what a runner or host context has left
 *   (`budget.maxSteps - ledger.steps`); null for none and metadata.
 * - `stepsCeiling`: the max_steps_ceiling setting, absolute on the run's
 *   whole count under every context.
 * - `used`: the steps the parked run this turn resumes had used, its
 *   children's included (the `harness_steps_used` its row wrote); a runner
 *   start is always a new run, so 0.
 * - `treeLeft` / `planLeft`: the run tree and plan grants, or null when no
 *   durable tree or plan step budget bounds the run. A spent one is 0, never
 *   a refusal: acting on recorded decisions costs no model step.
 * - `took`: the steps the run has taken since it began, for what it writes
 *   when it parks.
 */
export interface AllowanceFacts {
  kind: string;
  modeSteps: number;
  savedCeiling: number | null;
  contextLeft: number | null;
  stepsCeiling: number | null;
  used: number;
  treeLeft: number | null;
  planLeft: number | null;
  took: number;
}

/** A run's step allowance, in the corpus's key names and order (harness-plans step_allowance). */
export interface StepAllowance {
  ceiling: number;
  used: number;
  own_left: number;
  tree_left: number | null;
  plan_left: number | null;
  /** The least of own_left, tree_left and plan_left: the model steps this run may take. */
  allowance: number;
  /** No model step is left; the run still acts on the decisions it resumes. */
  acts_without_step: boolean;
  /** What the run writes on the row it parks on as `harness_steps_used`. */
  used_at_park: number;
  /** What it writes there as `harness_steps_left`; never read to decide a budget. */
  left_at_park: number;
}

const KINDS: readonly string[] = ['none', 'metadata', 'runner', 'host'];

/** A whole number of steps, clamped to 0; null stays null. */
function steps(value: number | null | undefined): number | null {
  return value === null || value === undefined ? null : Math.max(0, Math.trunc(value));
}

/**
 * @internal Used by the runtime; pinned in every language by the harness-plans corpus.
 */
export const RunBudgets = {
  /**
   * A run's step allowance, pure (harness-plans step_allowance).
   *
   * The own ceiling is what the runner's context leaves a child it starts;
   * else the ceiling saved with the child; else, for a host context, the
   * steps used plus what the context has left; else the mode's max_steps.
   * max_steps_ceiling then caps it. What is left of it is narrowed by a
   * host context (which can only lower it), the tree grant and the plan
   * grant.
   *
   * @throws RangeError for an unknown context kind, or a runner or host context without contextLeft
   */
  allowance(facts: AllowanceFacts): StepAllowance {
    const kind = facts.kind;
    if (!KINDS.includes(kind)) throw new RangeError('Unknown run context kind.');
    const contextLeft = steps(facts.contextLeft);
    if ((kind === 'runner' || kind === 'host') && contextLeft === null) throw new RangeError('A runner or host context states what it has left.');
    const saved = steps(facts.savedCeiling);
    const stepsCeiling = steps(facts.stepsCeiling);
    // A runner start is a new run, whatever its session's rows say.
    const used = kind === 'runner' ? 0 : (steps(facts.used) ?? 0);
    const tree = steps(facts.treeLeft);
    const plan = steps(facts.planLeft);
    const took = steps(facts.took) ?? 0;

    let ceiling = kind === 'runner' ? contextLeft!
      : saved ?? (kind === 'host' ? used + contextLeft! : (steps(facts.modeSteps) ?? 0));
    if (stepsCeiling !== null) ceiling = Math.min(ceiling, stepsCeiling);
    let ownLeft = Math.max(0, ceiling - used);
    if (kind === 'host') ownLeft = Math.min(ownLeft, contextLeft!);
    let left = ownLeft;
    if (tree !== null) left = Math.min(left, tree);
    if (plan !== null) left = Math.min(left, plan);
    const taken = Math.min(took, left);

    return {
      ceiling, used, own_left: ownLeft, tree_left: tree, plan_left: plan, allowance: left,
      acts_without_step: left === 0, used_at_park: used + taken, left_at_park: left - taken,
    };
  },
};
