import { PlanRefused } from './refusal.js';

/**
 * How a plan's steps are budgeted, and the plan's shared ledger (G-366).
 *
 * The app's developer chooses, in the plans config's `budget` or per plan
 * through host code (`PlanHandle.useBudget()`); never the model, never a
 * per-call option:
 *
 * - `per_step` (default): each step's turn gets the per-turn budget, as it
 *   always has.
 * - `shared`: every turn that serves the plan also spends from ONE ledger
 *   with `max_steps` model steps and, optionally, `max_cost_usd`. Each turn
 *   still keeps its own per-turn limits; the ledger bounds the total.
 * - `none`: no plan-level budget. Per-turn limits, the plan's own step and
 *   advance limits and agent spend ceilings still apply.
 *
 * Pure state in, state out, the same as the PHP reference's PlanBudget
 * (pinned by the `harness-plans` corpus). `Plans` keeps the state in the plan
 * store and changes it in one locked write per change.
 *
 * The ledger is a hold-and-settle account. Before a turn, the steps and cost
 * it may spend are HELD; after it, the hold is replaced by what it spent. Two
 * turns can therefore never both be granted the same remainder. A hold that
 * outlives the hold TTL (a turn that died without settling) is charged in
 * full: an unknown spend counts as spent, never as free.
 *
 * Costs are kept in whole micro-USD so every language adds them exactly.
 */
export type PlanBudgetMode = 'per_step' | 'shared' | 'none';

export interface PlanBudgetSettings {
  mode: PlanBudgetMode;
  max_steps: number | null;
  max_cost_micros: number | null;
}

/** The host config shape, `plans.budget`. */
export interface PlanBudgetConfig {
  mode?: PlanBudgetMode;
  max_steps?: number | null;
  max_cost_usd?: number | null;
}

export interface PlanBudgetHold {
  steps: number;
  cost_micros: number;
  at: number;
}

export interface PlanBudgetState extends PlanBudgetSettings {
  planId: string;
  steps: number;
  cost_micros: number;
  unmetered: number;
  holds: Record<string, PlanBudgetHold>;
}

export type PlanBudgetLimit = 'steps' | 'cost' | 'cost_unmetered';

export type PlanBudgetGrant = { steps: number; cost_micros: number | null } | { refused: typeof PLAN_BUDGET_EXHAUSTED; limit: PlanBudgetLimit };

/** The code a turn and a plan stop with when the shared ledger is spent. */
export const PLAN_BUDGET_EXHAUSTED = 'plan_budget_exhausted';

/** An invalid budget setting. Thrown at configuration, before anything runs. */
export class PlanBudgetInvalid extends TypeError {
  constructor(message: string) {
    super(message);
    this.name = 'PlanBudgetInvalid';
  }
}

const MODES: readonly string[] = ['per_step', 'shared', 'none'];
const KEYS: readonly string[] = ['mode', 'max_steps', 'max_cost_usd'];
const STEPS_CEILING = 10000;
const COST_CEILING_USD = 1000000;

function lower(a: number | null, b: number | null): number | null {
  return a === null ? b : (b === null ? a : Math.min(a, b));
}

export const PlanBudget = {
  MODES,
  EXHAUSTED: PLAN_BUDGET_EXHAUSTED,

  /**
   * Validate a budget setting. Throws `PlanBudgetInvalid` for anything
   * invalid: an unknown key or mode, a limit out of range, or `shared`
   * without `max_steps`.
   */
  settings(config: unknown, source = 'plans.budget', complete = true): PlanBudgetSettings {
    if (config === null || config === undefined) return { mode: 'per_step', max_steps: null, max_cost_micros: null };
    if (typeof config !== 'object' || Array.isArray(config)) throw new PlanBudgetInvalid(`${source} must be a map.`);
    const given = config as Record<string, unknown>;
    for (const key of Object.keys(given)) {
      if (!KEYS.includes(key)) throw new PlanBudgetInvalid(`${source} has an unknown setting.`);
    }
    const mode = given.mode ?? 'per_step';
    if (typeof mode !== 'string' || !MODES.includes(mode)) throw new PlanBudgetInvalid(`${source}.mode must be per_step, shared or none.`);
    const steps = given.max_steps ?? null;
    if (steps !== null && (typeof steps !== 'number' || !Number.isSafeInteger(steps) || steps < 1 || steps > STEPS_CEILING)) {
      throw new PlanBudgetInvalid(`${source}.max_steps must be a whole number between 1 and ${STEPS_CEILING}, or null.`);
    }
    const cost = given.max_cost_usd ?? null;
    if (cost !== null && (typeof cost !== 'number' || !Number.isFinite(cost) || cost < 0 || cost > COST_CEILING_USD)) {
      throw new PlanBudgetInvalid(`${source}.max_cost_usd must be a number of USD between 0 and ${COST_CEILING_USD}, or null.`);
    }
    if (complete && mode === 'shared' && steps === null) throw new PlanBudgetInvalid(`${source}.max_steps is required when the mode is shared.`);

    return { mode: mode as PlanBudgetMode, max_steps: steps as number | null, max_cost_micros: cost === null ? null : PlanBudget.micros(cost as number) };
  },

  /** The setting for one plan, chosen by host code: its mode, with each limit the lower of the host's and the configured one. */
  forPlan(configured: PlanBudgetSettings, requested: unknown): PlanBudgetSettings {
    const asked = PlanBudget.settings(requested, 'the plan budget', false);
    const chosen: PlanBudgetSettings = {
      mode: asked.mode,
      max_steps: lower(configured.max_steps, asked.max_steps),
      max_cost_micros: lower(configured.max_cost_micros, asked.max_cost_micros),
    };
    if (chosen.mode === 'shared' && chosen.max_steps === null) throw new PlanBudgetInvalid('the plan budget.max_steps is required when the mode is shared.');

    return chosen;
  },

  /** A plan's budget record, fixed from a setting. */
  open(planId: string, settings: PlanBudgetSettings): PlanBudgetState {
    return {
      planId,
      mode: settings.mode,
      max_steps: settings.max_steps,
      max_cost_micros: settings.max_cost_micros,
      steps: 0,
      cost_micros: 0,
      unmetered: 0,
      holds: {},
    };
  },

  /** A later host choice on a fixed record: the mode cannot change, and a limit can only be lowered. */
  narrow(state: PlanBudgetState, settings: PlanBudgetSettings): PlanBudgetState {
    if (settings.mode !== state.mode) throw PlanRefused.because('plan.budget_fixed');

    return {
      ...state,
      max_steps: lower(state.max_steps, settings.max_steps),
      max_cost_micros: lower(state.max_cost_micros, settings.max_cost_micros),
    };
  },

  /** Which limit is spent, or null. Holds do not count: they are spend that may yet be returned. */
  exhaustion(state: PlanBudgetState): PlanBudgetLimit | null {
    if (state.mode !== 'shared') return null;
    if (state.steps >= (state.max_steps ?? 0)) return 'steps';
    if (state.max_cost_micros !== null) {
      // A cost a provider did not report cannot be held to a cap: closed.
      if (state.unmetered > 0) return 'cost_unmetered';
      if (state.cost_micros >= state.max_cost_micros) return 'cost';
    }

    return null;
  },

  /**
   * Hold what one turn may spend: at most `steps` steps and `costMicros`
   * (null: no per-turn cost cap), never more than the ledger has left after
   * every other hold. Expired holds are charged in full first.
   */
  reserve(state: PlanBudgetState, holdId: string, steps: number, costMicros: number | null, now: number, holdTtl: number): [PlanBudgetState, PlanBudgetGrant] {
    const next = PlanBudget.expire(state, now, holdTtl);
    let heldSteps = 0;
    let heldCost = 0;
    for (const hold of Object.values(next.holds)) {
      heldSteps += hold.steps;
      heldCost += hold.cost_micros;
    }
    let limit = PlanBudget.exhaustion(next);
    const leftSteps = (next.max_steps ?? 0) - next.steps - heldSteps;
    const leftCost = next.max_cost_micros === null ? null : next.max_cost_micros - next.cost_micros - heldCost;
    if (limit === null && leftSteps <= 0) limit = 'steps';
    if (limit === null && leftCost !== null && leftCost <= 0) limit = 'cost';
    if (limit !== null || steps < 1) return [next, { refused: PLAN_BUDGET_EXHAUSTED, limit: limit ?? 'steps' }];
    const grantSteps = Math.min(steps, leftSteps);
    const grantCost = leftCost === null ? null : (costMicros === null ? leftCost : Math.min(costMicros, leftCost));
    const holds: Record<string, PlanBudgetHold> = { ...next.holds, [holdId]: { steps: grantSteps, cost_micros: grantCost ?? 0, at: now } };

    return [{ ...next, holds: sortedHolds(holds) }, { steps: grantSteps, cost_micros: grantCost }];
  },

  /** Replace a turn's hold with what it spent. `costMicros` null: the provider reported no cost. A hold already gone changes nothing. */
  settle(state: PlanBudgetState, holdId: string, steps: number, costMicros: number | null): PlanBudgetState {
    if (!Object.hasOwn(state.holds, holdId)) return state;
    const { [holdId]: _gone, ...holds } = state.holds;

    return {
      ...state,
      holds,
      steps: state.steps + Math.max(0, steps),
      unmetered: costMicros === null ? state.unmetered + 1 : state.unmetered,
      cost_micros: costMicros === null ? state.cost_micros : state.cost_micros + Math.max(0, costMicros),
    };
  },

  /** Charge every hold older than the TTL in full. */
  expire(state: PlanBudgetState, now: number, holdTtl: number): PlanBudgetState {
    let next = state;
    for (const [id, hold] of Object.entries(state.holds)) {
      if (now - hold.at >= holdTtl) {
        const { [id]: _gone, ...holds } = next.holds;
        next = { ...next, holds, steps: next.steps + hold.steps, cost_micros: next.cost_micros + hold.cost_micros };
      }
    }

    return next;
  },

  micros(usd: number): number {
    return Math.round(usd * 1_000_000);
  },
};

function sortedHolds(holds: Record<string, PlanBudgetHold>): Record<string, PlanBudgetHold> {
  const out: Record<string, PlanBudgetHold> = {};
  for (const id of Object.keys(holds).sort()) out[id] = holds[id]!;

  return out;
}
