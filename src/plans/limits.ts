import type { PlanBudgetConfig } from './budget.js';

/**
 * The bounds a plan runs inside (G-360).
 *
 * A value outside its range is REFUSED at construction, never clamped: an
 * operator who asked for 500 steps should learn that 40 is the most an update
 * can carry, not find out from a plan that was silently cut.
 */
export interface PlanLimitsInit {
  maxSteps?: number;
  maxAttempts?: number;
  maxPausesPerStep?: number;
  maxChanges?: number;
  maxAdvances?: number;
  contextMaxChars?: number;
  contextWindow?: number;
  eventMaxBytes?: number;
  requireApproval?: boolean;
  runSteps?: boolean;
  staleAfter?: number;
}

/** The host config shape, the same keys as the reference's `prism-harness.plans`. */
export interface PlansConfig {
  /** Off unless true, AND the session's mode sets `plans: true`. */
  enabled?: boolean;
  /** A person approves every plan before it runs. On by default; only this turns it off. */
  require_approval?: boolean;
  max_steps?: number;
  max_attempts?: number;
  max_pauses_per_step?: number;
  max_changes?: number;
  max_advances?: number;
  context?: { max_chars?: number; window?: number };
  /** `run_steps`: mix `plan_updated` steps (contract g282.3) into streamed runs. Off by default. */
  events?: { run_steps?: boolean; max_bytes?: number };
  reaper?: { stale_after?: number };
  /**
   * How a plan's steps are budgeted (G-366): `per_step` (default), `shared`
   * with `max_steps` (required) and optional `max_cost_usd`, or `none`. See
   * `PlanBudget`. Invalid values throw `PlanBudgetInvalid` when plans are used.
   */
  budget?: PlanBudgetConfig | null;
}

export class PlanLimits {
  /** Fancy contract g282.3: a plan carries at most 40 steps. */
  static readonly STEPS_CEILING = 40;

  /** Fancy contract g282.3: the first update is under 8 KB. */
  static readonly EVENT_BYTES_CEILING = 8192;

  readonly maxSteps: number;
  readonly maxAttempts: number;
  readonly maxPausesPerStep: number;
  readonly maxChanges: number;
  readonly maxAdvances: number;
  readonly contextMaxChars: number;
  readonly contextWindow: number;
  readonly eventMaxBytes: number;
  readonly requireApproval: boolean;
  readonly runSteps: boolean;
  readonly staleAfter: number;

  constructor(init: PlanLimitsInit = {}) {
    this.maxSteps = init.maxSteps ?? 20;
    this.maxAttempts = init.maxAttempts ?? 3;
    this.maxPausesPerStep = init.maxPausesPerStep ?? 1;
    this.maxChanges = init.maxChanges ?? 5;
    this.maxAdvances = init.maxAdvances ?? 60;
    this.contextMaxChars = init.contextMaxChars ?? 2000;
    this.contextWindow = init.contextWindow ?? 8;
    this.eventMaxBytes = init.eventMaxBytes ?? PlanLimits.EVENT_BYTES_CEILING;
    this.requireApproval = init.requireApproval ?? true;
    this.runSteps = init.runSteps ?? false;
    this.staleAfter = init.staleAfter ?? 300;
    const ranges: [string, number, number, number][] = [
      ['max_steps', this.maxSteps, 1, PlanLimits.STEPS_CEILING],
      ['max_attempts', this.maxAttempts, 1, 20],
      ['max_pauses_per_step', this.maxPausesPerStep, 0, 10],
      ['max_changes', this.maxChanges, 0, 50],
      ['max_advances', this.maxAdvances, 1, 10000],
      ['context.max_chars', this.contextMaxChars, 500, 20000],
      ['context.window', this.contextWindow, 0, PlanLimits.STEPS_CEILING],
      ['events.max_bytes', this.eventMaxBytes, 1024, PlanLimits.EVENT_BYTES_CEILING],
      ['reaper.stale_after', this.staleAfter, 1, 86400],
    ];
    for (const [name, value, min, max] of ranges) {
      if (!Number.isSafeInteger(value) || value < min || value > max) {
        throw new RangeError(`prism-harness.plans.${name} must be between ${min} and ${max}.`);
      }
    }
    for (const [name, value] of [['require_approval', this.requireApproval], ['events.run_steps', this.runSteps]] as const) {
      if (typeof value !== 'boolean') throw new TypeError(`prism-harness.plans.${name} must be true or false.`);
    }
    Object.freeze(this);
  }

  /** From the host config. A value of the wrong type is refused, never read as its default. */
  static fromConfig(config: PlansConfig = {}): PlanLimits {
    const int = (value: unknown, name: string): number | undefined => {
      if (value === undefined || value === null) return undefined;
      if (typeof value !== 'number' || !Number.isSafeInteger(value)) throw new TypeError(`prism-harness.plans.${name} must be a whole number.`);
      return value;
    };
    const bool = (value: unknown, name: string): boolean | undefined => {
      if (value === undefined || value === null) return undefined;
      if (typeof value !== 'boolean') throw new TypeError(`prism-harness.plans.${name} must be true or false.`);
      return value;
    };
    const section = (value: unknown): Record<string, unknown> =>
      typeof value === 'object' && value !== null && !Array.isArray(value) ? value as Record<string, unknown> : {};
    const context = section(config.context), events = section(config.events), reaper = section(config.reaper);
    const init: PlanLimitsInit = {};
    const set = <K extends keyof PlanLimitsInit>(key: K, value: PlanLimitsInit[K] | undefined): void => {
      if (value !== undefined) init[key] = value;
    };
    set('maxSteps', int(config.max_steps, 'max_steps'));
    set('maxAttempts', int(config.max_attempts, 'max_attempts'));
    set('maxPausesPerStep', int(config.max_pauses_per_step, 'max_pauses_per_step'));
    set('maxChanges', int(config.max_changes, 'max_changes'));
    set('maxAdvances', int(config.max_advances, 'max_advances'));
    set('contextMaxChars', int(context.max_chars, 'context.max_chars'));
    set('contextWindow', int(context.window, 'context.window'));
    set('eventMaxBytes', int(events.max_bytes, 'events.max_bytes'));
    set('requireApproval', bool(config.require_approval, 'require_approval'));
    set('runSteps', bool(events.run_steps, 'events.run_steps'));
    set('staleAfter', int(reaper.stale_after, 'reaper.stale_after'));

    return new PlanLimits(init);
  }

  toJSON(): Record<string, number | boolean> {
    return {
      max_steps: this.maxSteps,
      max_attempts: this.maxAttempts,
      max_pauses_per_step: this.maxPausesPerStep,
      max_changes: this.maxChanges,
      max_advances: this.maxAdvances,
      context_max_chars: this.contextMaxChars,
      context_window: this.contextWindow,
      event_max_bytes: this.eventMaxBytes,
      require_approval: this.requireApproval,
      run_steps: this.runSteps,
      stale_after: this.staleAfter,
    };
  }
}
