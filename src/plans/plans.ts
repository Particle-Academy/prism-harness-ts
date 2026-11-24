import { randomBytes } from 'node:crypto';
import { HarnessError } from '../errors.js';
import type { HarnessEvents } from '../events.js';
import type { JsonObject } from '../json.js';
import type { ModeRegistry } from '../modes.js';
import type { AgentRuntime } from '../runtime.js';
import type { Session } from '../session.js';
import { StepChannel, type RunStep } from '../steps.js';
import type { SessionStore } from '../stores/session-store.js';
import type { RunContext } from '../subagents.js';
import type { HarnessTool } from '../tools.js';
import { PlanIds } from './ids.js';
import { PlanLimits, type PlansConfig } from './limits.js';
import { PlanMachine, phpTrim, type Plan, type PlanStep } from './machine.js';
import { PlanRefused, PlansDisabled } from './refusal.js';
import { approvalDecisions, pendingApprovalCalls } from './thread.js';
import { planTools } from './tools.js';
import { PlanView, type PlanCounts } from './view.js';

export interface PlansOptions {
  /** The host's plans config, the same keys as the reference's `prism-harness.plans`. Read on every use. */
  config?: PlansConfig;
  /** Where each session's mode is resolved, to read its `plans` flag. */
  modes: ModeRegistry;
  /** `plan.updated` events go here. Counts and ids only. */
  events?: HarnessEvents;
  /** Mints plan and step ids. ULIDs unless replaced. */
  ids?: PlanIds;
  /** Where the live-plan index and the reaper heartbeat go. Defaults to the harness's durable store. */
  store?: SessionStore;
  /** How long one advance may hold the plan lock and its step claim. Seconds; 300 unless set. */
  lockTtlSeconds?: number;
}

type Watcher = (plan: Plan) => void;

/**
 * Agent plans (G-360): the store, the agent's tools, and the hooks the runtime
 * calls at the moments a plan has to look at the thread.
 *
 * OFF UNLESS BOTH SWITCHES ARE ON: the config's `enabled` and the mode's
 * `plans` flag. Off, no plan tool is offered, no context is added, no plan
 * step is emitted and `session.plan()` rejects with `PlansDisabled`.
 *
 * A person approves a plan before it runs. That is on by default, and only
 * the host's config (`require_approval`) turns it off: never a per-call
 * option, never anything the model can send.
 *
 * Config is read when it is used, so a host that changes it at runtime is
 * read correctly.
 */
export class Plans {
  static readonly INDEX_KEY = 'prism-harness:plans:live';
  static readonly HEARTBEAT_KEY = 'prism-harness:plans:heartbeat';

  readonly config: PlansConfig;
  readonly #modes: ModeRegistry;
  readonly #events: HarnessEvents | null;
  readonly #ids: PlanIds;
  readonly #lockTtl: number;
  #store: SessionStore | null;
  #storeSource: (() => SessionStore) | null = null;
  readonly #watchers = new Map<string, Map<number, Watcher>>();
  #watcherSeq = 0;
  #reapDispatcher: ((sessionKey: string, planId: string) => void | Promise<void>) | null = null;
  #clock: (() => number) | null = null;

  constructor(options: PlansOptions) {
    this.config = options.config ?? {};
    this.#modes = options.modes;
    this.#events = options.events ?? null;
    this.#ids = options.ids ?? new PlanIds();
    this.#store = options.store ?? null;
    const ttl = options.lockTtlSeconds ?? 300;
    if (!Number.isSafeInteger(ttl) || ttl < 1) throw new TypeError('lockTtlSeconds must be a positive whole number of seconds.');
    this.#lockTtl = ttl;
  }

  enabled(): boolean {
    return this.config.enabled === true;
  }

  limits(): PlanLimits {
    return PlanLimits.fromConfig(this.config);
  }

  /** Whether this session's current mode runs with plans. */
  async enabledFor(session: Session, mode?: string | null): Promise<boolean> {
    if (!this.enabled()) return false;
    try {
      return this.#modes.forAgent(session.agent).resolve(mode ?? await session.mode()).plans === true;
    } catch {
      return false;
    }
  }

  async handle(session: Session): Promise<PlanHandle> {
    if (!(await this.enabledFor(session))) throw new PlansDisabled();

    return new PlanHandle(session, this);
  }

  ids(): PlanIds {
    return this.#ids;
  }

  lockTtl(): number {
    return this.#lockTtl;
  }

  /** @internal Tests pin the clock. */
  useClock(clock: (() => number) | null): void {
    this.#clock = clock;
  }

  now(): number {
    return this.#clock !== null ? this.#clock() : Math.floor(Date.now() / 1000);
  }

  /** @internal Set by the harness: its durable store, unless one was given. */
  useIndexStore(source: () => SessionStore): void {
    this.#storeSource ??= source;
  }

  // -- Store -------------------------------------------------------------------

  async load(session: Session): Promise<Plan | null> {
    const payload = await session.planStore().get(key(session.key()));
    const plan = payload?.plan;

    return typeof plan === 'object' && plan !== null && !Array.isArray(plan) ? plan as unknown as Plan : null;
  }

  /** Write a plan, compare-and-set on its revision. Refused with `plan.conflict` when the stored revision is not the one read. */
  async write(session: Session, plan: Plan, expectedRevision: number | null): Promise<void> {
    const store = session.planStore();
    const name = key(session.key());
    await store.withLock(name, async () => {
      const current = (await store.get(name))?.plan;
      const live = typeof current === 'object' && current !== null && !Array.isArray(current) ? current as unknown as Plan : null;
      const revision = live === null ? null : (live.revision ?? null);
      if (revision !== expectedRevision && !(expectedRevision === null && live !== null && !PlanMachine.isLive(live))) {
        throw PlanRefused.because('plan.conflict');
      }
      await store.put(name, { plan: plan as unknown as JsonObject });
    }, 10, 5);

    await this.#indexed(session.key(), plan);
    this.#announce(session, plan);
  }

  /** Load, transform, and write when the transform changed anything. */
  async mutate(session: Session, change: (plan: Plan) => Plan): Promise<Plan | null> {
    const plan = await this.load(session);
    if (plan === null) return null;
    const next = change(plan);
    if (JSON.stringify(next) !== JSON.stringify(plan)) await this.write(session, next, plan.revision);

    return next;
  }

  /** The plan brought in line with the thread, written when it moved. */
  async reconciled(session: Session, superseding = false, runId: string | null = null): Promise<Plan | null> {
    const plan = await this.load(session);
    if (!PlanMachine.isLive(plan)) return plan;
    const rows = (await session.thread().messages()).map((entry) => entry.message);
    const pending = superseding ? [] : pendingApprovalCalls(rows);
    const decided = approvalDecisions(rows);

    return this.mutate(session, (current) => PlanMachine.reconcile(current, pending, decided, this.now(), runId, superseding));
  }

  async #indexed(sessionKey: string, plan: Plan): Promise<void> {
    const store = this.#indexStore();
    if (store === null) return;
    await store.withLock(Plans.INDEX_KEY, async () => {
      const index = { ...((await store.get(Plans.INDEX_KEY)) ?? {}) };
      if (PlanMachine.isLive(plan)) index[sessionKey] = plan.id;
      else delete index[sessionKey];
      await store.put(Plans.INDEX_KEY, index);
    }, 10, 5);
  }

  #indexStore(): SessionStore | null {
    if (this.#store !== null) return this.#store;
    try {
      return this.#storeSource?.() ?? null;
    } catch {
      return null;
    }
  }

  // -- Events and stream watchers ------------------------------------------------

  /** Watch one session's plan writes, for a streamed run. Returns the unwatch. */
  watch(sessionKey: string, watcher: Watcher): () => void {
    const id = ++this.#watcherSeq;
    let watchers = this.#watchers.get(sessionKey);
    if (watchers === undefined) {
      watchers = new Map();
      this.#watchers.set(sessionKey, watchers);
    }
    watchers.set(id, watcher);

    return () => {
      const current = this.#watchers.get(sessionKey);
      current?.delete(id);
      if (current?.size === 0) this.#watchers.delete(sessionKey);
    };
  }

  #announce(session: Session, plan: Plan): void {
    for (const watcher of [...(this.#watchers.get(session.key())?.values() ?? [])]) watcher(plan);
    this.#events?.emit({
      type: 'plan.updated',
      sessionKey: session.key(),
      runId: plan.lastRunId ?? '',
      planId: plan.id,
      revision: plan.revision,
      status: plan.status,
      holdingApprovalId: plan.holdingApprovalId,
      counts: PlanView.counts(plan),
      ...(session.agent === null ? {} : { harnessAgent: session.agent }),
    });
  }

  // -- Runtime hooks ---------------------------------------------------------------

  /** After a run ends: bind the approvals the turn staged, and notice the ones a new message superseded. */
  async afterTurn(session: Session): Promise<void> {
    if (!this.enabled()) return;
    try {
      await this.reconciled(session);
    } catch (error) {
      // The run is over either way; the next advance reconciles again.
      reportQuietly(error);
    }
  }

  /** After decisions are recorded, before the run resumes. */
  async decided(session: Session): Promise<void> {
    if (!this.enabled()) return;
    try {
      await this.reconciled(session);
    } catch (error) {
      // The decision is recorded; the plan reads it again at the end of the
      // run and on the next advance.
      reportQuietly(error);
    }
  }

  /**
   * The plan context for the turn about to start, or ''. A turn with a new
   * user message ends every pending approval, so the plan is reconciled as if
   * nothing were pending before it is rendered. Computed per run, never stored.
   */
  async context(session: Session, prompt: string): Promise<string> {
    if (!(await this.enabledFor(session))) return '';
    let plan: Plan | null;
    try {
      plan = await this.reconciled(session, prompt !== '', await this.runId(session));
    } catch (error) {
      reportQuietly(error);

      return '';
    }

    return plan === null ? '' : PlanView.context(plan, this.limits());
  }

  /** The plan tools for a run in this session, or none when plans are off. */
  async tools(session: Session, mode: string | null): Promise<HarnessTool[]> {
    if (!(await this.enabledFor(session, mode))) return [];

    return planTools(this, session);
  }

  /** @internal The id of the run in flight in this session, or null. */
  async runId(session: Session): Promise<string | null> {
    const run = await session.run();

    return run !== null && run.status === 'running' ? run.id : null;
  }

  // -- Reaper ----------------------------------------------------------------------

  /**
   * The callback the reaper hands a stalled plan to: normally "dispatch the job
   * that calls `advance()` for this session".
   */
  reapUsing(dispatch: (sessionKey: string, planId: string) => void | Promise<void>): void {
    this.#reapDispatcher = dispatch;
  }

  reapDispatcher(): ((sessionKey: string, planId: string) => void | Promise<void>) | null {
    return this.#reapDispatcher;
  }

  async heartbeat(): Promise<{ at: number } | null> {
    const beat = await this.#indexStore()?.get(Plans.HEARTBEAT_KEY);
    const at = beat?.at;

    return typeof at === 'number' && Number.isSafeInteger(at) ? { at } : null;
  }

  /**
   * Find active plans whose head step has no live claim and that have not
   * moved for `reaper.stale_after` seconds, and hand each to `dispatch`
   * (the registered one when none is given). Writes the heartbeat the doctor
   * checks. Returns how many it handed on. Schedule it every few minutes when
   * plans are advanced by a queue.
   */
  async sweep(dispatch: ((sessionKey: string, planId: string) => void | Promise<void>) | null = this.#reapDispatcher): Promise<number> {
    if (!this.enabled()) return 0;
    if (dispatch === null) throw new TypeError('No reaper callback is registered. Call plans.reapUsing() first.');
    const store = this.#indexStore();
    if (store === null) return 0;
    const now = this.now();
    const staleAfter = this.limits().staleAfter;
    const index = (await store.get(Plans.INDEX_KEY)) ?? {};
    let handed = 0;
    for (const [sessionKey, planId] of Object.entries(index)) {
      const plan = (await store.get(key(sessionKey)))?.plan as unknown as Plan | undefined;
      if (typeof plan !== 'object' || plan === null || plan.id !== planId) continue;
      if (PlanMachine.stalled(plan, now, staleAfter)) {
        await dispatch(sessionKey, String(planId));
        handed += 1;
      }
    }
    await store.put(Plans.HEARTBEAT_KEY, { at: now });

    return handed;
  }

  /**
   * What the doctor reports about the reaper. A plan driven by a queue needs
   * one, or one lost job stops the plan for ever with nothing to show for it.
   * With plans on and `queued: true`, a missing or stale heartbeat fails.
   */
  async doctor(options: { queued: boolean }): Promise<{ check: string; ok: boolean; detail: string } | null> {
    if (!this.enabled()) return null;
    let staleAfter: number;
    try {
      staleAfter = this.limits().staleAfter;
    } catch (error) {
      return { check: 'plans', ok: false, detail: error instanceof Error ? error.message : String(error) };
    }
    if (!options.queued) return { check: 'plans', ok: true, detail: 'enabled; plans are advanced in-process, so no reaper is needed' };
    const beat = await this.heartbeat();
    if (beat === null || this.now() - beat.at > 3 * staleAfter) {
      return { check: 'plans', ok: false, detail: 'enabled with a queue, and the plan reaper has not run recently. Schedule plans.sweep().' };
    }

    return { check: 'plans', ok: true, detail: 'enabled; the plan reaper is running' };
  }
}

function key(sessionKey: string): string {
  return `${sessionKey}:plan`;
}

/**
 * Report a failure by its class only. A store error's message can carry the
 * payload it was writing, which here would be plan text the model wrote, and
 * plan text never goes into a log.
 */
function reportQuietly(error: unknown): void {
  const name = error instanceof Error ? error.constructor.name : typeof error;
  process.emitWarning(`A plan update failed: ${name}.`);
}

/**
 * What one `advance()` did (G-360).
 *
 * `ran`: one turn ran on `stepId`, as run `runId`. Call `advance()` again to
 * settle it and run the next step.
 * `waiting`: the head step waits on a person (an approval or an answer).
 * `blocked`: the plan stopped on a limit; `reason` says which.
 * `done`: every step is closed.
 * `not_active`: there is no active plan; `planStatus` says what there is.
 */
export class AdvanceOutcome {
  constructor(
    readonly status: 'ran' | 'waiting' | 'blocked' | 'done' | 'not_active',
    readonly planStatus: string | null,
    readonly stepId: string | null = null,
    readonly runId: string | null = null,
    readonly reason: string | null = null,
  ) {
    Object.freeze(this);
  }

  ran(): boolean {
    return this.status === 'ran';
  }

  toJSON(): Record<string, string | null> {
    return { status: this.status, planStatus: this.planStatus, stepId: this.stepId, runId: this.runId, reason: this.reason };
  }
}

/**
 * The whole plan as stored, for the host's fetch-on-change endpoint (G-360).
 *
 * This is the one place the instruction, the step notes, the questions and
 * the answers are available. Events and run steps carry ids, titles, states
 * and counts only. Whoever serves this to a client decides who may read it.
 */
export class PlanSnapshot {
  readonly #plan: Plan;

  constructor(plan: Plan) {
    this.#plan = structuredClone(plan);
  }

  get id(): string { return this.#plan.id; }
  get status(): string { return this.#plan.status; }
  get revision(): number { return this.#plan.revision; }
  get holdingApprovalId(): string | null { return this.#plan.holdingApprovalId; }
  get blockedReason(): string | null { return this.#plan.blockedReason; }
  get steps(): PlanStep[] { return structuredClone(this.#plan.steps); }
  counts(): PlanCounts { return PlanView.counts(this.#plan); }

  /** The plan without the argument digests, which are the host's own bookkeeping. */
  toJSON(): Omit<Plan, 'draftDigest'> {
    const { draftDigest: _digest, ...plan } = structuredClone(this.#plan);
    if (plan.pendingChange !== null) {
      const { digest: _change, ...change } = plan.pendingChange;
      (plan as Record<string, unknown>).pendingChange = change;
    }
    plan.steps = plan.steps.map(({ pauseDigest: _pause, ...step }) => step as PlanStep);

    return plan;
  }
}

/**
 * One session's plan, for the host (G-360).
 *
 * `advance()` moves the plan at most ONE step: it settles the previous step
 * from what the thread and the run record say, checks the limits, and runs one
 * turn on the next step. Settling never trusts a job's return value and never
 * reads the model's prose: a step is done only when the agent called
 * plan_complete_step and the turn ended cleanly with nothing pending.
 *
 * `answerStep()`, `resume()` and `cancel()` are host calls. WHO MAY MAKE THEM
 * IS THE APPLICATION'S DECISION, as with `recordApproval()`: authorize first,
 * and name the person as the RunContext's `actor`. The actor is never taken
 * from the model.
 */
export class PlanHandle {
  constructor(private readonly session: Session, private readonly plans: Plans) {}

  /** The whole plan, the target of fetch-on-change; null when there is none. */
  async current(): Promise<PlanSnapshot | null> {
    const plan = await this.plans.load(this.session);

    return plan === null ? null : new PlanSnapshot(plan);
  }

  /** The plan context block as the model would read it now; '' when there is no live plan. */
  async context(): Promise<string> {
    const plan = await this.plans.load(this.session);

    return plan === null ? '' : PlanView.context(plan, this.plans.limits());
  }

  /** Settle, choose and run one turn on the next step. One advance at a time per session: a second one fails with `session_locked`. */
  async advance(context?: RunContext): Promise<AdvanceOutcome> {
    const runtime = this.#runtime();

    return this.session.holdPlanLock(async () => {
      const [outcome, step] = await this.#prepare();
      if (step === null) return outcome;
      const response = await runtime.send(this.session, prompt(step), undefined, context);

      return new AdvanceOutcome('ran', 'active', step.id, response.runId);
    }, this.plans.lockTtl());
  }

  /** An alias of `advance()`. */
  continuePlan(context?: RunContext): Promise<AdvanceOutcome> {
    return this.advance(context);
  }

  /** `advance()`, delivered as run steps. The generator's return value is the outcome. */
  streamAdvance(context?: RunContext): AsyncGenerator<RunStep, AdvanceOutcome, undefined> {
    const channel = new StepChannel();

    return channel.drain(() => this.session.holdPlanLock(async () => {
      const runtime = this.#runtime();
      const [outcome, step] = await this.#prepare();
      if (step === null) return outcome;
      const stream = runtime.streamSteps(this.session, prompt(step), undefined, context);
      let runId: string | null = null;
      for (;;) {
        const next = await stream.next();
        if (next.done === true) {
          runId ??= next.value.runId;
          break;
        }
        runId ??= next.value.runId;
        channel.push(next.value);
      }

      return new AdvanceOutcome('ran', 'active', step.id, runId);
    }, this.plans.lockTtl()));
  }

  /**
   * A person answers a question step. The answer is data the agent reads on
   * its next turn; it is never an approval and releases nothing held.
   */
  async answerStep(stepId: string, text: string, choice: string | null, context: RunContext): Promise<PlanSnapshot> {
    const plan = await this.plans.mutate(this.session, (current) => PlanMachine.answer(current, stepId, text, choice, context.actor, this.plans.now()));
    if (plan === null) throw PlanRefused.because('plan.not_answerable');

    return new PlanSnapshot(plan);
  }

  /** The host unblocks the plan. See `PlanMachine.resume()` for what is retried. */
  async resume(context: RunContext): Promise<PlanSnapshot> {
    actor(context);
    const plan = await this.plans.mutate(this.session, (current) => PlanMachine.resume(current, this.plans.ids(), this.plans.limits(), this.plans.now()));
    if (plan === null) throw PlanRefused.because('plan.not_resumable');

    return new PlanSnapshot(plan);
  }

  /** The host stops the plan: every open step is skipped. */
  async cancel(context: RunContext): Promise<PlanSnapshot> {
    actor(context);
    const plan = await this.plans.mutate(this.session, (current) => PlanMachine.cancel(current, this.plans.now()));
    if (plan === null) throw PlanRefused.because('plan.not_live');

    return new PlanSnapshot(plan);
  }

  #runtime(): AgentRuntime {
    const runtime = this.session.planRuntime();
    if (runtime === null) {
      throw new HarnessError('no_agent_runtime', 'Advancing a plan needs the harness to be built with a runtime.');
    }

    return runtime;
  }

  /** Settle, check limits, choose, and claim, under the plan lock. */
  async #prepare(): Promise<[AdvanceOutcome, PlanStep | null]> {
    const plan = await this.plans.reconciled(this.session);
    if (plan === null) return [new AdvanceOutcome('not_active', null), null];
    const run = await this.session.run();
    const runFailed = run !== null && run.status === 'failed';
    const now = this.plans.now();
    const [next, [kind, value]] = PlanMachine.prepare(plan, runFailed, this.plans.limits(), now, null);
    if (JSON.stringify(next) !== JSON.stringify(plan)) await this.plans.write(this.session, next, plan.revision);
    if (kind !== 'run') {
      switch (kind) {
        case 'waiting': return [new AdvanceOutcome('waiting', next.status, value), null];
        case 'blocked': return [new AdvanceOutcome('blocked', next.status, null, null, value), null];
        case 'done': return [new AdvanceOutcome('done', next.status), null];
        default: return [new AdvanceOutcome('not_active', next.status), null];
      }
    }
    const claimed = PlanMachine.claim(next, String(value), `advance_${randomBytes(6).toString('hex')}`, now + this.plans.lockTtl(), now);
    await this.plans.write(this.session, claimed, next.revision);
    const step = claimed.steps.find((candidate) => candidate.id === value);
    if (step === undefined) throw PlanRefused.because('plan.conflict');

    return [new AdvanceOutcome('ran', 'active', step.id), step];
  }
}

/** The host's fixed words. The step's own instruction reaches the model only through the fenced plan context, never as the user turn. */
function prompt(step: PlanStep): string {
  return `Continue the plan. Work on step ${step.position}.`;
}

function actor(context: RunContext): void {
  if (typeof context?.actor !== 'string' || phpTrim(context.actor) === '') throw PlanRefused.because('plan.no_actor');
}
