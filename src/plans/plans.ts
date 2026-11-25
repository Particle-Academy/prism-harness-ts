import { randomBytes } from 'node:crypto';
import { HarnessError } from '../errors.js';
import type { HarnessEvents } from '../events.js';
import type { JsonObject } from '../json.js';
import type { ModeRegistry } from '../modes.js';
import type { AgentResponse, AgentRuntime } from '../runtime.js';
import type { Session } from '../session.js';
import { StepChannel, type RunStep } from '../steps.js';
import type { SessionStore } from '../stores/session-store.js';
import { RunBudget, type RunContext } from '../subagents.js';
import type { HarnessTool } from '../tools.js';
import { PLAN_BUDGET_EXHAUSTED, PlanBudget, type PlanBudgetConfig, type PlanBudgetSettings, type PlanBudgetState } from './budget.js';
import { PlanIds } from './ids.js';
import { PlanLimits, type PlansConfig } from './limits.js';
import { PlanMachine, phpTrim, type Plan, type PlanStep } from './machine.js';
import { PlanRefused, PlansDisabled } from './refusal.js';
import { pausedApprovalIds, pausedRunState } from '../thread-rows.js';
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
  /** Turns spending from a shared plan budget, by run id, until they settle. */
  readonly #budgetTurns = new Map<string, { session: string; ledger: string; plan: string; context: RunContext; steps: number; cost: number; unmetered: number }>();
  /** Sessions whose advance() is running the turn of the step it claimed, counted per session. */
  readonly #advancing = new Map<string, number>();

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

  /** The configured plan budget (G-366), validated: an invalid mode or limit throws `PlanBudgetInvalid`. */
  budgetSettings(): PlanBudgetSettings {
    return PlanBudget.settings(this.config.budget ?? null);
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
    // An invalid plan budget fails here, before anything can run under it.
    this.budgetSettings();

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
    await this.#settleTurn(session);
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
    // No plan can be drafted under an invalid plan budget (G-366).
    this.budgetSettings();

    return planTools(this, session);
  }

  /** @internal The id of the run in flight in this session, or null. */
  async runId(session: Session): Promise<string | null> {
    const run = await session.run();

    return run !== null && run.status === 'running' ? run.id : null;
  }

  // -- Plan budget (G-366) -----------------------------------------------------------

  /** The budget record of this plan, or null when none was fixed yet. */
  async budgetRecord(session: Session, planId: string): Promise<PlanBudgetState | null> {
    const record = await session.planStore().get(budgetKey(session.key()));

    return record !== null && record.planId === planId ? record as unknown as PlanBudgetState : null;
  }

  /** The mode and limits this plan runs under: the fixed record, else config. */
  async budgetFor(session: Session, planId: string): Promise<PlanBudgetSettings> {
    const record = await this.budgetRecord(session, planId);

    return record === null ? this.budgetSettings() : { mode: record.mode, max_steps: record.max_steps, max_cost_micros: record.max_cost_micros };
  }

  /** Host code chooses this plan's budget. The first choice fixes it; after that the mode cannot change and a limit can only be lowered. */
  async useBudget(session: Session, planId: string, requested: unknown): Promise<PlanBudgetState> {
    const chosen = PlanBudget.forPlan(this.budgetSettings(), requested);

    return (await this.#budgetWrite(session, planId, chosen, (state) => [PlanBudget.narrow(state, chosen), null]))[0];
  }

  /** Which shared limit this plan has spent, or null. */
  async budgetExhaustion(session: Session, planId: string): Promise<string | null> {
    const record = await this.budgetRecord(session, planId);

    return record === null ? null : PlanBudget.exhaustion(PlanBudget.expire(record, this.now(), this.#lockTtl));
  }

  /** Announce that a plan stopped on its shared budget. */
  async announceBudgetExhausted(session: Session, planId: string, limit: string): Promise<void> {
    const record = await this.budgetRecord(session, planId) ?? PlanBudget.open(planId, await this.budgetFor(session, planId));
    try {
      this.#events?.emit({
        type: 'plan.budget_exhausted',
        sessionKey: session.key(),
        planId,
        code: PLAN_BUDGET_EXHAUSTED,
        limit,
        steps: record.steps,
        costUsd: record.cost_micros / 1_000_000,
        at: new Date().toISOString(),
        ...(session.agent === null ? {} : { harnessAgent: session.agent }),
      });
    } catch (error) {
      reportQuietly(error);
    }
  }

  /**
   * The shared budget this turn would spend from, or null.
   * @internal
   */
  async servingSharedPlan(session: Session, prompt = '', context?: RunContext): Promise<PlanBudgetSettings | null> {
    return (await this.#ledgerFor(session, prompt, context))?.settings ?? null;
  }

  /**
   * The shared plan budget a run spends from, as the session key the plan
   * belongs to and the plan id; null when it spends from none.
   * @internal Saved with a subagent child the run starts.
   */
  ledgerOf(runId: string): { session: string; plan: string } | null {
    const turn = this.#budgetTurns.get(runId);

    return turn === undefined ? null : { session: turn.ledger, plan: turn.plan };
  }

  /**
   * Mark the turn advance() runs for the step it claimed as the plan's own
   * until the returned release is called.
   * @internal
   */
  servingStep(session: Session): () => void {
    const name = session.key();
    this.#advancing.set(name, (this.#advancing.get(name) ?? 0) + 1);
    let released = false;

    return () => {
      if (released) return;
      released = true;
      const left = (this.#advancing.get(name) ?? 1) - 1;
      if (left <= 0) this.#advancing.delete(name);
      else this.#advancing.set(name, left);
    };
  }

  /**
   * Refuse, before anything is recorded, a decision on an approval the plan's
   * held step asked for when the shared budget its resume would spend from is
   * already spent, or when it is handed a run tree of its own. Recorded, it
   * would be consumed by a resume the budget then refuses, and the approved
   * call would never run.
   *
   * @internal Called by every path that records decisions.
   */
  async assertDecisionsFunded(session: Session, approvalIds: readonly string[], context?: RunContext): Promise<void> {
    if (!this.enabled() || approvalIds.length === 0) return;
    let target: { session: string; plan: string } | null = null;
    const plan = await this.enabledFor(session) ? await this.load(session) : null;
    const step = activeAgentStep(plan);
    if (plan !== null && step !== null && step.approvalIds.some((id) => approvalIds.includes(id))) {
      if ((await this.budgetFor(session, plan.id)).mode !== 'shared') return;
      target = { session: session.key(), plan: plan.id };
    } else if (context === undefined || context.metadataOnly) {
      // A subagent child resumed outside the runner spends from the shared
      // plan budget of the turn that started it.
      target = await this.#childLedger(session);
    }
    if (target === null) return;
    if (context !== undefined && !context.metadataOnly) throw PlanRefused.because('plan.budget_context');
    const record = await this.#budgetRecordAt(session.planStore(), target.session, target.plan);
    const spent = record === null ? null : PlanBudget.exhaustion(PlanBudget.expire(record, this.now(), this.#lockTtl));
    if (spent !== null) throw budgetSpent(spent);
  }

  /**
   * The run a turn executes under, when the turn is one of the plan's own and
   * its budget is shared: the per-turn budget, narrowed to what the plan's
   * ledger holds for it. Any other run is returned unchanged.
   *
   * The plan's own turns are the one advance() runs for the step it claimed
   * and the resume of an approval that step asked for. Any other turn on the
   * session (a person chatting while a step waits on them, or while a claim
   * whose worker died is still live) is not the plan's: it spends nothing
   * from the plan's budget and is never refused for it.
   *
   * A run handed a run tree of its own cannot also spend from the plan's
   * ledger, so it is refused.
   *
   * @internal Called by the runtime once a run's own budget is known.
   */
  async planTurn(session: Session, runId: string, run: RunContext, context: RunContext | undefined, prompt = ''): Promise<RunContext> {
    const target = await this.#ledgerFor(session, prompt, context);
    if (target === null) return run;
    if (context !== undefined && !context.metadataOnly) throw PlanRefused.because('plan.budget_context');

    const { ledger, budget } = run;
    const baseSteps = ledger.steps;
    const baseCost = ledger.costUsd;
    const turnCost = budget.maxCostUsd === null ? null : PlanBudget.micros(Math.max(0, budget.maxCostUsd - baseCost));
    const now = this.now();
    const ttl = this.#lockTtl;
    const [, grant] = await this.#budgetWriteAt(session.planStore(), target.session, target.plan, this.budgetSettings(),
      (state) => PlanBudget.reserve(state, runId, Math.max(0, budget.maxSteps - baseSteps), turnCost, now, ttl));
    if (grant === null || 'refused' in grant) throw budgetSpent(grant?.limit ?? 'steps');
    const narrowed = run.withBudget(new RunBudget(
      baseSteps + grant.steps,
      grant.cost_micros === null ? budget.maxCostUsd : baseCost + grant.cost_micros / 1_000_000,
      budget.maxSeconds,
      budget.maxOutputTokensPerCall,
    ));
    this.#budgetTurns.set(runId, { session: session.key(), ledger: target.session, plan: target.plan, context: narrowed, steps: baseSteps, cost: baseCost, unmetered: ledger.unmeteredRuns });

    return narrowed;
  }

  /**
   * The shared ledger a turn spends from: its own plan's, when the turn is
   * one of the plan's own; else, for a subagent child resumed outside the
   * runner, the ledger of the plan turn that started it. Null otherwise.
   */
  async #ledgerFor(session: Session, prompt: string, context?: RunContext): Promise<{ session: string; plan: string; settings: PlanBudgetSettings } | null> {
    if (!this.enabled()) return null;
    if (await this.enabledFor(session)) {
      const plan = await this.load(session);
      const step = activeAgentStep(plan);
      if (plan !== null && step !== null) {
        const own = this.#advancing.has(session.key())
          || (prompt === '' && step.approvalIds.length > 0
            && pausedApprovalIds((await session.thread().messages()).map((entry) => entry.message)).some((id) => step.approvalIds.includes(id)));
        if (own) {
          const settings = await this.budgetFor(session, plan.id);

          return settings.mode === 'shared' ? { session: session.key(), plan: plan.id, settings } : null;
        }
      }
    }

    return prompt === '' && (context === undefined || context.metadataOnly) ? this.#childLedger(session) : null;
  }

  /** The plan budget saved on a paused subagent child's row, while it is still that plan's shared budget. */
  async #childLedger(session: Session): Promise<{ session: string; plan: string; settings: PlanBudgetSettings } | null> {
    const saved = pausedRunState((await session.thread().messages()).map((entry) => entry.message))?.planLedger ?? null;
    if (saved === null) return null;
    const record = await this.#budgetRecordAt(session.planStore(), saved.session, saved.plan);
    if (record === null || record.mode !== 'shared') return null;

    return { ...saved, settings: { mode: 'shared', max_steps: record.max_steps, max_cost_micros: record.max_cost_micros } };
  }

  async #budgetRecordAt(store: SessionStore, sessionKey: string, planId: string): Promise<PlanBudgetState | null> {
    const record = await store.get(budgetKey(sessionKey));

    return record !== null && record.planId === planId ? record as unknown as PlanBudgetState : null;
  }

  /** Replace a finished turn's hold with what it spent. */
  async #settleTurn(session: Session): Promise<void> {
    const run = await session.run();
    const turn = run === null ? undefined : this.#budgetTurns.get(run.id);
    if (run === null || turn === undefined || turn.session !== session.key()) return;
    this.#budgetTurns.delete(run.id);
    const ledger = turn.context.ledger;
    const steps = ledger.steps - turn.steps;
    const cost = ledger.unmeteredRuns > turn.unmetered ? null : PlanBudget.micros(Math.max(0, ledger.costUsd - turn.cost));
    try {
      await this.#budgetWriteAt(session.planStore(), turn.ledger, turn.plan, this.budgetSettings(), (state) => [PlanBudget.settle(state, run.id, steps, cost), null]);
    } catch (error) {
      // The hold stays and is charged in full once it expires.
      reportQuietly(error);
    }
  }

  /** One locked change to this plan's budget record, opening it from `settings` when it is absent or belongs to an earlier plan. */
  async #budgetWrite<T>(
    session: Session,
    planId: string,
    settings: PlanBudgetSettings,
    change: (state: PlanBudgetState) => [PlanBudgetState, T],
  ): Promise<[PlanBudgetState, T]> {
    return this.#budgetWriteAt(session.planStore(), session.key(), planId, settings, change);
  }

  /** The same change to the budget record of the plan kept under another session key, in the same durable store (a subagent child's parent). */
  async #budgetWriteAt<T>(
    store: SessionStore,
    sessionKey: string,
    planId: string,
    settings: PlanBudgetSettings,
    change: (state: PlanBudgetState) => [PlanBudgetState, T],
  ): Promise<[PlanBudgetState, T]> {
    const name = budgetKey(sessionKey);

    return store.withLock(name, async () => {
      const stored = await store.get(name);
      const state = stored !== null && stored.planId === planId ? stored as unknown as PlanBudgetState : PlanBudget.open(planId, settings);
      const [next, result] = change(state);
      await store.put(name, next as unknown as JsonObject);

      return [next, result] as [PlanBudgetState, T];
    }, 10, 5);
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
 * settle it and run the next step. `response` is that run's response, so a
 * host can put the approvals it staged (`pendingApprovals`) in front of a
 * person. It is not part of `toJSON()`.
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
    readonly response: AgentResponse | null = null,
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
      let response: AgentResponse;
      // This turn, and only this one, is the step's own (G-366).
      const serving = this.plans.servingStep(this.session);
      try {
        response = await runtime.send(this.session, prompt(step), undefined, context);
      } catch (error) {
        const stopped = error instanceof HarnessError && error.code === 'run_not_permitted' || error instanceof HarnessError && error.code === PLAN_BUDGET_EXHAUSTED
          ? await this.#stoppedOnBudget() : null;
        if (stopped === null) throw error;

        return stopped;
      } finally {
        serving();
      }

      return new AdvanceOutcome('ran', 'active', step.id, response.runId, null, response);
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
      let response: AgentResponse | null = null;
      const serving = this.plans.servingStep(this.session);
      try {
        for (;;) {
          let next: IteratorResult<RunStep, AgentResponse>;
          try {
            next = await stream.next();
          } catch (error) {
            const stopped = error instanceof HarnessError && (error.code === 'run_not_permitted' || error.code === PLAN_BUDGET_EXHAUSTED)
              ? await this.#stoppedOnBudget() : null;
            if (stopped === null) throw error;

            return stopped;
          }
          if (next.done === true) {
            runId ??= next.value.runId;
            response = next.value;
            break;
          }
          runId ??= next.value.runId;
          channel.push(next.value);
        }
      } finally {
        serving();
      }

      return new AdvanceOutcome('ran', 'active', step.id, runId, null, response);
    }, this.plans.lockTtl()));
  }

  /**
   * Choose this plan's budget (G-366), in host code: `shared` with
   * `max_steps` and optionally `max_cost_usd`, `per_step`, or `none`. Each
   * limit is the lower of this and the configured one. The first choice, or
   * the plan's first shared turn, fixes the budget: after that the mode cannot
   * change (`plan.budget_fixed`) and a limit can only be lowered. Never
   * reachable from a tool or the model.
   */
  async useBudget(budget: PlanBudgetConfig): Promise<void> {
    const plan = await this.plans.load(this.session);
    if (!PlanMachine.isLive(plan)) throw PlanRefused.because('plan.no_plan');
    await this.plans.useBudget(this.session, plan!.id, budget);
  }

  /** The plan's budget as it stands: mode, limits and what a shared budget has spent. Null when there is no plan. */
  async budget(): Promise<{ mode: string; max_steps: number | null; max_cost_usd: number | null; steps: number; cost_usd: number } | null> {
    const plan = await this.plans.load(this.session);
    if (plan === null) return null;
    const settings = await this.plans.budgetFor(this.session, plan.id);
    const record = await this.plans.budgetRecord(this.session, plan.id);

    return {
      mode: settings.mode,
      max_steps: settings.max_steps,
      max_cost_usd: settings.max_cost_micros === null ? null : settings.max_cost_micros / 1_000_000,
      steps: record?.steps ?? 0,
      cost_usd: (record?.cost_micros ?? 0) / 1_000_000,
    };
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

  /** A turn refused because the plan's shared budget is spent stops the plan cleanly; null when the budget is not what stopped it. */
  async #stoppedOnBudget(): Promise<AdvanceOutcome | null> {
    const plan = await this.plans.load(this.session);
    if (plan === null || plan.status !== 'active' || await this.plans.budgetExhaustion(this.session, plan.id) === null) return null;

    return (await this.#prepare())[0];
  }

  /** Settle, check limits, choose, and claim, under the plan lock. */
  async #prepare(): Promise<[AdvanceOutcome, PlanStep | null]> {
    const plan = await this.plans.reconciled(this.session);
    if (plan === null) return [new AdvanceOutcome('not_active', null), null];
    const run = await this.session.run();
    const runFailed = run !== null && run.status === 'failed';
    const now = this.plans.now();
    const spent = plan.status === 'active' ? await this.plans.budgetExhaustion(this.session, plan.id) : null;
    const [next, [kind, value]] = PlanMachine.prepare(plan, runFailed, this.plans.limits(), now, null, spent !== null);
    if (JSON.stringify(next) !== JSON.stringify(plan)) await this.plans.write(this.session, next, plan.revision);
    if (kind === 'blocked' && value === PLAN_BUDGET_EXHAUSTED) await this.plans.announceBudgetExhausted(this.session, plan.id, String(spent));
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

/** The plan's head step when it is the agent's and claimed for a turn or held on an approval that turn asked for; null otherwise. */
function activeAgentStep(plan: Plan | null): PlanStep | null {
  if (plan === null || !['active', 'waiting', 'blocked'].includes(plan.status)) return null;
  const head = PlanMachine.headIndex(plan);
  if (head === null) return null;
  const step = plan.steps[head]!;

  return step.owner === 'agent' && ['running', 'held'].includes(step.state) ? step : null;
}

function budgetSpent(limit: string): HarnessError {
  return new HarnessError(PLAN_BUDGET_EXHAUSTED,
    `This plan step may not proceed (plan_budget_exhausted): the plan's shared ${limit} budget is spent.`);
}

function budgetKey(sessionKey: string): string {
  return `${sessionKey}:plan-budget`;
}

/** The host's fixed words. The step's own instruction reaches the model only through the fenced plan context, never as the user turn. */
function prompt(step: PlanStep): string {
  return `Continue the plan. Work on step ${step.position}.`;
}

function actor(context: RunContext): void {
  if (typeof context?.actor !== 'string' || phpTrim(context.actor) === '') throw PlanRefused.because('plan.no_actor');
}
