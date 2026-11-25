import { HarnessError } from './errors.js';

/**
 * How deep a run tree may nest.
 *
 * Budgets alone do bound a cycle — mode A calling B calling A terminates when
 * the steps run out — but they do not bound it CHEAPLY, and they do not bound
 * the child's address, which grows by `::sub::<name>` at every level. A tree
 * deep enough would produce session keys long enough to truncate in a store
 * with a fixed-width key and collide two distinct children onto one
 * conversation.
 *
 * Depth is also the honest limit to state: nobody debugs a six-deep agent tree,
 * and a config that produced one is a mistake worth reporting rather than
 * executing.
 */
export const MAX_DEPTH = 4;

/**
 * What a run is ALLOWED TO SPEND.
 *
 * `maxSteps` alone was never a budget. It bounds ITERATIONS, and twenty steps
 * each calling an expensive tool sits comfortably inside it — so a run could
 * respect its declared limit and still cost more than anyone intended. Cost and
 * wall-clock are the two a person actually cares about when they say "bounded".
 */
export class RunBudget {
  constructor(
    readonly maxSteps: number,
    readonly maxCostUsd: number | null = null,
    readonly maxSeconds: number | null = null,
    readonly maxOutputTokensPerCall: number | null = null,
  ) {
    if (maxOutputTokensPerCall !== null && (!Number.isSafeInteger(maxOutputTokensPerCall) || maxOutputTokensPerCall < 1)) throw new TypeError('Output ceilings must be positive integers or null.');
  }

  static fromConfig(config: Record<string, unknown>, defaultSteps = 8): RunBudget {
    const output = config.max_output_tokens_per_call;
    if (output !== undefined && output !== null && (typeof output !== 'number' || !Number.isSafeInteger(output) || output < 1)) throw new TypeError('Output ceilings must be positive integers or null.');
    return new RunBudget(
      numberOr(config.max_steps, defaultSteps),
      numberOrNull(config.max_cost_usd),
      numberOrNull(config.max_seconds),
      output ?? null,
    );
  }

  /**
   * The budget a CHILD actually gets, as a TREE-ABSOLUTE ceiling.
   *
   * BUDGETS NEST; THEY DO NOT RESET. A resetting budget is not a budget: a
   * parent limited to 8 steps that may spawn subagents each entitled to a fresh
   * 8 has no bound at all — it has a bound per node in a tree whose width it
   * also controls, which is unbounded spend wearing a limit's clothing.
   *
   * ## Absolute, not remaining — and this is a FIX, not a port
   *
   * The reference computes the child's budget as what REMAINS
   * (`min(declared, parent.maxSteps - ledger.steps)`) and then `exhaustion()`
   * compares the ledger's CUMULATIVE steps against it. Those two are in
   * different units, and the result is that a child is refused the moment its
   * parent has spent anything: parent 8, ledger 7, child declares 2 gives a
   * budget of 1, and `7 >= 1` is immediately exhausted — so the child gets
   * ZERO steps while the tree genuinely has one left. Verified against the
   * reference's own arithmetic, not inferred.
   *
   * Expressing the child's budget as a ceiling THE TREE MAY REACH makes both
   * halves the same unit and gives the right answer in every case: an unspent
   * parent still hands a child exactly what it declared, and a nearly-spent one
   * hands it exactly what is left.
   *
   * Recorded in the envelope's port gaps register as a divergence the reference
   * should adopt.
   */
  nestedWithin(parent: RunBudget, ledger: RunLedger): RunBudget {
    return new RunBudget(
      Math.min(parent.maxSteps, ledger.steps + this.maxSteps),
      lesser(parent.maxCostUsd, this.maxCostUsd === null ? null : ledger.costUsd + this.maxCostUsd),
      lesser(
        parent.maxSeconds,
        this.maxSeconds === null ? null : Math.trunc(ledger.elapsedSeconds()) + this.maxSeconds,
      ),
      lesser(parent.maxOutputTokensPerCall, this.maxOutputTokensPerCall),
    );
  }
}

/**
 * What a run TREE has actually spent, and whether it has been cancelled.
 *
 * SHARED BY REFERENCE from a parent to every descendant, which is the whole
 * point: budgets nest, and nesting is only real if the child's spend lands in
 * the same account the parent is measured against. A per-run ledger would let
 * each node report itself within budget while the tree went far past it.
 *
 * Mutable on purpose, and the only mutable thing here. Spend is a running
 * total; modelling it immutably would mean threading a new instance back up
 * through every return, and the one place that must not be missed is the
 * failure path.
 */
export class RunLedger {
  #steps = 0;

  #costUsd = 0;

  #unmeteredRuns = 0;

  /** Spend attributed to each named agent, in the SAME account as the tree total. */
  readonly #agentCost = new Map<string, number>();

  readonly #agentUnmetered = new Map<string, number>();

  #cancelled = false;

  #cancelReason: string | null = null;

  /**
   * The reason a ledger stopped by a person's cancel carries (G-285), so a
   * cancel is never confused with a spent budget.
   */
  static readonly CANCELLED = 'run cancelled';

  /**
   * Durable cancel flags this tree watches, one per run executing on it in
   * this process (G-285). Not part of any snapshot: each reads the store.
   */
  readonly #cancelProbes = new Map<number, () => Promise<boolean>>();

  #probeSequence = 0;

  constructor(
    readonly rootRunId: string,
    private readonly startedAt: number = Date.now(),
  ) {}

  static start(rootRunId: string): RunLedger {
    return new RunLedger(rootRunId);
  }

  get steps(): number {
    return this.#steps;
  }

  get costUsd(): number {
    return this.#costUsd;
  }

  get unmeteredRuns(): number {
    return this.#unmeteredRuns;
  }

  elapsedSeconds(): number {
    return (Date.now() - this.startedAt) / 1000;
  }

  recordSteps(steps: number): void {
    this.#steps += Math.max(0, steps);
  }

  /**
   * Charge a run's cost to the tree.
   *
   * NULL IS NOT ZERO. A provider's reported cost is nullable because not every
   * provider reports one, and folding that into `+= 0` would leave a cost
   * budget that can never trip — enforced in the documentation, absent at
   * runtime, and indistinguishable from a tree that genuinely spent nothing.
   * Counted separately so `exhaustion()` can say the cap is UNENFORCEABLE
   * instead of quietly failing open.
   */
  recordCost(usd: number | null, agent: string | null = null): void {
    if (usd === null) {
      this.#unmeteredRuns += 1;
      if (agent !== null) this.#agentUnmetered.set(agent, (this.#agentUnmetered.get(agent) ?? 0) + 1);

      return;
    }

    this.#costUsd += Math.max(0, usd);
    if (agent !== null) this.#agentCost.set(agent, (this.#agentCost.get(agent) ?? 0) + Math.max(0, usd));
  }

  /**
   * Charge what a named agent spent BEFORE an approval pause back to it, when
   * the paused run resumes as a new tree (G-321 E). A run that pauses for a
   * person and resumes is the same run, so its spend ceiling continues rather
   * than starting again at nothing.
   *
   * The agent's account only: that spend was already reported by the run that
   * paused, and counting it into this tree's total would report it twice.
   */
  carryAgentSpend(agent: string, usd: number, unmeteredRuns: number): void {
    this.#agentCost.set(agent, (this.#agentCost.get(agent) ?? 0) + Math.max(0, usd));
    if (unmeteredRuns > 0) this.#agentUnmetered.set(agent, (this.#agentUnmetered.get(agent) ?? 0) + unmeteredRuns);
  }

  /**
   * What one named agent has spent in this tree.
   *
   * A LABEL on the tree's own account, not a second account: the same charge
   * already counts toward `costUsd`, so an agent ceiling can only narrow what
   * the tree may spend, never add to it.
   */
  agentCostUsd(agent: string): number {
    return this.#agentCost.get(agent) ?? 0;
  }

  /** Calls by one named agent that reported no cost. */
  agentUnmeteredRuns(agent: string): number {
    return this.#agentUnmetered.get(agent) ?? 0;
  }

  /**
   * Refuse the next call by `agent` when it has reached its ceiling (G-321 E).
   *
   * Checked BEFORE the call, like every budget here. An agent whose provider
   * has reported no cost cannot be held to a ceiling at all, so that fails
   * CLOSED as `budget_unpriced` rather than spending against a cap nobody can
   * measure.
   */
  assertAgentCeiling(agent: string, ceilingUsd: number): void {
    if (this.agentUnmeteredRuns(agent) > 0) {
      throw new HarnessError(
        'budget_unpriced',
        `Agent [${agent}] has a spend ceiling, but its provider reported no cost, so spend against the ceiling is unknown.`,
      );
    }
    const spent = this.agentCostUsd(agent);
    if (spent >= ceilingUsd) {
      throw new HarnessError(
        'agent_spend_ceiling_exceeded',
        `Agent [${agent}] reached its spend ceiling (${spent.toFixed(4)} of ${ceilingUsd.toFixed(4)} USD used in this run tree).`,
      );
    }
  }

  /**
   * Stop the tree.
   *
   * COOPERATIVE rather than pre-emptive: a running tool cannot be interrupted,
   * and pretending otherwise would be the more dangerous lie. A half-executed
   * tool is precisely the state the durability layer exists to protect, so the
   * in-flight call is allowed to finish and the NEXT step is refused.
   */
  cancel(reason = 'cancelled'): void {
    this.#cancelled = true;
    this.#cancelReason = reason;
  }

  get cancelled(): boolean {
    return this.#cancelled;
  }

  /**
   * Watch a run's durable cancel flag (G-285). Every check this tree makes
   * before a step, a tool or a child reads it; once one reads true the tree
   * is cancelled, for every run on it, and stays cancelled. Returns the
   * unwatch, called when that run ends.
   *
   * @internal Registered by the runtime for each run it executes.
   */
  watchCancel(probe: () => Promise<boolean>): () => void {
    const id = ++this.#probeSequence;
    this.#cancelProbes.set(id, probe);

    return () => { this.#cancelProbes.delete(id); };
  }

  /**
   * Whether a person cancelled this tree (G-285), reading every watched flag
   * now. A budget stop, or a host's own `cancel()`, is not a person's cancel
   * and never reads as one.
   */
  async cancelRequested(): Promise<boolean> {
    if (this.#cancelled) return this.#cancelReason === RunLedger.CANCELLED;
    for (const probe of [...this.#cancelProbes.values()]) {
      if (await probe()) {
        // Sticky: set by whichever check saw it first, for the whole tree.
        if (!this.#cancelled) this.cancel(RunLedger.CANCELLED);

        return this.#cancelReason === RunLedger.CANCELLED;
      }
    }

    return false;
  }

  get cancelReason(): string | null {
    return this.#cancelReason;
  }

  remainingCost(budget: RunBudget): number | null {
    return budget.maxCostUsd === null ? null : Math.max(0, budget.maxCostUsd - this.#costUsd);
  }

  remainingSeconds(budget: RunBudget): number | null {
    return budget.maxSeconds === null
      ? null
      : Math.max(0, Math.trunc(budget.maxSeconds - this.elapsedSeconds()));
  }

  /**
   * Why the tree may not spend again — or null when it may.
   *
   * Returns a REASON rather than a boolean. The states are genuinely different
   * (cancelled / out of steps / out of money / out of time) and a caller that
   * cannot tell them apart writes one message for four causes, which is the
   * collapse this ecosystem keeps finding. See prism-parity decision 0020.
   */
  exhaustion(budget: RunBudget): string | null {
    if (this.#cancelled) {
      return this.#cancelReason ?? 'cancelled';
    }

    if (this.#steps >= budget.maxSteps) {
      return `step budget exhausted (${this.#steps} of ${budget.maxSteps} used)`;
    }

    if (budget.maxCostUsd !== null && this.#unmeteredRuns > 0) {
      // Failing CLOSED. A cost cap the provider gives us no numbers to enforce
      // is not a cap, and continuing would spend without limit under a budget
      // the operator believes is holding.
      return (
        `cost budget cannot be enforced: ${this.#unmeteredRuns} run(s) reported no cost, ` +
        `so spend against the ${budget.maxCostUsd.toFixed(4)} USD cap is unknown`
      );
    }

    if (budget.maxCostUsd !== null && this.#costUsd >= budget.maxCostUsd) {
      return `cost budget exhausted (${this.#costUsd.toFixed(4)} of ${budget.maxCostUsd.toFixed(4)} USD used)`;
    }

    if (budget.maxSeconds !== null && this.elapsedSeconds() >= budget.maxSeconds) {
      return `time budget exhausted (${Math.trunc(this.elapsedSeconds())}s of ${budget.maxSeconds}s used)`;
    }

    return null;
  }
}

/**
 * A nested agent a parent run may call, and the AUTHORITY IT GETS.
 *
 * The authority is DECLARED rather than inherited. A subagent that ran with
 * whatever its parent happened to hold would make "narrowed toolset" a
 * description instead of a constraint — and the narrowing is the entire reason
 * to reach for a subagent rather than another turn of the parent.
 */
export class Subagent {
  constructor(
    readonly name: string,
    readonly description: string,
    readonly mode: string,
    readonly budget: RunBudget,
    /**
     * The scope suffix the child's own session and thread live under.
     *
     * Deterministic, so a cold worker resuming the tree lands on the same child
     * conversation instead of starting a fresh one. Defaults to the name.
     */
    readonly scopeSuffix: string | null = null,
    /** The provider this subagent runs on, above any mode's (G-321 E). */
    readonly provider: string | null = null,
    /** The model this subagent runs on, above any mode's. */
    readonly model: string | null = null,
    /**
     * The named agent the child runs as. Only an agent the PARENT MODE lists
     * in its `agents`; the mode registry refuses any other.
     */
    readonly agent: string | null = null,
    /**
     * Opt in to the parent SESSION's provider/model override. Off by default:
     * an override belongs to the session that set it.
     */
    readonly inheritSessionModel: boolean = false,
  ) {}

  /**
   * The scope the child session resolves under.
   *
   * A DIFFERENT scope from the parent, which is what keeps this from
   * deadlocking. A session's lock is taken on its address, and a nested run
   * asking for the parent's address inside the parent's own lock would wait for
   * a lock it is already holding. Giving the child its own address removes the
   * contention rather than making the lock reentrant — which would let a child
   * mutate parent state mid-run, the precise thing the lock is for.
   */
  scopeUnder(parentScope: string): string {
    return `${parentScope}::sub::${this.scopeSuffix ?? this.name}`;
  }
}

/** `modeName` names the declaring mode in a refusal; it defaults to the subagent's own name. */
export function subagentFromConfig(name: string, config: Record<string, unknown>, modeName: string = name): Subagent {
  const description = config.description;
  const mode = config.mode;
  const scope = config.scope;
  const malformed = (detail: string) => new HarnessError('mode_malformed', `Harness mode [${modeName}] declares subagent [${name}] with ${detail}.`);
  for (const key of ['provider', 'model'] as const) {
    const value = config[key];
    if (value !== undefined && value !== null && (typeof value !== 'string' || value === '')) {
      throw malformed('a malformed provider/model: expected a nonempty string or null');
    }
  }
  const inherit = config.inherit_session_model;
  if (inherit !== undefined && typeof inherit !== 'boolean') {
    throw malformed('a malformed inherit_session_model: expected a boolean');
  }
  // Held to the agent-id rule EXACTLY: never trimmed and never case-folded, so
  // a near-miss is a configuration error rather than another agent's authority.
  const agent = config.agent;
  if (agent !== undefined && agent !== null && !isAgentId(agent)) {
    throw malformed('a malformed agent id: agent ids are lowercase identifiers of 1-64 ASCII bytes');
  }

  return new Subagent(
    name,
    typeof description === 'string' && description !== ''
      ? description
      : `Run the [${name}] subagent and return its result.`,
    typeof mode === 'string' && mode !== '' ? mode : name,
    RunBudget.fromConfig(config),
    typeof scope === 'string' ? scope : null,
    (config.provider as string | null | undefined) ?? null,
    (config.model as string | null | undefined) ?? null,
    (agent as string | null | undefined) ?? null,
    inherit === true,
  );
}

/** The agent-id rule `validateAgent()` enforces, as a predicate. */
export function isAgentId(value: unknown): value is string {
  return typeof value === 'string' && /^[a-z][a-z0-9_-]{0,63}$/.exec(value)?.[0] === value;
}

/** A provider and a model, either of which may be unset. */
export interface ModelChoice {
  provider: string | null;
  model: string | null;
}

/**
 * What a SUBAGENT's child run resolves its provider and model from, besides its
 * own mode and agent (G-321 E). Built by `SubagentRunner`; per field:
 *
 *   session (only when the subagent asked to inherit it)
 *   > declared (the subagent declaration)
 *   > the child's own mode
 *   > parentMode (only when the child runs in the parent's agent)
 *   > the child agent's default > global.
 *
 * The child session's own override slot is not consulted: the runner owns that
 * address, and copying the parent's override into it is exactly the drift this
 * replaced.
 */
export interface ModelBinding {
  session: ModelChoice;
  declared: ModelChoice;
  parentMode: ModelChoice;
}

/**
 * Where a run sits in its tree, and what the tree has left to spend.
 *
 * A run with NO context is a root: that is the ordinary case and stays free of
 * all of this.
 */
export interface RunMetadata {
  /**
   * The ACP turn this run serves, when it serves one. Carried onto every run
   * step as `turnId`; Harness never invents one.
   */
  turnId?: string | null;
  /**
   * Who is acting, as the APPLICATION identifies them. Carried onto a
   * `decision_recorded` step as `actor`. Set by the app that authorized the
   * decision, never read from the model or the request body, which is the
   * whole reason it lives here.
   */
  actor?: string | null;
  /**
   * True for a context built by `RunContext.metadata()`: it carries a turnId
   * and/or actor and states NO run tree, so the runtime executes the run
   * exactly as it would with no context at all.
   */
  metadataOnly?: boolean;
}

export class RunContext {
  readonly turnId: string | null;
  readonly actor: string | null;
  readonly metadataOnly: boolean;

  constructor(
    readonly ledger: RunLedger,
    readonly budget: RunBudget,
    readonly parentRunId: string | null = null,
    readonly depth: number = 0,
    metadata: RunMetadata = {},
    /** Set for a subagent's child run; see `ModelBinding`. Null resolves like a root run. */
    readonly binding: ModelBinding | null = null,
  ) {
    // Runtime guards: these arrive from an app's request handling, and an
    // annotation stops nothing that came out of a JSON body.
    for (const key of ['turnId', 'actor'] as const) {
      const value = metadata[key];
      if (value !== undefined && value !== null && (typeof value !== 'string' || value === '')) {
        throw new TypeError(`RunContext ${key} must be a nonempty string or null.`);
      }
    }
    if (metadata.metadataOnly !== undefined && typeof metadata.metadataOnly !== 'boolean') {
      throw new TypeError('RunContext metadataOnly must be a boolean.');
    }
    this.turnId = metadata.turnId ?? null;
    this.actor = metadata.actor ?? null;
    this.metadataOnly = metadata.metadataOnly ?? false;
  }

  static root(runId: string, budget: RunBudget): RunContext {
    return new RunContext(RunLedger.start(runId), budget);
  }

  /**
   * Only metadata — a turn and/or an actor — with no say in how the run
   * executes. The run takes the path and budget it would take with no
   * context; use `root()` to state a run tree.
   */
  static metadata(turnId: string | null = null, actor: string | null = null): RunContext {
    return new RunContext(RunLedger.start('metadata'), new RunBudget(0), null, 0, { turnId, actor, metadataOnly: true });
  }

  /** The same run, serving an ACP turn. */
  withTurn(turnId: string | null): RunContext {
    return new RunContext(this.ledger, this.budget, this.parentRunId, this.depth, { turnId, actor: this.actor, metadataOnly: this.metadataOnly }, this.binding);
  }

  /** The same run, acting for an actor the application has authorized. */
  withActor(actor: string | null): RunContext {
    return new RunContext(this.ledger, this.budget, this.parentRunId, this.depth, { turnId: this.turnId, actor, metadataOnly: this.metadataOnly }, this.binding);
  }

  /** The same run tree under a different budget, keeping its metadata. */
  withBudget(budget: RunBudget): RunContext {
    return new RunContext(this.ledger, budget, this.parentRunId, this.depth, { turnId: this.turnId, actor: this.actor, metadataOnly: this.metadataOnly }, this.binding);
  }

  /** The same run under another model binding, keeping its tree, budget and metadata. */
  withBinding(binding: ModelBinding | null): RunContext {
    return new RunContext(this.ledger, this.budget, this.parentRunId, this.depth, { turnId: this.turnId, actor: this.actor, metadataOnly: this.metadataOnly }, binding);
  }

  get rootRunId(): string {
    return this.ledger.rootRunId;
  }

  isChild(): boolean {
    return this.parentRunId !== null;
  }

  /**
   * The context a child inherits: SAME LEDGER, narrowed budget.
   *
   * Same ledger by reference is the load-bearing part. A child with its own
   * ledger would let every node report itself inside budget while the tree
   * spent without limit.
   */
  forChild(subagent: Subagent, parentRunId: string, binding: ModelBinding | null = null): RunContext {
    return new RunContext(
      this.ledger,
      subagent.budget.nestedWithin(this.budget, this.ledger),
      parentRunId,
      this.depth + 1,
      { turnId: this.turnId, actor: this.actor },
      binding,
    );
  }

  tooDeep(): boolean {
    return this.depth >= MAX_DEPTH;
  }
}

function lesser(a: number | null, b: number | null): number | null {
  if (a === null) return b;
  if (b === null) return a;

  return Math.min(a, b);
}

function numberOr(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? Math.trunc(value) : fallback;
}

function numberOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}
