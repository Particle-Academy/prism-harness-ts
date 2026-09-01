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
  ) {}

  static fromConfig(config: Record<string, unknown>, defaultSteps = 8): RunBudget {
    return new RunBudget(
      numberOr(config.max_steps, defaultSteps),
      numberOrNull(config.max_cost_usd),
      numberOrNull(config.max_seconds),
    );
  }

  /**
   * The budget a CHILD actually gets.
   *
   * BUDGETS NEST; THEY DO NOT RESET. A resetting budget is not a budget: a
   * parent limited to 8 steps that may spawn subagents each entitled to a fresh
   * 8 has no bound at all — it has a bound per node in a tree whose width it
   * also controls, which is unbounded spend wearing a limit's clothing.
   *
   * So a child gets the SMALLER of what it declares and what the tree has left.
   * A child may ask for less than it is offered; it may never ask for more than
   * remains.
   */
  nestedWithin(parent: RunBudget, ledger: RunLedger): RunBudget {
    return new RunBudget(
      Math.min(this.maxSteps, Math.max(0, parent.maxSteps - ledger.steps)),
      lesser(this.maxCostUsd, ledger.remainingCost(parent)),
      lesser(this.maxSeconds, ledger.remainingSeconds(parent)),
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

  #cancelled = false;

  #cancelReason: string | null = null;

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
  recordCost(usd: number | null): void {
    if (usd === null) {
      this.#unmeteredRuns += 1;

      return;
    }

    this.#costUsd += Math.max(0, usd);
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

export function subagentFromConfig(name: string, config: Record<string, unknown>): Subagent {
  const description = config.description;
  const mode = config.mode;
  const scope = config.scope;

  return new Subagent(
    name,
    typeof description === 'string' && description !== ''
      ? description
      : `Run the [${name}] subagent and return its result.`,
    typeof mode === 'string' && mode !== '' ? mode : name,
    RunBudget.fromConfig(config),
    typeof scope === 'string' ? scope : null,
  );
}

/**
 * Where a run sits in its tree, and what the tree has left to spend.
 *
 * A run with NO context is a root: that is the ordinary case and stays free of
 * all of this.
 */
export class RunContext {
  constructor(
    readonly ledger: RunLedger,
    readonly budget: RunBudget,
    readonly parentRunId: string | null = null,
    readonly depth: number = 0,
  ) {}

  static root(runId: string, budget: RunBudget): RunContext {
    return new RunContext(RunLedger.start(runId), budget);
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
  forChild(subagent: Subagent, parentRunId: string): RunContext {
    return new RunContext(
      this.ledger,
      subagent.budget.nestedWithin(this.budget, this.ledger),
      parentRunId,
      this.depth + 1,
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
