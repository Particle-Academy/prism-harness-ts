import { HarnessError } from './errors.js';
import type { JsonObject } from './json.js';
import { PlanBudget, type PlanBudgetGrant, type PlanBudgetState, type RunContextKind } from './plans/budget.js';
import type { Session } from './session.js';
import type { SessionStore } from './stores/session-store.js';
import { RunBudget, type RunContext, type RunLedger } from './subagents.js';

/** A run tree: the session key and run id of its root run. */
export interface RunTreeRef {
  session: string;
  run: string;
}

interface RootRun {
  session: string;
  tree: RunTreeRef;
  budget: RunBudget;
  ledger: RunLedger;
  steps: number;
  cost: number;
  unmetered: number;
  held: boolean;
  ttl: number;
}

/** Root runs of this process, by run id, until they end. */
const runs = new Map<string, RootRun>();

/**
 * The durable half of a run tree's budget, for subagent children resumed
 * outside the runner (the port of the reference's RunTrees).
 *
 * A child that stops on an approval is resumed later on its own session, in a
 * run the runner did not start. That run must not get a fresh root budget: it
 * continues the run tree it belonged to. While the tree runs, its budget is the
 * in-memory RunLedger every run in it shares. The first time a root run starts
 * a child, the tree's allowance is written to the durable store as a ledger
 * record (the shape PlanBudget keeps for a shared plan budget) with the root
 * run holding all of it; when the root run ends it settles what the whole tree
 * spent. A resumed child then holds what the tree has left before it runs,
 * settles what it spent when it ends, and is refused once the tree's allowance
 * is spent. A hold whose run never settled is charged in full once the run
 * lock's TTL has passed, so a crash can only over-charge the tree.
 *
 * @internal Used by the runtime.
 */
export const RunTrees = {
  /**
   * A run is starting. A turn on a child session the runner did not start (an
   * answer to its approval, a new message, or a run context the host built)
   * continues the tree `savedTree` names: it holds what that tree has left,
   * and its run is narrowed to the hold, so a host context can only lower
   * what the tree allows. Any other root run opens a tree of its own, kept in
   * memory until it starts a child. A run on the runner's own context already
   * spends from its tree and is left alone.
   *
   * @throws HarnessError `run_not_permitted` when the tree the child belongs to is spent
   */
  async rootRun(session: Session, runId: string, run: RunContext, context: RunContext | undefined, savedTree: RunTreeRef | null, ttl: number): Promise<RunContext> {
    const route = PlanBudget.route(RunTrees.contextKind(context), '', false, false, savedTree !== null, false).tree;
    if (route === 'none') return run;
    if (route === 'open') savedTree = null;
    const entry: RootRun = {
      session: session.key(), tree: savedTree ?? { session: session.key(), run: runId },
      budget: run.budget, ledger: run.ledger, steps: run.ledger.steps, cost: run.ledger.costUsd,
      unmetered: run.ledger.unmeteredRuns, held: false, ttl,
    };
    if (savedTree === null) {
      runs.set(runId, entry);

      return run;
    }

    const budget = run.budget;
    const turnCost = budget.maxCostUsd === null ? null : PlanBudget.micros(Math.max(0, budget.maxCostUsd - entry.cost));
    const now = Math.floor(Date.now() / 1000);
    const grant = await write(session.planStore(), savedTree, (state) => state === null
      // The tree's record is gone: closed, never a fresh budget.
      ? [null, { refused: PlanBudget.EXHAUSTED, limit: 'steps' } as PlanBudgetGrant]
      : PlanBudget.reserve(state, runId, Math.max(0, budget.maxSteps - entry.steps), turnCost, now, ttl));
    if ('refused' in grant) throw spent(grant.limit);

    const narrowed = run.withBudget(new RunBudget(
      entry.steps + grant.steps,
      grant.cost_micros === null ? budget.maxCostUsd : entry.cost + grant.cost_micros / 1_000_000,
      budget.maxSeconds,
      budget.maxOutputTokensPerCall,
    ));
    runs.set(runId, { ...entry, budget: narrowed.budget, ledger: narrowed.ledger, held: true });

    return narrowed;
  },

  /**
   * What kind of run context a turn was handed, for PlanBudget.route():
   * `runner` when it is a context of a run this runtime is running (its
   * ledger is that run's), `host` for any other context that states a run
   * tree.
   */
  contextKind(context: RunContext | null | undefined): RunContextKind {
    if (context == null) return 'none';
    if (context.metadataOnly) return 'metadata';

    return runs.get(context.rootRunId)?.ledger === context.ledger ? 'runner' : 'host';
  },

  /** The tree a run of this process belongs to, or null. */
  treeOf(runId: string): RunTreeRef | null {
    return runs.get(runId)?.tree ?? null;
  },

  /**
   * A child is starting under a run tree: the tree it belongs to, opened in
   * the durable store the first time its root starts a child. Null when the
   * tree's root is not a run of this runtime (a context the host built).
   */
  async childStarted(child: Session, context: RunContext): Promise<RunTreeRef | null> {
    const rootRunId = context.rootRunId;
    const entry = runs.get(rootRunId);
    if (entry === undefined) return null;
    if (!entry.held) {
      const settings = {
        mode: 'shared' as const,
        max_steps: Math.max(0, entry.budget.maxSteps - entry.steps),
        max_cost_micros: entry.budget.maxCostUsd === null ? null : PlanBudget.micros(Math.max(0, entry.budget.maxCostUsd - entry.cost)),
      };
      const now = Math.floor(Date.now() / 1000);
      await write(child.planStore(), entry.tree, (state) =>
        PlanBudget.reserve(state ?? PlanBudget.open(rootRunId, settings), rootRunId, Math.max(1, settings.max_steps), settings.max_cost_micros, now, entry.ttl));
      entry.held = true;
    }

    return entry.tree;
  },

  /**
   * Refuse, before anything is recorded, a decision on a child session whose
   * run tree has nothing left: recorded, it would be consumed by a resume the
   * tree then refuses, and the approved call would never run.
   *
   * @throws HarnessError `run_not_permitted`
   */
  async assertDecisionsFunded(session: Session, tree: RunTreeRef | null, context: RunContext | undefined, ttl: number): Promise<void> {
    if (tree === null || PlanBudget.route(RunTrees.contextKind(context), '', false, false, true, false).tree !== 'continue') return;
    const state = record(await session.planStore().get(key(tree)), tree);
    if (state === null) throw spent('steps');
    const [, grant] = PlanBudget.reserve(state, 'decision', 1, null, Math.floor(Date.now() / 1000), ttl);
    if ('refused' in grant) throw spent(grant.limit);
  },

  /** A run is over: a root run that holds part of a tree settles what the tree spent under it. */
  async finished(session: Session, runId: string): Promise<void> {
    const entry = runs.get(runId);
    if (entry === undefined || entry.session !== session.key()) return;
    runs.delete(runId);
    if (!entry.held) return;
    const steps = entry.ledger.steps - entry.steps;
    const cost = entry.ledger.unmeteredRuns > entry.unmetered ? null : PlanBudget.micros(Math.max(0, entry.ledger.costUsd - entry.cost));
    try {
      await write(session.planStore(), entry.tree, (state) => [state === null ? null : PlanBudget.settle(state, runId, steps, cost), null]);
    } catch (error) {
      // The hold stays and is charged in full once it expires.
      process.emitWarning(`A run tree budget update failed: ${error instanceof Error ? error.constructor.name : typeof error}.`);
    }
  },
};

/** One locked change to a tree's record. */
async function write<T>(store: SessionStore, tree: RunTreeRef, change: (state: PlanBudgetState | null) => [PlanBudgetState | null, T]): Promise<T> {
  const name = key(tree);

  return store.withLock(name, async () => {
    const [next, result] = change(record(await store.get(name), tree));
    if (next !== null) await store.put(name, next as unknown as JsonObject);

    return result;
  }, 10, 5);
}

function record(stored: JsonObject | null, tree: RunTreeRef): PlanBudgetState | null {
  return stored !== null && stored.planId === tree.run ? stored as unknown as PlanBudgetState : null;
}

function key(tree: RunTreeRef): string {
  return `${tree.session}:run-tree:${tree.run}`;
}

function spent(limit: string): HarnessError {
  return HarnessError.runNotPermitted(`This resumed subagent may not run: its run tree's ${limit} budget is spent.`);
}
