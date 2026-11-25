import { HarnessError } from './errors.js';
import type { JsonObject } from './json.js';
import { PlanBudget, type PlanBudgetGrant, type PlanBudgetState, type RunContextKind } from './plans/budget.js';
import type { ChildRefs, Session } from './session.js';
import type { SessionStore } from './stores/session-store.js';
import { RunBudget, type RunContext, type RunLedger } from './subagents.js';

/** A run tree: the session key and run id of its root run. */
export interface RunTreeRef {
  session: string;
  run: string;
}

/**
 * Which run tree a run belongs to (G-381), decided once when it begins:
 * `tree` is a durable tree of this runtime, `host` for a tree the host built,
 * `none` for no tree at all. `spends` is the live run whose tree budget it
 * spends from, `route` how its budget is routed (open a tree, continue a
 * saved one, or neither).
 */
export interface TreeMembership {
  tree: RunTreeRef | 'host' | 'none';
  spends: string | null;
  depth: number;
  route: 'open' | 'continue' | 'none';
}

/** The membership a parent run hands its child, as `membership()` reads it. */
export interface ParentMembership {
  tree: RunTreeRef | 'host' | 'none';
  spends: string | null;
  depth: number;
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

interface Member extends TreeMembership {
  session: string;
  ledger: RunLedger;
}

/** Runs of this process holding a run tree's budget, by run id, until they end. */
const runs = new Map<string, RootRun>();

/** The recorded membership of every live run of this process, by run id. */
const members = new Map<string, Member>();

/**
 * The child contexts the subagent runner built, held weakly, with the run
 * whose turn called the subagent tool and that run's session. Only these are
 * the runner's own.
 */
const childContexts = new WeakMap<RunContext, { parentRunId: string | null; parentSessionKey: string }>();

const KINDS: readonly string[] = ['none', 'metadata', 'runner', 'host'];

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
 * Which tree a run is in is decided ONCE, when it begins (membership()), and
 * recorded; every consumer reads that record (G-381).
 *
 * @internal Used by the runtime.
 */
export const RunTrees = {
  /**
   * Which run tree a run belongs to, pure (harness-plans tree_membership).
   * A run the runner started takes its parent run's tree, one level deeper,
   * and never reads what its session saved; a turn the runner did not start
   * continues the tree its session saved; a host context on a session that
   * saved none is in the host's tree; any other run opens a tree of its own,
   * except a root turn resuming its parked run (`resumedTree`), which stays
   * in the tree that run began in while spending from an entry of its own.
   *
   * @throws RangeError for a kind that is not none, metadata, runner or host
   */
  membership(kind: string, session: string, run: string, parent: ParentMembership | null, savedTree: RunTreeRef | null, savedDepth: number, contextDepth: number, resumedTree: RunTreeRef | null = null): TreeMembership {
    if (!KINDS.includes(kind)) throw new RangeError('Unknown run context kind.');
    savedDepth = Math.max(0, savedDepth);
    contextDepth = Math.max(0, contextDepth);
    if (kind === 'runner') {
      return parent === null
        ? { tree: 'none', spends: null, depth: contextDepth, route: 'none' }
        : { tree: copyTree(parent.tree), spends: parent.spends, depth: Math.max(0, parent.depth) + 1, route: 'none' };
    }
    if (savedTree !== null) {
      return { tree: { session: savedTree.session, run: savedTree.run }, spends: run,
        depth: kind === 'host' ? Math.max(savedDepth, contextDepth) : savedDepth, route: 'continue' };
    }
    if (kind === 'host') return { tree: 'host', spends: null, depth: Math.max(savedDepth, contextDepth), route: 'none' };

    const tree = resumedTree === null ? { session, run } : { session: resumedTree.session, run: resumedTree.run };

    return { tree, spends: run, depth: savedDepth, route: 'open' };
  },

  /**
   * The run tree and plan ledger the runner saves with a child, pure
   * (harness-plans kept_refs): the tree it is in, with the plan ledger of the
   * run it spends from. A child in the host's tree, or in none, keeps what it
   * saved before, unless that tree was cancelled.
   */
  keptRefs(membership: TreeMembership, planLedger: ChildRefs['planLedger'], savedTree: RunTreeRef | null, savedPlanLedger: ChildRefs['planLedger'], savedTreeCancelled: boolean): ChildRefs {
    if (membership.tree !== null && typeof membership.tree === 'object') return { tree: { session: membership.tree.session, run: membership.tree.run }, planLedger };

    return savedTreeCancelled ? { tree: null, planLedger: null } : { tree: savedTree, planLedger: savedPlanLedger };
  },

  /**
   * The membership of a run beginning now on `session` under `context` (the
   * context as handed in, metadata-only included). A runner context's parent
   * is the live run the runner marked, accepted only when that run is on the
   * session the runner named AND its ledger is the context's own: proof the
   * context came from that run's turn.
   */
  membershipOf(session: Session, runId: string, context: RunContext | null | undefined, saved: ChildRefs | null, savedDepth: number, resumedTree: RunTreeRef | null = null): TreeMembership {
    const kind = RunTrees.contextKind(context);
    let parent: ParentMembership | null = null;
    if (kind === 'runner' && context != null) {
      const mark = childContexts.get(context);
      const live = mark?.parentRunId == null ? undefined : members.get(mark.parentRunId);
      if (live !== undefined && live.session === mark!.parentSessionKey && live.ledger === context.ledger) {
        parent = { tree: live.tree, spends: live.spends, depth: live.depth };
      }
    }
    const contextDepth = context == null || context.metadataOnly ? 0 : context.depth;

    const bare = kind === 'none' || kind === 'metadata';

    return RunTrees.membership(kind, session.key(), runId, parent, saved?.tree ?? null, savedDepth, contextDepth, bare ? resumedTree : null);
  },

  /** Record a live run's membership, once its run context exists and before any tool runs. */
  enter(session: Session, runId: string, membership: TreeMembership, ledger: RunLedger): void {
    members.set(runId, { ...membership, session: session.key(), ledger });
  },

  /** The recorded membership of a live run of this process, or null. */
  memberOf(runId: string): TreeMembership | null {
    const member = members.get(runId);

    return member === undefined ? null : { tree: member.tree, spends: member.spends, depth: member.depth, route: member.route };
  },

  /**
   * A run is starting, its membership decided: a run that opens a tree is
   * kept in memory until it starts a child; a run that continues a saved tree
   * holds what that tree has left, and its run is narrowed to the hold, so a
   * host context can only lower what the tree allows. Any other run is left
   * alone.
   *
   * @throws HarnessError `run_not_permitted` when the tree the child belongs to is spent
   */
  async rootRun(session: Session, runId: string, run: RunContext, membership: TreeMembership, ttl: number): Promise<RunContext> {
    if (membership.route === 'none' || typeof membership.tree !== 'object') return run;
    const tree = membership.tree;
    const entry: RootRun = {
      session: session.key(), tree,
      budget: run.budget, ledger: run.ledger, steps: run.ledger.steps, cost: run.ledger.costUsd,
      unmetered: run.ledger.unmeteredRuns, held: false, ttl,
    };
    if (membership.route === 'open') {
      runs.set(runId, entry);

      return run;
    }

    const budget = run.budget;
    const turnCost = budget.maxCostUsd === null ? null : PlanBudget.micros(Math.max(0, budget.maxCostUsd - entry.cost));
    const now = Math.floor(Date.now() / 1000);
    const grant = await write(session.planStore(), tree, (state) => state === null
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
   * `runner` for the context the subagent runner built to start a child,
   * `metadata` for a metadata-only one, `host` for any other. An unmarked
   * context is the host's, whatever ledger or run ids it carries.
   */
  contextKind(context: RunContext | null | undefined): RunContextKind {
    if (context == null) return 'none';
    if (childContexts.has(context)) return 'runner';

    return context.metadataOnly ? 'metadata' : 'host';
  },

  /**
   * The subagent runner is starting a child under `context`, which it built,
   * from the turn of `parentRunId` on the session `parentSessionKey` (null
   * when no turn is calling it). Marks it as the runner's own.
   *
   * @internal Called by the subagent runner only.
   */
  startingChild(context: RunContext, parentRunId: string | null, parentSessionKey: string): RunContext {
    childContexts.set(context, { parentRunId, parentSessionKey });

    return context;
  },

  /** The kind of context a turn was handed, for bindingRoute(): the same as contextKind(). */
  bindingContextKind(context: RunContext | null | undefined): RunContextKind {
    return RunTrees.contextKind(context);
  },

  /**
   * A child is starting in the run tree its membership names: that tree,
   * opened in the durable store for the run it spends from the first time
   * that run starts a child. Null when it is in the host's tree or in none.
   */
  async childStarted(child: Session, membership: TreeMembership): Promise<RunTreeRef | null> {
    if (typeof membership.tree !== 'object' || membership.spends === null) return null;
    const spends = membership.spends;
    const entry = runs.get(spends);
    if (entry !== undefined && !entry.held) {
      const settings = {
        mode: 'shared' as const,
        max_steps: Math.max(0, entry.budget.maxSteps - entry.steps),
        max_cost_micros: entry.budget.maxCostUsd === null ? null : PlanBudget.micros(Math.max(0, entry.budget.maxCostUsd - entry.cost)),
      };
      const now = Math.floor(Date.now() / 1000);
      // A root resuming its parked run is in the tree that run began, but
      // spends from its own budget: when that tree's record already exists
      // (the parked run delegated before), it is left alone, so the resumed
      // root never takes from what the subagents parked in it have left.
      // Otherwise it is opened here, named by the tree's root run, and held
      // under the run that spends. Either way it is not tried again; settling
      // a hold that was never taken changes nothing.
      const tree = membership.tree;
      await write(child.planStore(), tree, (state) => state !== null && spends !== tree.run
        ? [null, null]
        : PlanBudget.reserve(state ?? PlanBudget.open(tree.run, settings), spends, Math.max(1, settings.max_steps), settings.max_cost_micros, now, entry.ttl));
      entry.held = true;
    }

    return membership.tree;
  },

  /**
   * Refuse, before anything is recorded, a decision on a child session whose
   * run tree has nothing left: recorded, it would be consumed by a resume the
   * tree then refuses, and the approved call would never run. The decision
   * turn's membership is decided as the turn's own would be.
   *
   * @throws HarnessError `run_not_permitted`
   */
  async assertDecisionsFunded(session: Session, saved: ChildRefs | null, context: RunContext | undefined, ttl: number): Promise<void> {
    const membership = RunTrees.membership(RunTrees.contextKind(context), session.key(), 'decision', null, saved?.tree ?? null, 0, 0);
    if (membership.route !== 'continue' || typeof membership.tree !== 'object') return;
    const state = record(await session.planStore().get(key(membership.tree)), membership.tree);
    if (state === null) throw spent('steps');
    const [, grant] = PlanBudget.reserve(state, 'decision', 1, null, Math.floor(Date.now() / 1000), ttl);
    if ('refused' in grant) throw spent(grant.limit);
  },

  /** A run is over: its membership is dropped, and a run that holds part of a tree settles what the tree spent under it. */
  async finished(session: Session, runId: string): Promise<void> {
    if (members.get(runId)?.session === session.key()) members.delete(runId);
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

function copyTree(tree: RunTreeRef | 'host' | 'none'): RunTreeRef | 'host' | 'none' {
  return typeof tree === 'object' ? { session: tree.session, run: tree.run } : tree;
}

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
