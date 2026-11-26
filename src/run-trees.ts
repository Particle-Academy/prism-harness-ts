import type { JsonObject } from './json.js';
import { PlanBudget, type PlanBudgetGrant, type PlanBudgetState, type RunContextKind } from './plans/budget.js';
import { RunBudgets, type AllowanceFacts, type StepAllowance } from './run-budgets.js';
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
  /**
   * The facts its step allowance was computed from when it began (G-387),
   * the tree and plan grants included, and its ledger's steps then; null
   * until the allowance is decided.
   */
  allowance: { facts: AllowanceFacts; startSteps: number } | null;
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
  enter(session: Session, runId: string, membership: TreeMembership, run: RunContext): void {
    members.set(runId, { ...membership, session: session.key(), ledger: run.ledger, allowance: null });
  },

  /**
   * Record a live run's step allowance (G-387), decided once when it began,
   * beside its membership. `run` is the run narrowed to it: a run holding a
   * tree's budget keeps that budget, the allowance its children's holds are
   * drawn from.
   */
  allowed(runId: string, facts: AllowanceFacts, startSteps: number, run: RunContext): void {
    const member = members.get(runId);
    if (member !== undefined) member.allowance = { facts: { ...facts }, startSteps };
    const entry = runs.get(runId);
    if (entry !== undefined) {
      entry.budget = run.budget;
      entry.ledger = run.ledger;
    }
  },

  /**
   * A live run's step allowance as recorded when it began, with what it has
   * taken since (its ledger's gain, a runner child's steps included): what
   * it writes on the row it parks on. Null for a run with none recorded.
   */
  allowanceOf(runId: string): StepAllowance | null {
    const member = members.get(runId);
    if (member === undefined || member.allowance === null) return null;
    const { facts, startSteps } = member.allowance;

    return RunBudgets.allowance({ ...facts, took: Math.max(0, member.ledger.steps - startSteps) });
  },

  /** The recorded membership of a live run of this process, or null. */
  memberOf(runId: string): TreeMembership | null {
    const member = members.get(runId);

    return member === undefined ? null : { tree: member.tree, spends: member.spends, depth: member.depth, route: member.route };
  },

  /**
   * A run is starting, its membership decided and its own steps left known
   * (`ownLeft`, RunBudgets.allowance): a run that opens a tree is kept in
   * memory until it starts a child; a run that continues a saved tree, or an
   * open root whose tree already has a record, holds what it may spend of
   * that tree, requested as `ownLeft`, in one locked write. Returns the run,
   * its cost narrowed to the hold, and the tree's step grant: null when no
   * durable tree bounds the run, 0 when the tree's steps or cost are spent,
   * its record is gone, or the run has no step of its own left. A spent
   * tree is never a refusal here (G-387): a resume still acts on the
   * decisions it resumes and takes no model step; the runtime refuses a new
   * prompt that is left no step. Any other run is left alone.
   */
  async holdTree(session: Session, runId: string, run: RunContext, membership: TreeMembership, ttl: number, ownLeft: number): Promise<{ run: RunContext; treeLeft: number | null }> {
    if (membership.route === 'none' || typeof membership.tree !== 'object') return { run, treeLeft: null };
    const tree = membership.tree;
    const entry: RootRun = {
      session: session.key(), tree,
      budget: run.budget, ledger: run.ledger, steps: run.ledger.steps, cost: run.ledger.costUsd,
      unmetered: run.ledger.unmeteredRuns, held: false, ttl,
    };
    if (membership.route === 'open') {
      // An open run whose tree already has a record is held to it, whatever
      // its run id: a root resumed on the stream path continues under the
      // parked run's own id, and the tree's allowance holds across the pause
      // there exactly as on a resume under a new id. One whose tree has no
      // record yet stays open: it opens the record when it first delegates
      // (RunTrees.holdStart). Two resumes of one parked root never overlap: the
      // runtime starts a resumed root under the session lock, so the second
      // sees the first's record. One with no step of its own left has
      // nothing of the tree's to hold.
      if (ownLeft < 1 || record(await session.planStore().get(key(tree)), tree) === null) {
        runs.set(runId, entry);

        return { run, treeLeft: null };
      }
    } else if (ownLeft < 1) return { run, treeLeft: 0 };

    const budget = run.budget;
    const turnCost = budget.maxCostUsd === null ? null : PlanBudget.micros(Math.max(0, budget.maxCostUsd - entry.cost));
    const now = Math.floor(Date.now() / 1000);
    const grant = await write(session.planStore(), tree, (state) => state === null
      // The tree's record is gone: closed, never a fresh budget.
      ? [null, { refused: PlanBudget.EXHAUSTED, limit: 'steps' } as PlanBudgetGrant]
      : PlanBudget.reserve(state, runId, ownLeft, turnCost, now, ttl));
    if ('refused' in grant) return { run, treeLeft: 0 };

    const narrowed = run.withBudget(new RunBudget(
      budget.maxSteps,
      grant.cost_micros === null ? budget.maxCostUsd : entry.cost + grant.cost_micros / 1_000_000,
      budget.maxSeconds,
      budget.maxOutputTokensPerCall,
    ));
    runs.set(runId, { ...entry, budget: narrowed.budget, ledger: narrowed.ledger, held: true });

    return { run: narrowed, treeLeft: grant.steps };
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
   * What the hold a runner start takes on its run tree would grant now
   * (G-387 round 7), read without a write: the tree is opened in the durable
   * store for the run it spends from the first time that run starts a child,
   * holding what that run has left. Null when the start takes no hold (a
   * child in the host's tree or in none, or a spending run that already
   * holds its tree); 0 when the tree's record has nothing left to grant.
   */
  async startGrant(child: Session, membership: TreeMembership): Promise<{ steps: number; request: TreeRequest } | null> {
    const request = treeRequest(membership);
    if (request === null) return null;
    const state = record(await child.planStore().get(key(request.tree)), request.tree);
    const [, grant] = PlanBudget.reserve(state ?? PlanBudget.open(request.tree.run, request.settings), request.spends,
      Math.max(1, request.settings.max_steps), request.settings.max_cost_micros, Math.floor(Date.now() / 1000), request.ttl);

    return { steps: 'refused' in grant ? 0 : grant.steps, request };
  },

  /**
   * A runner start commits its tree hold (G-387 round 7): the hold
   * startGrant() read, in ONE conditional locked write. If the tree no
   * longer grants all of `expected` (a sibling spent it since the read),
   * nothing is written and the start is refused. Returns whether the
   * record was opened by this write, for releaseStart().
   */
  async holdStart(child: Session, request: TreeRequest, expected: number): Promise<{ opened: boolean } | 'refused'> {
    const entry = runs.get(request.spends);
    if (entry === undefined || entry.held) return 'refused';
    const now = Math.floor(Date.now() / 1000);
    const held = await write<{ opened: boolean } | 'refused'>(child.planStore(), request.tree, (state) => {
      const [next, grant] = PlanBudget.reserve(state ?? PlanBudget.open(request.tree.run, request.settings), request.spends,
        Math.max(1, request.settings.max_steps), request.settings.max_cost_micros, now, request.ttl);
      if ('refused' in grant || grant.steps < expected) return [null, 'refused' as const];

      return [next, { opened: state === null }];
    });
    if (held !== 'refused') entry.held = true;

    return held;
  },

  /**
   * Release a hold holdStart() took, for a start that did not go ahead: a
   * record it opened is removed, otherwise its hold is dropped unspent, and
   * the spending run holds nothing again. Isolated: a failed release leaves
   * the hold, which is charged in full once it expires (an over-charge).
   */
  async releaseStart(child: Session, request: TreeRequest, held: { opened: boolean }): Promise<void> {
    const entry = runs.get(request.spends);
    if (entry !== undefined) entry.held = false;
    const store = child.planStore();
    const name = key(request.tree);
    try {
      await store.withLock(name, async () => {
        const state = record(await store.get(name), request.tree);
        if (state === null) return;
        if (held.opened && Object.keys(state.holds).every((id) => id === request.spends) && state.steps === 0 && state.cost_micros === 0) {
          await store.forget(name);
          return;
        }
        await store.put(name, PlanBudget.settle(state, request.spends, 0, 0) as unknown as JsonObject);
      }, 10, 5);
    } catch (error) {
      process.emitWarning(`A run tree budget update failed: ${error instanceof Error ? error.constructor.name : typeof error}.`);
    }
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

/** The hold a runner start would take on its tree: for the run it spends from, what that run has left. */
export interface TreeRequest {
  tree: RunTreeRef;
  spends: string;
  settings: { mode: 'shared'; max_steps: number; max_cost_micros: number | null };
  ttl: number;
}

function treeRequest(membership: TreeMembership): TreeRequest | null {
  if (typeof membership.tree !== 'object' || membership.spends === null) return null;
  const entry = runs.get(membership.spends);
  if (entry === undefined || entry.held) return null;
  // The tree's allowance is the spending run's own step allowance, as
  // recorded when it began: its budget was narrowed to it (allowed()).
  return {
    tree: { session: membership.tree.session, run: membership.tree.run },
    spends: membership.spends,
    settings: {
      mode: 'shared',
      max_steps: Math.max(0, entry.budget.maxSteps - entry.steps),
      max_cost_micros: entry.budget.maxCostUsd === null ? null : PlanBudget.micros(Math.max(0, entry.budget.maxCostUsd - entry.cost)),
    },
    ttl: entry.ttl,
  };
}

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

