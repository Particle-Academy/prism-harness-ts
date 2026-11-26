import { ParticipantRef, SessionAddress } from './address.js';
import { indexedStore, storeLockRank, StoreAddressIndex } from './address-index.js';
import type { JsonObject, JsonValue } from './json.js';
import { isJsonObject } from './json.js';
import type { SessionStore } from './stores/session-store.js';
import { StoreTaskSource } from './store-task-source.js';
import type { StoreTaskSourceOptions } from './store-task-source.js';
import { HarnessError } from './errors.js';
import { Thread } from './thread.js';
import { pausedApprovalIds, storedBinding } from './thread-rows.js';
import type { ModelBinding } from './subagents.js';
import type { AgentRuntime } from './runtime.js';
import { PlansDisabled } from './plans/refusal.js';
import type { PlanHandle, Plans } from './plans/plans.js';

/**
 * Who a session belongs to.
 *
 * A TYPE and an ID, not just an id: one application holds users and teams and
 * bots, and `7` means a different participant in each. The reference gets the
 * same pair from Eloquent's morph class and primary key.
 */
export interface Participant {
  type: string;
  id: string | number;
}

/** A subagent child's run tree and shared plan budget references (G-366). */
export interface ChildRefs {
  tree: { session: string; run: string } | null;
  planLedger: { session: string; plan: string } | null;
}

/** How many recent runs' trees a session keeps durably (G-381), as the reference does. */
const RUN_HISTORY = 20;

export interface RunState extends JsonObject {
  id: string;
  status: 'running' | 'completed' | 'failed';
  mode?: string;
  provider?: string;
  model?: string;
  started_at?: string;
  /** The run tree the run belongs to (G-381): a tree of this runtime, `host` or `none`. */
  tree?: { session: string; run: string } | 'host' | 'none';
  /** The depth the run runs at, decided with its tree. */
  depth?: number;
  /** The last run step `seq` this run emitted, so a resumed stream continues it. */
  last_seq?: number;
}

export interface SessionOptions {
  participant: Participant;
  scope: string;
  ephemeral: SessionStore;
  durable: SessionStore;
  /** How long ephemeral state lives. Null keeps it until forgotten. */
  ttlSeconds?: number | null;
  agent?: string | null;
  actor?: object;
  addressIndex?: StoreAddressIndex;
  maxAddressesPerParticipant?: number | null;
  /** Agent plans (G-360). Off unless given, enabled in its config, and turned on by the session's mode. */
  plans?: Plans;
  /** The runtime a plan's driver runs its turns on (`plan().advance()`). */
  runtime?: AgentRuntime;
}

/**
 * One participant's live runtime, reconstructed per turn.
 *
 * RESOLVED, NEVER HELD. Nothing survives in memory between turns, so a fresh
 * worker resolving the same address has to see the same active mode, the same
 * model and the same conversation as whatever set them. Everything here reads
 * through a store for that reason, rather than being a property that happens to
 * be populated.
 *
 * State is split deliberately:
 *
 *  - mode, model, provider and run bookkeeping are EPHEMERAL. Lose them and the
 *    next turn falls back to a default, which is a shrug.
 *  - the thread and stored capabilities are DURABLE.
 */
export class Session {
  readonly address: SessionAddress;
  readonly actor: object;
  readonly #addressIndex: StoreAddressIndex | undefined;
  readonly #rawDurable: SessionStore;
  readonly #rawEphemeral: SessionStore;
  readonly #lockStore: SessionStore;
  readonly scope: string;

  readonly participant: Participant;

  readonly #ephemeral: SessionStore;

  readonly #durable: SessionStore;

  readonly #ttlSeconds: number | null;

  readonly #plans: Plans | null;
  readonly #runtime: AgentRuntime | null;

  #cachedState: JsonObject | null = null;
  #resolutionOnly = false;
  #resolvedMode: string | null = null;

  constructor(options: SessionOptions) {
    if (options.participant instanceof ParticipantRef && options.actor === undefined) throw new Error('Typed participants require an explicit authorization actor.');
    const strictIdentity = options.agent != null || options.participant instanceof ParticipantRef;
    this.participant = options.participant instanceof ParticipantRef ? options.participant : (strictIdentity ? Object.freeze({ ...options.participant }) : options.participant);
    this.address = new SessionAddress(this.participant, options.scope, options.agent ?? null);
    this.actor = options.actor ?? this.participant;
    const index = options.addressIndex ?? (this.address.agent !== null || options.participant instanceof ParticipantRef
      ? new StoreAddressIndex(options.durable, options.maxAddressesPerParticipant ?? null) : undefined);
    this.#addressIndex = index;
    this.#rawDurable = options.durable;
    this.#rawEphemeral = options.ephemeral;
    this.scope = options.scope;
    this.#ephemeral = index ? indexedStore(options.ephemeral, index, this.address, 'ephemeral') : options.ephemeral;
    this.#durable = index ? indexedStore(options.durable, index, this.address, 'durable') : options.durable;
    this.#lockStore = index && storeLockRank(options.durable) > storeLockRank(options.ephemeral) ? this.#durable : this.#ephemeral;
    this.#ttlSeconds = options.ttlSeconds ?? null;
    this.#plans = options.plans ?? null;
    this.#runtime = options.runtime ?? null;
    for (const key of strictIdentity ? ['address', 'actor', 'participant', 'scope'] : ['address', 'actor']) Object.defineProperty(this, key, { writable: false, configurable: false });
  }

  /**
   * The address this session is resolved by.
   *
   * Participant AND scope, because one participant holds several unrelated
   * conversations at once and they must not collide.
   *
   * The type is HASHED rather than interpolated, and with sha1 truncated to 12
   * characters — byte for byte what the PHP reference produces. A class name
   * contains backslashes there, which make for awkward store keys and leak the
   * application's namespace layout into something visible in tooling. Matching
   * the reference exactly is what lets a PHP app and a TypeScript agent share
   * ONE store and resolve the same session.
   */
  key(): string {
    return this.address.agent === null && !(this.participant instanceof ParticipantRef)
      ? new SessionAddress(this.participant,this.scope).sessionKey() : this.address.sessionKey();
  }

  get agent(): string | null { return this.address.agent; }

  // -- ephemeral half --------------------------------------------------------

  async mode(): Promise<string | null> {
    if (this.#resolvedMode !== null) return this.#resolvedMode;
    return this.#readString('mode');
  }

  async usingMode(mode: string): Promise<this> {
    return this.#write('mode', mode);
  }

  async model(): Promise<string | null> {
    return this.#readString('model');
  }

  async usingModel(model: string): Promise<this> {
    return this.#write('model', model);
  }

  async provider(): Promise<string | null> {
    return this.#readString('provider');
  }

  async usingProvider(provider: string): Promise<this> {
    return this.#write('provider', provider);
  }

  async state(): Promise<JsonObject> {
    this.#cachedState ??= (await this.#ephemeral.get(this.#ephemeralKey())) ?? {};

    return this.#resolutionOnly ? structuredClone(this.#cachedState) : this.#cachedState;
  }

  /** Drop the ephemeral half. THE CONVERSATION IS UNTOUCHED. */
  async forget(): Promise<this> {
    this.#assertWritable();
    await this.#ephemeral.forget(this.#ephemeralKey());
    this.#cachedState = null;

    return this;
  }

  // -- durable half ----------------------------------------------------------

  /** The stored conversation this session is bound to. */
  thread(): Thread {
    this.#assertWritable();
    const store = this.#addressIndex ? indexedStore(this.#rawDurable, this.#addressIndex, this.address, 'durable', 'thread') : this.#durable;
    return new Thread(store, `${this.key()}:thread`);
  }

  async findThread(): Promise<Thread | null> {
    const thread = this.thread();
    return await thread.exists() ? thread : null;
  }

  /**
   * The agent task list this session is bound to.
   *
   * The DURABLE half, alongside the thread, for the same reason: a
   * half-finished task list that vanishes on a deploy is indistinguishable from
   * a finished one. A volatile durable slot has already been refused by the
   * store manager before this can be reached, and `StoreTaskSource` refuses one
   * again on its own -- the second check is what makes a source built directly
   * against a store as safe as one reached through here.
   */
  tasks(options: StoreTaskSourceOptions = {}): StoreTaskSource {
    this.#assertWritable();
    const store = this.#addressIndex ? indexedStore(this.#rawDurable, this.#addressIndex, this.address, 'durable', 'tasks') : this.#durable;
    return new StoreTaskSource(store, `${this.key()}:tasks`, options);
  }

  /**
   * This session's agent plan (G-360).
   *
   * Plans are optional and off unless the harness was given a `Plans` whose
   * config sets `enabled: true` AND the session's mode sets `plans: true`.
   *
   * @throws PlansDisabled (`plan.disabled`) when they are off
   */
  async plan(): Promise<PlanHandle> {
    this.#assertWritable();
    if (this.#plans === null) throw new PlansDisabled();

    return this.#plans.handle(this);
  }

  /** @internal The plans service this session was resolved with, or null. */
  planService(): Plans | null {
    return this.#plans;
  }

  /** @internal The runtime this session was resolved with, or null. */
  planRuntime(): AgentRuntime | null {
    return this.#runtime;
  }

  /**
   * Where this session's plan is stored: the durable store, always. A
   * half-finished plan that vanished on a deploy would read as finished.
   *
   * @internal
   */
  planStore(): SessionStore {
    return this.#durable;
  }

  /**
   * Hold the plan driver's lock while the callback runs, so one advance runs at
   * a time per session. A second advance does not wait: it fails at once with
   * `session_locked` and does nothing.
   *
   * @internal
   */
  async holdPlanLock<T>(callback: () => T | Promise<T>, ttlSeconds: number): Promise<T> {
    this.#assertWritable();

    return this.#lockStore.withLock(`${this.key()}:plan-advance`, callback, ttlSeconds, 0);
  }

  /**
   * The run tree and shared plan budget this session was last started under
   * by the subagent runner, as written by the runtime; null for a session the
   * runner never started, or one written before these were kept here. Every
   * turn on the child the runner did not start (an answer, a new message, a
   * host context) continues them rather than starting fresh.
   *
   * @internal Read by the runtime and Plans.
   */
  async childRefs(): Promise<ChildRefs | null> {
    const stored = (await this.#durable.get(this.#durableKey())) ?? {};
    const refs = stored.child_refs;
    if (!isJsonObject(refs)) return null;
    const pair = (value: JsonValue | undefined, first: string, second: string): Record<string, string> | null => {
      if (!isJsonObject(value)) return null;
      const a = value[first], b = value[second];

      return typeof a === 'string' && a !== '' && typeof b === 'string' && b !== '' ? { [first]: a, [second]: b } : null;
    };

    return {
      tree: pair(refs.tree, 'session', 'run') as ChildRefs['tree'],
      planLedger: pair(refs.plan_ledger, 'session', 'plan') as ChildRefs['planLedger'],
    };
  }

  /**
   * The provider and model binding the subagent runner gave this session's
   * child runs, or null for a session the runner never started (or one
   * started before the binding was kept here).
   *
   * Kept in the DURABLE half, beside the thread and its pending approvals: a
   * child is resumed by a person answering one of those approvals, or sent a
   * new message, long after the ephemeral half may have expired. Every turn
   * on the child the runner does not start resolves its provider and model
   * from this, never from the child's own override slot.
   *
   * @internal Read by the runtime.
   */
  async childBinding(): Promise<ModelBinding | null> {
    const stored = (await this.#durable.get(this.#durableKey())) ?? {};
    const binding = storedBinding(stored.child_binding);

    return binding === null ? null : { session: binding.session, declared: binding.declared, parentMode: binding.parent_mode };
  }

  /**
   * The depth the subagent runner started this session's child run at, or 0
   * for a session the runner never started. Every later turn on the child
   * runs at least this deep, so a resumed child cannot reset its nesting.
   *
   * @internal Read by the runtime.
   */
  async childDepth(): Promise<number> {
    const stored = (await this.#durable.get(this.#durableKey())) ?? {};
    const depth = stored.child_depth;

    return typeof depth === 'number' && Number.isSafeInteger(depth) && depth > 0 ? depth : 0;
  }

  /**
   * The step ceiling the subagent runner gave this session's child run when
   * it last started it (G-387): the least of its declared max_steps, what
   * its parent had left and max_steps_ceiling. Every turn on the child the
   * runner does not start is held to it: it is the delegation's budget for
   * its whole life, until the runner starts it again. Null for a session
   * with none saved; `invalid` for a stored value that is not a
   * non-negative safe integer (the runtime fails closed on it).
   *
   * @internal Read by the runtime.
   */
  async childCeiling(): Promise<number | null | 'invalid'> {
    const stored = (await this.#durable.get(this.#durableKey())) ?? {};
    if (!Object.hasOwn(stored, 'child_ceiling') || stored.child_ceiling === undefined) return null;
    const ceiling = stored.child_ceiling;

    return typeof ceiling === 'number' && Number.isSafeInteger(ceiling) && ceiling >= 0 ? ceiling : 'invalid';
  }

  /**
   * @internal Written by the runtime when the subagent runner starts a child
   * run, with the step ceiling it started it under (null keeps none). A
   * ceiling starts the delegation's step count again at 0, with `hold` (the
   * steps the starting run may take) held under its run id, in this one
   * write.
   */
  async rememberChildBinding(binding: ModelBinding, depth = 0, ceiling: number | null = null, hold: { runId: string; steps: number; at: number } | null = null, ttlSeconds = 300): Promise<this> {
    return this.#writeDurable((stored) => {
      const { child_ceiling: _old, child_steps: count, ...rest } = stored;
      // A new delegation never wipes a turn on the child that is still
      // running: its hold would be lost, and its steps counted nowhere. The
      // runner refuses such a delegation first; this refuses the race.
      if (ceiling !== null && hold !== null && hasLiveHold(count, hold.at, ttlSeconds)) {
        throw HarnessError.runNotPermitted('This subagent may not be started again: a turn on it is still running.');
      }
      const holds: JsonObject = hold === null || hold.steps < 1 ? {} : { [hold.runId]: { steps: hold.steps, at: hold.at } };

      return {
        ...rest,
        child_binding: {
          session: { ...binding.session },
          declared: { ...binding.declared },
          parent_mode: { ...binding.parentMode },
        },
        child_depth: Number.isSafeInteger(depth) && depth > 0 ? depth : 0,
        ...(ceiling === null ? {} : {
          child_ceiling: Number.isSafeInteger(ceiling) && ceiling > 0 ? ceiling : 0,
          child_steps: { steps: 0, holds, charged: {} },
        }),
      };
    });
  }

  /**
   * A runner start commits the child's records (G-387 round 7), in ONE
   * locked write: its binding, depth, the ceiling the runner gave it with
   * its count started again at 0 and `hold` held under the starting run's
   * id, and the run tree and plan ledger it belongs to. It re-checks that no
   * turn on the child holds part of its count (a resume that claimed since
   * the start read it): then nothing is written and false is returned.
   *
   * @internal Written by the runtime when a runner start commits.
   */
  async rememberChildStart(binding: ModelBinding, depth: number, ceiling: number, hold: { runId: string; steps: number; at: number }, ttlSeconds: number, refs: ChildRefs): Promise<boolean> {
    let written = false;
    this.#assertWritable();
    const name = this.#durableKey();
    await this.#durable.withLock(name, async () => {
      const stored = (await this.#durable.get(name)) ?? {};
      const { child_ceiling: _old, child_steps: count, ...rest } = stored;
      if (hasLiveHold(count, hold.at, ttlSeconds)) return;
      const holds: JsonObject = hold.steps < 1 ? {} : { [hold.runId]: { steps: hold.steps, at: hold.at } };
      await this.#durable.put(name, {
        ...rest,
        child_binding: {
          session: { ...binding.session },
          declared: { ...binding.declared },
          parent_mode: { ...binding.parentMode },
        },
        child_depth: Number.isSafeInteger(depth) && depth > 0 ? depth : 0,
        child_ceiling: Number.isSafeInteger(ceiling) && ceiling > 0 ? ceiling : 0,
        child_steps: { steps: 0, holds, charged: {} },
        child_refs: {
          tree: refs.tree === null ? null : { ...refs.tree },
          plan_ledger: refs.planLedger === null ? null : { ...refs.planLedger },
        },
      });
      written = true;
    }, 10, 5);

    return written;
  }

  /**
   * Claim, in ONE locked write, what a turn on this child session may spend
   * of its delegation's step budget (G-387). `claim` is handed the steps the
   * delegation has used since the runner last started it, live holds
   * included (an expired hold is charged in full first), or null when that
   * count is missing or malformed, and returns the steps to hold under
   * `runId`. Two turns on one child can therefore never both spend the same
   * remainder.
   *
   * @internal Called by the runtime when a turn the runner did not start begins.
   */
  async claimChildSteps(runId: string, now: number, ttlSeconds: number, claim: (used: number | null) => number): Promise<void> {
    await this.#writeDurable((stored) => {
      const read = childSteps(stored.child_steps);
      // A run id that already holds (or was charged) is not a new turn: two
      // live turns under one id would overwrite each other's hold. Closed.
      const count = read === null || Object.hasOwn(read.holds, runId) || Object.hasOwn(read.charged, runId) ? null : expired(read, now, ttlSeconds);
      const hold = Math.max(0, Math.trunc(claim(count === null ? null : count.steps + heldSteps(count.holds))));
      // Never repaired: a record that is not a count stays as it is.
      if (count === null) return stored;
      const holds: Record<string, { steps: number; at: number }> = { ...count.holds };
      if (hold > 0) holds[runId] = { steps: hold, at: now };

      return { ...stored, child_steps: { steps: count.steps, holds, charged: count.charged } };
    });
  }

  /**
   * True when a turn on this child session holds part of its delegation's
   * step budget and its hold has not expired: a turn on it is running.
   *
   * @internal Read by the subagent runner before it starts the child again.
   */
  async childTurnLive(now: number, ttlSeconds: number): Promise<boolean> {
    const stored = (await this.#durable.get(this.#durableKey())) ?? {};

    return hasLiveHold(stored.child_steps, now, ttlSeconds);
  }

  /**
   * Replace a turn's charge on its delegation's step budget with the steps it
   * took (G-387): its hold if it is still stored, or the full charge made
   * when another turn found it expired, never both. The run's own entry is
   * looked up BEFORE any other hold is expired, so a turn that outlived the
   * lock TTL is still charged what it took. A charge already gone (a new
   * runner start reset the count) changes nothing, and a record that is not
   * a count is never repaired.
   *
   * @internal Called by the runtime when the turn ends, parks or fails.
   */
  async settleChildSteps(runId: string, took: number, now: number, ttlSeconds: number): Promise<void> {
    await this.#writeDurable((stored) => {
      const read = childSteps(stored.child_steps);
      if (read === null) return stored;
      const actual = Math.max(0, Math.trunc(took));
      let steps = read.steps;
      const holds = { ...read.holds };
      const charged = { ...read.charged };
      if (Object.hasOwn(holds, runId)) {
        delete holds[runId];
        steps += actual;
      } else if (Object.hasOwn(charged, runId)) {
        steps = Math.max(0, steps - charged[runId]!) + actual;
        delete charged[runId];
      } else return stored;
      const count = expired({ steps, holds, charged }, now, ttlSeconds);

      return { ...stored, child_steps: { steps: count.steps, holds: count.holds, charged: count.charged } };
    });
  }

  /** @internal Written by the runtime when the subagent runner starts a child run. */
  async rememberChildRefs(refs: ChildRefs): Promise<this> {
    return this.#writeDurable((stored) => ({
      ...stored,
      child_refs: {
        tree: refs.tree === null ? null : { ...refs.tree },
        plan_ledger: refs.planLedger === null ? null : { ...refs.planLedger },
      },
    }));
  }

  /**
   * The durable run tree a recent run of this session began in (G-381), or
   * null for a run in the host's tree, in none, or not remembered.
   *
   * @internal Read by the runtime.
   */
  async runTree(runId: string): Promise<{ session: string; run: string } | null> {
    const stored = (await this.#durable.get(this.#runTreesKey())) ?? {};
    const runs = stored.runs;
    const entry = isJsonObject(runs) && Object.hasOwn(runs, runId) ? runs[runId] : undefined;
    const tree = isJsonObject(entry) ? entry.tree : undefined;
    if (!isJsonObject(tree)) return null;
    const { session, run } = tree;

    return typeof session === 'string' && session !== '' && typeof run === 'string' && run !== '' ? { session, run } : null;
  }

  /**
   * The durable run tree the run that issued this session's paused approvals
   * began in (G-381), so a root resuming that run stays in it, after the
   * ephemeral half has expired too; null when nothing is paused, the run is
   * not remembered, or it was in the host's tree or in none.
   *
   * @internal Read by the runtime.
   */
  async resumedRunTree(): Promise<{ session: string; run: string } | null> {
    for (const runId of await this.#parkedRunIds()) {
      const tree = await this.runTree(runId);
      if (tree !== null) return tree;
    }

    return null;
  }

  /** The runs that issued the approvals the thread is parked on. */
  async #parkedRunIds(): Promise<string[]> {
    const entries = await this.thread().messages();
    const runs: string[] = [];
    for (const approvalId of pausedApprovalIds(entries.map((entry) => entry.message))) {
      const issued = entries.find((entry) => Array.isArray(entry.message.tool_approval_requests)
        && (entry.message.tool_approval_requests as JsonObject[]).some((request) => isJsonObject(request) && request.approval_id === approvalId));
      if (issued?.runId != null && !runs.includes(issued.runId)) runs.push(issued.runId);
    }

    return runs;
  }

  /**
   * Keep a run's tree under its OWN durable key (G-381), so starting a run
   * never rewrites the session's durable record (child refs, binding,
   * capabilities a host may change from another process). Written in
   * beginRun under a store lock on its own key, since send() holds no
   * session lock and two turns may begin on one session at once. The newest
   * RUN_HISTORY runs are kept, by an explicit sequence rather than key
   * order, and a run whose approvals the thread is still parked on is never
   * dropped: a resume that keeps failing must not forget the tree it resumes.
   */
  async #rememberRunTree(id: string, tree: { session: string; run: string } | 'host' | 'none', depth: number): Promise<void> {
    this.#assertWritable();
    // Read before the lock: stale, it can only keep an extra entry or miss
    // pinning a run that parked a moment ago, which has the newest seq.
    const parked = new Set(await this.#parkedRunIds());
    const name = this.#runTreesKey();
    // Under a lock on that key: send() holds no session lock, and two turns
    // beginning together on one session must both be kept.
    await this.#durable.withLock(name, async () => {
      const stored = (await this.#durable.get(name)) ?? {};
      const runs: JsonObject = isJsonObject(stored.runs) ? { ...stored.runs } : {};
      const seq = typeof stored.seq === 'number' && Number.isSafeInteger(stored.seq) ? stored.seq + 1 : 1;
      runs[id] = { tree, depth, seq };
      const order = Object.keys(runs).map((runId) => {
        const entry = runs[runId];

        return { runId, at: isJsonObject(entry) && typeof entry.seq === 'number' ? entry.seq : 0 };
      }).sort((a, b) => b.at - a.at);
      const kept: JsonObject = {};
      let count = 0;
      for (const { runId } of order) {
        if (count < RUN_HISTORY || parked.has(runId)) {
          kept[runId] = runs[runId]!;
          count += 1;
        }
      }
      await this.#durable.put(name, { seq, runs: kept });
    }, 10, 5);
  }

  #runTreesKey(): string {
    return `${this.key()}:run-trees`;
  }

  async capability(name: string): Promise<JsonObject | null> {
    const stored = (await this.#durable.get(this.#durableKey())) ?? {};
    const capabilities = isJsonObject(stored.capabilities) ? stored.capabilities : {};
    const capability = capabilities[name];

    return isJsonObject(capability) ? (this.#resolutionOnly ? structuredClone(capability) : capability) : null;
  }

  async usingCapability(name: string, state: JsonObject): Promise<this> {
    return this.#writeCapabilities((capabilities) => ({ ...capabilities, [name]: state }));
  }

  async forgetCapability(name: string): Promise<this> {
    return this.#writeCapabilities(({ [name]: _dropped, ...rest }) => rest);
  }

  // -- runs ------------------------------------------------------------------

  async run(): Promise<RunState | null> {
    const run = (await this.state()).run;

    return isJsonObject(run) && typeof run.id === 'string' ? (run as RunState) : null;
  }

  /**
   * @param tree the run tree the run belongs to, decided once when it begins
   *   (RunTrees.membership): a tree of this runtime, `host` or `none` (G-381)
   * @param depth the depth it runs at, from the same decision
   */
  async beginRun(id: string, mode: string, provider: string, model: string, tree?: { session: string; run: string } | 'host' | 'none', depth?: number): Promise<this> {
    if (tree !== undefined) await this.#rememberRunTree(id, typeof tree === 'object' ? { session: tree.session, run: tree.run } : tree, depth ?? 0);

    return this.#write('run', {
      id,
      status: 'running',
      mode,
      provider,
      model,
      started_at: new Date().toISOString(),
      ...(tree === undefined ? {} : { tree: typeof tree === 'object' ? { session: tree.session, run: tree.run } : tree }),
      ...(depth === undefined ? {} : { depth }),
    });
  }

  /**
   * @param toolCalls the NAMES of the tools this run invoked, in order.
   *
   * NAMES ONLY, and that boundary is deliberate. "Which tools did this run
   * reach for" is what an operator needs to audit a guardrail, and a tool name
   * is not PII. ARGUMENTS are — `prism-opentelemetry` already carries them
   * behind an opt-in capture gate with a length cap, and recording them a
   * second time here, ungated, would quietly undo that decision for everyone
   * who installed both.
   */
  async completeRun(id: string, finishReason: string, toolCalls: readonly string[] = []): Promise<this> {
    return this.#finishRun(id, 'completed', {
      finish_reason: finishReason,
      tool_calls: [...toolCalls],
    });
  }

  async failRun(id: string, failure: string): Promise<this> {
    return this.#finishRun(id, 'failed', { failure });
  }

  /**
   * The last run step `seq` this run emitted, so a resumed stream continues it.
   * A run that is not the one in flight is left alone, as `completeRun()` does.
   *
   * @internal Written by the runtime after a stepped run closes.
   */
  async rememberStepSeq(id: string, seq: number): Promise<this> {
    if (!Number.isSafeInteger(seq) || seq < 0) throw new TypeError('A step sequence must be a nonnegative integer.');
    const current = await this.run();

    if (current === null || current.id !== id) return this;

    return this.#write('run', { ...current, last_seq: seq });
  }

  /**
   * Run something that MUST NOT HAPPEN TWICE.
   *
   * Two workers can hold the same session at the same moment: a queued job
   * finishing a run while the user sends another message is ordinary. Advance a
   * run inside this, not outside it.
   */
  async lock<T>(callback: (session: this) => T | Promise<T>, ttlSeconds = 10, waitSeconds = 5): Promise<T> {
    this.#assertWritable();
    return this.#lockStore.withLock(
      this.key(),
      async () => {
        // Re-read inside the lock. State written by whoever held it before us
        // is otherwise invisible to this instance, and acting on a stale read
        // is the thing the lock exists to prevent.
        this.#cachedState = null;

        return callback(this);
      },
      ttlSeconds,
      waitSeconds,
    );
  }

  // -- internals -------------------------------------------------------------

  #ephemeralKey(): string {
    return `${this.key()}:ephemeral`;
  }

  #durableKey(): string {
    return `${this.key()}:durable`;
  }

  /** Refuse explicit logical mutations; ordinary get() retains expired-entry cleanup. */
  async resolveReadOnly<T>(mode: string, resolve: (session: Session) => T | Promise<T>): Promise<T> {
    const view = new Session({ participant: this.participant, scope: this.scope, agent: this.agent, actor: this.actor,
      ephemeral: this.#rawEphemeral, durable: this.#rawDurable, addressIndex: this.#addressIndex, ttlSeconds: this.#ttlSeconds });
    view.#resolutionOnly = true; view.#resolvedMode = mode;
    try { return await resolve(view); }
    finally { view.#resolutionOnly = false; view.#cachedState = null; }
  }

  #assertWritable(): void {
    if (this.#resolutionOnly) throw new Error('Harness tool resolution is read-only.');
  }

  async #readString(key: string): Promise<string | null> {
    const value = (await this.state())[key];

    return typeof value === 'string' ? value : null;
  }

  async #write(key: string, value: JsonValue): Promise<this> {
    this.#assertWritable();
    const state = { ...(await this.state()), [key]: value };
    await this.#ephemeral.put(this.#ephemeralKey(), state, this.#ttlSeconds);
    this.#cachedState = state;

    return this;
  }

  async #writeCapabilities(
    mutate: (capabilities: JsonObject) => JsonObject,
  ): Promise<this> {
    return this.#writeDurable((stored) => ({
      ...stored,
      capabilities: mutate(isJsonObject(stored.capabilities) ? stored.capabilities : {}),
    }));
  }

  /**
   * Every read-modify-write of the session's durable record (capabilities
   * the host keeps, a child's binding and refs the runner saves) under one
   * lock on that record's key, so a write from one worker never undoes a
   * write from another: a capability revoked while a parent delegates stays
   * revoked.
   */
  async #writeDurable(change: (stored: JsonObject) => JsonObject): Promise<this> {
    this.#assertWritable();
    const name = this.#durableKey();
    await this.#durable.withLock(name, async () => {
      await this.#durable.put(name, change((await this.#durable.get(name)) ?? {}));
    }, 10, 5);

    return this;
  }

  async #finishRun(id: string, status: 'completed' | 'failed', extra: JsonObject): Promise<this> {
    const current = await this.run();

    // A run that is not the one in flight does not overwrite it. A late worker
    // reporting on a superseded run would otherwise mark the live one finished.
    if (current !== null && current.id !== id) {
      return this;
    }

    await this.#write('run', {
      ...(current ?? { id }),
      id,
      status,
      finished_at: new Date().toISOString(),
      ...extra,
    });

    // G-360: when the run is over, the plan reads the thread it left: the
    // approvals the run staged, and the ones a new message ended.
    await this.#plans?.afterTurn(this);

    return this;
  }
}

interface ChildCount {
  steps: number;
  holds: Record<string, { steps: number; at: number }>;
  /** The full charge made for each run whose hold expired, until that run settles. */
  charged: Record<string, number>;
}

/** A non-negative safe integer. */
function count(value: JsonValue | undefined): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0;
}

/**
 * A delegation's step count as stored, or null when the record, `steps` or
 * `holds` is missing, or any of them, a hold's `steps` or `at`, or a charge,
 * is not a non-negative safe integer: the turn then fails closed. `charged`
 * may be absent (none).
 */
function childSteps(value: JsonValue | undefined): ChildCount | null {
  if (!isJsonObject(value) || !isJsonObject(value.holds) || !count(value.steps)) return null;
  const holds: ChildCount['holds'] = {};
  for (const [id, hold] of Object.entries(value.holds)) {
    if (!isJsonObject(hold) || !count(hold.steps) || !count(hold.at)) return null;
    holds[id] = { steps: hold.steps, at: hold.at };
  }
  const charged: ChildCount['charged'] = {};
  if (value.charged !== undefined) {
    if (!isJsonObject(value.charged)) return null;
    for (const [id, steps] of Object.entries(value.charged)) {
      if (!count(steps)) return null;
      charged[id] = steps;
    }
  }

  return { steps: value.steps, holds, charged };
}

/** The count with every hold older than the TTL charged in full, and its charge kept under its run id. */
function expired(read: ChildCount, now: number, ttlSeconds: number): ChildCount {
  let steps = read.steps;
  const holds: ChildCount['holds'] = {};
  const charged = { ...read.charged };
  for (const [id, hold] of Object.entries(read.holds)) {
    if (now - hold.at >= ttlSeconds) {
      steps += hold.steps;
      charged[id] = (charged[id] ?? 0) + hold.steps;
    } else holds[id] = hold;
  }

  return { steps, holds, charged };
}

/** A stored count with a hold that has not expired. A record that is not a count has none (a runner start may replace it). */
function hasLiveHold(value: JsonValue | undefined, now: number, ttlSeconds: number): boolean {
  const read = childSteps(value);

  return read !== null && Object.values(read.holds).some((hold) => now - hold.at < ttlSeconds);
}

function heldSteps(holds: Record<string, { steps: number }>): number {
  return Object.values(holds).reduce((sum, hold) => sum + hold.steps, 0);
}
