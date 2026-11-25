import { ParticipantRef, SessionAddress } from './address.js';
import { indexedStore, storeLockRank, StoreAddressIndex } from './address-index.js';
import type { JsonObject, JsonValue } from './json.js';
import { isJsonObject } from './json.js';
import type { SessionStore } from './stores/session-store.js';
import { StoreTaskSource } from './store-task-source.js';
import type { StoreTaskSourceOptions } from './store-task-source.js';
import { Thread } from './thread.js';
import { pausedApprovalIds, pendingApprovalIds, storedBinding } from './thread-rows.js';
import type { ThreadMessage } from './thread.js';
import type { HarnessEvent } from './events.js';
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

export interface RunState extends JsonObject {
  id: string;
  status: 'running' | 'completed' | 'failed' | 'cancelled';
  mode?: string;
  provider?: string;
  model?: string;
  started_at?: string;
  /** The last run step `seq` this run emitted, so a resumed stream continues it. */
  last_seq?: number;
  /** The run's tags, so a cancel from another process carries them (G-285). */
  parent_run_id?: string | null;
  root_run_id?: string | null;
  turn_id?: string | null;
  /** When the run stopped on an approval. A parked run in this port stays `running`. */
  parked_at?: string;
  /** What a run a person cancelled would otherwise have ended as (G-285). */
  ended_as?: string;
}

/**
 * What `Session.cancel()` did (G-285). The same wire strings in every port.
 *
 * - `ok`: the cancel flag is set. A running run stops at its next step or
 *   tool; a run parked on an approval is settled now.
 * - `not-running`: one of this session's runs that has already ended.
 *   Nothing changed.
 * - `no-turn`: this session has no run at all. Nothing changed.
 * - `unsupported`: this session cannot hold a cancel another process can see
 *   (it was built without a runtime, or its lock store cannot lock). Nothing
 *   changed.
 * - `refused`: the id is not one of this session's runs: malformed, made up,
 *   or another session's. Nothing changed.
 */
export type CancelOutcome = 'ok' | 'not-running' | 'no-turn' | 'unsupported' | 'refused';

/** Where an approval stands (G-285). See `Session.approvalStatus()`. */
export type ApprovalStatus = 'pending' | 'approved' | 'denied' | 'cancelled' | 'superseded' | 'unknown';

/**
 * This port's run id: a lowercase RFC 4122 v4 UUID, as `randomUUID()` writes
 * it. Matched against the WHOLE string, with nothing trimmed: JavaScript's `$`
 * without the `m` flag does not accept a trailing newline.
 */
const RUN_ID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

/**
 * How many run ids a session remembers, so cancel() can tell one of its own
 * earlier runs (`not-running`) from an id it never issued (`refused`). The
 * same bound as the PHP reference.
 */
const RUN_HISTORY = 20;

/** The run tags `beginRun()` records (G-285). */
export interface RunTags {
  parentRunId?: string | null;
  rootRunId?: string | null;
  turnId?: string | null;
}

/** The run that asked for an approval id, read from the stored row that carries the request; null when no row issued it. */
function approvalRunId(entries: readonly ThreadMessage[], approvalId: string): string | null {
  for (let index = entries.length - 1; index >= 0; index -= 1) {
    const entry = entries[index]!;
    if (entry.message.type !== 'assistant' || !Array.isArray(entry.message.tool_approval_requests)) continue;
    for (const request of entry.message.tool_approval_requests) {
      if (isJsonObject(request) && request.approval_id === approvalId) {
        return typeof entry.runId === 'string' && entry.runId !== '' ? entry.runId : null;
      }
    }
  }

  return null;
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

  /** @internal Written by the runtime when the subagent runner starts a child run. */
  async rememberChildBinding(binding: ModelBinding): Promise<this> {
    this.#assertWritable();
    const stored = (await this.#durable.get(this.#durableKey())) ?? {};
    await this.#durable.put(this.#durableKey(), {
      ...stored,
      child_binding: {
        session: { ...binding.session },
        declared: { ...binding.declared },
        parent_mode: { ...binding.parentMode },
      },
    });

    return this;
  }

  /** @internal Written by the runtime when the subagent runner starts a child run. */
  async rememberChildRefs(refs: ChildRefs): Promise<this> {
    this.#assertWritable();
    const stored = (await this.#durable.get(this.#durableKey())) ?? {};
    await this.#durable.put(this.#durableKey(), {
      ...stored,
      child_refs: {
        tree: refs.tree === null ? null : { ...refs.tree },
        plan_ledger: refs.planLedger === null ? null : { ...refs.planLedger },
      },
    });

    return this;
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

  /**
   * This session's latest run, or null when it has none.
   *
   * A run parked on an approval when a person cancelled it reads `status`
   * `cancelled` (G-285): the cancel is durable even though the parked run had
   * no process to write it.
   */
  async run(): Promise<RunState | null> {
    const run = (await this.state()).run;
    if (!isJsonObject(run) || typeof run.id !== 'string') return null;
    const state = run as RunState;
    const idle = state.status !== 'running' || typeof state.parked_at === 'string';
    if (idle && state.status !== 'cancelled' && await this.cancelRequestedFor(state.id)) {
      return { ...state, status: 'cancelled' };
    }

    return state;
  }

  async beginRun(id: string, mode: string, provider: string, model: string, tags: RunTags = {}): Promise<this> {
    const state = await this.state();
    const earlier = Array.isArray(state.run_history) ? state.run_history.filter((entry): entry is string => typeof entry === 'string') : [];
    const history = [...new Set([...earlier, id])].slice(-RUN_HISTORY);

    return this.#writeAll({
      run: {
        id,
        status: 'running',
        mode,
        provider,
        model,
        started_at: new Date().toISOString(),
        // The run's tags, so a cancel from another process carries them.
        parent_run_id: tags.parentRunId ?? null,
        root_run_id: tags.rootRunId ?? null,
        turn_id: tags.turnId ?? null,
      },
      run_history: history,
    });
  }

  /**
   * Ask this session's run to stop (G-285), from any process.
   *
   * The flag is DURABLE: written to the session's durable store, so a worker
   * running the run sees it at its next check even when the caller is a
   * different process. The run checks it before every model step and before
   * every tool call, and stops at the first check after the flag is set. A
   * tool call already executing is NOT interrupted: it finishes, its result is
   * recorded, and the next step or tool is refused. The run then ends
   * `cancelled`, with `run.cancelled`, and never resumes. Cancelling a run
   * stops the whole run tree it belongs to: its subagents' child runs too.
   *
   * A run parked on an approval is cancelled at once: every approval it is
   * waiting on reads as settled-cancelled (`approvalStatus()`), and deciding
   * one is refused.
   *
   * The id must be this session's run, matched exactly: an id this session
   * never issued is `refused`, whatever session it belongs to.
   *
   * WHO MAY CANCEL IS THE APPLICATION'S DECISION, as for approvals: the
   * session is scoped to a participant, but whether this user may stop THIS
   * run is a question only the host can answer. Authorize before calling.
   */
  async cancel(runId: string): Promise<CancelOutcome> {
    this.#assertWritable();
    const runtime = this.#runtime;
    if (runtime === null || this.#lockStore.lockScope?.() === 'unsupported') return 'unsupported';
    if (typeof runId !== 'string' || !RUN_ID.test(runId)) return 'refused';

    const events: HarnessEvent[] = [];
    const outcome = await this.#underCancelLock(async (): Promise<CancelOutcome> => {
      // Read fresh: the run is written by whichever process executes it.
      this.#cachedState = null;
      const state = await this.state();
      const run = isJsonObject(state.run) && typeof state.run.id === 'string' ? state.run as RunState : null;
      if (run === null) return 'no-turn';
      if (run.id !== runId) {
        const history = Array.isArray(state.run_history) ? state.run_history : [];

        return history.includes(runId) ? 'not-running' : 'refused';
      }
      const tags = {
        runId,
        sessionKey: this.key(),
        parentRunId: typeof run.parent_run_id === 'string' ? run.parent_run_id : null,
        rootRunId: typeof run.root_run_id === 'string' ? run.root_run_id : null,
        turnId: typeof run.turn_id === 'string' ? run.turn_id : null,
        ...(this.agent === null ? {} : { harnessAgent: this.agent }),
      };
      const requested = await this.cancelRequestedFor(runId);
      // A parked run in this port keeps `running`; `parked_at` says no
      // process is executing it.
      if (run.status === 'running' && typeof run.parked_at !== 'string') {
        // Twice is harmless: the flag is already set, and the run stops
        // once. Only the first cancel announces it.
        if (!requested) {
          await this.#recordCancel(runId);
          events.push({ type: 'run.cancel_requested', ...tags, state: 'running', at: new Date().toISOString() });
        }

        return 'ok';
      }
      if (requested || run.status !== 'running' || !(await this.#parkedOn(runId))) return 'not-running';
      // Parked on a person: no process is running it, so it is settled here.
      // Its approvals now read as cancelled and cannot be decided.
      await this.#recordCancel(runId);
      const at = new Date().toISOString();
      events.push({ type: 'run.cancel_requested', ...tags, state: 'parked', at });
      events.push({ type: 'run.cancelled', ...tags, at });

      return 'ok';
    });
    for (const event of events) runtime.announce(event);

    return outcome;
  }

  /**
   * Where an approval stands (G-285): `pending`, `approved`, `denied`,
   * `cancelled` (its run was cancelled before anyone decided), `superseded`
   * (never decided, and a later turn ended it), or `unknown` (this thread
   * never issued it). An approval in a cancelled run never reads `pending`.
   */
  async approvalStatus(approvalId: string): Promise<ApprovalStatus> {
    const entries = await this.thread().messages();
    const runId = approvalRunId(entries, approvalId);
    if (runId === null) return 'unknown';
    // The FIRST answer to an approval is the answer.
    for (const entry of entries) {
      const row = entry.message;
      const answers = row.type === 'tool_result' && Array.isArray(row.tool_approval_responses) ? row.tool_approval_responses
        : row.type === 'tool_approval_response' ? [row] : [];
      for (const answer of answers) {
        if (isJsonObject(answer) && answer.approval_id === approvalId) return answer.approved === true ? 'approved' : 'denied';
      }
    }
    if (await this.cancelRequestedFor(runId)) return 'cancelled';

    return pendingApprovalIds(entries.map((entry) => entry.message)).includes(approvalId) ? 'pending' : 'superseded';
  }

  /**
   * Whether a person cancelled this run (G-285), read from the durable store.
   *
   * @internal Read by the runtime's cancel checks.
   */
  async cancelRequestedFor(runId: string): Promise<boolean> {
    const record = await this.#durable.get(this.#cancelKey());
    const runs = isJsonObject(record?.runs) ? record.runs : {};

    return Object.hasOwn(runs, runId);
  }

  /**
   * Whether an approval id belongs to a run a person cancelled (G-285).
   *
   * @internal Read by the decision writer.
   */
  async approvalCancelled(approvalId: string): Promise<boolean> {
    const runId = approvalRunId(await this.thread().messages(), approvalId);

    return runId !== null && await this.cancelRequestedFor(runId);
  }

  /**
   * Whether an empty-prompt turn would continue a run parked on approvals a
   * person cancelled (G-285). Such a turn is refused: a cancelled run never
   * resumes.
   *
   * @internal
   */
  async resumesCancelledRun(prompt: string): Promise<boolean> {
    if (prompt !== '') return false;
    const entries = await this.thread().messages();
    for (const approvalId of pausedApprovalIds(entries.map((entry) => entry.message))) {
      const runId = approvalRunId(entries, approvalId);
      if (runId !== null && await this.cancelRequestedFor(runId)) return true;
    }

    return false;
  }

  /**
   * End a run as cancelled (G-285).
   *
   * @internal Called by the runtime when a cancel stopped the run.
   */
  async cancelRun(id: string): Promise<this> {
    return this.#finishRun(id, 'cancelled', { failure: 'cancelled' });
  }

  /**
   * Whether this run ended cancelled (G-285).
   *
   * @internal
   */
  async runWasCancelled(id: string): Promise<boolean> {
    const run = await this.run();

    return run !== null && run.id === id && run.status === 'cancelled';
  }

  /**
   * Note that this run stopped on an approval, under the cancel lock (G-285).
   * False, and the run ended cancelled, when a person's cancel was already
   * set: a cancel that returned `ok` is never followed by a run left waiting.
   *
   * @internal Called by the runtime as a run parks.
   */
  async parkRun(id: string): Promise<boolean> {
    return this.#underCancelLock(async () => {
      const current = await this.run();
      if (current === null || current.id !== id) return true;
      if (await this.cancelRequestedFor(id)) {
        await this.#write('run', { ...current, status: 'cancelled', failure: 'cancelled', ended_as: 'awaiting_approval', finished_at: new Date().toISOString() });

        return false;
      }
      await this.#write('run', { ...current, parked_at: new Date().toISOString() });

      return true;
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

  #cancelKey(): string {
    return `${this.key()}:cancel`;
  }

  /** Whether this session's thread ends on approvals this run issued that nobody has decided. */
  async #parkedOn(runId: string): Promise<boolean> {
    const entries = await this.thread().messages();

    return pendingApprovalIds(entries.map((entry) => entry.message)).some((approvalId) => approvalRunId(entries, approvalId) === runId);
  }

  async #recordCancel(runId: string): Promise<void> {
    const record = await this.#durable.get(this.#cancelKey());
    const runs: JsonObject = isJsonObject(record?.runs) ? { ...record.runs } : {};
    delete runs[runId];
    runs[runId] = new Date().toISOString();
    // Bounded: the newest flags are the ones a run or an approval can still ask about.
    await this.#durable.put(this.#cancelKey(), { runs: Object.fromEntries(Object.entries(runs).slice(-RUN_HISTORY)) });
  }

  /**
   * Run a cancel decision, or a run's end, under this session's cancel lock,
   * so a cancel racing a run's last step cannot both return `ok` and leave the
   * run `completed`.
   *
   * Its own key, never the session lock, which the running run holds. The
   * order is always session lock, then cancel lock, so they cannot deadlock.
   */
  async #underCancelLock<T>(work: () => Promise<T>): Promise<T> {
    if (this.#lockStore.lockScope?.() === 'unsupported') return work();

    return this.#lockStore.withLock(`${this.key()}:cancel-lock`, work, 10, 10);
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
    return this.#writeAll({ [key]: value });
  }

  /** Several keys written together, in one store write. */
  async #writeAll(values: JsonObject): Promise<this> {
    this.#assertWritable();
    const state = { ...(await this.state()), ...values };
    await this.#ephemeral.put(this.#ephemeralKey(), state, this.#ttlSeconds);
    this.#cachedState = state;

    return this;
  }

  async #writeCapabilities(
    mutate: (capabilities: JsonObject) => JsonObject,
  ): Promise<this> {
    this.#assertWritable();
    const stored = (await this.#durable.get(this.#durableKey())) ?? {};
    const capabilities = isJsonObject(stored.capabilities) ? stored.capabilities : {};
    await this.#durable.put(this.#durableKey(), { ...stored, capabilities: mutate(capabilities) });

    return this;
  }

  async #finishRun(id: string, status: 'completed' | 'failed' | 'cancelled', extra: JsonObject): Promise<this> {
    // Under the cancel lock (G-285): a cancel that set the flag before this
    // point ends the run cancelled, even when its last step had already
    // finished, so a cancel that returned `ok` is never followed by a run
    // that reads `completed`. Whichever takes the lock second sees the other.
    const written = await this.#underCancelLock(async () => {
      const current = await this.run();

      // A run that is not the one in flight does not overwrite it. A late worker
      // reporting on a superseded run would otherwise mark the live one finished.
      if (current !== null && current.id !== id) {
        return false;
      }

      let ending: string = status, details = extra;
      const flagged = await this.cancelRequestedFor(id);
      if (status !== 'cancelled' && flagged) {
        details = { ...extra, failure: 'cancelled', ended_as: status };
        ending = 'cancelled';
      }
      // A run stopped because its TREE was cancelled carries its own flag
      // too, so its approvals read cancelled and it never resumes.
      if (ending === 'cancelled' && !flagged) await this.#recordCancel(id);
      await this.#write('run', {
        ...(current ?? { id }),
        id,
        status: ending,
        finished_at: new Date().toISOString(),
        ...details,
      });

      return true;
    });
    if (!written) return this;

    // G-360: when the run is over, the plan reads the thread it left: the
    // approvals the run staged, and the ones a new message ended.
    await this.#plans?.afterTurn(this);

    return this;
  }
}
