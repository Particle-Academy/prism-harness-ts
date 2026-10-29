import { ParticipantRef, SessionAddress } from './address.js';
import { indexedStore, storeLockRank, StoreAddressIndex } from './address-index.js';
import type { JsonObject, JsonValue } from './json.js';
import { isJsonObject } from './json.js';
import type { SessionStore } from './stores/session-store.js';
import { StoreTaskSource } from './store-task-source.js';
import type { StoreTaskSourceOptions } from './store-task-source.js';
import { Thread } from './thread.js';

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

export interface RunState extends JsonObject {
  id: string;
  status: 'running' | 'completed' | 'failed';
  mode?: string;
  provider?: string;
  model?: string;
  started_at?: string;
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

  async beginRun(id: string, mode: string, provider: string, model: string): Promise<this> {
    return this.#write('run', {
      id,
      status: 'running',
      mode,
      provider,
      model,
      started_at: new Date().toISOString(),
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
    this.#assertWritable();
    const stored = (await this.#durable.get(this.#durableKey())) ?? {};
    const capabilities = isJsonObject(stored.capabilities) ? stored.capabilities : {};
    await this.#durable.put(this.#durableKey(), { ...stored, capabilities: mutate(capabilities) });

    return this;
  }

  async #finishRun(id: string, status: 'completed' | 'failed', extra: JsonObject): Promise<this> {
    const current = await this.run();

    // A run that is not the one in flight does not overwrite it. A late worker
    // reporting on a superseded run would otherwise mark the live one finished.
    if (current !== null && current.id !== id) {
      return this;
    }

    return this.#write('run', {
      ...(current ?? { id }),
      id,
      status,
      finished_at: new Date().toISOString(),
      ...extra,
    });
  }
}
