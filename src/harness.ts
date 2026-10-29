import type { Participant } from './session.js';
import { Session } from './session.js';
import type { SessionStore } from './stores/session-store.js';
import { MemorySessionStore } from './stores/memory-session-store.js';
import type { StoreManagerOptions } from './stores/store-manager.js';
import { SessionStoreManager } from './stores/store-manager.js';
import { ParticipantAdapters, ParticipantRef, SessionAddress, validateAgent } from './address.js';
import { addressLimit, indexedStore, StoreAddressIndex } from './address-index.js';
import { StoreTaskSource } from './store-task-source.js';
import { Thread } from './thread.js';
import type { AgentRuntime, EffectiveTool } from './runtime.js';

export interface AgentHarnessConfig {
  ttlSeconds?: number | null;
  defaultScope?: string;
  defaultTaskList?: string;
}

export interface HarnessOptions extends Partial<StoreManagerOptions> {
  agents?: Record<string, AgentHarnessConfig>;
  defaultScope?: string;
  defaultTaskList?: string;
  runtime?: AgentRuntime;
  /** How long ephemeral state lives. Null keeps it until forgotten. */
  ttlSeconds?: number | null;
  participantAdapters?: ParticipantAdapters;
  /** Null (the default) leaves named address growth unlimited. */
  maxAddressesPerParticipant?: number | null;
}

/**
 * The entry point: `harness.for(participant).session('support')`.
 *
 * Mirrors the reference's `PrismHarness::for($user)->session('support')`,
 * including the two-step shape — the participant is chosen first and the scope
 * second, because one participant holds several unrelated conversations and the
 * pair is what addresses one of them.
 *
 * ## The default is deliberately NOT usable for durable state
 *
 * With no drivers configured, both slots resolve to an in-memory store, and the
 * manager then REFUSES it for the durable slot — because it reports itself
 * volatile and the durable slot holds approvals a human has not answered yet.
 * Constructing a harness therefore works, and asking it for durable state
 * fails loudly with a message that names the fix.
 *
 * That is not an oversight to be smoothed over. A package that silently
 * accepted an in-memory durable store would pass every test on one process and
 * lose a half-executed action the first time it was deployed on two.
 */
export class PrismHarness {
  readonly #stores: SessionStoreManager;

  readonly #ttlSeconds: number | null;
  readonly #participants: ParticipantAdapters;
  readonly #maxAddresses: number | null;
  readonly #runtime?: AgentRuntime;
  readonly #agents: Record<string, AgentHarnessConfig>;
  readonly #defaultScope: string;
  readonly #defaultTaskList: string;

  constructor(options: HarnessOptions = {}) {
    this.#runtime = options.runtime;
    this.#defaultScope = options.defaultScope ?? 'default';
    this.#defaultTaskList = options.defaultTaskList ?? 'default';
    validateHarnessConfig({ defaultScope: this.#defaultScope, defaultTaskList: this.#defaultTaskList });
    this.#agents = Object.fromEntries(Object.entries(options.agents ?? {}).map(([agent, config]) => {
      validateAgent(agent); validateHarnessConfig(config); return [agent, { ...config }];
    }));
    this.#ttlSeconds = options.ttlSeconds ?? null;
    this.#participants = options.participantAdapters ?? new ParticipantAdapters();
    this.#maxAddresses = addressLimit(options.maxAddressesPerParticipant ?? null);
    this.#stores = new SessionStoreManager({
      stores: options.stores,
      drivers: options.drivers ?? { default: () => new MemorySessionStore() },
    });
  }

  /** Bind to a participant. Returns a builder, because the scope comes next. */
  for(participant: Participant): PendingSession {
    return this.pending(participant, null);
  }

  forAgent(agent: string | null): AgentHarness {
    validateAgent(agent);
    return new AgentHarness(this, agent);
  }

  effectiveTools(agent: string | null, mode: string, participant: Participant, scope?: string): Promise<EffectiveTool[]> {
    if (!this.#runtime) throw new Error('Tool inspection requires a configured Harness runtime.');
    return this.#runtime.effectiveTools(this.pending(participant, agent).session(scope), mode);
  }

  pending(participant: Participant, agent: string | null, actor?: object): PendingSession {
    validateAgent(agent);
    const config = agent === null ? undefined : this.#agents[agent];
    const ttl = config?.ttlSeconds === undefined ? this.#ttlSeconds : config.ttlSeconds;
    return new PendingSession(participant, this.#stores, ttl, agent, actor, this.#participants, this.#maxAddresses, config?.defaultScope ?? this.#defaultScope);
  }

  tasks(list?: string, agent: string | null = null): StoreTaskSource {
    list ??= (agent === null ? undefined : this.#agents[agent]?.defaultTaskList) ?? this.#defaultTaskList;
    const address = new SessionAddress(null, `task-list:${list}`, agent);
    const durable = this.#stores.durable();
    return new StoreTaskSource(agent === null ? durable : indexedStore(durable, new StoreAddressIndex(durable, this.#maxAddresses), address, 'durable', 'tasks'),
      agent === null ? `${list}:tasks` : `agent-list:${address.indexKey()}:tasks`);
  }

  sharedThread(scope: string, agent: string | null = null): Thread {
    const address = new SessionAddress(null, scope, agent), durable = this.#stores.durable();
    return new Thread(agent === null ? durable : indexedStore(durable, new StoreAddressIndex(durable, this.#maxAddresses), address, 'durable', 'thread'), `thread:${address.threadKey()}`);
  }

  async findShared(scope: string, agent: string | null = null): Promise<Thread | null> {
    const thread = this.sharedThread(scope, agent);
    return await thread.exists() ? thread : null;
  }

  ephemeralStore(): SessionStore {
    return this.#stores.ephemeral();
  }

  durableStore(): SessionStore {
    return this.#stores.durable();
  }
}

export class PendingSession {
  constructor(
    private readonly participant: Participant,
    private readonly stores: SessionStoreManager,
    private readonly ttlSeconds: number | null,
    private readonly agent: string | null = null,
    private readonly actor?: object,
    private readonly participants = new ParticipantAdapters(),
    private readonly maxAddresses: number | null = null,
    private readonly defaultScope = 'default',
  ) {}

  asActor(actor: object): PendingSession {
    if (this.participant instanceof ParticipantRef) throw new Error('Typed participants require a registered actor adapter.');
    return new PendingSession(this.participant, this.stores, this.ttlSeconds, this.agent, actor, this.participants, this.maxAddresses, this.defaultScope);
  }

  /**
   * Resolve the session for a scope.
   *
   * The stores are resolved HERE, which is what makes the volatile-durable
   * guard fire when a session is opened rather than at some later moment when
   * an approval needs saving.
   */
  session(scope = this.defaultScope): Session {
    const actor = this.participant instanceof ParticipantRef ? this.participants.actor(this.participant, this.actor) : (this.actor ?? this.participant);
    const durable = this.stores.durable();
    return new Session({
      participant: this.participant,
      scope,
      ephemeral: this.stores.ephemeral(),
      durable,
      ttlSeconds: this.ttlSeconds,
      agent: this.agent,
      actor,
      addressIndex: this.agent !== null || this.participant instanceof ParticipantRef ? new StoreAddressIndex(durable, this.maxAddresses) : undefined,
    });
  }
}

/** A namespace-bound view, sharing stores without a mutable current-agent flag. */
export class AgentHarness {
  constructor(private readonly harness: PrismHarness, readonly agent: string | null) { validateAgent(agent); Object.defineProperty(this, 'agent', { writable: false, configurable: false }); }
  for(participant: Participant): PendingSession { return this.harness.pending(participant, this.agent); }
  effectiveTools(mode: string, participant: Participant, scope?: string): Promise<EffectiveTool[]> { return this.harness.effectiveTools(this.agent, mode, participant, scope); }
  tasks(list?: string): StoreTaskSource { return this.harness.tasks(list, this.agent); }
  sharedThread(scope: string): Thread { return this.harness.sharedThread(scope, this.agent); }
  findShared(scope: string): Promise<Thread | null> { return this.harness.findShared(scope, this.agent); }
}

function validateHarnessConfig(config: AgentHarnessConfig): void {
  if (config.ttlSeconds !== undefined && config.ttlSeconds !== null && (!Number.isSafeInteger(config.ttlSeconds) || config.ttlSeconds < 0)) throw new TypeError('Agent ephemeral TTL must be nonnegative whole seconds or null.');
  for (const key of ['defaultScope', 'defaultTaskList'] as const) {
    const value = config[key];
    if (value !== undefined && (typeof value !== 'string' || Buffer.from(value, 'utf8').toString('utf8') !== value || (key === 'defaultTaskList' && value.length === 0))) throw new TypeError('Agent scope/list defaults must be valid strings.');
  }
}
