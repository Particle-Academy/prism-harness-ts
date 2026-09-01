import type { Participant } from './session.js';
import { Session } from './session.js';
import type { SessionStore } from './stores/session-store.js';
import { MemorySessionStore } from './stores/memory-session-store.js';
import type { StoreManagerOptions } from './stores/store-manager.js';
import { SessionStoreManager } from './stores/store-manager.js';

export interface HarnessOptions extends Partial<StoreManagerOptions> {
  /** How long ephemeral state lives. Null keeps it until forgotten. */
  ttlSeconds?: number | null;
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

  constructor(options: HarnessOptions = {}) {
    this.#ttlSeconds = options.ttlSeconds ?? null;
    this.#stores = new SessionStoreManager({
      stores: options.stores,
      drivers: options.drivers ?? { default: () => new MemorySessionStore() },
    });
  }

  /** Bind to a participant. Returns a builder, because the scope comes next. */
  for(participant: Participant): PendingSession {
    return new PendingSession(participant, this.#stores, this.#ttlSeconds);
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
  ) {}

  /**
   * Resolve the session for a scope.
   *
   * The stores are resolved HERE, which is what makes the volatile-durable
   * guard fire when a session is opened rather than at some later moment when
   * an approval needs saving.
   */
  session(scope: string): Session {
    return new Session({
      participant: this.participant,
      scope,
      ephemeral: this.stores.ephemeral(),
      durable: this.stores.durable(),
      ttlSeconds: this.ttlSeconds,
    });
  }
}
