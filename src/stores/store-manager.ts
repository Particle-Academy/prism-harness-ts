import { HarnessError } from '../errors.js';
import type { SessionStore } from './session-store.js';
import { isDurable } from './session-store.js';

export const SLOT_EPHEMERAL = 'ephemeral';
export const SLOT_DURABLE = 'durable';

export type StoreFactory = () => SessionStore;

export interface StoreManagerOptions {
  /** Driver name per slot. Both default to whatever is registered as `default`. */
  stores?: Partial<Record<'ephemeral' | 'durable', string>>;
  /** Driver name → factory. Resolved lazily, and each driver is built at most once. */
  drivers: Record<string, StoreFactory>;
}

/**
 * Resolves the two state slots, and REFUSES a configuration that would lose work.
 *
 * State is split into two named slots rather than one store, because the halves
 * have genuinely different requirements:
 *
 *  - `ephemeral` — active mode, selected model, run bookkeeping. Losing it
 *    degrades to a default.
 *  - `durable` — threads and pending tool approvals. Losing it is a correctness
 *    failure, not a cache miss.
 *
 * The guard is the point of this class: a driver that reports itself volatile
 * is refused for the durable slot AT RESOLVE TIME. Accepting it and finding out
 * later is exactly the failure the package was written to avoid.
 */
export class SessionStoreManager {
  readonly #drivers: Record<string, StoreFactory>;

  readonly #slots: Record<string, string>;

  readonly #resolved = new Map<string, SessionStore>();

  constructor(options: StoreManagerOptions) {
    this.#drivers = options.drivers;
    this.#slots = {
      [SLOT_EPHEMERAL]: options.stores?.ephemeral ?? 'default',
      [SLOT_DURABLE]: options.stores?.durable ?? 'default',
    };
  }

  ephemeral(): SessionStore {
    return this.slot(SLOT_EPHEMERAL);
  }

  durable(): SessionStore {
    return this.slot(SLOT_DURABLE);
  }

  slot(slot: string): SessionStore {
    const cached = this.#resolved.get(slot);
    if (cached !== undefined) return cached;

    const name = this.#slots[slot] ?? 'default';
    const factory = this.#drivers[name];

    if (factory === undefined) {
      throw HarnessError.unknownStoreDriver(slot, name);
    }

    const store = factory();

    // Checked HERE rather than at construction, so it fires in the same place
    // whether the store is configured up front, swapped in a test, or changed
    // at runtime — and so a misconfiguration cannot lie dormant until the first
    // approval needs saving.
    if (slot === SLOT_DURABLE && !isDurable(store.durability())) {
      throw HarnessError.volatileDurableStore(slot, name);
    }

    this.#resolved.set(slot, store);

    return store;
  }
}
