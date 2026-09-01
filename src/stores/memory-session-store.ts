import type { JsonObject } from '../json.js';
import { clone } from '../json.js';
import { HarnessError } from '../errors.js';
import type { Durability, SessionStore } from './session-store.js';

interface Entry {
  payload: JsonObject;
  expiresAt: number | null;
}

/**
 * State in this process's memory. VOLATILE, and it says so.
 *
 * The right home for the ephemeral slot in a test or a single-process tool, and
 * REFUSED for the durable slot by the store manager — not as a technicality:
 * the contents do not outlive the process, and the durable slot holds approvals
 * a human has not answered yet.
 *
 * The lock is real but PROCESS-LOCAL. Two workers cannot see each other's
 * locks, which is the whole reason a deployment uses something else; this
 * serialises callers within one process and claims nothing beyond that.
 */
export class MemorySessionStore implements SessionStore {
  readonly #entries = new Map<string, Entry>();

  /** Key → the promise chain currently serialising work on it. */
  readonly #locks = new Map<string, Promise<unknown>>();

  durability(): Durability {
    return 'volatile';
  }

  async get(key: string): Promise<JsonObject | null> {
    const entry = this.#entries.get(key);

    if (entry === undefined) {
      return null;
    }

    if (entry.expiresAt !== null && entry.expiresAt <= Date.now()) {
      this.#entries.delete(key);

      return null;
    }

    // A copy, so a caller mutating what it read cannot reach into the store.
    return clone(entry.payload);
  }

  async put(key: string, payload: JsonObject, ttlSeconds: number | null = null): Promise<void> {
    this.#entries.set(key, {
      payload: clone(payload),
      expiresAt: ttlSeconds === null ? null : Date.now() + ttlSeconds * 1000,
    });
  }

  async forget(key: string): Promise<void> {
    this.#entries.delete(key);
  }

  async withLock<T>(
    key: string,
    callback: () => T | Promise<T>,
    _ttlSeconds = 10,
    waitSeconds = 5,
  ): Promise<T> {
    const previous = this.#locks.get(key) ?? Promise.resolve();

    let release!: () => void;
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });

    // CLAIMED SYNCHRONOUSLY, before the first `await`.
    //
    // Registering after awaiting is a check-then-act race: two callers arriving
    // in the same tick both read an empty slot, both decide they hold the lock,
    // and both run. That is the same shape as an `exists?`-then-`create` on the
    // file driver, which is why that one uses exclusive creation instead — and
    // it is not theoretical, it is what the serialisation test caught here.
    //
    // The chain resolves only once the previous holder has finished AND this
    // one has released, so waiters queue in arrival order.
    const chained = previous.then(
      () => held,
      () => held,
    );
    this.#locks.set(key, chained);

    // The wait is bounded. A callback that never returns would otherwise hold
    // every later caller forever, and "the lock timed out" is a far better
    // failure than a request that hangs until something upstream gives up.
    const acquired = await Promise.race([
      previous.then(
        () => true,
        () => true,
      ),
      sleep(waitSeconds * 1000).then(() => false),
    ]);

    if (!acquired) {
      // Release before throwing, or everyone queued behind this caller waits on
      // a lock that was never actually taken.
      this.#releaseSlot(key, chained, release);

      throw HarnessError.sessionLocked(key, waitSeconds);
    }

    try {
      return await callback();
    } finally {
      this.#releaseSlot(key, chained, release);
    }
  }

  #releaseSlot(key: string, chained: Promise<unknown>, release: () => void): void {
    release();

    // Only clear the slot if nobody queued behind us. Deleting a slot another
    // caller has already chained onto would let a third caller start while the
    // second is still running.
    if (this.#locks.get(key) === chained) {
      this.#locks.delete(key);
    }
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, ms);
    // Do not hold the event loop open just to fail a lock later.
    timer.unref?.();
  });
}
