import type { JsonObject } from '../json.js';

/**
 * Whether a store's contents survive a deploy.
 *
 * This is the distinction the whole state layer turns on. An in-memory map or a
 * Redis used as a cache is the natural home for live session state — and a
 * cache is disposable by definition. Something has to say which of the two a
 * configured store actually is, because the package cannot detect it and
 * guessing wrong loses a half-executed agent action rather than a cheap value.
 */
export type Durability =
  /**
   * Contents may vanish at any time — a flush, an eviction, a deploy.
   *
   * Only safe for state whose loss degrades to a default: the active mode, the
   * selected model, run bookkeeping. Ask again and you get a sensible answer.
   */
  | 'volatile'
  /**
   * Contents survive until deliberately removed.
   *
   * Required for anything whose loss is a correctness failure rather than an
   * inconvenience — a pending tool approval is a half-executed action waiting
   * on a human, and it has to outlive the request, the worker, and a deploy.
   */
  | 'durable';

export function isDurable(durability: Durability): boolean {
  return durability === 'durable';
}

/**
 * Where session state lives between turns.
 *
 * A server handles a request and moves on, so a session cannot be an object
 * held in memory the way a single-process agent's is — it has to be
 * reconstructed from a store every time. This is that store.
 *
 * Implementations declare their OWN durability rather than having it inferred.
 * Only the application knows whether its Redis is persistent or a disposable
 * cache, and the difference decides whether losing the contents is a shrug or a
 * lost agent action.
 */
export interface SessionStore {
  /** Null when nothing is stored. */
  get(key: string): Promise<JsonObject | null>;

  /** `ttlSeconds` of null keeps the payload until it is removed. */
  put(key: string, payload: JsonObject, ttlSeconds?: number | null): Promise<void>;

  forget(key: string): Promise<void>;

  /**
   * Run the callback while holding an EXCLUSIVE lock on the key.
   *
   * Two workers can resolve the same session at the same moment — a queued job
   * finishing a run while the user sends another message is ordinary, not
   * exotic. Whatever must not happen twice goes in here.
   *
   * Returns the callback's value. Throws `HarnessError` code `session_locked`
   * if the lock cannot be acquired within `waitSeconds`, rather than running
   * the callback anyway.
   */
  withLock<T>(
    key: string,
    callback: () => T | Promise<T>,
    ttlSeconds?: number,
    waitSeconds?: number,
  ): Promise<T>;

  /**
   * Whether this store's contents survive a deploy.
   *
   * Read by the manager when a slot is resolved: a store that reports itself
   * volatile is refused for durable state instead of silently accepting it.
   */
  durability(): Durability;
}
