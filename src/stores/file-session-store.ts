import { createHash } from 'node:crypto';
import { mkdirSync } from 'node:fs';
import { mkdir, open, readFile, rename, rm, unlink, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import type { JsonObject } from '../json.js';
import { isJsonObject } from '../json.js';
import { HarnessError } from '../errors.js';
import type { Durability, SessionStore } from './session-store.js';

/**
 * State on disk, as one JSON file per key. DURABLE, and it says so.
 *
 * The default durable driver for a port with no database and no dependencies.
 * A real deployment points the durable slot at a database instead; this exists
 * so the package WORKS ON INSTALL rather than requiring infrastructure before
 * the first session can be opened — the same reason the PHP reference defaults
 * both slots to `database`.
 *
 * ## Two properties that matter more than speed
 *
 * **Writes are atomic.** A payload is written to a temporary file and renamed
 * over the target, because `rename` is atomic on both POSIX and Windows. A
 * partial write here is a corrupted thread, and a process killed mid-write is
 * ordinary rather than exotic.
 *
 * **The lock is cross-process.** It is an exclusive-create lockfile (`wx`),
 * which is the one primitive that is atomic on every filesystem worth
 * supporting — unlike an "is it there?" check followed by a create, which has a
 * window between the two. Two workers on one machine genuinely exclude each
 * other. Two workers on different machines sharing a network filesystem do NOT
 * reliably, and no file lock can promise that; use a database or Redis there.
 */
export class FileSessionStore implements SessionStore {
  readonly #directory: string;

  constructor(directory: string) {
    this.#directory = directory;
    mkdirSync(directory, { recursive: true });
  }

  durability(): Durability {
    return 'durable';
  }

  async get(key: string): Promise<JsonObject | null> {
    let raw: string;

    try {
      raw = await readFile(this.#pathFor(key), 'utf8');
    } catch (error) {
      if (isMissing(error)) return null;
      throw error;
    }

    let document: unknown;

    try {
      document = JSON.parse(raw);
    } catch (error) {
      throw HarnessError.unmappableContent(`the stored payload for [${key}] is not valid JSON`, {
        cause: error,
      });
    }

    if (!isJsonObject(document)) {
      throw HarnessError.unmappableContent(`the stored payload for [${key}] is not an object`);
    }

    const expiresAt = document.expires_at;

    if (typeof expiresAt === 'number' && expiresAt <= Date.now()) {
      await this.forget(key);

      return null;
    }

    return isJsonObject(document.payload) ? document.payload : null;
  }

  async put(key: string, payload: JsonObject, ttlSeconds: number | null = null): Promise<void> {
    const target = this.#pathFor(key);
    const temporary = `${target}.${process.pid}.${Date.now()}.tmp`;
    const document = JSON.stringify({
      key,
      payload,
      expires_at: ttlSeconds === null ? null : Date.now() + ttlSeconds * 1000,
    });

    await mkdir(this.#directory, { recursive: true });
    await writeFile(temporary, document, 'utf8');

    try {
      // Atomic on both POSIX and Windows. A reader either sees the whole old
      // payload or the whole new one, never half of either.
      await rename(temporary, target);
    } catch (error) {
      await rm(temporary, { force: true });
      throw error;
    }
  }

  async forget(key: string): Promise<void> {
    await rm(this.#pathFor(key), { force: true });
  }

  async withLock<T>(
    key: string,
    callback: () => T | Promise<T>,
    ttlSeconds = 10,
    waitSeconds = 5,
  ): Promise<T> {
    const lockPath = `${this.#pathFor(key)}.lock`;
    const deadline = Date.now() + waitSeconds * 1000;

    for (;;) {
      try {
        // `wx` fails if the file exists. Exclusive CREATION is atomic; a
        // check-then-create is not, and the window between the two is exactly
        // where two workers both decide they hold the lock.
        const handle = await open(lockPath, 'wx');
        await handle.writeFile(String(Date.now() + ttlSeconds * 1000), 'utf8');
        await handle.close();

        try {
          return await callback();
        } finally {
          await this.#release(lockPath);
        }
      } catch (error) {
        if (!isHeld(error)) throw error;

        // A lock whose TTL has passed belonged to a process that died holding
        // it. Left alone it would wedge the key forever, which is worse than
        // the small race in reclaiming it — and the reclaim is itself a
        // create-exclusive, so only one waiter can win.
        if (await this.#expired(lockPath)) {
          await unlink(lockPath).catch(() => undefined);
          continue;
        }

        if (Date.now() >= deadline) {
          throw HarnessError.sessionLocked(key, waitSeconds);
        }

        await sleep(25);
      }
    }
  }

  /**
   * Give the lock up, and NEVER leave it held on the way out.
   *
   * Deleting is the normal path, but on Windows it can fail: another waiter
   * attempting to create the same path holds a transient handle, and the unlink
   * then raises EPERM. Swallowing that — which is what this did first — LEAKS
   * THE LOCK. The waiter then sees a lockfile whose TTL is still in the future,
   * has no way to know its holder is gone, and blocks for the whole wait before
   * failing. Not theoretical: `prism-harness-py` hit exactly this the first
   * time two threads recorded a message concurrently, and this port had the
   * same defect with no test that happened to reach it.
   *
   * So if the file cannot be removed, it is rewritten with an ALREADY-PAST
   * expiry. Any waiter reclaims it on its next attempt, which is the same path
   * a genuinely dead holder takes.
   */
  async #release(lockPath: string): Promise<void> {
    for (let attempt = 0; attempt < 5; attempt += 1) {
      try {
        await unlink(lockPath);

        return;
      } catch (error) {
        if (isMissing(error)) return;
        await sleep(5);
      }
    }

    // Nothing left to try but mark it dead. The TTL is the backstop, and it is
    // why the lockfile carries one at all.
    await writeFile(lockPath, '0', 'utf8').catch(() => undefined);
  }

  async #expired(lockPath: string): Promise<boolean> {
    try {
      const expiresAt = Number(await readFile(lockPath, 'utf8'));

      return Number.isFinite(expiresAt) && expiresAt <= Date.now();
    } catch {
      // Gone between the failed create and this read: treat it as not expired
      // and let the next attempt take it cleanly.
      return false;
    }
  }

  /**
   * One file per key, named by a digest.
   *
   * A session key contains colons, which are legal on POSIX and not on Windows,
   * and it is long enough to run into path ceilings once a scope is appended.
   * The digest sidesteps both, and the key itself is stored INSIDE the file so
   * the mapping stays inspectable.
   */
  #pathFor(key: string): string {
    return join(this.#directory, `${createHash('sha256').update(key).digest('hex').slice(0, 32)}.json`);
  }
}

function isMissing(error: unknown): boolean {
  return typeof error === 'object' && error !== null && (error as { code?: string }).code === 'ENOENT';
}

/**
 * "The lock is already held" — under four different names.
 *
 * POSIX says EEXIST. WINDOWS DOES NOT, reliably: deleting a file that another
 * handle still has open leaves it in a pending-delete state, and a create
 * attempt against that returns EPERM, EACCES or EBUSY instead. That window is
 * not rare — it is exactly the moment one caller releases the lock while the
 * next is trying to take it, which is the common case under contention rather
 * than an edge.
 *
 * Treating those as "held, retry" is also the safe direction to be wrong in: a
 * genuine permission problem retries until the wait expires and then reports
 * `session_locked`, which is a survivable failure. Treating them as "not held"
 * would hand the same lock to two callers.
 */
function isHeld(error: unknown): boolean {
  const code = typeof error === 'object' && error !== null ? (error as { code?: string }).code : undefined;

  return code === 'EEXIST' || code === 'EPERM' || code === 'EACCES' || code === 'EBUSY';
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, ms);
    timer.unref?.();
  });
}
