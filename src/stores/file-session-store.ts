import { createHash } from 'node:crypto';
import { mkdirSync } from 'node:fs';
import { mkdir, open, readFile, rename, rm, stat, truncate, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import type { JsonObject } from '../json.js';
import { isJsonObject } from '../json.js';
import { HarnessError } from '../errors.js';
import { withFileOwnerLock, pendingFileOwnerReleases } from './file-owner-lock.js';
import { fileLockIdentityStatus } from './file-lock-identity.js';
import type { FileLockIdentityStatus } from './file-lock-identity.js';
import type { PendingFileLockRelease } from './file-owner-lock.js';
import type { AppendLogStore, Durability, SessionStore } from './session-store.js';
import { matchesLog, readLogIndex, saveLogIndex, scanLog } from './file-session-log.js';
import type { LogIndex } from './file-session-log.js';

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
 * **The lock is cross-process.** Decision 0022 uses an atomically published
 * owner directory. TTL never proves death. Local filesystems on one host only;
 * SMB/NFS are unsupported. Stop all old workers before upgrading and inspect
 * legacy timestamp lockfiles explicitly. Use a database or Redis across hosts.
 */
export class FileSessionStore implements SessionStore, AppendLogStore {
  readonly #directory: string;

  /**
   * Disposable counts bound to the actual file's size, times and identity.
   * A new instance can reuse the small sidecar; a mismatch streams a recount.
   */
  readonly #logs = new Map<string, LogIndex>();

  constructor(directory: string) {
    this.#directory = directory;
    mkdirSync(directory, { recursive: true });
  }

  durability(): Durability {
    return 'durable';
  }

  /** Local filesystem on one host; SMB/NFS sharing is unsupported. */
  sharingScope(): 'host' { return 'host'; }
  lockScope(): 'host' | 'process' | 'distributed' | 'unsupported' { return 'host'; }

  /** Process discovery snapshot; this observer never starts metadata discovery. */
  identityStatus(): FileLockIdentityStatus { return fileLockIdentityStatus(); }

  async get(key: string): Promise<JsonObject | null> {
    let raw: string;

    try {
      // Node's whole-file helper closes its own handle on both outcomes before
      // resolving; JSON decoding and expiry housekeeping happen after it closes.
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
      await replaceStateFile(temporary, target);
    } catch (error) {
      await rm(temporary, { force: true });
      throw error;
    }
  }

  async forget(key: string): Promise<void> {
    await rm(this.#pathFor(key), { force: true });
    await rm(this.#logPathFor(key), { force: true });
    await rm(this.#logPathFor(key) + '.index.json', { force: true }).catch(() => undefined);
    this.#logs.delete(key);
  }

  /**
   * One JSON document per line, appended through a single write and then
   * synced, so a message that `record()` returned is on disk. Before writing,
   * a TORN tail (a previous write cut off by a crash) is truncated back to the
   * last complete line; appending after it would glue the new line onto the
   * fragment and corrupt both.
   */
  async appendLog(key: string, entries: readonly JsonObject[]): Promise<void> {
    const path = this.#logPathFor(key);
    await mkdir(this.#directory, { recursive: true });
    const before = await this.#repairTail(path);
    const beforeStat = await stat(path).catch((error: unknown) => {
      if (isMissing(error)) return null;
      throw error;
    });
    const data = entries.map((entry) => `${JSON.stringify(entry)}\n`).join('');
    const handle = await open(path, 'a');

    try {
      if (data !== '') await handle.write(data);
      await handle.sync();
    } finally {
      await handle.close();
    }

    const known = this.#logs.get(key);
    const previousCount = before === 0 ? 0
      : known !== undefined && beforeStat !== null && matchesLog(known, beforeStat) ? known.count : null;
    if (previousCount !== null) {
      const after = await stat(path);
      if (after.size === before + Buffer.byteLength(data)) {
        await this.#rememberLog(key, { size: after.size, mtimeMs: after.mtimeMs, ctimeMs: after.ctimeMs,
          ino: after.ino, dev: after.dev, count: previousCount + entries.length });
      } else this.#logs.delete(key);
    } else {
      this.#logs.delete(key);
    }
  }

  async readLog(key: string): Promise<JsonObject[] | null> {
    const entries: JsonObject[] = [];
    let index: LogIndex;
    try {
      index = await scanLog(this.#logPathFor(key), (line, number) => {
        let entry: unknown;
        try {
          entry = JSON.parse(line);
        } catch (error) {
          throw HarnessError.unmappableContent(`line ${number} of the thread log for [${key}] is not valid JSON`, { cause: error });
        }
        if (!isJsonObject(entry)) {
          throw HarnessError.unmappableContent(`line ${number} of the thread log for [${key}] is not an object`);
        }
        entries.push(entry);
      });
    } catch (error) {
      if (isMissing(error)) return null;
      throw error;
    }

    await this.#rememberLog(key, index);
    return entries;
  }

  async logLength(key: string): Promise<number | null> {
    let current;

    try {
      current = await stat(this.#logPathFor(key));
    } catch (error) {
      if (isMissing(error)) return null;
      throw error;
    }

    const known = this.#logs.get(key);
    if (known !== undefined && matchesLog(known, current)) return known.count;
    const disk = await readLogIndex(this.#logPathFor(key), current);
    if (disk !== null) {
      this.#logs.set(key, disk);
      return disk.count;
    }
    try {
      const index = await scanLog(this.#logPathFor(key));
      await this.#rememberLog(key, index);
      return index.count;
    } catch (error) {
      if (isMissing(error)) return null;
      throw error;
    }
  }

  async clearLog(key: string): Promise<void> {
    await mkdir(this.#directory, { recursive: true });
    await writeFile(this.#logPathFor(key), '', 'utf8');
    const current = await stat(this.#logPathFor(key));
    await this.#rememberLog(key, { size: current.size, mtimeMs: current.mtimeMs,
      ctimeMs: current.ctimeMs, ino: current.ino, dev: current.dev, count: 0 });
  }

  async #rememberLog(key: string, index: LogIndex): Promise<void> {
    if (index.size < 0) { this.#logs.delete(key); return; }
    this.#logs.set(key, index);
    await saveLogIndex(this.#logPathFor(key), index);
  }

  /** Cut a torn tail back to the last newline. Returns the file's size afterwards. */
  async #repairTail(path: string): Promise<number> {
    let size: number;

    try {
      size = (await stat(path)).size;
    } catch (error) {
      if (isMissing(error)) return 0;
      throw error;
    }

    if (size === 0) return 0;

    const handle = await open(path, 'r');

    try {
      const last = Buffer.alloc(1);
      await handle.read(last, 0, 1, size - 1);
      if (last[0] === 0x0a) return size;

      // Walk back a chunk at a time: one message can be larger than any chunk.
      const chunk = Buffer.alloc(64 * 1024);
      let end = size;

      while (end > 0) {
        const start = Math.max(0, end - chunk.length);
        const { bytesRead } = await handle.read(chunk, 0, end - start, start);
        const newline = chunk.subarray(0, bytesRead).lastIndexOf(0x0a);

        if (newline !== -1) {
          await handle.close();
          await truncate(path, start + newline + 1);

          return start + newline + 1;
        }

        end = start;
      }

      await handle.close();
      await truncate(path, 0);

      return 0;
    } finally {
      await handle.close().catch(() => undefined);
    }
  }

  async withLock<T>(
    key: string,
    callback: () => T | Promise<T>,
    ttlSeconds = 10,
    waitSeconds = 5,
  ): Promise<T> {
    return withFileOwnerLock(this.#pathFor(key) + '.lock', key, callback, ttlSeconds, waitSeconds);
  }

  /** Pending release diagnostics for this directory; length is the pending count. */
  pendingReleases(): readonly PendingFileLockRelease[] {
    return pendingFileOwnerReleases(this.#directory);
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
    return join(this.#directory, `${digestOf(key)}.json`);
  }

  /** The key's append-only log, beside its payload: same digest, `.jsonl`. */
  #logPathFor(key: string): string {
    return join(this.#directory, `${digestOf(key)}.jsonl`);
  }
}

/** A key's file name. Colons are illegal on Windows and keys run long, so it is digested. */
function digestOf(key: string): string {
  return createHash('sha256').update(key).digest('hex').slice(0, 32);
}

function isMissing(error: unknown): boolean {
  return typeof error === 'object' && error !== null && (error as { code?: string }).code === 'ENOENT';
}

/** An open Windows target can block replacement even with delete sharing. */
async function replaceStateFile(temporary: string, target: string): Promise<void> {
  try {
    await rename(temporary, target);
    return;
  } catch (original) {
    if (process.platform !== 'win32' || !isReplacePermissionError(original)) throw original;
    const deadline = performance.now() + 750;
    while (true) {
      const remaining = deadline - performance.now();
      if (remaining <= 0) throw original;
      // This timer stays referenced: an awaited write must keep the process alive.
      await delay(Math.min(10, remaining));
      if (performance.now() >= deadline) throw original;
      try {
        await rename(temporary, target);
        return;
      } catch (error) {
        if (!isReplacePermissionError(error)) throw error;
      }
    }
  }
}

function isReplacePermissionError(error: unknown): boolean {
  const code = typeof error === 'object' && error !== null ? (error as { code?: string }).code : undefined;
  return code === 'EPERM' || code === 'EACCES';
}
