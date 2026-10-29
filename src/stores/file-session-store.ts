import { createHash } from 'node:crypto';
import { mkdirSync } from 'node:fs';
import { mkdir, open, readFile, rename, rm, stat, truncate, unlink, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import type { JsonObject } from '../json.js';
import { isJsonObject } from '../json.js';
import { HarnessError } from '../errors.js';
import { withFileOwnerLock, pendingFileOwnerReleases } from './file-owner-lock.js';
import type { PendingFileLockRelease } from './file-owner-lock.js';
import type { AppendLogStore, Durability, SessionStore } from './session-store.js';

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
   * What this instance last saw of each log: its size on disk and how many
   * complete entries that size holds. Lets `logLength()` answer without reading
   * the log, and stays honest across processes because it is only trusted while
   * the file is still exactly that size. Another writer changes the size, and
   * the next call recounts.
   */
  readonly #logs = new Map<string, { size: number; count: number }>();

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
    await rm(this.#logPathFor(key), { force: true });
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
    const data = entries.map((entry) => `${JSON.stringify(entry)}\n`).join('');
    const handle = await open(path, 'a');

    try {
      if (data !== '') await handle.write(data);
      await handle.sync();
    } finally {
      await handle.close();
    }

    const known = this.#logs.get(key);
    if (known !== undefined && known.size === before) {
      this.#logs.set(key, { size: before + Buffer.byteLength(data), count: known.count + entries.length });
    } else {
      this.#logs.delete(key);
    }
  }

  async readLog(key: string): Promise<JsonObject[] | null> {
    let raw: string;

    try {
      raw = await readFile(this.#logPathFor(key), 'utf8');
    } catch (error) {
      if (isMissing(error)) return null;
      throw error;
    }

    const lines = raw.split('\n');
    // Everything after the last newline is a line that never finished. Empty
    // when the file ends cleanly, a fragment when a crash cut a write short.
    lines.pop();
    const entries = lines.map((line, index) => {
      let entry: unknown;

      try {
        entry = JSON.parse(line);
      } catch (error) {
        throw HarnessError.unmappableContent(`line ${index + 1} of the thread log for [${key}] is not valid JSON`, {
          cause: error,
        });
      }

      if (!isJsonObject(entry)) {
        throw HarnessError.unmappableContent(`line ${index + 1} of the thread log for [${key}] is not an object`);
      }

      return entry;
    });

    this.#logs.set(key, { size: Buffer.byteLength(raw), count: entries.length });

    return entries;
  }

  async logLength(key: string): Promise<number | null> {
    let size: number;

    try {
      size = (await stat(this.#logPathFor(key))).size;
    } catch (error) {
      if (isMissing(error)) return null;
      throw error;
    }

    const known = this.#logs.get(key);
    if (known !== undefined && known.size === size) return known.count;

    return (await this.readLog(key))?.length ?? null;
  }

  async clearLog(key: string): Promise<void> {
    await mkdir(this.#directory, { recursive: true });
    await writeFile(this.#logPathFor(key), '', 'utf8');
    this.#logs.set(key, { size: 0, count: 0 });
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
