import { createHash, randomUUID } from 'node:crypto';
import type { Stats } from 'node:fs';
import { open, rename, rm, stat, writeFile } from 'node:fs/promises';

/** Advisory metadata only. A log remains authoritative if this file is lost. */
export interface LogIndex {
  size: number;
  mtimeMs: number;
  ctimeMs: number;
  ino: number;
  dev: number;
  count: number;
}

export function matchesLog(index: LogIndex, current: Stats): boolean {
  return index.size === current.size && index.mtimeMs === current.mtimeMs
    && index.ctimeMs === current.ctimeMs && index.ino === current.ino && index.dev === current.dev;
}

function indexFor(snapshot: Stats, count: number): LogIndex {
  return { size: snapshot.size, mtimeMs: snapshot.mtimeMs, ctimeMs: snapshot.ctimeMs,
    ino: snapshot.ino, dev: snapshot.dev, count };
}

/**
 * Read a fixed file snapshot in bounded byte chunks. Only newline-terminated
 * records exist; the unfinished tail is deliberately ignored. Counting never
 * decodes log JSON. Decoding retains at most one serialized record at a time.
 */
export async function scanLog(path: string, onLine?: (line: string, number: number) => void): Promise<LogIndex> {
  const handle = await open(path, 'r');
  try {
    const snapshot = await handle.stat();
    const chunk = Buffer.alloc(64 * 1024);
    let offset = 0;
    let count = 0;
    let pieces: Buffer[] = [];
    while (offset < snapshot.size) {
      const { bytesRead } = await handle.read(chunk, 0, Math.min(chunk.length, snapshot.size - offset), offset);
      if (bytesRead === 0) break;
      offset += bytesRead;
      let start = 0;
      let newline = chunk.indexOf(0x0a, start);
      while (newline !== -1 && newline < bytesRead) {
        count++;
        if (onLine !== undefined) {
          const last = chunk.subarray(start, newline);
          const line = pieces.length === 0 ? last : Buffer.concat([...pieces, last]);
          onLine(line.toString('utf8'), count);
          pieces = [];
        }
        start = newline + 1;
        newline = chunk.indexOf(0x0a, start);
      }
      if (onLine !== undefined && start < bytesRead) pieces.push(Buffer.from(chunk.subarray(start, bytesRead)));
    }
    const index = indexFor(snapshot, count);
    // A concurrent truncate/append invalidates caching, even if a reader can
    // still return the complete lines it observed. A negative size never hits.
    if (offset !== snapshot.size || !matchesLog(index, await handle.stat())) index.size = -1;
    return index;
  } finally {
    await handle.close();
  }
}

function checksum(index: LogIndex): string {
  return createHash('sha256').update(JSON.stringify(index)).digest('hex');
}

export async function readLogIndex(path: string, current: Stats): Promise<LogIndex | null> {
  try {
    const handle = await open(path + '.index.json', 'r');
    try {
      const { size } = await handle.stat();
      if (size > 1024 || size === 0) return null;
      const buffer = Buffer.alloc(1024);
      const { bytesRead } = await handle.read(buffer, 0, buffer.length, 0);
      const document: unknown = JSON.parse(buffer.subarray(0, bytesRead).toString('utf8'));
      if (typeof document !== 'object' || document === null) return null;
      const value = document as Record<string, unknown>;
      if (value.version !== 1 || typeof value.checksum !== 'string') return null;
      for (const field of ['size', 'mtimeMs', 'ctimeMs', 'ino', 'dev', 'count']) {
        if (typeof value[field] !== 'number' || !Number.isFinite(value[field])) return null;
      }
      const index: LogIndex = { size: value.size as number, mtimeMs: value.mtimeMs as number,
        ctimeMs: value.ctimeMs as number, ino: value.ino as number, dev: value.dev as number, count: value.count as number };
      if (!Number.isSafeInteger(index.count) || index.count < 0 || index.count > index.size
        || !matchesLog(index, current) || value.checksum !== checksum(index)) return null;
      return index;
    } finally {
      await handle.close();
    }
  } catch {
    // Missing, torn, inaccessible and malformed indexes are all cache misses.
    return null;
  }
}

export async function saveLogIndex(path: string, index: LogIndex): Promise<void> {
  const temporary = path + '.' + randomUUID() + '.index.tmp';
  try {
    if (!matchesLog(index, await stat(path))) return;
    // Fixed small write cost regardless of the thread's length. No fsync policy
    // change: this cache is disposable; appendLog still syncs the actual log.
    const body = JSON.stringify({ version: 1, ...index, checksum: checksum(index) });
    await writeFile(temporary, body.padEnd(384, ' '), { encoding: 'utf8', flag: 'wx', mode: 0o600 });
    await rename(temporary, path + '.index.json');
  } catch {
    // A successful durable append must not become a failure due to its cache.
  } finally {
    await rm(temporary, { force: true }).catch(() => undefined);
  }
}
