import { createHash } from 'node:crypto';
import { mkdtemp, open, readFile, readdir, rm, stat, utimes, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';

const io = vi.hoisted(() => ({ wholeLogReads: 0, logOpens: 0, afterLogRead: null as (() => Promise<void>) | null }));
vi.mock('node:fs/promises', async (original) => {
  const fs = await original<typeof import('node:fs/promises')>();
  return { ...fs,
    readFile: (...args: Parameters<typeof fs.readFile>) => {
      if (String(args[0]).endsWith('.jsonl')) io.wholeLogReads++;
      return fs.readFile(...args);
    },
    open: async (...args: Parameters<typeof fs.open>) => {
      const handle = await fs.open(...args);
      if (String(args[0]).endsWith('.jsonl') && args[1] === 'r') {
        io.logOpens++;
        const read = handle.read.bind(handle);
        return Object.assign(handle, { read: async (...a: Parameters<typeof read>) => {
          const result = await read(...a);
          const callback = io.afterLogRead;
          io.afterLogRead = null;
          if (callback !== null) await callback();
          return result;
        } });
      }
      return handle;
    },
  };
});
const { FileSessionStore } = await import('../src/index.js');
const owned: string[] = [];
afterEach(async () => {
  for (const directory of owned.splice(0)) await rm(directory, { recursive: true, force: true });
  vi.restoreAllMocks();
  io.afterLogRead = null;
});
async function fixture(): Promise<{ directory: string; file: string; index: string; store: InstanceType<typeof FileSessionStore> }> {
  const directory = await mkdtemp(join(tmpdir(), 'prism-large-log-'));
  owned.push(directory);
  const file = join(directory, createHash('sha256').update('large').digest('hex').slice(0, 32) + '.jsonl');
  return { directory, file, index: file + '.index.json', store: new FileSessionStore(directory) };
}

describe('streamed file session logs', () => {
  it('counts newline boundaries on a cache miss without readLog, readFile or decoding log JSON', async () => {
    const { file, store } = await fixture();
    await writeFile(file, 'not JSON\n{"torn":');
    const read = vi.spyOn(store, 'readLog');
    io.wholeLogReads = 0;
    expect(await store.logLength('large')).toBe(1);
    expect(read).not.toHaveBeenCalled();
    expect(io.wholeLogReads).toBe(0);
    await expect(store.readLog('large')).rejects.toMatchObject({ code: 'unmappable_content' });
  });

  it('reuses a validated small sidecar after a new instance without reopening the log', async () => {
    const { directory, file, index, store } = await fixture();
    await writeFile(file, '{"n":1}\n{"n":2}\n');
    expect(await store.logLength('large')).toBe(2);
    expect((await stat(index)).size).toBeLessThan(1024);
    io.logOpens = 0;
    expect(await new FileSessionStore(directory).logLength('large')).toBe(2);
    expect(io.logOpens).toBe(0);
  });

  it('rebuilds tampered, malformed and stale sidecars instead of trusting their count', async () => {
    const { directory, file, index, store } = await fixture();
    await writeFile(file, '{"n":1}\n');
    expect(await store.logLength('large')).toBe(1);
    const original = await readFile(index, 'utf8');
    const tampered: Record<string, unknown> = JSON.parse(original);
    tampered.count = 0;
    for (const raw of [JSON.stringify(tampered), '{', original.replace('"version":1', '"version":99')]) {
      await writeFile(index, raw);
      io.logOpens = 0;
      expect(await new FileSessionStore(directory).logLength('large')).toBe(1);
      expect(io.logOpens).toBeGreaterThan(0);
    }
    await writeFile(file, '{"n":1}\n{"n":2}\n');
    expect(await new FileSessionStore(directory).logLength('large')).toBe(2);
  });

  it('invalidates both memory and disk caches when a same-size replacement changes mtime', async () => {
    const { directory, file, store } = await fixture();
    const first = '{"n":1}        \n';
    const second = '{"n":1}\n{"n":2}\n';
    expect(Buffer.byteLength(first)).toBe(Buffer.byteLength(second));
    await writeFile(file, first);
    expect(await store.logLength('large')).toBe(1);
    const before = await stat(file);
    await writeFile(file, second);
    await utimes(file, before.atime, new Date(before.mtimeMs + 2000));
    expect(await store.logLength('large')).toBe(2);
    expect(await new FileSessionStore(directory).logLength('large')).toBe(2);
  });

  it('preserves UTF-8, CRLF, chunk-split records and the ignored unterminated tail', async () => {
    const { file, store } = await fixture();
    const text = 'a'.repeat(65526) + '☃終🙂';
    await writeFile(file, JSON.stringify({ text }) + '\r\n{"n":2}\n{"unterminated":true}');
    io.wholeLogReads = 0;
    await expect(store.readLog('large')).resolves.toEqual([{ text }, { n: 2 }]);
    expect(io.wholeLogReads).toBe(0);
    expect(await store.logLength('large')).toBe(2);
    await store.appendLog('large', [{ n: 3 }]);
    expect(await store.readLog('large')).toEqual([{ text }, { n: 2 }, { n: 3 }]);
    expect(await store.logLength('large')).toBe(3);
  });

  it('keeps corruption and non-object line numbers, while dropping even a valid unterminated tail', async () => {
    const { file, store } = await fixture();
    for (const line of ['oops', '[]', 'null', '']) {
      await writeFile(file, '{"n":1}\n' + line + '\n{"ignored":');
      await expect(store.readLog('large')).rejects.toMatchObject({ code: 'unmappable_content', message: expect.stringContaining('line 2') });
    }
    await writeFile(file, '{"ignored":true}');
    expect(await store.readLog('large')).toEqual([]);
  });

  it('handles missing, empty, clear and forget without leaving a stale index', async () => {
    const { file, index, store } = await fixture();
    expect(await store.readLog('large')).toBeNull();
    expect(await store.logLength('large')).toBeNull();
    await writeFile(file, '');
    expect(await store.readLog('large')).toEqual([]);
    await store.appendLog('large', [{ n: 1 }]);
    expect(await store.logLength('large')).toBe(1);
    await store.clearLog('large');
    expect(await store.logLength('large')).toBe(0);
    await store.forget('large');
    expect(await store.logLength('large')).toBeNull();
    await expect(stat(index)).rejects.toMatchObject({ code: 'ENOENT' });
  });

  it('keeps a durable append successful when the advisory index cannot be replaced', async () => {
    const { index, store } = await fixture();
    const fs = await import('node:fs/promises');
    await fs.mkdir(index);
    await store.appendLog('large', [{ n: 1 }]);
    expect(await store.logLength('large')).toBe(1);
    expect(await store.readLog('large')).toEqual([{ n: 1 }]);
  });

  it('does not publish a stale count when another writer appends during a rebuild', async () => {
    const { directory, file, index, store } = await fixture();
    await writeFile(file, '{"n":1}\n');
    io.afterLogRead = async () => {
      const fs = await import('node:fs/promises');
      await fs.appendFile(file, '{"n":2}\n');
    };
    expect(await store.logLength('large')).toBe(1);
    await expect(stat(index)).rejects.toMatchObject({ code: 'ENOENT' });
    expect(await store.logLength('large')).toBe(2);
    expect(await new FileSessionStore(directory).logLength('large')).toBe(2);
  });

  it('reads a real log larger than the V8 string limit through the streaming path', async () => {
    const { file, store, directory } = await fixture();
    // Actual bytes, not a mocked stream. Padding is JSON whitespace so the
    // returned object array stays small even on the smallest supported runner.
    const count = 513;
    const chunk = Buffer.alloc(1024 * 1024, 0x20);
    const row = Buffer.from('{"text":"UTF8-終🙂"}');
    chunk.set(row); chunk[chunk.length - 1] = 0x0a;
    const handle = await open(file, 'w');
    try { for (let i = 0; i < count; i++) await handle.writeFile(chunk); }
    finally { await handle.close(); }
    expect((await stat(file)).size).toBeGreaterThan(512 * 1024 * 1024);
    io.wholeLogReads = 0;
    expect(await store.logLength('large')).toBe(count);
    const rows = await store.readLog('large');
    expect(rows).toHaveLength(count);
    expect(rows?.[0]).toEqual({ text: 'UTF8-終🙂' });
    expect(rows?.at(-1)).toEqual({ text: 'UTF8-終🙂' });
    expect(io.wholeLogReads).toBe(0);
    expect((await readdir(directory)).filter(name => name.endsWith('.tmp'))).toEqual([]);
  }, 120_000);
});
