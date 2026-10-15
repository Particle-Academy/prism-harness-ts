import { mkdtemp, readdir, readFile, writeFile as rawWriteFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';

// Every byte the store hands to the filesystem, whichever call it uses. Bytes
// WRITTEN, not wall time: the cost Genie 2 paid was a full rewrite of a 9 MB
// thread per chat row, and that is a byte count before it is a latency.
const written = { bytes: 0 };

vi.mock('node:fs/promises', async (importOriginal) => {
  const actual = await importOriginal<typeof import('node:fs/promises')>();
  const size = (data: unknown): number =>
    typeof data === 'string' ? Buffer.byteLength(data) : data instanceof Uint8Array ? data.byteLength : 0;

  return {
    ...actual,
    writeFile: async (...args: Parameters<typeof actual.writeFile>) => {
      written.bytes += size(args[1]);
      return actual.writeFile(...args);
    },
    appendFile: async (...args: Parameters<typeof actual.appendFile>) => {
      written.bytes += size(args[1]);
      return actual.appendFile(...args);
    },
    open: async (...args: Parameters<typeof actual.open>) => {
      const handle = await actual.open(...args);
      const write = handle.write.bind(handle) as (...a: unknown[]) => Promise<unknown>;
      const writeFile = handle.writeFile.bind(handle);
      const appendFile = handle.appendFile.bind(handle);
      return Object.assign(handle, {
        write: (...a: unknown[]) => { written.bytes += size(a[0]); return write(...a); },
        writeFile: (data: string | Uint8Array, options?: unknown) => {
          written.bytes += size(data); return writeFile(data, options as never);
        },
        appendFile: (data: string | Uint8Array, options?: unknown) => {
          written.bytes += size(data); return appendFile(data, options as never);
        },
      });
    },
  };
});

const { FileSessionStore, Thread } = await import('../src/index.js');

const directories: string[] = [];
afterEach(() => { directories.length = 0; });

async function directory(): Promise<string> {
  const path = await mkdtemp(join(tmpdir(), 'prism-harness-thread-log-'));
  directories.push(path);
  return path;
}

function message(index: number): { type: string; content: string } {
  return { type: 'user', content: `message number ${String(index).padStart(6, '0')}` };
}

async function bytesForOneAppend(thread: InstanceType<typeof Thread>, index: number): Promise<number> {
  written.bytes = 0;
  await thread.record([message(index)]);
  return written.bytes;
}

async function logFile(path: string): Promise<string> {
  const names = (await readdir(path)).filter((name) => name.endsWith('.jsonl'));
  expect(names).toHaveLength(1);
  return join(path, names[0]!);
}

describe('the thread as an append-only log', () => {
  it('writes the same bytes per append at 10 messages and at 30,000', async () => {
    const small = new Thread(new FileSessionStore(await directory()), 'small:thread');
    await small.record(Array.from({ length: 10 }, (_, i) => message(i)));
    const atTen = await bytesForOneAppend(small, 10);

    const large = new Thread(new FileSessionStore(await directory()), 'large:thread');
    await large.record(Array.from({ length: 30_000 }, (_, i) => message(i)));
    const atThirtyThousand = await bytesForOneAppend(large, 30_000);

    // One message's worth either way. The position grows by a few digits, so
    // "flat" means within a handful of bytes, not identical.
    expect(atTen).toBeGreaterThan(0);
    expect(atTen).toBeLessThan(1024);
    expect(atThirtyThousand - atTen).toBeLessThan(16);
    expect(await large.count()).toBe(30_001);
  }, 60_000);

  it('drops a torn last line after a crash, and the next append lands cleanly after it', async () => {
    const path = await directory();
    const thread = new Thread(new FileSessionStore(path), 'torn:thread');
    await thread.record([message(1), message(2)]);

    // A process killed mid-write leaves a prefix of a line with no terminator.
    const file = await logFile(path);
    await rawWriteFile(file, `${await readFile(file, 'utf8')}{"position":3,"message":{"type":"us`, 'utf8');

    const reopened = new Thread(new FileSessionStore(path), 'torn:thread');
    expect((await reopened.messages()).map((entry) => entry.position)).toEqual([1, 2]);
    expect(await reopened.count()).toBe(2);

    const [third] = await reopened.record([message(3)]);
    expect(third?.position).toBe(3);
    expect((await reopened.messages()).map((entry) => entry.message.content)).toEqual([
      message(1).content, message(2).content, message(3).content,
    ]);
    // Every line on disk is whole JSON: the torn bytes were cut, not glued on.
    for (const line of (await readFile(file, 'utf8')).split('\n').filter(Boolean)) {
      expect(() => JSON.parse(line)).not.toThrow();
    }
  });

  it('refuses a corrupt line that is NOT the last, rather than silently dropping history', async () => {
    const path = await directory();
    const thread = new Thread(new FileSessionStore(path), 'corrupt:thread');
    await thread.record([message(1), message(2)]);
    const file = await logFile(path);
    const lines = (await readFile(file, 'utf8')).split('\n');
    lines[0] = '{"position":1,"message":';
    await rawWriteFile(file, lines.join('\n'), 'utf8');

    await expect(new Thread(new FileSessionStore(path), 'corrupt:thread').messages())
      .rejects.toMatchObject({ code: 'unmappable_content' });
  });

  it('reads an existing whole-file thread, then appends to it without loss or renumbering', async () => {
    const path = await directory();
    const store = new FileSessionStore(path);
    // Exactly what 0.6.0's Thread.record() stored.
    await store.put('legacy:thread', { messages: [1, 2, 3].map((position) => ({
      position, message: message(position), run_id: 'run-0', recorded_at: '2026-10-01T00:00:00.000Z',
    })) });

    const thread = new Thread(store, 'legacy:thread');
    expect((await thread.messages()).map((entry) => entry.position)).toEqual([1, 2, 3]);
    expect(await thread.count()).toBe(3);

    const [fourth] = await thread.record([message(4)], 'run-1');
    expect(fourth?.position).toBe(4);

    const reread = await new Thread(new FileSessionStore(path), 'legacy:thread').messages();
    expect(reread.map((entry) => [entry.position, entry.runId])).toEqual([
      [1, 'run-0'], [2, 'run-0'], [3, 'run-0'], [4, 'run-1'],
    ]);
    expect(reread[0]?.recordedAt).toBe('2026-10-01T00:00:00.000Z');
  });

  it('assigns positions correctly when two store instances share a directory', async () => {
    // Two processes, as far as the store can tell: no shared memory, one lock.
    const path = await directory();
    const one = new Thread(new FileSessionStore(path), 'shared:thread');
    const two = new Thread(new FileSessionStore(path), 'shared:thread');

    await one.record([message(1)]);
    await two.record([message(2)]);
    await one.record([message(3)]);

    expect(await one.count()).toBe(3);
    expect((await two.messages()).map((entry) => entry.position)).toEqual([1, 2, 3]);
  });

  it('starts again from position 1 after clear()', async () => {
    const thread = new Thread(new FileSessionStore(await directory()), 'clear:thread');
    await thread.record([message(1), message(2)]);
    await thread.clear();
    expect(await thread.count()).toBe(0);
    expect((await thread.record([message(3)]))[0]?.position).toBe(1);
  });
});
