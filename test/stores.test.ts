import { mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { describe, expect, it } from 'vitest';
import {
  FileSessionStore,
  HarnessError,
  MemorySessionStore,
  SessionStoreManager,
  type SessionStore,
} from '../src/index.js';

async function fileStore(): Promise<FileSessionStore> {
  return new FileSessionStore(await mkdtemp(join(tmpdir(), 'prism-harness-store-')));
}

/**
 * Both drivers must answer the same way. A test written against only the
 * in-memory one proves nothing about the driver a deployment actually uses,
 * and the two differ in exactly the places that matter — copying, expiry, and
 * whether a lock crosses a process.
 */
describe.each<[string, () => Promise<SessionStore>]>([
  ['memory', async () => new MemorySessionStore()],
  ['file', fileStore],
])('%s store', (_name, make) => {
  it('returns null for a key it has never seen', async () => {
    expect(await (await make()).get('nothing')).toBeNull();
  });

  it('round-trips a payload', async () => {
    const store = await make();
    await store.put('k', { mode: 'plan', depth: 2 });

    expect(await store.get('k')).toEqual({ mode: 'plan', depth: 2 });
  });

  it('forgets a key', async () => {
    const store = await make();
    await store.put('k', { a: 1 });
    await store.forget('k');

    expect(await store.get('k')).toBeNull();
  });

  it('expires a payload once its ttl has passed', async () => {
    const store = await make();
    await store.put('k', { a: 1 }, -1);

    expect(await store.get('k')).toBeNull();
  });

  it('hands back a COPY, so a caller cannot mutate what is stored', async () => {
    // Only the in-memory driver could ever get this wrong, which is exactly why
    // it is tested on both: a bug that appears on one driver is the worst kind,
    // because the in-memory one is what tests run against.
    const store = await make();
    await store.put('k', { list: [1, 2] });

    const first = (await store.get('k')) as { list: number[] };
    first.list.push(3);

    expect(await store.get('k')).toEqual({ list: [1, 2] });
  });

  it('runs a locked callback and returns its value', async () => {
    expect(await (await make()).withLock('k', () => 'done')).toBe('done');
  });

  it('releases the lock when the callback throws', async () => {
    const store = await make();

    await expect(
      store.withLock('k', () => {
        throw new Error('boom');
      }),
    ).rejects.toThrow('boom');

    // Would hang or time out if the lock leaked.
    expect(await store.withLock('k', () => 'free')).toBe('free');
  });

  it('SERIALISES two callers on the same key', async () => {
    const store = await make();
    const order: string[] = [];

    const first = store.withLock('k', async () => {
      order.push('first-in');
      await new Promise((resolve) => setTimeout(resolve, 30));
      order.push('first-out');
    });
    const second = store.withLock('k', () => {
      order.push('second-in');
    });

    await Promise.all([first, second]);

    // The second caller must not begin before the first has finished.
    expect(order).toEqual(['first-in', 'first-out', 'second-in']);
  });

  it('does not serialise DIFFERENT keys against each other', async () => {
    const store = await make();
    let released = false;

    const held = store.withLock('a', async () => {
      await new Promise((resolve) => setTimeout(resolve, 50));
      released = true;
    });

    await store.withLock('b', () => undefined);
    expect(released).toBe(false);

    await held;
  });

  it('throws session_locked rather than running the callback anyway', async () => {
    const store = await make();
    let ranWhileHeld = false;

    const held = store.withLock('k', async () => {
      await new Promise((resolve) => setTimeout(resolve, 200));
    });

    await expect(
      store.withLock(
        'k',
        () => {
          ranWhileHeld = true;
        },
        10,
        // A wait short enough to expire while the first caller still holds it.
        0.05,
      ),
    ).rejects.toMatchObject({ code: 'session_locked' });

    expect(ranWhileHeld).toBe(false);
    await held;
  });
});

describe('MemorySessionStore', () => {
  it('reports itself VOLATILE, which is what gets it refused for durable state', () => {
    expect(new MemorySessionStore().durability()).toBe('volatile');
  });
});

describe('FileSessionStore', () => {
  it('reports itself durable', async () => {
    expect((await fileStore()).durability()).toBe('durable');
  });

  it('survives a new store instance over the same directory', async () => {
    // The point of the driver. An in-memory store would return null here, and
    // that difference is the whole reason durability is declared rather than
    // inferred.
    const directory = await mkdtemp(join(tmpdir(), 'prism-harness-store-'));
    await new FileSessionStore(directory).put('k', { approval: 'pending' });

    expect(await new FileSessionStore(directory).get('k')).toEqual({ approval: 'pending' });
  });

  it('reclaims a lock whose holder died', async () => {
    // Left alone, a stale lockfile wedges the key forever — worse than the
    // small race in reclaiming it.
    const directory = await mkdtemp(join(tmpdir(), 'prism-harness-store-'));
    const store = new FileSessionStore(directory);
    await store.put('k', {});

    const files = await readFile(join(directory, (await lockName(directory, store))), 'utf8').catch(
      () => null,
    );
    expect(files).toBeNull(); // no lock is held yet

    // Plant an already-expired lock, as a killed process would leave behind.
    await writeFile(await lockPath(directory, store), String(Date.now() - 1000), 'utf8');

    expect(await store.withLock('k', () => 'recovered')).toBe('recovered');
  });

  it('refuses a stored payload that is not valid JSON, by code', async () => {
    const directory = await mkdtemp(join(tmpdir(), 'prism-harness-store-'));
    const store = new FileSessionStore(directory);
    await store.put('k', { a: 1 });
    await writeFile(await filePath(directory, store), 'not json', 'utf8');

    await expect(store.get('k')).rejects.toMatchObject({ code: 'unmappable_content' });
  });
});

describe('SessionStoreManager', () => {
  it('REFUSES a volatile store for the durable slot', async () => {
    // The guard the package exists for. Accepting it and finding out later is
    // exactly the failure it was written to avoid.
    const manager = new SessionStoreManager({
      drivers: { memory: () => new MemorySessionStore() },
      stores: { ephemeral: 'memory', durable: 'memory' },
    });

    expect(() => manager.ephemeral()).not.toThrow();
    expect(() => manager.durable()).toThrowError(HarnessError);

    try {
      manager.durable();
    } catch (error) {
      expect((error as HarnessError).code).toBe('unsafe_state_configuration');
      // The message has to name the fix, not just the fault.
      expect((error as HarnessError).message).toContain('durable driver');
    }
  });

  it('accepts a durable store for the durable slot', async () => {
    const directory = await mkdtemp(join(tmpdir(), 'prism-harness-store-'));
    const manager = new SessionStoreManager({
      drivers: { memory: () => new MemorySessionStore(), files: () => new FileSessionStore(directory) },
      stores: { ephemeral: 'memory', durable: 'files' },
    });

    expect(manager.durable().durability()).toBe('durable');
  });

  it('names the slot and the driver when a driver is not registered', () => {
    const manager = new SessionStoreManager({ drivers: {}, stores: { durable: 'redis' } });

    expect(() => manager.durable()).toThrowError(/redis/);
  });

  it('builds each driver at most once', () => {
    let built = 0;
    const manager = new SessionStoreManager({
      drivers: {
        memory: () => {
          built += 1;

          return new MemorySessionStore();
        },
      },
      stores: { ephemeral: 'memory', durable: 'memory' },
    });

    manager.ephemeral();
    manager.ephemeral();

    expect(built).toBe(1);
  });
});

// The store hashes its key to a filename; these mirror that so a test can reach
// the file it wrote without the store exposing its layout.
async function filePath(directory: string, _store: FileSessionStore): Promise<string> {
  const { createHash } = await import('node:crypto');

  return join(directory, `${createHash('sha256').update('k').digest('hex').slice(0, 32)}.json`);
}

async function lockPath(directory: string, store: FileSessionStore): Promise<string> {
  return `${await filePath(directory, store)}.lock`;
}

async function lockName(directory: string, store: FileSessionStore): Promise<string> {
  return (await lockPath(directory, store)).slice(directory.length + 1);
}
