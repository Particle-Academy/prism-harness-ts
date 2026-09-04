import { mkdtemp, readFile, utimes, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
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
 * A FINISHED lock stamp: the expiry, then the terminator the reader requires.
 *
 * Every test below that plants a lockfile by hand goes through here, so none of
 * them can accidentally plant a half-written one and then assert the reader's
 * behaviour on the wrong input. The format is shared with the PHP and Python
 * ports; `writes a stamp in the SHARED format` asserts the real writer emits
 * exactly this, which is the only thing keeping this helper honest.
 */
function lockStamp(expiresAt: number): string {
  // The newline as a code point rather than an escape: this file is edited by
  // tooling that has mangled `\` sequences here before, and a terminator that
  // silently became something else would make every test below assert the
  // opposite of what it claims.
  return `${String(Math.trunc(expiresAt))}${String.fromCharCode(10)}`;
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

    // MUTUAL EXCLUSION is what both drivers promise: the two critical sections
    // must not overlap. WHICH of them wins is not promised by the file driver
    // and cannot be — both callers race an exclusive `open`, and the loser is
    // decided by the OS. Asserting the arrival order here instead failed about
    // one run in twenty, and a gate that is red one time in twenty is a gate
    // nobody believes. The ordering guarantee the in-memory driver does make is
    // asserted on its own, below.
    expect(order).toHaveLength(3);
    expect(order).toContain('second-in');
    expect(order.indexOf('first-out')).toBe(order.indexOf('first-in') + 1);
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

    // Wait for the first caller to ACTUALLY hold the lock before the second
    // tries. Starting both and assuming the first wins is not safe on the file
    // driver: both callers race an exclusive `open` and the OS picks, so the
    // "waiter" sometimes took the lock cleanly and this test failed claiming a
    // promise had resolved. That was the test being wrong, not the lock.
    let acquired!: () => void;
    const holding = new Promise<void>((resolve) => {
      acquired = resolve;
    });

    const held = store.withLock('k', async () => {
      acquired();
      await new Promise((resolve) => setTimeout(resolve, 200));
    });

    await holding;

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

describe('memory store lock ordering', () => {
  it('queues waiters in ARRIVAL order', async () => {
    // The in-memory driver chains waiters onto a promise, so it can promise
    // FIFO and does. Asserted here rather than in the shared block because the
    // file driver cannot promise it: two callers race an exclusive create and
    // the OS picks. Keeping the stronger assertion where it is true beats
    // deleting it because one driver of two cannot meet it.
    const store = new MemorySessionStore();
    const order: string[] = [];

    const runners = ['a', 'b', 'c'].map((name) =>
      store.withLock('k', async () => {
        order.push(name);
        await new Promise((resolve) => setTimeout(resolve, 5));
      }),
    );

    await Promise.all(runners);

    expect(order).toEqual(['a', 'b', 'c']);
  });
});

/**
 * The window between creating the lockfile and stamping it.
 *
 * Taking the lock is `open(…, 'wx')` and THEN writing the expiry — two
 * operations, with a moment where the file exists and is empty. A waiter
 * reading it in that moment used to compute `Number('') === 0`, conclude the
 * lock had expired in 1970, delete it, and take a lock somebody was actively
 * holding. That is the same key handed to two callers, which is the one thing
 * this lock exists to prevent.
 *
 * It was not theoretical and it was not rare: it failed the suite above about
 * one run in eight, as a timing flake, which is how it survived.
 *
 * These tests recreate the window directly rather than racing for it, because a
 * test that has to win a race to fail is a test that passes for the wrong
 * reason most of the time.
 */
describe('file store: a lockfile that exists but is not yet stamped', () => {
  async function lockedDirectory(): Promise<{ store: FileSessionStore; lockPath: string }> {
    const directory = await mkdtemp(join(tmpdir(), 'prism-harness-lockwindow-'));
    const digest = createHash('sha256').update('k').digest('hex').slice(0, 32);

    return { store: new FileSessionStore(directory), lockPath: join(directory, `${digest}.json.lock`) };
  }

  it('is treated as HELD, not as expired in 1970', async () => {
    const { store, lockPath } = await lockedDirectory();
    let ranWhileHeld = false;

    await writeFile(lockPath, '', 'utf8');

    await expect(
      store.withLock(
        'k',
        () => {
          ranWhileHeld = true;
        },
        10,
        0.05,
      ),
    ).rejects.toMatchObject({ code: 'session_locked' });

    expect(ranWhileHeld).toBe(false);
  });

  it('is still reclaimed once it is older than the ttl', async () => {
    // The other half of the fix. Treating an empty lockfile as held FOREVER
    // would let a process that died inside that window wedge the key, which is
    // a worse failure than the one being fixed. The file's own age is the
    // bound, so nothing new has to be written to make it recoverable.
    const { store, lockPath } = await lockedDirectory();

    await writeFile(lockPath, '', 'utf8');

    const longAgo = new Date(Date.now() - 60_000);
    await utimes(lockPath, longAgo, longAgo);

    expect(await store.withLock('k', () => 'taken', 10, 0.5)).toBe('taken');
  });

  it('is treated as HELD when the stamp is TRUNCATED, not expired in 1970', async () => {
    // The empty case reached through a different door, and the more dangerous
    // one because it does not look broken. A torn write leaves a PREFIX of the
    // timestamp — `17356899` where `1735689900000` was meant — and every prefix
    // of a number is a smaller number. So it parses, it is finite, and it is
    // decades in the past: the reader concludes the holder is long dead and
    // deletes a lock somebody is holding right now.
    //
    // Refusing anything that is not terminated is what tells "torn" apart from
    // "small". Unreadable has to mean WAIT.
    const { store, lockPath } = await lockedDirectory();
    let ranWhileHeld = false;

    await writeFile(lockPath, String(Date.now() + 10_000).slice(0, 8), 'utf8');

    await expect(
      store.withLock(
        'k',
        () => {
          ranWhileHeld = true;
        },
        10,
        0.05,
      ),
    ).rejects.toMatchObject({ code: 'session_locked' });

    expect(ranWhileHeld).toBe(false);
  });

  it('is still reclaimed once a TRUNCATED stamp is older than the ttl', async () => {
    // Same bound as the empty case, and for the same reason: a torn write from
    // a process that then died must not wedge the key forever.
    const { store, lockPath } = await lockedDirectory();

    await writeFile(lockPath, String(Date.now() + 10_000).slice(0, 8), 'utf8');

    const longAgo = new Date(Date.now() - 60_000);
    await utimes(lockPath, longAgo, longAgo);

    expect(await store.withLock('k', () => 'taken', 10, 0.5)).toBe('taken');
  });

  it('writes a stamp its OWN reader accepts', async () => {
    // A ROUND TRIP through the real writer and the real reader. Every other
    // test in this block plants a lockfile by hand, so all of them would stay
    // green if the writer stopped emitting the terminator the reader now
    // demands — Python's mutation run found exactly that hole in its own
    // suite. This is the test that closes it.
    //
    // The outer lock is taken with a ttl of zero, so its stamp is already in
    // the past by the time anything reads it. The inner caller therefore has to
    // PARSE what the writer wrote in order to reclaim: a stamp the reader
    // cannot read falls through to the mtime bound, which has not elapsed, and
    // the inner call would time out instead.
    const { store } = await lockedDirectory();

    const reclaimed = await store.withLock(
      'k',
      async () => store.withLock('k', () => 'reclaimed', 10, 1),
      0,
      5,
    );

    expect(reclaimed).toBe('reclaimed');
  });

  it('writes a stamp in the SHARED format: an integer and a terminator', async () => {
    // The lockfile format is cross-language surface. A PHP, TypeScript or
    // Python worker pointed at one store directory has to agree on it, and
    // Python already requires the trailing newline: without it every lock this
    // port takes reads as unterminated there, so a Python worker waits our
    // locks out rather than reclaiming a dead one. Safe direction, real
    // divergence. Asserted on the bytes, from inside a held lock.
    const { store, lockPath } = await lockedDirectory();

    const raw = await store.withLock('k', async () => readFile(lockPath, 'utf8'), 10, 5);

    expect(raw.endsWith('\n')).toBe(true);
    expect(raw.slice(0, -1)).toMatch(/^\d+$/);
    expect(Number.isInteger(Number(raw.slice(0, -1)))).toBe(true);
  });

  it('still reclaims a stamped lock whose expiry has passed', async () => {
    // The pre-existing behaviour, asserted so the fix above cannot quietly take
    // it away: a lock left by a dead holder must not wedge the key.
    const { store, lockPath } = await lockedDirectory();

    await writeFile(lockPath, lockStamp(Date.now() - 1_000), 'utf8');

    expect(await store.withLock('k', () => 'taken', 10, 0.5)).toBe('taken');
  });

  it('does not reclaim a stamped lock whose expiry is in the future', async () => {
    const { store, lockPath } = await lockedDirectory();
    let ran = false;

    await writeFile(lockPath, lockStamp(Date.now() + 60_000), 'utf8');

    await expect(
      store.withLock(
        'k',
        () => {
          ran = true;
        },
        10,
        0.05,
      ),
    ).rejects.toMatchObject({ code: 'session_locked' });

    expect(ran).toBe(false);
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

    // Plant an already-expired lock, as a killed process would leave behind —
    // in the REAL format, terminator and all. Without the newline this is a
    // torn write rather than an old one, and the reader is right to wait.
    await writeFile(await lockPath(directory, store), lockStamp(Date.now() - 1000), 'utf8');

    expect(await store.withLock('k', () => 'recovered')).toBe('recovered');
  });

  it('survives many contending callers', async () => {
    // A general concurrency assertion: every caller gets through, none is
    // starved. It does NOT discriminate the release-leak bug below — verified
    // by reverting the fix and watching this still pass — so it is not the
    // regression test for it, and saying so matters more than the green tick.
    const store = await fileStore();
    let entered = 0;

    await Promise.all(
      Array.from({ length: 12 }, () =>
        store.withLock('contended', async () => {
          entered += 1;
          await new Promise((resolve) => setTimeout(resolve, 1));
        }),
      ),
    );

    expect(entered).toBe(12);
  }, 20_000);

  it('leaves NO lockfile behind after a normal release', async () => {
    // The property `#release` exists to guarantee. Its retry-then-mark-dead
    // fallback cannot be induced portably — an unlink only fails while another
    // caller happens to hold the path, which is a Windows timing window — so
    // this asserts the outcome that IS observable everywhere, and the test
    // below covers the state that fallback leaves behind.
    const directory = await mkdtemp(join(tmpdir(), 'prism-harness-store-'));
    const store = new FileSessionStore(directory);
    await store.withLock('k', () => undefined);

    await expect(readFile(await lockPath(directory, store), 'utf8')).rejects.toMatchObject({
      code: 'ENOENT',
    });
  });

  it('reclaims a lock marked dead, WITHOUT waiting out the ttl', async () => {
    // The contract `#release`'s fallback depends on: when it cannot delete the
    // lockfile it rewrites it with an already-past expiry rather than leaving
    // it held, and a waiter must then reclaim it on the next attempt instead of
    // blocking for the full wait. `prism-harness-py` hit that leak on its first
    // concurrent test; this port had the same defect.
    const directory = await mkdtemp(join(tmpdir(), 'prism-harness-store-'));
    const store = new FileSessionStore(directory);
    await store.put('k', {});
    // `#release` writes a TERMINATED zero, so this plants what it plants.
    await writeFile(await lockPath(directory, store), lockStamp(0), 'utf8');

    const started = Date.now();
    // A wait far shorter than the 10s lock ttl: reclaiming has to come from the
    // expiry marker, not from the lock timing out.
    const result = await store.withLock('k', () => 'reclaimed', 10, 0.5);

    expect(result).toBe('reclaimed');
    expect(Date.now() - started).toBeLessThan(400);
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
