import { mkdir, mkdtemp, readdir, readFile, utimes, writeFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { nativeLockIdentity } from '../src/stores/file-lock-identity.js';
import { parseFileLockOwner } from '../src/stores/file-owner-lock.js';
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

async function plantOwner(path:string,pid:number,released:boolean):Promise<void> {
  const identity=nativeLockIdentity(); expect(identity?.pid_namespace).toBeTypeOf('string');
  const nonce=randomBytes(16).toString('hex');
  await mkdir(path);
  await writeFile(join(path,'owner-'+nonce+'.json'),JSON.stringify({
    version:2,pid,identity,nonce,released,expires_at:Math.floor(Date.now()/1000)+60,
  })+'\n');
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
    // Keep A inside an explicit barrier. Filesystem scheduling is not a clock
    // guarantee; B must enter while A remains held, regardless of elapsed time.
    for (const forceGlobalKey of [false, true]) {
      const store = await make();
      let entered!: () => void, release!: () => void, released = false, callbacks = 0;
      const ready = new Promise<void>(done => { entered = done; });
      const barrier = new Promise<void>(done => { release = done; });
      const held = store.withLock('a', async () => {
        entered(); await barrier; released = true;
      });
      try {
        await Promise.race([ready, held]);
        const second = store.withLock(forceGlobalKey ? 'a' : 'b', () => {
          callbacks++; expect(released).toBe(false);
        }, 10, 0.1);
        if (forceGlobalKey) {
          // Deliberately collapse both logical keys onto one actual lock: the
          // same apparatus must observe refusal and zero B callbacks.
          await expect(second).rejects.toMatchObject({ code: 'session_locked' });
          expect(callbacks).toBe(0);
        } else {
          await second; expect(callbacks).toBe(1);
        }
        expect(released).toBe(false);
      } finally { release(); await held; }
    }
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

/** The old timestamp protocol never grants v2 recovery authority, at any age. */
describe('file store: legacy timestamp files require explicit maintenance', () => {
  it.each(['', '123', '1735689', '1\n', '0\n', String(Date.now()+60_000)+'\n'])('refuses legacy bytes %j even with an old mtime', async stamp => {
    const directory = await mkdtemp(join(tmpdir(), 'prism-harness-legacy-'));
    const store = new FileSessionStore(directory), path = await lockPath(directory, store);
    await writeFile(path, stamp, 'utf8');
    const longAgo = new Date(0); await utimes(path, longAgo, longAgo);
    let callbacks = 0;
    await expect(store.withLock('k', () => {callbacks++;},10,0)).rejects.toMatchObject({
      code:'session_locked',message:expect.stringContaining('legacy_lock_requires_maintenance'),
    });
    expect(callbacks).toBe(0);
    expect(await readFile(path,'utf8')).toBe(stamp);
  });

  it('reads the complete real writer record before any callback and refuses to steal its expired hint', async () => {
    const directory=await mkdtemp(join(tmpdir(),'prism-harness-roundtrip-'));
    const store=new FileSessionStore(directory), path=await lockPath(directory,store);
    let bytes=Buffer.alloc(0), filename='';
    await store.withLock('k',async()=>{
      [filename]=(await readdir(path));
      bytes=await readFile(join(path,filename));
      const record=parseFileLockOwner(bytes,filename);
      expect(record).not.toBeNull(); expect(record!.released).toBe(false);
      expect(bytes.at(-1)).toBe(10);
      // Same key is still held even though the real writer emitted an old TTL.
      await expect(store.withLock('k',()=>{throw Error('stolen');},10,0)).rejects.toMatchObject({code:'session_locked'});
    },-1000);
    // Writer/reader roundtrip after the callback: unreleased live record holds;
    // the same record explicitly released is recoverable, irrespective of TTL.
    await mkdir(path); await writeFile(join(path,filename),bytes);
    await expect(store.withLock('k',()=>1,10,0)).rejects.toMatchObject({code:'session_locked'});
    const released={...JSON.parse(bytes.toString('utf8')),released:true};
    await writeFile(join(path,filename),JSON.stringify(released)+'\n');
    expect(await store.withLock('k',()=>7,10,0.5)).toBe(7);
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

  it('reclaims a proven-dead same-identity holder without using expiry as proof', async () => {
    const directory=await mkdtemp(join(tmpdir(),'prism-harness-dead-'));
    const store=new FileSessionStore(directory);
    const child=spawnSync(process.execPath,['-e',''],{windowsHide:true,timeout:5000});
    expect(child.status).toBe(0); expect(child.error).toBeUndefined();
    await plantOwner(await lockPath(directory,store),child.pid,false);
    expect(await store.withLock('k',()=> 'recovered')).toBe('recovered');
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

  it('reclaims an explicitly released same-identity owner without waiting for TTL', async () => {
    const directory=await mkdtemp(join(tmpdir(),'prism-harness-released-'));
    const store=new FileSessionStore(directory);
    await plantOwner(await lockPath(directory,store),process.pid,true);
    const started=performance.now();
    expect(await store.withLock('k',()=> 'reclaimed',10,0.5)).toBe('reclaimed');
    expect(performance.now()-started).toBeLessThan(400);
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
