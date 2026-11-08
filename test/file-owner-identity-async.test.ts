import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

type Done = (error: Error | null, stdout: string, stderr: string) => void;
const io = vi.hoisted(() => ({ sync: vi.fn(), async: vi.fn(), pending: [] as Done[] }));
vi.mock('node:child_process', () => ({
  spawnSync: (...args: unknown[]) => io.sync(...args),
  execFile: (...args: unknown[]) => { io.async(...args.slice(0, 3)); io.pending.push(args[3] as Done); return {}; },
}));
const originalPlatform = Object.getOwnPropertyDescriptor(process, 'platform')!;
const directories: string[] = [];
const output = (platform: string) => platform === 'win32'
  ? 'boot=638000000000000000\ncontainer=absent\n'
  : '11111111-1111-4111-8111-111111111111\n';
beforeEach(() => {
  vi.resetModules(); io.sync.mockReset(); io.async.mockReset(); io.pending.length = 0;
  Object.defineProperty(process, 'platform', { ...originalPlatform, value: 'win32' });
  vi.stubEnv('SystemRoot', 'C:\\Windows');
  io.sync.mockImplementation(() => ({ status: 0, stdout: output(process.platform), stderr: '' }));
});
afterEach(async () => {
  vi.restoreAllMocks(); vi.unstubAllEnvs(); Object.defineProperty(process, 'platform', originalPlatform);
  for (const directory of directories.splice(0)) {
    expect(dirname(resolve(directory))).toBe(resolve(tmpdir()));
    expect(basename(directory)).toMatch(/^prism-async-identity-/);
    await rm(directory, { recursive: true, force: true });
  }
});
const tick = () => new Promise<void>(resolve => setImmediate(resolve));
async function dispatched(): Promise<void> {
  const deadline = Date.now() + 2000;
  while (io.async.mock.calls.length + io.sync.mock.calls.length === 0 && Date.now() < deadline) {
    await new Promise<void>(resolve => setTimeout(resolve, 10));
  }
}
const finish = (error: Error | null = null) => {
  const callback = io.pending.shift(); expect(callback).toBeTypeOf('function');
  callback!(error, error === null ? output(process.platform) : '', '');
};

it.each(['win32', 'darwin'] as const)('never calls spawnSync on the actual %s lock path, and waits without blocking the loop', async platform => {
  Object.defineProperty(process, 'platform', { ...originalPlatform, value: platform });
  const { FileSessionStore } = await import('../src/index.js');
  const directory = await mkdtemp(join(tmpdir(), 'prism-async-identity-')); directories.push(directory);
  const store = new FileSessionStore(directory); let entered = false;
  const acquisition = store.withLock('owned', () => { entered = true; });
  try {
    await dispatched();
    expect(io.sync).not.toHaveBeenCalled();
    expect(io.async).toHaveBeenCalledTimes(1);
    expect(entered).toBe(false);
    let heartbeat = false; await new Promise<void>(resolve => setTimeout(() => { heartbeat = true; resolve(); }, 0));
    expect(heartbeat).toBe(true); expect(entered).toBe(false);
    finish(); await acquisition; expect(entered).toBe(true);
  } finally {
    while (io.pending.length > 0) finish();
    await acquisition;
  }
});

it('shares one in-flight identity sample across concurrent first locks', async () => {
  const { FileSessionStore } = await import('../src/index.js');
  const directory = await mkdtemp(join(tmpdir(), 'prism-async-identity-')); directories.push(directory);
  const store = new FileSessionStore(directory); let entered = 0;
  const acquisitions = Array.from({ length: 12 }, (_, i) => store.withLock('key-' + i, () => { entered++; }));
  try { await dispatched(); expect(io.async).toHaveBeenCalledTimes(1); expect(io.sync).not.toHaveBeenCalled(); expect(entered).toBe(0); finish(); await Promise.all(acquisitions); expect(entered).toBe(12); }
  finally { while (io.pending.length > 0) finish(); await Promise.all(acquisitions); }
});

it('shares the same promise, publishes status only after completion, and freezes the result', async () => {
  const identity = await import('../src/stores/file-lock-identity.js');
  const one = identity.nativeLockIdentity(), two = identity.nativeLockIdentity();
  try {
    expect(one).toBeInstanceOf(Promise); expect(two).toBe(one);
    expect(identity.fileLockIdentityStatus().sampled).toBe(false);
    finish(); const result = await one;
    expect(result).toMatchObject({ platform: 'win32', pid_namespace: 'win32:kernel' });
    expect(Object.isFrozen(result)).toBe(true);
    expect(await identity.nativeLockIdentity()).toBe(result);
    expect(io.async).toHaveBeenCalledTimes(1);
  } finally { while (io.pending.length > 0) finish(); }
});

it('keeps failure and the sixty-second retry asynchronous and shares the retry sample', async () => {
  let now = 0; vi.spyOn(performance, 'now').mockImplementation(() => now);
  const identity = await import('../src/stores/file-lock-identity.js');
  const failed = identity.nativeLockIdentity();
  try {
    expect(io.async).toHaveBeenCalledTimes(1); finish(new Error('owned query failure')); expect(await failed).toBeNull();
    now = 59999; expect(await identity.nativeLockIdentity()).toBeNull(); expect(io.async).toHaveBeenCalledTimes(1);
    expect(identity.fileLockIdentityStatus()).toMatchObject({ definitive: false, retry_after_ms: 1, code: 'file_lock_identity_unavailable' });
    now = 60000; const retried = Array.from({ length: 12 }, () => identity.nativeLockIdentity());
    expect(io.async).toHaveBeenCalledTimes(2); expect(retried.every(p => p === retried[0])).toBe(true);
    await tick(); finish(); const results = await Promise.all(retried);
    expect(results.every(value => value?.pid_namespace === 'win32:kernel')).toBe(true);
    now = 99999999; expect(await identity.nativeLockIdentity()).toBe(results[0]); expect(io.async).toHaveBeenCalledTimes(2);
    expect(io.sync).not.toHaveBeenCalled();
  } finally { while (io.pending.length > 0) finish(); }
});

it('exports optional async warmup and a pure status helper without sampling on observation', async () => {
  const api = await import('../src/index.js');
  const proposed = api as typeof api & { warmLockIdentity: () => Promise<void>; fileLockIdentityStatus: () => { sampled: boolean } };
  expect(proposed.warmLockIdentity).toBeTypeOf('function'); expect(proposed.fileLockIdentityStatus).toBeTypeOf('function');
  expect(proposed.fileLockIdentityStatus().sampled).toBe(false); expect(io.async).not.toHaveBeenCalled(); expect(io.sync).not.toHaveBeenCalled();
  const one = proposed.warmLockIdentity(), two = proposed.warmLockIdentity();
  try { expect(io.async).toHaveBeenCalledTimes(1); finish(); await Promise.all([one, two]); expect(proposed.fileLockIdentityStatus().sampled).toBe(true); }
  finally { while (io.pending.length > 0) finish(); }
});
