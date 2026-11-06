import { createHash, randomBytes } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { lstat, mkdir, mkdtemp, readFile, unlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, expect, it, vi } from 'vitest';
import { FileSessionStore } from '../src/index.js';
import { nativeLockIdentity } from '../src/stores/file-lock-identity.js';
import type { FileLockOwner } from '../src/stores/file-owner-lock.js';

vi.mock('node:fs/promises', async importOriginal => {
  const actual = await importOriginal<typeof import('node:fs/promises')>();
  return { ...actual, unlink: vi.fn(actual.unlink) };
});
const actual = await vi.importActual<typeof import('node:fs/promises')>('node:fs/promises');
const vector = JSON.parse(readFileSync(new URL('./fixtures/file-lock-cleanup-os-refusal.json', import.meta.url), 'utf8')) as {
  id: string; os_errors: { ts: string[] }; required_cases: string[];
};
if (vector.id !== 'cleanup-refused-by-os' || JSON.stringify(vector.os_errors.ts) !== '["EPERM","EBUSY"]'
  || JSON.stringify(vector.required_cases) !== '["held-then-cleared","budget-exhausted","unrelated-error"]') {
  throw new Error('Unknown or incomplete canonical cleanup-refusal vector');
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.mocked(unlink).mockReset().mockImplementation(actual.unlink);
});

async function fixture() {
  const directory = await mkdtemp(join(tmpdir(), 'prism-cleanup-held-g242-'));
  const store = new FileSessionStore(directory);
  const lock = join(directory, createHash('sha256').update('k').digest('hex').slice(0, 32) + '.json.lock');
  const identity = nativeLockIdentity();
  expect(identity?.pid_namespace).toBeTypeOf('string');
  const owner: FileLockOwner = {
    version: 2, pid: process.pid, identity: identity!, nonce: randomBytes(16).toString('hex'),
    released: true, expires_at: 0,
  };
  const filename = join(lock, 'owner-' + owner.nonce + '.json');
  const temporary = join(lock, 'owner-' + owner.nonce + '.release.tmp');
  const bytes = JSON.stringify(owner) + '\n';
  await mkdir(lock);
  await writeFile(filename, bytes);
  await writeFile(temporary, bytes);
  return { store, lock, filename, temporary, bytes };
}

it.each(vector.os_errors.ts)('waits and re-inspects when temporary cleanup is blocked by %s', async code => {
  const { store, lock, filename, temporary } = await fixture();
  const timer = vi.spyOn(globalThis, 'setTimeout');
  vi.spyOn(performance, 'now').mockReturnValue(0);
  let denied = 0, callbacks = 0;
  vi.mocked(unlink).mockImplementation(async path => {
    if (String(path) === temporary) {
      if (denied++ === 0) throw Object.assign(new Error('open Windows handle'), { code });
      expect(callbacks).toBe(0);
      expect(timer.mock.calls.some(call => call[1] === 25)).toBe(true);
    }
    return actual.unlink(path);
  });
  expect(await store.withLock('k', async () => {
    expect(timer.mock.calls.some(call => call[1] === 25)).toBe(true);
    await expect(readFile(filename)).rejects.toMatchObject({ code: 'ENOENT' });
    callbacks++;
    return 7;
  })).toBe(7);
  expect(denied).toBe(2);
  expect(callbacks).toBe(1);
  await expect(lstat(lock)).rejects.toMatchObject({ code: 'ENOENT' });
});

it.each(vector.os_errors.ts)('refuses within the existing exhausted wait budget for %s', async code => {
  const { store, filename, temporary, bytes } = await fixture();
  let callbacks = 0, denied = 0;
  vi.mocked(unlink).mockImplementation(async path => {
    if (String(path) === temporary) {
      denied++;
      throw Object.assign(new Error('open Windows handle'), { code });
    }
    return actual.unlink(path);
  });
  await expect(store.withLock('k', () => { callbacks++; }, 10, 0)).rejects.toMatchObject({ code: 'session_locked' });
  expect(denied).toBe(1);
  expect(callbacks).toBe(0);
  expect(await readFile(filename, 'utf8')).toBe(bytes);
  expect(await readFile(temporary, 'utf8')).toBe(bytes);
});

it('propagates unrelated cleanup I/O errors without granting entry', async () => {
  const { store, filename, temporary, bytes } = await fixture();
  const error = Object.assign(new Error('device error'), { code: 'EIO' });
  let callbacks = 0;
  vi.mocked(unlink).mockImplementation(async path => {
    if (String(path) === temporary) throw error;
    return actual.unlink(path);
  });
  await expect(store.withLock('k', () => { callbacks++; }, 10, 0)).rejects.toBe(error);
  expect(callbacks).toBe(0);
  expect(await readFile(filename, 'utf8')).toBe(bytes);
  expect(await readFile(temporary, 'utf8')).toBe(bytes);
});
