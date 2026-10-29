import { createHash } from 'node:crypto';
import { mkdtemp, lstat, readFile, readdir, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { expect, it, vi } from 'vitest';
import { FileSessionStore } from '../src/index.js';
import * as identities from '../src/stores/file-lock-identity.js';

async function fixture(): Promise<{ store: FileSessionStore; lock: string }> {
  const directory = await mkdtemp(join(tmpdir(), 'prism-owner-lock-'));
  const digest = createHash('sha256').update('k').digest('hex').slice(0, 32);
  return { store: new FileSessionStore(directory), lock: join(directory, digest + '.json.lock') };
}

it('publishes a complete v2 owner directory before entering its callback', async () => {
  const { store, lock } = await fixture();
  await store.withLock('k', async () => {
    expect((await lstat(lock)).isDirectory()).toBe(true);
    const files = await readdir(lock);
    expect(files).toHaveLength(1);
    expect(files[0]).toMatch(/^owner-[0-9a-f]{32}\.json$/);
    const bytes = await readFile(join(lock, files[0]!), 'utf8');
    expect(bytes.endsWith('\n')).toBe(true);
    const record = JSON.parse(bytes);
    expect(Object.keys(record).sort()).toEqual(['expires_at', 'identity', 'nonce', 'pid', 'released', 'version']);
    expect(record.version).toBe(2); expect(record.pid).toBe(process.pid); expect(record.released).toBe(false);
    expect(record.nonce).toMatch(/^[0-9a-f]{32}$/);
    expect(files[0]).toBe('owner-' + record.nonce + '.json');
    expect(Number.isInteger(record.expires_at)).toBe(true);
    expect(record.expires_at).toBeLessThan(Date.now() / 1000 + 20);
    expect(record.identity.platform).toBe(process.platform);
    expect(record.identity.boot_id).toBeTypeOf('string');
    expect(record.identity.pid_namespace).toBeTypeOf('string');
  });
});

it('refuses an expired legacy lock with the stable maintenance diagnostic', async () => {
  const { store, lock } = await fixture();
  await writeFile(lock, '1\n', 'utf8');
  let ran = 0;
  await expect(store.withLock('k', () => { ran++; }, 10, 0)).rejects.toMatchObject({ code: 'session_locked', message: expect.stringContaining('legacy_lock_requires_maintenance') });
  expect(ran).toBe(0);
  await expect(store.withLock('k', () => { ran++; }, 10, 0)).rejects.toMatchObject({ message: expect.stringContaining(lock) });
  await expect(store.withLock('k', () => { ran++; }, 10, 0)).rejects.toMatchObject({ message: expect.stringContaining('stop_old_workers_then_inspect_and_archive') });
  expect(await readFile(lock, 'utf8')).toBe('1\n');
});

it('retains the legacy diagnostic when native identity is unavailable',async()=>{
 const{store,lock}=await fixture();await writeFile(lock,'1\n');
 const identity=vi.spyOn(identities,'nativeLockIdentity').mockReturnValue(null);
 try{
  await expect(store.withLock('k',()=>{throw Error('legacy callback');},10,0)).rejects.toMatchObject({code:'session_locked',message:expect.stringContaining('legacy_lock_requires_maintenance')});
  expect(identity).not.toHaveBeenCalled();
 }finally{identity.mockRestore();}
 expect(await readFile(lock,'utf8')).toBe('1\n');
});

it('propagates callback filesystem errors without interpreting them as contention', async () => {
  const { store } = await fixture();
  const original = Object.assign(new Error('callback failure'), { code: 'EEXIST' });
  let callbacks = 0;
  await expect(store.withLock('k', () => { callbacks++; throw original; }, 10, 0.12)).rejects.toBe(original);
  expect(callbacks).toBe(1);
});
