import { randomBytes } from 'node:crypto';
import { mkdir, lstat, readdir, readFile, symlink, writeFile, rmdir, unlink } from 'node:fs/promises';
import { expect, it, vi } from 'vitest';
import { FileSessionStore } from '../src/index.js';
import { parseFileLockOwner, cleanFileLockOwner } from '../src/stores/file-owner-lock.js';
import type { FileLockOwner } from '../src/stores/file-owner-lock.js';
import { nativeLockIdentity } from '../src/stores/file-lock-identity.js';
import { mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createHash } from 'node:crypto';

function owner(): FileLockOwner {
  const identity = nativeLockIdentity();
  expect(identity?.pid_namespace).toBeTypeOf('string');
  return { version: 2, pid: process.pid, identity: identity!, nonce: randomBytes(16).toString('hex'), released: false, expires_at: 0 };
}
const name = (value: FileLockOwner) => 'owner-' + value.nonce + '.json';
async function fixture() {
  const directory = await mkdtemp(join(tmpdir(), 'prism-owner-vectors-'));
  return { store: new FileSessionStore(directory), lock: join(directory, createHash('sha256').update('k').digest('hex').slice(0,32) + '.json.lock') };
}
async function plant(lock: string, value: FileLockOwner) {
  await mkdir(lock);
  await writeFile(join(lock, name(value)), JSON.stringify(value) + '\n', 'utf8');
}
const raw = (value: unknown) => Buffer.from(JSON.stringify(value) + '\n');

it('refuses a canonical symlink or Windows junction without touching its target',async()=>{
 const{store,lock}=await fixture(),target=lock+'.target';
 await mkdir(target);await writeFile(join(target,'sentinel'),'keep');
 await symlink(target,lock,'junction');
 await expect(store.withLock('k',()=>{throw Error('unknown canonical alias');},10,0)).rejects.toMatchObject({code:'session_locked'});
 expect((await lstat(lock)).isSymbolicLink()).toBe(true);
 expect(await readFile(join(target,'sentinel'),'utf8')).toBe('keep');
});

it('parses a native complete writer record but refuses closed-schema violations', () => {
  const valid = owner(), filename = name(valid);
  expect(parseFileLockOwner(raw(valid), filename)).toEqual(valid);
  const bad: unknown[] = [
    {...valid, pid: true}, {...valid, pid: 0}, {...valid, pid: 4294967296},
    {...valid, version: true}, {...valid, released: 0}, {...valid, expires_at: 0.1},
    {...valid, unknown: true}, {...valid, identity: null},
    {...valid, identity: {...valid.identity, unknown: 1}},
    {...valid, identity: {...valid.identity, platform: ['linux']}},
    {...valid, identity: {...valid.identity, boot_id: ''}},
    {...valid, identity: {...valid.identity, boot_id: '\ud800'}},
    {...valid, identity: {...valid.identity, boot_id: 'é'.repeat(129)}},
    {...valid, identity: {...valid.identity, pid_namespace: ''}},
    {...valid, nonce: 'a'.repeat(32)},
  ];
  for (const value of bad) expect(parseFileLockOwner(raw(value), filename)).toBeNull();
  for (const bytes of [
    raw(valid).subarray(0, -1), Buffer.from('{"version":2,"pid":\n'),
    Buffer.from(JSON.stringify(valid).replace('"version":2', '"version":2,"version":2') + '\n'),
    Buffer.from(JSON.stringify(valid).replace('"version":2', '"version":2.0') + '\n'),
    Buffer.concat([Buffer.from([0xef,0xbb,0xbf]), raw(valid)]),
    Buffer.concat([Buffer.from([0xff]),raw(valid)]), Buffer.alloc(4097,32),
  ]) expect(parseFileLockOwner(bytes, filename)).toBeNull();
});

it.each(['boot','realm','platform','foreign-released','unknown'])('holds %s identity without a PID probe', async kind => {
  const {store,lock} = await fixture(), value = owner();
  if (kind === 'boot') value.identity = {...value.identity, boot_id:'different-boot'};
  if (kind === 'realm' || kind === 'foreign-released') value.identity = {...value.identity, pid_namespace:'foreign-realm'};
  if (kind === 'platform') value.identity = {...value.identity, platform:process.platform === 'win32' ? 'linux' : 'win32'};
  if (kind === 'foreign-released') value.released = true;
  if (kind === 'unknown') { value.identity = {...value.identity,pid_namespace:null}; value.released = true; }
  await plant(lock,value);
  const before=await readFile(join(lock,name(value)));
  const probe=vi.spyOn(process,'kill');
  try {
    await expect(store.withLock('k',()=>{throw Error('must not enter');},10,0)).rejects.toMatchObject({code:'session_locked'});
    expect(probe).not.toHaveBeenCalled();
    expect(await readFile(join(lock,name(value)))).toEqual(before);
  } finally {probe.mockRestore();}
});

it('holds a live owner after its old TTL and reclaims an explicitly released owner', async () => {
  const {store,lock} = await fixture(), value=owner(); await plant(lock,value);
  await expect(store.withLock('k',()=>1,10,0)).rejects.toMatchObject({code:'session_locked'});
  value.released=true; await writeFile(join(lock,name(value)),JSON.stringify(value)+'\n');
  expect(await store.withLock('k',()=>7,10,0.1)).toBe(7);
  await expect(lstat(lock)).rejects.toMatchObject({code:'ENOENT'});
});

it('unknown liveness holds a matching owner and keeps bytes intact', async () => {
  const {store,lock}=await fixture(), value=owner(); await plant(lock,value);
  const probe=vi.spyOn(process,'kill').mockImplementation(()=>{throw Object.assign(Error('denied'),{code:'EPERM'});});
  try {await expect(store.withLock('k',()=>1,10,0)).rejects.toMatchObject({code:'session_locked'});}
  finally {probe.mockRestore();}
  expect(await readFile(join(lock,name(value)),'utf8')).toBe(JSON.stringify(value)+'\n');
});

it('an unexpected entry prevents even released-owner cleanup', async () => {
  const {store,lock}=await fixture(), value=owner(); value.released=true; await plant(lock,value);
  await writeFile(join(lock,'unexpected'),'keep');
  await expect(store.withLock('k',()=>1,10,0)).rejects.toMatchObject({code:'session_locked'});
  expect((await readdir(lock)).sort()).toEqual([name(value),'unexpected'].sort());
});

it('a stale observer missing its old nonce cannot remove an empty canonical directory', async () => {
  const {lock}=await fixture(), value=owner(); await mkdir(lock);
  await cleanFileLockOwner(lock,value);
  expect((await lstat(lock)).isDirectory()).toBe(true);
});

it('a stale cleanup cannot remove a populated replacement', async () => {
  const {lock}=await fixture(), old=owner(), replacement=owner(); await plant(lock,old);
  await unlink(join(lock,name(old))); await rmdir(lock); await plant(lock,replacement);
  await cleanFileLockOwner(lock,old);
  expect(await readFile(join(lock,name(replacement)),'utf8')).toBe(JSON.stringify(replacement)+'\n');
});

it('executes native empty-directory publication semantics', async () => {
  const {store,lock}=await fixture(); await mkdir(lock);
  if(process.platform==='win32') {
    await expect(store.withLock('k',()=>1,10,0)).rejects.toMatchObject({code:'session_locked'});
    expect(await readdir(lock)).toEqual([]);
  } else expect(await store.withLock('k',()=>1,10,0)).toBe(1);
});
