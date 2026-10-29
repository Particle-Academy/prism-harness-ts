import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { mkdtemp, readFile, readdir, rename } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { FileSessionStore } from '../src/index.js';

vi.mock('node:fs/promises', async (importActual) => {
  const actual = await importActual<typeof import('node:fs/promises')>();
  return { ...actual, rename: vi.fn(actual.rename) };
});
const actual = await vi.importActual<typeof import('node:fs/promises')>('node:fs/promises');
const replace = vi.mocked(rename);
beforeEach(() => { replace.mockImplementation(actual.rename); });
afterEach(() => { replace.mockClear(); vi.unstubAllGlobals(); });

async function heldReader(path: string, sharing: string): Promise<() => Promise<void>> {
  const command = "$ErrorActionPreference='Stop'; $p=[Console]::In.ReadLine(); $r=[IO.File]::Open($p,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::ReadWrite); try { [Console]::Out.WriteLine('ready'); [Console]::Out.Flush(); [Console]::In.ReadLine() | Out-Null } finally { $r.Dispose() }";
  const ordinaryWindows = process.platform === 'win32' && sharing === 'ordinary';
  const child = ordinaryWindows
    ? spawn('powershell.exe', ['-NoLogo', '-NoProfile', '-NonInteractive', '-Command', command], { windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'], timeout: 10000 })
    : spawn(process.execPath, [resolve('test/fixtures/file-state-reader.mjs'), path], { windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'], timeout: 10000 });
  let stdout = '', stderr = '';
  child.stderr.on('data', data => { stderr += data; });
  const exited = new Promise<void>((done, reject) => {
    child.once('error', reject);
    child.once('close', code => code === 0 ? done() : reject(Error(`Reader exit ${String(code)}: ${stderr}`)));
  });
  const ready = new Promise<void>((done, reject) => {
    child.stdout.on('data', data => { stdout += data; if (stdout.includes('ready')) done(); });
    exited.then(() => { if (!stdout.includes('ready')) reject(Error('Reader exited before barrier')); }, reject);
  });
  if (ordinaryWindows) child.stdin.write(path + '\n');
  try { await ready; } catch (error) { child.stdin.end(); await exited.catch(() => undefined); throw error; }
  let closed = false;
  return async () => { if (!closed) { closed = true; child.stdin.end(); } await exited; };
}

for (const sharing of ['ordinary', 'share-delete']) {
  it(`publishes complete state after a native ${sharing} reader closes`, async () => {
    const directory = await mkdtemp(join(tmpdir(), 'prism-state-write-')), store = new FileSessionStore(directory);
    await store.put('k', { revision: 0 });
    const target = join(directory, createHash('sha256').update('k').digest('hex').slice(0, 32) + '.json');
    const close = await heldReader(target, sharing);
    let blocked!: () => void;
    const failedAttempt = new Promise<void>(done => { blocked = done; });
    const errors: unknown[] = [];
    replace.mockImplementation(async (from, to) => {
      try { await actual.rename(from, to); } catch (error) { errors.push(error); blocked(); throw error; }
    });
    const payload = { revision: 1, body: '界'.repeat(70000) };
    const write = store.put('k', payload).then(() => ({ error: undefined }), (error: unknown) => ({ error }));
    try {
      if (process.platform === 'win32') {
        await Promise.race([failedAttempt, delay(3000).then(() => { throw Error('Missing real permission failure'); })]);
        await close();
      }
      expect((await write).error).toBeUndefined();
    } finally { await close(); }
    expect(errors.length > 0).toBe(process.platform === 'win32');
    expect(await store.get('k')).toEqual(payload);
    expect(await readdir(directory)).toEqual([target.split(/[\\/]/).at(-1)]);
  }, 15000);

  it(`preserves the original error, old state and no temp for a permanent ${sharing} reader`, async () => {
    const directory = await mkdtemp(join(tmpdir(), 'prism-state-write-')), store = new FileSessionStore(directory);
    await store.put('k', { revision: 0 });
    const target = join(directory, createHash('sha256').update('k').digest('hex').slice(0, 32) + '.json');
    const originalBytes = await readFile(target), close = await heldReader(target, sharing), errors: unknown[] = [];
    replace.mockImplementation(async (from, to) => {
      try { await actual.rename(from, to); } catch (error) { errors.push(error); throw error; }
    });
    const started = performance.now();
    let failure: unknown;
    try { await store.put('k', { revision: 1 }); } catch (error) { failure = error; }
    try {
      if (process.platform === 'win32') {
        expect(performance.now() - started).toBeGreaterThanOrEqual(700);
        expect(performance.now() - started).toBeLessThan(2000);
        expect(errors.length).toBeGreaterThan(1);
        expect(failure).toBe(errors[0]);
        expect(failure).toMatchObject({ code: expect.stringMatching(/^(EPERM|EACCES)$/) });
        expect(await readFile(target)).toEqual(originalBytes);
      } else { expect(failure).toBeUndefined(); expect(errors).toEqual([]); }
      expect(await readdir(directory)).toEqual([target.split(/[\\/]/).at(-1)]);
    } finally { await close(); }
    expect(await store.get('k')).toEqual({ revision: process.platform === 'win32' ? 0 : 1 });
  }, 15000);
}

it.each(['EIO', 'ENOENT', 'EBUSY'])('does not retry Windows %s errors or replace their identity', async code => {
  const directory = await mkdtemp(join(tmpdir(), 'prism-state-error-')), store = new FileSessionStore(directory);
  await store.put('k', { revision: 0 });
  const failure = Object.assign(new Error('synthetic IO'), { code });
  vi.stubGlobal('process', { ...process, platform: 'win32' });
  replace.mockClear().mockRejectedValue(failure);
  await expect(store.put('k', { revision: 1 })).rejects.toBe(failure);
  expect(replace).toHaveBeenCalledTimes(1);
  expect((await readdir(directory)).filter(name => name.endsWith('.tmp'))).toEqual([]);
});

it('keeps an otherwise idle child alive until the awaited write settles', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'prism-state-idle-')), store = new FileSessionStore(directory);
  await store.put('k', { revision: 0 });
  const target = join(directory, createHash('sha256').update('k').digest('hex').slice(0, 32) + '.json');
  const originalBytes = await readFile(target), close = await heldReader(target, 'share-delete');
  const script = "import{pathToFileURL}from'node:url';const{FileSessionStore}=await import(pathToFileURL(process.argv[1]).href);const start=performance.now();let code=null;try{await new FileSessionStore(process.argv[2]).put('k',{revision:1});}catch(error){code=error.code;}console.log(JSON.stringify({code,elapsedMs:performance.now()-start}));";
  try {
    const child = spawn(process.execPath, ['--input-type=module', '-e', script, resolve('dist/index.js'), directory], { windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'], timeout: 5000 });
    let stdout = '', stderr = '';
    child.stdout.on('data', data => { stdout += data; });
    child.stderr.on('data', data => { stderr += data; });
    const exit = await new Promise<number | null>((done, reject) => { child.once('error', reject); child.once('close', done); });
    expect(exit, stderr).toBe(0);
    const result = JSON.parse(stdout);
    if (process.platform === 'win32') {
      expect(result.code).toMatch(/^(EPERM|EACCES)$/);
      expect(result.elapsedMs).toBeGreaterThanOrEqual(700);
      expect(result.elapsedMs).toBeLessThan(2000);
      expect(await readFile(target)).toEqual(originalBytes);
    } else expect(result.code).toBeNull();
    expect((await readdir(directory)).filter(name => name.endsWith('.tmp'))).toEqual([]);
  } finally { await close(); }
});

it.each(['EPERM', 'EACCES'])('does not retry POSIX %s errors', async code => {
  const directory = await mkdtemp(join(tmpdir(), 'prism-state-error-')), store = new FileSessionStore(directory);
  await store.put('k', { revision: 0 });
  const failure = Object.assign(new Error('synthetic POSIX permission'), { code });
  vi.stubGlobal('process', { ...process, platform: 'linux' });
  replace.mockClear().mockRejectedValue(failure);
  await expect(store.put('k', { revision: 1 })).rejects.toBe(failure);
  expect(replace).toHaveBeenCalledTimes(1);
  expect((await readdir(directory)).filter(name => name.endsWith('.tmp'))).toEqual([]);
});
