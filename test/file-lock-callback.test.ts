import { appendFile, mkdtemp, readFile, writeFile, lstat } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { expect, it } from 'vitest';

// The same assertions can execute a hash-verified published package for red proof.
const packageRoot = process.env.PRISM_FILE_LOCK_PACKAGE;
const api: typeof import('../src/index.js') = packageRoot
  ? await import(pathToFileURL(join(packageRoot, 'dist/index.js')).href)
  : await import('../src/index.js');
const { FileSessionStore, Session } = api;

async function fixture() {
  const directory = await mkdtemp(join(tmpdir(), 'prism-lock-callback-'));
  const store = new FileSessionStore(directory);
  const session = new Session({ participant: { type: 'test', id: 'lock-regression' },
    scope: 'lock-regression', ephemeral: store, durable: store });
  const digest = createHash('sha256').update(session.key()).digest('hex').slice(0, 32);
  return { directory, session, lock: join(directory, digest + '.json.lock') };
}

it.each(['EEXIST', 'EPERM', 'EACCES', 'EBUSY'])('propagates callback %s once after a counted write', async code => {
  const { directory, session, lock } = await fixture();
  const counted = join(directory, 'counted.log'), collision = join(directory, 'exists');
  await writeFile(collision, 'existing');
  let callbacks = 0, original: unknown;
  const callback = async () => {
    callbacks++;
    await appendFile(counted, 'effect\n');
    try {
      if (code === 'EEXIST') await writeFile(collision, 'new', { flag: 'wx' });
      else throw Object.assign(new Error('callback failure'), { code });
    } catch (error) { original ??= error; throw error; }
  };
  let failure: unknown;
  try { await session.lock(callback, 10, 0.12); } catch (error) { failure = error; }
  expect(callbacks).toBe(1);
  expect((await readFile(counted, 'utf8')).trim().split('\n')).toHaveLength(1);
  expect(failure).toBe(original);
  expect(failure).toMatchObject({ code });
  await expect(lstat(lock)).rejects.toMatchObject({ code: 'ENOENT' });
});

it('a separate waiter completes its bounded refusal without a host keepalive', async () => {
  const { directory, session, lock } = await fixture();
  let outcome: { code: number | null; signal: string | null; stdout: string; stderr: string } | undefined;
  await session.lock(async () => {
    const child = spawn(process.execPath, [
      resolve('test/fixtures/file-lock-waiter.mjs'),
      join(packageRoot ?? resolve('.'), 'dist/index.js'), directory,
    ], { stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true, timeout: 5000 });
    let stdout = '', stderr = '';
    child.stdout.setEncoding('utf8'); child.stderr.setEncoding('utf8');
    child.stdout.on('data', (data: string) => { stdout += data; });
    child.stderr.on('data', (data: string) => { stderr += data; });
    outcome = await new Promise((done, reject) => {
      child.on('error', reject);
      child.on('close', (code, signal) => { done({ code, signal, stdout, stderr }); });
    });
    // The holder is still inside its callback throughout the child's entire wait.
    expect((await lstat(lock)).isFile()).toBe(true);
  });
  expect(outcome).toBeDefined();
  expect(outcome!.code, outcome!.stderr).toBe(0);
  expect(outcome!.signal).toBeNull();
  expect(JSON.parse(outcome!.stdout)).toMatchObject({ callbacks: 0, code: 'session_locked' });
}, 10_000);

