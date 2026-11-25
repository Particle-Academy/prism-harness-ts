import { spawn } from 'node:child_process';
import { resolve } from 'node:path';
import { expect, it } from 'vitest';

// A lock wait on the memory store keeps the process alive until it ends. A
// bare script whose only pending work is that wait must get `session_locked`
// back, not exit mid-await (code 13) because the wait's timer was unref'd.
// And an acquired lock must not hold the process open for the rest of the
// wait. Run against the built package, in a child process, as a script runs.

function run(script: string): Promise<{ code: number | null; stdout: string; stderr: string; ms: number }> {
  const started = Date.now();
  const child = spawn(process.execPath, ['--input-type=module', '-e', script, resolve('dist/index.js')], {
    windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'], timeout: 20000,
  });
  let stdout = '', stderr = '';
  child.stdout.on('data', (chunk) => { stdout += String(chunk); });
  child.stderr.on('data', (chunk) => { stderr += String(chunk); });

  return new Promise((done) => {
    child.on('close', (code) => done({ code, stdout, stderr, ms: Date.now() - started }));
  });
}

const prelude = `
const { pathToFileURL } = await import('node:url');
const { MemorySessionStore } = await import(pathToFileURL(process.argv[1]).href);
const store = new MemorySessionStore();
`;

it('returns session_locked to a script whose only pending work is the lock wait', async () => {
  const result = await run(`${prelude}
// The holder never finishes and schedules nothing.
void store.withLock('k', () => new Promise(() => {}), 10, 5);
try {
  await store.withLock('k', () => 'ran', 10, 0.3);
  console.log('acquired');
} catch (error) {
  console.log(error.code);
}
`);

  expect(result.stderr).toBe('');
  expect(result.stdout.trim()).toBe('session_locked');
  expect(result.code).toBe(0);
});

it('does not hold the process open for the rest of the wait once the lock is acquired', async () => {
  const result = await run(`${prelude}
console.log(await store.withLock('k', () => 'ran', 10, 15));
`);

  expect(result.stdout.trim()).toBe('ran');
  expect(result.code).toBe(0);
  expect(result.ms).toBeLessThan(10000);
}, 25000);
