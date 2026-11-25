import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, readdir, readFile, unlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, expect, it, vi } from 'vitest';
import { FileSessionStore } from '../src/index.js';

// A released owner record is reclaimed by deleting it, and on Windows another
// process (an antivirus scan, the indexer) can hold the file it just wrote for
// a moment: the delete fails with EPERM. The lock reads that as "held" and
// tries again until its wait runs out. That is a wait that is too short, not
// a race: the record is never taken while the delete fails, the callback
// never runs early, and a wait long enough for the handle to close reclaims
// it. Made deterministic here by failing the delete a fixed number of times.

vi.mock('node:fs/promises', async (importActual) => {
  const actual = await importActual<typeof import('node:fs/promises')>();
  return { ...actual, unlink: vi.fn(actual.unlink) };
});
const actual = await vi.importActual<typeof import('node:fs/promises')>('node:fs/promises');
const remove = vi.mocked(unlink);

afterEach(() => { remove.mockImplementation(actual.unlink); });

/** Fail the next `count` deletes of a lock owner record with EPERM, as a held Windows handle does. */
function heldHandle(count: number): { left: () => number } {
  let left = count;
  remove.mockImplementation(async (path) => {
    // The published lock's record only, never the caller's own candidate.
    if (left > 0 && /\.json\.lock[\\/]owner-[0-9a-f]{32}\.json$/.test(String(path))) {
      left -= 1;
      throw Object.assign(new Error('operation not permitted'), { code: 'EPERM' });
    }
    return actual.unlink(path);
  });

  return { left: () => left };
}

it('reclaims a released record only once a transiently held delete succeeds, never before', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'prism-harness-transient-'));
  const store = new FileSessionStore(directory);
  const path = join(directory, `${createHash('sha256').update('k').digest('hex').slice(0, 32)}.json.lock`);
  let filename = '', bytes = Buffer.alloc(0);
  await store.withLock('k', async () => {
    [filename] = await readdir(path);
    bytes = await readFile(join(path, filename!));
  });
  // The same owner record, explicitly released, back in place.
  await mkdir(path);
  await writeFile(join(path, filename), JSON.stringify({ ...JSON.parse(bytes.toString('utf8')), released: true }) + '\n');

  const held = heldHandle(30);
  let ran = 0;
  // No wait: the delete fails, so the record is not taken and nothing runs.
  await expect(store.withLock('k', () => { ran += 1; }, 10, 0)).rejects.toMatchObject({ code: 'session_locked' });
  expect(ran).toBe(0);
  expect(await readdir(path)).toEqual([filename]);
  expect(held.left()).toBeGreaterThan(0);

  // A wait that outlasts the held handle reclaims it, once.
  expect(await store.withLock('k', () => { ran += 1; return 7; }, 10, 10)).toBe(7);
  expect(ran).toBe(1);
  expect(held.left()).toBe(0);
});
