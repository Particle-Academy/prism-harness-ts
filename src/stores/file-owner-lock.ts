import { randomBytes } from 'node:crypto';
import { lstat, mkdir, readdir, readFile, rename, rmdir, unlink, writeFile } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import { HarnessError } from '../errors.js';
import { nativeLockIdentity, sameLockIdentity, lockPidIsDead } from './file-lock-identity.js';
import type { LockIdentity } from './file-lock-identity.js';

export interface FileLockOwner {
  version: 2;
  pid: number;
  identity: LockIdentity;
  nonce: string;
  released: boolean;
  expires_at: number;
}
interface ReleaseProgress {
  lockPath: string;
  owner: FileLockOwner;
  unlinked: boolean;
  flushing?: Promise<void>;
}
// Failed release is observable, then retried by this process's next lock call.
// Key by nonce: old and replacement owners of one path must never overwrite
// each other's pending state. No background timer keeps an idle worker alive.
const pendingReleases = new Map<string, ReleaseProgress>();
export interface PendingFileLockRelease {
  readonly key_digest: string;
  readonly nonce: string;
}
/** Snapshot only: no filesystem access, retries, raw keys, paths or PIDs. */
export function pendingFileOwnerReleases(directory: string): readonly PendingFileLockRelease[] {
  const parent = resolve(directory);
  const rows = [...pendingReleases.values()].filter(progress=>dirname(progress.lockPath)===parent)
    .map(progress=>Object.freeze({key_digest:basename(progress.lockPath,'.json.lock'),nonce:progress.owner.nonce}));
  rows.sort((a,b)=>a.key_digest===b.key_digest?(a.nonce<b.nonce?-1:a.nonce>b.nonce?1:0):(a.key_digest<b.key_digest?-1:1));
  return Object.freeze(rows);
}
const OWNER = /^owner-([0-9a-f]{32})\.json$/;
const HELD = new Set(['EEXIST', 'ENOTEMPTY', 'EISDIR', 'ENOTDIR', 'EPERM', 'EACCES', 'EBUSY']);
function code(error: unknown): string | undefined { return (error as NodeJS.ErrnoException | null)?.code; }
function sleep(ms: number): Promise<void> { return new Promise(done => { setTimeout(done, ms); }); }
function object(value: unknown): value is Record<string, unknown> { return typeof value === 'object' && value !== null && !Array.isArray(value); }
function keys(value: Record<string, unknown>, expected: string[]): boolean { return Object.keys(value).sort().join(',') === expected.sort().join(','); }
function bounded(value: unknown): value is string { return typeof value === 'string' && value.length > 0 && Buffer.byteLength(value, 'utf8') <= 256 && Buffer.from(value, 'utf8').toString('utf8') === value; }

/** Closed, bounded record; duplicate keys and fractional/exponent number tokens refuse. */
export function parseFileLockOwner(raw: Buffer, filename: string): FileLockOwner | null {
  if (raw.length > 4096 || raw.length === 0 || raw.at(-1) !== 10) return null;
  try {
    const text = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(raw);
    const value: unknown = JSON.parse(text);
    const tokens = text.match(/"(?:\\.|[^"\\])*"|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|[{}\[\]:,]|true|false|null/g) ?? [];
    const seen = new Set<string>();
    for (let i = 0; i < tokens.length; i++) {
      const token = tokens[i]!;
      if (token.startsWith('"') && tokens[i + 1] === ':') {
        const key = JSON.parse(token) as string;
        if (seen.has(key)) return null;
        seen.add(key);
      } else if (/^-?\d/.test(token) && (!/^-?\d+$/.test(token) || !Number.isSafeInteger(Number(token)))) return null;
    }
    const match = OWNER.exec(filename);
    if (!match || !object(value) || !keys(value, ['version','pid','identity','nonce','released','expires_at'])) return null;
    if (value.version !== 2 || typeof value.pid !== 'number' || !Number.isInteger(value.pid) || value.pid < 1 || value.pid > 4294967295 ||
        typeof value.released !== 'boolean' || typeof value.expires_at !== 'number' || !Number.isSafeInteger(value.expires_at) || value.nonce !== match[1]) return null;
    const id = value.identity;
    if (!object(id) || !keys(id, ['platform','boot_id','pid_namespace']) ||
        typeof id.platform !== 'string' || !['linux','darwin','win32'].includes(id.platform) || !bounded(id.boot_id) ||
        !(id.pid_namespace === null || bounded(id.pid_namespace))) return null;
    return value as unknown as FileLockOwner;
  } catch { return null; }
}
function filename(owner: FileLockOwner): string { return 'owner-' + owner.nonce + '.json'; }
function payload(owner: FileLockOwner): string { return JSON.stringify(owner) + '\n'; }

/** Unlink the observed nonce first. Missing old filename NEVER authorizes rmdir. */
export async function cleanFileLockOwner(lockPath: string, owner: FileLockOwner, temporary = false, progress?: ReleaseProgress): Promise<void> {
  if (temporary) await unlink(join(lockPath, 'owner-' + owner.nonce + '.release.tmp')).catch(error => { if (code(error) !== 'ENOENT') throw error; });
  if (!progress?.unlinked) for (let attempt = 0; ; attempt++) {
    try { await unlink(join(lockPath, filename(owner))); if (progress) progress.unlinked = true; break; }
    catch (error) {
      if (code(error) === 'ENOENT') return;
      if (attempt >= 4 || !HELD.has(code(error) ?? '')) throw error;
      await sleep(5);
    }
  }
  for (let attempt = 0; ; attempt++) {
    try { await rmdir(lockPath); return; }
    catch (error) {
      // A replacement is populated, or somebody already removed the empty dir.
      if (['ENOENT','ENOTEMPTY','EEXIST'].includes(code(error) ?? '')) return;
      if (attempt >= 4 || !HELD.has(code(error) ?? '')) throw error;
      await sleep(5);
    }
  }
}

async function inspect(lockPath: string, local: LockIdentity): Promise<'legacy' | 'held' | 'retry'> {
  let owner: FileLockOwner | null, temporary: boolean;
  try {
    const info = await lstat(lockPath);
    if (info.isSymbolicLink()) return 'held';
    if (info.isFile()) return 'legacy';
    if (!info.isDirectory()) return 'held';
    const entries = await readdir(lockPath);
    const names = entries.filter(name => OWNER.test(name));
    if (names.length !== 1) return 'held';
    const name = names[0]!, release = name.replace('.json', '.release.tmp');
    if (entries.some(entry => entry !== name && entry !== release)) return 'held';
    temporary = entries.includes(release);
    for (const entry of entries) {
      const stat = await lstat(join(lockPath, entry));
      if (!stat.isFile() || stat.isSymbolicLink() || stat.size > 4096) return 'held';
    }
    owner = parseFileLockOwner(await readFile(join(lockPath, name)), name);
    // Entire established identity must match BEFORE the first liveness probe.
    if (!owner || !sameLockIdentity(owner.identity, local)) return 'held';
    if (!owner.released && !lockPidIsDead(owner.pid)) return 'held';
  } catch (error) {
    if (code(error) === 'ENOENT') return 'retry';
    if (HELD.has(code(error) ?? '')) return 'held';
    throw error;
  }
  await cleanFileLockOwner(lockPath, owner, temporary);
  return 'retry';
}

async function release(progress: ReleaseProgress): Promise<void> {
  const {lockPath,owner} = progress;
  if (progress.unlinked) { await cleanFileLockOwner(lockPath,owner,false,progress); return; }
  const name = filename(owner);
  try {
    const observed = parseFileLockOwner(await readFile(join(lockPath, name)), name);
    if (!observed || observed.nonce !== owner.nonce) throw new Error('File lock owner changed before release');
    if (observed.released) { await cleanFileLockOwner(lockPath,owner,true,progress); return; }
  } catch (error) { if (code(error) === 'ENOENT') return; throw error; }
  const temporary = join(lockPath, 'owner-' + owner.nonce + '.release.tmp');
  // The callback has finished; close the full release record before publishing it.
  // While this exact live owner is still unreleased, nobody may reclaim it.
  // Retrying its own known nonce temporary can safely replace a partial write.
  await writeFile(temporary, payload({ ...owner, released: true }), { encoding: 'utf8', flag: 'w' });
  for (let attempt = 0; ; attempt++) {
    try { await rename(temporary, join(lockPath, name)); break; }
    catch (error) {
      if (attempt >= 4 || !HELD.has(code(error) ?? '')) throw error;
      await sleep(5);
    }
  }
  await cleanFileLockOwner(lockPath, owner, false, progress);
}

async function flushPendingReleases(): Promise<void> {
  for (const progress of [...pendingReleases.values()]) {
    if (pendingReleases.get(progress.owner.nonce) !== progress) continue;
    if (!progress.flushing) progress.flushing = release(progress);
    const operation = progress.flushing;
    try {
      await operation;
      if (pendingReleases.get(progress.owner.nonce) === progress) pendingReleases.delete(progress.owner.nonce);
    } finally { if (progress.flushing === operation) progress.flushing = undefined; }
  }
}

/** Decision 0022. Acquisition retry never contains callback execution. */
export async function withFileOwnerLock<T>(path: string, key: string, callback: () => T | Promise<T>, ttl: number, wait: number): Promise<T> {
  if (!Number.isFinite(ttl) || !Number.isFinite(wait) || wait < 0 || !Number.isSafeInteger(Math.floor(Date.now()/1000 + ttl))) throw new RangeError('Invalid file lock duration');
  await flushPendingReleases();
  const lockPath = resolve(path);
  try {
    if ((await lstat(lockPath)).isFile()) throw new HarnessError('session_locked', 'legacy_lock_requires_maintenance: ' + lockPath + '; stop_old_workers_then_inspect_and_archive');
  } catch (error) { if (code(error) !== 'ENOENT') throw error; }
  const identity = nativeLockIdentity();
  if (!identity) throw new HarnessError('session_locked', 'file_lock_identity_unavailable: ' + lockPath);
  const owner: FileLockOwner = { version: 2, pid: process.pid, identity, nonce: randomBytes(16).toString('hex'), released: false, expires_at: Math.floor(Date.now()/1000 + ttl) };
  const candidate = lockPath + '.candidate-' + owner.nonce;
  if (dirname(candidate) !== dirname(lockPath) || basename(candidate) === basename(lockPath)) throw new Error('Lock publication must stay inside its parent');
  const deadline = performance.now() + wait * 1000;
  await mkdir(candidate);
  let published = false;
  try {
    await writeFile(join(candidate, filename(owner)), payload(owner), { encoding:'utf8', flag:'wx' });
    for (;;) {
      if (pendingReleases.size) await flushPendingReleases();
      // Node/Windows rename can replace a regular file. Refuse stale legacy
      // locks before publication; stopping ALL legacy workers is mandatory.
      try {
        const existing = await lstat(lockPath);
        if (existing.isFile()) throw new HarnessError('session_locked', 'legacy_lock_requires_maintenance: ' + lockPath + '; stop_old_workers_then_inspect_and_archive');
        if (existing.isSymbolicLink() || !existing.isDirectory()) throw HarnessError.sessionLocked(key,wait);
      } catch (error) { if (code(error) !== 'ENOENT') throw error; }
      try { await rename(candidate, lockPath); published = true; break; }
      catch (error) { if (!HELD.has(code(error) ?? '')) throw error; }
      const state = await inspect(lockPath, identity);
      if (state === 'legacy') throw new HarnessError('session_locked', 'legacy_lock_requires_maintenance: ' + lockPath + '; stop_old_workers_then_inspect_and_archive');
      if (state === 'retry' && performance.now() < deadline) continue;
      if (performance.now() >= deadline) throw HarnessError.sessionLocked(key, wait);
      await sleep(25);
    }
  } finally {
    if (!published) await cleanFileLockOwner(candidate, owner);
  }
  // Exactly one execution, with exact original error identity on normal cleanup.
  try { return await callback(); }
  finally {
    const progress: ReleaseProgress = {lockPath,owner,unlinked:false};
    try { await release(progress); }
    catch (error) {
      pendingReleases.set(owner.nonce,progress);
      process.emitWarning(JSON.stringify({key_digest:basename(lockPath,'.json.lock'),nonce:owner.nonce}), {code:'file_lock_release_pending'});
      throw error;
    }
  }
}
