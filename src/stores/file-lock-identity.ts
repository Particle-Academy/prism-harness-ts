import { readFileSync, readlinkSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { windowsLockIdentity } from './file-lock-windows-identity.js';

export interface LockIdentity {
  readonly platform: 'linux' | 'darwin' | 'win32';
  readonly boot_id: string;
  readonly pid_namespace: string | null;
}
let sampled = false;
let established: LockIdentity | null = null;
let retryAt = 0;
let definitive = false;
let durationMs = 0;
export interface FileLockIdentityStatus {
  readonly sampled: boolean;
  readonly recovery_available: boolean;
  readonly definitive: boolean;
  readonly retry_after_ms: number | null;
  readonly sample_duration_ms: number;
  readonly code: 'file_lock_identity_unavailable' | null;
}
/** Process metadata snapshot only; never runs a query or performs I/O. */
export function fileLockIdentityStatus(): FileLockIdentityStatus {
  return Object.freeze({sampled,recovery_available:established?.pid_namespace != null,definitive,
    retry_after_ms:!sampled||definitive?null:Math.max(0,retryAt-performance.now()),
    sample_duration_ms:durationMs,code:sampled&&established?.pid_namespace==null?'file_lock_identity_unavailable':null});
}
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/** Definitive metadata is permanent; transient failure is sampled at most once per minute. */
export function nativeLockIdentity(): LockIdentity | null {
  if (!sampled || (!definitive && performance.now() >= retryAt)) {
    const start = performance.now();
    const identity = readNativeLockIdentity();
    durationMs = performance.now()-start;
    established = identity === null ? null : Object.freeze(identity);
    definitive = identity !== null && (identity.pid_namespace !== null || identity.platform === 'win32');
    retryAt = performance.now()+60000;
    sampled = true;
  }
  return established;
}

function readNativeLockIdentity(): LockIdentity | null {
  let identity: LockIdentity | null = null;
  try {
    if (process.platform === 'win32') identity = windowsLockIdentity();
    else if (process.platform === 'linux') {
      const boot = readFileSync('/proc/sys/kernel/random/boot_id', 'utf8').trim();
      if (!UUID.test(boot)) return null;
      let realm: string | null = null;
      try { const value = readlinkSync('/proc/self/ns/pid'); if (/^pid:\[[0-9]+\]$/.test(value)) realm = value; } catch { /* unknown */ }
      identity = { platform: 'linux', boot_id: boot.toLowerCase(), pid_namespace: realm };
    } else if (process.platform === 'darwin') {
      const result = spawnSync('/usr/sbin/sysctl', ['-n', 'kern.bootsessionuuid'],
        { encoding: 'utf8', timeout: 15000, maxBuffer: 65536 });
      const boot = result.stdout?.trim();
      if (result.error || result.status !== 0 || !boot || !UUID.test(boot)) return null;
      identity = { platform: 'darwin', boot_id: boot.toLowerCase(), pid_namespace: 'darwin:kernel' };
    }
  } catch { return null; }
  return identity;
}
export function sameLockIdentity(owner: LockIdentity, local: LockIdentity): boolean {
  return owner.pid_namespace !== null && local.pid_namespace !== null &&
    owner.platform === local.platform && owner.boot_id === local.boot_id &&
    owner.pid_namespace === local.pid_namespace;
}

/** Read-only signal-zero probe. Permission errors and reused live PIDs remain held. */
export function lockPidIsDead(pid: number): boolean {
  try { process.kill(pid, 0); return false; }
  catch (error) { return (error as NodeJS.ErrnoException).code === 'ESRCH'; }
}
