import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import type { LockIdentity } from '../src/stores/file-lock-identity.js';

const reader = vi.hoisted(() => vi.fn<() => LockIdentity | null>());
vi.mock('../src/stores/file-lock-windows-identity.js', () => ({ windowsLockIdentity: reader }));
const originalPlatform = Object.getOwnPropertyDescriptor(process, 'platform')!;

beforeEach(() => {
  vi.resetModules();
  reader.mockReset();
  Object.defineProperty(process, 'platform', { ...originalPlatform, value: 'win32' });
});
afterEach(() => { vi.restoreAllMocks(); Object.defineProperty(process, 'platform', originalPlatform); });

const host: LockIdentity = { platform: 'win32', boot_id: 'cim-utc-ticks:638000000000000000', pid_namespace: 'win32:kernel' };
it.each([null, { ...host, pid_namespace: null }, host])('retains the first identity within the retry interval: %j', async first => {
  reader.mockReturnValueOnce(first).mockReturnValue(host);
  const { nativeLockIdentity } = await import('../src/stores/file-lock-identity.js');
  const initial = nativeLockIdentity();
  expect(initial).toEqual(first);
  expect(nativeLockIdentity()).toBe(initial);
  expect(reader).toHaveBeenCalledTimes(1);
  if (initial !== null) expect(Object.isFrozen(initial)).toBe(true);
});

it('re-samples transient failure after sixty seconds and then caches definitive success', async () => {
  let now=0; vi.spyOn(performance,'now').mockImplementation(()=>now);
  reader.mockReturnValueOnce(null).mockReturnValue(host);
  const {nativeLockIdentity,fileLockIdentityStatus}=await import('../src/stores/file-lock-identity.js');
  expect(fileLockIdentityStatus().sampled).toBe(false); expect(reader).not.toHaveBeenCalled();
  expect(nativeLockIdentity()).toBeNull(); now=59999;
  expect(nativeLockIdentity()).toBeNull(); expect(reader).toHaveBeenCalledTimes(1);
  expect(fileLockIdentityStatus()).toMatchObject({definitive:false,code:'file_lock_identity_unavailable',retry_after_ms:1});
  now=60000;expect(nativeLockIdentity()).toEqual(host);expect(reader).toHaveBeenCalledTimes(2);
  now=99999999;expect(nativeLockIdentity()).toEqual(host);expect(reader).toHaveBeenCalledTimes(2);
  expect(fileLockIdentityStatus()).toMatchObject({definitive:true,recovery_available:true,retry_after_ms:null,code:null});
  expect(Object.isFrozen(fileLockIdentityStatus())).toBe(true);
});

it('positive container detection is definitive even though it cannot authorize recovery', async () => {
  let now=0;vi.spyOn(performance,'now').mockImplementation(()=>now);
  reader.mockReturnValueOnce({...host,pid_namespace:null}).mockReturnValue(host);
  const {nativeLockIdentity,fileLockIdentityStatus}=await import('../src/stores/file-lock-identity.js');
  expect(nativeLockIdentity()?.pid_namespace).toBeNull();now=99999999;
  expect(nativeLockIdentity()?.pid_namespace).toBeNull();expect(reader).toHaveBeenCalledTimes(1);
  expect(fileLockIdentityStatus()).toMatchObject({definitive:true,recovery_available:false,retry_after_ms:null});
});

it('retains a failed sample instead of retrying discovery during acquisition', async () => {
  reader.mockImplementationOnce(() => { throw new Error('owned unavailable reader'); }).mockReturnValue(host);
  const { nativeLockIdentity } = await import('../src/stores/file-lock-identity.js');
  expect(nativeLockIdentity()).toBeNull();
  expect(nativeLockIdentity()).toBeNull();
  expect(reader).toHaveBeenCalledTimes(1);
});
