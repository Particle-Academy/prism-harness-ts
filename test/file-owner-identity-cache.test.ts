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
afterEach(() => Object.defineProperty(process, 'platform', originalPlatform));

const host: LockIdentity = { platform: 'win32', boot_id: 'cim-utc-ticks:638000000000000000', pid_namespace: 'win32:kernel' };
it.each([null, { ...host, pid_namespace: null }, host])('keeps the first identity result, including unknown: %j', async first => {
  reader.mockReturnValueOnce(first).mockReturnValue(host);
  const { nativeLockIdentity } = await import('../src/stores/file-lock-identity.js');
  const initial = nativeLockIdentity();
  expect(initial).toEqual(first);
  expect(nativeLockIdentity()).toBe(initial);
  expect(reader).toHaveBeenCalledTimes(1);
  if (initial !== null) expect(Object.isFrozen(initial)).toBe(true);
});

it('retains a failed sample instead of retrying discovery during acquisition', async () => {
  reader.mockImplementationOnce(() => { throw new Error('owned unavailable reader'); }).mockReturnValue(host);
  const { nativeLockIdentity } = await import('../src/stores/file-lock-identity.js');
  expect(nativeLockIdentity()).toBeNull();
  expect(nativeLockIdentity()).toBeNull();
  expect(reader).toHaveBeenCalledTimes(1);
});
