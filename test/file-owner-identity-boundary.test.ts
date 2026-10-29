import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { windowsLockIdentity } from '../src/stores/file-lock-windows-identity.js';

const invoke = vi.hoisted(() => vi.fn());
vi.mock('node:child_process', () => ({ spawnSync: invoke }));

function guard(source: string): void {
  source = source.replace(/^\s*\/\/.*$/gm, '');
  const imports = [...source.matchAll(/from ['"]([^'"]+)['"]/g)].map(value => value[1]).sort();
  assert.deepEqual(imports, ['./file-lock-identity.js', 'node:child_process', 'node:path']);
  assert(!/\b(?:require|import)\s*\(|process\.binding|\b(?:Add-Type|DllImport|IsProcessInJob|QueryInformationJobObject|CreateJobObject|OpenJobObject|Silo|NtQueryInformationProcess)\b/i.test(source));
  assert(!/GetValue\s*\(|-EncodedCommand|\$\{/.test(source));
  assert.match(source, /PSModuleAutoloadingPreference = 'None'/);
  assert.match(source, /Import-Module -Name \(\$PSHOME \+ '\\Modules\\CimCmdlets\\CimCmdlets\.psd1'\)/);
  assert.match(source, /CimCmdlets\\Get-CimInstance -ClassName Win32_OperatingSystem/);
  assert.match(source, /ToUniversalTime\(\)\.Ticks\.ToString/);
  assert.match(source, /RegistryView\]::Registry64/);
  assert.match(source, /GetValueNames\(\) -contains 'ContainerType'/);
}
it('binds the fixed metadata script and rejects compilation, value reads and Job APIs', () => {
  const source = readFileSync(new URL('../src/stores/file-lock-windows-identity.ts', import.meta.url), 'utf8');
  guard(source);
  for (const addition of [
    '\nIsProcessInJob(0,0,0);',
    '\nimport value from "node:os";',
    '\nconst bad = "Add-Type";',
    '\nconst bad = "GetValue(\\u0027ContainerType\\u0027)";',
  ]) expect(() => guard(source + addition)).toThrow();
});

it('also guards the identity dispatcher import boundary against alternate Job readers', () => {
  const source = readFileSync(new URL('../src/stores/file-lock-identity.ts', import.meta.url), 'utf8');
  const check = (value: string) => {
    const imports = [...value.matchAll(/from ['"]([^'"]+)['"]/g)].map(match => match[1]).sort();
    assert.deepEqual(imports, ['./file-lock-windows-identity.js', 'node:child_process', 'node:fs']);
    assert(!/IsProcessInJob|QueryInformationJobObject|CreateJobObject|OpenJobObject|NtQueryInformationProcess|Add-Type|DllImport|process\.binding|\b(?:require|import)\s*\(/i.test(value));
  };
  check(source);
  for (const addition of ['\nIsProcessInJob(0,0,0);', '\nimport native from "node:vm";']) expect(() => check(source + addition)).toThrow();
});

beforeEach(() => {
  invoke.mockReset();
  vi.stubEnv('SystemRoot', 'C:\\Windows');
  vi.stubEnv('TEMP', 'C:\\owned-temp');
  vi.stubEnv('TMP', 'C:\\owned-temp');
  vi.stubEnv('PSModulePath', 'C:\\untrusted-modules');
});
afterEach(() => vi.unstubAllEnvs());

it('runs only the absolute first-party binary with a fixed script and constructed environment', () => {
  invoke.mockReturnValue({ status: 0, stderr: '', stdout: 'boot=638000000000000000\r\ncontainer=absent\r\n' });
  expect(windowsLockIdentity()).toEqual({
    platform: 'win32', boot_id: 'cim-utc-ticks:638000000000000000', pid_namespace: 'win32:kernel',
  });
  expect(invoke).toHaveBeenCalledTimes(1);
  const [binary, argv, options] = invoke.mock.calls[0]!;
  expect(binary).toBe('C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe');
  expect(argv.slice(0, 4)).toEqual(['-NoProfile', '-NonInteractive', '-NoLogo', '-Command']);
  expect(argv).toHaveLength(5);
  expect(argv[4]).not.toContain('C:\\Windows');
  expect(options.env).toEqual({
    SystemRoot: 'C:\\Windows', windir: 'C:\\Windows', TEMP: 'C:\\owned-temp', TMP: 'C:\\owned-temp',
  });
  expect(options).toMatchObject({ timeout: 15000, windowsHide: true, shell: false, maxBuffer: 16384 });
});
it('keeps a positively detected container realm unknown', () => {
  invoke.mockReturnValue({ status: 0, stderr: '', stdout: 'boot=638000000000000000\ncontainer=present\n' });
  expect(windowsLockIdentity()?.pid_namespace).toBeNull();
});
it.each([
  'boot=0\ncontainer=absent\n',
  'boot=3155378976000000000\ncontainer=absent\n',
  'boot=638000000000000000\ncontainer=unknown\n',
  'boot=638000000000000000\ncontainer=absent',
  'boot=638000000000000000\ncontainer=absent\nextra\n',
  'boot=638000000000000000\ncontainer=absent\n\n',
  '\nboot=638000000000000000\ncontainer=absent\n',
])('refuses anything outside the strict two-line metadata format: %j', stdout => {
  invoke.mockReturnValue({ status: 0, stderr: '', stdout });
  expect(windowsLockIdentity()).toBeNull();
});
it.each([
  { status: 1, stderr: '', stdout: 'boot=638000000000000000\ncontainer=absent\n' },
  { status: 0, stderr: 'owned warning', stdout: 'boot=638000000000000000\ncontainer=absent\n' },
  { status: null, error: { code: 'ETIMEDOUT' }, stderr: '', stdout: '' },
])('refuses errors, timeout and stderr', result => {
  invoke.mockReturnValue(result);
  expect(windowsLockIdentity()).toBeNull();
});
it('never searches PATH when SystemRoot is unavailable or relative', () => {
  for (const root of ['', 'Windows', '\\Windows', 'C:Windows', '\\\\host\\Windows']) {
    vi.stubEnv('SystemRoot', root);
    expect(windowsLockIdentity()).toBeNull();
  }
  expect(invoke).not.toHaveBeenCalled();
});
