import { spawnSync } from 'node:child_process';
import { win32 } from 'node:path';
import type { LockIdentity } from './file-lock-identity.js';

// Master2797: fixed metadata only, no compilation, no value payload or Job state.
// Resolve the first-party CIM module from PSHOME, never a user module search path.
const COMMAND = String.raw`
$ErrorActionPreference = 'Stop'
$PSModuleAutoloadingPreference = 'None'
try {
  Import-Module -Name ($PSHOME + '\Modules\Microsoft.PowerShell.Utility\Microsoft.PowerShell.Utility.psd1') -ErrorAction Stop
  Import-Module -Name ($PSHOME + '\Modules\CimCmdlets\CimCmdlets.psd1') -ErrorAction Stop
  $boot = (CimCmdlets\Get-CimInstance -ClassName Win32_OperatingSystem -ErrorAction Stop).LastBootUpTime
  if ($null -eq $boot) { throw 'Missing boot identity' }
  $ticks = $boot.ToUniversalTime().Ticks.ToString([Globalization.CultureInfo]::InvariantCulture)
  $base = [Microsoft.Win32.RegistryKey]::OpenBaseKey([Microsoft.Win32.RegistryHive]::LocalMachine, [Microsoft.Win32.RegistryView]::Registry64)
  try {
    $key = $base.OpenSubKey('SYSTEM\CurrentControlSet\Control')
    if ($null -eq $key) { throw 'Missing registry key' }
    try { $present = $key.GetValueNames() -contains 'ContainerType' }
    finally { $key.Dispose() }
  } finally { $base.Dispose() }
  [Console]::Out.WriteLine('boot=' + $ticks)
  if ($present) { [Console]::Out.WriteLine('container=present') }
  else { [Console]::Out.WriteLine('container=absent') }
} catch { exit 1 }
`;

export function windowsLockIdentity(): LockIdentity | null {
  const root = process.env.SystemRoot;
  if (!root || !/^[a-z]:[\\/]/i.test(root) || !win32.isAbsolute(root)) return null;
  const env: NodeJS.ProcessEnv = { SystemRoot: root, windir: root };
  for (const key of ['TEMP', 'TMP'] as const) if (process.env[key]) env[key] = process.env[key];
  const binary = win32.join(env.SystemRoot!, 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe');
  try {
    const result = spawnSync(binary, ['-NoProfile', '-NonInteractive', '-NoLogo', '-Command', COMMAND],
      { env, encoding: 'utf8', timeout: 15000, maxBuffer: 16384, windowsHide: true, shell: false });
    if (result.error || result.status !== 0 || result.stderr !== '') return null;
    const match = /^boot=([1-9][0-9]{0,18})\r?\ncontainer=(present|absent)\r?\n$/.exec(result.stdout);
    if (!match || match[0] !== result.stdout || BigInt(match[1]!) > 3155378975999999999n) return null;
    return { platform: 'win32', boot_id: 'cim-utc-ticks:' + match[1],
      pid_namespace: match[2] === 'absent' ? 'win32:kernel' : null };
  } catch { return null; }
}
