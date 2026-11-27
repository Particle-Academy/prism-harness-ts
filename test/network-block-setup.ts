import { syncBuiltinESMExports } from 'node:module';
import http from 'node:http';
import https from 'node:https';
import net from 'node:net';
import tls from 'node:tls';

// Every test file runs with the network blocked: a test that reaches a real
// host fails at once, whatever client it uses. Local IPC (a socket or pipe
// path) is not network and stays open. A test may still stub fetch itself;
// restoring its stub brings back this block, never the real fetch.

const marker = Symbol.for('prism-harness.network-blocked');
const state = globalThis as unknown as Record<symbol, boolean>;

function refuse(what: string): never {
  throw new Error(`A test tried to use the network (${what}). Tests run offline.`);
}

if (state[marker] !== true) {
  state[marker] = true;

  globalThis.fetch = (async () => refuse('fetch')) as typeof fetch;
  if ('WebSocket' in globalThis) (globalThis as Record<string, unknown>).WebSocket = function WebSocket() { refuse('WebSocket'); };

  // A connection to a host and port is network; one to a path (a Unix socket
  // or a Windows named pipe) is local IPC.
  const isPath = (args: unknown[]): boolean => typeof args[0] === 'string' && Number.isNaN(Number(args[0]))
    || (typeof args[0] === 'object' && args[0] !== null && typeof (args[0] as { path?: unknown }).path === 'string');
  for (const name of ['connect', 'createConnection'] as const) {
    const original = net[name] as (...args: unknown[]) => net.Socket;
    (net as unknown as Record<string, unknown>)[name] = (...args: unknown[]) => (isPath(args) ? original(...args) : refuse(`net.${name}`));
  }
  (tls as unknown as Record<string, unknown>).connect = () => refuse('tls.connect');
  for (const [label, module] of [['http', http], ['https', https]] as const) {
    (module as unknown as Record<string, unknown>).request = () => refuse(`${label}.request`);
    (module as unknown as Record<string, unknown>).get = () => refuse(`${label}.get`);
  }
  syncBuiltinESMExports();
}
