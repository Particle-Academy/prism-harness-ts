import { pathToFileURL } from 'node:url';
const { FileSessionStore, Session } = await import(pathToFileURL(process.argv[2]).href);
const store = new FileSessionStore(process.argv[3]);
const session = new Session({ participant: { type: 'test', id: 'lock-regression' },
  scope: 'lock-regression', ephemeral: store, durable: store });
let callbacks = 0, code = null;
// Intentionally no IPC channel, fixture timer, host keepalive or signal handler.
try { await session.lock(() => { callbacks++; }, 10, 0.12); }
catch (error) { code = error.code; }
console.log(JSON.stringify({ callbacks, code }));

