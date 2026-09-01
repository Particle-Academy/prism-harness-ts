# Prism Harness for TypeScript

Durable agent sessions — threads, session state and store drivers. The
TypeScript port of [`particle-academy/prism-harness`](https://github.com/Particle-Academy/prism-harness).

Zero runtime dependencies. Node 20+.

This package is private while coordinated parity work is in progress.

```ts
import { FileSessionStore, MemorySessionStore, PrismHarness } from '@particle-academy/prism-harness';

const harness = new PrismHarness({
  drivers: {
    memory: () => new MemorySessionStore(),
    files: () => new FileSessionStore('./storage/harness'),
  },
  stores: { ephemeral: 'memory', durable: 'files' },
});

const session = harness.for({ type: 'User', id: 7 }).session('support');

await session.usingMode('plan').then((s) => s.usingModel('claude-sonnet-4-5'));

await session.lock(async (live) => {
  // whatever must not happen twice
  await live.beginRun('run-1', 'plan', 'anthropic', 'claude-sonnet-4-5');
});
```

## Resolved, never held

A server handles a request and moves on, so a session cannot be an object kept
in memory the way a single-process agent's is. Every call rebuilds one from a
store, which is what makes a fresh worker see the same mode, model and
conversation as the request that set them.

## The two halves

State is split into named slots, because the halves have genuinely different
requirements:

| Slot | Holds | Losing it means |
|---|---|---|
| `ephemeral` | active mode, selected model, run bookkeeping | falls back to a default |
| `durable` | threads, stored capabilities | work is gone |

**A store that reports itself volatile is REFUSED for the durable slot**, at the
moment a session is opened. That is the guard the package exists for: a cache is
disposable by definition, and the durable slot holds approvals a human has not
answered yet. Accepting it and finding out later is exactly the failure this
package was written to avoid.

Construct a `PrismHarness` with no drivers and it works — and then refuses
durable state, loudly, with a message that names the fix. That is deliberate. A
package that silently accepted an in-memory durable store would pass every test
on one process and lose a half-executed action the first time it ran on two.

## The same address as PHP

`Session.key()` is byte for byte what the reference builds: `session:` plus the
sha1 of the participant type truncated to 12, the participant id, and the scope.
Matching exactly is what lets a PHP app and a TypeScript agent **share one store
and resolve the same session**. A different digest would give two conversations
that look identical and never meet.

## Threads

`record()` assigns positions inside the store lock, reading the current length
and writing the new messages as one operation. Two turns landing concurrently
would otherwise both read position 4 and both write position 5, silently losing
a message — the race the reference tracks as prism-harness#2.

## Drivers

- **`MemorySessionStore`** — volatile, and says so. Right for the ephemeral slot
  in a test or a single-process tool. Its lock is real but process-local.
- **`FileSessionStore`** — durable. Atomic writes (write-then-rename) and a
  cross-process lock built on exclusive file creation, which is the one
  primitive that is atomic on every filesystem worth supporting. Two workers on
  one machine genuinely exclude each other; two machines over a network
  filesystem do not, and no file lock can promise that — use a database there.

Implement `SessionStore` for anything else. Declare your own `durability()`:
only you know whether your Redis is persistent or a disposable cache, and that
declaration is an assertion about your infrastructure, not a preference.
