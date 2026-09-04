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
| `durable` | threads, stored capabilities, agent task lists | work is gone |

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

## Agent task lists

An agent given a goal has to keep working across many requests. It needs a list
of what remains, and that list has to survive the request, the worker, a crash
and a deploy.

```ts
const tasks = session.tasks();

await tasks.addMany(['read the issue', 'write the patch', 'run the gates']);

const task = await tasks.claim('worker-1');   // ONE atomic call

if (task !== null) {
  // …do the work…
  await tasks.release(task, 'worker-1', 'done');
}

await tasks.pending();   // a COUNT, which is what a loop needs to stop
```

There is **no task model, no schema and no migration**. Two contracts and
adapters: `AgentTaskSource` is where tasks come from — `claim`, `release`,
`pending`, `find` — and `AgentTask` is one unit of work. `find()` is on the
contract because `release()` takes a task while every external caller holds
only an id: a tool call carries `{"id": "t-1"}` and an HTTP route has
`/tasks/t-1`.

`StoreTaskSource` is the default, backed by the durable slot; a consumer with an
existing table wraps their own record with `asAgentTask()` or mixes
`withAgentTask()` into their class. Both satisfy one contract.

Four states — `todo`, `claimed`, `done`, `failed` — and the transitions are
pinned across all three languages:

| | |
|---|---|
| `claim()` is one call | Read-then-mark is two operations with a window between them, and two workers arriving in that window get the same task. |
| A claim carries an owner AND an expiry | Five minutes by default. An expired claim returns the task to **`todo`**, never to `failed`: a worker dying is not the task failing, and conflating them burns a retry that never ran. |
| `claimed` is written before the work begins | So "started and died" is distinguishable from "never started". |
| `done` and `failed` are terminal | Re-releasing one is an error, not a silent no-op. |
| A worker may extend its own lease | Only while it still holds it, and bounded by the RUN. `RunLedger.exhaustion()` refuses outright — cancelled, out of steps, out of money or out of time — and `remainingSeconds()` bounds what is granted. There is no second timeout to set, or to forget to set. The new expiry is `now + granted` even when that shortens the lease: the grant is what the run can still afford. |
| A lease must be whole positive seconds | Zero or less is refused rather than clamped, and a fraction is refused rather than truncated — both are a value quietly becoming a different value, and `claimed_until` is an integer timestamp so a fractional lease could never have been honoured anyway. The check is `Number.isInteger`, which also refuses `NaN`, the infinities, and the string a JSON config would have delivered past the type annotation. |
| `release()` names the worker | Not in the spec's sketch, and added because without it a completion tool can close any task in the list — including one another worker is midway through. A release by anyone but the holder is refused, and the refusal does not say who the holder is. |

**A volatile store is refused**, the same way the durable slot is. A
half-finished task list that vanishes on a deploy is indistinguishable from a
finished one.

**An empty worker id or task id is refused, and nothing is trimmed first.**
`trim`, `strip` and `String.prototype.trim` each strip a different set of
codepoints, so trimming would have three ports disagreeing about whether the
same id is blank. A single space is a legal id. That is the `prism-human-plus`
G-36 lesson applied before it costs anything.

**An agent cannot mark its own task complete.** `release()` is the
application's call, made from evidence. A consumer who wants the agent to close
its own tasks builds `agentCompletionTool()` and authorizes it through the
existing `ToolAuthorizer` — which must be enabled, because an authorizer that is
off allows every call and would grant the authority while reading as a control.
If the model can declare its own work done, "run until the goal is met" becomes
"run until it decides it is met", and a run that has stalled ends by declaring
victory.

That tool **requires an explicit `outcome`** of `done` or `failed`. Anything
else — `"complete"`, `null`, or a missing argument — is refused rather than
guessed at, because `outcome` is the one value the model supplies and it decides
a terminal state. Guessing would let a malformed call produce the more
privileged result.

It also **checks the holder itself** rather than trusting the source to. It is
typed to the `AgentTaskSource` contract, and an interface cannot make an
implementation check anything — `release(task, worker, outcome)` reads like
"find it and set the state", which is what a third party will write. A task
whose holder cannot be established at all is refused too: `AgentTask` is three
methods, so a conforming source may expose no holder, and reading that silence
as permission is the same mistake as inferring `done` from a missing outcome.

Stopping is the existing `RunBudget` — cost, turns and wall-clock. Dependency
ordering is not here: drive [`fancy-flow`](https://github.com/Particle-Academy/fancy-flow)
for that. A task list that grows a scheduler has rebuilt a workflow engine
badly.

## Drivers

- **`MemorySessionStore`** — volatile, and says so. Right for the ephemeral slot
  in a test or a single-process tool. Its lock is real but process-local.
- **`FileSessionStore`** — durable. Atomic writes (write-then-rename) and a
  cross-process lock built on exclusive file creation, which is the one
  primitive that is atomic on every filesystem worth supporting. Two workers on
  one machine genuinely exclude each other; two machines over a network
  filesystem do not, and no file lock can promise that — use a database there.

  Its **lockfile format is shared with the PHP and Python ports**, because a
  worker in any of the three may be pointed at one store directory: the expiry
  in milliseconds, then a newline, written in a single write. The terminator is
  load-bearing. Without it there is nothing to tell a complete expiry from the
  first half of one, and every prefix of a timestamp is a smaller, *older*
  timestamp — so a torn write reads as a lock that expired decades ago and gets
  deleted out from under the worker holding it. Anything unterminated is treated
  as held, bounded by the file's own age so a process that died mid-write cannot
  wedge the key either.

Implement `SessionStore` for anything else. Declare your own `durability()`:
only you know whether your Redis is persistent or a disposable cache, and that
declaration is an assertion about your infrastructure, not a preference.
