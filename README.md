# Prism Harness for TypeScript

Durable agent sessions — threads, session state and store drivers. The
TypeScript port of [`particle-academy/prism-harness`](https://github.com/Particle-Academy/prism-harness).

Zero runtime dependencies. Node 22+.

```
npm install @particle-academy/prism-harness
```

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

## Named agents

Select a stable agent id in trusted application code, using
[a-z][a-z0-9_-]{0,63}. The existing calls without an agent retain their addresses.
Each named agent has its own tools, providers, modes, session state, threads and
default task list.

```ts
import { AgentRuntime, ModeRegistry, ToolRegistry } from '@particle-academy/prism-harness';

const tools = new ToolRegistry(), modes = new ModeRegistry();
tools.forAgent('support').register({ name: 'lookup', handle: () => 'found' }, 'app.support');
modes.forAgent('support').register('reply', { tools: ['lookup'] }, 'app.support').setDefault('reply');
const runtime = new AgentRuntime({ tools, modes,
  client: async () => ({ text: 'done', finishReason: 'stop' }),
}); // Replace the synthetic client when connecting a model.
const session = harness.forAgent('support').for({ type: 'User', id: 7 }).session('case:42');
const descriptors = await runtime.effectiveTools(session, 'reply');
```

Import ToolRegistry, ModeRegistry and AgentRuntime from this package. When the
harness is constructed with runtime, its bound agent also exposes
effectiveTools('reply', participant, scope). Descriptors contain name, owner and
source and come from the model offer resolver. Inspection executes no handler,
creates no session, thread or index entry and makes no model/provider request
or logical state change. Reads may physically purge expired entries, exactly as
an ordinary store read does. Explicit state writes during resolution refuse.

For named agents, the mode list is the ceiling even when a caller supplies tool
names. Legacy sessions retain caller-list selection by default; set
`AgentRuntimeOptions.narrowRequestedTools` to `true` to apply the ceiling there.
Named agents inherit no legacy authority. Builtins require explicit registration and
mode selection. Named-agent duplicates are refused; use
tools.forAgent('support').replace(replacement, 'app.support', 'app.support.v2')
for an intentional replacement.

Legacy registries retain duplicate overwrite and existing cross-kind precedence.
Use `new ToolRegistry({ refuseDuplicates: true })` and
`new ModeRegistry({ refuseDuplicates: true })` to refuse legacy duplicates.
Named session addresses are immutable and validated; legacy participant/scope
values retain their existing mutable behavior.

AgentRuntime's agents map accepts provider/model, max_steps_ceiling,
max_output_tokens_per_call and lock_ttl/lock_wait overrides. Unset scalars fall
back to config; explicit null clears nullable defaults. Caps narrow the existing
shared RunContext ledger and output budget. ToolAuthorizer's agents map supplies
enabled/offer/call policy overrides, retaining invocation-time rechecks.
PrismHarness's agents map separately supplies ttlSeconds, defaultScope and
defaultTaskList. Typed ParticipantRef values require a ParticipantAdapters actor
resolver; a child keeps the already-bound actor. Existing plain participants
continue to work.

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

## Attachments

A turn can carry media alongside its prompt:

```ts
import { Image } from '@particle-academy/prism';

await runtime.send(session, 'What is wrong with this layout?', undefined, undefined, [
  Image.fromBase64(screenshot, 'image/png'),
]);
```

Pass `prism-ts` media objects or media already serialized. Attachments are stored
with the turn in the shape `UserMessage.fromObject()` rebuilds. Each one must be
an image, document, audio or video that carries its bytes, a provider file id or
document chunks. The same rules as the PHP reference and the Python port apply,
pinned across all three by prism-parity's `harness-turn-attachments` corpus.

**Refused, with a `HarnessError` whose `code` names the problem:**

| Code | When |
|---|---|
| `attachment_by_reference` | built from a URL or read from a file |
| `attachment_not_media` | anything other than those four media types |
| `attachment_empty` | no bytes, no file id and no chunks |
| `attachment_without_prompt` | attachments with an empty prompt |

A refusal happens before a run starts: no run, no events, nothing in the thread.

## Provider options per mode

A mode can declare `provider_options`. They reach your model client unchanged,
as `request.providerOptions`, on every step of every run in that mode. Pass them
to your provider call (with `prism-ts`, `withProviderOptions()`):

```ts
import { ModeRegistry } from '@particle-academy/prism-harness';

new ModeRegistry({
  modes: { overseer: { system_prompt: '...', provider_options: { thinking: { type: 'adaptive' } } } },
});
```

A value that is not a map is refused when the mode is resolved, with
`mode_malformed`.

## What your model client receives

`request.messages` is the thread, oldest first, in the shape the PHP reference
stores Prism's messages:

| `type` | Carries |
|---|---|
| `user` | `content`, and `additional_content` when the turn has attachments |
| `assistant` | `content`, `tool_calls` (`id`, `name`, `arguments`, `result_id`, `reasoning_id`, `reasoning_summary`), `additional_content`, `tool_approval_requests` |
| `tool_result` | `tool_results` (`tool_call_id`, `tool_name`, `args`, `result`, `tool_call_result_id`, `artifacts`), `tool_approval_responses` |

Consecutive tool result rows reach the client as one, so each call's result is
sent to the provider once. Return a call's provider ids as `resultId`,
`reasoningId` and `reasoningSummary`, and provider state for the turn as
`additionalContent`: they are recorded and come back in `messages`.

## Structured turns

When the answer is a document rather than prose, hand the turn a schema:

```ts
const schema = {
  name: 'plan',
  type: 'object',
  properties: {
    title: { type: 'string' },
    steps: { type: 'array', items: { type: 'string' } },
  },
  required: ['title', 'steps'],
};

const response = await runtime.sendStructured(session, 'Plan the release', schema);

response.structured; // { title: 'Ship it', steps: ['write', 'test'] }
response.text; // the document as the model wrote it
```

It is the same run as `send()` — the mode's system prompt, its tools, the step
budget, the events — asking the provider for structured output. Your client
receives the schema as `request.schema` and returns what it parsed as
`response.structured`; with `prism-ts` that is `Prism.structured().withSchema()`
and the response's `structured`.

**The thread keeps the text, with the parsed object beside it.** The assistant
message is the raw document, and `structured` rides along in the row's
`additional_content`. A later turn replays the conversation as messages and reads
the text, so a thread that contains a structured answer reads like any other.

**A document that misses the schema is refused, not repaired.**

```ts
import { HarnessError } from '@particle-academy/prism-harness';

try {
  const plan = (await runtime.sendStructured(session, brief, schema)).structured;
} catch (error) {
  if (error instanceof HarnessError) {
    error.code; // structured_schema_violation, or structured_unreadable
    error.problems; // every way it missed, not the first
    error.document; // what the model actually said
  }
}
```

Nothing is coerced, nothing is trimmed to the fields that fit, and the result is
never an empty document. An empty plan settles a batch as `done`, which reads
exactly like a considered answer of "nothing to propose" — the failure this
refusal exists to prevent. The exchange is still recorded, and the run is marked
failed: a thread that omits the answer it did not like cannot explain the retry
sitting next to it. The run row and the `run.failed` event name the CODE, not the
message — the message quotes the values that missed, and an event carrying those
would ship model output to every listener, which is the same reason tool
arguments are names-only here.

The check reads the schema's own JSON Schema, so a hand-written schema is held to
the same terms. `schemaProblems()` is exported if you want it directly. It checks
declared types, required keys, enum members, array items, and — where a schema
closes itself — keys nobody declared. It does not read `$ref`, `allOf`, `oneOf`
or the numeric and string facets; what it cannot read, it passes, rather than
reporting a constraint it did not actually check. The same rules as the PHP
reference, message for message, so a document refused in one language is refused
in the other for the same stated reason.

## Failed model steps

A client response with `finishReason: 'error'` or `'unknown'` ends the run with
`HarnessError.code === 'model_step_failed'`. Its `finishReason` identifies which
of those two finishes occurred; the error message contains no model output.
The step, cost, and assistant response are recorded before the error is raised.
Tools requested by that response are refused, with no pending approval authority,
so approving or resuming later cannot execute them. A new turn can still be sent.

## Approvals

A mode names the tools a person must approve:

```ts
new ModeRegistry({
  modes: {
    guarded: { system_prompt: '...', tools: ['read', 'delete'], requires_approval: ['delete'] },
  },
});
```

A step that calls one stops with `finishReason: 'awaiting_approval'`. The calls
that need nobody have already run. Record a decision for every pending approval,
then resume with an empty prompt:

```ts
import { recordApproval } from '@particle-academy/prism-harness';

const response = await runtime.send(session, 'Clean up the failed run');

if (response.finishReason === 'awaiting_approval') {
  for (const pending of response.pendingApprovals) {
    await recordApproval(session, pending.id, true); // or false, 'not on production'
  }

  await runtime.send(session, '');
}
```

On resume the approved calls run once and denied calls return their reason, and
then the model continues. The model is not asked to make the calls again. A
pending call with no decision is refused with "No approval response provided".
A call that has a result never runs again. Who may approve is your application's
decision: authorize before calling `recordApproval()`.

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
- **`FileSessionStore`** — durable. Payload writes remain atomic. Cross-process
  local locks follow [decision 0022](https://github.com/Particle-Academy/prism-parity/blob/main/docs/decisions/0022-local-file-lock-ownership.md):
  a complete nonce-bound owner directory is published atomically; TTL expiry
  never proves that a live holder died. Recovery requires a matching established
  native boot/PID realm and a proven dead or explicitly released owner. Local
  filesystems on one host only; SMB/NFS are unsupported.

  State readers close their handles before decoding the document. On Windows,
  replacement of a state file can fail while another process holds it open,
  even with delete sharing. Permission failures retry for up to 750 ms. If the
  reader stays open or access is permanently denied, the original native error
  is raised and the temporary file is removed; the previous state stays intact.
  Other errors and POSIX replacement do not retry. Callbacks are never replayed.

  On Windows, the first file-lock operation makes one fixed, read-only metadata
  query using the system Windows PowerShell and CIM module, with a fifteen-second
  timeout. Established identities and detected container conditions are cached
  for the process lifetime. Transient query failures are cached for sixty seconds,
  then permit a new sample. With unknown identity a worker can acquire a free
  lock and release its own owner record, but unknown owners and unknown probers
  never authorize PID probes or recovery. A crashed unknown owner's lock stays
  held for explicit operator inspection. `FileSessionStore.identityStatus()` is
  an in-memory snapshot and performs no query or filesystem work. No runtime
  compilation or extra dependency is used.

  **BREAKING CHANGE: Stop all old workers before upgrading the file-lock
  protocol.** Inspect and archive stale legacy timestamp lockfiles explicitly.
  The library refuses them with `session_locked` and
  `legacy_lock_requires_maintenance`, naming the canonical path and action
  `stop_old_workers_then_inspect_and_archive`. On Windows, Node can replace a
  legacy file created between preflight and rename; leaving old workers running
  can break exclusion. No rolling upgrade is supported. The hard stop is a
  correctness requirement in both ports.

  A blocked Windows reader can delay release. The callback may already have
  completed when release raises an error; the library does not replay it.
  Pending cleanup is retried at the owning process's next file-lock operation.
  The `file_lock_release_pending` warning includes only the key digest and nonce.
  Read `store.pendingReleases()` for this directory's pending digest/nonce entries;
  the snapshot's length is the count. Querying does not retry or write anything.

  **Threads are append-only logs** (`<digest>.jsonl` beside the payload file):
  one line per message, one write and an `fsync` per `record()`, so recording
  costs the size of the new messages however long the thread already is.
  Before 0.7.0, every message re-read and rewrote the whole conversation. A
  thread stored that way is read as before and moved into the log the first
  time it grows. A line cut off by a crash is dropped, and the next append
  starts after the last complete line. A damaged line anywhere else is refused
  with `unmappable_content` rather than silently skipped.

  The thread log is **not yet shared with the PHP and Python ports**: a Python
  worker pointed at the same directory does not read it.

Implement `SessionStore` for anything else. Declare your own `durability()`:
only you know whether your Redis is persistent or a disposable cache, and that
declaration is an assertion about your infrastructure, not a preference.
