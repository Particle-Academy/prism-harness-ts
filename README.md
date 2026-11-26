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

await session.usingMode('plan').then((s) => s.usingModel('claude-sonnet-5-5'));

await session.lock(async (live) => {
  // whatever must not happen twice
  await live.beginRun('run-1', 'plan', 'anthropic', 'claude-sonnet-5-5');
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

The mode list is the ceiling for named and legacy sessions. Caller-supplied
names narrow it; omitting names uses the mode's full list, and an empty list
offers no tools. A requested `*` selects the mode's ceiling. Other names outside
the ceiling fail before the model is called.

Legacy callers that intentionally need the previous replacement behavior can
set `AgentRuntimeOptions.allowLegacyToolListWidening` to `true`. This permits
widening only in the legacy namespace; the configured authorizer still filters
the offer and tool calls. Named agents always retain their ceiling. Remove an
old explicit `narrowRequestedTools: false`: it now raises a migration error.
Explicit `true` remains supported, and conflicts with the widening opt-out.
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

## Provider and model per mode, subagent and agent

`request.provider` and `request.model` are resolved field by field. For a run:

1. the session override (`usingProvider()` / `usingModel()`);
2. the mode's `provider` / `model`;
3. the agent's `provider` / `model` (`agents.<id>` on `AgentRuntime`);
4. the runtime `config`, then `'unknown'`.

A subagent's child run resolves:

1. the parent session's override, only when the declaration sets
   `inherit_session_model: true`;
2. the subagent declaration's `provider` / `model`;
3. the child's own mode;
4. the parent's mode, only when the child runs in the parent's agent;
5. the child agent's default, then the runtime `config`.

A subagent can run as another named agent with `agent`, but only when the parent
mode lists that agent in `agents`. Any other agent, and any id that is padded or
not lowercase, is refused with `mode_malformed` when the mode is resolved. Ids
are never trimmed or case-folded.

```ts
new ModeRegistry({ agents: {
  planner: { owner: 'app', modes: { plan: {
    model: 'big-model',
    agents: ['researcher'],
    subagents: { research: { mode: 'research', agent: 'researcher' } },
  } } },
  researcher: { owner: 'app', modes: { research: {} } },
} });
new AgentRuntime({ /* ... */ agents: {
  planner: { provider: 'anthropic', model: 'planner-model' },
  researcher: { provider: 'my-vllm', model: 'small-model', spend_ceiling: 0.25 },
} });
```

`spend_ceiling` is the most an agent may spend, in USD, within one run tree. It is
checked against the tree's own ledger before each model call, so it never opens a
second account. An agent at its ceiling is refused with
`agent_spend_ceiling_exceeded`. If its provider reported no cost, it is refused
with `budget_unpriced`, because the ceiling cannot be measured. A refused
subagent returns to the parent as an `exhausted` result, and the parent keeps
running.

The ceiling covers one run tree; there is no per-day window. A run that pauses
for an approval and is resumed (`send(session, '')` or `streamDecision()`) is
the same run: the agent's spend so far is stored with the approval request and
counts against the ceiling after the resume, as the step count does. A subagent's
child run resumed that way, outside the runner, also keeps the provider and model
it was bound to, and continues the run tree it was started in rather than getting
a fresh root budget. So does every other turn on the child session the runner did
not start: a new message sent to it instead of an answer, and a turn handed a run
context the host built, which is narrowed to what the tree (and the plan budget
below) has left and can never widen it. The first time a root run starts a child, the tree's step and
cost allowance is kept in the durable store, and the root run settles what the
whole tree spent when it ends. The resumed child holds what the tree has left
before it calls a provider and settles what it spent when it ends. Once the
tree's steps are spent, a decision on the child's approval is still accepted:
acting on it costs no model step, so the resume runs the approved call and
finishes with `budget_exhausted` without calling a provider. A tree whose COST
budget is spent refuses the decision (`run_not_permitted`) with nothing
recorded. A hold whose run never settled is charged in full once the run lock's
TTL has passed. When the child was started by a turn that spent from a shared
plan budget, the resumed child spends from that plan budget too.

How many model steps any run may take is decided once, when it begins, and
never re-derived (G-387): the least of its own ceiling less the steps it has
used, what a host context offers, and what its run tree and shared plan budget
grant it. Its own ceiling is, for a child the runner starts, what the runner's
context leaves it (its declared `max_steps` within its parent's remainder);
for any later turn on that child the runner did not start, that same ceiling,
saved with the child; otherwise the mode's `max_steps`. `max_steps_ceiling`
caps it under every context. The steps used are written on the row a run parks
on (`harness_steps_used`, its subagent children's steps included) and its
resume continues from them; a value there that is not a non-negative whole
number fails closed. A child started under a run
context the host built itself, outside any turn, has no tree record and is
resumed on its own root budget. Every other `send()` starts a new run tree with the full
ceiling, and so does each `plan.advance()` called without a context. To hold
the steps of a plan to one ceiling, pass the same `RunContext` to every
`advance(context)`; that context's budget then bounds the whole plan, its step
limit included.

A subagent tool built with `SubagentRunner.tool(subagent, parent, context, parentRunId)`
and called by a turn on `parent` runs its child in THAT turn's run tree: the
child spends from the turn's ledger, so from the turn's budget and, for a plan
step, from the plan's shared budget. The `context` it was built with can then
only narrow the turn's limits, by what it has left; it never widens them and
its own ledger is not charged. Called outside a turn, `context` is the run tree,
as before.

Every run event of a named agent carries `harnessAgent`. Legacy events do not.

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
decision: authorize before calling `recordApproval()`. `recordDecisions(session,
[{ approvalId, approved, reason }])` records several decisions as one row.

Recording a decision takes the session lock, which is not re-entrant: do not
call these from inside your own `session.lock(...)` callback. `recordApproval()`
and `recordDecisions()` wait up to 30 s for the lock; `runtime.decide(session,
approvalId, approved, reason?)` records the same decision using the session's
agent `lock_ttl` / `lock_wait`. A caller that cannot get the lock gets
`session_locked`, and nothing is recorded.

A decision must answer an approval the session is waiting on. `recordApproval()`,
`recordDecisions()` and `streamDecision()` reject with a `HarnessError` coded
`approval_not_pending` for an approval id the run never issued, or one already
answered, and refuse the whole batch when any one decision is like that. Nothing
is written and nothing is sent, so a made-up id cannot cancel a real one, and a
decision with nothing pending starts no model turn.

## Run steps for a UI

`streamSteps()` takes the same arguments as `send()` and yields run steps as the
run happens: thinking, text, tool calls and results, approval requests, errors
and step boundaries, each tagged for a client to render. The generator's return
value is the run's `AgentResponse`.

```ts
const steps = runtime.streamSteps(session, 'Refactor the billing job', undefined, RunContext.metadata(turnId, userId));

for await (const step of steps) {
  JSON.stringify(step); // contractVersion "g282.1", runId, parentRunId, rootRunId, turnId, seq, at, type, ...
}
```

- `seq` counts from 1 per run with no gaps; render in that order.
- A tool call carries `startedAt`. Its result carries `ok`, `result` or
  `error { code, message }`, `refused`, `durationMs` and `durationExact`.
  - `refused` is true for an expected refusal: a denied approval, a tool no
    longer offered, tool authorization, or a gate. A tool that threw is
    `ok: false, refused: false`.
  - `durationExact` is false when the step made several tool calls or the tool
    ran while resuming an approval. The duration is then an upper bound.
- An `error` step's message is a fixed generic text.
- `RunContext.metadata(turnId, actor)` sets `turnId` and `actor` without
  changing how the run executes.
- The stream holds the session lock for the whole turn. If another holder has
  it, the stream fails with `session_locked` after `lock_wait` seconds without
  calling the model.
- Once started, a run finishes even if you stop reading.

`send()` is unchanged. To also receive every step as a `run.stepped` event,
build the runtime with `stepEvents: true`. It is off by default because steps
carry tool arguments, results and model text, and anything that records or
forwards every event would otherwise capture them.

### Streaming an approval

`streamDecision(session, decisions, context)` checks the decisions when called,
then, once the stream starts and under the session lock, records them in the
thread exactly as `recordApproval()` does and resumes the run. The first step for
each decision is `decision_recorded` (`approvalId`, `verdict`, `reason`,
`actor`); the run's steps follow, continuing its `seq`. Decisions are saved
before any tool runs, so resuming again runs nothing twice: streaming the same
decision again is refused with `approval_not_pending`. The resumed run emits
`run.resumed`, not a second `run.started`. `actor` comes from your `RunContext`,
never from the model.

```ts
for await (const step of runtime.streamDecision(session, [{ approvalId, approved: true }], RunContext.metadata(turnId, userId))) {
  render(step);
}
```

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

## Agent plans

A plan is an optional, ordered list of steps that an agent writes for a goal.
A person approves it, and then the host moves it forward one step per turn. A
plan says what to do next and when to stop for a person. It grants nothing:
every tool call a step makes still goes through the mode's tools, approvals,
the tool policy and the run budget. A plan step is a separate thing from a task
in the list above, and a done plan step never closes a task.

### Turning plans on

Plans are off unless both switches are on: the `Plans` config's `enabled`, and
the mode's `plans` flag. Give the harness the same `Plans` its runtime's
sessions use:

```ts
import { AgentRuntime, ModeRegistry, Plans, PrismHarness, RunContext, diagnose } from '@particle-academy/prism-harness';

const modes = new ModeRegistry({ modes: { study: { tools: ['search'], plans: true } } });
const plans = new Plans({ modes, events, config: { enabled: true } });
const runtime = new AgentRuntime({ client, modes, tools, events });
const harness = new PrismHarness({ drivers, stores, runtime, plans });
```

When either switch is off, no plan tool is offered, nothing is added to the
system prompt, no plan step is streamed, and `await session.plan()` rejects
with `PlansDisabled` (code `plan.disabled`). A mode's `plans` must be `true` or
`false`; anything else is refused with `mode_malformed`.

### A person approves every plan

A drafted plan waits for a person before any step runs. The draft is an
ordinary tool approval, answered with `recordApproval()`, `recordDecisions()`
or `streamDecision()`, then resumed with `send(session, '')`. Turning this off
is a host decision made in config (`require_approval: false`). There is no
per-call option, and nothing the model sends can change it. Plan changes and
approval pauses always go to a person.

### The agent's tools

| tool | asks a person | what it does |
|---|---|---|
| `plan_draft(goal, steps)` | yes | Proposes the plan. Each step is the agent's own (`owner: agent`, with an `instruction`) or a question for the person (`owner: human`, `interaction: input`, with a `question`). |
| `plan_complete_step(summary)` | no | Reports the current step done. The next `advance()` settles it. |
| `plan_pause_for_person(kind, question, options?)` | `approve`: yes | Stops before the current step. `approve` is a tool approval; `input` waits for `answerStep()`. |
| `plan_propose_change(reason, base_revision, steps)` | yes | Proposes a new list for the steps that can still be replaced. Every other step is kept by id. |

Each plan tool carries a JSON Schema for its arguments as `parameters`, for
your model client to send to its provider. No tool takes an approval id, and
only `plan_propose_change` takes step ids (to keep steps unchanged). A refusal
is a fixed sentence and a code, such as `[plan.text]`.

The plan tools decide for themselves which calls need a person. A mode's
`requires_approval` (even `'*'`) and the host tool policy never apply to them,
and no tool policy can name one: `ToolPolicy.fromJson()` refuses a plan tool in
any spelling.

### Bounds

A plan has at most 40 steps (`max_steps`, 20 by default). A title is at most
160 characters, an instruction at most 600, and a note at most 200, counted in
Unicode code points. The first update must stay under 8 KB. Text with line
breaks, control, bidi or zero-width characters, guillemets, a lone surrogate,
or the plan frame's markers is refused. A plan over a bound is refused when it
is drafted, and is never truncated.

### Driving a plan

```ts
const plan = await session.plan();

await plan.advance();                   // settle the last step, run one turn on the next
for await (const step of plan.streamAdvance()) render(step); // the same, as run steps
(await plan.current())?.toJSON();       // the whole plan, for a fetch-on-change endpoint

// Host calls: authorize first, and name the person as the actor.
await plan.answerStep(stepId, 'Paris', null, RunContext.metadata(null, 'user:42'));
await plan.resume(RunContext.metadata(null, 'user:42'));
await plan.cancel(RunContext.metadata(null, 'user:42'));
```

`advance()` runs at most one step and returns an `AdvanceOutcome`: `ran`,
`waiting`, `blocked` (with a `reason`), `done` or `not_active`. A step is done
only when the agent called `plan_complete_step` and the turn ended cleanly with
nothing pending. A turn that ends without the tool sends the step back to
`todo`, and it counts as an attempt. Completion is never read from the model's
text. The model's turn is always "Continue the plan. Work on step N."; the
step's instruction reaches it only in the plan block of the system prompt.

When a step's turn asks for a tool approval, the step is held on that
approval. Once a person answers it, call `advance()` again. A refused approval
fails the step. It is never shown as done. An answer to a question step never
releases a held step.

Limits: `max_attempts` per step (3), `max_pauses_per_step` (1),
`max_changes` per plan (5), and `max_advances` per plan (60). Hitting one
blocks the plan. `resume()` adds a retry step where that makes sense. It never
replays a step that paused too often, and it never resumes a plan that ran out
of advances.

Only one `advance()` runs at a time per session. A second one fails at once
with `session_locked` and changes nothing.

### Plan budgets

How a plan's steps are budgeted is the app's choice, set in the plans config's
`budget` or per plan with `plan.useBudget({...})`. The model and per-call
options cannot choose it or widen it.

| mode | what it means |
|---|---|
| `per_step` (default) | Each step's turn gets the per-turn budget, as before. |
| `shared` | The plan's own turns also spend from one ledger: `max_steps` model steps in total (required) and optionally `max_cost_usd`. Per-turn limits still apply to each turn. When the ledger is spent, the plan is blocked with `plan_budget_exhausted` and a `plan.budget_exhausted` event is emitted once. |
| `none` | No plan-level budget. Per-turn limits, the plan's own limits and agent spend ceilings still apply. |

```ts
const plans = new Plans({ modes, events, config: { enabled: true, budget: { mode: 'shared', max_steps: 40, max_cost_usd: 2 } } });

// or for one plan, in host code:
await (await session.plan()).useBudget({ mode: 'shared', max_steps: 12 });
await (await session.plan()).budget(); // mode, limits, and what a shared budget has spent
```

A shared ledger lives in the plan store, so it carries across approval pauses,
restarts and workers. Each turn holds what it may spend before it starts and
settles what it spent when it ends, in one locked write each, so two turns are
never granted the same remainder. A hold whose turn never settled is charged in
full once the plan lock TTL has passed. A plan's budget is fixed when it is
first chosen or first used: after that its mode cannot change
(`plan.budget_fixed`) and its limits can only be lowered. A step whose turn is
handed a run tree of its own is refused (`plan.budget_context`) rather than run
outside the ledger. A turn on a subagent child that a plan's turn started, handed
a host-built context, is narrowed to what the plan budget has left instead. An invalid mode or limit throws `PlanBudgetInvalid` when
plans are used, before anything runs. A turn whose ledger has no step left
takes no model step and acts on its decisions only; one whose ledger's cost is
spent fails with `plan_budget_exhausted`.

The plan's own turns are the turn `advance()` runs for the step it claimed and
the resume of an approval that step asked for. Any other turn on the session,
such as the person chatting while a step waits on their approval, spends
nothing from the plan's budget and is never refused for it. A decision on an
approval the step asked for (`recordApproval()`, `recordDecisions()`,
`runtime.decide()` or `streamDecision()`) is accepted even when the shared
budget's steps are spent: the resume runs the approved call and takes no model
step, so an approval is never accepted and then not acted on. The next
`advance()` then blocks the plan. A decision is refused with
`plan_budget_exhausted`, recording nothing, only when the budget's COST is
spent.

### Hosts with queues

If a queued job calls `advance()`, register a reaper callback and run the sweep
on a schedule:

```ts
plans.reapUsing((sessionKey, planId) => queue.dispatch('advance-plan', { sessionKey }));
setInterval(() => void plans.sweep(), 5 * 60 * 1000);

diagnose({ modes, stores, plans: await plans.doctor({ queued: true }) });
```

With plans on and `queued: true`, the doctor fails when the sweep has not run
recently.

### What a client sees

- A `plan.updated` event on every write. It carries ids, the revision, the
  status and counts only. Fetch `current()` for the rest.
- With the config's `events: { run_steps: true }`, `streamSteps()`,
  `streamDecision()` and `streamAdvance()` carry `plan_updated` steps
  (contract `g282.3`). The first is a full snapshot, and later ones carry only
  the steps that changed. Every other step stays `g282.1`. Leave this off until
  your consumer knows `g282.3`.
- The plan block the model reads each turn is built per run and never stored
  in the thread. Instructions, notes, questions and answers do not appear in
  events, plan steps or warnings.

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

  Windows CIM and macOS boot-identity queries run asynchronously. Locks await a
  shared pending sample; successful identity stays cached, and a transient
  failure waits sixty seconds before another async sample. Hosts can optionally
  call `await warmLockIdentity()` at startup. The exported
  `fileLockIdentityStatus()` and the store's `identityStatus()` only observe the
  current cache and never start a query. Unknown identity still cannot authorize
  recovery, and the ownership/nonce/death checks are unchanged.

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

  Reading decodes one completed line at a time, avoiding the whole-file
  JavaScript string limit. `readLog()` still returns the complete array, so
  memory for the decoded messages and the largest single record is required.
  `logLength()` counts newline boundaries without parsing message JSON; record
  validation happens when history is read. Its disposable `<digest>.jsonl.index.json`
  sidecar contains only a count and file metadata, never message content. The
  count is reused only while size, modification/change times and file identity
  match; invalid or missing indexes are rebuilt by streaming the log. The
  checksum detects damaged cache metadata; it is not an authentication boundary.
  Removing a sidecar is safe and does not remove history. Log writes retain
  their existing per-append `fsync`; the advisory sidecar adds no durability claim.

  The thread log is **not yet shared with the PHP and Python ports**: a Python
  worker pointed at the same directory does not read it.

Implement `SessionStore` for anything else. Declare your own `durability()`:
only you know whether your Redis is persistent or a disposable cache, and that
declaration is an assertion about your infrastructure, not a preference.
