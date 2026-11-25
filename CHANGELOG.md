# Changelog

## Unreleased

### Added

- `bindingRoute()`: the one rule for which provider/model binding a turn runs
  on and whether it is saved with the session, now asked by the runtime and
  pinned by the harness-plans corpus (hpl-0056..hpl-0058) in all three
  languages. Behaviour is unchanged.
- Plan budgets (G-366): the plans config's `budget.mode` is `per_step` (the
  default, unchanged behaviour), `shared` (one ledger of `max_steps` model
  steps and optionally `max_cost_usd` across the plan's own turns, kept in
  the plan store) or `none`. Host code can choose it per plan with
  `PlanHandle.useBudget()`; `PlanHandle.budget()` reports it. A spent shared
  budget blocks the plan with `plan_budget_exhausted`, emits
  `plan.budget_exhausted` once, and cannot be resumed. See the README's
  "Plan budgets".
- A shared plan budget charges only the plan's own turns: the turn
  `advance()` runs for the step it claimed and the resume of an approval that
  step asked for. A person chatting while a step is held, or while a claim
  whose worker died is still live, is no longer charged to the plan or
  refused once it is spent. A decision on the held step's approval is refused
  with `plan_budget_exhausted` before anything is recorded when the shared
  budget is spent, on every decision path, and the next `advance()` blocks
  the plan once, so an approval is never recorded and then never run.
- A subagent tool built with `SubagentRunner.tool()` and called by a turn on
  its parent session now runs its child in that turn's run tree, so the
  child's spend counts against the turn's budget and a shared plan budget.
  The run context the host built the tool with can only narrow the turn's
  limits; before, the child spent from that context's own ledger, outside the
  turn and outside the plan budget.
- A subagent child that stopped on an approval and is resumed outside the
  runner continues the run tree it was started in instead of a fresh root
  budget. The first time a root run starts a child, the tree's allowance is
  kept in the durable store and the root run settles what the tree spent when
  it ends; the resumed child holds what is left, settles what it spent, and
  is refused with `run_not_permitted` once it is spent. A decision on that
  child's approval is refused the same way before anything is recorded. When
  the tree spent from a shared plan budget, the resumed child spends from it
  too, and its decision is refused with `plan_budget_exhausted` once the plan
  budget is spent. A child started under a host-built context outside a turn
  keeps the previous behaviour.

### Fixed

- A subagent child the runner starts again under a run tree the host built
  is no longer held against the run tree its session saved before (which
  refused it once that tree was spent, or charged it to that tree), and that
  saved tree is kept rather than replaced with nothing, so a later turn the
  runner does not start still continues it instead of a fresh root budget.
  The runner's own context is the runner's for the budget routes too, as in
  the PHP reference.
- A subagent child's depth is saved with its binding, and every later turn on
  the child runs at least that deep, a host context included, as in the PHP
  reference: a resumed child can no longer reset its nesting depth.

- A subagent child the runner starts under a run tree the HOST built (the
  parent's turn was handed a `RunContext` the host made) now saves its
  provider/model binding with the child session, as the PHP reference does.
  Before, only a child under a tree this runtime opened was saved, so the
  child's next turn outside the runner ran on the default provider. Only the
  context the runner itself builds and marks counts as the runner's; a
  context the host builds, or binds again, never saves a binding. Pinned by
  harness-plans hpl-0059 and hpl-0060.

- A subagent child's provider and model binding is now kept with the child
  session's durable data, as the PHP reference keeps it, not only on the row
  a paused run stopped on. A new message to a child session, a resume, and a
  turn under a context the host built all run on the binding the runner gave
  the child; before, a new message resolved its provider and model like a
  root run. A child session with saved run tree references and no saved
  binding is refused with `child_binding_missing`, the reference's code,
  before any request. A context the host supplies never replaces the
  binding or run tree a child saved: only the runner's own context writes
  them.
- `MemorySessionStore.withLock()` no longer unrefs its lock-wait timer while
  the wait is awaited. A script whose only pending work was a lock wait used
  to exit with code 13 (unsettled top-level await) instead of getting
  `session_locked`. The timer is cleared as soon as the lock is acquired.

- Every turn on a subagent child session that the runner did not start
  continues the child's run tree and the shared plan budget it was started
  under, not only the empty-prompt resume: a new message sent to the child
  instead of an answer to its approval no longer gets a fresh root budget,
  and is refused with `run_not_permitted` (or `plan_budget_exhausted`) once
  they are spent. The references are now kept on the child session as well as
  on the row it paused on, so a turn with a new message can read them. This
  matches the PHP reference.
- A run context the host hands a turn on such a child (`streamDecision()` or
  `send(child, '', …, context)`) is narrowed to what the run tree and the
  plan budget have left, and charged to them, instead of running on its own
  budget outside both. It is refused only when they are spent. The runner's
  own context for a child it is running is not charged twice.
- Which budgets a turn spends from is decided by one pure function,
  `PlanBudget.route()`, pinned across languages by the harness-plans corpus
  0.1.5 (hpl-0053..hpl-0055).
- The run-tree check before a decision is recorded uses the agent's
  `lock_ttl`, as the turn does, instead of a fixed 300 seconds.

- Agent plans (G-360), the port of the reference's: an agent drafts an ordered
  plan with `plan_draft`, a person approves it, and `session.plan().advance()`
  runs one step per turn. Off unless the new `Plans` service's config sets
  `enabled` and the mode sets `plans: true`. A step is done only when the agent
  called `plan_complete_step` and the turn ended cleanly; a refused approval
  fails its step. Plans are bounded (40 steps, 160-character titles, a first
  update under 8 KB) and refused, never truncated, when over. `plan_updated`
  run steps (contract `g282.3`) are off unless the config's
  `events.run_steps` is on; `plan.updated` events carry ids and counts only.
  See the README's "Agent plans".
- `ToolPolicy` refuses a policy naming a plan tool, in any spelling
  (`RESERVED_TOOLS`).

### Changed

- Approval requests are now written with `arguments_format: 1`, which records
  that the call's arguments were stored as the model sent them.
- An approved call whose request has no `arguments_format` and whose stored
  arguments are empty no longer runs on resume. 0.11.0 and earlier stored a
  call whose arguments arrived as a bare number or boolean as no arguments, so
  the two cannot be told apart. The call's result, recorded with no arguments,
  says the approval expired and asks the model to call the tool again; a new
  call needs a new approval. A `tool_approval_expired` event records it. Approvals pending
  with no arguments when you upgrade expire once, including ones written by the
  previous release.

### Fixed

- An approval has one answer. `recordApproval()` and `recordDecisions()` now
  check that an id is pending and record the decision under the session lock,
  so two decisions racing for the same id (two reviewers, a retry) cannot both
  be recorded: the second is refused with `approval_not_pending`, and one that
  cannot get the lock fails with `session_locked` having recorded nothing.
  Where a thread already holds two decisions for one id, the FIRST one is the
  answer for the resume, the tool scope and the plan alike; before, the resume
  acted on the later one, so a denial followed by an approval ran the call.
  The provider replay view is unchanged.
  **Behaviour change: recording a decision now takes the session lock.** The
  lock is not re-entrant, so a host that calls `recordApproval()` or
  `recordDecisions()` from inside its own `session.lock(...)` callback now waits
  and then fails with `session_locked`, recording nothing. Those two use fixed
  values (300 s TTL, 30 s wait); the new `runtime.decide(session, approvalId,
  approved, reason?)` records the same decision with the session's agent
  `lock_ttl` / `lock_wait`, as `streamDecision()` does.
- Tool call arguments holding values JSON has no form for are unusable, like
  unparsable arguments: `undefined` in a list, a function, a symbol, a bigint,
  or an object that is not a plain object (a `Date`, a `Map`, a class
  instance). Such a call asks nobody and runs nothing, and the model is told.
  These are exactly the values its digest cannot represent. Before, a gated call with
  such a value threw a `TypeError` out of `send()` when its decision event was
  digested, and a `Date` was digested as `{}` while being sent back as text.
  An `undefined` property still reads as an absent one.
- A named agent's `spend_ceiling` now holds across an approval pause. The
  agent's spend so far is stored on the row that asks for the approval
  (`harness_agent_spend`) and counted again when the run resumes, instead of
  starting at nothing. A subagent's child run resumed outside the runner keeps
  the provider and model it was bound to (`harness_binding`) instead of
  resolving them like a root run. Each `send()` with a prompt, and each
  `plan.advance()` without a context, still starts a new run tree; see the
  README.
- Whether a stored call's arguments are unusable is read from the turn being
  resumed only, so an older turn's call with a reused id no longer refuses a
  newer approved call.
- The file store reads a stored `arguments_format` written as anything but
  `1` (for example `1.0`) as no marker, as the PHP and Python harnesses do.
- A decision for an approval the session is not waiting on is refused with
  `approval_not_pending` (G-354). Before, a made-up approval id recorded while a
  call was pending closed that call as "No approval response provided" on
  resume, and one streamed while nothing was pending started a model turn with
  no new user message. `recordApproval()`, `recordDecisions()` and
  `streamDecision()` now refuse an id the run never issued, one already
  answered, or the same id twice in one batch, before anything is written or
  sent. Streaming the same decision again is refused rather than replayed.
- A run's step limit (`max_steps`, and `max_steps_ceiling` for a named agent)
  now covers the whole run, including every pause for an approval and every
  resume after one. A resume (`send(session, '')` or `streamDecision()`) continues the paused run's step count
  instead of starting a new one, also when another worker resumes it from
  persisted state. The count is read from the stored thread rows, so no new
  state is written. A resume that finds the limit already reached still runs
  the calls a person approved, then finishes with `budget_exhausted` without
  another model call. A new prompt starts a new run with its own limit, as
  before.

### Added

- `ToolPolicy`, a host-owned list of approval-gated tools that run without
  asking a person (G-326). Built with `ToolPolicy.fromJson({ revision, allow:
  [{ server?, tool }], honor })`, the same JSON shape as the PHP and Python
  harnesses. Passed per launch as `AgentRuntime({ toolPolicy })` and replaced
  live with `runtime.setToolPolicy()`; the default `ToolPolicy.none()` changes
  nothing. Matching is exact: a padded, look-alike or case-variant name never
  matches, an entry without `server` never matches a tool that has one, and
  there are no wildcards. Confirmation and approval tools, malformed entries,
  and tools declaring `tier: 'destructive'` are refused with
  `tool_policy_refused`. `honor: false` turns the list off. A replacement
  applies from the next call; a call already waiting keeps waiting, and a call
  the old policy allowed is refused if the new one no longer does. The policy
  is never sent to the model and has no tool surface.
- `HarnessTool` takes optional `server` (the MCP server a tool comes from) and
  `tier` (`read_only`, `write`, `destructive`; advisory).
  `ToolRegistry.declarations()` lists what the static tools declare.
- A tool that arrives through `registerProvider` is scoped by the registry:
  its `server` is the provider's registered name, whatever the tool itself
  declares. A provider that forgets to set `server`, or a tool that claims
  another server, can no longer make a provider's tool match an unscoped
  entry or another server's entry. Only a natively registered tool matches an
  entry without `server`. Name a provider (`registerProvider(provider, owner,
  name)`) to give a policy a stable scope to list; an unnamed one is scoped as
  `provider:<n>`. `ToolAuthorizer`'s call-time wrapper now keeps `server` and
  `tier`, which it used to drop.
- Events `tool_auto_allowed`, `tool_approval_asked`, `tool_approved` and
  `tool_denied` for every call to an approval-gated tool, carrying
  `policyRevision`, `server`, `tool`, `argsDigest` (sha256 of the arguments'
  RFC 8785 canonical JSON; never the arguments), `decidedBy` (`policy` or
  `person`) and `harnessAgent`. They are emitted whether or not a policy is
  set, so a listener that expects only `run.*` events will now also see these.
- `canonicalJson()` and `argsDigest()`.
- Modes take optional `provider`, `model` and `agents` keys. Subagent declarations
  take optional `provider`, `model`, `agent` and `inherit_session_model` keys
  (G-321 E). For a run, the order is session > mode > agent > global.
- A subagent may run as another named agent with `agent`, but only when the parent
  mode lists that agent in `agents`. An unlisted agent, or a padded or mixed-case
  id, is refused with `mode_malformed` when the mode is resolved. Ids are never
  trimmed or case-folded.
- `agents.<id>.spend_ceiling` on `AgentRuntime` caps what that agent spends in one
  run tree. It is checked against the same root ledger before each model call. An
  agent over its ceiling is refused with `agent_spend_ceiling_exceeded`. One whose
  provider reported no cost is refused with `budget_unpriced`. A refused subagent
  returns to its parent as `exhausted`. Per-day windows are not implemented.
- Every run event of a named agent carries `harnessAgent`. Legacy events are
  unchanged.
- `RunLedger.agentCostUsd()`, `agentUnmeteredRuns()` and `assertAgentCeiling()`.
  `RunLedger.recordCost()` takes an optional agent label.
- `AgentRuntime.modeOf()`, `RunContext.binding`, a `binding` argument on
  `forChild()`, and the `ModelBinding`, `ModelChoice` and `HarnessAgentTag` types.
- `isAgentId()`.

### Changed (breaking)

- **A subagent's child no longer inherits the parent session's provider/model
  override.** `SubagentRunner` used to copy the parent's override onto the child
  session. For a legacy parent with no override, it wrote `'unknown'`. It now
  writes nothing. The child resolves its own subagent declaration first, then its
  own mode, then the parent's mode (same agent only), then the agent default and
  the global default. To keep the old behaviour, set
  `inherit_session_model: true` on the declaration. The child session's own
  override slot is not read for a subagent run, so a value an earlier version
  wrote there has no effect. The rule is the same in PHP and Python and is pinned
  by the `harness-agent-binding` corpus.

## 0.11.2

Security-only release on the 0.11 line. Nothing else from main is included.

### Changed

- Approval requests are now written with `arguments_format: 1`, which records
  that the call's arguments were stored as the model sent them.
- An approved call whose request has no `arguments_format` and whose stored
  arguments are empty no longer runs on resume. 0.11.0 stored a call whose
  arguments arrived as a bare number or boolean as no arguments, so the two
  cannot be told apart. The call's result, recorded with no arguments, says
  the approval expired and asks the model to call the tool again; a new call
  needs a new approval. A `tool_approval_expired` event (`runId`,
  `sessionKey`, `tool`, `argsDigest` of no arguments, `decidedBy: 'policy'`;
  never the arguments) records it. Only the turn being resumed decides this,
  so a call id reused across turns does not expire a marked approval.

### Upgrading

- **One-time expiry.** Approvals that are pending when you upgrade, on a call
  stored with no arguments, expire once instead of running, including ones
  written by 0.11.1, which wrote no marker either. The model is told to call
  the tool again, and the person is asked again. Approvals on calls with
  named arguments are not affected. Resuming a thread written by another
  language's harness is not supported for this check.
- If your code writes approval request rows itself, write
  `arguments_format: 1` only for calls stored exactly as the model sent them.

## 0.11.1

### Fixed

- A run's step limit (`max_steps`, and `max_steps_ceiling` for a named agent)
  now covers the whole run, including every pause for an approval and every
  resume after one. A resume (`send(session, '')` or `streamDecision()`)
  continues the paused run's step count instead of starting a new one, also
  when another worker resumes it from persisted state. The count is read from
  the stored thread rows, so no new state is written. A resume that finds the
  limit already reached still runs the calls a person approved, then finishes
  with `budget_exhausted` without another model call. A new prompt starts a
  new run with its own limit, as before.
- A tool call whose arguments are not a JSON object (unparsed text, a list,
  a number, a map with integer-like keys, or values a follow-up request could
  not carry) is recorded with no arguments, asks nobody to approve it, is not
  passed to a call policy or handler, and is answered to the model as not run.
  The same check applies when an approved call is resumed from stored rows.
  An empty string, `0`, `null`, `[]` and `{}` all mean no arguments, and a
  tool taking none still runs with `{}`. JSON text naming values is read as
  those values.

### Upgrading

- Resolve or deny any approval that is pending when you upgrade if its tool
  call had arguments that were not an object. 0.11.0 stored some of those
  calls (a bare number or boolean) as `{}`, so an approved resume runs them
  with no arguments.

No breaking changes.

## 0.11.0

### Added

- `AgentRuntime.streamSteps()` yields tagged run steps (contract `g282.1`,
  byte-compatible with prism-harness PHP 0.25.0) and returns the run's response.
- `AgentRuntime.streamDecision()` records approval decisions under the session
  lock, then streams the resumed run under the parked run's id, continuing its
  `seq` and emitting `run.resumed`.
- `recordDecisions()`, `RunContext.metadata()`, `withTurn()` and `withActor()`.
- Opt-in `run.stepped` events with `stepEvents: true`.

`send()`, `sendStructured()` and `recordApproval()` behave as before.

## 0.10.3

### Fixed

- Read Windows and macOS lock identity with asynchronous native queries, so a
  first lock or retry keeps the host event loop responsive.
- Share one pending sample across concurrent locks while preserving successful
  identity caching, the sixty-second retry and all metadata validation.
- Export optional `warmLockIdentity()` and the read-only
  `fileLockIdentityStatus()` helper for host startup and diagnostics.

No breaking changes.

## 0.10.2

### Fixed

- Stream file-session log records instead of decoding the entire file as one
  string, so logs larger than the JavaScript string limit can reopen.
- Count completed log lines without loading message history. A disposable small
  sidecar avoids repeated scans after a new store instance; changed file metadata
  or an invalid index causes a streaming rebuild.
- Preserve incomplete-tail recovery, corrupt-record errors, cross-process locks
  and the existing sync on each durable append.

No breaking changes.

## 0.10.0

### Fixed

- Fixed: on Windows, a lock cleanup refused because another worker still holds the file now waits instead of crashing.

### Breaking changes

- Caller-supplied tool names now narrow a legacy session's mode list by default,
  through both `send()` and `sendStructured()`. A name outside the ceiling is
  refused before a model call; an empty list offers no tools.
- To retain legacy replacement behavior, explicitly set
  `allowLegacyToolListWidening: true`. The authorizer still applies, and named
  agents always enforce their mode ceiling.
- Remove an explicit `narrowRequestedTools: false`; it now raises a migration
  error naming the new opt-out. Explicit `true` remains valid, but combining it
  with the widening opt-out is a configuration error.

Legacy duplicate-registration and address defaults retain their existing behavior.
