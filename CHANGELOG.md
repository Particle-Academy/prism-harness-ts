# Changelog

## Unreleased

### Added

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
