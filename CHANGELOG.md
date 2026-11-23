# Changelog

## Unreleased

### Added

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
