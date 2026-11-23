# Changelog

## Unreleased

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
