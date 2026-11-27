# Changelog

## Unreleased

### Fixed: ended approvals in steps that repeat a call id, and cleanup on success (G-387 round 13)

- An approval a runner start ended reads ended when ANY result row of its
  request's own turn carries the start's mark naming it. It was read only
  from the first result row carrying its call id, so in a step where an
  ungated call (or a call with unusable arguments) shared that id and
  already had a result, the plan could count a person's approval of a call
  that never ran as approved work. Pinned by the harness-thread-rows corpus.
- Every decision recorded on a step whose calls repeat a tool call id is
  recorded as a denial with the repeated-id reason, on every decision path
  (`recordApproval()`, `recordDecisions()`, `decide()`, `streamDecision()`),
  as the PHP reference records it. A `decision_recorded` step still reports
  the decision as given, as the reference streams it. Pinned by the harness-approval-pairing corpus.
- A call's result is paired with it by its position in the step, never by
  its tool call id: an ungated call's result no longer makes a gated call
  sharing its id look finished, nor takes its approval in the plan's view of
  the pending calls.
- A run releases its holds BEFORE `run.finished`, as it already did before
  `run.failed`, and a delegation's step-count settle is tried twice, so one
  failed store read at the end of a completed delegation no longer leaves the
  child's count over-charged. A completed start is never undone by its
  cleanup.
- `sendStructured()` on a turn that stops before any document (it parks on an
  approval, or a budget stops it) returns the structured response with
  `structured: null` and its pending approvals; the field was missing.
- With the G-392 property tests (now in the suite, with a nightly long run):
  every gated call in a step gets exactly one recorded outcome, a result
  belongs to the call at its position, and a request pairs only with a call
  that has no result of its own, so an ungated call never takes an approval
  or a repeated-id denial.
- A decision applies only to the step whose approvals were requested: the
  decision checks, the repeated-id rule, the resume and the plan readers all
  take that step from one selector (the last assistant row, unless a user row
  follows it). A row recorded after the park can no longer make one of them
  read another step: a decision on a step a later assistant or user row
  superseded is refused on every path, and the resume acts on nothing of it.
- A step's results are read from every result row after it, up to the next
  assistant row, so a row of another kind between them (a system row, an
  empty result row, a host's own row) can no longer hide a call's result and
  let a resume act on it again.
- A runner start that fails after its records are written but before its run
  begins (a plan watch for a streamed turn) now ends its own view and is
  undone, as every later failure is.

### Changed: one gate for a runner start (G-387)

- A runner start (`SubagentRunner.run()`, and the subagent tool that calls it)
  is decided by ONE pure gate, `RunStarts.gate()`, pinned across languages by
  the harness-plans corpus 0.1.10 (`start_gate`, hpl-0084..hpl-0089). Every
  fact is read first and nothing is written before the gate answers: the task,
  depth, the cancel and time of the context the child would run under and of
  the context the tool was wired with, the steps the start's allowance would
  give, cost (a context's cost budget and the child agent's `spend_ceiling`,
  spent or unmeterable), a turn on the child still running, an approval on it
  a person can still answer, and what the run tree and plan would grant. The
  first stop in that order is the reason the parent's model reads, in the
  harness's own words.
- A refused start writes nothing: not the child's binding, depth, ceiling,
  count, run tree refs or mode, no run state or run history, no tree hold.
- On start it commits in order: the run tree hold, in a conditional locked
  write refused if the tree was spent since the gate read it; the child's
  records, in one locked write that is refused if a turn on the child claimed
  its count since (the tree hold is then released); then the run. Once the run
  has begun, a later stop only stops model steps; the start is never refused
  after its records were written.
- The child agent's `spend_ceiling` now refuses the start (`exhausted`) before
  the child's run begins, rather than failing the child's run on its first
  step; no `run.started`/`run.failed` event is emitted for that child.
- A subagent task that is not text now reaches the parent's model as "the
  task is not text" (it was reported as empty), and a cancelled run's
  refusal no longer repeats the host's cancel reason.

### Fixed: a runner start after its records are written (G-387 round 8)

- A runner start is narrowed from the steps its context had when the gate
  read them: a step spent on a shared context after that read is counted
  against the child, never added back on top.
- The child's mode is saved in the same locked write as its other records.
  A start that fails after that write and before its task row is recorded
  (the run cannot begin, the tool offer or a plan read throws) restores the
  child's records and mode as they were, only if nobody has written them
  since (otherwise only the start's own count hold is settled), and releases
  its holds (see round 9: this is now housekeeping only).
- That write also re-checks, under the child's lock, for an approval a person
  could answer that parked since the gate read the child: the start is then
  refused (`denied`, waiting on a person) with nothing written.
- Whether a pending approval could be answered is decided by the known
  reasons only (no binding, too deep). A store read that fails there now
  fails the start (`failed`, nothing written) instead of counting the
  approval unanswerable.
- A runner start's tree hold is released only by the start that took it.
- Everything the parent's model reads of a child is harness-written and chosen
  by code: the gate's reasons, a fixed reason for each stop after the start
  (cancelled, steps, cost, time), a fixed reason for a child agent's spend
  ceiling or an unmetered cost, and otherwise the class name and harness code.
  A host's cancel reason no longer reaches the parent's tool result or a later
  provider request; `AgentResponse.stoppedBecause` keeps it for the host. An
  error's `code` is read once, guarded, and a refusal's outcome comes from its
  code, never from the thrown object. A task lease that cannot be extended
  names the stop, not the host's cancel reason.
- `RunBudget` (and `RunBudget.fromConfig`) refuses a cost or time limit that
  is `NaN`, infinite or negative (`TypeError`); `fromConfig` used to read such
  a value as no limit. A runner start counts a non-finite limit as spent.

### Fixed: a runner start's child records are provisional until its task row (G-387 round 9)

- Everything a runner start saves on its child (binding, depth, step ceiling
  and count, run tree and plan references, mode) carries the start's run id
  and reads as ABSENT until the start's task row, the user row recorded under
  that run id, is in the child's thread. Until then a turn on the child the
  runner did not start, a decision on its approvals (`recordApproval`,
  `decide`, `streamDecision`), the start gate's answerability read and the
  delegation step count all fail closed (`child_binding_missing`, no
  allowance), and a binding on the child's paused row is not used either.
  Confirmation is read from the task row, never from another write: a start
  that fails before its task row leaves nothing answerable that was not, even
  when the store also fails while it cleans up. The round-8 restore is now
  housekeeping and its failure changes nothing; a count hold it could not
  release expires after `lock_ttl`. Records with no mark (written before this
  change) behave as before; a mark that is not a run id fails closed.
- The child's run state and run history do the same: until the task row,
  `session.run()` is the run the child had before the start (a parked or
  finished run stays current after a failed start), and the attempted run has
  no run tree entry.
- A runner start acts on no recorded decision. Its task row ends every call
  the child was still waiting on, recorded as not run in the same write,
  whatever decision is recorded on it (one recorded while an earlier binding
  was alive included).
- A run's failure handler cleans up before it reads anything from the error,
  reads it only guarded, and emits `run.failed` even when the failed-run write
  itself fails. An error whose prototype or `code` cannot be read is framed as
  a failure by class name.
- Only the codes this package defines are reported as harness codes in a
  framed failure, a run row or `run.failed`; a `HarnessError` built with any
  other code is reported as `HarnessError error`.
- Configured limits stay strict: a `max_cost_usd`, `max_seconds` or
  `max_steps` that is present but not a finite non-negative number (`'NAN'`,
  `'INF'`, `'-INF'`, `'0,50'`, a quoted number) is a `TypeError`, never "no
  limit", in `RunBudget.fromConfig`, the runtime config, modes and plan
  budgets (now pinned by tests).

### Fixed: one ended-mark shape, and run events that cannot break a run (G-387 round 12)

- The row a runner start writes when it ends the calls its child was parked
  on now has the PHP reference's shape, so a thread reads the same in every
  language: one result row holding only the ended calls, each with the fixed
  result `Not run: the delegation this call was proposed under ended.`, and
  marked beside (never inside) the results with
  `harness_ended_by_start: {approval_ids: [...]}`, the calls' own approvals in
  call order. Only that mark makes a call read ended: the approval named by
  it, on the first result of its call after its request row. Pinned by the
  harness-thread-rows corpus (thr-0014). The per-entry mark of round 11 was
  never released and is no longer written or read.
- A throwing event listener can no longer break a run: a listener's failure
  is reported by its class name only (the warning no longer carries the
  listener's own text), by a conversion that cannot throw, whatever the
  listener threw.
- Every run event is emitted inside the run's failure handling: a throw from
  emitting `run.started` (a host's own events object may throw anything)
  fails the run with the class name only, undoes a runner start that had not
  recorded its task, releases its holds, emits `run.failed` and reaches the
  caller. No run is left `running`, and no child stays busy until `lock_ttl`.
- On a failed run, the run tree hold and the delegation hold are released
  before `run.failed` is emitted, so a listener never sees a hold that is
  about to be released.
- A tool that throws a value that cannot be converted to text fails as a
  tool result naming its class, instead of failing the run.
- The child-state invariant matrix covers more phases (inside `run.started`
  and `tool_denied` listeners, a failed start measured inside `run.failed`),
  more readers (provisional, plans, the plan's run, the paused row, busy, the
  resumed tree, the tool policy) and actions (a resume, a second start, a
  decision or turn on the start's own Session, and that Session used from
  outside its run).

### Fixed: a runner start's own reads, and how an ended call is recognised (G-387 round 11)

- A call a runner start's task row ended is recognised only by a mark the
  harness writes on the row (round 12: `harness_ended_by_start`, below),
  never by the "Not run" text: a tool whose result
  is exactly that text is an ordinary result, and its approval stays
  approved. The mark applies only to the approval it names, asked for before
  it, so a provider that reuses tool call ids across turns can no longer make
  a later approval (a plan draft, a plan step) read as ended.
- Inside a runner start's own run, the child's Session (and the read-only
  view a tool provider is given) reads what the start committed: its mode,
  binding, run tree, depth, ceiling and run. The plan context of the start
  follows its mode and run. Every other Session object on the child, and any
  decision or turn, even one entered on the start's own Session, keeps the
  provisional rule until the task row.
- A start's task is marked recorded the moment its row is written, before
  any event for the calls it ended: a failure after that never undoes it.
- Warnings about a failed budget, plan or step-sequence update name the
  error's class by the named-class rule only, never a path or its message.

### Fixed: provisional child records, round 10 (G-387)

- A child whose records a runner start left provisional reads as the run
  tree and shared plan budget it was last CONFIRMED with, kept beside the
  provisional records, never as none. A later start outside a parent turn (a
  direct `SubagentRunner.run()`, or the subagent tool called with the
  context it was wired with) keeps them, so every later turn on the child
  stays in the root's run tree, bounded by it, and spends from the same plan
  budget. A child with nothing confirmed before still reads as having none.
- A decision a person recorded that no resume has acted on yet refuses a
  runner start exactly as an undecided approval does (`denied`, waiting on a
  person, nothing written), at the gate and in the locked re-check of the
  child's records: a model re-delegating can no longer end a call a person
  decided. A resume then acts on the decision as before. A decision nobody
  can act on any more (its binding gone, too deep) still does not block.
- A host `ToolProvider`, factory or offer policy may read the child's session
  state (`run()`, `childBinding()`, `childRefs()`, `childDepth()`,
  `childCeiling()`, `mode()`) while a runner start resolves its tools: the
  read that confirms a provisional record is allowed in the read-only
  resolution view, which still refuses every write.
- The `tool_denied` events (and run steps) for the calls a runner start ends
  as not run are emitted only after those rows are recorded with its task
  row; a store failure on that write reports none.
- A call a runner start ended as not run no longer counts as approved work in
  the child's plan: its approval reads as superseded, whatever decision was
  recorded on it.
- Pinned by tests (unchanged behaviour): the mode a start wrote reads as the
  mode before it (or none) until its task row; a thrown value that is not an
  `Error` fails the run with the same cleanup and still reaches the caller;
  a framed failure, `run.failed` and the run row name an identifier-shaped
  class only, never a path, file or line (an anonymous class, or a class or
  constructor whose name carries a path, reads as `Error`).

### Changed: no exception text in run rows or `run.failed` (G-387 round 8)

- A failed run's row (`session.run().failure`) and its `run.failed` event now
  carry a fixed description with the error's class name and harness code,
  `The run failed (HarnessError model_step_failed).`, never the exception's
  message. This package has no content-capture setting, so exception text is
  never written there. A structured run's schema violation still records its
  code alone. The exception is still thrown to the caller.

## 0.12.0

Includes every fix released on the 0.11 line in 0.11.1 and 0.11.2.

### Changed: one authority for a run's step allowance (G-387)

- How many model steps a run may take is decided once, when it begins, by one
  pure function, `RunBudgets.allowance()`, pinned across languages by the
  harness-plans corpus 0.1.9 (`step_allowance`, hpl-0072..hpl-0083), and
  recorded beside the run's tree membership. The run is narrowed to it once;
  the step gate, the row it parks on, the subagent runner's pre-check, the
  run-tree and plan holds and both decision checks read that record or that
  budget.
- A subagent child's ceiling is the runner's grant (its declared `max_steps`
  within what its parent had left, and `max_steps_ceiling`), saved with the
  child's binding when the runner starts it. It is the delegation's budget for
  its whole life: every turn on the child the runner did not start (a resume
  with no context, a metadata context or a host context, a new message, an
  empty send) claims from it, in one locked write against a durable count of
  the steps the child has used since the runner last started it, so two turns
  running at once never both spend the same remainder. A turn is charged
  what it took when it ends, parks or fails, even when it ran past the lock
  TTL; a hold whose turn never settles is charged in full once the lock TTL
  has passed, and replaced by what the turn took if it settles after all,
  never counted twice. A count, or any part of it, that is not a
  non-negative whole number fails closed (no model step) and is never
  repaired; only a runner start writes a fresh one. A runner start begins
  the count again, and is refused while a turn on the child is still
  running. A host context can only lower what is left, never raise it.
- A subagent child session fails closed when it has no saved ceiling, or
  one that is not a non-negative whole number: its approved calls still run,
  it takes no model step, and a new message to it is refused before anything
  is recorded. A session is a child when its address is one the runner
  builds (`::sub::` in its scope), including a child a released version
  parked, or when it has saved run tree references, a saved binding, or a
  parked row that names a run tree. `::sub::` in a scope is reserved for
  subagent children: a host scope containing it is treated as a child. Any
  other session runs on its mode's
  `max_steps`, or on a host context's budget.
- A step budget that is not a whole, non-negative number is refused:
  `new RunBudget(NaN)`, `Infinity`, `undefined`, `-1` or `1.5`, and
  `RunBudget.fromConfig({ max_steps: 1.5 })`, throw `TypeError`; a run
  context whose budget is not a number (a host-built object) has no step
  left.
- The row a run parks on now records `harness_steps_used`: the steps the run
  had used, its subagent children's included. Its resume continues from that
  count, never from fewer than the model steps the session's own rows show; a
  row written before the field existed is counted from its rows, and a value
  that is not a non-negative whole number fails closed (no model step), as
  does a missing value on a child session that saved its ceiling.
  `harness_steps_left` is still written (it now includes plan narrowing), but
  nothing reads it to decide a budget.
- `max_steps_ceiling` bounds a run's whole count under every context.
- A root run under a host-supplied run context runs on that context's budget,
  not its mode's `max_steps` (still capped by `max_steps_ceiling`).
- A spent budget, steps or cost (the run's own, its run tree's or a shared
  plan's), never refuses a decision and never refuses a resume: acting on an
  approved call makes no provider call. The decision is recorded, the resume
  runs the approved call and finishes with `budget_exhausted` without a model
  call, on `send()`, `streamDecision()` and `sendStructured()` alike. An
  approved subagent call on such a run is acted on: the call runs, and its
  result is the runner's "did not run" refusal (the step budget is
  exhausted), so its child takes no step. This replaces the
  `run_not_permitted` / `plan_budget_exhausted` decision refusals described
  below.
- A decision its resume could not act on is refused before anything is
  written, on every decision path (`recordApproval()`, `recordDecisions()`,
  `decide()`, `streamDecision()`): a subagent child whose binding is missing
  (`child_binding_missing`), a run nested too deep, and a supplied run context
  whose ledger was cancelled or whose time is spent (`run_not_permitted`; a
  batch of denials only is not refused for spent time, as recording a denial
  runs nothing, but is for a cancelled context); and
  a plan step handed a run tree of its own (`plan.budget_context`). A resume
  under a cancelled run context, or one whose time is spent, acts on none of
  the recorded decisions (that a denial-only batch started, too): they stay
  recorded for a later resume that can act on them. A new prompt under such a
  context, on a session parked on approvals, is refused (`run_not_permitted`)
  before anything is recorded, so it cannot end the parked turn; a runner
  start refused that way leaves the child's saved records unchanged.
- The subagent runner refuses (`denied`, a framed result, nothing written to
  the child) a task that is not text, or that consists only of the blank
  codepoints U+0009-U+000D, U+0020, U+0085, U+00A0, U+1680, U+2000-U+200D,
  U+2028, U+2029, U+202F, U+205F, U+2060, U+3000 and U+FEFF (the same set in
  every port); a delegation into a child that is waiting on a person to
  answer an approval that person could still answer; and one into a child
  on which a turn is still running. The subagent tool checks the host
  context it was wired with first: a cancelled one, or one with nothing
  left (an unenforceable cost cap included), returns the framed `cancelled`
  or `exhausted` result before any child starts, inside a turn too. Every
  failure while it starts or runs the child is framed as `failed`, naming
  the error's class (an identifier, else `Error`) and a HarnessError's code,
  never its message.
- A host-supplied `RunContext` belongs to one session's turn: its ledger is
  the account every run under it spends from, so one context shared across
  sessions charges each child the steps the others took on it.
- A turn with a NEW prompt (it resumes no decisions) that is left no model
  step is refused before anything is recorded, as before:
  `plan_budget_exhausted` when the shared plan budget is what has nothing left
  for it (also a fresh `advance()` whose remainder a live turn holds), and
  `run_not_permitted` otherwise (its own saved ceiling, its run tree or a cost
  budget spent).

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
