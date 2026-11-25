import { PLAN_BUDGET_EXHAUSTED } from './budget.js';
import { PlanIds } from './ids.js';
import { PlanLimits } from './limits.js';
import { PlanRefused } from './refusal.js';
import { PlanText, acceptablePlanText, planTextLength, wellFormed } from './text.js';
import { firstEventBytes } from './view.js';

/** One step as stored. Every field is always present: a dropped key is different bytes in another language. */
export interface PlanStep {
  id: string;
  position: number;
  title: string;
  instruction: string | null;
  owner: 'agent' | 'human';
  interaction: 'input' | 'approve' | null;
  question: string | null;
  options: string[];
  state: string;
  approvalIds: string[];
  approvedIds: string[];
  supersededApprovalIds: string[];
  pauseDigest: string | null;
  claim: { worker: string; until: number } | null;
  attempts: number;
  pauses: number;
  lastSettle: string | null;
  report: { summary: string } | null;
  summary: string | null;
  answer: { text: string; choice: string | null; actor: string; at: string } | null;
  completedBy: string | null;
  failure: string | null;
  retryOf: string | null;
}

export interface PendingChange {
  approvalId: string | null;
  baseRevision: number;
  reason: string;
  digest: string;
  order: string[];
  newSteps: PlanStep[];
}

/** A plan as stored. The field order is the reference's, so the stored JSON is the same bytes. */
export interface Plan {
  id: string;
  anchorRunId: string | null;
  revision: number;
  status: string;
  goal: string;
  holdingApprovalId: string | null;
  pendingChange: PendingChange | null;
  blockedReason: string | null;
  draftDigest: string;
  counters: { advances: number; changes: number; settledByTool: number; settledWithoutTool: number };
  lastRunId: string | null;
  createdAt: string;
  updatedAt: string;
  lastAdvancedAt: string | null;
  steps: PlanStep[];
}

/** A pending approval at the transcript tail: its id, the tool it was asked for, and the digest of its stored arguments. */
export interface PendingCall {
  id: string;
  name: string;
  digest: string;
}

export interface DraftInput {
  title: string;
  instruction: string | null;
  owner: 'agent' | 'human';
  interaction: 'input' | 'approve' | null;
  question: string | null;
  options: string[];
}

export interface CheckedDraft {
  goal: string;
  steps: DraftInput[];
}

/** `['run', stepId]`, `['waiting', stepId]`, `['blocked', reason]`, `['done', null]` or `['not_active', status]`. */
export type PrepareOutcome = [string, string | null];

type Map_ = Record<string, unknown>;

// -- PHP array semantics, for input decoded from JSON --------------------------
//
// The reference reads the model's arguments as PHP arrays, where a JSON object
// and a JSON list are both "arrays" and `{}` and `[]` are the same value. These
// reproduce exactly the checks it makes, so a malformed argument is refused
// with the same code in both languages.

function isPlainObject(value: unknown): value is Map_ {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** A PHP list: a JS array, or an object whose keys are exactly "0".."n-1". */
function listLike(value: Map_): boolean {
  const keys = Object.keys(value);

  return keys.every((key, index) => key === String(index));
}

/** The values of anything PHP would call a list (`[]` and `{}` included), or null. */
function asList(value: unknown): unknown[] | null {
  if (Array.isArray(value)) return value;
  if (isPlainObject(value) && listLike(value)) return Object.values(value);

  return null;
}

/** Anything PHP would accept as a map that is not a non-empty list, or null. */
function asMap(value: unknown): Map_ | null {
  if (Array.isArray(value)) return value.length === 0 ? {} : null;
  if (isPlainObject(value)) return Object.keys(value).length > 0 && listLike(value) ? null : value;

  return null;
}

/** Keys of an array-ish value, as PHP's array_keys would list them. */
function keysOf(value: unknown): string[] {
  if (Array.isArray(value)) return value.map((_, index) => String(index));
  if (isPlainObject(value)) return Object.keys(value);

  return [];
}

function has(value: Map_, key: string): boolean {
  return Object.hasOwn(value, key);
}

/** `$value[$key] ?? null`. */
function get(value: Map_, key: string): unknown {
  return value[key] ?? null;
}

function diff(values: readonly string[], minus: readonly string[]): string[] {
  return values.filter((value) => !minus.includes(value));
}

function same(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

function copy<T>(value: T): T {
  return structuredClone(value);
}

/** PHP's trim(): space, tab, newline, carriage return, NUL and vertical tab. */
export function phpTrim(value: string): string {
  // Index scans, not a regular expression: an anchored alternation over the
  // same class backtracks quadratically on a long run of trimmed characters
  // that does not reach the end, and the actor comes from the host's request.
  const trimmed = ' \t\n\r\0\x0B';
  let start = 0;
  let end = value.length;
  while (start < end && trimmed.includes(value[start]!)) start++;
  while (end > start && trimmed.includes(value[end - 1]!)) end--;

  return value.slice(start, end);
}

/**
 * Every transition a plan can make, as pure functions over the stored plan
 * (G-360). No I/O, no clock, no randomness: the caller passes the time, the
 * id generator and the facts read from the thread, so the same inputs give the
 * same plan in PHP, TypeScript and Python. The `harness-plans` parity corpus
 * drives these functions directly.
 *
 * A PLAN GRANTS NOTHING. Nothing here runs a tool, decides an approval or
 * touches the task list. Approvals are core approvals, decided through
 * `recordApproval()` / `streamDecision()`; this only reads which ids are
 * pending or answered, and those ids come from the persisted thread, never
 * from a tool argument.
 *
 * Every function takes the plan by value: the argument is never changed.
 * Refusals throw `PlanRefused` with a constant message.
 */
export const PlanMachine = {
  DRAFT: 'plan_draft',
  COMPLETE: 'plan_complete_step',
  PAUSE: 'plan_pause_for_person',
  CHANGE: 'plan_propose_change',
  /** Every plan tool name. A pending approval for one of these never holds an agent step. */
  TOOLS: ['plan_draft', 'plan_complete_step', 'plan_pause_for_person', 'plan_propose_change'] as readonly string[],
  LIVE: ['proposed', 'active', 'waiting', 'blocked'] as readonly string[],
  TERMINAL: ['done', 'failed', 'skipped'] as readonly string[],

  // -- Drafting ---------------------------------------------------------------

  /** Check a draft at the tool boundary. Refused, never truncated. */
  checkDraft(args: unknown, limits: PlanLimits, authorized = true): CheckedDraft {
    if (!authorized) {
      // The host's call policy refused this call: nothing is checked,
      // recorded or proposed, and the live-plan slot stays free.
      throw PlanRefused.because('plan.not_authorized');
    }
    const map = isPlainObject(args) || Array.isArray(args) ? args : null;
    if (map === null || Array.isArray(map) || !has(map, 'goal') || !has(map, 'steps') || diff(keysOf(map), ['goal', 'steps']).length > 0) {
      throw PlanRefused.because('plan.shape');
    }
    const steps = asList(map.steps);
    if (steps === null) throw PlanRefused.because('plan.shape');
    if (!acceptablePlanText(map.goal, PlanText.GOAL)) throw PlanRefused.because('plan.text');
    if (steps.length < 1 || steps.length > limits.maxSteps) throw PlanRefused.because('plan.bounds');

    return { goal: map.goal, steps: steps.map((step) => PlanMachine.checkStepInput(step)) };
  },

  /**
   * One step as the model wrote it: `{title, instruction, owner, interaction?, question?, options?}`.
   * A person's step is a question (`interaction: input`); a person's approval
   * mid-plan is a core approval, raised with plan_pause_for_person.
   */
  checkStepInput(input: unknown): DraftInput {
    const keys = ['title', 'instruction', 'owner', 'interaction', 'question', 'options'];
    const step = asMap(input);
    if (step === null || diff(keysOf(step), keys).length > 0) throw PlanRefused.because('plan.shape');
    const owner = get(step, 'owner');
    const interaction = get(step, 'interaction');
    const question = get(step, 'question');
    const rawOptions = get(step, 'options');
    const instruction = get(step, 'instruction');
    if (owner === 'agent') {
      if (interaction !== null || question !== null || rawOptions !== null) throw PlanRefused.because('plan.shape');
    } else if (owner === 'human') {
      if (interaction !== 'input' || question === null) throw PlanRefused.because('plan.shape');
    } else {
      throw PlanRefused.because('plan.shape');
    }
    const options = rawOptions === null ? null : asList(rawOptions);
    if (rawOptions !== null && (options === null || options.length > PlanText.OPTIONS)) throw PlanRefused.because('plan.shape');
    if (!acceptablePlanText(get(step, 'title'), PlanText.TITLE)) throw PlanRefused.because('plan.text');
    if (owner === 'agent' && !acceptablePlanText(instruction, PlanText.INSTRUCTION)) throw PlanRefused.because('plan.text');
    if (owner === 'human' && instruction !== null && !acceptablePlanText(instruction, PlanText.INSTRUCTION)) throw PlanRefused.because('plan.text');
    if (question !== null && !acceptablePlanText(question, PlanText.QUESTION)) throw PlanRefused.because('plan.text');
    for (const option of options ?? []) {
      if (!acceptablePlanText(option, PlanText.OPTION)) throw PlanRefused.because('plan.text');
    }

    return {
      title: step.title as string,
      instruction: instruction as string | null,
      owner,
      interaction: owner === 'human' ? 'input' : null,
      question: question as string | null,
      options: (options ?? []) as string[],
    };
  },

  /** A new plan from a checked draft: `proposed`, revision 1. */
  propose(draft: CheckedDraft, digest: string, ids: PlanIds, now: number, runId: string | null, limits: PlanLimits): Plan {
    let plan: Plan = {
      id: ids.plan(),
      anchorRunId: runId,
      revision: 1,
      status: 'proposed',
      goal: draft.goal,
      holdingApprovalId: null,
      pendingChange: null,
      blockedReason: null,
      draftDigest: digest,
      counters: { advances: 0, changes: 0, settledByTool: 0, settledWithoutTool: 0 },
      lastRunId: runId,
      createdAt: iso(now),
      updatedAt: iso(now),
      lastAdvancedAt: null,
      steps: [],
    };
    for (const input of draft.steps) plan.steps.push(newStep(ids.step(), input));
    plan = renumber(plan);
    if (firstEventBytes(plan) > limits.eventMaxBytes) throw PlanRefused.because('plan.size');

    return plan;
  },

  /** The person approved the draft: the approved call ran with its stored arguments, and they must be the arguments that were proposed. */
  activate(original: Plan, digest: string, now: number, runId: string | null): Plan {
    const plan = copy(original);
    if (plan.status !== 'proposed' || plan.draftDigest !== digest) throw PlanRefused.because('plan.not_waiting');
    plan.status = 'active';
    plan.holdingApprovalId = null;

    return written(plan, now, runId);
  },

  // -- Reading the thread: held, superseded, moved, decided ---------------------

  /**
   * Bring the plan in line with the thread (G-360 §1.4).
   *
   * `pending` lists every approval pending at the transcript tail, with the
   * tool it was asked for and the digest of its stored arguments. `decided`
   * maps every answered approval id to whether it was approved. Both are read
   * from the persisted thread; nothing here accepts an id from a tool argument
   * or from model text.
   *
   * @param newTurn a turn with a new user message is starting: every pending approval is about to end
   */
  reconcile(original: Plan, pending: readonly PendingCall[], decided: ReadonlyMap<string, boolean>, now: number, runId: string | null, newTurn = false): Plan {
    if (!PlanMachine.LIVE.includes(original.status)) return original;
    let plan = copy(original);
    const pendingIds = pending.map((entry) => entry.id);

    const known = boundIds(plan);
    const find = (name: string, digest: string | null): string | null => {
      for (const entry of pending) {
        if (entry.name === name && entry.digest === digest && !known.includes(entry.id)) {
          known.push(entry.id);

          return entry.id;
        }
      }

      return null;
    };

    // The draft.
    if (plan.status === 'proposed') {
      plan.holdingApprovalId ??= find(PlanMachine.DRAFT, plan.draftDigest);
      const holding = plan.holdingApprovalId;
      if (holding === null) {
        plan = reject(plan, 'approval_withdrawn');
      } else if (decided.has(holding)) {
        if (decided.get(holding) === false) plan = reject(plan, null);
      } else if (!pendingIds.includes(holding)) {
        plan = reject(plan, 'approval_withdrawn');
      }
    }

    // A proposed change.
    if (plan.pendingChange !== null) {
      const change = plan.pendingChange;
      change.approvalId ??= find(PlanMachine.CHANGE, change.digest);
      const id = change.approvalId;
      if (id === null || (decided.has(id) && decided.get(id) === false) || (!decided.has(id) && !pendingIds.includes(id))) {
        plan.pendingChange = null;
        if (plan.status === 'waiting') plan.status = 'active';
      }
    }

    // A person's approval step, raised with plan_pause_for_person.
    const snapshot = copy(plan.steps);
    snapshot.forEach((original, i) => {
      if (original.owner !== 'human' || original.interaction !== 'approve' || original.state !== 'held') return;
      let step = copy(original);
      if (step.approvalIds.length === 0) {
        const bound = find(PlanMachine.PAUSE, step.pauseDigest);
        if (bound === null) {
          plan.steps[i] = closed(step, 'failed', 'host', 'approval_withdrawn');
          plan = block(plan, 'approval_withdrawn');

          return;
        }
        step.approvalIds = [bound];
        plan.steps[i] = copy(step);
      }
      const id = step.approvalIds[0]!;
      if (decided.has(id)) {
        const approved = decided.get(id) === true;
        step.approvedIds = approved ? [id] : [];
        plan.steps[i] = closed(step, approved ? 'done' : 'skipped', 'person', approved ? null : 'declined');
      } else if (!pendingIds.includes(id)) {
        step = { ...step, supersededApprovalIds: [...step.supersededApprovalIds, id], approvalIds: [] };
        plan.steps[i] = closed(step, 'failed', 'host', 'approval_withdrawn');
        plan = block(plan, 'approval_withdrawn');
      }
    });

    // The agent's head step and the approvals its turn staged. Not at the
    // start of a turn with a new message: that turn may stage the card the
    // hold moves to, so the hold is judged when the turn is over.
    const head = headIndex(plan);
    if (!newTurn && head !== null && ['active', 'waiting'].includes(plan.status)) {
      const step = plan.steps[head]!;
      if (step.owner === 'agent' && ['running', 'held'].includes(step.state)) {
        plan = reconcileAgentStep(plan, head, pending, pendingIds, decided, [...known]);
      }
    }

    plan = normalised(plan);

    return same(plan, original) ? original : written(plan, now, runId);
  },

  // -- The driver -------------------------------------------------------------

  /**
   * Settle the previous head from persisted state, check the limits, and
   * choose what happens next.
   */
  prepare(original: Plan, runFailed: boolean, limits: PlanLimits, now: number, runId: string | null, budgetExhausted = false): [Plan, PrepareOutcome] {
    if (original.status !== 'active') return [original, ['not_active', original.status]];
    let plan = settle(copy(original), runFailed, limits);

    const head = headIndex(plan);
    let outcome: PrepareOutcome;
    if (plan.status === 'blocked') {
      outcome = ['blocked', plan.blockedReason];
    } else if (head === null) {
      plan = normalised(plan);
      outcome = ['done', null];
    } else {
      const step = plan.steps[head]!;
      if (step.owner === 'human' && step.state === 'todo') {
        step.state = 'waiting_input';
        outcome = ['waiting', step.id];
      } else if (step.state !== 'todo') {
        outcome = ['waiting', step.id];
      } else if (plan.counters.advances >= limits.maxAdvances) {
        plan = block(plan, 'budget');
        outcome = ['blocked', 'budget'];
      } else if (budgetExhausted) {
        // G-366: the plan's shared budget is spent. It is only ever lowered,
        // so resume() refuses this as it refuses 'budget'.
        plan = block(plan, PLAN_BUDGET_EXHAUSTED);
        outcome = ['blocked', PLAN_BUDGET_EXHAUSTED];
      } else {
        outcome = ['run', step.id];
      }
    }
    plan = normalised(plan);

    return [same(plan, original) ? original : written(plan, now, runId), outcome];
  },

  /** Claim the head step for one turn. */
  claim(original: Plan, stepId: string, worker: string, until: number, now: number): Plan {
    const plan = copy(original);
    const head = headIndex(plan);
    if (plan.status !== 'active' || head === null || plan.steps[head]!.id !== stepId || plan.steps[head]!.state !== 'todo') {
      throw PlanRefused.because('plan.conflict');
    }
    const step = plan.steps[head]!;
    step.state = 'running';
    step.claim = { worker, until };
    step.attempts += 1;
    step.report = null;
    plan.counters.advances += 1;
    plan.lastAdvancedAt = iso(now);

    return written(plan, now, null);
  },

  /** plan_complete_step: the agent reports its running step done. It does NOT advance; the next advance settles it. */
  report(original: Plan, summary: unknown, now: number, runId: string | null): Plan {
    const plan = copy(original);
    const head = headIndex(plan);
    if (!['active', 'waiting'].includes(plan.status) || head === null
      || plan.steps[head]!.owner !== 'agent' || plan.steps[head]!.state !== 'running') {
      throw PlanRefused.because('plan.no_running_step');
    }
    if (!acceptablePlanText(summary, PlanText.SUMMARY)) throw PlanRefused.because('plan.text');
    plan.steps[head]!.report = { summary };

    return written(plan, now, runId);
  },

  /**
   * plan_pause_for_person: insert a person's step before the agent's running
   * step. A second pause on one step blocks the plan, and the blocked plan is
   * returned with the refusal so it can be written.
   *
   * @returns the plan, and a refusal code or null
   */
  pause(original: Plan, input: unknown, digest: string | null, ids: PlanIds, limits: PlanLimits, now: number, runId: string | null, authorized = true): [Plan, string | null] {
    if (!authorized) {
      // The host's call policy refused this call: nothing is checked,
      // recorded or proposed, and the live-plan slot stays free.
      throw PlanRefused.because('plan.not_authorized');
    }
    const args = asMap(input);
    if (args === null || diff(keysOf(args), ['kind', 'question', 'options']).length > 0
      || !['approve', 'input'].includes(get(args, 'kind') as string)) {
      throw PlanRefused.because('plan.shape');
    }
    const kind = args.kind as 'approve' | 'input';
    const rawOptions = get(args, 'options');
    const options = rawOptions === null ? null : asList(rawOptions);
    if (rawOptions !== null && (options === null || options.length > PlanText.OPTIONS || (kind === 'approve' && options.length > 0))) {
      throw PlanRefused.because('plan.shape');
    }
    if (!acceptablePlanText(get(args, 'question'), PlanText.TITLE)) throw PlanRefused.because('plan.text');
    for (const option of options ?? []) {
      if (!acceptablePlanText(option, PlanText.OPTION)) throw PlanRefused.because('plan.text');
    }
    let plan = copy(original);
    const head = headIndex(plan);
    if (plan.status !== 'active' || head === null || plan.steps[head]!.owner !== 'agent' || plan.steps[head]!.state !== 'running') {
      throw PlanRefused.because('plan.no_running_step');
    }
    const step = plan.steps[head]!;
    if (step.pauses >= limits.maxPausesPerStep) {
      plan.steps[head] = closed(step, 'failed', 'host', 'pause_limit');
      plan = block(plan, 'pause_limit');

      return [written(normalised(plan), now, runId), 'plan.pause_limit'];
    }
    step.pauses += 1;
    step.state = 'todo';
    step.claim = null;
    step.report = null;
    step.attempts = Math.max(0, step.attempts - 1);

    const question = args.question as string;
    const human = newStep(ids.step(), {
      title: question,
      instruction: null,
      owner: 'human',
      interaction: kind,
      question,
      options: (options ?? []) as string[],
    });
    human.state = kind === 'approve' ? 'held' : 'waiting_input';
    human.pauseDigest = kind === 'approve' ? digest : null;
    plan.steps.splice(head, 0, human);
    if (plan.steps.length > limits.maxSteps) throw PlanRefused.because('plan.bounds');

    return [written(normalised(renumber(plan)), now, runId), null];
  },

  /**
   * A person answers a question step. The actor is the host's, never the
   * model's. An answer is data for the next turn; it releases no approval.
   */
  answer(original: Plan, stepId: unknown, text: unknown, choice: unknown, actor: unknown, now: number): Plan {
    if (typeof actor !== 'string' || phpTrim(actor) === '') throw PlanRefused.because('plan.no_actor');
    const plan = copy(original);
    let index: number | null = null;
    plan.steps.forEach((step, i) => {
      if (step.id === stepId) index = i;
    });
    if (index === null || !['active', 'waiting'].includes(plan.status)) throw PlanRefused.because('plan.not_answerable');
    const step = plan.steps[index]!;
    if (step.owner !== 'human' || step.interaction !== 'input' || step.state !== 'waiting_input') {
      throw PlanRefused.because('plan.not_answerable');
    }
    const said = text ?? '';
    const picked = choice ?? null;
    if (typeof said !== 'string' || !wellFormed(said) || planTextLength(said) > PlanText.ANSWER
      || (picked !== null && !step.options.includes(picked as string))
      || (said === '' && picked === null)) {
      throw PlanRefused.because('plan.answer_invalid');
    }
    step.answer = { text: said, choice: picked as string | null, actor, at: iso(now) };
    plan.steps[index] = closed(step, 'done', 'person', null);

    return written(normalised(plan), now, null);
  },

  /**
   * plan_propose_change: a new list for the steps that can still be replaced.
   * Only `todo`, agent-owned steps can be replaced; every other step must
   * appear unchanged, by id, in its order, with every finished step first. A
   * stale revision, an unknown id and a step that cannot be replaced get the
   * same refusal.
   */
  proposeChange(original: Plan, input: unknown, digest: string, ids: PlanIds, limits: PlanLimits, now: number, runId: string | null, authorized = true): Plan {
    if (!authorized) {
      // The host's call policy refused this call: nothing is checked,
      // recorded or proposed, and the live-plan slot stays free.
      throw PlanRefused.because('plan.not_authorized');
    }
    const args = asMap(input);
    const entries = args === null ? null : asList(get(args, 'steps'));
    if (args === null || diff(keysOf(args), ['reason', 'base_revision', 'steps']).length > 0 || entries === null) {
      throw PlanRefused.because('plan.shape');
    }
    const plan = copy(original);
    if (plan.status !== 'active' || get(args, 'base_revision') !== plan.revision) throw PlanRefused.because('plan.change_refused');
    if (plan.counters.changes >= limits.maxChanges) throw PlanRefused.because('plan.change_limit');
    if (!acceptablePlanText(get(args, 'reason'), PlanText.REASON)) throw PlanRefused.because('plan.text');
    if (entries.length < 1 || entries.length > limits.maxSteps) throw PlanRefused.because('plan.bounds');

    const existing = new Map(plan.steps.map((step) => [step.id, step]));
    const order: string[] = [];
    const fresh: PlanStep[] = [];
    for (const raw of entries) {
      let entry: unknown = raw;
      if (isPlainObject(entry) && has(entry, 'id') && entry.id === null) {
        const { id: _dropped, ...rest } = entry;
        entry = rest;
      }
      if (isPlainObject(entry) && has(entry, 'id')) {
        const { id, ...rest } = entry;
        if (Object.values(rest).some((value) => value !== null)
          || !PlanIds.isStepId(id) || !existing.has(id) || order.includes(id)) {
          throw PlanRefused.because('plan.change_refused');
        }
        order.push(id);
        continue;
      }
      // The id is minted before the step is checked, as the reference does:
      // a refused entry still advances the generator.
      const stepId = ids.step();
      const step = newStep(stepId, PlanMachine.checkStepInput(entry));
      fresh.push(step);
      order.push(step.id);
    }

    const fixed = plan.steps.filter((step) => !replaceable(step)).map((step) => step.id);
    // Every fixed step is kept, in its order.
    if (!same(order.filter((id) => fixed.includes(id)), fixed)) throw PlanRefused.because('plan.change_refused');
    // Finished steps come first; a fixed head stays the head.
    let seenOpen = false;
    for (const id of order) {
      const known = existing.get(id);
      const terminal = known !== undefined && PlanMachine.TERMINAL.includes(known.state);
      if (terminal && seenOpen) throw PlanRefused.because('plan.change_refused');
      if (!terminal) {
        if (!seenOpen) {
          const head = headIndex(plan);
          if (head !== null && !replaceable(plan.steps[head]!) && plan.steps[head]!.id !== id) throw PlanRefused.because('plan.change_refused');
        }
        seenOpen = true;
      }
    }
    const replaced = [...existing.keys()].filter((id) => !order.includes(id));
    if (order.length + replaced.length > limits.maxSteps) throw PlanRefused.because('plan.bounds');

    plan.pendingChange = {
      approvalId: null,
      baseRevision: plan.revision,
      reason: args.reason as string,
      digest,
      order,
      newSteps: fresh,
    };
    plan.status = 'waiting';
    plan.counters.changes += 1;
    if (firstEventBytes(plan) > limits.eventMaxBytes) throw PlanRefused.because('plan.size');

    return written(plan, now, runId);
  },

  /** The person approved the change: the approved call ran with its stored arguments, which must be the proposed ones. */
  applyChange(original: Plan, digest: string, now: number, runId: string | null): Plan {
    const plan = copy(original);
    const change = plan.pendingChange;
    if (plan.status !== 'waiting' || change === null || change.digest !== digest) throw PlanRefused.because('plan.not_waiting');
    const byId = new Map([...plan.steps, ...change.newSteps].map((step) => [step.id, step]));
    const steps = change.order.map((id) => byId.get(id)!);
    for (const step of plan.steps) {
      if (!change.order.includes(step.id)) steps.push(closed(step, 'skipped', 'host', 'replaced'));
    }
    plan.steps = steps;
    plan.pendingChange = null;
    plan.status = 'active';

    return written(normalised(renumber(plan)), now, runId);
  },

  /**
   * The host unblocks a plan. A step that failed for lack of a person's card
   * or of attempts gets a retry step after it; a step that paused too often
   * is not replayed; a plan that ran out of advances stays blocked.
   */
  resume(original: Plan, ids: PlanIds, limits: PlanLimits, now: number): Plan {
    if (original.status !== 'blocked' || original.blockedReason === 'budget' || original.blockedReason === PLAN_BUDGET_EXHAUSTED) {
      throw PlanRefused.because('plan.not_resumable');
    }
    const plan = copy(original);
    const reason = plan.blockedReason;
    if (reason !== null && ['attempts_exhausted', 'no_completion', 'approval_withdrawn'].includes(reason)) {
      let failed: number | null = null;
      plan.steps.forEach((step, i) => {
        if (step.owner === 'agent' && step.state === 'failed' && step.failure === reason) failed = i;
      });
      if (failed !== null) {
        if (plan.steps.length + 1 > limits.maxSteps) throw PlanRefused.because('plan.bounds');
        const old = plan.steps[failed]!;
        const retry = newStep(ids.step(), {
          title: old.title, instruction: old.instruction, owner: 'agent', interaction: null, question: null, options: [],
        });
        retry.retryOf = old.id;
        plan.steps.splice(failed + 1, 0, retry);
      }
    }
    plan.status = 'active';
    plan.blockedReason = null;

    return written(normalised(renumber(plan)), now, null);
  },

  cancel(original: Plan, now: number): Plan {
    if (!PlanMachine.LIVE.includes(original.status)) throw PlanRefused.because('plan.not_live');
    const plan = copy(original);
    plan.steps = plan.steps.map((step) => PlanMachine.TERMINAL.includes(step.state) ? step : closed(step, 'skipped', 'host', 'cancelled'));
    plan.pendingChange = null;
    plan.holdingApprovalId = null;
    plan.status = 'cancelled';

    return written(plan, now, null);
  },

  // -- Helpers ----------------------------------------------------------------

  headIndex(plan: Plan): number | null {
    return headIndex(plan);
  },

  /**
   * Whether the reaper should hand this plan back to the host: it is active,
   * its head is the agent's step with no live claim (todo, or running on a
   * lease that ran out), and it has not moved for `staleAfter` seconds. A step
   * waiting on a person is never stalled.
   */
  stalled(plan: Plan, now: number, staleAfter: number): boolean {
    const head = headIndex(plan);
    if (plan.status !== 'active' || head === null) return false;
    const step = plan.steps[head]!;
    const claimLive = step.claim !== null && typeof step.claim === 'object' && step.claim.until > now;
    const idle = (step.owner === 'agent' && step.state === 'todo') || (step.state === 'running' && !claimLive);
    const parsed = Date.parse(String(plan.lastAdvancedAt ?? plan.updatedAt));

    return idle && (Number.isNaN(parsed) || now - Math.floor(parsed / 1000) >= staleAfter);
  },

  isLive(plan: Plan | null | undefined): plan is Plan {
    return typeof plan === 'object' && plan !== null && PlanMachine.LIVE.includes(plan.status);
  },
};

function reject(plan: Plan, reason: string | null): Plan {
  plan.status = 'rejected';
  plan.blockedReason = reason;
  plan.holdingApprovalId = null;
  plan.steps = plan.steps.map((step) => PlanMachine.TERMINAL.includes(step.state) ? step : closed(step, 'skipped', 'host', 'rejected'));

  return plan;
}

function reconcileAgentStep(
  plan: Plan,
  index: number,
  pending: readonly PendingCall[],
  pendingIds: readonly string[],
  decided: ReadonlyMap<string, boolean>,
  known: readonly string[],
): Plan {
  const step = copy(plan.steps[index]!);
  const live = diff(step.approvalIds, step.approvedIds);

  for (const id of live) {
    if (decided.has(id) && decided.get(id) === false) {
      // A person refused work this step needed: never shown as done.
      plan.steps[index] = closed(step, 'failed', 'person', 'approval_refused');

      return plan;
    }
  }
  const stillPending: string[] = [];
  const superseded: string[] = [];
  for (const id of live) {
    if (decided.has(id)) step.approvedIds.push(id);
    else if (pendingIds.includes(id)) stillPending.push(id);
    else superseded.push(id);
  }
  const fresh = pending.filter((entry) => !PlanMachine.TOOLS.includes(entry.name) && !known.includes(entry.id)).map((entry) => entry.id);
  step.approvalIds = [...diff(step.approvalIds, superseded), ...fresh];
  step.supersededApprovalIds = [...step.supersededApprovalIds, ...superseded];

  if (stillPending.length > 0 || fresh.length > 0) {
    step.state = 'held';
  } else if (superseded.length > 0) {
    // The person's card was lost and nothing replaced it.
    plan.steps[index] = closed(step, 'failed', 'host', 'approval_withdrawn');

    return block(plan, 'approval_withdrawn');
  } else {
    step.state = 'running';
  }
  plan.steps[index] = step;

  return plan;
}

function settle(plan: Plan, runFailed: boolean, limits: PlanLimits): Plan {
  const head = headIndex(plan);
  if (head === null) return plan;
  let step = plan.steps[head]!;
  if (step.owner !== 'agent') return plan;
  if (step.state === 'running') {
    if (step.report !== null && typeof step.report === 'object' && !runFailed) {
      step.summary = step.report.summary;
      plan.steps[head] = closed(step, 'done', 'agent-report', null);
      plan.counters.settledByTool += 1;

      return plan;
    }
    step.state = 'todo';
    step.claim = null;
    step.report = null;
    step.lastSettle = runFailed ? 'run_failed' : 'no_report';
    if (!runFailed) plan.counters.settledWithoutTool += 1;
  }
  step = plan.steps[head]!;
  if (step.state === 'todo' && step.attempts >= limits.maxAttempts) {
    const reason = step.lastSettle === 'no_report' ? 'no_completion' : 'attempts_exhausted';
    plan.steps[head] = closed(step, 'failed', 'host', reason);
    plan = block(plan, reason);
  }

  return plan;
}

function headIndex(plan: Plan): number | null {
  const index = plan.steps.findIndex((step) => !PlanMachine.TERMINAL.includes(step.state));

  return index === -1 ? null : index;
}

function newStep(id: string, input: DraftInput): PlanStep {
  return {
    id,
    position: 0,
    title: input.title,
    instruction: input.instruction,
    owner: input.owner,
    interaction: input.interaction,
    question: input.question,
    options: input.options,
    state: 'todo',
    approvalIds: [],
    approvedIds: [],
    supersededApprovalIds: [],
    pauseDigest: null,
    claim: null,
    attempts: 0,
    pauses: 0,
    lastSettle: null,
    report: null,
    summary: null,
    answer: null,
    completedBy: null,
    failure: null,
    retryOf: null,
  };
}

function replaceable(step: PlanStep): boolean {
  return step.owner === 'agent' && step.state === 'todo';
}

function closed(original: PlanStep, state: string, by: string, failure: string | null): PlanStep {
  return { ...copy(original), state, completedBy: by, failure, claim: null, report: null };
}

function block(plan: Plan, reason: string): Plan {
  plan.status = 'blocked';
  plan.blockedReason = reason;

  return plan;
}

function renumber(plan: Plan): Plan {
  plan.steps.forEach((step, i) => {
    step.position = i + 1;
  });

  return plan;
}

/**
 * Finish an active plan whose steps are all closed, and point
 * holdingApprovalId at the one approval that is actually holding it.
 */
function normalised(plan: Plan): Plan {
  if (['active', 'waiting'].includes(plan.status) && plan.pendingChange === null && headIndex(plan) === null) plan.status = 'done';
  if (plan.status === 'proposed') return plan;
  let holding: string | null = null;
  if (plan.pendingChange !== null && plan.pendingChange.approvalId !== null) {
    holding = plan.pendingChange.approvalId;
  } else {
    const head = headIndex(plan);
    if (head !== null && plan.steps[head]!.state === 'held') {
      const step = plan.steps[head]!;
      holding = diff(step.approvalIds, step.approvedIds)[0] ?? null;
    }
  }
  plan.holdingApprovalId = ['active', 'waiting'].includes(plan.status) ? holding : null;

  return plan;
}

/** Every approval id the plan is already bound to. */
function boundIds(plan: Plan): string[] {
  const ids: string[] = [];
  if (typeof plan.holdingApprovalId === 'string') ids.push(plan.holdingApprovalId);
  if (plan.pendingChange !== null && typeof plan.pendingChange.approvalId === 'string') ids.push(plan.pendingChange.approvalId);
  for (const step of plan.steps) ids.push(...step.approvalIds, ...step.supersededApprovalIds);

  return [...new Set(ids)];
}

function written(plan: Plan, now: number, runId: string | null): Plan {
  plan.revision += 1;
  plan.updatedAt = iso(now);
  plan.lastRunId = runId;

  return plan;
}

/** `YYYY-MM-DDTHH:MM:SSZ`, UTC, no fraction. */
export function iso(seconds: number): string {
  return new Date(seconds * 1000).toISOString().replace(/\.\d{3}Z$/, 'Z');
}
