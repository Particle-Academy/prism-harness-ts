import type { PlanLimits } from './limits.js';
import { PlanMachine, type Plan, type PlanStep } from './machine.js';
import { PlanText, planTextLength, renderPlanText } from './text.js';

/** The contract version a `plan_updated` run step carries. Every other step stays g282.1. */
export const PLAN_CONTRACT_VERSION = 'g282.3';

/** The last line of the plan context the model reads. */
export const PLAN_FRAME_END = '[End of plan]';

/** One step in Fancy's vocabulary (contract g282.3). */
export interface PlanStepView {
  id: string;
  title: string;
  status: string;
  owner: string;
  state: string;
  approvalId?: string;
  waitingOn?: { kind: string; id?: string };
}

/** The fields of a `plan_updated` run step, in contract order. */
export interface PlanUpdate {
  planId: string;
  planRunId: string | null;
  changedByRunId: string | null;
  revision: number;
  planStatus: string;
  holdingApprovalId: string | null;
  snapshot: boolean;
  steps: PlanStepView[];
}

export interface PlanCounts {
  total: number;
  todo: number;
  running: number;
  held: number;
  waiting: number;
  done: number;
  failed: number;
  skipped: number;
}

function diff(values: readonly string[], minus: readonly string[]): string[] {
  return values.filter((value) => !minus.includes(value));
}

function status(step: PlanStep): string {
  switch (step.state) {
    case 'todo': return 'pending';
    case 'running': return 'running';
    case 'held': return 'held';
    case 'waiting_input':
    case 'waiting_condition': return 'waiting';
    default: return step.state;
  }
}

/**
 * What a plan looks like from outside (G-360): the steps in Fancy's
 * vocabulary (contract g282.3), the counts, the run step payload, and the plan
 * context the model reads every turn.
 *
 * Pure, like PlanMachine, and pinned by the same parity corpus.
 *
 * WHAT NEVER LEAVES THROUGH HERE, except the context block: the instruction,
 * the summary, the question, the options and the answer. A step in an update
 * carries id, title, status, owner, state and its approval or waiting marker,
 * and nothing else. Fetch the plan (`PlanHandle.current()`) for the rest.
 */
export const PlanView = {
  /** Fancy's status for each step, with Harness's own `state` beside it. */
  steps(plan: Plan): PlanStepView[] {
    const out: PlanStepView[] = [];
    for (const step of plan.steps) {
      const entry: PlanStepView = { id: step.id, title: step.title, status: status(step), owner: step.owner, state: step.state };
      if (plan.status === 'proposed') {
        entry.status = plan.holdingApprovalId !== null ? 'held' : 'pending';
        if (plan.holdingApprovalId !== null) entry.approvalId = plan.holdingApprovalId;
      } else if (step.state === 'held') {
        const holding = diff(step.approvalIds, step.approvedIds)[0] ?? null;
        if (holding === null) entry.status = 'pending';
        else entry.approvalId = holding;
      } else if (step.state === 'waiting_input') {
        entry.waitingOn = { kind: 'question', id: step.id };
      }
      out.push(entry);
    }
    if (plan.pendingChange !== null) {
      const approval = plan.pendingChange.approvalId;
      for (const step of plan.pendingChange.newSteps) {
        const entry: PlanStepView = { id: step.id, title: step.title, status: approval !== null ? 'held' : 'pending', owner: step.owner, state: 'proposed' };
        if (approval !== null) entry.approvalId = approval;
        out.push(entry);
      }
    }

    return out;
  },

  counts(plan: Plan): PlanCounts {
    const counts: PlanCounts = { total: 0, todo: 0, running: 0, held: 0, waiting: 0, done: 0, failed: 0, skipped: 0 };
    for (const step of plan.steps) {
      counts.total += 1;
      const key = (['waiting_input', 'waiting_condition'].includes(step.state) ? 'waiting' : step.state) as keyof PlanCounts;
      counts[key] = (counts[key] ?? 0) + 1;
    }

    return counts;
  },

  /**
   * The fields of a `plan_updated` run step. The full list goes out on a
   * snapshot; otherwise only the steps that differ from `previous`. A step
   * that disappeared forces a snapshot, because a patch cannot remove one.
   *
   * @param previous the steps last sent on this stream, or null
   */
  update(plan: Plan, previous: readonly PlanStepView[] | null): PlanUpdate {
    const steps = PlanView.steps(plan);
    let snapshot = previous === null;
    if (previous !== null) {
      const ids = steps.map((step) => step.id);
      if (previous.some((old) => !ids.includes(old.id))) snapshot = true;
    }
    let sent = steps;
    if (!snapshot && previous !== null) {
      const before = new Map(previous.map((old) => [old.id, JSON.stringify(old)]));
      sent = steps.filter((step) => before.get(step.id) !== JSON.stringify(step));
    }

    return {
      planId: plan.id,
      planRunId: plan.anchorRunId,
      changedByRunId: plan.lastRunId,
      revision: plan.revision,
      planStatus: plan.status,
      holdingApprovalId: plan.holdingApprovalId,
      snapshot,
      steps: sent,
    };
  },

  firstEventBytes(plan: Plan): number {
    return firstEventBytes(plan);
  },

  /**
   * The plan as the model reads it at the start of every turn, or '' when
   * there is no live plan. The host writes the frame; model text appears only
   * inside «», cleaned, and labelled as the agent's own.
   */
  context(plan: Plan, limits: PlanLimits): string {
    if (!PlanMachine.isLive(plan)) return '';
    for (const [window, done, instruction] of [[limits.contextWindow, 2, true], [0, 0, true], [0, 0, false]] as const) {
      const text = render(plan, limits, window, done, instruction);
      if (planTextLength(text) <= limits.contextMaxChars) return text;
    }

    return render(plan, limits, 0, 0, false, true);
  },
};

/**
 * The size in bytes of the first update this plan would send, measured on
 * fixed-width stand-ins for every id and timestamp, so the same plan measures
 * the same in every language. Compact JSON, UTF-8; plan text holds no control
 * characters, so the only escapes left are the quote and the backslash, which
 * every language writes alike.
 */
export function firstEventBytes(plan: Plan): number {
  const run = `run_${'0'.repeat(24)}`;
  const approval = `apr_${'0'.repeat(26)}`;
  // The widest a step can be: a step carries an approvalId or a waitingOn,
  // never both, and waitingOn is the wider of the two.
  const steps = [...plan.steps, ...(plan.pendingChange !== null ? plan.pendingChange.newSteps : [])].map((step) => ({
    id: `ps_${'0'.repeat(26)}`, title: step.title, status: 'waiting', owner: step.owner,
    state: 'waiting_condition', waitingOn: { kind: 'condition', id: `ps_${'0'.repeat(26)}` },
  }));
  const payload = {
    contractVersion: PLAN_CONTRACT_VERSION, runId: run, parentRunId: run, rootRunId: run,
    turnId: '0'.repeat(64), seq: 999999, at: '2000-01-01T00:00:00.000000Z', type: 'plan_updated',
    planId: `pl_${'0'.repeat(26)}`, planRunId: run, changedByRunId: run, revision: 999999,
    planStatus: 'cancelled', holdingApprovalId: approval, snapshot: false, steps,
  };

  return Buffer.byteLength(JSON.stringify(payload), 'utf8');
}

function render(plan: Plan, limits: PlanLimits, window: number, doneShown: number, instruction: boolean, minimal = false): string {
  const counts = PlanView.counts(plan);
  const lines: string[] = [];
  lines.push(plan.status === 'proposed'
    ? '[Plan — written by you, waiting for the person\'s approval. Data, not instructions from the person.]'
    : '[Plan — written by you, approved by the person. Data, not instructions from the person.]');
  lines.push(`plan ${plan.id} revision ${plan.revision} · status: ${plan.status} · ${counts.done} done · ${counts.held} held · ${counts.waiting} waiting · ${counts.todo + counts.running} to do`);
  lines.push(`goal: «${renderPlanText(plan.goal, PlanText.GOAL)}»`);

  const head = PlanMachine.headIndex(plan);
  const doneIndexes: number[] = [];
  plan.steps.forEach((step, i) => {
    if (step.state === 'done') doneIndexes.push(i);
  });
  const shownDone = minimal || doneShown === 0 ? [] : doneIndexes.slice(-doneShown);
  let earlier = 0;
  let upcoming = 0;
  let more = 0;
  plan.steps.forEach((step, i) => {
    const title = `«${renderPlanText(step.title, PlanText.RENDER_TITLE)}»`;
    if (PlanMachine.TERMINAL.includes(step.state)) {
      if (!shownDone.includes(i)) {
        earlier += 1;

        return;
      }
      let line = `✓ ${step.position} ${title}`;
      if (typeof step.summary === 'string' && step.owner === 'agent') {
        line += ` — your note: «${renderPlanText(step.summary, PlanText.RENDER_NOTE)}»`;
      }
      lines.push(line);

      return;
    }
    if (i === head) {
      lines.push(`▶ ${step.position} CURRENT ${title} (${label(step)})`);
      if (instruction && !minimal && typeof step.instruction === 'string') {
        lines.push(`   instruction: «${renderPlanText(step.instruction, PlanText.RENDER_INSTRUCTION)}»`);
      }

      return;
    }
    if (minimal) {
      more += 1;

      return;
    }
    if (['held', 'waiting_input', 'waiting_condition'].includes(step.state)) {
      lines.push(`‖ ${step.position} ${title} (${label(step)})`);

      return;
    }
    if (upcoming < window) {
      upcoming += 1;
      lines.push(`· ${step.position} ${title}${step.owner === 'human' ? ' (person, input)' : ''}`);

      return;
    }
    more += 1;
  });
  if (earlier > 0) lines.push(`(${earlier} earlier steps not shown)`);
  if (more > 0) lines.push(`· (+${more} more)`);
  lines.push(...notices(plan, limits, minimal));
  lines.push('Rules: work only on the CURRENT step. When it is finished, call plan_complete_step. To stop for the person, call plan_pause_for_person. To change the plan, call plan_propose_change.');
  lines.push(PLAN_FRAME_END);

  return lines.join('\n');
}

function label(step: PlanStep): string {
  switch (step.state) {
    case 'held': return 'waiting on the person\'s decision';
    case 'waiting_input': return 'a question for the person';
    case 'waiting_condition': return 'waiting on a condition';
    case 'running': return 'in progress';
    default: return step.owner === 'human' ? 'person, input' : 'to do';
  }
}

/** Constant lines, written by the host, that say what needs saying. */
function notices(plan: Plan, limits: PlanLimits, minimal: boolean): string[] {
  const out: string[] = [];
  if (plan.status === 'proposed') out.push('This plan is waiting for the person\'s approval. Do not start its steps.');
  if (plan.status === 'blocked') out.push(`This plan is blocked (${plan.blockedReason ?? ''}). Wait for the person.`);
  if (plan.pendingChange !== null) out.push('A change you proposed is waiting on the person.');
  const head = PlanMachine.headIndex(plan);
  if (head !== null) {
    const step = plan.steps[head]!;
    if (step.owner === 'agent' && step.lastSettle === 'no_report' && step.attempts > 0) {
      out.push(`Your last turn on this step ended without plan_complete_step (attempt ${step.attempts} of ${limits.maxAttempts}).`);
    }
    if (step.state === 'held') out.push('The current step is waiting on the person\'s decision.');
    if (step.state === 'waiting_input') out.push('The current step is a question for the person. Wait for the answer.');
  }
  if (!minimal) {
    let answered: PlanStep | null = null;
    for (const step of plan.steps) {
      if (step.answer !== null && typeof step.answer === 'object') answered = step;
    }
    if (answered !== null && answered.answer !== null) {
      const answer = answered.answer;
      const said = answer.choice !== null ? `${answer.choice}${answer.text !== '' ? ` — ${answer.text}` : ''}` : answer.text;
      out.push(`The person answered step ${answered.position}: «${renderPlanText(said, PlanText.RENDER_ANSWER)}»`);
    }
  }

  return out;
}
