import { readFileSync, writeFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  HarnessError,
  PlanIds,
  PlanLimits,
  PlanMachine,
  PlanRefused,
  PlanView,
  ToolPolicy,
  renderPlanText,
  type Plan,
  type PendingCall,
} from '../src/index.js';

// The cross-language plans corpus (G-360, prism-parity suites/harness-plans).
// Each case runs its ops through the pure PlanMachine and PlanView, the same
// interpreter as the reference's tests/PlanCorpusTest.php. With
// HARNESS_PLANS_RECORD set to a path, the outcomes are written there for
// tools/generate-harness-plans.mjs --merge instead of being compared.

type Op = Record<string, any>;

interface PlansCase {
  id: string;
  title: string;
  now?: number;
  limits?: Record<string, number>;
  ops: Op[];
  expected: unknown;
}

const corpus = JSON.parse(readFileSync(new URL('./fixtures/harness-plans.json', import.meta.url), 'utf8')) as { cases: PlansCase[] };

function runPlansCase(entry: PlansCase): { results: unknown[]; final: unknown } {
  const given = entry.limits ?? {};
  const limits = new PlanLimits({
    maxSteps: given.max_steps ?? 20,
    maxAttempts: given.max_attempts ?? 3,
    maxPausesPerStep: given.max_pauses_per_step ?? 1,
    maxChanges: given.max_changes ?? 5,
    maxAdvances: given.max_advances ?? 60,
    contextMaxChars: given.context_max_chars ?? 2000,
    contextWindow: given.context_window ?? 8,
    eventMaxBytes: given.event_max_bytes ?? 8192,
  });
  let counter = 0;
  const ids = new PlanIds((prefix) => `${prefix}_${String(++counter).padStart(26, '0')}`);
  const state: { plan: Plan | null; marked?: ReturnType<typeof PlanView.steps> | null } = { plan: null, marked: null };
  const results: unknown[] = [];
  for (const op of entry.ops) {
    const now = (op.now ?? entry.now ?? 1767225600) as number;
    try {
      results.push(plansOp(op, state, ids, limits, now));
    } catch (error) {
      if (!(error instanceof PlanRefused)) throw error;
      results.push({ refused: error.code });
    }
  }

  return { results, final: state.plan === null ? null : plansFinal(state.plan, limits) };
}

function plansOp(op: Op, state: { plan: Plan | null; marked?: ReturnType<typeof PlanView.steps> | null }, ids: PlanIds, limits: PlanLimits, now: number): unknown {
  const run = 'run_case';
  const needs = (): Plan => {
    if (state.plan === null) throw PlanRefused.because('plan.no_plan');

    return state.plan;
  };
  switch (op.op) {
    case 'draft': {
      const draft = PlanMachine.checkDraft(op.args, limits, op.authorized ?? true);
      if (PlanMachine.isLive(state.plan)) throw PlanRefused.because('plan.exists');
      state.plan = PlanMachine.propose(draft, op.digest, ids, now, run, limits);

      return { ok: true };
    }
    case 'activate':
      state.plan = PlanMachine.activate(needs(), op.digest, now, run);

      return { ok: true };
    case 'reconcile': {
      const decided = new Map<string, boolean>(Object.entries(op.decided as Record<string, boolean>).map(([id, approved]) => [String(id), approved]));
      state.plan = PlanMachine.reconcile(needs(), op.pending as PendingCall[], decided, now, null, op.new_turn ?? false);

      return { ok: true };
    }
    case 'prepare': {
      const [plan, outcome] = PlanMachine.prepare(needs(), op.run_failed, limits, now, null);
      state.plan = plan;

      return { outcome };
    }
    case 'claim': {
      const current = needs();
      const head = PlanMachine.headIndex(current);
      if (head === null) throw PlanRefused.because('plan.conflict');
      state.plan = PlanMachine.claim(current, current.steps[head]!.id, 'worker', op.until, now);

      return { ok: true };
    }
    case 'report':
      state.plan = PlanMachine.report(needs(), op.summary, now, run);

      return { ok: true };
    case 'pause': {
      const [plan, code] = PlanMachine.pause(needs(), op.args, op.digest ?? null, ids, limits, now, run, op.authorized ?? true);
      state.plan = plan;

      return code === null ? { ok: true } : { refused: code };
    }
    case 'answer': {
      const current = needs();
      const stepId = current.steps[op.step]?.id ?? 'ps_missing';
      state.plan = PlanMachine.answer(current, stepId, op.text, op.choice, op.actor, now);

      return { ok: true };
    }
    case 'change': {
      const current = needs();
      const args = { ...op.args };
      if (args.base_revision === 'current') args.base_revision = current.revision;
      args.steps = (args.steps as Op[]).map((entry) => Object.hasOwn(entry, 'ref') ? { id: current.steps[entry.ref]!.id } : entry);
      state.plan = PlanMachine.proposeChange(current, args, op.digest, ids, limits, now, run, op.authorized ?? true);

      return { ok: true };
    }
    case 'apply_change':
      state.plan = PlanMachine.applyChange(needs(), op.digest, now, run);

      return { ok: true };
    case 'resume':
      state.plan = PlanMachine.resume(needs(), ids, limits, now);

      return { ok: true };
    case 'cancel':
      state.plan = PlanMachine.cancel(needs(), now);

      return { ok: true };
    case 'mark':
      state.marked = PlanView.steps(needs());

      return { ok: true };
    case 'update_since_mark':
      return { update: PlanView.update(needs(), state.marked ?? null) };
    case 'stalled':
      return { stalled: PlanMachine.stalled(needs(), now, op.stale_after) };
    case 'policy':
      try {
        ToolPolicy.fromJson({ revision: 'r1', allow: [{ tool: op.name }], honor: true });

        return { refused: false };
      } catch (error) {
        if (error instanceof HarnessError && error.code === 'tool_policy_refused') return { refused: true };
        throw error;
      }
    case 'render':
      return { rendered: renderPlanText(op.value, op.max) };
    default:
      throw new Error(`Unknown op ${String(op.op)}`);
  }
}

function plansFinal(plan: Plan, limits: PlanLimits): unknown {
  const change = plan.pendingChange;

  return {
    status: plan.status,
    revision: plan.revision,
    holdingApprovalId: plan.holdingApprovalId,
    blockedReason: plan.blockedReason,
    counters: plan.counters,
    pendingChange: change === null ? null : {
      approvalId: change.approvalId, order: change.order, newSteps: change.newSteps.map((step) => step.id),
    },
    steps: plan.steps.map((step) => ({
      id: step.id, position: step.position, title: step.title, owner: step.owner,
      interaction: step.interaction, state: step.state, attempts: step.attempts, pauses: step.pauses,
      approvalIds: step.approvalIds, approvedIds: step.approvedIds, supersededApprovalIds: step.supersededApprovalIds,
      completedBy: step.completedBy, failure: step.failure, retryOf: step.retryOf,
      summary: step.summary, answer: step.answer,
    })),
    update: PlanView.update(plan, null),
    counts: PlanView.counts(plan),
    context: PlanView.context(plan, limits),
    first_event_bytes: PlanView.firstEventBytes(plan),
  };
}

const record = process.env.HARNESS_PLANS_RECORD;

describe('the harness-plans corpus (G-360)', () => {
  if (typeof record === 'string' && record !== '') {
    it('records the plans corpus', () => {
      const out: Record<string, unknown> = {};
      for (const entry of corpus.cases) out[entry.id] = runPlansCase(entry);
      writeFileSync(record, JSON.stringify(out, null, 4));
      expect(Object.keys(out)).not.toHaveLength(0);
    });

    return;
  }

  it('vendors all 39 cases', () => {
    expect(corpus.cases).toHaveLength(39);
  });

  it.each(corpus.cases.map((entry) => [entry.id, entry] as const))('%s runs as the reference recorded it', (_id, entry) => {
    const actual = runPlansCase(entry);
    // Compared as JSON text as well, so key order and a list-versus-map difference cannot hide.
    expect(actual).toEqual(entry.expected);
    expect(JSON.stringify(actual)).toBe(JSON.stringify(entry.expected));
  });
});
