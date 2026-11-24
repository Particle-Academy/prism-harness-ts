import type { JsonObject } from '../json.js';
import type { Session } from '../session.js';
import { argsDigest } from '../tool-policy.js';
import type { HarnessTool } from '../tools.js';
import { PlanMachine, type Plan } from './machine.js';
import type { Plans } from './plans.js';
import { PlanRefused, planRefusalForModel } from './refusal.js';
import { approvedCallsAwaitingResult } from './thread.js';

/** A plan tool decides for itself, per call, whether a person must approve it. */
type ApprovalGate = (args: JsonObject) => Promise<boolean>;

/**
 * The plan tools' own approval gates, keyed by the tool object the harness
 * built. Only a tool built here is in this map, so no registered tool can
 * claim to gate itself and slip out from under its mode's approval list.
 */
const GATES = new WeakMap<HarnessTool, ApprovalGate>();

/**
 * The approval gate of a plan tool, or null for every other tool.
 *
 * A plan tool gates itself: a draft, a change and an approval pause are core
 * approvals bound to their exact arguments. The runtime consults this INSTEAD
 * of the mode's approval list and the host tool policy, so neither a mode
 * gating `*` nor a permissive policy can wrap or bypass it.
 *
 * @internal
 */
export function planApprovalGate(tool: HarnessTool | undefined): ApprovalGate | null {
  return tool === undefined ? null : (GATES.get(tool) ?? null);
}

function digest(args: JsonObject): string {
  return argsDigest(args);
}

const STRING = { type: 'string' } as const;

function stepSchema(withId: boolean): JsonObject {
  const properties: JsonObject = {
    title: { type: 'string', description: 'A short title for the step, at most 160 characters, one line.' },
    instruction: { type: ['string', 'null'], description: 'What to do in this step, at most 600 characters, one line. Required for your own steps.' },
    owner: { type: 'string', enum: ['agent', 'human'], description: 'agent for your own step; human for a question to the person.' },
    interaction: { type: ['string', 'null'], enum: ['input', null], description: 'input for a question to the person; omit for your own steps.' },
    question: { type: ['string', 'null'], description: 'The question for the person, for a human step.' },
    options: { type: ['array', 'null'], items: { ...STRING }, description: 'Suggested answers for a human step; free text is always allowed.' },
  };

  return withId
    ? { type: 'object', description: 'One step.', properties: { id: { type: ['string', 'null'], description: 'To keep an existing step unchanged, its id and nothing else.' }, ...properties } }
    : { type: 'object', description: 'One step.', properties, required: ['title', 'owner'] };
}

interface PlanTool extends HarnessTool {
  /** JSON Schema for the arguments, for a model client to send to its provider. */
  readonly parameters: JsonObject;
}

/**
 * The agent's plan tools, built per run for one session (G-360).
 *
 * NO TOOL TAKES A STEP ID OR AN APPROVAL ID. Each acts on the plan's head
 * step, which the driver chose, so a model that cannot name a step cannot act
 * on someone else's, nor probe which ids exist. plan_propose_change is the one
 * exception, and only to keep existing steps unchanged by id.
 *
 * WHAT THE PERSON APPROVED IS WHAT IS APPLIED. plan_draft, plan_propose_change
 * and plan_pause_for_person (kind approve) are core approvals. The proposal is
 * recorded when the runtime asks whether the call needs approval; it is applied
 * only when the runtime runs the call a person approved, and only when that
 * call's arguments are the ones proposed. A call that reaches its handler
 * without a recorded approval for exactly these arguments applies nothing.
 *
 * Every refusal handed back to the model is a constant sentence and a code.
 *
 * @internal Built by `Plans.tools()`.
 */
export function planTools(plans: Plans, session: Session): HarnessTool[] {
  /** Refusal codes found while deciding approval, by tool and digest. */
  const refusals = new Map<string, string>();

  /** A person approved exactly this call, and it has not run yet. Read from the thread. */
  const approved = async (tool: string, hash: string): Promise<boolean> =>
    approvedCallsAwaitingResult((await session.thread().messages()).map((entry) => entry.message), tool, hash) > 0;

  const proposal = async (args: JsonObject, hash: string): Promise<Plan> => {
    const limits = plans.limits();
    const draft = PlanMachine.checkDraft(args, limits);
    if (PlanMachine.isLive(await plans.load(session))) throw PlanRefused.because('plan.exists');

    return PlanMachine.propose(draft, hash, plans.ids(), plans.now(), await plans.runId(session), limits);
  };

  const refusedCode = (error: unknown): string => {
    if (error instanceof PlanRefused) return error.code;
    throw error;
  };

  const applyPause = async (args: JsonObject, hash: string | null): Promise<string | null> => {
    let code: string | null = null;
    let plan: Plan | null;
    try {
      const runId = await plans.runId(session);
      plan = await plans.mutate(session, (current) => {
        const [next, refusal] = PlanMachine.pause(current, args, hash, plans.ids(), plans.limits(), plans.now(), runId);
        code = refusal;

        return next;
      });
    } catch (error) {
      return refusedCode(error);
    }

    return plan === null ? 'plan.no_running_step' : code;
  };

  const draft: PlanTool = {
    name: PlanMachine.DRAFT,
    description: 'Propose a plan: a goal and an ordered list of steps. A person approves it before any step starts. Steps are your own work (owner agent, with an instruction) or a question for the person (owner human, interaction input, with a question).',
    parameters: {
      type: 'object',
      properties: {
        goal: { type: 'string', description: 'What the plan is for, at most 200 characters, one line.' },
        steps: { type: 'array', description: 'The steps, in order.', items: stepSchema(false) },
      },
      required: ['goal', 'steps'],
    },
    async handle(args: JsonObject): Promise<string> {
      const hash = digest(args);
      try {
        if (await approved(PlanMachine.DRAFT, hash)) {
          const runId = await plans.runId(session);
          const plan = await plans.mutate(session, (current) => PlanMachine.activate(current, hash, plans.now(), runId));
          if (plan === null) return planRefusalForModel('plan.not_waiting');

          return 'The person approved the plan. It is active; the host will run its steps one at a time.';
        }
        const refused = refusals.get(PlanMachine.DRAFT + hash);
        if (refused !== undefined) return planRefusalForModel(refused);
        if (plans.limits().requireApproval) return planRefusalForModel('plan.not_waiting');
        // Approval switched off by the host's config: the plan starts as written.
        let plan = await proposal(args, hash);
        plan = PlanMachine.activate(plan, hash, plans.now(), await plans.runId(session));
        await plans.write(session, plan, (await plans.load(session))?.revision ?? null);

        return 'The plan is active; the host will run its steps one at a time.';
      } catch (error) {
        return planRefusalForModel(refusedCode(error));
      }
    },
  };
  GATES.set(draft, async (args) => {
    if (!plans.limits().requireApproval) return false;
    const hash = digest(args);
    try {
      const plan = await proposal(args, hash);
      await plans.write(session, plan, (await plans.load(session))?.revision ?? null);
    } catch (error) {
      refusals.set(PlanMachine.DRAFT + hash, refusedCode(error));

      return false;
    }

    return true;
  });

  const complete: PlanTool = {
    name: PlanMachine.COMPLETE,
    description: 'Report that the CURRENT plan step is finished. It does not start the next step; stop work after calling it.',
    parameters: {
      type: 'object',
      properties: { summary: { type: 'string', description: 'What you did, at most 200 characters, one line.' } },
      required: ['summary'],
    },
    async handle(args: JsonObject): Promise<string> {
      const keys = Object.keys(args);
      if (keys.length !== 1 || keys[0] !== 'summary') return planRefusalForModel('plan.shape');
      try {
        const runId = await plans.runId(session);
        const plan = await plans.mutate(session, (current) => PlanMachine.report(current, args.summary, plans.now(), runId));
        if (plan === null) return planRefusalForModel('plan.no_running_step');
      } catch (error) {
        return planRefusalForModel(refusedCode(error));
      }

      return 'Noted. The step will be settled after this turn. Stop work on it now.';
    },
  };
  GATES.set(complete, async () => false);

  const pause: PlanTool = {
    name: PlanMachine.PAUSE,
    description: 'Stop for the person before continuing the CURRENT step. kind approve asks the person to approve going on; kind input asks a question. End your turn after calling it.',
    parameters: {
      type: 'object',
      properties: {
        kind: { type: 'string', enum: ['approve', 'input'], description: 'approve or input' },
        question: { type: 'string', description: 'What to ask the person, at most 160 characters, one line.' },
        options: { type: ['array', 'null'], items: { ...STRING }, description: 'Suggested answers for kind input; free text is always allowed.' },
      },
      required: ['kind', 'question'],
    },
    async handle(args: JsonObject): Promise<string> {
      const hash = digest(args);
      if (args.kind === 'approve') {
        if (await approved(PlanMachine.PAUSE, hash)) return 'The person approved. Continue the current step.';

        return planRefusalForModel(refusals.get(PlanMachine.PAUSE + hash) ?? 'plan.no_running_step');
      }
      const code = await applyPause(args, null);

      return code !== null
        ? planRefusalForModel(code)
        : 'The question is with the person. End your turn now; the plan continues after they answer.';
    },
  };
  GATES.set(pause, async (args) => {
    if (args.kind !== 'approve') return false;
    const hash = digest(args);
    const code = await applyPause(args, hash);
    if (code !== null) {
      refusals.set(PlanMachine.PAUSE + hash, code);

      return false;
    }

    return true;
  });

  const change: PlanTool = {
    name: PlanMachine.CHANGE,
    description: 'Propose a new list for the plan\'s remaining steps. Keep every finished, running, held or person step unchanged by giving only its id. A person approves the change before it applies.',
    parameters: {
      type: 'object',
      properties: {
        reason: { type: 'string', description: 'Why the plan should change, at most 200 characters, one line.' },
        base_revision: { type: 'number', description: 'The plan revision you are changing, from the plan context.' },
        steps: { type: 'array', description: 'The new list of steps, in order.', items: stepSchema(true) },
      },
      required: ['reason', 'base_revision', 'steps'],
    },
    async handle(args: JsonObject): Promise<string> {
      const hash = digest(args);
      if (!(await approved(PlanMachine.CHANGE, hash))) return planRefusalForModel(refusals.get(PlanMachine.CHANGE + hash) ?? 'plan.not_waiting');
      try {
        const runId = await plans.runId(session);
        await plans.mutate(session, (current) => PlanMachine.applyChange(current, hash, plans.now(), runId));
      } catch (error) {
        return planRefusalForModel(refusedCode(error));
      }

      return 'The person approved the change. The plan now follows the new steps.';
    },
  };
  GATES.set(change, async (args) => {
    const hash = digest(args);
    try {
      const runId = await plans.runId(session);
      const plan = await plans.mutate(session, (current) => PlanMachine.proposeChange(current, args, hash, plans.ids(), plans.limits(), plans.now(), runId));
      if (plan === null) throw PlanRefused.because('plan.no_plan');
    } catch (error) {
      refusals.set(PlanMachine.CHANGE + hash, refusedCode(error));

      return false;
    }

    return true;
  });

  return [draft, complete, pause, change];
}
