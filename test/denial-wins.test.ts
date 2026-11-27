import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  ToolRegistry,
  recordApproval,
  recordDecisions,
  type AgentResponse,
  type JsonObject,
  type LlmResponse,
  type RunStep,
  type Session,
} from '../src/index.js';
import { toolResultRow } from '../src/thread-rows.js';

// G-401: a DENIAL for an approval is final. Once a denial is recorded, through
// recordApproval, recordDecisions or streamDecision, in one batch or later, or
// already in the thread, the call never runs; a later approval records
// nothing. A denial after an approval revokes it, as long as the call has not
// run. An identical re-send records nothing. Offline; every store is in memory.

const network = vi.fn(() => { throw new Error('A test tried to use the network.'); });
beforeEach(() => { vi.stubGlobal('fetch', network); });
afterEach(() => { expect(network).not.toHaveBeenCalled(); vi.unstubAllGlobals(); });

class DurableMemory extends MemorySessionStore {
  override durability(): 'durable' {
    return 'durable';
  }
}

const modes = new ModeRegistry({
  default: 'guarded',
  modes: { guarded: { system_prompt: 'Careful.', tools: ['echo'], max_steps: 6, requires_approval: ['echo'] } },
});

interface Rig {
  session: Session;
  agent: AgentRuntime;
  ran: number[];
  approvalId: string;
}

async function parked(): Promise<Rig> {
  const harness = new PrismHarness({
    drivers: { memory: () => new MemorySessionStore(), durable: () => new DurableMemory() },
    stores: { ephemeral: 'memory', durable: 'durable' },
  });
  const session = harness.for({ type: 'User', id: 1 }).session('first-decision');
  const ran: number[] = [];
  let model = 0;
  const agent = new AgentRuntime({
    modes,
    tools: new ToolRegistry().register({ name: 'echo', handle: (args: JsonObject) => { ran.push(Number(args.n)); return `ran:${String(args.n)}`; } }),
    client: async (): Promise<LlmResponse> => {
      model += 1;
      return model === 1 ? { text: '', finishReason: 'tool_calls', toolCalls: [{ id: 'c1', name: 'echo', arguments: { n: 1 } }] } : { text: 'done', finishReason: 'stop' };
    },
  });
  const [pending] = (await agent.send(session, 'go')).pendingApprovals;
  return { session, agent, ran, approvalId: pending!.id };
}

async function drain(stream: AsyncGenerator<RunStep, AgentResponse, undefined>): Promise<RunStep[]> {
  const steps: RunStep[] = [];
  for (;;) {
    const next = await stream.next();
    if (next.done === true) return steps;
    steps.push(next.value);
  }
}

/** Every decision stored for `approvalId`, in thread order. */
async function decisionsFor(session: Session, approvalId: string): Promise<boolean[]> {
  return (await session.thread().messages()).map((entry) => entry.message).filter((row) => row.type === 'tool_result')
    .flatMap((row) => (row.tool_approval_responses as JsonObject[]) ?? [])
    .filter((entry) => entry.approval_id === approvalId).map((entry) => entry.approved === true);
}

async function outcome(session: Session): Promise<string[]> {
  return [...new Set((await session.thread().messages()).map((entry) => entry.message).filter((row) => row.type === 'tool_result')
    .flatMap((row) => row.tool_results as JsonObject[]).filter((entry) => (entry.args as JsonObject).n === 1).map((entry) => String(entry.result)))];
}

describe('G-401: a denial for an approval is final', () => {
  it('recordApproval twice, deny then approve: the denial stands, nothing runs', async () => {
    const r = await parked();
    await recordApproval(r.session, r.approvalId, false, 'no');
    await recordApproval(r.session, r.approvalId, true);
    await r.agent.send(r.session, '');

    expect(r.ran).toEqual([]);
    expect(await outcome(r.session)).toEqual(['no']);
    expect(await decisionsFor(r.session, r.approvalId)).toEqual([false, false]);
  });

  it('one recordDecisions batch [deny, approve]: the denial stands', async () => {
    const r = await parked();
    await recordDecisions(r.session, [{ approvalId: r.approvalId, approved: false, reason: 'no' }, { approvalId: r.approvalId, approved: true }]);
    await r.agent.send(r.session, '');

    expect(r.ran).toEqual([]);
    expect(await outcome(r.session)).toEqual(['no']);
  });

  it('recordDecisions twice, deny then approve: the denial stands', async () => {
    const r = await parked();
    await recordDecisions(r.session, [{ approvalId: r.approvalId, approved: false, reason: 'no' }]);
    await recordDecisions(r.session, [{ approvalId: r.approvalId, approved: true }]);
    await r.agent.send(r.session, '');

    expect(r.ran).toEqual([]);
    expect(await outcome(r.session)).toEqual(['no']);
  });

  it('recordApproval(deny) then streamDecision([approve]): the denial stands, and the stream reports it', async () => {
    const r = await parked();
    await recordApproval(r.session, r.approvalId, false, 'no');
    const steps = await drain(r.agent.streamDecision(r.session, [{ approvalId: r.approvalId, approved: true }]));

    expect(r.ran).toEqual([]);
    expect(await outcome(r.session)).toEqual(['no']);
    const recorded = steps.filter((step) => step.type === 'decision_recorded');
    expect(recorded.map((step) => [step.approvalId, step.verdict, step.reason])).toEqual([[r.approvalId, 'denied', 'no']]);
  });

  it('deny, a turn that fails before acting, then approve: the denial stands', async () => {
    const r = await parked();
    await recordApproval(r.session, r.approvalId, false, 'no');
    await expect(r.agent.send(r.session, '', undefined, undefined, [{ type: 'image', url: 'https://example.test/x.png' }])).rejects.toThrow();
    await recordApproval(r.session, r.approvalId, true);
    await r.agent.send(r.session, '');

    expect(r.ran).toEqual([]);
    expect(await outcome(r.session)).toEqual(['no']);
  });

  it('approve then deny, before a resume: the denial revokes the approval, nothing runs', async () => {
    const r = await parked();
    await recordApproval(r.session, r.approvalId, true);
    await recordApproval(r.session, r.approvalId, false, 'no');
    await r.agent.send(r.session, '');
    await r.agent.send(r.session, '');

    expect(r.ran).toEqual([]);
    expect(await outcome(r.session)).toEqual(['no']);
  });

  it('one recordDecisions batch [approve, deny]: the denial stands', async () => {
    const r = await parked();
    await recordDecisions(r.session, [{ approvalId: r.approvalId, approved: true }, { approvalId: r.approvalId, approved: false, reason: 'no' }]);
    await r.agent.send(r.session, '');

    expect(r.ran).toEqual([]);
    expect(await outcome(r.session)).toEqual(['no']);
  });

  it('recordApproval(approve) then streamDecision([deny]): the denial stands, and the stream reports it', async () => {
    const r = await parked();
    await recordApproval(r.session, r.approvalId, true);
    const steps = await drain(r.agent.streamDecision(r.session, [{ approvalId: r.approvalId, approved: false, reason: 'no' }]));

    expect(r.ran).toEqual([]);
    expect(await outcome(r.session)).toEqual(['no']);
    expect(steps.filter((step) => step.type === 'decision_recorded').map((step) => [step.verdict, step.reason])).toEqual([['denied', 'no']]);
  });

  it('a denial after the call ran records it, and the call does not run again', async () => {
    const r = await parked();
    await recordApproval(r.session, r.approvalId, true);
    await r.agent.send(r.session, '');
    await recordApproval(r.session, r.approvalId, false, 'too late');
    await r.agent.send(r.session, '');

    expect(r.ran).toEqual([1]);
    expect(await outcome(r.session)).toEqual(['ran:1']);
  });

  it('a deny-then-approve pair already in the thread (written through Thread.record, as an earlier version or two racing writers can leave it): the denial stands when it is read, nothing runs', async () => {
    const r = await parked();
    await r.session.thread().record([toolResultRow([], [{ approval_id: r.approvalId, approved: false, reason: 'no' }])]);
    await r.session.thread().record([toolResultRow([], [{ approval_id: r.approvalId, approved: true, reason: null }])]);
    await r.agent.send(r.session, '');
    await r.agent.send(r.session, '');

    expect(r.ran).toEqual([]);
    expect(await outcome(r.session)).toEqual(['no']);
  });

  it('an approve-then-deny pair already in the thread: the denial stands when it is read, nothing runs', async () => {
    const r = await parked();
    await r.session.thread().record([toolResultRow([], [{ approval_id: r.approvalId, approved: true, reason: null }])]);
    await r.session.thread().record([toolResultRow([], [{ approval_id: r.approvalId, approved: false, reason: 'no' }])]);
    await r.agent.send(r.session, '');
    await r.agent.send(r.session, '');

    expect(r.ran).toEqual([]);
    expect(await outcome(r.session)).toEqual(['no']);
  });

  it('a pair in ONE row written through Thread.record, approve first: the denial stands', async () => {
    const r = await parked();
    await r.session.thread().record([toolResultRow([], [{ approval_id: r.approvalId, approved: true, reason: null }, { approval_id: r.approvalId, approved: false, reason: 'no' }])]);
    await r.agent.send(r.session, '');

    expect(r.ran).toEqual([]);
    expect(await outcome(r.session)).toEqual(['no']);
  });

  it('a pair in ONE row written through Thread.record, deny first: the denial stands', async () => {
    const r = await parked();
    await r.session.thread().record([toolResultRow([], [{ approval_id: r.approvalId, approved: false, reason: 'no' }, { approval_id: r.approvalId, approved: true, reason: null }])]);
    await r.agent.send(r.session, '');

    expect(r.ran).toEqual([]);
    expect(await outcome(r.session)).toEqual(['no']);
  });

  it('a re-sent identical decision records nothing and runs nothing', async () => {
    const r = await parked();
    await recordApproval(r.session, r.approvalId, true);
    const before = (await r.session.thread().messages()).length;
    await recordApproval(r.session, r.approvalId, true);
    await recordDecisions(r.session, [{ approvalId: r.approvalId, approved: true }]);

    expect((await r.session.thread().messages()).length).toBe(before);
    await r.agent.send(r.session, '');
    await drain(r.agent.streamDecision(r.session, [{ approvalId: r.approvalId, approved: true }]));
    expect(r.ran).toEqual([1]);
    expect(await decisionsFor(r.session, r.approvalId)).toEqual([true, true]);
  });
});
