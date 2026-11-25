import { mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  FileSessionStore,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  ToolRegistry,
  recordApproval,
  recordDecisions,
  type AgentResponse,
  type ApprovalDecision,
  type LlmRequest,
  type LlmResponse,
  type RunStep,
  type Session,
} from '../src/index.js';

// G-354. A decision is an answer to a question the run asked. An approval id
// the run never issued, or one already answered, is not an answer to anything:
// recorded while a call was pending, the resume closed that call as "No
// approval response provided"; streamed while nothing was pending, it started
// a model turn with no new user message, in which an ungated tool could run.

const modes = new ModeRegistry({
  default: 'guarded',
  modes: { guarded: { system_prompt: 'Careful.', tools: ['echo', 'shout'], max_steps: 4, requires_approval: ['echo'] } },
});

beforeEach(() => {
  // No test here may reach a network.
  vi.stubGlobal('fetch', () => { throw new Error('network blocked in tests'); });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

interface Rig {
  session: Session;
  agent: AgentRuntime;
  requests: LlmRequest[];
  counts: Record<string, number>;
}

async function rig(toolCalls = [{ id: 'c1', name: 'echo', arguments: { value: 'x' } }]): Promise<Rig> {
  const directory = await mkdtemp(join(tmpdir(), 'prism-harness-g354-'));
  const harness = new PrismHarness({
    drivers: { memory: () => new MemorySessionStore(), files: () => new FileSessionStore(directory) },
    stores: { ephemeral: 'memory', durable: 'files' },
  });
  const session = harness.for({ type: 'User', id: 1 }).session('support');
  await session.usingProvider('anthropic');
  await session.usingModel('claude-sonnet-5-5');
  const requests: LlmRequest[] = [];
  const counts: Record<string, number> = {};
  const tools = new ToolRegistry()
    .register({ name: 'echo', handle: (args) => { counts.echo = (counts.echo ?? 0) + 1; return `echoed:${String(args.value ?? '')}`; } })
    .register({ name: 'shout', handle: (args) => { counts.shout = (counts.shout ?? 0) + 1; return `SHOUTED:${String(args.value ?? '')}`; } });
  const client = async (request: LlmRequest): Promise<LlmResponse> => {
    requests.push(request);

    return requests.length === 1
      ? { text: '', finishReason: 'tool_calls', toolCalls }
      : { text: `Finished after ${requests.length - 1}.`, finishReason: 'stop' };
  };

  return { session, agent: new AgentRuntime({ client, modes, tools }), requests, counts };
}

async function drain(stream: AsyncGenerator<RunStep, AgentResponse, undefined>): Promise<void> {
  for (;;) {
    if ((await stream.next()).done === true) return;
  }
}

async function rows(session: Session): Promise<string> {
  return JSON.stringify((await session.thread().messages()).map(entry => entry.message));
}

type Path = (rig: Rig, decisions: ApprovalDecision[]) => Promise<void>;

const paths: Record<string, Path> = {
  recordApproval: async ({ session }, decisions) => {
    for (const decision of decisions) await recordApproval(session, decision.approvalId, decision.approved, decision.reason ?? null);
  },
  recordDecisions: async ({ session }, decisions) => recordDecisions(session, decisions),
  streamDecision: async ({ session, agent }, decisions) => drain(agent.streamDecision(session, decisions)),
};

const each = Object.keys(paths);

describe('a decision for an approval that is not pending (G-354)', () => {
  it.each(each)('%s: refuses a made-up id while a call is pending, and the real one still works afterwards', async (path) => {
    const r = await rig();
    const first = await r.agent.send(r.session, 'go');
    const before = await rows(r.session);

    await expect(paths[path]!(r, [{ approvalId: 'approval-never-issued', approved: true }]))
      .rejects.toMatchObject({ code: 'approval_not_pending' });

    expect(r.requests).toHaveLength(1);
    expect(await rows(r.session)).toBe(before);
    expect(r.counts).toEqual({});

    await recordApproval(r.session, first.pendingApprovals[0]!.id, true);
    const resumed = await r.agent.send(r.session, '');

    expect(r.counts).toEqual({ echo: 1 });
    expect(resumed.text).toBe('Finished after 1.');
    expect(JSON.stringify(r.requests[1]?.messages)).not.toContain('No approval response provided');
  });

  it.each(each)('%s: refuses a made-up id next to the real one, and records neither', async (path) => {
    const r = await rig();
    const first = await r.agent.send(r.session, 'go');
    const before = await rows(r.session);

    // recordApproval writes one at a time, so the real id is LAST: the made-up
    // one must be refused before anything is written.
    await expect(paths[path]!(r, [
      { approvalId: 'approval-never-issued', approved: true },
      { approvalId: first.pendingApprovals[0]!.id, approved: true },
    ])).rejects.toMatchObject({ code: 'approval_not_pending' });

    expect(r.requests).toHaveLength(1);
    expect(await rows(r.session)).toBe(before);
  });

  it.each(each)('%s: refuses a made-up id while nothing is pending: no request, nothing written', async (path) => {
    const r = await rig([{ id: 'c1', name: 'shout', arguments: { value: 'y' } }]);
    await r.agent.send(r.session, 'go');
    const before = await rows(r.session);

    expect(r.requests).toHaveLength(2);
    expect(r.counts).toEqual({ shout: 1 });

    await expect(paths[path]!(r, [{ approvalId: 'approval-never-issued', approved: true }]))
      .rejects.toMatchObject({ code: 'approval_not_pending' });

    expect(r.requests).toHaveLength(2);
    expect(await rows(r.session)).toBe(before);
    expect(r.counts).toEqual({ shout: 1 });
  });

  it.each(each)('%s: refuses a decision on a session that has never had a turn', async (path) => {
    const r = await rig();

    await expect(paths[path]!(r, [{ approvalId: 'approval-never-issued', approved: true }]))
      .rejects.toMatchObject({ code: 'approval_not_pending' });

    expect(r.requests).toHaveLength(0);
    expect(await rows(r.session)).toBe('[]');
  });

  it.each(each)('%s: the real id works exactly as before', async (path) => {
    const r = await rig();
    const first = await r.agent.send(r.session, 'go');

    await paths[path]!(r, [{ approvalId: first.pendingApprovals[0]!.id, approved: true }]);
    if (path !== 'streamDecision') await r.agent.send(r.session, '');

    expect(r.counts).toEqual({ echo: 1 });
    expect(r.requests).toHaveLength(2);
  });

  it.each(each)('%s: refuses a repeat of an id already answered: no request, nothing written', async (path) => {
    const r = await rig();
    const first = await r.agent.send(r.session, 'go');
    await recordApproval(r.session, first.pendingApprovals[0]!.id, true);
    await r.agent.send(r.session, '');
    const before = await rows(r.session);

    await expect(paths[path]!(r, [{ approvalId: first.pendingApprovals[0]!.id, approved: true }]))
      .rejects.toMatchObject({ code: 'approval_not_pending' });

    expect(r.requests).toHaveLength(2);
    expect(await rows(r.session)).toBe(before);
    expect(r.counts).toEqual({ echo: 1 });
  });

  it.each(each)('%s: refuses a repeat recorded before the run resumed', async (path) => {
    const r = await rig();
    const first = await r.agent.send(r.session, 'go');
    await recordApproval(r.session, first.pendingApprovals[0]!.id, false, 'no');
    const before = await rows(r.session);

    await expect(paths[path]!(r, [{ approvalId: first.pendingApprovals[0]!.id, approved: true }]))
      .rejects.toMatchObject({ code: 'approval_not_pending' });

    expect(r.requests).toHaveLength(1);
    expect(await rows(r.session)).toBe(before);
  });

  it('refuses the same id twice in one batch', async () => {
    const r = await rig();
    const first = await r.agent.send(r.session, 'go');
    const before = await rows(r.session);
    const id = first.pendingApprovals[0]!.id;

    await expect(recordDecisions(r.session, [{ approvalId: id, approved: false }, { approvalId: id, approved: true }]))
      .rejects.toMatchObject({ code: 'approval_not_pending' });

    expect(r.requests).toHaveLength(1);
    expect(await rows(r.session)).toBe(before);
  });
});
