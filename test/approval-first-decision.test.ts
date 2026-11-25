import { mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  FileSessionStore,
  HarnessError,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  ToolRegistry,
  recordApproval,
  toolResultRow,
  type JsonObject,
  type LlmRequest,
  type LlmResponse,
  type Session,
} from '../src/index.js';

// One answer per approval, and the same answer for every reader. Two decisions
// for one id racing each other (a double click, two reviewers, a retry) must
// not both be recorded, and a thread that already holds two (written before
// this was enforced) must be read as its FIRST answer by the resume, the
// tool scope and the plan alike.

const modes = new ModeRegistry({
  default: 'guarded',
  modes: { guarded: { system_prompt: 'Careful.', tools: ['echo'], max_steps: 4, requires_approval: ['echo'] } },
});

beforeEach(() => {
  vi.stubGlobal('fetch', () => { throw new Error('network blocked in tests'); });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

async function rig(): Promise<{ session: Session; agent: AgentRuntime; requests: LlmRequest[]; ran: () => number }> {
  const directory = await mkdtemp(join(tmpdir(), 'prism-harness-first-decision-'));
  const harness = new PrismHarness({
    drivers: { memory: () => new MemorySessionStore(), files: () => new FileSessionStore(directory) },
    stores: { ephemeral: 'memory', durable: 'files' },
  });
  const session = harness.for({ type: 'User', id: 1 }).session('support');
  await session.usingProvider('anthropic');
  await session.usingModel('claude-sonnet-5-5');
  const requests: LlmRequest[] = [];
  let runs = 0;
  const tools = new ToolRegistry().register({ name: 'echo', handle: () => { runs += 1; return 'echoed'; } });
  const client = async (request: LlmRequest): Promise<LlmResponse> => {
    requests.push(request);

    return requests.length === 1
      ? { text: '', finishReason: 'tool_calls', toolCalls: [{ id: 'c1', name: 'echo', arguments: { value: 'x' } }] }
      : { text: 'done', finishReason: 'stop' };
  };

  return { session, agent: new AgentRuntime({ client, modes, tools }), requests, ran: () => runs };
}

async function decisions(session: Session): Promise<JsonObject[]> {
  return (await session.thread().messages())
    .map((entry) => entry.message)
    .filter((row) => row.type === 'tool_result')
    .flatMap((row) => (Array.isArray(row.tool_approval_responses) ? row.tool_approval_responses : []) as JsonObject[]);
}

describe('one answer per approval', () => {
  it('records only one of two decisions racing for the same id, and refuses the other', async () => {
    const { session, agent, ran } = await rig();
    const id = (await agent.send(session, 'go')).pendingApprovals[0]!.id;

    const settled = await Promise.allSettled([
      recordApproval(session, id, false, 'no'),
      recordApproval(session, id, true),
    ]);

    expect(settled.filter((outcome) => outcome.status === 'fulfilled')).toHaveLength(1);
    const refused = settled.find((outcome) => outcome.status === 'rejected') as PromiseRejectedResult;
    expect((refused.reason as HarnessError).code).toBe('approval_not_pending');

    const recorded = await decisions(session);
    expect(recorded).toHaveLength(1);

    await agent.send(session, '');
    // Whichever answer was recorded is the one that acted.
    expect(ran()).toBe(recorded[0]!.approved === true ? 1 : 0);
  });

  it('acts on the FIRST of two stored decisions: a denial followed by an approval never runs the call', async () => {
    const { session, agent, requests, ran } = await rig();
    const id = (await agent.send(session, 'go')).pendingApprovals[0]!.id;
    await recordApproval(session, id, false, 'no');
    // A second answer for the same id, as a race before this fix could leave it.
    await session.thread().record([toolResultRow([], [{ approval_id: id, approved: true, reason: null }])]);

    await agent.send(session, '');

    expect(ran()).toBe(0);
    const results = (requests.at(-1)!.messages.filter((row) => row.type === 'tool_result')
      .flatMap((row) => row.tool_results as JsonObject[]));
    expect(results.map((entry) => entry.result)).toEqual(['no']);
  });
});
