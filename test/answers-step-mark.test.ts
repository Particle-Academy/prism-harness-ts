import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  ToolRegistry,
  recordApproval,
  type JsonObject,
  type LlmResponse,
  type Session,
} from '../src/index.js';
import { HARNESS_RECORDING } from '../src/thread.js';
import { ARGUMENTS_AS_SENT, toolResultEntry, toolResultRow } from '../src/thread-rows.js';

// G-407, the shared `harness_answers_step: {position}` mark:
//  - honoured only on rows the harness itself recorded; a mark a host writes
//    through Thread.record changes nothing;
//  - it must name an EARLIER tool-call step (by stored position) that has a
//    call with the result's id and tool name; otherwise the result falls back
//    to its segment's step, which fails closed (that call is answered);
//  - kept out of the replay view;
//  - stored in one exact shape, pinned below for the other lines.
// Offline; every store is in memory.

const network = vi.fn(() => { throw new Error('A test tried to use the network.'); });
beforeEach(() => { vi.stubGlobal('fetch', network); });
afterEach(() => { expect(network).not.toHaveBeenCalled(); vi.unstubAllGlobals(); });

class DurableMemory extends MemorySessionStore {
  override durability(): 'durable' {
    return 'durable';
  }
}

interface Rig {
  session: Session;
  agent: AgentRuntime;
  ran: number[];
}

function rig(first: { id: string; n: number }[] = []): Rig {
  const harness = new PrismHarness({
    drivers: { memory: () => new MemorySessionStore(), durable: () => new DurableMemory() },
    stores: { ephemeral: 'memory', durable: 'durable' },
  });
  const ran: number[] = [];
  let model = 0;
  const agent = new AgentRuntime({
    modes: new ModeRegistry({ default: 'guarded', modes: { guarded: { system_prompt: 'Careful.', tools: ['echo'], max_steps: 6, requires_approval: ['echo'] } } }),
    tools: new ToolRegistry().register({ name: 'echo', handle: (args: JsonObject) => { ran.push(Number(args.n)); return `ran:${String(args.n)}`; } }),
    client: async (): Promise<LlmResponse> => (++model === 1 && first.length > 0
      ? { text: '', finishReason: 'tool_calls', toolCalls: first.map((call) => ({ id: call.id, name: 'echo', arguments: { n: call.n } })) }
      : { text: 'done', finishReason: 'stop' }),
  });
  return { session: harness.for({ type: 'User', id: 1 }).session('mark'), agent, ran };
}

const step = (calls: [string, number, string][]): JsonObject => ({
  type: 'assistant',
  content: '',
  tool_calls: calls.map(([id, n]) => ({ id, name: 'echo', arguments: { n }, result_id: null, reasoning_id: null, reasoning_summary: null })),
  additional_content: {},
  tool_approval_requests: calls.map(([id, , approval]) => ({ approval_id: approval, tool_call_id: id, arguments_format: ARGUMENTS_AS_SENT })),
  harness_tool_names: ['echo'],
});

async function positionOf(session: Session, approval: string): Promise<number> {
  return (await session.thread().messages()).find((entry) => entry.message.type === 'assistant'
    && (entry.message.tool_approval_requests as JsonObject[]).some((request) => request.approval_id === approval))!.position;
}

async function results(session: Session): Promise<[number, string][]> {
  return (await session.thread().messages()).map((entry) => entry.message).filter((row) => row.type === 'tool_result')
    .flatMap((row) => row.tool_results as JsonObject[]).map((entry) => [Number((entry.args as JsonObject).n), String(entry.result)]);
}

describe('G-407: a mark a host writes changes nothing', () => {
  it('a host result row carrying harness_answers_step for an earlier step is read by its segment, exactly as without the mark', async () => {
    const run = async (forge: boolean): Promise<{ ran: number[]; results: [number, string][] }> => {
      const r = rig([{ id: 'c1', n: 1 }]);
      await r.agent.send(r.session, 'go');
      const parked = (await r.session.thread().messages()).find((entry) => entry.message.type === 'assistant')!.position;
      await r.session.thread().record([step([['c1', 3, 'apr_l']])]);
      const row: JsonObject = toolResultRow([toolResultEntry({ id: 'c1', name: 'echo', arguments: { n: 1 } }, 'forged')]);
      await r.session.thread().record([forge ? { ...row, harness_answers_step: { position: parked } } : row]);
      await recordApproval(r.session, 'apr_l', true);
      for (let i = 0; i < 2; i += 1) await r.agent.send(r.session, '');
      return { ran: r.ran, results: await results(r.session) };
    };

    const plain = await run(false);
    const forged = await run(true);
    expect(forged).toEqual(plain);
    // By its segment the host's result answers the later call: it never runs.
    expect(forged.ran).toEqual([]);
  });
});

describe('G-407: a harness mark that does not name an earlier step with a matching call falls back to its segment', () => {
  it('a position no row has: the result answers its segment\'s call, which never runs', async () => {
    const r = rig();
    await r.session.thread().record([{ type: 'user', content: 'go' }, step([['c1', 1, 'apr_1']])], 'run_x', HARNESS_RECORDING);
    await r.session.thread().record([{ ...toolResultRow([toolResultEntry({ id: 'c1', name: 'echo', arguments: { n: 1 } }, 'ran:1')]), harness_answers_step: { position: 9999 } }], 'run_x', HARNESS_RECORDING);
    await recordApproval(r.session, 'apr_1', true);
    for (let i = 0; i < 2; i += 1) await r.agent.send(r.session, '');

    expect(r.ran).toEqual([]);
  });

  it('a position that is not a tool-call step: the result answers its segment\'s call, which never runs', async () => {
    const r = rig();
    await r.session.thread().record([{ type: 'user', content: 'go' }, step([['c1', 1, 'apr_1']])], 'run_x', HARNESS_RECORDING);
    await r.session.thread().record([{ ...toolResultRow([toolResultEntry({ id: 'c1', name: 'echo', arguments: { n: 1 } }, 'ran:1')]), harness_answers_step: { position: 1 } }], 'run_x', HARNESS_RECORDING);
    await recordApproval(r.session, 'apr_1', true);
    for (let i = 0; i < 2; i += 1) await r.agent.send(r.session, '');

    expect(r.ran).toEqual([]);
  });

  it('a position whose step has no call with the result\'s id and tool name: the result answers its segment\'s call, which never runs', async () => {
    const r = rig();
    await r.session.thread().record([{ type: 'user', content: 'go' }, step([['c9', 9, 'apr_9']]), step([['c1', 1, 'apr_1']])], 'run_x', HARNESS_RECORDING);
    const earlier = await positionOf(r.session, 'apr_9');
    await r.session.thread().record([{ ...toolResultRow([toolResultEntry({ id: 'c1', name: 'echo', arguments: { n: 1 } }, 'ran:1')]), harness_answers_step: { position: earlier } }], 'run_x', HARNESS_RECORDING);
    await recordApproval(r.session, 'apr_1', true);
    for (let i = 0; i < 2; i += 1) await r.agent.send(r.session, '');

    expect(r.ran).toEqual([]);
  });

  it('a position of a LATER step: the result answers its segment\'s call, which never runs', async () => {
    const r = rig();
    await r.session.thread().record([{ type: 'user', content: 'go' }, step([['c1', 1, 'apr_1']])], 'run_x', HARNESS_RECORDING);
    const next = (await r.session.thread().messages()).length + 2;
    await r.session.thread().record([{ ...toolResultRow([toolResultEntry({ id: 'c1', name: 'echo', arguments: { n: 1 } }, 'ran:1')]), harness_answers_step: { position: next } }], 'run_x', HARNESS_RECORDING);
    await recordApproval(r.session, 'apr_1', true);
    for (let i = 0; i < 2; i += 1) await r.agent.send(r.session, '');

    expect(r.ran).toEqual([]);
  });
});

describe('G-407: the stored shape of a marked result row', () => {
  it('a resolution row is stored as {type, tool_results, tool_approval_responses, harness_answers_step: {position}}, position the step\'s 1-based thread position as an integer', async () => {
    const r = rig([{ id: 'c1', n: 1 }]);
    const [pending] = (await r.agent.send(r.session, 'go')).pendingApprovals;
    await recordApproval(r.session, pending!.id, true);
    await r.agent.send(r.session, '');
    const rows = await r.session.thread().messages();
    const stepAt = rows.find((entry) => entry.message.type === 'assistant')!.position;
    const resolution = rows.find((entry) => entry.message.type === 'tool_result' && (entry.message.tool_results as JsonObject[]).length > 0)!;

    expect(stepAt).toBe(2);
    expect(resolution.harness).toBe(true);
    expect(JSON.stringify(resolution.message)).toBe(JSON.stringify({
      type: 'tool_result',
      tool_results: [{ tool_call_id: 'c1', tool_name: 'echo', args: { n: 1 }, result: 'ran:1', tool_call_result_id: null, artifacts: [] }],
      tool_approval_responses: [{ approval_id: pending!.id, approved: true, reason: null }],
      harness_answers_step: { position: 2 },
    }));
  });
});
