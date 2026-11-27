import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  ToolRegistry,
  recordApproval,
  recordDecisions,
  type JsonObject,
  type LlmResponse,
  type LlmToolCall,
  type Session,
} from '../src/index.js';
import { ARGUMENTS_AS_SENT } from '../src/thread-rows.js';

// A call's result belongs to the step that asked for it, even when another
// tool-call step was recorded while the call ran: the harness's result row
// names the step it answers. So the call that ran is never denied as
// replaced, and a later call that reuses its id is never credited with its
// result. Two identical gated calls in one step get one outcome each, and a
// spent run writes nothing more on later resumes. Deterministic: the writes
// happen from inside the running tool. Offline; every store is in memory.

const REPLACED = 'Tool call was not run because a later step replaced it';
const REPEATED = 'Tool call was not run because another call in this step has the same id';

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
  freeRan: number[];
  during: (() => Promise<void>) | null;
}

function rig(first: LlmToolCall[], maxSteps = 6): Rig {
  const harness = new PrismHarness({
    drivers: { memory: () => new MemorySessionStore(), durable: () => new DurableMemory() },
    stores: { ephemeral: 'memory', durable: 'durable' },
  });
  const r: Rig = { session: harness.for({ type: 'User', id: 1 }).session('own-step'), agent: undefined as unknown as AgentRuntime, ran: [], freeRan: [], during: null };
  const once = async (): Promise<void> => {
    const during = r.during;
    r.during = null;
    if (during !== null) await during();
  };
  let model = 0;
  r.agent = new AgentRuntime({
    modes: new ModeRegistry({ default: 'guarded', modes: { guarded: { system_prompt: 'Careful.', tools: ['echo', 'free'], max_steps: maxSteps, requires_approval: ['echo'] } } }),
    tools: new ToolRegistry()
      .register({ name: 'echo', handle: async (args: JsonObject) => { r.ran.push(Number(args.n)); await once(); return `ran:${String(args.n)}`; } })
      .register({ name: 'free', handle: async (args: JsonObject) => { r.freeRan.push(Number(args.n)); await once(); return `free:${String(args.n)}`; } }),
    client: async (): Promise<LlmResponse> => (++model === 1 ? { text: '', finishReason: 'tool_calls', toolCalls: first } : { text: 'done', finishReason: 'stop' }),
  });
  return r;
}

/** A tool-call step as a host records it through Thread.record. */
const hostStep = (calls: [string, number, string | null][]): JsonObject => ({
  type: 'assistant',
  content: '',
  tool_calls: calls.map(([id, n]) => ({ id, name: 'echo', arguments: { n }, result_id: null, reasoning_id: null, reasoning_summary: null })),
  additional_content: {},
  tool_approval_requests: calls.flatMap(([id, , approval]) => (approval === null ? [] : [{ approval_id: approval, tool_call_id: id, arguments_format: ARGUMENTS_AS_SENT }])),
  harness_tool_names: ['echo', 'free'],
});

/** Every result entry naming the call carrying `n`, in thread order. */
async function results(session: Session, n: number): Promise<string[]> {
  return (await session.thread().messages()).map((entry) => entry.message).filter((row) => row.type === 'tool_result')
    .flatMap((row) => row.tool_results as JsonObject[]).filter((entry) => (entry.args as JsonObject).n === n).map((entry) => String(entry.result));
}

describe('a tool-call step recorded while an approved call runs', () => {
  for (const reused of [true, false]) {
    it(`the host records a step${reused ? ' reusing the running call\'s id' : ''} from inside the running call: the call keeps its result, is never denied as replaced, and the later call is decided by its own approval`, async () => {
      const r = rig([{ id: 'c1', name: 'echo', arguments: { n: 1 } }]);
      const [pending] = (await r.agent.send(r.session, 'go')).pendingApprovals;
      await recordApproval(r.session, pending!.id, true);
      r.during = async () => { await r.session.thread().record([hostStep([[reused ? 'c1' : 'c9', 3, 'apr_later']])]); };
      await r.agent.send(r.session, '');
      await recordApproval(r.session, 'apr_later', false, 'not this');
      for (let i = 0; i < 2; i += 1) await r.agent.send(r.session, '');

      expect(r.ran).toEqual([1]);
      expect([...new Set(await results(r.session, 1))]).toEqual(['ran:1']);
      expect(await results(r.session, 1)).not.toContain(REPLACED);
      expect([...new Set(await results(r.session, 3))]).toEqual(['not this']);
    });
  }

  it('the host records a step reusing an ungated call\'s id while that call runs, when the step is taken: the ungated call keeps its result, and the later call is decided by its own approval', async () => {
    const r = rig([{ id: 'c1', name: 'free', arguments: { n: 1 } }, { id: 'c2', name: 'echo', arguments: { n: 2 } }]);
    r.during = async () => { await r.session.thread().record([hostStep([['c1', 3, 'apr_later']])]); };
    await r.agent.send(r.session, 'go');
    await recordApproval(r.session, 'apr_later', true);
    for (let i = 0; i < 2; i += 1) await r.agent.send(r.session, '');

    expect(r.freeRan).toEqual([1]);
    expect([...new Set(await results(r.session, 1))]).toEqual(['free:1']);
    expect(r.ran).toEqual([3]);
    expect([...new Set(await results(r.session, 3))]).toEqual(['ran:3']);
    expect([...new Set(await results(r.session, 2))]).toEqual([REPLACED]);
  });
});

describe('two identical gated calls in one step', () => {
  it('same id, tool and arguments: one outcome each, nothing runs, and a spent run writes nothing on later resumes', async () => {
    const r = rig([{ id: 'c1', name: 'echo', arguments: { n: 1 } }, { id: 'c1', name: 'echo', arguments: { n: 1 } }], 1);
    const parked = await r.agent.send(r.session, 'go');
    await recordDecisions(r.session, parked.pendingApprovals.map((p) => ({ approvalId: p.id, approved: true })));
    await r.agent.send(r.session, '');
    const rows = (await r.session.thread().messages()).length;
    for (let i = 0; i < 3; i += 1) await r.agent.send(r.session, '');

    expect(r.ran).toEqual([]);
    expect((await r.session.thread().messages()).length).toBe(rows);
    // One stored row holds both outcomes.
    const resolution = (await r.session.thread().messages()).map((entry) => entry.message)
      .filter((row) => row.type === 'tool_result' && (row.tool_results as JsonObject[]).length > 0);
    expect(resolution).toHaveLength(1);
    expect((resolution[0]!.tool_results as JsonObject[]).map((entry) => entry.result)).toEqual([REPEATED, REPEATED]);
  });
});
