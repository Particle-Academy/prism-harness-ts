import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  ToolRegistry,
  recordApproval,
  type JsonObject,
  type JsonValue,
  type LlmResponse,
  type Session,
} from '../src/index.js';
import { ARGUMENTS_AS_SENT, toolResultRow } from '../src/thread-rows.js';

// G-408: one rule decides which rows are tool-call steps, for every reader:
// an assistant row with at least one tool call that is an object. A row whose
// tool_calls hold no objects (strings, numbers, null, arrays) is an assistant
// row without tool calls. The calls a resume answers, the results it counts,
// the replaced check and the step its result row names all come from that one
// selection, so an approved call runs at most once and ends with exactly one
// outcome. Offline; every store is in memory.

const REPLACED = 'Tool call was not run because a later step replaced it';

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

/** A run whose one step is spent when it parks: a resume makes no model call. */
function spent(): Rig {
  const harness = new PrismHarness({
    drivers: { memory: () => new MemorySessionStore(), durable: () => new DurableMemory() },
    stores: { ephemeral: 'memory', durable: 'durable' },
  });
  const ran: number[] = [];
  const agent = new AgentRuntime({
    modes: new ModeRegistry({ default: 'guarded', modes: { guarded: { system_prompt: 'Careful.', tools: ['echo', 'free'], max_steps: 1, requires_approval: ['echo'] } } }),
    tools: new ToolRegistry().register({ name: 'echo', handle: (args: JsonObject) => { ran.push(Number(args.n)); return `ran:${String(args.n)}`; } }).register({ name: 'free', handle: (args: JsonObject) => { ran.push(Number(args.n)); return 'free'; } }),
    client: async (): Promise<LlmResponse> => ({ text: 'done', finishReason: 'stop' }),
  });
  return { session: harness.for({ type: 'User', id: 1 }).session('non-object'), agent, ran };
}

const parkedStep: JsonObject = {
  type: 'assistant',
  content: '',
  tool_calls: [{ id: 'c1', name: 'echo', arguments: { n: 1 }, result_id: null, reasoning_id: null, reasoning_summary: null }],
  additional_content: {},
  tool_approval_requests: [{ approval_id: 'apr_s', tool_call_id: 'c1', arguments_format: ARGUMENTS_AS_SENT }],
  harness_tool_names: ['echo', 'free'],
};

const JUNK: [string, JsonValue][] = [
  ['strings', ['x']],
  ['a JSON string of a call', ['{"id":"c1","name":"echo","arguments":{"n":1}}']],
  ['null', [null]],
  ['a number', [7]],
  ['an array of strings', [['c1', 'echo']]],
  ['a string, not an array', '{"id":"c1"}'],
];

async function outcomes(session: Session): Promise<string[]> {
  return (await session.thread().messages()).map((entry) => entry.message).filter((row) => row.type === 'tool_result' && Array.isArray(row.tool_results))
    .flatMap((row) => row.tool_results as JsonObject[]).filter((entry) => (entry.args as JsonObject).n === 1).map((entry) => String(entry.result));
}

describe('G-408: an assistant row whose tool_calls hold no objects is not a tool-call step', () => {
  for (const [label, junk] of JUNK) {
    for (const requests of [true, false]) {
      it(`after the approval, the host records a copy of the parked step whose tool_calls are ${label}${requests ? ', keeping its approval requests' : ', with no requests'}: the approved call runs at most once, has exactly one outcome, and later resumes answer nothing again`, async () => {
        const r = spent();
        await r.session.thread().record([{ type: 'user', content: 'go' }, parkedStep], 'run_x');
        await recordApproval(r.session, 'apr_s', true);
        await r.session.thread().record([{
          type: 'assistant', content: '', tool_calls: junk, additional_content: {},
          tool_approval_requests: requests ? [{ approval_id: 'apr_s', tool_call_id: 'c1', arguments_format: ARGUMENTS_AS_SENT }] : [],
          harness_tool_names: ['echo', 'free'],
        }]);
        await r.agent.send(r.session, '').catch(() => undefined);
        // Result rows: a later resume answers nothing again. (With nothing
        // parked, a resume is a model turn and may record its answer.)
        const resultRows = async (): Promise<number> => (await r.session.thread().messages()).filter((entry) => entry.message.type === 'tool_result').length;
        const rows = await resultRows();
        for (let i = 0; i < 2; i += 1) await r.agent.send(r.session, '').catch(() => undefined);

        expect(r.ran.length).toBeLessThanOrEqual(1);
        const got = await outcomes(r.session);
        expect(got).toHaveLength(1);
        // The host's assistant row replaced the parked step before it ran.
        expect(got).toEqual([REPLACED]);
        expect(r.ran).toEqual([]);
        expect(await resultRows()).toBe(rows);
      });
    }
  }

  it('a host copy of the parked step that mixes non-objects with a NEW call and keeps the approval request: the request asks for nothing, so the approval never runs the new call', async () => {
    const r = spent();
    await r.session.thread().record([{ type: 'user', content: 'go' }, parkedStep], 'run_x');
    await recordApproval(r.session, 'apr_s', true);
    await r.session.thread().record([{
      type: 'assistant', content: '', additional_content: {}, harness_tool_names: ['echo', 'free'],
      tool_calls: ['x', { id: 'c1', name: 'free', arguments: { n: 9 }, result_id: null, reasoning_id: null, reasoning_summary: null }, null],
      tool_approval_requests: [{ approval_id: 'apr_s', tool_call_id: 'c1', arguments_format: ARGUMENTS_AS_SENT }],
    }]);
    // The approval again, written straight to the thread after the copy.
    await r.session.thread().record([toolResultRow([], [{ approval_id: 'apr_s', approved: true, reason: null }])]);
    for (let i = 0; i < 3; i += 1) await r.agent.send(r.session, '').catch(() => undefined);

    expect(r.ran).toEqual([]);
    expect(await outcomes(r.session)).toEqual([REPLACED]);
  });

  it('a parked step whose tool_calls mix an object call with non-objects is a step: its object call is decided, once', async () => {
    const r = spent();
    await r.session.thread().record([{ type: 'user', content: 'go' }, { ...parkedStep, tool_calls: ['x', ...(parkedStep.tool_calls as JsonObject[]), null, 3] }], 'run_x');
    await recordApproval(r.session, 'apr_s', true);
    for (let i = 0; i < 3; i += 1) await r.agent.send(r.session, '').catch(() => undefined);

    expect(r.ran).toEqual([1]);
    expect(await outcomes(r.session)).toEqual(['ran:1']);
  });
});
