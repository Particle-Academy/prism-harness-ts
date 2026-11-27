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
import { ARGUMENTS_AS_SENT, toolResultEntry, toolResultRow } from '../src/thread-rows.js';

// G-397, absolute segment ownership: a result counts only for calls of the
// step whose segment holds it (after that step, before the next tool-call
// step), matched by exact parsed arguments, then by id and name, then by id.
// An earlier step's gated calls with no result of their own get the replaced
// denial. Every gated call ends with exactly one outcome, and nothing runs
// twice. Offline; every store is in memory.

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

function rig(maxSteps: number, firstCalls: { id: string; n: number }[] = []): Rig {
  const harness = new PrismHarness({
    drivers: { memory: () => new MemorySessionStore(), durable: () => new DurableMemory() },
    stores: { ephemeral: 'memory', durable: 'durable' },
  });
  const session = harness.for({ type: 'User', id: 1 }).session('segments');
  const ran: number[] = [];
  let model = 0;
  const agent = new AgentRuntime({
    modes: new ModeRegistry({ default: 'guarded', modes: { guarded: { system_prompt: 'Careful.', tools: ['echo', 'free'], max_steps: maxSteps, requires_approval: ['echo'] } } }),
    tools: new ToolRegistry()
      .register({ name: 'echo', handle: (args: JsonObject) => { ran.push(Number(args.n)); return `ran:${String(args.n)}`; } })
      .register({ name: 'free', handle: () => 'free' }),
    client: async (): Promise<LlmResponse> => {
      model += 1;
      return model === 1 && firstCalls.length > 0
        ? { text: '', finishReason: 'tool_calls', toolCalls: firstCalls.map((call) => ({ id: call.id, name: 'echo', arguments: { n: call.n } })) }
        : { text: 'done', finishReason: 'stop' };
    },
  });
  return { session, agent, ran };
}

const step = (calls: [string, JsonValue, string][]): JsonObject => ({
  type: 'assistant',
  content: '',
  tool_calls: calls.map(([id, args]) => ({ id, name: 'echo', arguments: args, result_id: null, reasoning_id: null, reasoning_summary: null })),
  additional_content: {},
  tool_approval_requests: calls.map(([id, , approval]) => ({ approval_id: approval, tool_call_id: id, arguments_format: ARGUMENTS_AS_SENT })),
  harness_tool_names: ['echo', 'free'],
});

/** Every result row entry, in thread order, as [n, result] (n null when its arguments carry none). */
async function entries(session: Session): Promise<[number | null, string][]> {
  return (await session.thread().messages()).map((entry) => entry.message).filter((row) => row.type === 'tool_result')
    .flatMap((row) => (Array.isArray(row.tool_results) ? row.tool_results as JsonObject[] : [{ args: {}, result: row.result } as JsonObject]))
    .map((entry) => [typeof (entry.args as JsonObject).n === 'number' ? (entry.args as JsonObject).n as number : null, String(entry.result)]);
}

describe('G-397: a result counts only for the step whose segment holds it', () => {
  it('the newest call has a recorded result whose arguments differ from its own (lossy) but name an earlier same-id call: the newest call is done and never runs; the earlier call is denied as replaced', async () => {
    const r = rig(6);
    await r.session.thread().record([
      { type: 'user', content: 'go' },
      step([['c1', { n: 2 }, 'apr_old']]),
      step([['c1', { n: 1 }, 'apr_new']]),
      // The newest call's own result, recorded with arguments that do not match it.
      toolResultRow([toolResultEntry({ id: 'c1', name: 'echo', arguments: { n: 2 } }, 'ran:redacted')]),
    ]);
    await recordApproval(r.session, 'apr_new', true);
    for (let i = 0; i < 3; i += 1) await r.agent.send(r.session, '');

    expect(r.ran).toEqual([]);
    const all = await entries(r.session);
    expect(all.filter(([, result]) => result === REPLACED)).toEqual([[2, REPLACED]]);
    expect(all.filter(([n]) => n === 1)).toEqual([]);
  });

  it('a result recorded in the newest segment naming an earlier call that already has its replaced denial answers the newest call: one outcome each, nothing runs', async () => {
    const r = rig(6, [{ id: 'c1', n: 1 }]);
    await r.agent.send(r.session, 'go');
    await r.session.thread().record([step([['c1', { n: 3 }, 'apr_later']])]);
    await recordApproval(r.session, 'apr_later', false, 'wait');
    await r.agent.send(r.session, '');
    await r.session.thread().record([step([['c1', { n: 5 }, 'apr_l2']])]);
    await r.session.thread().record([toolResultRow([toolResultEntry({ id: 'c1', name: 'echo', arguments: { n: 1 } }, 'forged:1')])]);
    await recordApproval(r.session, 'apr_l2', true);
    for (let i = 0; i < 3; i += 1) await r.agent.send(r.session, '');

    expect(r.ran).toEqual([]);
    const all = await entries(r.session);
    // The first call: its one replaced denial. The second: its denial. The
    // third: the result recorded in its segment; no replaced denial for it.
    expect(all.filter(([n, result]) => n === 1 && result === REPLACED)).toHaveLength(1);
    expect([...new Set(all.filter(([n]) => n === 3).map(([, result]) => result))]).toEqual(['wait']);
    expect(all.filter(([n]) => n === 5)).toEqual([]);
  });

  for (const earlier of [true, false]) {
    it(`a 0.3.0 result row (no arguments) after a modern step answers that step's call by id${earlier ? ', not an earlier same-id call' : ''}: it never runs`, async () => {
      const r = rig(1);
      const rows: JsonObject[] = [{ type: 'user', content: 'go' }];
      if (earlier) rows.push(step([['c1', {}, 'apr_s']]));
      rows.push(step([['c1', { n: 1 }, 'apr_l']]));
      await r.session.thread().record(rows, 'run_x');
      await r.session.thread().record([{ type: 'tool_result', tool_call_id: 'c1', name: 'echo', result: 'legacy' }]);
      await recordApproval(r.session, 'apr_l', true);
      for (let i = 0; i < 3; i += 1) await r.agent.send(r.session, '');

      expect(r.ran).toEqual([]);
      const replaced = (await entries(r.session)).filter(([, result]) => result === REPLACED);
      expect(replaced).toHaveLength(earlier ? 1 : 0);
    });
  }

  it('a result naming a parked call, recorded after a later step reusing its id (arguments stored as a JSON string), answers the later call; the run is spent and resumed three times: nothing runs, the parked call is replaced once', async () => {
    const r = rig(1);
    await r.session.thread().record([{ type: 'user', content: 'go' }, step([['c1', { n: 1 }, 'apr_s']])], 'run_x');
    await r.session.thread().record([step([['c1', '{"n":3}', 'apr_later']])]);
    await r.session.thread().record([toolResultRow([toolResultEntry({ id: 'c1', name: 'echo', arguments: { n: 1 } }, 'forged:1')])]);
    await recordApproval(r.session, 'apr_later', true);
    for (let i = 0; i < 3; i += 1) await r.agent.send(r.session, '');

    expect(r.ran).toEqual([]);
    expect((await entries(r.session)).filter(([, result]) => result === REPLACED)).toEqual([[1, REPLACED]]);
  });

  it('arguments whose keys are in another order are the same arguments: the approved call runs once', async () => {
    const r = rig(1);
    await r.session.thread().record([{ type: 'user', content: 'one' },
      step([['c1', { n: 1, m: 2 }, 'apr_e']]),
      toolResultRow([toolResultEntry({ id: 'c1', name: 'echo', arguments: { m: 2, n: 1 } }, 'ran:old')], [{ approval_id: 'apr_e', approved: true, reason: null }]),
      { type: 'user', content: 'two' },
      step([['c1', '{"m":2,"n":1}', 'apr_s']])], 'run_x');
    await recordApproval(r.session, 'apr_s', true);
    for (let i = 0; i < 3; i += 1) await r.agent.send(r.session, '');

    expect(r.ran).toEqual([1]);
  });
});
