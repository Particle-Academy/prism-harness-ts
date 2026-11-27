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

// G-397: a result is matched to its call by id and by the call's arguments in
// the form the harness uses them (parsed), whatever form they are stored in:
// a JSON string, as the PHP reference stores them, or an object. A result in
// the newest step's segment that matches one of its calls answers that call,
// never an earlier call that reuses the id and arguments. So an approved call
// runs once, however many resumes follow. Offline; every store is in memory.

const network = vi.fn(() => { throw new Error('A test tried to use the network.'); });
beforeEach(() => { vi.stubGlobal('fetch', network); });
afterEach(() => { expect(network).not.toHaveBeenCalled(); vi.unstubAllGlobals(); });

class DurableMemory extends MemorySessionStore {
  override durability(): 'durable' {
    return 'durable';
  }
}

const tools = (ran: number[]): ToolRegistry => new ToolRegistry()
  .register({ name: 'echo', handle: (args: JsonObject) => { ran.push(Number(args.n)); return `ran:${String(args.n)}`; } })
  .register({ name: 'free', handle: () => 'free' });

function mode(maxSteps: number): ModeRegistry {
  return new ModeRegistry({ default: 'guarded', modes: { guarded: { system_prompt: 'Careful.', tools: ['echo', 'free'], max_steps: maxSteps, requires_approval: ['echo'] } } });
}

function session(): Session {
  const harness = new PrismHarness({
    drivers: { memory: () => new MemorySessionStore(), durable: () => new DurableMemory() },
    stores: { ephemeral: 'memory', durable: 'durable' },
  });
  return harness.for({ type: 'User', id: 1 }).session('stored-arguments');
}

const step = (id: string, args: JsonValue, approval: string): JsonObject => ({
  type: 'assistant',
  content: '',
  tool_calls: [{ id, name: 'echo', arguments: args, result_id: null, reasoning_id: null, reasoning_summary: null }],
  additional_content: {},
  tool_approval_requests: [{ approval_id: approval, tool_call_id: id, arguments_format: ARGUMENTS_AS_SENT }],
  harness_tool_names: ['echo', 'free'],
});

/** An earlier answered step with `earlier` arguments (or none), then a parked step whose one call has `parked` arguments. */
async function thread(s: Session, earlier: JsonValue | null, parked: JsonValue): Promise<void> {
  const rows: JsonObject[] = [];
  if (earlier !== null) {
    rows.push(
      { type: 'user', content: 'one' },
      step('c1', earlier, 'apr_e'),
      toolResultRow([toolResultEntry({ id: 'c1', name: 'echo', arguments: { n: 1 } }, 'ran:old')], [{ approval_id: 'apr_e', approved: true, reason: null }]),
      { type: 'assistant', content: 'ok', tool_calls: [], tool_approval_requests: [] },
    );
  }
  rows.push({ type: 'user', content: 'two' }, step('c1', parked, 'apr_s'));
  await s.thread().record(rows, 'run_x');
}

async function outcomesAfterPark(s: Session): Promise<string[]> {
  const rows = (await s.thread().messages()).map((entry) => entry.message);
  let parked = rows.length - 1;
  while (parked >= 0 && !(rows[parked]!.type === 'assistant' && Array.isArray(rows[parked]!.tool_calls) && (rows[parked]!.tool_calls as JsonValue[]).length > 0)) parked -= 1;
  return [...new Set(rows.slice(parked + 1).filter((row) => row.type === 'tool_result')
    .flatMap((row) => row.tool_results as JsonObject[]).map((entry) => String(entry.result)))];
}

const SHAPES: [string, JsonValue | null, JsonValue][] = [
  ['arguments stored as a JSON string, an earlier step reusing the id with the same arguments', { n: 1 }, '{"n":1}'],
  ['arguments stored as a JSON string on both steps', '{"n":1}', '{"n":1}'],
  ['arguments stored as objects, an earlier step reusing the id with the same arguments', { n: 1 }, { n: 1 }],
  ['arguments stored as a JSON string, no earlier step', null, '{"n":1}'],
];

describe('G-397: an approved call runs once, whatever form its arguments are stored in', () => {
  for (const [label, earlier, parked] of SHAPES) {
    it(`${label}; the run's steps are spent, so no resume calls the model`, async () => {
      const s = session();
      const ran: number[] = [];
      let model = 0;
      const agent = new AgentRuntime({ modes: mode(1), tools: tools(ran), client: async (): Promise<LlmResponse> => { model += 1; return { text: 'done', finishReason: 'stop' }; } });
      await thread(s, earlier, parked);
      await recordApproval(s, 'apr_s', true);
      for (let i = 0; i < 3; i += 1) await agent.send(s, '');

      expect(model).toBe(0);
      expect(ran).toEqual([1]);
      expect(await outcomesAfterPark(s)).toEqual(['ran:1']);
    });

    it(`${label}; the provider fails after each resolution`, async () => {
      const s = session();
      const ran: number[] = [];
      const agent = new AgentRuntime({ modes: mode(6), tools: tools(ran), client: async (): Promise<LlmResponse> => { throw new Error('The provider is down (scripted).'); } });
      await thread(s, earlier, parked);
      await recordApproval(s, 'apr_s', true);
      for (let i = 0; i < 3; i += 1) await expect(agent.send(s, '')).rejects.toThrow('scripted');

      expect(ran).toEqual([1]);
      expect(await outcomesAfterPark(s)).toEqual(['ran:1']);
    });

    it(`${label}; the model answers after the resolution: one run, one outcome`, async () => {
      const s = session();
      const ran: number[] = [];
      const agent = new AgentRuntime({ modes: mode(6), tools: tools(ran), client: async (): Promise<LlmResponse> => ({ text: 'done', finishReason: 'stop' }) });
      await thread(s, earlier, parked);
      await recordApproval(s, 'apr_s', true);
      for (let i = 0; i < 3; i += 1) await agent.send(s, '');

      expect(ran).toEqual([1]);
      expect(await outcomesAfterPark(s)).toEqual(['ran:1']);
    });
  }
});
