import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  ToolRegistry,
  recordDecisions,
  type JsonObject,
  type LlmRequest,
  type LlmResponse,
} from '../src/index.js';

// G-402: the replay a provider is sent keeps every result of a step. Results
// are folded by call id, tool name and arguments, occurrence by occurrence,
// never by call id alone, so two calls that share an id keep a result each.
// Offline; every store is in memory.

const network = vi.fn(() => { throw new Error('A test tried to use the network.'); });
beforeEach(() => { vi.stubGlobal('fetch', network); });
afterEach(() => { expect(network).not.toHaveBeenCalled(); vi.unstubAllGlobals(); });

class DurableMemory extends MemorySessionStore {
  override durability(): 'durable' {
    return 'durable';
  }
}

describe('G-402: the replay keeps a result for each call that shares an id', () => {
  for (const [label, calls] of [
    ['two gated calls with the same id and different arguments', [{ id: 'c', name: 'echo', arguments: { n: 1 } }, { id: 'c', name: 'echo', arguments: { n: 2 } }]],
    ['an ungated and a gated call with the same id and different arguments', [{ id: 'c', name: 'free', arguments: { n: 1 } }, { id: 'c', name: 'echo', arguments: { n: 2 } }]],
  ] as [string, JsonObject[]][]) {
    it(`${label}: the next request sends one result per call`, async () => {
      const harness = new PrismHarness({
        drivers: { memory: () => new MemorySessionStore(), durable: () => new DurableMemory() },
        stores: { ephemeral: 'memory', durable: 'durable' },
      });
      const session = harness.for({ type: 'User', id: 1 }).session('replay');
      const requests: LlmRequest[] = [];
      const agent = new AgentRuntime({
        modes: new ModeRegistry({ default: 'guarded', modes: { guarded: { system_prompt: 'Careful.', tools: ['echo', 'free'], max_steps: 6, requires_approval: ['echo'] } } }),
        tools: new ToolRegistry().register({ name: 'echo', handle: () => 'ok' }).register({ name: 'free', handle: (args: JsonObject) => `free:${String(args.n)}` }),
        client: async (request: LlmRequest): Promise<LlmResponse> => {
          requests.push(request);
          return requests.length === 1 ? { text: '', finishReason: 'tool_calls', toolCalls: calls as never } : { text: 'done', finishReason: 'stop' };
        },
      });
      const parked = await agent.send(session, 'go');
      await recordDecisions(session, parked.pendingApprovals.map((p) => ({ approvalId: p.id, approved: true })));
      await agent.send(session, '');

      const sent = requests.at(-1)!.messages as JsonObject[];
      const results = sent.filter((row) => row.type === 'tool_result').flatMap((row) => row.tool_results as JsonObject[]);
      expect(results.map((entry) => [entry.tool_call_id, entry.tool_name, (entry.args as JsonObject).n]).sort())
        .toEqual(calls.map((call) => [call.id, call.name, (call.arguments as JsonObject).n]).sort());
    });
  }
});
