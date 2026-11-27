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
  type Session,
} from '../src/index.js';

// G-405: the replay a provider is sent carries each result of a step exactly
// once, in its step's segment, whatever rows sit between the result rows: a
// new message, a host's system row, the decisions. Offline; every store is in
// memory.

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
  requests: LlmRequest[];
}

function rig(): Rig {
  const harness = new PrismHarness({
    drivers: { memory: () => new MemorySessionStore(), durable: () => new DurableMemory() },
    stores: { ephemeral: 'memory', durable: 'durable' },
  });
  const session = harness.for({ type: 'User', id: 1 }).session('replay-once');
  const requests: LlmRequest[] = [];
  const agent = new AgentRuntime({
    modes: new ModeRegistry({ default: 'guarded', modes: { guarded: { system_prompt: 'Careful.', tools: ['echo', 'free'], max_steps: 6, requires_approval: ['echo'] } } }),
    tools: new ToolRegistry().register({ name: 'echo', handle: () => 'ok' }).register({ name: 'free', handle: () => 'free ok' }),
    client: async (request: LlmRequest): Promise<LlmResponse> => {
      requests.push(request);
      return requests.length === 1
        ? { text: '', finishReason: 'tool_calls', toolCalls: [{ id: 'f1', name: 'free', arguments: { n: 1 } }, { id: 'e1', name: 'echo', arguments: { n: 2 } }] }
        : { text: 'done', finishReason: 'stop' };
    },
  });
  return { session, agent, requests };
}

/** Every tool result the last request carried, as [id, name, n]. */
function sent(r: Rig): [string, string, number][] {
  return (r.requests.at(-1)!.messages as JsonObject[]).filter((row) => row.type === 'tool_result')
    .flatMap((row) => row.tool_results as JsonObject[])
    .map((entry) => [String(entry.tool_call_id), String(entry.tool_name), Number((entry.args as JsonObject).n)]);
}

describe('G-405: each result of a step is replayed exactly once', () => {
  it('a new message after the decision: the resume and the new turn send each result once', async () => {
    const r = rig();
    const [pending] = (await r.agent.send(r.session, 'go')).pendingApprovals;
    await recordDecisions(r.session, [{ approvalId: pending!.id, approved: true }]);
    await r.agent.send(r.session, 'And another thing.');

    expect(sent(r).sort()).toEqual([['e1', 'echo', 2], ['f1', 'free', 1]]);
  });

  for (const [label, row] of [
    ['a system row', { type: 'system', content: 'A host note.' }],
    ['an empty result row', { type: 'tool_result', tool_results: [], tool_approval_responses: [] }],
  ] as [string, JsonObject][]) {
    it(`${label} between the decision and the resume: each result is sent once`, async () => {
      const r = rig();
      const [pending] = (await r.agent.send(r.session, 'go')).pendingApprovals;
      await recordDecisions(r.session, [{ approvalId: pending!.id, approved: true }]);
      await r.session.thread().record([row]);
      await r.agent.send(r.session, '');

      expect(sent(r).sort()).toEqual([['e1', 'echo', 2], ['f1', 'free', 1]]);
    });

    it(`${label} between the park and the decision: each result is sent once`, async () => {
      const r = rig();
      const [pending] = (await r.agent.send(r.session, 'go')).pendingApprovals;
      await r.session.thread().record([row]);
      await recordDecisions(r.session, [{ approvalId: pending!.id, approved: true }]);
      await r.agent.send(r.session, '');

      expect(sent(r).sort()).toEqual([['e1', 'echo', 2], ['f1', 'free', 1]]);
    });
  }
});
