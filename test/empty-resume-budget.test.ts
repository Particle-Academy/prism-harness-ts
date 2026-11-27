import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  ToolRegistry,
  recordApproval,
  type AgentResponse,
  type LlmResponse,
} from '../src/index.js';

// An empty send continues the turn it follows: its steps count from the
// turn's last user row, so no number of empty resumes takes a turn past
// max_steps, in steps or in model calls. Offline; every store is in memory.

const network = vi.fn(() => { throw new Error('A test tried to use the network.'); });
beforeEach(() => { vi.stubGlobal('fetch', network); });
afterEach(() => { expect(network).not.toHaveBeenCalled(); vi.unstubAllGlobals(); });

class DurableMemory extends MemorySessionStore {
  override durability(): 'durable' {
    return 'durable';
  }
}

describe('the step budget across empty resumes', () => {
  for (const after of ['an ungated call', 'an answer'] as const) {
    it(`max_steps 4: park, approve, the model then keeps returning ${after}; six empty sends after the run ends: steps and model calls never exceed 4`, async () => {
      const harness = new PrismHarness({
        drivers: { memory: () => new MemorySessionStore(), durable: () => new DurableMemory() },
        stores: { ephemeral: 'memory', durable: 'durable' },
      });
      const session = harness.for({ type: 'User', id: 1 }).session('budget');
      let model = 0, gated = 0, free = 0;
      const agent = new AgentRuntime({
        modes: new ModeRegistry({ default: 'guarded', modes: { guarded: { system_prompt: 'Careful.', tools: ['echo', 'free'], max_steps: 4, requires_approval: ['echo'] } } }),
        tools: new ToolRegistry().register({ name: 'echo', handle: () => { gated += 1; return 'ok'; } }).register({ name: 'free', handle: () => { free += 1; return 'ok'; } }),
        client: async (): Promise<LlmResponse> => {
          model += 1;
          if (model === 1) return { text: '', finishReason: 'tool_calls', toolCalls: [{ id: 'c1', name: 'echo', arguments: { n: 1 } }] };
          return after === 'an ungated call'
            ? { text: '', finishReason: 'tool_calls', toolCalls: [{ id: `f${model}`, name: 'free', arguments: { n: model } }] }
            : { text: 'done', finishReason: 'stop' };
        },
      });
      const [pending] = (await agent.send(session, 'go')).pendingApprovals;
      await recordApproval(session, pending!.id, true);
      const responses: AgentResponse[] = [await agent.send(session, '')];
      for (let i = 0; i < 6; i += 1) responses.push(await agent.send(session, ''));

      expect(model).toBeLessThanOrEqual(4);
      for (const response of responses) expect(response.steps).toBeLessThanOrEqual(4);
      expect(gated).toBe(1);
      expect(free).toBeLessThanOrEqual(3);
    });
  }

  it('a new message after the turn ends begins a new turn, with a budget of its own', async () => {
    const harness = new PrismHarness({
      drivers: { memory: () => new MemorySessionStore(), durable: () => new DurableMemory() },
      stores: { ephemeral: 'memory', durable: 'durable' },
    });
    const session = harness.for({ type: 'User', id: 1 }).session('budget-new');
    let model = 0;
    const agent = new AgentRuntime({
      modes: new ModeRegistry({ default: 'chat', modes: { chat: { system_prompt: 'Brief.', tools: ['free'], max_steps: 2 } } }),
      tools: new ToolRegistry().register({ name: 'free', handle: () => 'ok' }),
      client: async (): Promise<LlmResponse> => { model += 1; return { text: '', finishReason: 'tool_calls', toolCalls: [{ id: `f${model}`, name: 'free', arguments: {} }] }; },
    });
    expect((await agent.send(session, 'one')).finishReason).toBe('budget_exhausted');
    expect(model).toBe(2);
    expect((await agent.send(session, '')).finishReason).toBe('budget_exhausted');
    expect(model).toBe(2);
    expect((await agent.send(session, 'two')).finishReason).toBe('budget_exhausted');
    expect(model).toBe(4);
  });
});
