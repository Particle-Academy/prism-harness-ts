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

// G-401, the two shapes every line is checked for: a decision is stored, no
// turn acts on it (none runs, or one fails before acting), then the other
// decision arrives. Whatever rule decides which decision stands, a call a
// person denied never runs. Offline; every store is in memory.

const network = vi.fn(() => { throw new Error('A test tried to use the network.'); });
beforeEach(() => { vi.stubGlobal('fetch', network); });
afterEach(() => { expect(network).not.toHaveBeenCalled(); vi.unstubAllGlobals(); });

class DurableMemory extends MemorySessionStore {
  override durability(): 'durable' {
    return 'durable';
  }
}

async function parked(): Promise<{ session: Session; agent: AgentRuntime; ran: number[]; approvalId: string }> {
  const harness = new PrismHarness({
    drivers: { memory: () => new MemorySessionStore(), durable: () => new DurableMemory() },
    stores: { ephemeral: 'memory', durable: 'durable' },
  });
  const session = harness.for({ type: 'User', id: 1 }).session('g401-a-d');
  const ran: number[] = [];
  let model = 0;
  const agent = new AgentRuntime({
    modes: new ModeRegistry({ default: 'guarded', modes: { guarded: { system_prompt: 'Careful.', tools: ['echo'], max_steps: 6, requires_approval: ['echo'] } } }),
    tools: new ToolRegistry().register({ name: 'echo', handle: (args: JsonObject) => { ran.push(Number(args.n)); return 'ok'; } }),
    client: async (): Promise<LlmResponse> => (++model === 1 ? { text: '', finishReason: 'tool_calls', toolCalls: [{ id: 'c1', name: 'echo', arguments: { n: 1 } }] } : { text: 'done', finishReason: 'stop' }),
  });
  const [pending] = (await agent.send(session, 'go')).pendingApprovals;
  return { session, agent, ran, approvalId: pending!.id };
}

const failedTurn = async (r: { session: Session; agent: AgentRuntime }): Promise<void> => {
  await expect(r.agent.send(r.session, '', undefined, undefined, [{ type: 'image', url: 'https://example.test/x.png' }])).rejects.toThrow();
};

describe('G-401 (a): a stored denial no turn acted on, then an approval', () => {
  for (const between of ['no turn', 'a turn that fails before acting'] as const) {
    it(`${between} between them: the denied call never runs`, async () => {
      const r = await parked();
      await recordApproval(r.session, r.approvalId, false, 'no');
      if (between !== 'no turn') await failedTurn(r);
      await recordApproval(r.session, r.approvalId, true).catch(() => undefined);
      for (let i = 0; i < 3; i += 1) await r.agent.send(r.session, '');

      expect(r.ran).toEqual([]);
    });
  }
});

describe('G-401 (d): a stored approval no turn acted on, then a denial', () => {
  for (const between of ['no turn', 'a turn that fails before acting'] as const) {
    it(`${between} between them: the denied call never runs`, async () => {
      const r = await parked();
      await recordApproval(r.session, r.approvalId, true);
      if (between !== 'no turn') await failedTurn(r);
      await recordApproval(r.session, r.approvalId, false, 'no').catch(() => undefined);
      for (let i = 0; i < 3; i += 1) await r.agent.send(r.session, '');

      expect(r.ran).toEqual([]);
    });
  }
});
