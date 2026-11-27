import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  HarnessEvents,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  ToolRegistry,
  recordApproval,
  type HarnessEvent,
  type JsonObject,
  type LlmResponse,
  type RunStep,
  type Session,
} from '../src/index.js';

// Two checks on the release line. Offline; every store is in memory.
//  - A structured turn that stops for a person never hands back the model's
//    document: no document has been checked against the schema yet.
//  - A row recorded through the public Thread.record counts as outside the
//    run even while the runtime itself is recording that run's rows: the
//    harness's own recording is a property of each of its writes, never a
//    window that host code can fall into.

const REPLACED = 'Tool call was not run because a later step replaced it';
const SCHEMA: JsonObject = { type: 'object', properties: { title: { type: 'string' } }, required: ['title'] };

const network = vi.fn(() => { throw new Error('A test tried to use the network.'); });
beforeEach(() => { vi.stubGlobal('fetch', network); });
afterEach(() => { expect(network).not.toHaveBeenCalled(); vi.unstubAllGlobals(); });

class DurableMemory extends MemorySessionStore {
  override durability(): 'durable' {
    return 'durable';
  }
}

function session(): Session {
  const harness = new PrismHarness({
    drivers: { memory: () => new MemorySessionStore(), durable: () => new DurableMemory() },
    stores: { ephemeral: 'memory', durable: 'durable' },
  });
  return harness.for({ type: 'User', id: 1 }).session('checks');
}

const modes = new ModeRegistry({ default: 'guarded', modes: { guarded: { system_prompt: 'Careful.', tools: ['echo'], max_steps: 6, requires_approval: ['echo'] } } });

describe('a structured turn that parks for approval', () => {
  // As an Anthropic-shaped client returns it: the step that calls a tool also
  // carries text, here a document that breaks the schema.
  const step: LlmResponse = {
    text: '{"title": 12345, "role": "admin"}',
    finishReason: 'tool_calls',
    toolCalls: [{ id: 'c1', name: 'echo', arguments: { n: 1 } }],
    structured: { title: 12345, role: 'admin' },
  };

  // stop_reason tool_use maps to tool_calls; end_turn maps to stop, and the
  // tool_use block is still a call (prism-ts keeps tool_use blocks whatever
  // the stop reason). Both shapes keep the gated call and return no document.
  for (const finishReason of ['tool_calls', 'stop'] as const) {
    it(`sendStructured (finish ${finishReason}) keeps the gated call, parks on it, and returns no document`, async () => {
      const s = session();
      const ran: number[] = [];
      const agent = new AgentRuntime({ modes, tools: new ToolRegistry().register({ name: 'echo', handle: (args: JsonObject) => { ran.push(Number(args.n)); return 'ok'; } }), client: async () => ({ ...step, finishReason }) });
      const parked = await agent.sendStructured(s, 'go', SCHEMA);

      expect(parked.finishReason).toBe('awaiting_approval');
      expect(parked.pendingApprovals.map((p) => [p.toolCallId, p.tool])).toEqual([['c1', 'echo']]);
      expect((parked as { structured?: unknown }).structured).toBeUndefined();
      expect(ran).toEqual([]);
    });
  }

  it('and a structured resume after it is checked against the schema before anything is returned', async () => {
    const s = session();
    let model = 0;
    const agent = new AgentRuntime({
      modes,
      tools: new ToolRegistry().register({ name: 'echo', handle: () => 'ok' }),
      client: async () => (++model === 1 ? step : { text: '{"title": 12345, "role": "admin"}', finishReason: 'stop', structured: { title: 12345, role: 'admin' } }),
    });
    const [pending] = (await agent.sendStructured(s, 'go', SCHEMA)).pendingApprovals;
    await recordApproval(s, pending!.id, true);

    await expect(agent.sendStructured(s, '', SCHEMA)).rejects.toMatchObject({ code: 'structured_schema_violation' });
  });
});

describe('the harness\'s own recording is per write, never a window', () => {
  it('a listener that records a user row through Thread.record under the run\'s id while the runtime records the parked step: the row is outside, it supersedes, the call is denied as replaced, and a later decision is refused', async () => {
    const s = session();
    const events = new HarnessEvents();
    const pending: Promise<unknown>[] = [];
    events.listen((event: HarnessEvent) => {
      // Fires from inside the runtime's own turn, right after it recorded the
      // parked step's rows.
      if (event.type === 'run.stepped' && (event.step as RunStep).type === 'approval_request') {
        pending.push(s.thread().record([{ type: 'user', content: 'From a listener.' }], event.runId));
      }
    });
    const ran: number[] = [];
    let model = 0;
    const agent = new AgentRuntime({
      modes,
      events,
      stepEvents: true,
      tools: new ToolRegistry().register({ name: 'echo', handle: (args: JsonObject) => { ran.push(Number(args.n)); return 'ok'; } }),
      client: async (): Promise<LlmResponse> => (++model === 1
        ? { text: '', finishReason: 'tool_calls', toolCalls: [{ id: 'c1', name: 'echo', arguments: { n: 1 } }] }
        : { text: 'done', finishReason: 'stop' }),
    });
    const [approval] = (await agent.send(s, 'go')).pendingApprovals;
    await Promise.all(pending);
    expect(pending).toHaveLength(1);

    const rows = await s.thread().messages();
    const user = rows.find((entry) => entry.message.type === 'user' && entry.message.content === 'From a listener.')!;
    expect(user.runId).toBe(rows.find((entry) => entry.message.type === 'assistant')!.runId);
    expect(user.harness).toBeUndefined();

    await expect(recordApproval(s, approval!.id, true)).rejects.toMatchObject({ code: 'approval_not_pending' });
    await agent.send(s, '');
    expect(ran).toEqual([]);
    const results = (await s.thread().messages()).flatMap((entry) => entry.message.type === 'tool_result' ? entry.message.tool_results as JsonObject[] : []);
    expect(results.map((entry) => entry.result)).toEqual([REPLACED]);
  });
});
