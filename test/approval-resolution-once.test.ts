import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  HarnessEvents,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  ToolRegistry,
  recordApproval,
  type Durability,
  type JsonObject,
  type LlmResponse,
  type Session,
} from '../src/index.js';

// An approved call runs once, whatever else fails around it. The shapes the
// 0.26.8 reviews found in other ports: a tool that throws during approval
// resolution must not leave the EARLIER resolved calls unrecorded, so the next
// resume re-runs them (P-3); and a stream resume of decisions recorded but not
// yet resolved must not run an approved call twice when an event listener
// throws (P-2). Driven through send / recordApproval / streamSteps /
// streamDecision with a scripted model; fetch is refused.

beforeEach(() => {
  vi.stubGlobal('fetch', () => { throw new Error('A test tried to use the network.'); });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

class DurableMemoryStore extends MemorySessionStore {
  override durability(): Durability {
    return 'durable';
  }
}

class Rig {
  readonly ran: Record<string, number> = {};
  readonly runtime: AgentRuntime;
  readonly harness: PrismHarness;
  readonly session: Session;
  #turns = 0;

  constructor(events?: HarnessEvents) {
    const modes = new ModeRegistry({
      default: 'guarded',
      modes: { guarded: { system_prompt: 'Careful.', tools: ['first', 'second'], requires_approval: ['first', 'second'], max_steps: 10 } },
    });
    const tools = new ToolRegistry()
      .register({ name: 'first', handle: () => { this.ran.first = (this.ran.first ?? 0) + 1; return 'first ran'; } })
      .register({ name: 'second', handle: () => { this.ran.second = (this.ran.second ?? 0) + 1; throw new Error('second exploded'); } });
    this.runtime = new AgentRuntime({
      modes,
      tools,
      ...(events === undefined ? {} : { events, stepEvents: true }),
      client: async (): Promise<LlmResponse> => this.#turns++ === 0
        ? { text: '', finishReason: 'tool_calls', toolCalls: [{ id: 'c1', name: 'first', arguments: {} }, { id: 'c2', name: 'second', arguments: {} }] }
        : { text: 'Done.', finishReason: 'stop' },
    });
    const durable = new DurableMemoryStore(), ephemeral = new MemorySessionStore();
    this.harness = new PrismHarness({
      drivers: { memory: () => ephemeral, durable: () => durable },
      stores: { ephemeral: 'memory', durable: 'durable' },
      runtime: this.runtime,
    });
    this.session = this.harness.for({ type: 'User', id: 1 }).session('support');
  }

  async pending(): Promise<string[]> {
    const rows = (await this.session.thread().messages()).map((entry) => entry.message);
    const answered = new Set(rows.flatMap((row) => Array.isArray(row.tool_approval_responses) ? row.tool_approval_responses as JsonObject[] : [])
      .map((response) => response.approval_id));

    return rows.flatMap((row) => Array.isArray(row.tool_approval_requests) ? row.tool_approval_requests as JsonObject[] : [])
      .map((request) => request.approval_id as string).filter((id) => !answered.has(id));
  }
}

const drain = async (steps: AsyncGenerator<unknown, unknown, undefined>) => { for await (const _ of steps) { /* drained */ } };

describe('an approved call runs once', () => {
  it('when a later approved tool in the same resume throws (P-3)', async () => {
    const r = new Rig();
    await r.runtime.send(r.session, 'go');
    for (const id of await r.pending()) await recordApproval(r.session, id, true);

    await r.runtime.send(r.session, '').catch(() => undefined);
    await r.runtime.send(r.session, '').catch(() => undefined);

    expect(r.ran).toEqual({ first: 1, second: 1 });
  });

  it.each(['streamSteps', 'streamDecision'] as const)('on a stream resume when an event listener throws (P-2, %s)', async (how) => {
    const events = new HarnessEvents();
    events.listen(() => { throw new Error('listener exploded'); });
    const r = new Rig(events);
    const quiet = process.emitWarning;
    process.emitWarning = () => {};
    try {
      await r.runtime.send(r.session, 'go');
      const ids = await r.pending();
      if (how === 'streamSteps') {
        for (const id of ids) await recordApproval(r.session, id, true);
        await drain(r.runtime.streamSteps(r.session, '')).catch(() => undefined);
      } else {
        await drain(r.runtime.streamDecision(r.session, ids.map((approvalId) => ({ approvalId, approved: true })))).catch(() => undefined);
      }
      await r.runtime.send(r.session, '').catch(() => undefined);
      await drain(r.runtime.streamSteps(r.session, '')).catch(() => undefined);
    } finally {
      process.emitWarning = quiet;
    }

    expect(r.ran).toEqual({ first: 1, second: 1 });
  });
});
