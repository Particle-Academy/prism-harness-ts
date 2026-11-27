import { describe, expect, it } from 'vitest';
import {
  AgentRuntime,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  ToolRegistry,
  assistantRow,
  recordApproval,
  type JsonObject,
  type LlmRequest,
  type LlmResponse,
  type Session,
} from '../src/index.js';

// G-412. What the provider is sent: every tool call is answered by the tool result
// row directly after its step, even when a host row (a user row, or an
// assistant text row) was recorded while the call ran. A request with a tool
// call and no result straight after it is refused by the providers, on every
// later turn. Checked at the request seam, for an approved call (it has a
// started row) and an ungated one. Offline.

/** The in-memory store, declared durable for the test. */
class Durable extends MemorySessionStore {
  override durability(): 'durable' {
    return 'durable';
  }
}

type Kind = 'approved' | 'ungated';
type HostRow = 'user' | 'assistant text';

function rig(kind: Kind, hostRow: HostRow) {
  const harness = new PrismHarness({
    drivers: { memory: () => new MemorySessionStore(), durable: () => new Durable() },
    stores: { ephemeral: 'memory', durable: 'durable' },
  });
  const session = harness.for({ type: 'User', id: 1 }).session('replay-order');
  const requests: LlmRequest[] = [];
  const ran: number[] = [];
  let model = 0;
  const record = async (live: Session): Promise<void> => {
    await live.thread().record([hostRow === 'user'
      ? { type: 'user', content: 'a message that arrived while the tool ran' }
      : assistantRow('a note the host wrote while the tool ran', [])]);
  };
  const agent = new AgentRuntime({
    modes: new ModeRegistry({ default: 'm', modes: { m: { system_prompt: 'S.', tools: ['work'], max_steps: 6, requires_approval: kind === 'approved' ? ['work'] : [] } } }),
    tools: new ToolRegistry().register({ name: 'work', handle: async (args: JsonObject) => { ran.push(Number(args.n)); await record(session); return `worked:${String(args.n)}`; } }),
    client: async (request: LlmRequest): Promise<LlmResponse> => {
      requests.push(JSON.parse(JSON.stringify(request)) as LlmRequest);
      return ++model === 1
        ? { text: '', finishReason: 'tool_calls', toolCalls: [{ id: 'w1', name: 'work', arguments: { n: 1 } }] }
        : { text: 'done', finishReason: 'stop' };
    },
  });
  return { session, agent, requests, ran };
}

/** Every tool call in a request whose step is not followed directly by a result row answering it. */
function unanswered(request: LlmRequest): string[] {
  const rows = request.messages as JsonObject[];
  return rows.flatMap((row, at) => {
    if (row.type !== 'assistant' || !Array.isArray(row.tool_calls) || row.tool_calls.length === 0) return [];
    const next = rows[at + 1];
    const answered = new Set(next?.type === 'tool_result' ? (next.tool_results as JsonObject[]).map((entry) => entry.tool_call_id) : []);
    return (row.tool_calls as JsonObject[]).map((call) => String(call.id)).filter((id) => !answered.has(id)).map((id) => `request row ${at}: ${id}`);
  });
}

describe('G-412 replay order: a tool result goes directly after its own step', () => {
  for (const kind of ['approved', 'ungated'] as const) {
    for (const hostRow of ['user', 'assistant text'] as const) {
      it(`${kind} call, a ${hostRow} row recorded while it ran: every request answers every call after its step`, async () => {
        const r = rig(kind, hostRow);
        const first = await r.agent.send(r.session, 'go');
        if (kind === 'approved') {
          await recordApproval(r.session, first.pendingApprovals[0]!.id, true);
          await r.agent.send(r.session, '');
        }
        await r.agent.send(r.session, 'next message');
        await r.agent.send(r.session, '');

        expect(r.ran).toEqual([1]);
        expect(r.requests.length).toBeGreaterThanOrEqual(3);
        expect(r.requests.flatMap(unanswered)).toEqual([]);
        // The host row is still sent, after the result.
        const last = r.requests.at(-1)!.messages as JsonObject[];
        const step = last.findIndex((row) => row.type === 'assistant' && Array.isArray(row.tool_calls) && row.tool_calls.length > 0);
        const host = last.findIndex((row) => JSON.stringify(row).includes('while the tool ran'));
        expect(host).toBeGreaterThan(step + 1);
      });
    }
  }
});

/** A durable store that refuses ONE write: the first whose newest row is a result row holding `marker`. */
class RefusesOnce extends Durable {
  constructor(private armed: boolean, private readonly marker: string) {
    super();
  }

  override async put(key: string, payload: JsonObject, ttlSeconds: number | null = null): Promise<void> {
    const messages = Array.isArray(payload.messages) ? payload.messages as JsonObject[] : [];
    const newest = messages.at(-1)?.message as JsonObject | undefined;
    if (this.armed && newest?.type === 'tool_result' && JSON.stringify(newest.tool_results ?? []).includes(this.marker)) {
      this.armed = false;
      throw new Error('The store refused the write (scripted).');
    }
    await super.put(key, payload, ttlSeconds);
  }
}

function plainRig(options: { refuse?: string; gated?: boolean }) {
  const harness = new PrismHarness({
    drivers: { memory: () => new MemorySessionStore(), durable: () => new RefusesOnce(options.refuse !== undefined, options.refuse ?? '') },
    stores: { ephemeral: 'memory', durable: 'durable' },
  });
  const session = harness.for({ type: 'User', id: 1 }).session('replay-order-plain');
  const requests: LlmRequest[] = [];
  const ran: number[] = [];
  let model = 0;
  const agent = new AgentRuntime({
    modes: new ModeRegistry({ default: 'm', modes: { m: { system_prompt: 'S.', tools: ['work'], max_steps: 6, requires_approval: options.gated === true ? ['work'] : [] } } }),
    tools: new ToolRegistry().register({ name: 'work', handle: (args: JsonObject) => { ran.push(Number(args.n)); return `worked:${String(args.n)}`; } }),
    client: async (request: LlmRequest): Promise<LlmResponse> => {
      requests.push(JSON.parse(JSON.stringify(request)) as LlmRequest);
      return ++model === 1
        ? { text: '', finishReason: 'tool_calls', toolCalls: [{ id: 'w1', name: 'work', arguments: { n: 1 } }] }
        : { text: 'done', finishReason: 'stop' };
    },
  });
  return { session, agent, requests, ran };
}

async function resultsFor(session: Session, id: string): Promise<string[]> {
  return (await session.thread().messages()).flatMap((entry) => (entry.message.type === 'tool_result' ? entry.message.tool_results as JsonObject[] : []))
    .filter((entry) => entry.tool_call_id === id).map((entry) => String(entry.result));
}

describe('G-412: a call no resolution answers is still answered in every request', () => {
  it('an ungated call whose result write was refused: the next send stores it as interrupted, once, and never runs it again', async () => {
    const r = plainRig({ refuse: 'worked:1' });
    await r.agent.send(r.session, 'go').catch(() => undefined);
    expect(r.ran).toEqual([1]);
    await r.agent.send(r.session, 'next message');
    await r.agent.send(r.session, '');

    expect(r.ran).toEqual([1]);
    expect(await resultsFor(r.session, 'w1')).toEqual(['Tool call was interrupted, it may have run']);
    expect(r.requests.length).toBeGreaterThanOrEqual(3);
    expect(r.requests.flatMap(unanswered)).toEqual([]);
  });

  it('a parked step a host row superseded: its call is answered right after it in every request, and never runs', async () => {
    const r = plainRig({ gated: true });
    const parked = await r.agent.send(r.session, 'go');
    await recordApproval(r.session, parked.pendingApprovals[0]!.id, true);
    await r.session.thread().record([assistantRow('a note the host wrote', [])]);
    await r.agent.send(r.session, '');
    await r.agent.send(r.session, 'next message');

    expect(r.ran).toEqual([]);
    expect(r.requests.length).toBeGreaterThanOrEqual(3);
    expect(r.requests.flatMap(unanswered)).toEqual([]);
  });
});
