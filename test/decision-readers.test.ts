import { describe, expect, it } from 'vitest';
import {
  AgentRuntime,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  ToolRegistry,
  recordApproval,
  toolResultRow,
  type JsonObject,
  type JsonValue,
  type LlmRequest,
  type LlmResponse,
  type Session,
} from '../src/index.js';

// G-411: what the decision readers say must match what happened to the call.
// A denial a host writes straight into the thread after the call started does
// not become the decision recorded beside its outcome: the answer says the
// call was interrupted, and its decision is the approval it started under.
// Offline.

const INTERRUPTED = 'Tool call was interrupted, it may have run';

/** A durable store that refuses ONE write, when armed: the first whose newest row is a result row holding `marker`. */
class FailingOnce extends MemorySessionStore {
  constructor(private armed: boolean, private readonly marker: string) {
    super();
  }

  override durability(): 'durable' {
    return 'durable';
  }

  override async put(key: string, payload: JsonObject, ttlSeconds: number | null = null): Promise<void> {
    const messages = Array.isArray(payload.messages) ? payload.messages as JsonValue[] : [];
    const newest = (messages.at(-1) as JsonObject | undefined)?.message as JsonObject | undefined;
    if (this.armed && newest?.type === 'tool_result' && JSON.stringify(newest.tool_results ?? []).includes(this.marker)) {
      this.armed = false;
      throw new Error('The store refused the write (scripted).');
    }
    await super.put(key, payload, ttlSeconds);
  }
}

function rig(options: { failResult?: boolean; maxSteps?: number } = {}) {
  const harness = new PrismHarness({
    drivers: { memory: () => new MemorySessionStore(), durable: () => new FailingOnce(options.failResult === true, 'ran:1') },
    stores: { ephemeral: 'memory', durable: 'durable' },
  });
  const session = harness.for({ type: 'User', id: 1 }).session('decision-readers');
  const ran: number[] = [];
  const requests: LlmRequest[] = [];
  let model = 0;
  const agent = new AgentRuntime({
    modes: new ModeRegistry({ default: 'guarded', modes: { guarded: { system_prompt: 'Careful.', tools: ['echo'], max_steps: options.maxSteps ?? 6, requires_approval: ['echo'] } } }),
    tools: new ToolRegistry().register({ name: 'echo', handle: (args: JsonObject) => { ran.push(Number(args.n)); return `ran:${String(args.n)}`; } }),
    client: async (request: LlmRequest): Promise<LlmResponse> => {
      requests.push(JSON.parse(JSON.stringify(request)) as LlmRequest);
      return ++model === 1
        ? { text: '', finishReason: 'tool_calls', toolCalls: [{ id: 'c1', name: 'echo', arguments: { n: 1 } }] }
        : { text: 'done', finishReason: 'stop' };
    },
  });
  return { session, agent, ran, requests };
}

async function rows(session: Session): Promise<JsonObject[]> {
  return (await session.thread().messages()).map((entry) => entry.message);
}

/** A denial written straight into the thread, as a host could, not through the decision API. */
async function hostDenial(session: Session, approvalId: string): Promise<void> {
  await session.thread().record([toolResultRow([], [{ approval_id: approvalId, approved: false, reason: 'host wrote a denial' }])]);
}

describe('G-411 F1: a denial written straight into the thread after the call started', () => {
  it('the replay answers interrupted with the approval the call started under; nothing is stored for the call (G-414)', async () => {
    const r = rig({ failResult: true });
    const parked = await r.agent.send(r.session, 'go');
    const approvalId = parked.pendingApprovals[0]!.id;
    await recordApproval(r.session, approvalId, true);
    await r.agent.send(r.session, '').catch(() => undefined);
    expect(r.ran).toEqual([1]);
    await hostDenial(r.session, approvalId);
    await r.agent.send(r.session, '');
    expect(r.ran).toEqual([1]);

    // Replay-only (G-414): no stored outcome for the call.
    expect((await rows(r.session)).filter((row) => row.type === 'tool_result' && (row.tool_results as JsonObject[]).length > 0)).toEqual([]);

    const sent = (r.requests.at(-1)!.messages as JsonObject[]).filter((row) => row.type === 'tool_result' && (row.tool_results as JsonObject[]).length > 0);
    expect(sent.map((row) => (row.tool_results as JsonObject[]).map((entry) => entry.result))).toEqual([[INTERRUPTED]]);
    expect(sent.map((row) => row.tool_approval_responses)).toEqual([[{ approval_id: approvalId, approved: true, reason: null }]]);
  });

  it('a denial written before the call started still keeps it from running', async () => {
    const r = rig();
    const parked = await r.agent.send(r.session, 'go');
    const approvalId = parked.pendingApprovals[0]!.id;
    await recordApproval(r.session, approvalId, true);
    await hostDenial(r.session, approvalId);
    await r.agent.send(r.session, '');
    expect(r.ran).toEqual([]);
  });
});
