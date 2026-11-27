import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  ToolRegistry,
  recordApproval,
  type JsonObject,
  type LlmRequest,
  type LlmResponse,
  type Session,
} from '../src/index.js';

// G-414: "Tool call was interrupted, it may have run" is a REPLAY-ONLY answer.
// It is built for the request the model is sent and never stored. A call
// that is still running in one worker while another resolves the session
// keeps exactly ONE stored outcome, its real result, and every request the
// model is sent still answers every call. Two workers share the stores; the
// lock store is a host driver that honours ttlSeconds, so the lease can expire
// while the call runs. The clock is faked (Date only). Offline.

const TTL = 5;
const INTERRUPTED = 'Tool call was interrupted, it may have run';

class Durable extends MemorySessionStore {
  override durability(): 'durable' { return 'durable'; }
}

/** A host lock store whose lock EXPIRES after ttlSeconds by the clock, as the interface allows. */
class ExpiringLocks extends MemorySessionStore {
  #held = new Map<string, { until: number; token: number }>();
  #next = 0;
  override async withLock<T>(key: string, callback: () => T | Promise<T>, ttlSeconds = 10): Promise<T> {
    const current = this.#held.get(key);
    if (current !== undefined && current.until > Date.now()) {
      throw Object.assign(new Error('The session is locked (test driver).'), { code: 'session_locked' });
    }
    const token = ++this.#next;
    this.#held.set(key, { until: Date.now() + ttlSeconds * 1000, token });
    try {
      return await callback();
    } finally {
      if (this.#held.get(key)?.token === token) this.#held.delete(key);
    }
  }
}

interface Worker { agent: AgentRuntime; session: Session; requests: LlmRequest[] }

function workers(gated: boolean) {
  const ephemeral = new ExpiringLocks();
  const durable = new Durable();
  const runs: string[] = [];
  let during: (() => Promise<void>) | null = null;
  const make = (name: string): Worker => {
    const session = new PrismHarness({ drivers: { e: () => ephemeral, d: () => durable }, stores: { ephemeral: 'e', durable: 'd' } })
      .for({ type: 'User', id: 1 }).session('one-outcome');
    const requests: LlmRequest[] = [];
    let model = 0;
    const agent = new AgentRuntime({
      config: { lock_ttl: TTL, lock_wait: 0 },
      modes: new ModeRegistry({ default: 'm', modes: { m: { system_prompt: 'S.', tools: ['slow'], max_steps: 4, requires_approval: gated ? ['slow'] : [] } } }),
      tools: new ToolRegistry().register({ name: 'slow', handle: async (args: JsonObject) => {
        runs.push(`${name}:${String(args.n)}`);
        const hook = during;
        during = null;
        if (hook !== null) {
          // The call outlives the lease, and the other worker acts meanwhile.
          vi.setSystemTime(Date.now() + (TTL + 5) * 1000);
          await hook();
        }
        return `slow:${String(args.n)}`;
      } }),
      client: async (request: LlmRequest): Promise<LlmResponse> => {
        requests.push(JSON.parse(JSON.stringify(request)) as LlmRequest);
        return ++model === 1
          ? { text: '', finishReason: 'tool_calls', toolCalls: [{ id: `${name}1`, name: 'slow', arguments: { n: name === 'A' ? 1 : 2 } }] }
          : { text: `${name} done`, finishReason: 'stop' };
      },
    });
    return { agent, session, requests };
  };
  return { a: make('A'), b: make('B'), runs, during: (hook: () => Promise<void>) => { during = hook; } };
}

/** Every stored result for `id`, in order. */
async function stored(session: Session, id: string): Promise<string[]> {
  return (await session.thread().messages())
    .flatMap(({ message }) => (message.type === 'tool_result' ? message.tool_results as JsonObject[] : []))
    .filter((entry) => entry.tool_call_id === id)
    .map((entry) => String(entry.result));
}

/** The tool calls in a request whose step is not followed directly by a result row answering them. */
function unanswered(request: LlmRequest): string[] {
  const rows = request.messages as JsonObject[];
  return rows.flatMap((row, at) => {
    if (row.type !== 'assistant' || !Array.isArray(row.tool_calls) || row.tool_calls.length === 0) return [];
    const next = rows[at + 1];
    const answered = new Set(next?.type === 'tool_result' ? (next.tool_results as JsonObject[]).map((entry) => entry.tool_call_id) : []);
    return (row.tool_calls as JsonObject[]).map((call) => String(call.id)).filter((id) => !answered.has(id));
  });
}

/** What a request answers `id` with, in the row directly after its step. */
function answerIn(request: LlmRequest, id: string): string | undefined {
  const rows = request.messages as JsonObject[];
  const step = rows.findIndex((row) => row.type === 'assistant' && Array.isArray(row.tool_calls) && (row.tool_calls as JsonObject[]).some((call) => call.id === id));
  const next = rows[step + 1];
  return next?.type === 'tool_result' ? String((next.tool_results as JsonObject[]).find((entry) => entry.tool_call_id === id)?.result) : undefined;
}

afterEach(() => { vi.useRealTimers(); });

describe('G-414: a call still running elsewhere keeps one stored outcome', () => {
  it('an approved call: B resumes after the lease expired; A\'s result is the one stored outcome, it ran once, every request answers every call', async () => {
    vi.useFakeTimers({ toFake: ['Date'] });
    const w = workers(true);
    const parked = await w.a.agent.send(w.a.session, 'go');
    await recordApproval(w.a.session, parked.pendingApprovals[0]!.id, true);
    let bRequest: LlmRequest | undefined;
    w.during(async () => {
      await w.b.agent.send(w.b.session, '');
      bRequest = w.b.requests.at(-1);
    });
    await w.a.agent.send(w.a.session, '');

    expect(w.runs).toEqual(['A:1']);
    expect(await stored(w.a.session, 'A1')).toEqual(['slow:1']);
    // While A's call ran, B was shown it as interrupted, without storing that.
    expect(bRequest).toBeDefined();
    expect(answerIn(bRequest!, 'A1')).toBe(INTERRUPTED);
    expect([...w.a.requests, ...w.b.requests].flatMap(unanswered)).toEqual([]);
  });

  it('an ungated call: B sends a new message while it runs; A\'s result is the one stored outcome, and every request answers every call', async () => {
    vi.useFakeTimers({ toFake: ['Date'] });
    const w = workers(false);
    let bRequest: LlmRequest | undefined;
    w.during(async () => {
      await w.b.agent.send(w.b.session, 'from B');
      bRequest = w.b.requests[0];
    });
    await w.a.agent.send(w.a.session, 'go');

    expect(w.runs).toEqual(['A:1', 'B:2']);
    expect(await stored(w.a.session, 'A1')).toEqual(['slow:1']);
    expect(answerIn(bRequest!, 'A1')).toBe(INTERRUPTED);
    expect([...w.a.requests, ...w.b.requests].flatMap(unanswered)).toEqual([]);
  });
});
