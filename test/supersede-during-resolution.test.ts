import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  Thread,
  ToolRegistry,
  recordApproval,
  type JsonObject,
  type LlmResponse,
  type Session,
} from '../src/index.js';

// G-404: a write that could supersede a step while its approved call is
// running, or between a new message's writes, never gives a call a second
// outcome, a false "replaced" denial, or a second run. Deterministic: the
// writes happen from inside the running tool, or from a hook on the thread's
// own record(), never from timing. Offline; every store is in memory.

const REPLACED = 'Tool call was not run because a later step replaced it';

const network = vi.fn(() => { throw new Error('A test tried to use the network.'); });
beforeEach(() => { vi.stubGlobal('fetch', network); });
afterEach(() => { expect(network).not.toHaveBeenCalled(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });

class DurableMemory extends MemorySessionStore {
  override durability(): 'durable' {
    return 'durable';
  }
}

interface Rig {
  session: Session;
  agent: AgentRuntime;
  ran: number[];
  /** Runs inside the echo tool, while the approved call is executing. */
  during: (() => Promise<void>) | null;
}

function rig(): Rig {
  const harness = new PrismHarness({
    drivers: { memory: () => new MemorySessionStore(), durable: () => new DurableMemory() },
    stores: { ephemeral: 'memory', durable: 'durable' },
  });
  const r: Rig = { session: harness.for({ type: 'User', id: 1 }).session('race'), agent: undefined as unknown as AgentRuntime, ran: [], during: null };
  let model = 0;
  r.agent = new AgentRuntime({
    modes: new ModeRegistry({ default: 'guarded', modes: { guarded: { system_prompt: 'Careful.', tools: ['echo'], max_steps: 6, requires_approval: ['echo'] } } }),
    tools: new ToolRegistry().register({
      name: 'echo',
      handle: async (args: JsonObject) => {
        r.ran.push(Number(args.n));
        const during = r.during;
        r.during = null;
        if (during !== null) await during();
        return `ran:${String(args.n)}`;
      },
    }),
    client: async (): Promise<LlmResponse> => {
      model += 1;
      return model === 1 ? { text: '', finishReason: 'tool_calls', toolCalls: [{ id: 'c1', name: 'echo', arguments: { n: 1 } }] } : { text: 'done', finishReason: 'stop' };
    },
  });
  return r;
}

/** Every distinct result recorded for the call, and how many replaced denials name it. */
/** Thread positions of the first row matching each predicate. */
async function positions(session: Session, ...matches: ((row: JsonObject) => boolean)[]): Promise<number[]> {
  const all = (await session.thread().messages()).map((entry) => entry.message);
  return matches.map((match) => all.findIndex(match));
}
const isResult = (row: JsonObject): boolean => row.type === 'tool_result' && Array.isArray(row.tool_results) && (row.tool_results as JsonObject[]).some((entry) => entry.result === 'ran:1');

async function outcomes(session: Session): Promise<{ results: string[]; replaced: number }> {
  const entries = (await session.thread().messages()).map((entry) => entry.message).filter((row) => row.type === 'tool_result')
    .flatMap((row) => row.tool_results as JsonObject[]).filter((entry) => (entry.args as JsonObject).n === 1);
  return { results: [...new Set(entries.map((entry) => String(entry.result)))], replaced: entries.filter((entry) => entry.result === REPLACED).length };
}

describe('G-404: a superseding write while the approved call runs', () => {
  it('the host records a user row through Thread.record from inside the running call: one run, one outcome, no replaced denial', async () => {
    const r = rig();
    const [pending] = (await r.agent.send(r.session, 'go')).pendingApprovals;
    await recordApproval(r.session, pending!.id, true);
    r.during = async () => { await r.session.thread().record([{ type: 'user', content: 'Meanwhile.' }]); };
    await r.agent.send(r.session, '');
    for (let i = 0; i < 2; i += 1) await r.agent.send(r.session, '');

    expect(r.ran).toEqual([1]);
    expect(await outcomes(r.session)).toEqual({ results: ['ran:1'], replaced: 0 });
    // The user row really landed while the call ran: before its result.
    const [user, result] = await positions(r.session, (row) => row.type === 'user' && row.content === 'Meanwhile.', isResult);
    expect(user).toBeGreaterThan(-1);
    expect(user).toBeLessThan(result!);
  });

  it('a new message on another worker while the call runs waits for the session lock: one run, one outcome, no replaced denial', async () => {
    const r = rig();
    const [pending] = (await r.agent.send(r.session, 'go')).pendingApprovals;
    await recordApproval(r.session, pending!.id, true);
    let other: Promise<unknown> | null = null;
    r.during = async () => {
      // Started, not awaited: it has to wait for the resolution under way.
      other = r.agent.send(r.session, 'A new message.');
      await Promise.resolve();
    };
    await r.agent.send(r.session, '');
    expect(other).not.toBeNull();
    await other;
    await r.agent.send(r.session, '');

    expect(r.ran).toEqual([1]);
    expect(await outcomes(r.session)).toEqual({ results: ['ran:1'], replaced: 0 });
    // The new message waited: its row comes after the call's result.
    const [result, user] = await positions(r.session, isResult, (row) => row.type === 'user' && row.content === 'A new message.');
    expect(user).toBeGreaterThan(result!);
  });
});

describe('G-404: a new message writes the answers before its own row', () => {
  it('a decision recorded between the new message\'s answer row and its user row runs nothing, and every call keeps one outcome', async () => {
    const r = rig();
    const [pending] = (await r.agent.send(r.session, 'go')).pendingApprovals;
    const record = Thread.prototype.record;
    let between: unknown = 'not reached';
    vi.spyOn(Thread.prototype, 'record').mockImplementation(async function (this: Thread, ...args: Parameters<Thread['record']>) {
      if (args[0].some((row) => row.type === 'user' && row.content === 'A new message.')) {
        between = await recordApproval(r.session, pending!.id, true).then(() => 'accepted', (error: unknown) => (error as { code?: string }).code);
      }
      return record.apply(this, args);
    });
    await r.agent.send(r.session, 'A new message.');
    vi.restoreAllMocks();
    await r.agent.send(r.session, '');
    await r.agent.send(r.session, '');

    // The step was answered first (no decision: refused), so the late
    // decision is for an answered call: it is recorded and acts on nothing.
    expect(between).toBe('accepted');
    expect(r.ran).toEqual([]);
    expect(await outcomes(r.session)).toEqual({ results: ['No approval response provided'], replaced: 0 });
  });

  it('the new message\'s user row write fails after its answer row: the call keeps its one outcome and never runs', async () => {
    const r = rig();
    const [pending] = (await r.agent.send(r.session, 'go')).pendingApprovals;
    const record = Thread.prototype.record;
    vi.spyOn(Thread.prototype, 'record').mockImplementation(async function (this: Thread, ...args: Parameters<Thread['record']>) {
      if (args[0].some((row) => row.type === 'user' && row.content === 'A new message.')) throw new Error('The store refused the write (scripted).');
      return record.apply(this, args);
    });
    await expect(r.agent.send(r.session, 'A new message.')).rejects.toThrow('scripted');
    vi.restoreAllMocks();
    await recordApproval(r.session, pending!.id, true);
    await r.agent.send(r.session, '');

    expect(r.ran).toEqual([]);
    expect(await outcomes(r.session)).toEqual({ results: ['No approval response provided'], replaced: 0 });
  });
});
