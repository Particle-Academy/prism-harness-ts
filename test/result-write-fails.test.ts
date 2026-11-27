import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  ToolRegistry,
  recordApproval,
  type AgentResponse,
  type JsonObject,
  type JsonValue,
  type LlmRequest,
  type LlmResponse,
  type RunStep,
  type Session,
} from '../src/index.js';
import { HARNESS_RECORDING } from '../src/thread.js';
import { threadView } from '../src/thread-rows.js';

// G-410: a call whose tool ran but whose result could not be written is never
// run again automatically. Before the resolver runs an approved call it
// records a started row for it; a later resume that finds a started call with
// no result answers it as interrupted (it may have run) and never runs it
// again. Offline; the durable store refuses one write, chosen by what it holds.

const INTERRUPTED = 'Tool call was interrupted, it may have run';

const network = vi.fn(() => { throw new Error('A test tried to use the network.'); });
beforeEach(() => { vi.stubGlobal('fetch', network); });
afterEach(() => { expect(network).not.toHaveBeenCalled(); vi.unstubAllGlobals(); });

/** A durable store that refuses ONE write: the first whose newest row is a result row holding `marker`. */
class FailingOnce extends MemorySessionStore {
  armed = true;
  constructor(private readonly marker: string) {
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

interface Rig {
  session: Session;
  agent: AgentRuntime;
  ran: number[];
  free: number[];
  /** Every request the model was sent. */
  requests: LlmRequest[];
}

function rig(marker: string, calls: { id: string; name: string; n: number }[]): Rig {
  const store = new FailingOnce(marker);
  const harness = new PrismHarness({
    drivers: { memory: () => new MemorySessionStore(), durable: () => store },
    stores: { ephemeral: 'memory', durable: 'durable' },
  });
  const ran: number[] = [];
  const free: number[] = [];
  const requests: LlmRequest[] = [];
  let model = 0;
  const agent = new AgentRuntime({
    modes: new ModeRegistry({ default: 'guarded', modes: { guarded: { system_prompt: 'Careful.', tools: ['echo', 'free'], max_steps: 6, requires_approval: ['echo'] } } }),
    tools: new ToolRegistry()
      .register({ name: 'echo', handle: (args: JsonObject) => { ran.push(Number(args.n)); return `ran:${String(args.n)}`; } })
      .register({ name: 'free', handle: (args: JsonObject) => { free.push(Number(args.n)); return `free:${String(args.n)}`; } }),
    client: async (request: LlmRequest): Promise<LlmResponse> => {
      requests.push(JSON.parse(JSON.stringify(request)) as LlmRequest);
      return ++model === 1
        ? { text: '', finishReason: 'tool_calls', toolCalls: calls.map((call) => ({ id: call.id, name: call.name, arguments: { n: call.n } })) }
        : { text: 'done', finishReason: 'stop' };
    },
  });
  return { session: harness.for({ type: 'User', id: 1 }).session('write-fails'), agent, ran, free, requests };
}

async function drain(stream: AsyncGenerator<RunStep, AgentResponse, undefined>): Promise<void> {
  for (;;) if ((await stream.next()).done === true) return;
}

/** What the newest request answers each call with, in the result row directly after its step. */
function sentAnswers(requests: readonly LlmRequest[]): Record<string, string> {
  const rows = (requests.at(-1)?.messages ?? []) as JsonObject[];
  const out: Record<string, string> = {};
  rows.forEach((row, at) => {
    if (row.type !== 'assistant' || !Array.isArray(row.tool_calls)) return;
    const next = rows[at + 1];
    for (const call of row.tool_calls as JsonObject[]) {
      const entry = next?.type === 'tool_result' ? (next.tool_results as JsonObject[]).find((result) => result.tool_call_id === call.id) : undefined;
      out[String(call.id)] = entry === undefined ? '(unanswered)' : String(entry.result);
    }
  });
  return out;
}

async function outcomes(session: Session, n: number): Promise<string[]> {
  return [...new Set((await session.thread().messages()).map((entry) => entry.message).filter((row) => row.type === 'tool_result')
    .flatMap((row) => row.tool_results as JsonObject[]).filter((entry) => (entry.args as JsonObject).n === n).map((entry) => String(entry.result)))];
}

describe('a call whose result could not be written', () => {
  for (const path of ['send', 'streamDecision', 'streamSteps'] as const) {
    it(`the approved call ran and its resolution row was refused (${path}): later resumes never run it again; nothing is stored for it, and the replay answers it as interrupted`, async () => {
      const r = rig('ran:1', [{ id: 'c1', name: 'echo', n: 1 }]);
      const [pending] = (await r.agent.send(r.session, 'go')).pendingApprovals;
      if (path === 'streamDecision') {
        await expect(drain(r.agent.streamDecision(r.session, [{ approvalId: pending!.id, approved: true }]))).rejects.toThrow('scripted');
      } else {
        await recordApproval(r.session, pending!.id, true);
        await expect(path === 'send' ? r.agent.send(r.session, '') : drain(r.agent.streamSteps(r.session, ''))).rejects.toThrow('scripted');
      }
      expect(r.ran).toEqual([1]);
      for (let i = 0; i < 3; i += 1) await r.agent.send(r.session, '');

      expect(r.ran).toEqual([1]);
      // Replay-only (G-414): a result that lands later would be its one outcome.
      expect(await outcomes(r.session, 1)).toEqual([]);
      expect(sentAnswers(r.requests)).toEqual({ c1: INTERRUPTED });
    });
  }

  it('an ungated call ran when its step was taken and its result row was refused: later resumes never run it again', async () => {
    // Ungated calls run when their step is taken, never by the resolver, so
    // they get no started row (the shared shape is per approved call).
    const r = rig('free:2', [{ id: 'f1', name: 'free', n: 2 }, { id: 'c1', name: 'echo', n: 1 }]);
    await expect(r.agent.send(r.session, 'go')).rejects.toThrow('scripted');
    expect(r.free).toEqual([2]);
    for (let i = 0; i < 3; i += 1) await r.agent.send(r.session, '').catch(() => undefined);

    expect(r.free).toEqual([2]);
  });
});

// The shared started-row bytes (harness-thread-rows thr-0022, the PHP
// reference's Thread::startedPayload), one stored row.
const THR_0022 = '[{"type":"tool_result","tool_results":[],"tool_approval_responses":[],"harness_started":{"tool_call_id":"call_paris","tool_name":"weather","approval_id":"apr_1"},"harness_answers_step":{"position":2}}]';

describe('G-410: the started row', () => {
  function weatherRig(during: (session: Session) => Promise<void> = async () => {}): { session: Session; agent: AgentRuntime; ran: string[]; requests: LlmRequest[] } {
    const harness = new PrismHarness({
      drivers: { memory: () => new MemorySessionStore(), durable: () => new FailingOnce('never') },
      stores: { ephemeral: 'memory', durable: 'durable' },
    });
    const session = harness.for({ type: 'User', id: 1 }).session('started');
    const ran: string[] = [];
    const requests: LlmRequest[] = [];
    const agent = new AgentRuntime({
      modes: new ModeRegistry({ default: 'guarded', modes: { guarded: { system_prompt: 'Careful.', tools: ['weather'], max_steps: 6, requires_approval: ['weather'] } } }),
      tools: new ToolRegistry().register({ name: 'weather', handle: async (args: JsonObject) => { ran.push(String(args.city)); await during(session); return `Sunny in ${String(args.city)}`; } }),
      client: async (request: LlmRequest): Promise<LlmResponse> => {
        requests.push(JSON.parse(JSON.stringify(request)) as LlmRequest);
        return { text: 'done', finishReason: 'stop' };
      },
    });
    return { session, agent, ran, requests };
  }
  const parked = (calls: [string, string, string][]): JsonObject[] => [
    { type: 'user', content: 'Weather?' },
    {
      type: 'assistant', content: '', additional_content: {}, harness_tool_names: ['weather'],
      tool_calls: calls.map(([id, city]) => ({ id, name: 'weather', arguments: { city }, result_id: null, reasoning_id: null, reasoning_summary: null })),
      tool_approval_requests: calls.map(([id, , approval]) => ({ approval_id: approval, tool_call_id: id, arguments_format: 1 })),
    },
  ];

  it('is written before the approved call runs, in the shared bytes (thr-0022)', async () => {
    let started: unknown = null;
    const r = weatherRig(async (session) => {
      started = (await session.thread().messages()).filter((entry) => entry.harness === true && Object.hasOwn(entry.message, 'harness_started')).map((entry) => entry.message);
    });
    await r.session.thread().record(parked([['call_paris', 'Paris', 'apr_1']]));
    await recordApproval(r.session, 'apr_1', true);
    await r.agent.send(r.session, '');

    expect(r.ran).toEqual(['Paris']);
    expect(JSON.stringify(started)).toBe(THR_0022);
    // Kept out of the replay: one result for the call, never the started row.
    const replay = threadView((await r.session.thread().messages()).map((entry) => entry.message));
    expect(JSON.stringify(replay)).not.toContain('harness_started');
  });

  it('a started row with no result (the worker died) reads interrupted in the replay, nothing is stored for it, and the call is never run again', async () => {
    const r = weatherRig();
    await r.session.thread().record(parked([['call_paris', 'Paris', 'apr_1']]));
    await recordApproval(r.session, 'apr_1', true);
    await r.session.thread().record([{ type: 'tool_result', tool_results: [], tool_approval_responses: [], harness_started: { tool_call_id: 'call_paris', tool_name: 'weather', approval_id: 'apr_1' }, harness_answers_step: { position: 2 } }], null, HARNESS_RECORDING);
    for (let i = 0; i < 3; i += 1) await r.agent.send(r.session, '');

    expect(r.ran).toEqual([]);
    // Replay-only (G-414): nothing is stored, and every request answers it as interrupted.
    expect(await outcomes2(r.session, 'Paris')).toEqual([]);
    expect(sentAnswers(r.requests)).toEqual({ call_paris: INTERRUPTED });
  });

  it('a started row a host records through Thread.record is ignored: the approved call runs once', async () => {
    const r = weatherRig();
    await r.session.thread().record(parked([['call_paris', 'Paris', 'apr_1']]));
    await recordApproval(r.session, 'apr_1', true);
    await r.session.thread().record([{ type: 'tool_result', tool_results: [], tool_approval_responses: [], harness_started: { tool_call_id: 'call_paris', tool_name: 'weather', approval_id: 'apr_1' }, harness_answers_step: { position: 2 } }]);
    for (let i = 0; i < 3; i += 1) await r.agent.send(r.session, '');

    expect(r.ran).toEqual(['Paris']);
    expect(await outcomes2(r.session, 'Paris')).toEqual(['Sunny in Paris']);
  });

  it('a harness started row naming a call the step does not have is ignored (segment ownership): the approved call runs once', async () => {
    const r = weatherRig();
    await r.session.thread().record(parked([['call_paris', 'Paris', 'apr_1']]));
    await recordApproval(r.session, 'apr_1', true);
    await r.session.thread().record([{ type: 'tool_result', tool_results: [], tool_approval_responses: [], harness_started: { tool_call_id: 'call_rome', tool_name: 'weather', approval_id: 'apr_9' }, harness_answers_step: { position: 2 } }], null, HARNESS_RECORDING);
    for (let i = 0; i < 2; i += 1) await r.agent.send(r.session, '');

    expect(r.ran).toEqual(['Paris']);
  });

  it('a denial recorded while an earlier call of the same step runs: the later call is not started and does not run', async () => {
    let denied = false;
    const r = weatherRig(async (session) => {
      if (denied) return;
      denied = true;
      await recordApproval(session, 'apr_2', false, 'changed my mind');
    });
    await r.session.thread().record(parked([['call_paris', 'Paris', 'apr_1'], ['call_rome', 'Rome', 'apr_2']]));
    await recordApproval(r.session, 'apr_1', true);
    await recordApproval(r.session, 'apr_2', true);
    for (let i = 0; i < 2; i += 1) await r.agent.send(r.session, '');

    expect(r.ran).toEqual(['Paris']);
    expect(await outcomes2(r.session, 'Rome')).toEqual(['Not run: this call is not the one that was approved, or it is no longer open.']);
  });
});

async function outcomes2(session: Session, city: string): Promise<string[]> {
  return [...new Set((await session.thread().messages()).map((entry) => entry.message).filter((row) => row.type === 'tool_result')
    .flatMap((row) => row.tool_results as JsonObject[]).filter((entry) => (entry.args as JsonObject).city === city).map((entry) => String(entry.result)))];
}
