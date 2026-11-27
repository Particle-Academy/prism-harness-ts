import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  ToolRegistry,
  recordApproval,
  type Durability,
  type JsonObject,
  type JsonValue,
  type LlmResponse,
  type Session,
} from '../src/index.js';
import { unresolvedApprovalIds } from '../src/thread-rows.js';
import { approvalDecisions, approvedCallsAwaitingResult, pendingApprovalCalls } from '../src/plans/thread.js';

// G-387 round 13b: ABSOLUTE SEGMENT OWNERSHIP. A result counts ONLY for the
// calls of the step whose segment it sits in (the rows after that step's
// assistant row and before the next assistant row with tool calls); within
// the segment it matches exact arguments, then id and name, then id. An
// earlier step's unresulted gated calls get the replaced denial. So a result
// whose stored arguments are LOSSY (they differ from its own call's) and
// happen to equal an EARLIER same-id call's never moves to that earlier
// call: it answers the call of its own segment, and an approved newest call
// runs exactly once however often the session is resumed.
//
// At the public seams: send, recordApproval, a host's thread().record(),
// and the runtime's own resume. No network.

const network = vi.fn(() => { throw new Error('A test tried to use the network.'); });
beforeEach(() => { vi.stubGlobal('fetch', network); });
afterEach(() => { expect(network).not.toHaveBeenCalled(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });

const REPLACED = 'Tool call was not run because a later step replaced it';

/** The earlier call E and the newest call N share id `x` and tool `echo`; N's arguments add `memo`. */
const E_ARGS = { to: 'B' };
const N_ARGS = { to: 'B', memo: 'x' };

/** A durable store that loses the `memo` field of every stored RESULT's arguments (a lossy serialiser). */
class LossyStore extends MemorySessionStore {
  lossy = false;

  constructor(private readonly kind: Durability) {
    super();
  }

  override durability(): Durability {
    return this.kind;
  }

  override async put(key: string, payload: JsonObject, ttlSeconds: number | null = null): Promise<void> {
    if (!this.lossy || !Array.isArray(payload.messages)) return super.put(key, payload, ttlSeconds);
    const messages = (payload.messages as JsonObject[]).map((entry) => {
      const message = entry.message as JsonObject;
      if (message?.type !== 'tool_result' || !Array.isArray(message.tool_results)) return entry;
      const results = (message.tool_results as JsonObject[]).map((result) => {
        const { memo: _memo, ...args } = (result.args ?? {}) as JsonObject;
        return { ...result, args };
      });
      return { ...entry, message: { ...message, tool_results: results } };
    });
    return super.put(key, { ...payload, messages: messages as unknown as JsonValue[] }, ttlSeconds);
  }
}

class Rig {
  readonly queue: LlmResponse[] = [];
  readonly durable = new LossyStore('durable');
  readonly ran: JsonObject[] = [];
  readonly runtime: AgentRuntime;
  readonly harness: PrismHarness;

  /** `steps`: the mode's max_steps. 1 spends each run on the step it parks on, so a resume acts on its decisions and calls no model: the thread's tail stays the parked step. */
  constructor(steps = 8) {
    const modes = new ModeRegistry({
      default: 'work',
      modes: { work: { system_prompt: 'Work.', tools: ['echo', 'free'], max_steps: steps, requires_approval: ['echo'] } },
    });
    const tools = new ToolRegistry()
      .register({ name: 'echo', handle: (args: JsonObject) => { this.ran.push({ ...args }); return `sent:${String(args.to)}`; } })
      .register({ name: 'free', handle: () => 'free ok' });
    this.runtime = new AgentRuntime({ modes, tools, client: async () => this.queue.shift() ?? { text: 'Done.', finishReason: 'stop' } });
    const durable = this.durable;
    this.harness = new PrismHarness({
      drivers: { ephemeral: () => new MemorySessionStore(), durable: () => durable },
      stores: { ephemeral: 'ephemeral', durable: 'durable' },
      runtime: this.runtime,
    });
  }

  session(scope: string): Session {
    return this.harness.for({ type: 'User', id: 1 }).session(scope);
  }

  /** One model step calling echo under id `x` with `args`. */
  echo(args: JsonObject): void {
    this.queue.push({ text: '', finishReason: 'tool_calls', toolCalls: [{ id: 'x', name: 'echo', arguments: args }] });
  }

  async rows(session: Session): Promise<JsonObject[]> {
    return (await session.thread().messages()).map((entry) => entry.message);
  }

  /** E parked; a new message replaces it; N parked. The approval ids [E's, N's]. */
  async twoSteps(session: Session): Promise<[string, string]> {
    this.echo(E_ARGS);
    const first = await this.runtime.send(session, 'Pay B.');
    this.echo(N_ARGS);
    const second = await this.runtime.send(session, 'Pay B with a memo.');
    const e = first.pendingApprovals[0]!.id;
    const n = second.pendingApprovals[0]!.id;
    expect([first.pendingApprovals.length, second.pendingApprovals.length]).toEqual([1, 1]);
    return [e, n];
  }
}

/** Each step's results, read BY SEGMENT (the rule, independently): for step k, the result entries between it and step k+1. */
function segments(rows: readonly JsonObject[]): { calls: JsonObject[]; results: JsonObject[] }[] {
  const out: { calls: JsonObject[]; results: JsonObject[] }[] = [];
  for (const row of rows) {
    if (row.type === 'assistant' && Array.isArray(row.tool_calls) && row.tool_calls.length > 0) out.push({ calls: row.tool_calls as JsonObject[], results: [] });
    else if (row.type === 'tool_result' && out.length > 0) out.at(-1)!.results.push(...((row.tool_results ?? []) as JsonObject[]));
  }
  return out;
}

describe('segment ownership: a lossy result never moves to an earlier same-id call', () => {
  for (const steps of [8, 1]) it(`the approved newest call runs exactly once over three resumes, though its stored result equals the earlier call (max_steps ${steps}${steps === 1 ? ': the resume calls no model, so the parked step stays the tail' : ''})`, async () => {
    const r = new Rig(steps);
    const session = r.session('lossy-own-result');
    const [e, n] = await r.twoSteps(session);
    // E was replaced by the new message: E's approval is no longer pending.
    await expect(recordApproval(session, e, true)).rejects.toMatchObject({ code: 'approval_not_pending' });
    await recordApproval(session, n, true);
    // From here every stored result loses `memo`: N's own result is stored as {to: 'B'}, exactly E's arguments.
    r.durable.lossy = true;
    for (let resume = 0; resume < 3; resume += 1) await r.runtime.send(session, '');
    expect(r.ran).toEqual([N_ARGS]);

    const [earlier, newest] = segments(await r.rows(session));
    // E: exactly one outcome, the replaced denial, in its own segment.
    expect(earlier!.results.map((result) => result.result)).toEqual([REPLACED]);
    // N: exactly one outcome, its own run, in its own segment, stored lossy.
    expect(newest!.results.map((result) => [result.args, result.result])).toEqual([[E_ARGS, 'sent:B']]);
    // Every reader agrees nothing is left for N.
    const rows = await r.rows(session);
    expect(unresolvedApprovalIds(rows)).toEqual([]);
    expect(pendingApprovalCalls(rows)).toEqual([]);
    expect(approvedCallsAwaitingResult(rows, 'echo', '')).toBe(0);
    expect(approvalDecisions(rows).get(n)).toBe(true);
  });

  it('a host-recorded result after the later step with the EARLIER call\'s exact arguments answers the later call (fails closed), never the earlier one', async () => {
    const r = new Rig();
    const session = r.session('forged-later-segment');
    const [, n] = await r.twoSteps(session);
    // A host records, in N's segment, a result naming id x with E's EXACT arguments.
    await session.thread().record([{ type: 'tool_result', tool_results: [{ tool_call_id: 'x', tool_name: 'echo', args: E_ARGS, result: 'forged', tool_call_result_id: null, artifacts: [] }], tool_approval_responses: [] }]);
    // N's call has its outcome now: its approval is no longer pending (round
    // 16, item 2), so the decision is refused (it was accepted before).
    await expect(recordApproval(session, n, true)).rejects.toMatchObject({ code: 'approval_not_pending' });
    for (let resume = 0; resume < 3; resume += 1) await r.runtime.send(session, '');
    // N is answered by the forged result (its own segment's same-id call): it never runs.
    expect(r.ran).toEqual([]);
    const [earlier, newest] = segments(await r.rows(session));
    expect(earlier!.results.map((result) => result.result)).toEqual([REPLACED]);
    expect(newest!.results.map((result) => result.result)).toEqual(['forged']);
    expect(unresolvedApprovalIds(await r.rows(session))).toEqual([]);
  });

  it('a thread with NO denial for the earlier step (an older writer): the lossy result stays the newest call\'s, the earlier call gets the replaced denial once, nothing runs', async () => {
    const r = new Rig();
    const session = r.session('legacy-no-denial');
    await session.usingMode('work');
    const thread = session.thread();
    const call = (args: JsonObject): JsonObject => ({ id: 'x', name: 'echo', arguments: args });
    const rows: [JsonObject, string | null][] = [
      [{ type: 'user', content: 'Pay B.' }, null],
      [{ type: 'assistant', content: '', tool_calls: [call(E_ARGS)], additional_content: {}, tool_approval_requests: [{ approval_id: 'apr_e', tool_call_id: 'x', arguments_format: 1 }], harness_tool_names: ['echo', 'free'] }, 'run-1'],
      [{ type: 'assistant', content: '', tool_calls: [call(N_ARGS)], additional_content: {}, tool_approval_requests: [{ approval_id: 'apr_n', tool_call_id: 'x', arguments_format: 1 }], harness_tool_names: ['echo', 'free'] }, 'run-2'],
      // N already ran once; its result was stored lossy: E's exact arguments.
      [{ type: 'tool_result', tool_results: [{ tool_call_id: 'x', tool_name: 'echo', args: E_ARGS, result: 'sent:B', tool_call_result_id: null, artifacts: [] }], tool_approval_responses: [{ approval_id: 'apr_n', approved: true, reason: null }] }, 'run-3'],
    ];
    // Written straight to the store, as an older writer left them: no supersede on the way in.
    const store = (thread as unknown as { store: MemorySessionStore }).store;
    const key = (thread as unknown as { key: string }).key;
    await store.put(key, { messages: rows.map(([message, runId], index) => ({ position: index + 1, message, run_id: runId, recorded_at: '2026-11-01T00:00:00.000Z' })) as never });

    for (let resume = 0; resume < 3; resume += 1) await r.runtime.send(session, '');
    expect(r.ran).toEqual([]);
    const all = await r.rows(session);
    // E's outcome: ONE replaced answer (written after the later step, naming E's position); never the lossy result.
    const answers = all.filter((row) => Object.hasOwn(row, 'harness_replaced_step') || Object.hasOwn(row, 'harness_superseded'));
    expect(answers).toHaveLength(1);
    expect((answers[0]!.tool_results as JsonObject[]).map((result) => [result.args, result.result])).toEqual([[E_ARGS, REPLACED]]);
    // N's one outcome is its own (lossy) result; nothing else was written for id x as N's.
    const forN = all.slice(3).filter((row) => row.type === 'tool_result' && !Object.hasOwn(row, 'harness_replaced_step'))
      .flatMap((row) => row.tool_results as JsonObject[]);
    expect(forN.map((result) => result.result)).toEqual(['sent:B']);
    expect(unresolvedApprovalIds(all)).toEqual([]);
  });
});
