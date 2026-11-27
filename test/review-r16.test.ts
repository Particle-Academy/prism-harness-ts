import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  HarnessError,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  Session,
  ToolRegistry,
  recordApproval,
  recordDecisions,
  type Durability,
  type JsonObject,
  type LlmRequest,
  type LlmResponse,
  type RuntimeConfig,
} from '../src/index.js';
import {
  endedByStart,
  pendingApprovalIds,
  placedRows,
  replayView,
  supersededApprovals,
  turnDecisions,
  unresolvedApprovalIds,
} from '../src/thread-rows.js';
import { approvalDecisions, pendingApprovalCalls } from '../src/plans/thread.js';

// G-387 round 16 (BRIEF-R16), at the public seams: send, recordApproval /
// recordDecisions, a host's thread().record(), raw stored rows, and a tool
// that acts on its own session while it runs. Every interleaving is forced by
// a HOOK (a tool that acts while it runs, a store write that fails), never by
// timing. No network; every store is in memory.

const network = vi.fn(() => { throw new Error('A test tried to use the network.'); });
beforeEach(() => { vi.stubGlobal('fetch', network); });
afterEach(() => { expect(network).not.toHaveBeenCalled(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });

const REPLACED = 'Tool call was not run because a later step replaced it';
const INTERRUPTED = 'Tool call was interrupted, it may have run';
const NO_RESPONSE = 'No approval response provided';
const DENIED = 'User denied tool execution';

/** A memory store whose writes of a thread payload can be made to fail, by the last message they carry; it counts whole-thread reads. */
class Store extends MemorySessionStore {
  failWhen: ((message: JsonObject) => boolean) | null = null;
  threadReads = 0;

  constructor(private readonly kind: Durability) {
    super();
  }

  override durability(): Durability {
    return this.kind;
  }

  override async get(key: string): Promise<JsonObject | null> {
    if (key.endsWith(':thread')) this.threadReads += 1;
    return super.get(key);
  }

  override async put(key: string, payload: JsonObject, ttlSeconds: number | null = null): Promise<void> {
    const messages = Array.isArray(payload.messages) ? payload.messages as JsonObject[] : [];
    const last = messages.at(-1)?.message as JsonObject | undefined;
    const fail = this.failWhen;
    if (fail !== null && last !== undefined && fail(last)) {
      this.failWhen = null;
      throw new Error('the store write failed');
    }
    return super.put(key, payload, ttlSeconds);
  }

  /** Append raw stored entries to a thread, as an import or a sync would (no harness reader, no binding). */
  async raw(key: string, messages: JsonObject[], runId: string | null = null): Promise<void> {
    const stored = (await super.get(key)) ?? {};
    const existing = Array.isArray(stored.messages) ? stored.messages as JsonObject[] : [];
    await super.put(key, {
      ...stored,
      messages: [...existing, ...messages.map((message, index) => ({ position: existing.length + index + 1, message, run_id: runId, recorded_at: '' }))],
    });
  }
}

class Rig {
  readonly requests: LlmRequest[] = [];
  readonly queue: LlmResponse[] = [];
  readonly durable = new Store('durable');
  readonly ephemeral = new Store('volatile');
  readonly runtime: AgentRuntime;
  readonly harness: PrismHarness;
  readonly echoed: number[] = [];
  readonly freed: number[] = [];
  /** Runs inside the echo tool, once, before it returns. */
  duringEcho: (() => Promise<void>) | null = null;
  /** What the echo hook produced (a write's outcome, an error code). */
  during: string[] = [];
  /** The model fails ONCE when set (the resumed turn's model call). */
  failOnce = false;

  constructor(maxSteps = 8, config: RuntimeConfig = {}) {
    const modes = new ModeRegistry({
      default: 'work',
      modes: { work: { system_prompt: 'Work.', tools: ['echo', 'free'], max_steps: maxSteps, requires_approval: ['echo'] } },
    });
    const tools = new ToolRegistry()
      .register({ name: 'echo', handle: async (args: JsonObject) => {
        this.echoed.push(Number(args.n));
        const hook = this.duringEcho;
        this.duringEcho = null;
        if (hook !== null) await hook();
        return `ran:${String(args.n)}`;
      } })
      .register({ name: 'free', handle: (args: JsonObject) => { this.freed.push(Number(args.n)); return `free:${String(args.n)}`; } });
    this.runtime = new AgentRuntime({
      modes, tools, config,
      client: async (request) => {
        this.requests.push(request);
        if (this.failOnce) { this.failOnce = false; throw new Error('the model failed'); }
        return this.queue.shift() ?? { text: 'Done.', finishReason: 'stop' };
      },
    });
    const durable = this.durable, ephemeral = this.ephemeral;
    this.harness = new PrismHarness({ drivers: { ephemeral: () => ephemeral, durable: () => durable }, stores: { ephemeral: 'ephemeral', durable: 'durable' }, runtime: this.runtime });
  }

  session(scope = 'chat'): Session {
    return this.harness.for({ type: 'User', id: 1 }).session(scope) as unknown as Session;
  }

  threadKey(): string {
    return (this.session().thread() as unknown as { key: string }).key;
  }

  calls(calls: [string, JsonObject, string][]): void {
    this.queue.push({ text: '', finishReason: 'tool_calls', toolCalls: calls.map(([name, args, id]) => ({ id, name, arguments: args })) });
  }

  text(text: string): void {
    this.queue.push({ text, finishReason: 'stop' });
  }

  async rows(): Promise<JsonObject[]> {
    return await this.session().thread().readerRows();
  }

  /** Parked on the given calls (default: one gated echo, c1); their approval ids. */
  async parked(calls: [string, JsonObject, string][] = [['echo', { n: 1 }, 'c1']]): Promise<string[]> {
    this.calls(calls);
    const response = await this.runtime.send(this.session(), 'Go');
    expect(response.finishReason).toBe('awaiting_approval');
    return response.pendingApprovals.map((pending) => pending.id);
  }

  /** The run id the parked step was recorded under. */
  async parkedRunId(): Promise<string> {
    const entry = (await this.session().thread().messages()).filter((candidate) => candidate.message.type === 'assistant').at(-1)!;
    return entry.runId!;
  }
}

/** Every outcome recorded for the call named `tool` carrying token `n`, COUNTED (an exact copy carried again is not counted twice). */
function outcomes(rows: JsonObject[], n: number, tool = 'echo'): string[] {
  const earlier = new Set<string>();
  const out: string[] = [];
  for (const row of rows) {
    if (row.type !== 'tool_result') continue;
    const here: string[] = [];
    for (const entry of (row.tool_results ?? []) as JsonObject[]) {
      if (entry.tool_name !== tool || (entry.args as JsonObject).n !== n) continue;
      const copy = JSON.stringify([entry.tool_call_id, entry.args, entry.result]);
      if (earlier.has(copy)) continue;
      here.push(copy);
      out.push(String(entry.result));
    }
    for (const copy of here) earlier.add(copy);
  }
  return out;
}

/** The replayed decision entries for `approval`, in the replay order. */
function replayedDecisions(rows: JsonObject[], approval: string): boolean[] {
  return replayView(rows).filter((row) => row.type === 'tool_result')
    .flatMap((row) => (row.tool_approval_responses ?? []) as JsonObject[])
    .filter((entry) => entry.approval_id === approval).map((entry) => entry.approved === true);
}

/** Every provider request answers every tool call it replays, exactly once, before the next assistant or user row. */
function everyCallAnswered(request: LlmRequest): string[] {
  const problems: string[] = [];
  const messages = request.messages as JsonObject[];
  for (const [index, row] of messages.entries()) {
    if (row.type !== 'assistant' || !Array.isArray(row.tool_calls) || row.tool_calls.length === 0) continue;
    const answered: string[] = [];
    for (let at = index + 1; at < messages.length && messages[at]!.type !== 'assistant' && messages[at]!.type !== 'user'; at += 1) {
      for (const entry of (messages[at]!.tool_results ?? []) as JsonObject[]) answered.push(String(entry.tool_call_id));
    }
    for (const call of row.tool_calls as JsonObject[]) {
      const count = answered.filter((id) => id === call.id).length;
      if (count !== 1) problems.push(`${String(call.id)}:${count}`);
    }
  }
  return problems;
}

const code = (error: { code?: string }): string => error.code ?? 'threw';
const rawDenial = (approval: string): JsonObject => ({ type: 'tool_result', tool_results: [], tool_approval_responses: [{ approval_id: approval, approved: false, reason: null }] });
const rawApproval = (approval: string): JsonObject => ({ type: 'tool_result', tool_results: [], tool_approval_responses: [{ approval_id: approval, approved: true, reason: null }] });

// ---------------------------------------------------------------------------
// Item 1 (TS run19 F1): a raw denial counts only while its call has NO
// started row and NO result before it; the start guard re-checks that the
// approval still STANDS immediately before it writes the started row.
// ---------------------------------------------------------------------------

describe('item 1: a raw denial after the call started or ran', () => {
  it('D1: a raw denial of c2 recorded while c1 runs (c2 not started yet) stands: c2 never runs, its ONE outcome is that denial, every reader agrees', async () => {
    const r = new Rig();
    const [a1, a2] = await r.parked([['echo', { n: 1 }, 'c1'], ['echo', { n: 2 }, 'c2']]);
    await recordDecisions(r.session(), [{ approvalId: a1!, approved: true }, { approvalId: a2!, approved: true }]);
    r.duringEcho = async () => { r.during.push(await r.session().thread().record([rawDenial(a2!)]).then(() => 'recorded', code)); };
    await r.runtime.send(r.session(), '');
    for (let i = 0; i < 2; i += 1) await r.runtime.send(r.session(), '');
    expect(r.during).toEqual(['recorded']);
    const rows = await r.rows();
    expect(r.echoed).toEqual([1]);
    expect(outcomes(rows, 1)).toEqual(['ran:1']);
    expect(outcomes(rows, 2)).toEqual([DENIED]);
    expect(approvalDecisions(rows).get(a2!)).toBe(false);
    expect(approvalDecisions(rows).get(a1!)).toBe(true);
    for (const decided of replayedDecisions(rows, a2!)) expect(decided).toBe(false);
  });

  it('D1b: a raw denial of the RUNNING call (its started row is before it) is ignored: it ran, every reader and the replay say approved', async () => {
    const r = new Rig();
    const [a1] = await r.parked();
    await recordApproval(r.session(), a1!, true);
    r.duringEcho = async () => { await r.session().thread().record([rawDenial(a1!)]); };
    await r.runtime.send(r.session(), '');
    const rows = await r.rows();
    expect(r.echoed).toEqual([1]);
    expect(outcomes(rows, 1)).toEqual(['ran:1']);
    expect(approvalDecisions(rows).get(a1!)).toBe(true);
    expect(replayedDecisions(rows, a1!).length).toBeGreaterThan(0);
    for (const decided of replayedDecisions(rows, a1!)) expect(decided).toBe(true);
  });

  it('D2: a raw denial recorded AFTER the call ran changes nothing: approved in the thread, the plan and the replay sent to the provider', async () => {
    const r = new Rig();
    const [a1] = await r.parked();
    await recordApproval(r.session(), a1!, true);
    r.text('Resumed.');
    await r.runtime.send(r.session(), '');
    // A raw row through the host API, and the same row written straight into storage.
    await r.session().thread().record([rawDenial(a1!)]);
    await r.durable.raw(r.threadKey(), [rawDenial(a1!)]);
    const rows = await r.rows();
    expect(approvalDecisions(rows).get(a1!)).toBe(true);
    for (const decided of replayedDecisions(rows, a1!)) expect(decided).toBe(true);
    for (const row of placedRows(rows)) {
      for (const entry of (row.tool_approval_responses ?? []) as JsonObject[]) if (entry.approval_id === a1) expect(entry.approved).toBe(true);
    }
    expect(r.echoed).toEqual([1]);
  });

  it('D3: a raw denial recorded while the call is parked (approved, not started) still revokes it: the call never runs', async () => {
    const r = new Rig();
    const [a1] = await r.parked();
    await recordApproval(r.session(), a1!, true);
    await r.session().thread().record([rawDenial(a1!)]);
    await r.runtime.send(r.session(), '');
    const rows = await r.rows();
    expect(r.echoed).toEqual([]);
    expect(outcomes(rows, 1)).toEqual([DENIED]);
    expect(approvalDecisions(rows).get(a1!)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Item 2 (TS run19 F2): an approval whose call already has an OUTCOME is not
// pending, by the same pairing as the unresolved reader and the replay.
// ---------------------------------------------------------------------------

describe('item 2: a closed call\'s approval is not pending', () => {
  it('B0-1: max_steps 1, an empty resume with no decision closes the call; its approval is no longer pending and a decision on it is refused', async () => {
    const r = new Rig(1);
    const [a1] = await r.parked();
    await r.runtime.send(r.session(), '');
    const rows = await r.rows();
    expect(outcomes(rows, 1)).toEqual([NO_RESPONSE]);
    expect(pendingApprovalIds(rows)).toEqual([]);
    expect(pendingApprovalCalls(rows)).toEqual([]);
    expect(unresolvedApprovalIds(rows)).toEqual([]);
    expect(await recordApproval(r.session(), a1!, true).then(() => 'accepted', code)).toBe('approval_not_pending');
    expect(approvalDecisions(await r.rows()).has(a1!)).toBe(false);
    await r.runtime.send(r.session(), '');
    expect(r.echoed).toEqual([]);
  });

  it('B0-2: the resumed model call fails after the resolution: the closed approval is not pending', async () => {
    const r = new Rig();
    const [a1] = await r.parked();
    r.failOnce = true;
    await expect(r.runtime.send(r.session(), '')).rejects.toThrow();
    const rows = await r.rows();
    expect(outcomes(rows, 1)).toEqual([NO_RESPONSE]);
    expect(pendingApprovalIds(rows)).toEqual([]);
    expect(await recordApproval(r.session(), a1!, true).then(() => 'accepted', code)).toBe('approval_not_pending');
  });

  it('B0-3: a raw host result for the parked call closes it: not pending, and the decision is refused', async () => {
    const r = new Rig();
    const [a1] = await r.parked();
    await r.session().thread().record([{ type: 'tool_result', tool_results: [{ tool_call_id: 'c1', tool_name: 'echo', args: { n: 1 }, result: 'host:1', tool_call_result_id: null, artifacts: [] }], tool_approval_responses: [] }]);
    const rows = await r.rows();
    expect(pendingApprovalIds(rows)).toEqual([]);
    expect(unresolvedApprovalIds(rows)).toEqual([]);
    expect(await recordApproval(r.session(), a1!, true).then(() => 'accepted', code)).toBe('approval_not_pending');
  });
});

// ---------------------------------------------------------------------------
// Item 3: ungated calls get a write-ahead started row too; the replay gives
// ANY tool call with no stored result the interrupted result.
// ---------------------------------------------------------------------------

describe('item 3: ungated calls and the interrupted replay', () => {
  it('U1: an ungated-only step whose results write fails: the call ran once; a new message writes NO replaced denial for it; the replay answers it as interrupted', async () => {
    const r = new Rig();
    r.calls([['free', { n: 7 }, 'f1']]);
    r.durable.failWhen = (message) => message.type === 'tool_result' && JSON.stringify(message).includes('free:7');
    await expect(r.runtime.send(r.session(), 'Go')).rejects.toThrow();
    expect(r.freed).toEqual([7]);
    r.text('Next answer.');
    await r.runtime.send(r.session(), 'Next.');
    const rows = await r.rows();
    expect(outcomes(rows, 7, 'free')).not.toContain(REPLACED);
    expect(r.freed).toEqual([7]);
    const last = r.requests.at(-1)!;
    expect(everyCallAnswered(last)).toEqual([]);
    const replayed = (last.messages as JsonObject[]).filter((row) => row.type === 'tool_result')
      .flatMap((row) => (row.tool_results ?? []) as JsonObject[]).filter((entry) => entry.tool_call_id === 'f1');
    expect(replayed.map((entry) => entry.result)).toEqual([INTERRUPTED]);
  });

  it('U2: an ungated call beside a gated one, the ungated results write fails: the next provider request answers BOTH calls; the ungated call never runs again and never reads "not run"', async () => {
    const r = new Rig();
    r.calls([['free', { n: 7 }, 'f1'], ['echo', { n: 1 }, 'c1']]);
    r.durable.failWhen = (message) => message.type === 'tool_result' && JSON.stringify(message).includes('free:7');
    await expect(r.runtime.send(r.session(), 'Go')).rejects.toThrow();
    expect(r.freed).toEqual([7]);
    const pending = pendingApprovalIds(await r.rows());
    expect(pending).toHaveLength(1);
    await recordApproval(r.session(), pending[0]!, true);
    await r.runtime.send(r.session(), '');
    expect(r.echoed).toEqual([1]);
    expect(r.freed).toEqual([7]);
    for (const request of r.requests.slice(1)) expect(everyCallAnswered(request)).toEqual([]);
    await r.runtime.send(r.session(), 'Next.');
    const rows = await r.rows();
    expect(outcomes(rows, 7, 'free')).not.toContain(REPLACED);
    for (const request of r.requests.slice(1)) expect(everyCallAnswered(request)).toEqual([]);
  });

  it('U3: a raw stored step with a call that has no result and is not the paused step (a later user row): the replay gives it ONE interrupted result', async () => {
    const r = new Rig();
    await r.runtime.send(r.session(), 'Hello');
    await r.durable.raw(r.threadKey(), [
      { type: 'assistant', content: '', tool_calls: [{ id: 'u9', name: 'free', arguments: { n: 9 } }], additional_content: {}, tool_approval_requests: [] },
      { type: 'user', content: 'Later.' },
    ]);
    const view = replayView(await r.rows());
    const answers = view.filter((row) => row.type === 'tool_result').flatMap((row) => (row.tool_results ?? []) as JsonObject[]).filter((entry) => entry.tool_call_id === 'u9');
    expect(answers.map((entry) => entry.result)).toEqual([INTERRUPTED]);
    await r.runtime.send(r.session(), 'More.');
    expect(everyCallAnswered(r.requests.at(-1)!)).toEqual([]);
  });

  it('N1: a tool\'s own nested send parks a NEW gated step mid-run: every provider request stays valid, the inner step stays pending and is never replaced by the outer turn, and approving it runs it once', async () => {
    const r = new Rig();
    const [a1] = await r.parked();
    await recordApproval(r.session(), a1!, true);
    let inner: string[] = [];
    r.duringEcho = async () => {
      r.calls([['echo', { n: 9 }, 'c9']]);
      const response = await r.runtime.send(r.session(), 'Inner.');
      r.during.push(response.finishReason);
      inner = response.pendingApprovals.map((pending) => pending.id);
    };
    const requests = r.requests.length;
    const outer = await r.runtime.send(r.session(), '');
    expect(r.during).toEqual(['awaiting_approval']);
    // The outer resume stopped with no model step of its own and reports the inner step's approvals.
    expect(r.requests.length).toBe(requests + 1);
    expect(outer.finishReason).toBe('awaiting_approval');
    expect(outer.pendingApprovals.map((pending) => [pending.id, pending.toolCallId, pending.tool])).toEqual([[inner[0], 'c9', 'echo']]);
    for (const request of r.requests) expect(everyCallAnswered(request)).toEqual([]);
    let rows = await r.rows();
    expect(outcomes(rows, 1)).toEqual(['ran:1']);
    expect(outcomes(rows, 9)).toEqual([]);
    expect(pendingApprovalIds(rows)).toEqual(inner);
    expect(supersededApprovals(rows).has(inner[0]!)).toBe(false);
    await recordApproval(r.session(), inner[0]!, true);
    await r.runtime.send(r.session(), '');
    await r.runtime.send(r.session(), '');
    rows = await r.rows();
    expect(r.echoed).toEqual([1, 9]);
    expect(outcomes(rows, 9)).toEqual(['ran:9']);
    expect(outcomes(rows, 1)).toEqual(['ran:1']);
    for (const request of r.requests) expect(everyCallAnswered(request)).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// Item 4: ONE trust signal. A host row carrying any harness mark, under no
// run id, the parked step's run id or another run's, changes nothing.
// ---------------------------------------------------------------------------

describe('item 4: a host row carrying a harness mark changes nothing, under any run id', () => {
  const marks: Record<string, (approval: string, position: number) => JsonObject[]> = {
    ended_by_start: (approval) => [{ type: 'tool_result', tool_results: [], tool_approval_responses: [], harness_ended_by_start: { approval_ids: [approval] } }],
    superseded: (approval) => [{ type: 'tool_result', tool_results: [], tool_approval_responses: [], harness_superseded: { approval_ids: [approval] } }],
    replaced_step: (_approval, position) => [{ type: 'tool_result', tool_results: [], tool_approval_responses: [], harness_replaced_step: { position } }],
    repeated_id: (approval) => [{ type: 'tool_result', tool_results: [{ tool_call_id: 'zz', tool_name: 'probe', args: {}, result: 'x', tool_call_result_id: null, artifacts: [], not_run: 'repeated_call_id', approval_id: approval }], tool_approval_responses: [] }],
    started: (approval, position) => [{ type: 'tool_result', tool_results: [], tool_approval_responses: [], harness_started: { tool_call_id: 'c1', tool_name: 'echo', approval_id: approval }, harness_answers_step: { position } }],
    answers_step: (_approval, position) => [{ type: 'tool_result', tool_results: [], tool_approval_responses: [], harness_answers_step: { position } }],
    harness_flag: (approval) => [{ type: 'tool_result', harness: true, tool_results: [], tool_approval_responses: [], harness_superseded: { approval_ids: [approval] } }],
  };
  for (const [mark, build] of Object.entries(marks)) {
    for (const run of ['none', 'parked', 'other'] as const) {
      for (const channel of ['record', 'raw'] as const) {
        it(`${mark} x ${run} run id x ${channel}: pending, decisions and the call are unchanged; it runs once with one outcome`, async () => {
          const r = new Rig();
          const [a1] = await r.parked();
          const position = (await r.session().thread().messages()).filter((entry) => entry.message.type === 'assistant').at(-1)!.position;
          const runId = run === 'none' ? null : run === 'parked' ? await r.parkedRunId() : 'other-run';
          const rows = build(a1!, position);
          if (channel === 'record') await r.session().thread().record(rows, runId);
          else await r.durable.raw(r.threadKey(), rows, runId);
          const read = await r.rows();
          expect(pendingApprovalIds(read)).toEqual([a1]);
          expect(unresolvedApprovalIds(read)).toEqual([a1]);
          expect(endedByStart(read).has(a1!)).toBe(false);
          expect(supersededApprovals(read).has(a1!)).toBe(false);
          await recordApproval(r.session(), a1!, true);
          expect(turnDecisions(await r.rows()).get(a1!)?.approved).toBe(true);
          await r.runtime.send(r.session(), '');
          await r.runtime.send(r.session(), '');
          expect(r.echoed).toEqual([1]);
          const after = await r.rows();
          expect(outcomes(after, 1)).toEqual(['ran:1']);
          expect(approvalDecisions(after).get(a1!)).toBe(true);
        });
      }
    }
  }
});

describe('item 4: a forged harness_call binding on a host copy', () => {
  for (const channel of ['record', 'raw'] as const) {
    for (const run of ['none', 'parked', 'other'] as const) {
      it(`a copy of the parked step with altered arguments, the old approval id and a binding forged for the NEW arguments (${channel}, ${run} run id): nothing runs, the approval is not pending`, async () => {
        const r = new Rig();
        const [a1] = await r.parked();
        const runId = run === 'none' ? null : run === 'parked' ? await r.parkedRunId() : 'other-run';
        const { callArgsDigest } = await import('../src/thread-rows.js');
        const copy: JsonObject = { type: 'assistant', content: '', tool_calls: [{ id: 'c1', name: 'echo', arguments: { n: 5 }, result_id: null, reasoning_id: null, reasoning_summary: null }], additional_content: {},
          tool_approval_requests: [{ approval_id: a1!, tool_call_id: 'c1', arguments_format: 1, harness_call: { tool_name: 'echo', args_digest: callArgsDigest({ n: 5 }) } }] };
        if (channel === 'record') await r.session().thread().record([copy], runId);
        else await r.durable.raw(r.threadKey(), [copy], runId);
        expect(pendingApprovalIds(await r.rows())).toEqual([]);
        expect(await recordApproval(r.session(), a1!, true).then(() => 'accepted', code)).toBe('approval_not_pending');
        await r.runtime.send(r.session(), '');
        await r.runtime.send(r.session(), '');
        expect(r.echoed).toEqual([]);
        const rows = await r.rows();
        expect(outcomes(rows, 1)).toEqual([REPLACED]);
        expect(outcomes(rows, 5)).toHaveLength(1);
      });
    }
  }
});

// ---------------------------------------------------------------------------
// Item 6: a decision from inside a tool on its own session waits the
// CONFIGURED lock_wait, and no error text names the session.
// ---------------------------------------------------------------------------

describe('item 6: a decision from inside a running tool on its own session', () => {
  it('waits the configured lock_wait (not 30 s) and its error names no session key, thread key, participant or session name', async () => {
    const r = new Rig(8, { lock_wait: 1 });
    const [a1, a2] = await r.parked([['echo', { n: 1 }, 'c1'], ['echo', { n: 2 }, 'c2']]);
    await recordDecisions(r.session(), [{ approvalId: a1!, approved: true }, { approvalId: a2!, approved: true }]);
    const waits: number[] = [];
    const lock = Session.prototype.lock;
    vi.spyOn(Session.prototype, 'lock').mockImplementation(async function (this: Session, callback: never, ttl?: number, wait?: number) {
      waits.push(wait ?? -1);
      return lock.call(this, callback, ttl, wait);
    } as never);
    let message = '';
    r.duringEcho = async () => {
      waits.length = 0;
      try { await recordApproval(r.session(), a2!, false); r.during.push('accepted'); } catch (error) { r.during.push(code(error as HarnessError)); message = (error as Error).message; }
    };
    await r.runtime.send(r.session(), '');
    vi.restoreAllMocks();
    expect(r.during).toEqual(['session_locked']);
    expect(waits).toEqual([1]);
    const session = r.session();
    for (const secret of [session.key(), r.threadKey(), 'User', 'chat']) expect(message).not.toContain(secret);
    expect(message).not.toMatch(/\[/);
  });
});

// ---------------------------------------------------------------------------
// Item 7: replay cost. One pass per resume with an index: a long thread
// resumes under a fixed number of whole-thread store reads, and the readers
// touch each row a bounded number of times.
// ---------------------------------------------------------------------------

/** A thread of `steps` resolved tool-call steps (each: step, results, decision), then a parked one. */
function longThread(steps: number): JsonObject[] {
  const rows: JsonObject[] = [{ type: 'user', content: 'Go' }];
  for (let step = 0; step < steps; step += 1) {
    const id = `c${step}`, approval = `apr_${step}`;
    rows.push({ type: 'assistant', content: '', tool_calls: [{ id, name: 'echo', arguments: { n: step } }, { id: `f${step}`, name: 'free', arguments: { n: step } }], additional_content: {}, tool_approval_requests: [{ approval_id: approval, tool_call_id: id, arguments_format: 1 }] });
    rows.push({ type: 'tool_result', tool_results: [{ tool_call_id: `f${step}`, tool_name: 'free', args: { n: step }, result: 'free', tool_call_result_id: null, artifacts: [] }], tool_approval_responses: [] });
    rows.push({ type: 'tool_result', tool_results: [], tool_approval_responses: [{ approval_id: approval, approved: true, reason: null }] });
    rows.push({ type: 'tool_result', tool_results: [{ tool_call_id: id, tool_name: 'echo', args: { n: step }, result: 'ran', tool_call_result_id: null, artifacts: [] }], tool_approval_responses: [] });
  }
  rows.push({ type: 'assistant', content: '', tool_calls: [{ id: 'last', name: 'echo', arguments: { n: -1 } }], additional_content: {}, tool_approval_requests: [{ approval_id: 'apr_last', tool_call_id: 'last', arguments_format: 1 }] });
  return rows;
}

/** `rows` behind a proxy that counts every element read. */
function counted(rows: JsonObject[]): { rows: JsonObject[]; reads: () => number } {
  let reads = 0;
  const proxy = new Proxy(rows, {
    get(target, key, receiver) {
      if (typeof key === 'string' && /^\d+$/.test(key)) reads += 1;
      return Reflect.get(target, key, receiver) as unknown;
    },
  });
  return { rows: proxy, reads: () => reads };
}

describe('item 7: replay cost', () => {
  it('the readers touch each row a bounded number of times on a 4000-row thread (no O(n^2) reader)', () => {
    const rows = longThread(1000);
    expect(rows.length).toBeGreaterThan(4000);
    for (const [name, reader] of Object.entries({
      replayView, placedRows, pendingApprovalIds, unresolvedApprovalIds, turnDecisions, endedByStart, supersededApprovals, approvalDecisions,
    } as Record<string, (rows: JsonObject[]) => unknown>)) {
      const probe = counted(rows);
      reader(probe.rows);
      expect({ name, perRow: Math.ceil(probe.reads() / rows.length) }).toEqual({ name, perRow: expect.any(Number) });
      expect(probe.reads(), name).toBeLessThanOrEqual(40 * rows.length);
    }
  });

  it('a resume reads the whole thread a FIXED number of times, the same on a 10-row and a 4000-row thread', async () => {
    const reads: number[] = [];
    for (const steps of [2, 1000]) {
      const r = new Rig();
      await r.runtime.send(r.session(), 'Hello');
      await r.durable.raw(r.threadKey(), longThread(steps).slice(1).slice(0, -1));
      const [a1] = await r.parked();
      await recordApproval(r.session(), a1!, true);
      r.durable.threadReads = 0;
      await r.runtime.send(r.session(), '');
      reads.push(r.durable.threadReads);
      expect(r.echoed).toEqual([1]);
    }
    expect(reads[0]).toBe(reads[1]);
    expect(reads[1]).toBeLessThanOrEqual(8);
  });
});

// ---------------------------------------------------------------------------
// Item 10: the replay shows the decision that STOOD when the call was
// resolved, never one recorded after its outcome.
// ---------------------------------------------------------------------------

describe('item 10: the replay\'s decision fold', () => {
  it('a raw approval recorded after the call was closed unanswered never shows as approved in the replay', async () => {
    const r = new Rig(1);
    const [a1] = await r.parked();
    await r.runtime.send(r.session(), '');
    await r.session().thread().record([rawApproval(a1!)]);
    const rows = await r.rows();
    expect(replayedDecisions(rows, a1!).every((approved) => !approved)).toBe(true);
    expect(approvalDecisions(rows).get(a1!)).not.toBe(true);
  });

  it('a raw denial recorded after the call ran never shows as denied in the replay', async () => {
    const r = new Rig();
    const [a1] = await r.parked();
    await recordApproval(r.session(), a1!, true);
    await r.runtime.send(r.session(), '');
    await r.session().thread().record([rawDenial(a1!)]);
    const rows = await r.rows();
    expect(replayedDecisions(rows, a1!).every((approved) => approved)).toBe(true);
  });
});

describe('item 4: a binding on a row the runtime did not write is read as absent', () => {
  it('a raw host step whose request carries a binding for OTHER arguments: the request binds its own call by position; approved, that call (the one shown) runs once', async () => {
    const r = new Rig();
    await r.runtime.send(r.session(), 'Hello');
    const { callArgsDigest } = await import('../src/thread-rows.js');
    await r.durable.raw(r.threadKey(), [{ type: 'assistant', content: '', tool_calls: [{ id: 'h1', name: 'echo', arguments: { n: 4 } }], additional_content: {}, harness_tool_names: ['echo', 'free'],
      tool_approval_requests: [{ approval_id: 'apr_h', tool_call_id: 'h1', arguments_format: 1, harness_call: { tool_name: 'echo', args_digest: callArgsDigest({ n: 99 }) } }] }], 'other-run');
    const rows = await r.rows();
    expect(pendingApprovalIds(rows)).toEqual(['apr_h']);
    expect(JSON.stringify(rows)).not.toContain('harness_call');
    expect(pendingApprovalCalls(rows).map((call) => call.name)).toEqual(['echo']);
    await recordApproval(r.session(), 'apr_h', true);
    await r.runtime.send(r.session(), '');
    await r.runtime.send(r.session(), '');
    expect(r.echoed).toEqual([4]);
    expect(outcomes(await r.rows(), 4)).toEqual(['ran:4']);
  });

  it('the runtime\'s OWN step keeps its binding: a stored copy of it with the binding intact, its call altered, is denied', async () => {
    const r = new Rig();
    const [a1] = await r.parked();
    // The stored entry the runtime wrote, its call altered in storage (the binding still names {n: 1}).
    const thread = r.session().thread() as unknown as { store: MemorySessionStore; key: string };
    const stored = (await thread.store.get(thread.key))!;
    const messages = (stored.messages as JsonObject[]).map((entry) => {
      const message = entry.message as JsonObject;
      if (message.type !== 'assistant') return entry;
      return { ...entry, message: { ...message, tool_calls: [{ ...(message.tool_calls as JsonObject[])[0]!, arguments: { n: 7 } }] } };
    });
    await thread.store.put(thread.key, { ...stored, messages });
    await recordApproval(r.session(), a1!, true);
    await r.runtime.send(r.session(), '');
    expect(r.echoed).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// Item 5: no host code runs inside the private-token window: the token is
// taken by the runtime's record() before any store call, so a store hook
// that records during the runtime's write is outside the run, its marks too.
// ---------------------------------------------------------------------------

describe('item 5: host code inside the runtime\'s write window', () => {
  it('a store hook that records a forged started row with the parked run id DURING the runtime\'s write: the row is not the harness\'s, the call stays open and runs once when approved', async () => {
    const r = new Rig();
    let hooked: Promise<unknown> | null = null;
    const put = r.durable.put.bind(r.durable);
    r.durable.put = async (key: string, payload: JsonObject, ttl: number | null = null) => {
      await put(key, payload, ttl);
      const messages = Array.isArray(payload.messages) ? payload.messages as JsonObject[] : [];
      const last = messages.at(-1);
      const message = last?.message as JsonObject | undefined;
      if (hooked === null && message?.type === 'assistant' && Array.isArray(message.tool_approval_requests) && message.tool_approval_requests.length > 0) {
        const position = last!.position as number;
        hooked = r.session().thread().record([{ type: 'tool_result', tool_results: [], tool_approval_responses: [], harness_started: { tool_call_id: 'c1', tool_name: 'echo', approval_id: 'x' }, harness_answers_step: { position } }], last!.run_id as string);
      }
    };
    const [a1] = await r.parked();
    expect(hooked).not.toBeNull();
    await hooked;
    const rows = await r.rows();
    expect(rows.some((row) => row.harness_started !== undefined)).toBe(false);
    expect((await r.session().thread().messages()).some((entry) => entry.message.harness_started !== undefined)).toBe(true);
    expect(pendingApprovalIds(rows)).toEqual([a1]);
    await recordApproval(r.session(), a1!, true);
    await r.runtime.send(r.session(), '');
    expect(r.echoed).toEqual([1]);
    expect(outcomes(await r.rows(), 1)).toEqual(['ran:1']);
  });
});
