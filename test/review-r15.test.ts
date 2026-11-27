import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  HarnessError,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  Thread,
  ToolRegistry,
  recordApproval,
  recordDecisions,
  threadView,
  type Durability,
  type JsonObject,
  type JsonValue,
  type LlmRequest,
  type LlmResponse,
  type RuntimeConfig,
} from '../src/index.js';
import * as rowsModule from '../src/thread-rows.js';
import {
  ARGUMENTS_AS_SENT,
  endedByStart,
  pausedApprovalIds,
  pendingApprovalIds,
  supersededApprovals,
  unresolvedApprovalIds,
} from '../src/thread-rows.js';
import { approvalDecisions, pendingApprovalCalls } from '../src/plans/thread.js';

// G-387 round 15 (BRIEF-R15), at the public seams: send, recordApproval /
// recordDecisions, a host's thread().record(), and a tool that writes to its
// own session while it runs. Every interleaving is forced by a HOOK (a tool
// that acts while it runs, a store write that fails), never by timing. No
// network; every store is in memory.

const network = vi.fn(() => { throw new Error('A test tried to use the network.'); });
beforeEach(() => { vi.stubGlobal('fetch', network); });
afterEach(() => { expect(network).not.toHaveBeenCalled(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });

const REPLACED = 'Tool call was not run because a later step replaced it';
const NEW_MESSAGE = 'A new message.';
// Read through the module object, so a name the source does not export yet
// reads undefined and the test fails on its expectation, not on its import.
const rows15 = rowsModule as unknown as Record<string, unknown>;
const STARTED_KEY = (): string => rows15.STARTED_KEY as string;
const ANSWERS_STEP_KEY = (): string => rows15.ANSWERS_STEP_KEY as string;
const INTERRUPTED = (): string => rows15.INTERRUPTED_RESULT as string;

/** A memory store whose writes of a thread payload can be made to fail, by the last message they carry. */
class Store extends MemorySessionStore {
  failWhen: ((message: JsonObject) => boolean) | null = null;

  constructor(private readonly kind: Durability) {
    super();
  }

  override durability(): Durability {
    return this.kind;
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
}

class Rig {
  readonly requests: LlmRequest[] = [];
  readonly queue: LlmResponse[] = [];
  readonly durable = new Store('durable');
  readonly ephemeral = new Store('volatile');
  readonly runtime: AgentRuntime;
  readonly harness: PrismHarness;
  readonly echoed: number[] = [];
  /** Runs inside the echo tool, once, before it returns. */
  duringEcho: (() => Promise<void>) | null = null;
  /** What the echo hook produced (a write's outcome, an error code). */
  during: string[] = [];

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
      .register({ name: 'free', handle: (args: JsonObject) => `free:${String(args.n)}` });
    this.runtime = new AgentRuntime({
      modes, tools, config,
      client: async (request) => {
        this.requests.push(request);
        return this.queue.shift() ?? { text: 'Done.', finishReason: 'stop' };
      },
    });
    const durable = this.durable, ephemeral = this.ephemeral;
    this.harness = new PrismHarness({ drivers: { ephemeral: () => ephemeral, durable: () => durable }, stores: { ephemeral: 'ephemeral', durable: 'durable' }, runtime: this.runtime });
  }

  session(scope = 'chat'): ReturnType<ReturnType<PrismHarness['for']>['session']> {
    return this.harness.for({ type: 'User', id: 1 }).session(scope);
  }

  calls(calls: [string, JsonObject, string][]): void {
    this.queue.push({ text: '', finishReason: 'tool_calls', toolCalls: calls.map(([name, args, id]) => ({ id, name, arguments: args })) });
  }

  /** The thread as the harness's readers take it (only the harness's own rows carry its marks). */
  async rows(scope = 'chat'): Promise<JsonObject[]> {
    return await this.session(scope).thread().readerRows();
  }

  /** Parked on the given calls (default: one gated echo, c1); their approval ids. */
  async parked(calls: [string, JsonObject, string][] = [['echo', { n: 1 }, 'c1']]): Promise<string[]> {
    this.calls(calls);
    const response = await this.runtime.send(this.session(), 'Go');
    expect(response.finishReason).toBe('awaiting_approval');
    return response.pendingApprovals.map((pending) => pending.id);
  }
}

/**
 * Every outcome recorded for the echo call carrying token `n`, COUNTED (a
 * multiset): an exact copy of an entry an EARLIER row holds is the same
 * outcome carried again (the resolution row carries the step's results), and
 * is not counted twice; two entries in one row are two outcomes.
 */
function echoOutcomes(rows: JsonObject[], n: number): string[] {
  const earlier = new Set<string>();
  const out: string[] = [];
  for (const row of rows) {
    if (row.type !== 'tool_result') continue;
    const here: string[] = [];
    for (const entry of (row.tool_results ?? []) as JsonObject[]) {
      if (entry.tool_name !== 'echo' || (entry.args as JsonObject).n !== n) continue;
      const copy = JSON.stringify([entry.tool_call_id, entry.args, entry.result]);
      if (earlier.has(copy)) continue;
      here.push(copy);
      out.push(String(entry.result));
    }
    for (const copy of here) earlier.add(copy);
  }
  return out;
}

const code = (error: { code?: string }): string => error.code ?? 'threw';

/** A host's assistant step with one echo call, and its approval request when `approval` is given. */
function hostStep(id: string, n: number, approval: string | null): JsonObject {
  return {
    type: 'assistant', content: '',
    tool_calls: [{ id, name: 'echo', arguments: { n }, result_id: null, reasoning_id: null, reasoning_summary: null }],
    additional_content: {},
    tool_approval_requests: approval === null ? [] : [{ approval_id: approval, tool_call_id: id, arguments_format: ARGUMENTS_AS_SENT }],
    harness_tool_names: ['echo', 'free'],
  };
}

// ---------------------------------------------------------------------------
// Item 1: the write-ahead `started` row. A superseding write during an
// approved call is never refused and never denies the call; the call keeps
// exactly one outcome, its real result.
// ---------------------------------------------------------------------------

describe('item 1: a superseding write while an approved call runs', () => {
  const kinds: Record<string, JsonObject> = {
    'a user row': { type: 'user', content: 'Host row.' },
    'an assistant text row': { type: 'assistant', content: 'Host answer.', tool_calls: [], tool_approval_requests: [] },
    'a tool-call step reusing the running call\'s id': hostStep('c1', 3, 'apr_later'),
    'a tool-call step with another id': hostStep('c9', 3, 'apr_later'),
  };

  for (const [label, row] of Object.entries(kinds)) {
    it(`a host records ${label} from inside the running call: the write is accepted, the call keeps ONE outcome (its real result), no replaced denial names it`, async () => {
      const r = new Rig();
      const [id] = await r.parked();
      await recordApproval(r.session(), id!, true);
      r.duringEcho = async () => { r.during.push(await r.session().thread().record([row]).then(() => 'recorded', code)); };
      await r.runtime.send(r.session(), '');
      expect(r.during).toEqual(['recorded']);

      const rows = await r.rows();
      expect(r.echoed).toEqual([1]);
      expect(echoOutcomes(rows, 1)).toEqual(['ran:1']);
      expect(supersededApprovals(rows).has(id!)).toBe(false);
      expect(approvalDecisions(rows).get(id!)).toBe(true);
      // The started row was written before the call ran, in its own segment.
      const step = rows.findIndex((candidate) => candidate.type === 'assistant');
      const started = rows.findIndex((candidate) => STARTED_KEY() !== undefined && Object.hasOwn(candidate, STARTED_KEY()));
      expect(started).toBeGreaterThan(step);
      expect(rows[started]![ANSWERS_STEP_KEY()]).toEqual({ position: step + 1 });
      // Two more resumes change nothing: it never runs again, nothing more is written for it.
      const count = rows.length;
      for (let i = 0; i < 2; i += 1) await r.runtime.send(r.session(), '');
      expect(r.echoed.filter((n) => n === 1)).toEqual([1]);
      expect(echoOutcomes(await r.rows(), 1)).toEqual(['ran:1']);
      if (row.type !== 'assistant' || (row.tool_calls as JsonValue[]).length === 0) expect((await r.rows()).length).toBeGreaterThanOrEqual(count);
    });
  }

  it('a tool-call step reusing the call\'s id, recorded mid-call, the run then continuing: the run\'s next row replaces the host\'s step (ONE denial for its call), and the earlier result never answers it', async () => {
    const r = new Rig();
    const [id] = await r.parked();
    await recordApproval(r.session(), id!, true);
    r.duringEcho = async () => { await r.session().thread().record([hostStep('c1', 3, 'apr_later')]); };
    await r.runtime.send(r.session(), '');
    for (let i = 0; i < 2; i += 1) await r.runtime.send(r.session(), '');
    const rows = await r.rows();
    expect(r.echoed).toEqual([1]);
    expect(echoOutcomes(rows, 1)).toEqual(['ran:1']);
    expect(echoOutcomes(rows, 3)).toEqual([REPLACED]);
    expect(await recordApproval(r.session(), 'apr_later', true).then(() => 'accepted', code)).toBe('approval_not_pending');
  });

  it('a tool-call step reusing the call\'s id, recorded mid-call on a SPENT run (no model step after): the result stays with its call, the later call is decided by ITS OWN approval and runs only when approved', async () => {
    const r = new Rig(1);
    const [id] = await r.parked();
    await recordApproval(r.session(), id!, true);
    r.duringEcho = async () => { await r.session().thread().record([hostStep('c1', 3, 'apr_later')]); };
    await r.runtime.send(r.session(), '');
    const rows = await r.rows();
    expect(echoOutcomes(rows, 1)).toEqual(['ran:1']);
    expect(echoOutcomes(rows, 3)).toEqual([]);
    // The later call is open and pending on its own approval; the earlier result never answered it.
    expect(pendingApprovalIds(rows)).toEqual(['apr_later']);
    await recordApproval(r.session(), 'apr_later', true);
    await r.runtime.send(r.session(), '');
    await r.runtime.send(r.session(), '');
    expect(r.echoed).toEqual([1, 3]);
    expect(echoOutcomes(await r.rows(), 3)).toEqual(['ran:3']);
  });

  it('a new message from another run while the call runs: it does not wait on the session lock, writes no replaced denial for the running call, and the call keeps its one result', async () => {
    const r = new Rig(8, { lock_wait: 1 });
    const [id] = await r.parked();
    await recordApproval(r.session(), id!, true);
    r.duringEcho = async () => { r.during.push(await r.runtime.send(r.session(), NEW_MESSAGE).then(() => 'ran', code)); };
    await r.runtime.send(r.session(), '');
    expect(r.during).toEqual(['ran']);
    const rows = await r.rows();
    expect(r.echoed).toEqual([1]);
    expect(echoOutcomes(rows, 1)).toEqual(['ran:1']);
    expect(supersededApprovals(rows).has(id!)).toBe(false);
  });

  it('two approved calls; a new message lands while the FIRST runs: the first keeps its result, the second gets ONE replaced denial and never runs', async () => {
    const r = new Rig();
    const ids = await r.parked([['echo', { n: 1 }, 'c1'], ['echo', { n: 2 }, 'c2']]);
    await recordDecisions(r.session(), ids.map((approvalId) => ({ approvalId, approved: true })));
    r.duringEcho = async () => { await r.session().thread().record([{ type: 'user', content: 'Host row.' }]); };
    await r.runtime.send(r.session(), '');
    for (let i = 0; i < 2; i += 1) await r.runtime.send(r.session(), '');
    const rows = await r.rows();
    expect(r.echoed).toEqual([1]);
    expect(echoOutcomes(rows, 1)).toEqual(['ran:1']);
    expect(echoOutcomes(rows, 2)).toEqual([REPLACED]);
  });
});

describe('item 1: a tool that writes to its OWN session while it runs', () => {
  const kinds: Record<string, JsonObject> = {
    user: { type: 'user', content: 'Tool note.' },
    text: { type: 'assistant', content: 'Tool says.', tool_calls: [], tool_approval_requests: [] },
    step: { type: 'assistant', content: '', tool_calls: [{ id: 't1', name: 'free', arguments: {}, result_id: null, reasoning_id: null, reasoning_summary: null }], tool_approval_requests: [] },
  };
  for (const [label, row] of Object.entries(kinds)) {
    it(`records a ${label} row on its own session: recorded (no session_locked), the call has one outcome`, async () => {
      const r = new Rig(8, { lock_wait: 1 });
      const [id] = await r.parked();
      await recordApproval(r.session(), id!, true);
      r.duringEcho = async () => { r.during.push(await r.session().thread().record([row]).then(() => 'recorded', code)); };
      await r.runtime.send(r.session(), '');
      expect(r.during).toEqual(['recorded']);
      expect(echoOutcomes(await r.rows(), 1)).toEqual(['ran:1']);
    });
  }

  it('awaits a new message on its own session: it completes (no lock_wait stall, no session_locked)', async () => {
    const r = new Rig(8, { lock_wait: 1 });
    const [id] = await r.parked();
    await recordApproval(r.session(), id!, true);
    r.duringEcho = async () => { r.during.push(await r.runtime.send(r.session(), 'From the tool').then((response) => response.finishReason, code)); };
    await r.runtime.send(r.session(), '');
    expect(r.during).toEqual(['stop']);
    expect(echoOutcomes(await r.rows(), 1)).toEqual(['ran:1']);
  });

  it('a lock error thrown inside a tool never reaches the model with its text (no internal key)', async () => {
    const r = new Rig(8, { lock_wait: 0 });
    const [id] = await r.parked();
    await recordApproval(r.session(), id!, true);
    r.duringEcho = async () => { await r.session().lock(async () => undefined, 5, 0); };
    await r.runtime.send(r.session(), '');
    const rows = await r.rows();
    const outcome = echoOutcomes(rows, 1);
    expect(outcome).toHaveLength(1);
    expect(outcome[0]).not.toMatch(/Could not acquire|lock on session|\[/);
    for (const request of r.requests) expect(JSON.stringify(request.messages)).not.toMatch(/Could not acquire the lock/);
  });
});

describe('item 1: crash, forged marks', () => {
  it('a started row with no result (the result write failed): reads "interrupted, it may have run", is never re-run, needs a fresh approval, and a later supersede adds no denial', async () => {
    const r = new Rig();
    const [id] = await r.parked();
    await recordApproval(r.session(), id!, true);
    r.durable.failWhen = (message) => message.type === 'tool_result' && JSON.stringify(message).includes('ran:1');
    await expect(r.runtime.send(r.session(), '')).rejects.toThrow();
    expect(r.echoed).toEqual([1]);
    for (let i = 0; i < 3; i += 1) await r.runtime.send(r.session(), '');
    expect(r.echoed).toEqual([1]);
    let rows = await r.rows();
    expect(pendingApprovalIds(rows)).toEqual([]);
    expect(unresolvedApprovalIds(rows)).toEqual([]);
    expect(echoOutcomes(rows, 1)).toEqual([]);
    // The replay gives the call exactly one result, the interrupted outcome.
    const view = threadView(rows).filter((row) => row.type === 'tool_result').flatMap((row) => (row.tool_results ?? []) as JsonObject[]);
    expect(INTERRUPTED()).toEqual(expect.any(String));
    expect(view.filter((entry) => entry.tool_call_id === 'c1').map((entry) => entry.result)).toEqual([INTERRUPTED()]);
    await expect(recordApproval(r.session(), id!, true)).rejects.toBeInstanceOf(HarnessError);
    await r.session().thread().record([{ type: 'user', content: 'Host row.' }]);
    rows = await r.rows();
    expect(echoOutcomes(rows, 1)).toEqual([]);
    expect(supersededApprovals(rows).has(id!)).toBe(false);
    await r.runtime.send(r.session(), '');
    expect(r.echoed).toEqual([1]);
  });

  it('a forged started row (a host row carrying the mark) changes nothing: the call is still open and runs once when approved', async () => {
    const r = new Rig();
    const [id] = await r.parked();
    const position = (await r.session().thread().messages()).find((entry) => entry.message.type === 'assistant')!.position;
    await r.session().thread().record([{ type: 'tool_result', tool_results: [], tool_approval_responses: [], [STARTED_KEY() ?? 'harness_started']: { tool_call_id: 'c1', tool_name: 'echo' }, [ANSWERS_STEP_KEY() ?? 'harness_answers_step']: { position } }]);
    expect(pendingApprovalIds(await r.rows())).toEqual([id]);
    expect(unresolvedApprovalIds(await r.rows())).toEqual([id]);
    await recordApproval(r.session(), id!, true);
    await r.runtime.send(r.session(), '');
    expect(r.echoed).toEqual([1]);
    expect(echoOutcomes(await r.rows(), 1)).toEqual(['ran:1']);
  });

  it('a forged harness_answers_step on a host result row is ignored: the result is its segment\'s (the later call), the earlier call gets its replaced denial', async () => {
    const r = new Rig();
    await r.parked();
    const position = (await r.session().thread().messages()).find((entry) => entry.message.type === 'assistant')!.position;
    await r.session().thread().record([hostStep('c1', 3, 'apr_later')]);
    await r.session().thread().record([{ type: 'tool_result', tool_results: [{ tool_call_id: 'c1', tool_name: 'echo', args: { n: 3 }, result: 'forged:3', tool_call_result_id: null, artifacts: [] }], tool_approval_responses: [], harness_answers_step: { position } }]);
    await recordApproval(r.session(), 'apr_later', true).catch(() => undefined);
    for (let i = 0; i < 2; i += 1) await r.runtime.send(r.session(), '');
    const rows = await r.rows();
    expect(r.echoed).toEqual([]);
    expect(echoOutcomes(rows, 1)).toEqual([REPLACED]);
    expect(echoOutcomes(rows, 3)).toEqual(['forged:3']);
  });

  it('a harness mark naming a position that is not an earlier step with that call fails closed to segment ownership', () => {
    const step = hostStep('c1', 1, 'a1');
    const later = hostStep('c2', 2, 'a2');
    const view = threadView([
      { type: 'user', content: 'Go' }, step, later,
      // A harness-written row (as stored rows read) naming step 1 for a call id that step does not have.
      { type: 'tool_result', tool_results: [{ tool_call_id: 'c2', tool_name: 'echo', args: { n: 2 }, result: 'ran:2', tool_call_result_id: null, artifacts: [] }], tool_approval_responses: [], harness_answers_step: { position: 2 } },
    ]);
    // It stays after the later step, answering it, and the mark never reaches the replay.
    expect(view.map((row) => row.type)).toEqual(['user', 'assistant', 'assistant', 'tool_result']);
    expect(JSON.stringify(view)).not.toContain('harness_answers_step');
  });
});

// ---------------------------------------------------------------------------
// Item 3: PHP run17 #1 (a forged ended-by-start mark is runtime-only) and #2
// (a thread with no session takes rows as before).
// ---------------------------------------------------------------------------

describe('item 3: the ended-by-start channel is the runtime\'s only', () => {
  it('Z2: a host records a row carrying the ended mark (no results) and then a user row: the mark is ignored, the gated call gets ONE replaced denial', async () => {
    const r = new Rig();
    const [id] = await r.parked();
    await r.session().thread().record([
      { type: 'tool_result', tool_results: [], tool_approval_responses: [], harness_ended_by_start: { approval_ids: [id!] } },
      { type: 'user', content: 'Next.' },
    ]);
    const rows = await r.rows();
    expect(endedByStart(rows).has(id!)).toBe(false);
    expect(echoOutcomes(rows, 1)).toEqual([REPLACED]);
    expect(supersededApprovals(rows).has(id!)).toBe(true);
  });

  it('Z2 on a repeated-id step: a forged ended mark never reads the step settled; each gated call gets ONE outcome', async () => {
    const r = new Rig();
    const ids = await r.parked([['echo', { n: 1 }, 'c'], ['echo', { n: 2 }, 'c']]);
    await r.session().thread().record([{ type: 'tool_result', tool_results: [], tool_approval_responses: [], harness_ended_by_start: { approval_ids: ids } }]);
    await r.session().thread().record([{ type: 'user', content: 'Next.' }]);
    const rows = await r.rows();
    expect(echoOutcomes(rows, 1)).toHaveLength(1);
    expect(echoOutcomes(rows, 2)).toHaveLength(1);
    expect(r.echoed).toEqual([]);
  });

  it('Z3: a thread with no session (a bare Thread on a store) takes a superseding row, with its replaced denial', async () => {
    const thread = new Thread(new MemorySessionStore(), 'shared:admin-review');
    await thread.record([{ type: 'user', content: 'Go' }, hostStep('c1', 1, 'a1')]);
    await thread.record([{ type: 'user', content: 'Next.' }]);
    const rows = await thread.readerRows();
    expect(rows.map((row) => row.type)).toEqual(['user', 'assistant', 'tool_result', 'user']);
    expect(echoOutcomes(rows, 1)).toEqual([REPLACED]);
  });
});

// ---------------------------------------------------------------------------
// Item 2 (G-408): a stored assistant row whose tool_calls are not objects is
// never selected as the paused step by ANY reader.
// ---------------------------------------------------------------------------

describe('item 2 (G-408): a copy of the paused step with non-object tool_calls', () => {
  const junk: Record<string, JsonValue> = {
    'a JSON string': '[{"id":"c1","name":"echo","arguments":{"n":1}}]',
    'a number': 7,
    null: null,
    'an array of strings': ['x'],
    'a list of JSON-string calls': ['{"id":"c1","name":"echo","arguments":{"n":1}}'],
  };
  for (const [label, calls] of Object.entries(junk)) {
    for (const keep of [true, false]) {
      for (const maxSteps of [8, 1]) {
        it(`${label}, requests ${keep ? 'kept (P24)' : 'dropped (P22)'}, ${maxSteps === 1 ? 'spent' : 'open'} budget: never run, ONE outcome, no row per resume, no reader takes it for the step`, async () => {
          const r = new Rig(maxSteps);
          const [id] = await r.parked();
          await recordApproval(r.session(), id!, true);
          await r.session().thread().record([{
            type: 'assistant', content: '', tool_calls: calls, additional_content: {},
            tool_approval_requests: keep ? [{ approval_id: id!, tool_call_id: 'c1', arguments_format: ARGUMENTS_AS_SENT }] : [],
            harness_tool_names: ['echo', 'free'],
          }]);
          const counts: number[] = [];
          for (let i = 0; i < 3; i += 1) {
            await r.runtime.send(r.session(), '').catch(() => undefined);
            counts.push((await r.rows()).filter((row) => row.type === 'tool_result').length);
          }
          const rows = await r.rows();
          expect(r.echoed).toEqual([]);
          expect(echoOutcomes(rows, 1)).toEqual([REPLACED]);
          expect(new Set(counts).size).toBe(1);
          expect(pendingApprovalIds(rows)).toEqual([]);
          expect(unresolvedApprovalIds(rows)).toEqual([]);
          expect(pausedApprovalIds(rows)).toEqual([]);
          expect(pendingApprovalCalls(rows)).toEqual([]);
          expect(supersededApprovals(rows).has(id!)).toBe(true);
          expect(approvalDecisions(rows).get(id!)).not.toBe(true);
          expect(await recordApproval(r.session(), id!, true).then(() => 'accepted', code)).toBe('approval_not_pending');
        });
      }
    }
  }

  it('recorded in-run (from inside the running approved call), requests kept: the call keeps its one result and the copy is never the step', async () => {
    const r = new Rig();
    const [id] = await r.parked();
    await recordApproval(r.session(), id!, true);
    r.duringEcho = async () => {
      await r.session().thread().record([{ type: 'assistant', content: '', tool_calls: ['x'], additional_content: {}, tool_approval_requests: [{ approval_id: id!, tool_call_id: 'c1', arguments_format: ARGUMENTS_AS_SENT }] }]);
    };
    await r.runtime.send(r.session(), '');
    for (let i = 0; i < 2; i += 1) await r.runtime.send(r.session(), '');
    const rows = await r.rows();
    expect(r.echoed).toEqual([1]);
    expect(echoOutcomes(rows, 1)).toEqual(['ran:1']);
    expect(pendingApprovalIds(rows)).toEqual([]);
    expect(await recordApproval(r.session(), id!, true).then(() => 'accepted', code)).toBe('approval_not_pending');
  });
});

// ---------------------------------------------------------------------------
// Added (coordinator): an approval belongs ONLY to the step that FIRST asked
// for it (TS 0.11.x #92). A recorded copy of the paused step that carries a
// NEW call and keeps the OLD approval request never lets that approval run
// the new call, decided before or after the copy.
// ---------------------------------------------------------------------------

describe('added: an approval belongs only to the step that first asked for it', () => {
  const copies: Record<string, JsonValue> = {
    'a mixed list (non-objects and a new call)': ['x', { id: 'c1', name: 'echo', arguments: { n: 5 }, result_id: null, reasoning_id: null, reasoning_summary: null }, null, 3],
    'a valid step with a new call under the old call id': [{ id: 'c1', name: 'echo', arguments: { n: 5 }, result_id: null, reasoning_id: null, reasoning_summary: null }],
  };
  for (const [label, calls] of Object.entries(copies)) {
    for (const when of ['before', 'after'] as const) {
      for (const maxSteps of [8, 1]) {
        it(`${label}, the old approval decided ${when} the copy, ${maxSteps === 1 ? 'spent' : 'open'} budget: the new call never runs on the old approval`, async () => {
          const r = new Rig(maxSteps);
          const [id] = await r.parked();
          if (when === 'before') await recordApproval(r.session(), id!, true);
          await r.session().thread().record([{
            type: 'assistant', content: '', tool_calls: calls, additional_content: {},
            tool_approval_requests: [{ approval_id: id!, tool_call_id: 'c1', arguments_format: ARGUMENTS_AS_SENT }],
            harness_tool_names: ['echo', 'free'],
          }]);
          if (when === 'after') expect(await recordApproval(r.session(), id!, true).then(() => 'accepted', code)).toBe('approval_not_pending');
          for (let i = 0; i < 3; i += 1) await r.runtime.send(r.session(), '').catch(() => undefined);
          const rows = await r.rows();
          expect(r.echoed).toEqual([]);
          expect(echoOutcomes(rows, 1)).toEqual([REPLACED]);
          expect(pendingApprovalIds(rows)).toEqual([]);
          expect(pausedApprovalIds(rows)).toEqual([]);
          expect(approvalDecisions(rows).get(id!)).not.toBe(true);
        });
      }
    }
  }
});

// ---------------------------------------------------------------------------
// Added (Py run19 Info): clearing a thread while a started row has no result.
// ---------------------------------------------------------------------------

describe('added: Thread.clear() while an approved call runs', () => {
  it('the cleared thread never reads the call as not run and never runs it again', async () => {
    const r = new Rig();
    const [id] = await r.parked();
    await recordApproval(r.session(), id!, true);
    r.duringEcho = async () => { await r.session().thread().clear(); };
    await r.runtime.send(r.session(), '');
    for (let i = 0; i < 2; i += 1) await r.runtime.send(r.session(), '');
    const rows = await r.rows();
    expect(r.echoed).toEqual([1]);
    expect(rows.flatMap((row) => (row.tool_results ?? []) as JsonObject[]).map((entry) => entry.result)).not.toContain(REPLACED);
    expect(pendingApprovalIds(rows)).toEqual([]);
    expect(unresolvedApprovalIds(rows)).toEqual([]);
    // The result that landed after the clear names a step the thread no longer has: it is nobody's.
    expect(threadView(rows).filter((row) => row.type === 'tool_result').flatMap((row) => (row.tool_results ?? []) as JsonObject[]).map((entry) => entry.result)).not.toContain(INTERRUPTED());
  });
});

// ---------------------------------------------------------------------------
// Added (PHP 0.26.10 audit): a thread an OLDER version wrote. Rows in the
// v0.11.2 shapes (its assistantRow, its user row, its decision row, entries
// with no harness flag; that line wrote no replaced denial at all), stored
// straight to the store.
// ---------------------------------------------------------------------------

describe('added: upgrade from a thread v0.11.2 wrote', () => {
  const step = (id: string, n: number, approval: string): JsonObject => ({
    type: 'assistant', content: '',
    tool_calls: [{ id, name: 'echo', arguments: { n }, result_id: null, reasoning_id: null, reasoning_summary: null }],
    additional_content: {},
    tool_approval_requests: [{ approval_id: approval, tool_call_id: id, arguments_format: 1 }],
    harness_tool_names: ['echo', 'free'],
  });
  const decided = (approval: string): [JsonObject, null] => [{ type: 'tool_result', tool_results: [], tool_approval_responses: [{ approval_id: approval, approved: true, reason: null }] }, null];

  async function legacy(r: Rig, rows: [JsonObject, string | null][]): Promise<void> {
    const thread = r.session().thread() as unknown as { store: MemorySessionStore; key: string };
    await thread.store.put(thread.key, { messages: rows.map(([message, runId], index) => ({ position: index + 1, message, run_id: runId, recorded_at: '2026-11-01T00:00:00.000Z' })) as never });
  }

  for (const approvedBefore of [true, false]) {
    it(`a step v0.11.2 replaced with a USER row${approvedBefore ? ', approved before it' : ''}: approved and resumed, nothing runs, ONE outcome, the decision refused`, async () => {
      const r = new Rig();
      await legacy(r, [
        [{ type: 'user', content: 'Go' }, null],
        [step('c1', 1, 'apr_old'), 'run-x'],
        ...(approvedBefore ? [decided('apr_old')] : []),
        [{ type: 'user', content: 'Something else.' }, 'run-y'],
        [{ type: 'assistant', content: 'Sure.', tool_calls: [], additional_content: {}, tool_approval_requests: [] }, 'run-y'],
      ]);
      expect(await recordApproval(r.session(), 'apr_old', true).then(() => 'accepted', code)).toBe('approval_not_pending');
      for (let i = 0; i < 3; i += 1) await r.runtime.send(r.session(), '');
      const rows = await r.rows();
      expect(r.echoed).toEqual([]);
      expect(echoOutcomes(rows, 1)).toEqual([REPLACED]);
      expect(approvalDecisions(rows).get('apr_old')).not.toBe(true);
      expect(await recordApproval(r.session(), 'apr_old', true).then(() => 'accepted', code)).toBe('approval_not_pending');
    });

    it(`a step v0.11.2 replaced by a LATER step${approvedBefore ? ', approved before it' : ''}: approved and resumed, nothing of it runs, ONE outcome, the decision refused`, async () => {
      const r = new Rig();
      await legacy(r, [
        [{ type: 'user', content: 'Go' }, null],
        [step('c1', 1, 'apr_old'), 'run-x'],
        ...(approvedBefore ? [decided('apr_old')] : []),
        [step('c2', 2, 'apr_new'), 'run-y'],
      ]);
      expect(await recordApproval(r.session(), 'apr_old', true).then(() => 'accepted', code)).toBe('approval_not_pending');
      await recordApproval(r.session(), 'apr_new', false, 'not now');
      for (let i = 0; i < 3; i += 1) await r.runtime.send(r.session(), '');
      const rows = await r.rows();
      expect(r.echoed).toEqual([]);
      expect(echoOutcomes(rows, 1)).toEqual([REPLACED]);
      expect(echoOutcomes(rows, 2)).toEqual(['not now']);
      expect(approvalDecisions(rows).get('apr_old')).not.toBe(true);
    });
  }
});

// ---------------------------------------------------------------------------
// Added (PHP 0.26.10 audit): directed decision orders.
// ---------------------------------------------------------------------------

describe('added: directed decision orders', () => {
  it('(a) a DENIAL is stored, then a resume is refused session_locked, then an APPROVE arrives: refused, and the call never runs', async () => {
    const r = new Rig(8, { lock_wait: 0 });
    const [id] = await r.parked();
    await recordApproval(r.session(), id!, false, 'no');
    // Another worker holds the session lock: the resume is refused session_locked.
    let locked = '';
    await r.session().lock(async () => { locked = await r.runtime.send(r.session(), '').then(() => 'ran', code); }, 5, 0);
    expect(locked).toBe('session_locked');
    expect(await recordApproval(r.session(), id!, true).then(() => 'accepted', code)).toBe('approval_not_pending');
    for (let i = 0; i < 2; i += 1) await r.runtime.send(r.session(), '');
    expect(r.echoed).toEqual([]);
    expect(echoOutcomes(await r.rows(), 1)).toEqual(['no']);
  });

  it('(d) an APPROVAL is stored, the turn never runs, then a DENY arrives: DENIED (the recommended G-401 option), the call never runs, one outcome', async () => {
    const r = new Rig();
    const [id] = await r.parked();
    await recordApproval(r.session(), id!, true);
    expect(await recordApproval(r.session(), id!, false, 'changed my mind').then(() => 'accepted', code)).toBe('accepted');
    // Nothing overrides the denial now.
    expect(await recordApproval(r.session(), id!, true).then(() => 'accepted', code)).toBe('approval_not_pending');
    for (let i = 0; i < 2; i += 1) await r.runtime.send(r.session(), '');
    const rows = await r.rows();
    expect(r.echoed).toEqual([]);
    expect(echoOutcomes(rows, 1)).toEqual(['changed my mind']);
    expect(approvalDecisions(rows).get(id!)).toBe(false);
  });

  it('(d) once the call has STARTED (its result write then failed), a deny is refused: the approval was acted on', async () => {
    const r = new Rig();
    const [id] = await r.parked();
    await recordApproval(r.session(), id!, true);
    r.durable.failWhen = (message) => message.type === 'tool_result' && JSON.stringify(message).includes('ran:1');
    await expect(r.runtime.send(r.session(), '')).rejects.toThrow();
    expect(await recordApproval(r.session(), id!, false, 'too late').then(() => 'accepted', code)).toBe('approval_not_pending');
    await r.runtime.send(r.session(), '');
    expect(r.echoed).toEqual([1]);
    expect(approvalDecisions(await r.rows()).get(id!)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Added (PHP run17 Info 6a, all three): a parked structured turn's RAW view.
// TS returns no raw provider steps on a response (steps is a count), so the
// unchecked document can only leak through the response object itself.
// ---------------------------------------------------------------------------

describe('added: a parked structured response carries the unchecked document nowhere', () => {
  it('no field of the parked response but its transcript text holds the model\'s document (structured is null, no raw steps)', async () => {
    const r = new Rig();
    r.queue.push({ text: '{"title":12345,"role":"admin"}', structured: { title: 12345, role: 'admin' }, finishReason: 'tool_calls', toolCalls: [{ id: 'c1', name: 'echo', arguments: { n: 1 } }] });
    const parked = await r.runtime.sendStructured(r.session(), 'Go', { type: 'object', properties: { title: { type: 'string' } }, required: ['title'] });
    expect(parked.finishReason).toBe('awaiting_approval');
    expect(parked.structured).toBeNull();
    expect(typeof parked.steps).toBe('number');
    // `text` is the transcript, as documented; nothing else carries the document.
    const { text: _transcript, ...rest } = parked;
    expect(JSON.stringify(rest)).not.toContain('admin');
  });
});
