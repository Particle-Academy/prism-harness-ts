import { AsyncLocalStorage } from 'node:async_hooks';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  Session,
  Thread,
  ToolRegistry,
  recordApproval,
  recordDecisions,
  threadView,
  type Durability,
  type JsonObject,
  type LlmRequest,
  type LlmResponse,
} from '../src/index.js';
import { pendingApprovalIds, supersededApprovals, supersededRow, unresolvedApprovalIds } from '../src/thread-rows.js';
import { approvalDecisions } from '../src/plans/thread.js';

// G-387 round 14 (BRIEF-R14), at the public seams: send, streamDecision,
// recordApproval / recordDecisions, a host's thread().record(), and the
// runtime's own resume. Every concurrent interleaving is forced by a HOOK
// (a store write held open, a tool that acts while it runs), never by
// timing. No network; every store is in memory.

const network = vi.fn(() => { throw new Error('A test tried to use the network.'); });
beforeEach(() => { vi.stubGlobal('fetch', network); });
afterEach(() => { expect(network).not.toHaveBeenCalled(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });

const REPLACED = 'Tool call was not run because a later step replaced it';
const NEW_MESSAGE = 'A new message.';

/** A memory store whose writes of a thread payload can be held open or made to fail, by what they carry. */
class Store extends MemorySessionStore {
  readonly keys = new Set<string>();
  /** Runs inside the write (inside the thread lock) of a payload whose last message matches; once. */
  onWrite: { match: (message: JsonObject) => boolean; run: () => Promise<void> } | null = null;

  constructor(private readonly kind: Durability) {
    super();
  }

  override durability(): Durability {
    return this.kind;
  }

  override async put(key: string, payload: JsonObject, ttlSeconds: number | null = null): Promise<void> {
    this.keys.add(key);
    const messages = Array.isArray(payload.messages) ? payload.messages as JsonObject[] : [];
    const last = messages.at(-1)?.message as JsonObject | undefined;
    const hook = this.onWrite;
    if (hook !== null && last !== undefined && hook.match(last)) {
      this.onWrite = null;
      await hook.run();
    }
    return super.put(key, payload, ttlSeconds);
  }
}

/** Who is acting, for the hook signals below. */
const actor = new AsyncLocalStorage<string>();

/**
 * Signals from inside the harness, keyed by actor: `queued` when that actor
 * asks for a session lock someone else holds (it will wait), `read` when it
 * reads the thread. A hook waits for whichever comes first, so the forced
 * interleaving is the same on every machine and never a sleep.
 */
class Signals {
  #held = 0;
  readonly #waiters: { who: string; what: 'queued' | 'read' | 'done'; resolve: () => void }[] = [];
  readonly #seen = new Set<string>();

  install(): void {
    const signals = this;
    const lock = Session.prototype.lock;
    vi.spyOn(Session.prototype, 'lock').mockImplementation(async function (this: Session, callback: never, ttl?: number, wait?: number) {
      const who = actor.getStore();
      if (who !== undefined && signals.#held > 0) signals.fire(who, 'queued');
      return lock.call(this, (async (live: Session) => {
        signals.#held += 1;
        try {
          return await (callback as (session: Session) => Promise<unknown>)(live);
        } finally {
          signals.#held -= 1;
        }
      }) as never, ttl, wait);
    } as never);
    const messages = Thread.prototype.messages;
    vi.spyOn(Thread.prototype, 'messages').mockImplementation(async function (this: Thread) {
      const who = actor.getStore();
      const result = await messages.call(this);
      if (who !== undefined) signals.fire(who, 'read');
      return result;
    });
    const readerRows = Thread.prototype.readerRows;
    vi.spyOn(Thread.prototype, 'readerRows').mockImplementation(async function (this: Thread) {
      const who = actor.getStore();
      const result = await readerRows.call(this);
      if (who !== undefined) signals.fire(who, 'read');
      return result;
    });
  }

  fire(who: string, what: 'queued' | 'read' | 'done'): void {
    this.#seen.add(`${who}:${what}`);
    for (const waiter of this.#waiters.splice(0)) {
      if (this.#seen.has(`${waiter.who}:${waiter.what}`)) waiter.resolve(); else this.#waiters.push(waiter);
    }
  }

  /** Resolves with the first of `whats` that `who` reaches. */
  async first(who: string, whats: ('queued' | 'read' | 'done')[]): Promise<string> {
    return await Promise.race(whats.map((what) => new Promise<string>((resolve) => {
      if (this.#seen.has(`${who}:${what}`)) return resolve(what);
      this.#waiters.push({ who, what, resolve: () => resolve(what) });
    })));
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

  constructor() {
    const modes = new ModeRegistry({
      default: 'work',
      modes: { work: { system_prompt: 'Work.', tools: ['echo', 'free'], max_steps: 8, requires_approval: ['echo'] } },
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
      modes, tools,
      client: async (request) => {
        this.requests.push(request);
        return this.queue.shift() ?? { text: 'Done.', finishReason: 'stop' };
      },
    });
    const durable = this.durable, ephemeral = this.ephemeral;
    this.harness = new PrismHarness({ drivers: { ephemeral: () => ephemeral, durable: () => durable }, stores: { ephemeral: 'ephemeral', durable: 'durable' }, runtime: this.runtime });
  }

  /** A fresh Session object on the same session, as another worker builds one. */
  session(scope = 'chat'): Session {
    return this.harness.for({ type: 'User', id: 1 }).session(scope);
  }

  calls(calls: [string, JsonObject, string][]): void {
    this.queue.push({ text: '', finishReason: 'tool_calls', toolCalls: calls.map(([name, args, id]) => ({ id, name, arguments: args })) });
  }

  /** The thread as the harness's readers take it (round 15: only the harness's own rows carry its marks). */
  async rows(scope = 'chat'): Promise<JsonObject[]> {
    return await this.session(scope).thread().readerRows();
  }

  /** Parked on one gated echo (c1); its approval id. */
  async parked(scope = 'chat', calls: [string, JsonObject, string][] = [['echo', { n: 1 }, 'c1']]): Promise<string[]> {
    this.calls(calls);
    const response = await this.runtime.send(this.session(scope), 'Go');
    expect(response.finishReason).toBe('awaiting_approval');
    return response.pendingApprovals.map((pending) => pending.id);
  }
}

/** Every outcome recorded for the echo call with token `n` in the step's segment: each result entry once (an exact copy is the same one). */
function echoOutcomes(rows: JsonObject[], n: number): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const row of rows) {
    if (row.type !== 'tool_result') continue;
    for (const entry of (row.tool_results ?? []) as JsonObject[]) {
      if (entry.tool_name !== 'echo' || (entry.args as JsonObject).n !== n) continue;
      const copy = JSON.stringify([entry.tool_call_id, entry.args, entry.result]);
      if (seen.has(copy)) continue;
      seen.add(copy);
      out.push(String(entry.result));
    }
  }
  return out;
}

const code = (error: { code?: string }): string => error.code ?? 'threw';

// ---------------------------------------------------------------------------
// Item 1 (G-404; TS run17 F2): a superseding write against the decision and
// resolution paths. One write for the replaced denial and the superseding
// row; the decision check inside the thread lock; the resolver's in-flight
// claim; every pending reader honouring the superseded mark.
// ---------------------------------------------------------------------------

describe('item 1: a new message and a decision (W1, W2)', () => {
  it('W1: the new message\'s write fails: NOTHING of it is stored (no denial without its row), the readers agree, and the decision is acted on exactly once', async () => {
    const r = new Rig();
    const [id] = await r.parked();
    r.durable.onWrite = { match: (message) => message.type === 'user' && message.content === NEW_MESSAGE, run: async () => { throw new Error('STORE-SECRET write failed'); } };
    await expect(r.runtime.send(r.session(), NEW_MESSAGE)).rejects.toThrow();

    const rows = await r.rows();
    // No replaced denial was recorded without the row that replaced the step.
    expect(rows.some((row) => Object.hasOwn(row, 'harness_superseded'))).toBe(false);
    expect(rows.some((row) => row.type === 'user' && row.content === NEW_MESSAGE)).toBe(false);
    // The two decision readers agree: the step is still answerable.
    expect(pendingApprovalIds(rows)).toEqual([id]);
    expect(unresolvedApprovalIds(rows)).toEqual([id]);

    await recordApproval(r.session(), id!, true);
    await r.runtime.send(r.session(), '');
    await r.runtime.send(r.session(), '');
    expect(r.echoed).toEqual([1]);
    expect(echoOutcomes(await r.rows(), 1)).toEqual(['ran:1']);
  });

  it('W1b: a thread holding a replaced denial with no superseding row after it (as a two-write supersede could leave one): every decision path refuses, nothing runs, one outcome', async () => {
    const r = new Rig();
    const [id] = await r.parked();
    // The exact row the replaced denial is, stored alone as the HARNESS's
    // own entry (round 15: a host row carrying the mark is read without it,
    // so only a stored harness entry can be this thread).
    const denial = supersededRow(await r.rows())!;
    expect(denial.harness_superseded).toEqual({ approval_ids: [id] });
    const thread = r.session().thread() as unknown as { store: MemorySessionStore; key: string };
    const payload = (await thread.store.get(thread.key))!;
    const stored = payload.messages as JsonObject[];
    await thread.store.put(thread.key, { ...payload, messages: [...stored, { position: stored.length + 1, message: denial as unknown as JsonObject, run_id: null, recorded_at: '2026-11-27T00:00:00.000Z', harness: true }] });

    const rows = await r.rows();
    expect(pendingApprovalIds(rows)).toEqual([]);
    expect(unresolvedApprovalIds(rows)).toEqual([]);
    await expect(recordApproval(r.session(), id!, true)).rejects.toMatchObject({ code: 'approval_not_pending' });
    await expect(recordDecisions(r.session(), [{ approvalId: id!, approved: true }])).rejects.toMatchObject({ code: 'approval_not_pending' });
    await expect(r.runtime.decide(r.session(), id!, true)).rejects.toMatchObject({ code: 'approval_not_pending' });
    const stream = r.runtime.streamDecision(r.session(), [{ approvalId: id!, approved: true }]);
    await expect((async () => { for await (const _step of stream) { /* drain */ } })()).rejects.toMatchObject({ code: 'approval_not_pending' });
    await r.runtime.send(r.session(), '');
    expect(r.echoed).toEqual([]);
    expect(echoOutcomes(await r.rows(), 1)).toEqual([REPLACED]);
    expect(approvalDecisions(await r.rows()).get(id!)).not.toBe(true);
  });

  it('W2: a decision made while the new message\'s write is in flight is refused approval_not_pending, and nothing of it is stored', async () => {
    const r = new Rig();
    const signals = new Signals();
    signals.install();
    const [id] = await r.parked();
    let decision: Promise<string> | null = null;
    r.durable.onWrite = {
      match: (message) => message.type === 'user' && message.content === NEW_MESSAGE,
      run: async () => {
        decision = actor.run('decider', () => recordApproval(r.session(), id!, true).then(() => 'accepted', code));
        // Held until the decision has either read the thread (no lock
        // serialises it with this write) or queued on the session lock
        // this write holds.
        await signals.first('decider', ['read', 'queued']);
      },
    };
    await r.runtime.send(r.session(), NEW_MESSAGE);
    expect(decision).not.toBeNull();
    expect(await decision!).toBe('approval_not_pending');

    const rows = await r.rows();
    expect(rows.flatMap((row) => (row.tool_approval_responses ?? []) as JsonObject[]).filter((entry) => entry.approval_id === id)).toEqual([]);
    expect(approvalDecisions(rows).get(id!)).not.toBe(true);
    expect(echoOutcomes(rows, 1)).toEqual([REPLACED]);
    await r.runtime.send(r.session(), '');
    expect(r.echoed).toEqual([]);
  });
});

describe('item 1: a supersede while an approved call is running (PR1, PR2)', () => {
  it('PR2 (round 15): a new message sent while the resolver runs an approved call does NOT wait for it: the call has exactly ONE outcome, its real result, and is never read as replaced', async () => {
    const r = new Rig();
    const signals = new Signals();
    signals.install();
    const [id] = await r.parked();
    await recordApproval(r.session(), id!, true);
    let other: Promise<string> | null = null;
    r.duringEcho = async () => {
      other = actor.run('B', () => r.runtime.send(r.session(), NEW_MESSAGE).then(() => 'ran', code));
      void other.then(() => signals.fire('B', 'done'));
      await signals.first('B', ['queued', 'done']);
    };
    await r.runtime.send(r.session(), '');
    expect(await other!).toBe('ran');

    const rows = await r.rows();
    expect(r.echoed).toEqual([1]);
    expect(echoOutcomes(rows, 1)).toEqual(['ran:1']);
    expect(supersededApprovals(rows).has(id!)).toBe(false);
    expect(approvalDecisions(rows).get(id!)).toBe(true);
    // The new message landed while the call ran (after its started row), and
    // the real result after it, naming its step.
    const result = rows.findIndex((row) => row.type === 'tool_result' && ((row.tool_results ?? []) as JsonObject[]).some((entry) => entry.result === 'ran:1'));
    const message = rows.findIndex((row) => row.type === 'user' && row.content === NEW_MESSAGE);
    const started = rows.findIndex((row) => Object.hasOwn(row, 'harness_started'));
    expect(started).toBeGreaterThan(-1);
    expect(started).toBeLessThan(message);
    expect(result).toBeGreaterThan(message);
    expect(rows[result]!.harness_answers_step).toEqual({ position: rows.findIndex((row) => row.type === 'assistant') + 1 });
  });

  for (const kind of ['user', 'assistant_text', 'assistant_tool_step'] as const) {
    it(`PR1 (${kind}, round 15): a host row recorded while the resolver runs an approved call is ACCEPTED, with no denial for the running call; the call has ONE outcome; the same row recorded after is accepted with no denial`, async () => {
      const r = new Rig();
      const [id] = await r.parked();
      await recordApproval(r.session(), id!, true);
      const row: JsonObject = kind === 'user' ? { type: 'user', content: 'Host row.' }
        : kind === 'assistant_text' ? { type: 'assistant', content: 'Host answer.', tool_calls: [], tool_approval_requests: [] }
          : { type: 'assistant', content: '', tool_calls: [{ id: 'later', name: 'free', arguments: {} }], tool_approval_requests: [] };
      let during = '';
      let count = 0;
      r.duringEcho = async () => {
        during = await r.session().thread().record([row]).then(() => 'recorded', code);
        count = (await r.rows()).length;
      };
      await r.runtime.send(r.session(), '');
      expect(during).toBe('recorded');

      const rows = await r.rows();
      expect(r.echoed).toEqual([1]);
      expect(echoOutcomes(rows, 1)).toEqual(['ran:1']);
      expect(supersededApprovals(rows).has(id!)).toBe(false);
      // No replaced denial ever names the running call (a host tool-call step
      // is itself replaced by the run's next row: its own call gets one).
      const deniedC1 = (all: JsonObject[]): boolean => all.some((candidate) => Object.hasOwn(candidate, 'harness_superseded')
        && ((candidate.tool_results ?? []) as JsonObject[]).some((entry) => entry.tool_call_id === 'c1'));
      expect(deniedC1(rows)).toBe(false);
      expect(count).toBeGreaterThan(0);
      // After the resolution, the host's row is an ordinary row.
      expect(await r.session().thread().record([row]).then(() => 'recorded', code)).toBe('recorded');
      expect(deniedC1(await r.rows())).toBe(false);
      expect(echoOutcomes(await r.rows(), 1)).toEqual(['ran:1']);
    });
  }

  it('PR1 on a stream (streamDecision holds the lock for its whole turn, round 15): a host row during the call is accepted the same way, with no denial for it', async () => {
    const r = new Rig();
    const [id] = await r.parked();
    let during = '';
    r.duringEcho = async () => { during = await r.session().thread().record([{ type: 'user', content: 'Host row.' }]).then(() => 'recorded', code); };
    for await (const _step of r.runtime.streamDecision(r.session(), [{ approvalId: id!, approved: true }])) { /* drain */ }
    expect(during).toBe('recorded');
    expect(echoOutcomes(await r.rows(), 1)).toEqual(['ran:1']);
    expect(supersededApprovals(await r.rows()).has(id!)).toBe(false);
  });
});

describe('item 1: new channels', () => {
  /** The thread payload key of a session: the key a thread write puts its messages under (its claim is `<key>:in-flight`). */
  async function threadKey(r: Rig): Promise<string> {
    for (const key of r.durable.keys) {
      if (key.endsWith(':in-flight')) continue;
      const payload = await r.durable.get(key);
      if (payload !== null && Array.isArray(payload.messages) && payload.messages.length > 0) return key;
    }
    throw new Error('no thread key');
  }

  it('round 15: a claim key an older build left (live, naming the step) refuses nothing: a host row supersedes the step as before', async () => {
    const r = new Rig();
    const [id] = await r.parked();
    const key = await threadKey(r);
    const position = (await r.session().thread().messages()).find((entry) => entry.message.type === 'assistant')!.position;
    await r.durable.put(`${key}:in-flight`, { v: 1, position, token: 'old', until: Date.now() + 60_000 });
    expect(await r.session().thread().record([{ type: 'user', content: 'Host row.' }]).then(() => 'recorded', code)).toBe('recorded');
    expect(supersededApprovals(await r.rows()).has(id!)).toBe(true);
    expect(r.echoed).toEqual([]);
  });

  it('round 15: a resolution whose result write fails leaves the started row: the call is never run again (it reads interrupted), and no claim or key is left behind', async () => {
    const r = new Rig();
    const [id] = await r.parked();
    await recordApproval(r.session(), id!, true);
    r.durable.onWrite = { match: (message) => message.type === 'tool_result' && JSON.stringify(message).includes('ran:1'), run: async () => { throw new Error('STORE-SECRET write failed'); } };
    await expect(r.runtime.send(r.session(), '')).rejects.toThrow();
    await r.runtime.send(r.session(), '');
    expect(r.echoed).toEqual([1]);
    const key = await threadKey(r);
    expect(await r.durable.get(`${key}:in-flight`)).toBeNull();
    expect((await r.rows()).filter((row) => Object.hasOwn(row, 'harness_started'))).toHaveLength(1);
  });

  it('a new message on a session with nothing open takes no session lock (a stream holding it does not block it)', async () => {
    const r = new Rig();
    r.queue.push({ text: 'Hi.', finishReason: 'stop' });
    await r.runtime.send(r.session(), 'Hello');
    const lock = vi.spyOn(Session.prototype, 'lock');
    await r.runtime.send(r.session(), 'Again');
    expect(lock).not.toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// Item 3, TS I-1: a copy of an earlier result in the same step answers
// nothing new (as 0.11.x reads it).
// ---------------------------------------------------------------------------

describe('item 3 (I-1): a copy of an earlier result answers nothing new', () => {
  it('an ungated call ran; a host records an exact copy of its result; the gated call sharing its id is still OPEN: a superseding row gives it its one replaced denial', async () => {
    const r = new Rig();
    const [id] = await r.parked('chat', [['free', { n: 1 }, 'c'], ['echo', { n: 2 }, 'c']]);
    expect(r.freed).toEqual([1]);
    const result = (await r.rows()).find((row) => row.type === 'tool_result')!;
    // The exact copy (id, tool, arguments and result) of the ungated call's result, in a row of its own.
    await r.session().thread().record([{ type: 'tool_result', tool_results: result.tool_results!, tool_approval_responses: [] }]);
    expect(unresolvedApprovalIds(await r.rows())).toEqual([id]);
    await r.session().thread().record([{ type: 'user', content: 'Host row.' }]);
    const rows = await r.rows();
    expect(supersededApprovals(rows).has(id!)).toBe(true);
    expect(echoOutcomes(rows, 2)).toEqual([REPLACED]);
    expect(r.echoed).toEqual([]);
  });

  it('channel: two IDENTICAL ungated calls in one step each keep their result (a copy is only one from an EARLIER row): a superseding row denies only the gated call', async () => {
    const r = new Rig();
    const [id] = await r.parked('chat', [['free', { n: 1 }, 'c'], ['free', { n: 1 }, 'c'], ['echo', { n: 2 }, 'd']]);
    expect(r.freed).toEqual([1, 1]);
    await r.session().thread().record([{ type: 'user', content: 'Host row.' }]);
    const rows = await r.rows();
    const denial = rows.find((row) => Object.hasOwn(row, 'harness_superseded'))!;
    expect(((denial.tool_results ?? []) as JsonObject[]).map((entry) => entry.tool_call_id)).toEqual(['d']);
    expect(supersededApprovals(rows).has(id!)).toBe(true);
  });

  it('distinct ids: the copy of an ungated result never answers the gated call by the id-only fallback', async () => {
    const r = new Rig();
    // Two calls under distinct ids; a copy of free's result recorded under the gated call's id-free fallback cannot exist, so the copy answers nothing.
    const [id] = await r.parked('chat', [['free', { n: 1 }, 'f'], ['echo', { n: 2 }, 'e']]);
    const result = (await r.rows()).find((row) => row.type === 'tool_result')!;
    await r.session().thread().record([{ type: 'tool_result', tool_results: result.tool_results!, tool_approval_responses: [] }]);
    await recordApproval(r.session(), id!, true);
    await r.runtime.send(r.session(), '');
    expect(r.echoed).toEqual([2]);
    expect(r.freed).toEqual([1]);
    expect(echoOutcomes(await r.rows(), 2)).toEqual(['ran:2']);
  });
});

// ---------------------------------------------------------------------------
// Item 5: the replay emits each result EXACTLY ONCE, in its own segment.
// ---------------------------------------------------------------------------

/** How often each result (id, tool, arguments) occurs in the tool_result rows a provider request carries. */
function emitted(request: LlmRequest): Map<string, number> {
  const counts = new Map<string, number>();
  for (const row of request.messages as JsonObject[]) {
    if (row.type !== 'tool_result') continue;
    for (const entry of (row.tool_results ?? []) as JsonObject[]) {
      const key = JSON.stringify([entry.tool_call_id, entry.tool_name, entry.args]);
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
  }
  return counts;
}

describe('item 5: a result is replayed once', () => {
  it('a new message on a parked step (an ungated call ran, the gated one decided): the next request carries each result once', async () => {
    const r = new Rig();
    const [id] = await r.parked('chat', [['free', { n: 1 }, 'f'], ['echo', { n: 2 }, 'e']]);
    await recordApproval(r.session(), id!, true);
    await r.runtime.send(r.session(), NEW_MESSAGE);
    for (const count of emitted(r.requests.at(-1)!).values()) expect(count).toBe(1);
    expect(r.echoed).toEqual([]);
  });

  it('a row of another kind between the decision and the resume (G-399): the resolution row\'s carried copy is not replayed a second time', async () => {
    const r = new Rig();
    const [id] = await r.parked('chat', [['free', { n: 1 }, 'f'], ['echo', { n: 2 }, 'e']]);
    await recordApproval(r.session(), id!, true);
    await r.session().thread().record([{ type: 'system', content: 'A host note.' }]);
    await r.runtime.send(r.session(), '');
    expect(r.echoed).toEqual([2]);
    const counts = emitted(r.requests.at(-1)!);
    expect([...counts.values()]).toEqual([1, 1]);
  });

  it('a stored thread whose resolution row came after a new message (the 0.26.x order): each result is replayed once, in its own segment', () => {
    const step = { type: 'assistant', content: '', tool_calls: [
      { id: 'f', name: 'free', arguments: { n: 1 }, result_id: null, reasoning_id: null, reasoning_summary: null },
      { id: 'e', name: 'echo', arguments: { n: 2 }, result_id: null, reasoning_id: null, reasoning_summary: null },
    ], additional_content: {}, tool_approval_requests: [{ approval_id: 'a', tool_call_id: 'e' }] };
    const entry = (id: string, name: string, n: number, result: string): JsonObject => ({ tool_call_id: id, tool_name: name, args: { n }, result, tool_call_result_id: null, artifacts: [] });
    const view = threadView([
      { type: 'user', content: 'Go' },
      step,
      { type: 'tool_result', tool_results: [entry('f', 'free', 1, 'free:1')], tool_approval_responses: [] },
      { type: 'user', content: NEW_MESSAGE },
      { type: 'tool_result', tool_results: [entry('f', 'free', 1, 'free:1'), entry('e', 'echo', 2, 'denied')], tool_approval_responses: [{ approval_id: 'a', approved: false, reason: null }] },
    ]);
    const counts = emitted({ messages: view } as unknown as LlmRequest);
    expect([...counts.entries()].sort()).toEqual([[JSON.stringify(['e', 'echo', { n: 2 }]), 1], [JSON.stringify(['f', 'free', { n: 1 }]), 1]]);
    // Each in its own segment (after the step, no later step), and the copy
    // replaced in place: the later row shows only what is new in it.
    expect(view.map((row) => row.type)).toEqual(['user', 'assistant', 'tool_result', 'user', 'tool_result']);
    expect(((view[4]!.tool_results ?? []) as JsonObject[]).map((entry) => entry.tool_call_id)).toEqual(['e']);
  });
});

// ---------------------------------------------------------------------------
// Item 6 (G-405): a structured turn parked for approval carries NO document,
// with an ANTHROPIC-shaped provider fake that keeps the model's text (here a
// bad-typed JSON document) on the tool-call step.
// ---------------------------------------------------------------------------

describe('item 6 (G-405): a parked structured turn has no document', () => {
  const SCHEMA: JsonObject = { type: 'object', properties: { ok: { type: 'boolean' } }, required: ['ok'] };

  for (const shape of ['text and a parsed document beside the calls', 'text only beside the calls'] as const) {
    it(`Anthropic-shaped step (${shape}): structured is null, and the bad document is never returned nor stored as the answer`, async () => {
      const r = new Rig();
      r.queue.push({
        text: '{"ok":"not a boolean"}',
        ...(shape === 'text and a parsed document beside the calls' ? { structured: { ok: 'not a boolean' } } : {}),
        finishReason: 'tool_calls',
        toolCalls: [{ id: 'c1', name: 'echo', arguments: { n: 1 } }],
      });
      const parked = await r.runtime.sendStructured(r.session(), 'Go', SCHEMA);
      expect(parked.finishReason).toBe('awaiting_approval');
      expect(parked.structured).toBeNull();
      expect(parked.pendingApprovals).toHaveLength(1);
      const step = (await r.rows()).find((row) => row.type === 'assistant')!;
      expect(Object.hasOwn((step.additional_content ?? {}) as JsonObject, 'structured')).toBe(false);
      // The resume returns a document only once it is validated.
      await recordApproval(r.session(), parked.pendingApprovals[0]!.id, true);
      r.queue.push({ text: '{"ok":true}', finishReason: 'stop', structured: { ok: true } });
      const done = await r.runtime.sendStructured(r.session(), '', SCHEMA);
      expect(done.structured).toEqual({ ok: true });
      expect(r.echoed).toEqual([1]);
    });
  }

  it('with an ungated call beside the gated one (tools named): a parked step returns no document either', async () => {
    const r = new Rig();
    r.queue.push({ text: '{"ok":"not a boolean"}', structured: { ok: 'not a boolean' }, finishReason: 'tool_calls', toolCalls: [{ id: 'c1', name: 'echo', arguments: { n: 1 } }, { id: 'c2', name: 'free', arguments: { n: 2 } }] });
    const parked = await r.runtime.sendStructured(r.session(), 'Go', SCHEMA, ['echo', 'free']);
    expect(parked.finishReason).toBe('awaiting_approval');
    expect(parked.structured).toBeNull();
    expect(r.freed).toEqual([2]);
  });
});

// ---------------------------------------------------------------------------
// Item 7 (G-406): the runtime's in-run marker covers exactly its own record
// call. A host write made INSIDE that window (a store hook or listener that
// re-enters, naming the run's id) is OUTSIDE the run.
// ---------------------------------------------------------------------------

describe('item 7 (G-406): a write made inside the runtime\'s own record window is outside the run', () => {
  it('a store hook on the parked step\'s write records a user row naming the run id: it supersedes the step (one denial), it is never the run\'s own', async () => {
    const r = new Rig();
    let reentry: Promise<unknown> | null = null;
    let runId = '';
    const original = Thread.prototype.record;
    // Learn the run id the way a listener would: from the write in flight.
    vi.spyOn(Thread.prototype, 'record').mockImplementation(async function (this: Thread, ...args: Parameters<Thread['record']>) {
      if (runId === '' && typeof args[1] === 'string') runId = args[1];
      return original.apply(this, args);
    });
    r.durable.onWrite = {
      match: (message) => message.type === 'assistant' && Array.isArray(message.tool_approval_requests) && message.tool_approval_requests.length > 0,
      run: async () => {
        // Scheduled from inside the runtime's write; it lands once the thread lock is free.
        reentry = Promise.resolve().then(() => r.session().thread().record([{ type: 'user', content: 'Re-entered.' }], runId));
      },
    };
    r.calls([['echo', { n: 1 }, 'c1']]);
    const parked = await r.runtime.send(r.session(), 'Go');
    expect(reentry).not.toBeNull();
    await reentry;
    const rows = await r.rows();
    const id = parked.pendingApprovals[0]!.id;
    expect(rows.some((row) => row.type === 'user' && row.content === 'Re-entered.')).toBe(true);
    expect(supersededApprovals(rows).has(id)).toBe(true);
    await expect(recordApproval(r.session(), id, true)).rejects.toMatchObject({ code: 'approval_not_pending' });
  });

  it("control: the runtime's own rows inside its run still never supersede its step", async () => {
    const r = new Rig();
    const [id] = await r.parked();
    expect(pendingApprovalIds(await r.rows())).toEqual([id]);
    expect((await r.rows()).some((row) => Object.hasOwn(row, 'harness_superseded'))).toBe(false);
  });
});
