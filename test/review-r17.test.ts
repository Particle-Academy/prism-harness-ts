import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  HarnessEvents,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  Session,
  ToolRegistry,
  recordApproval,
  type Durability,
  type JsonObject,
  type LlmRequest,
  type LlmResponse,
  type LlmToolCall,
} from '../src/index.js';
import { pausedRunState, pendingApprovalIds, replayView, unresolvedApprovalIds } from '../src/thread-rows.js';

// G-387 round 17 (BRIEF-R17), at the public seams: send, recordApproval, a
// host's thread().record(), raw stored rows, a tool that acts on its own
// session while it runs, a listener, and a store write that fails. Every
// interleaving is forced by a HOOK, never by timing. No network; every store
// is in memory. Every check of the replay reads the request ACTUALLY SENT.

const network = vi.fn(() => { throw new Error('A test tried to use the network.'); });
beforeEach(() => { vi.stubGlobal('fetch', network); });
afterEach(() => { expect(network).not.toHaveBeenCalled(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });

const REPLACED = 'Tool call was not run because a later step replaced it';
const INTERRUPTED = 'Tool call was interrupted, it may have run';

/** A memory store whose writes can be made to fail, by the last message they carry. */
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
  readonly events = new HarnessEvents();
  readonly runtime: AgentRuntime;
  readonly harness: PrismHarness;
  readonly echoed: number[] = [];
  readonly freed: number[] = [];
  /** Runs once inside the named tool, before it returns. */
  during: Record<'echo' | 'free', (() => Promise<void>) | null> = { echo: null, free: null };
  /** What a hook produced (a write's outcome, a nested send's finish or code). */
  seen: string[] = [];

  constructor(maxSteps = 8) {
    const modes = new ModeRegistry({
      default: 'work',
      modes: { work: { system_prompt: 'Work.', tools: ['echo', 'free', 'other'], max_steps: maxSteps, requires_approval: ['echo', 'other'] } },
    });
    const hooked = async (tool: 'echo' | 'free'): Promise<void> => {
      const hook = this.during[tool];
      this.during[tool] = null;
      if (hook !== null) await hook();
    };
    const tools = new ToolRegistry()
      .register({ name: 'echo', handle: async (args: JsonObject) => { this.echoed.push(Number(args.n)); await hooked('echo'); return `ran:${String(args.n)}`; } })
      .register({ name: 'free', handle: async (args: JsonObject) => { this.freed.push(Number(args.n)); await hooked('free'); return `free:${String(args.n)}`; } })
      .register({ name: 'other', handle: () => 'other' });
    this.runtime = new AgentRuntime({
      modes, tools, events: this.events,
      client: async (request) => {
        this.requests.push(request);
        return this.queue.shift() ?? { text: 'Done.', finishReason: 'stop' };
      },
    });
    const durable = this.durable, ephemeral = this.ephemeral;
    this.harness = new PrismHarness({ drivers: { ephemeral: () => ephemeral, durable: () => durable }, stores: { ephemeral: 'ephemeral', durable: 'durable' }, runtime: this.runtime });
  }

  session(): Session {
    return this.harness.for({ type: 'User', id: 1 }).session('chat') as unknown as Session;
  }

  threadKey(): string {
    return (this.session().thread() as unknown as { key: string }).key;
  }

  calls(calls: [string, JsonObject, string, Partial<LlmToolCall>?][]): LlmResponse {
    return { text: '', finishReason: 'tool_calls', toolCalls: calls.map(([name, args, id, extra]) => ({ id, name, arguments: args, ...(extra ?? {}) })) };
  }

  async rows(): Promise<JsonObject[]> {
    return await this.session().thread().readerRows();
  }

  async lastStepRunId(): Promise<string> {
    return (await this.session().thread().messages()).filter((entry) => entry.message.type === 'assistant').at(-1)!.runId!;
  }
}

const code = (error: { code?: string }): string => error.code ?? 'threw';

/**
 * THE INVARIANT (BRIEF-R17 item 1): every tool call of every step a request
 * carries is answered EXACTLY ONCE, in that step's OWN result run: the
 * tool_result rows directly after the step, before ANY row of another kind
 * (a user row, an assistant text row, a system row). Anthropic and OpenAI
 * both reject anything else.
 */
function unansweredInOwnRun(messages: readonly JsonObject[]): string[] {
  const problems: string[] = [];
  for (const [index, row] of messages.entries()) {
    if (row.type !== 'assistant' || !Array.isArray(row.tool_calls) || row.tool_calls.length === 0) continue;
    const answered: string[] = [];
    for (let at = index + 1; at < messages.length && messages[at]!.type === 'tool_result'; at += 1) {
      for (const entry of (messages[at]!.tool_results ?? []) as JsonObject[]) answered.push(String(entry.tool_call_id));
    }
    const ids = (row.tool_calls as JsonObject[]).map((call) => String(call.id));
    for (const id of new Set(ids)) {
      const want = ids.filter((candidate) => candidate === id).length;
      const got = answered.filter((candidate) => candidate === id).length;
      if (got !== want) problems.push(`${index}:${id}:${got}/${want}`);
    }
  }
  return problems;
}

/** Every SENT request, and the replay of the stored rows, satisfy the invariant. */
async function everyRequestValid(r: Rig): Promise<void> {
  for (const [index, request] of r.requests.entries()) expect({ request: index, problems: unansweredInOwnRun(request.messages) }).toEqual({ request: index, problems: [] });
  // The stored rows as the next request would replay them (a step still
  // parked for a person is answered by its resume or by what replaces it).
  const rows = await r.rows();
  if (pendingApprovalIds(rows).length === 0) expect(unansweredInOwnRun(replayView(rows))).toEqual([]);
}

/** The outcomes recorded for the call `tool` / token `n`, a copy carried again counted once. */
function outcomes(rows: JsonObject[], n: number, tool: string): string[] {
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

// ---------------------------------------------------------------------------
// Item 1 (TS run20 F1, the BLOCK): a result is placed directly after its OWN
// step, past any row of another kind written while the call ran.
// ---------------------------------------------------------------------------

type Action = 'user' | 'text' | 'user_text' | 'nested_text' | 'nested_park';
const ACTIONS: Action[] = ['user', 'text', 'user_text', 'nested_text', 'nested_park'];

/** What the tool does while it runs. */
function act(r: Rig, action: Action): () => Promise<void> {
  return async () => {
    const thread = r.session().thread();
    if (action === 'user') r.seen.push(await thread.record([{ type: 'user', content: 'Host row.' }]).then(() => 'recorded', code));
    if (action === 'text') r.seen.push(await thread.record([{ type: 'assistant', content: 'Host note.', tool_calls: [], tool_approval_requests: [] }]).then(() => 'recorded', code));
    if (action === 'user_text') r.seen.push(await thread.record([{ type: 'user', content: 'Host row.' }, { type: 'assistant', content: 'Host answer.', tool_calls: [], tool_approval_requests: [] }]).then(() => 'recorded', code));
    if (action === 'nested_text') {
      r.queue.unshift({ text: 'Inner answer.', finishReason: 'stop' });
      r.seen.push(await r.runtime.send(r.session(), 'Inner.').then((response) => response.finishReason, code));
    }
    if (action === 'nested_park') {
      r.queue.unshift(r.calls([['echo', { n: 9 }, 'c9']]));
      r.seen.push(await r.runtime.send(r.session(), 'Inner.').then((response) => response.finishReason, code));
    }
  };
}

describe('item 1: each result is placed directly after its own step', () => {
  for (const action of ACTIONS) {
    it(`approved call, ${action} while it runs: every request sent answers every call in its own result run`, async () => {
      const r = new Rig();
      r.queue.push(r.calls([['echo', { n: 1 }, 'c1']]));
      const parked = await r.runtime.send(r.session(), 'Go');
      await recordApproval(r.session(), parked.pendingApprovals[0]!.id, true);
      r.during.echo = act(r, action);
      r.queue.push({ text: 'Resumed.', finishReason: 'stop' });
      const resumed = await r.runtime.send(r.session(), '').then((response) => response.finishReason, code);
      expect(resumed).toBe(action === 'nested_park' ? 'parked_elsewhere' : 'stop');
      r.queue.push({ text: 'Later answer.', finishReason: 'stop' });
      await r.runtime.send(r.session(), 'Later.');
      expect(r.echoed).toEqual([1]);
      expect(outcomes(await r.rows(), 1, 'echo')).toEqual(['ran:1']);
      await everyRequestValid(r);
    });

    it(`ungated call, ${action} while it runs: every request sent answers every call in its own result run`, async () => {
      const r = new Rig();
      r.queue.push(r.calls([['free', { n: 1 }, 'f1']]));
      r.during.free = act(r, action);
      r.queue.push({ text: 'Outer.', finishReason: 'stop' });
      const first = await r.runtime.send(r.session(), 'Go').then((response) => response.finishReason, code);
      expect(first).toBe(action === 'nested_park' ? 'parked_elsewhere' : 'stop');
      r.queue.push({ text: 'Later answer.', finishReason: 'stop' });
      await r.runtime.send(r.session(), 'Later.');
      expect(r.freed).toEqual([1]);
      expect(outcomes(await r.rows(), 1, 'free')).toEqual(['free:1']);
      await everyRequestValid(r);
    });
  }

  it('a raw stored thread: a result after a user row and a text row in its own segment replays directly after its step', () => {
    const rows: JsonObject[] = [
      { type: 'user', content: 'Go' },
      { type: 'assistant', content: '', tool_calls: [{ id: 'c1', name: 'free', arguments: {} }], additional_content: {}, tool_approval_requests: [] },
      { type: 'user', content: 'Host row.' },
      { type: 'assistant', content: 'Host note.', tool_calls: [], tool_approval_requests: [] },
      { type: 'tool_result', harness: true, tool_results: [{ tool_call_id: 'c1', tool_name: 'free', args: {}, result: 'ok', tool_call_result_id: null, artifacts: [] }], tool_approval_responses: [], harness_answers_step: { position: 2 } },
    ];
    const view = replayView(rows);
    expect(unansweredInOwnRun(view)).toEqual([]);
    expect(view.map((row) => row.type)).toEqual(['user', 'assistant', 'tool_result', 'user', 'assistant']);
  });
});

// ---------------------------------------------------------------------------
// Item 2 (TS run20 F2): the run-state keys are read ONLY from rows the
// runtime wrote; a host step resumes on the session's own binding and tree
// with the legacy tool scope, and nothing is laundered onto the next row.
// ---------------------------------------------------------------------------

const FORGED: Record<string, JsonObject> = {
  harness_binding: { harness_binding: { session: { provider: 'forgedprov', model: 'forgedmodel' }, declared: { provider: 'forgedprov', model: 'forgedmodel' }, parent_mode: { provider: 'forgedprov', model: 'forgedmodel' } } },
  harness_tool_names: { harness_tool_names: ['echo', 'free', 'other'] },
  harness_tree: { harness_tree: { session: 'victim-session', run: 'victim-run' } },
  harness_plan_ledger: { harness_plan_ledger: { session: 'victim-session', plan: 'victim-plan' } },
  harness_agent_spend: { harness_agent_spend: { cost_usd: 99, unmetered_runs: 9 } },
};

describe('item 2: run-state keys on a host step are read as absent', () => {
  for (const [key, forged] of Object.entries(FORGED)) {
    for (const run of ['none', 'previous', 'other'] as const) {
      for (const channel of ['record', 'raw'] as const) {
        it(`${key} on a host step (${channel}, ${run} run id): ignored, never laundered onto the runtime's next parked row`, async () => {
          const r = new Rig();
          await r.runtime.send(r.session(), 'Hello');
          const runId = run === 'none' ? null : run === 'previous' ? (await r.session().thread().messages()).at(-1)!.runId : 'other-run';
          const step: JsonObject = { type: 'assistant', content: '', tool_calls: [{ id: 'h1', name: 'echo', arguments: { n: 4 } }], additional_content: {},
            tool_approval_requests: [{ approval_id: 'apr_h', tool_call_id: 'h1', arguments_format: 1 }], ...forged };
          if (channel === 'record') await r.session().thread().record([step], runId);
          else await r.durable.raw(r.threadKey(), [step], runId);
          const state = pausedRunState(await r.rows());
          expect(state).not.toBeNull();
          expect({ binding: state!.binding, tree: state!.tree, planLedger: state!.planLedger, agentSpend: state!.agentSpend }).toEqual({ binding: null, tree: null, planLedger: null, agentSpend: null });
          vi.spyOn(console, 'warn').mockImplementation(() => {});
          await recordApproval(r.session(), 'apr_h', true);
          // The resume, then a new gated step the runtime parks on itself.
          r.queue.push(r.calls([['echo', { n: 5 }, 'c5']]));
          const resumed = await r.runtime.send(r.session(), '');
          expect(resumed.finishReason).toBe('awaiting_approval');
          expect(r.echoed).toEqual([4]);
          const after = r.requests.slice(1);
          for (const request of after) expect([request.provider, request.model]).not.toContain('forgedprov');
          // The legacy scope: only the approved call's tool is offered on.
          expect(after[0]!.tools.map((tool) => tool.name)).toEqual(['echo']);
          const own = (await r.session().thread().messages()).filter((entry) => entry.message.type === 'assistant').at(-1)!.message;
          const text = JSON.stringify(own);
          for (const forgedValue of ['forgedprov', 'victim-session', 'victim-run', 'victim-plan']) expect(text).not.toContain(forgedValue);
          expect((own.harness_agent_spend as JsonObject | undefined)?.cost_usd ?? 0).not.toBe(99);
        });
      }
    }
  }
});

// ---------------------------------------------------------------------------
// Item 2 (Py run21 F1, checked here): the nested-park stop never trusts a
// run id. A tool that records a gated step under the OUTER run's own id
// still stops the outer run.
// ---------------------------------------------------------------------------

describe('item 2: the nested-park stop never trusts a run id', () => {
  for (const run of ['none', 'outer', 'other'] as const) {
    it(`a tool records a new gated step under ${run} run id while an approved call runs: the outer run stops (parked_elsewhere), the step stays pending`, async () => {
      const r = new Rig();
      r.queue.push(r.calls([['echo', { n: 1 }, 'c1']]));
      const parked = await r.runtime.send(r.session(), 'Go');
      await recordApproval(r.session(), parked.pendingApprovals[0]!.id, true);
      r.during.echo = async () => {
        const runId = run === 'none' ? null : run === 'other' ? 'other-run' : (await r.session().thread().messages()).filter((entry) => entry.message.type === 'assistant').at(-1)!.runId;
        await r.session().thread().record([{ type: 'assistant', content: '', tool_calls: [{ id: 'h9', name: 'echo', arguments: { n: 9 } }], additional_content: {}, tool_approval_requests: [{ approval_id: 'apr_h9', tool_call_id: 'h9', arguments_format: 1 }] }], runId);
      };
      const before = r.requests.length;
      expect(await r.runtime.send(r.session(), '').then(() => 'ran', code)).toBe('parked_elsewhere');
      expect(r.requests.length).toBe(before);
      expect(pendingApprovalIds(await r.rows())).toEqual(['apr_h9']);
      await everyRequestValid(r);
    });
  }
});

// ---------------------------------------------------------------------------
// Item 3 (TS run20 F3, Py run21 F2): nested parks and ungated outcomes.
// ---------------------------------------------------------------------------

describe('item 3: nested parks and ungated outcomes', () => {
  it('NP1: an UNGATED call\'s own send parks a newer step in the step\'s first pass: never awaiting_approval on an ended approval; the inner step is not replaced', async () => {
    const r = new Rig();
    r.queue.push(r.calls([['free', { n: 1 }, 'f1'], ['echo', { n: 2 }, 'c2']]));
    let inner: string[] = [];
    r.during.free = async () => {
      r.queue.unshift(r.calls([['echo', { n: 9 }, 'c9']]));
      const response = await r.runtime.send(r.session(), 'Inner.');
      inner = response.pendingApprovals.map((pending) => pending.id);
    };
    const outer = await r.runtime.send(r.session(), 'Go').then((response) => response, (error: { code?: string }) => error);
    const rows = await r.rows();
    const pending = pendingApprovalIds(rows);
    expect(inner).toHaveLength(1);
    expect(pending).toEqual(inner);
    if ('finishReason' in outer && typeof outer.finishReason === 'string') {
      for (const approval of (outer as unknown as { pendingApprovals: readonly { id: string }[] }).pendingApprovals) expect(pending).toContain(approval.id);
    } else {
      expect(code(outer as { code?: string })).toBe('parked_elsewhere');
    }
    expect(r.freed).toEqual([1]);
    expect(outcomes(rows, 1, 'free')).toEqual(['free:1']);
    expect(outcomes(rows, 2, 'echo')).toEqual([REPLACED]);
    // The inner step is approvable and runs once.
    await recordApproval(r.session(), inner[0]!, true);
    await r.runtime.send(r.session(), '');
    expect(r.echoed).toEqual([9]);
    await everyRequestValid(r);
  });

  for (const row of ['user', 'text'] as const) {
    it(`P4: a listener records a ${row} row when the step asks for approval: the ungated call is NOT run after its replaced denial (one outcome), and the response lists no approval that is not pending`, async () => {
      const r = new Rig();
      const writes: Promise<unknown>[] = [];
      r.events.listen((event) => {
        if (event.type !== 'tool_approval_asked' || writes.length > 0) return;
        writes.push(r.session().thread().record([row === 'user' ? { type: 'user', content: 'Notified.' } : { type: 'assistant', content: 'Waiting for you.', tool_calls: [], tool_approval_requests: [] }]));
      });
      r.queue.push(r.calls([['free', { n: 1 }, 'u1'], ['echo', { n: 2 }, 'c1']]));
      const response = await r.runtime.send(r.session(), 'Go');
      await Promise.all(writes);
      const rows = await r.rows();
      const free = outcomes(rows, 1, 'free');
      expect(free).toHaveLength(1);
      expect(r.freed.length === 0 ? [REPLACED] : ['free:1']).toEqual(free);
      expect(outcomes(rows, 2, 'echo')).toHaveLength(1);
      const pending = pendingApprovalIds(rows);
      for (const approval of response.pendingApprovals) expect(pending).toContain(approval.id);
      await everyRequestValid(r);
    });
  }

  it('UF: the ungated started-row write fails beside a gated call: the ungated call never runs, gets ONE "not run" outcome, and every later request answers it', async () => {
    const r = new Rig();
    r.queue.push(r.calls([['free', { n: 7 }, 'f1'], ['echo', { n: 1 }, 'c1']]));
    r.durable.failWhen = (message) => message.type === 'tool_result' && message.harness_started !== undefined;
    await expect(r.runtime.send(r.session(), 'Go')).rejects.toThrow();
    expect(r.freed).toEqual([]);
    const pending = pendingApprovalIds(await r.rows());
    expect(pending).toHaveLength(1);
    await recordApproval(r.session(), pending[0]!, true);
    r.queue.push({ text: 'Resumed.', finishReason: 'stop' });
    await r.runtime.send(r.session(), '');
    expect(r.echoed).toEqual([1]);
    expect(r.freed).toEqual([]);
    const rows = await r.rows();
    const f1 = outcomes(rows, 7, 'free');
    expect(f1).toHaveLength(1);
    expect(f1[0]).toMatch(/^Not run/);
    expect(unresolvedApprovalIds(rows)).toEqual([]);
    r.queue.push({ text: 'Later.', finishReason: 'stop' });
    await r.runtime.send(r.session(), 'Next.');
    expect(outcomes(await r.rows(), 7, 'free')).toEqual(f1);
    await everyRequestValid(r);
  });
});

// ---------------------------------------------------------------------------
// Item 4 (PHP run19 #4, checked here): the replay's synthesised answer
// carries the call's OWN provider ids, never null, for every provider shape.
// ---------------------------------------------------------------------------

describe('item 4: a synthesised answer keeps the call\'s own provider ids', () => {
  const SHAPES: Record<string, { id: string; resultId?: string }> = {
    openai_chat: { id: 'call_abc' },
    openai_responses: { id: 'fc_abc', resultId: 'call_abc' },
    anthropic: { id: 'toolu_abc' },
  };
  for (const [shape, ids] of Object.entries(SHAPES)) {
    it(`${shape}: an ungated call whose result write failed replays its interrupted answer under id ${ids.id}${ids.resultId === undefined ? '' : ` / ${ids.resultId}`}`, async () => {
      const r = new Rig();
      r.queue.push(r.calls([['free', { n: 3 }, ids.id, ids.resultId === undefined ? {} : { resultId: ids.resultId }]]));
      r.durable.failWhen = (message) => message.type === 'tool_result' && JSON.stringify(message).includes('free:3');
      await expect(r.runtime.send(r.session(), 'Go')).rejects.toThrow();
      r.queue.push({ text: 'Next answer.', finishReason: 'stop' });
      await r.runtime.send(r.session(), 'Next.');
      const sent = r.requests.at(-1)!.messages;
      const step = sent.find((row) => row.type === 'assistant' && Array.isArray(row.tool_calls) && row.tool_calls.length > 0)!;
      const call = (step.tool_calls as JsonObject[])[0]!;
      const answer = sent.filter((row) => row.type === 'tool_result').flatMap((row) => (row.tool_results ?? []) as JsonObject[])[0]!;
      expect(answer.result).toBe(INTERRUPTED);
      expect(answer.tool_call_id).toBe(ids.id);
      expect(answer.tool_call_id).toBe(call.id);
      expect(answer.tool_call_result_id).toBe(ids.resultId ?? null);
      expect(answer.tool_call_result_id).toBe(call.result_id);
      if (ids.resultId !== undefined) expect(answer.tool_call_result_id).not.toBeNull();
      expect(answer.tool_call_id).not.toBeNull();
      expect(answer.tool_call_id).not.toBe('');
      await everyRequestValid(r);
    });
  }
});

// ---------------------------------------------------------------------------
// Item 5 (Py run21 I1, checked here): the readers' one-pass index never goes
// stale when a caller replaces a row of its own array in place.
// ---------------------------------------------------------------------------

describe('item 5: the row index is never stale', () => {
  it('a row replaced in place at the same length is read afresh', () => {
    const rows: JsonObject[] = [
      { type: 'user', content: 'Go' },
      { type: 'assistant', content: '', tool_calls: [{ id: 'c1', name: 'echo', arguments: {} }], additional_content: {}, tool_approval_requests: [{ approval_id: 'a1', tool_call_id: 'c1', arguments_format: 1 }] },
      { type: 'system', content: 'note' },
    ];
    expect(pendingApprovalIds(rows)).toEqual(['a1']);
    rows[2] = { type: 'tool_result', tool_results: [{ tool_call_id: 'c1', tool_name: 'echo', args: {}, result: 'host', tool_call_result_id: null, artifacts: [] }], tool_approval_responses: [] };
    expect(pendingApprovalIds(rows)).toEqual([]);
    expect(pendingApprovalIds([...rows])).toEqual([]);
  });
});
