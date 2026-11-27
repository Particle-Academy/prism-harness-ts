import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  HarnessEvents,
  MemorySessionStore,
  ModeRegistry,
  Plans,
  PrismHarness,
  RunBudget,
  RunContext,
  Session,
  SubagentRunner,
  ToolRegistry,
  recordApproval,
  threadView,
  type Durability,
  type HarnessEvent,
  type JsonObject,
  type LlmRequest,
  type LlmResponse,
  type ModeConfig,
} from '../src/index.js';
import { endedByStart } from '../src/thread-rows.js';

// G-387 round 12: the round-11 review findings, at the public seams (the
// subagent TOOL inside a parent turn, SubagentRunner.run, send,
// recordApproval, event listeners, a host HarnessEvents). No network; every
// store is in memory.

const network = vi.fn(() => { throw new Error('A test tried to use the network.'); });
beforeEach(() => { vi.stubGlobal('fetch', network); });
afterEach(() => { expect(network).not.toHaveBeenCalled(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });

class Store extends MemorySessionStore {
  readonly keys = new Set<string>();

  constructor(private readonly kind: Durability) {
    super();
  }

  override durability(): Durability {
    return this.kind;
  }

  override async put(key: string, payload: JsonObject, ttlSeconds: number | null = null): Promise<void> {
    this.keys.add(key);
    await super.put(key, payload, ttlSeconds);
  }
}

interface RigOptions {
  plans?: boolean;
  /** The events object the runtime emits on (a host may pass its own). */
  events?: HarnessEvents;
  /** What the model call does for the child's requests (default: the scripted queue). */
  childClient?: () => Promise<LlmResponse>;
  /** What echo does when it runs. */
  echo?: () => string;
  /** The parent's model call fails once a child has asked the model. */
  failParentAfterChild?: boolean;
}

class Rig {
  readonly queue: LlmResponse[] = [];
  readonly requests: LlmRequest[] = [];
  readonly durable = new Store('durable');
  readonly ephemeral = new Store('volatile');
  readonly events: HarnessEvents;
  readonly registry: ToolRegistry;
  readonly modes: ModeRegistry;
  readonly plans: Plans | null;
  readonly runtime: AgentRuntime;
  readonly harness: PrismHarness;
  readonly ran: Record<string, number> = {};
  #call = 0;

  constructor(options: RigOptions = {}) {
    this.events = options.events ?? new HarnessEvents();
    this.modes = new ModeRegistry({
      default: 'lead',
      modes: {
        lead: { system_prompt: 'Lead.', tools: ['echo', 'a'], max_steps: 8, requires_approval: ['echo'], subagents: { a: { mode: 'work', max_steps: 8 } } } as ModeConfig,
        work: { system_prompt: 'Work.', tools: ['echo', 'boom'], max_steps: 8, requires_approval: ['echo'], ...(options.plans === true ? { plans: true } : {}) } as ModeConfig,
      },
    });
    this.registry = new ToolRegistry()
      .register({ name: 'echo', handle: () => { this.ran.echo = (this.ran.echo ?? 0) + 1; return options.echo?.() ?? 'ok'; } })
      .register({ name: 'boom', handle: () => { throw Object.create(null); } });
    this.plans = options.plans === true ? new Plans({ modes: this.modes, config: { enabled: true }, events: this.events }) : null;
    this.runtime = new AgentRuntime({
      modes: this.modes, tools: this.registry, events: this.events,
      client: async (request) => {
        this.requests.push(request);
        if (request.systemPrompt.startsWith('Work.') && options.childClient !== undefined) return options.childClient();
        if (options.failParentAfterChild === true && request.systemPrompt.startsWith('Lead.') && this.childRequests() > 0) throw new Error('MODEL-SECRET');
        return this.queue.shift() ?? { text: 'Done.', finishReason: 'stop' };
      },
    });
    const durable = this.durable, ephemeral = this.ephemeral;
    this.harness = new PrismHarness({
      drivers: { ephemeral: () => ephemeral, durable: () => durable },
      stores: { ephemeral: 'ephemeral', durable: 'durable' },
      runtime: this.runtime,
      ...(this.plans === null ? {} : { plans: this.plans }),
    });
    this.registry.register(new SubagentRunner(this.harness, this.runtime)
      .tool(this.modes.resolve('lead').subagents.a!, this.session(), RunContext.root('host', new RunBudget(100)), 'host_run'));
  }

  session(scope = 'chat'): Session {
    return this.harness.for({ type: 'User', id: 1 }).session(scope);
  }

  child(): Session {
    return this.session('chat::sub::a');
  }

  calls(calls: [string, JsonObject][]): void {
    this.queue.push({ text: '', finishReason: 'tool_calls', toolCalls: calls.map(([name, args]) => ({ id: `call_${++this.#call}`, name, arguments: args })) });
  }

  text(text = 'Done.'): void {
    this.queue.push({ text, finishReason: 'stop' });
  }

  async run(task = 'Work.', context: RunContext = RunContext.root('root', new RunBudget(8))): Promise<{ outcome: string; text: string }> {
    return new SubagentRunner(this.harness, this.runtime).run(this.modes.resolve('lead').subagents.a!, this.session(), context, 'root', task);
  }

  childRequests(): number {
    return this.requests.filter((request) => request.systemPrompt.startsWith('Work.')).length;
  }

  /** The child parked on `calls`, then left too deep to answer: a runner start may end them. */
  async parkedUnanswerable(calls: [string, JsonObject][]): Promise<string[]> {
    await this.session().usingMode('lead');
    this.calls(calls);
    await this.run('First');
    const key = `${this.child().key()}:durable`;
    await this.durable.put(key, { ...((await this.durable.get(key)) ?? {}), child_depth: 1_000 });

    return approvalsIn(await childRows(this));
  }
}

/** A start through the runner directly, or through the subagent tool in a parent turn; the outcome word. */
async function start(r: Rig, via: 'runner' | 'tool', task: string, child: () => void = () => r.text('Child done.')): Promise<string> {
  if (via === 'runner') {
    child();
    return (await r.run(task)).outcome;
  }
  r.calls([['a', { task }]]);
  child();
  r.text('Parent done.');
  expect((await r.runtime.send(r.session(), 'Delegate', undefined, RunContext.root('root', new RunBudget(8)))).text).toBe('Parent done.');
  const rows = (await r.session().thread().messages()).map((entry) => entry.message);
  const framed = String(rows.filter((row) => row.type === 'tool_result').flatMap((row) => row.tool_results as JsonObject[]).pop()?.result ?? '');

  return /did not run: it (\w+)/.exec(framed)?.[1] ?? 'completed';
}

async function childRows(r: Rig): Promise<JsonObject[]> {
  return (await r.child().thread().messages()).map((entry) => entry.message);
}

function approvalsIn(rows: JsonObject[]): string[] {
  return rows.flatMap((row) => Array.isArray(row.tool_approval_requests) ? (row.tool_approval_requests as JsonObject[]).map((request) => request.approval_id as string) : []);
}

/** The tool_result row a start's task row was written with (the row directly before the user row named `task`). */
async function endedRow(r: Rig, task: string): Promise<JsonObject | undefined> {
  const rows = await childRows(r);
  const index = rows.findIndex((row) => row.type === 'user' && row.content === task);

  return index > 0 && rows[index - 1]!.type === 'tool_result' ? rows[index - 1] : undefined;
}

// ---------------------------------------------------------------------------
// Item 2: ONE ended-mark shape in all three languages, the PHP reference's:
// `harness_ended_by_start: {approval_ids: [...]}` on the result row, beside
// (never inside) the results.
// ---------------------------------------------------------------------------

describe("item 2: the ended mark is the reference's shape, on the row, and only that shape is read", () => {
  for (const via of ['runner', 'tool'] as const) {
    it(`a start through the ${via} ends two parked calls: the row carries harness_ended_by_start {approval_ids} in call order, after the decisions; no entry is marked`, async () => {
      const r = new Rig();
      const parked = await r.parkedUnanswerable([['echo', { n: 1 }], ['echo', { n: 2 }]]);
      expect(parked).toHaveLength(2);

      expect(await start(r, via, 'Start over')).toBe('completed');

      const row = (await endedRow(r, 'Start over'))!;
      expect(row).toBeDefined();
      expect(Object.keys(row)).toEqual(['type', 'tool_results', 'tool_approval_responses', 'harness_ended_by_start']);
      expect(row.harness_ended_by_start).toEqual({ approval_ids: parked });
      for (const entry of row.tool_results as JsonObject[]) {
        expect(Object.keys(entry)).toEqual(['tool_call_id', 'tool_name', 'args', 'result', 'tool_call_result_id', 'artifacts']);
      }
      expect([...endedByStart(await childRows(r))].sort()).toEqual([...parked].sort());
      expect(r.ran.echo).toBeUndefined();
      // The approvals read ended: never pending, never acted on.
      for (const id of parked) await expect(recordApproval(r.child(), id, true)).rejects.toMatchObject({ code: 'approval_not_pending' });
    });
  }

  it('the replay view never carries the mark to a provider', async () => {
    const r = new Rig();
    await r.parkedUnanswerable([['echo', { n: 1 }]]);
    expect(await start(r, 'runner', 'Start over')).toBe('completed');
    const replay = JSON.stringify(threadView(await childRows(r)));
    expect(replay).not.toContain('harness_ended_by_start');
    expect(replay).not.toContain('not_run');
    const sent = JSON.stringify(r.requests.at(-1)!.messages);
    expect(sent).not.toContain('harness_ended_by_start');
  });

  const asked: JsonObject = { type: 'assistant', content: '', tool_calls: [{ id: 'c1', name: 'echo', arguments: {} }, { id: 'c2', name: 'echo', arguments: {} }], tool_approval_requests: [{ approval_id: 'apr_1', tool_call_id: 'c1' }, { approval_id: 'apr_2', tool_call_id: 'c2' }] };
  const entry = (id: string, extra: JsonObject = {}): JsonObject => ({ tool_call_id: id, tool_name: 'echo', args: {}, result: 'Not run: the delegation this call was proposed under ended.', tool_call_result_id: null, artifacts: [], ...extra });
  const user: JsonObject = { type: 'user', content: 'New task' };

  it('a row the PHP reference wrote reads ended (row level)', () => {
    const php: JsonObject = { type: 'tool_result', tool_results: [entry('c1'), entry('c2')], tool_approval_responses: [], harness_ended_by_start: { approval_ids: ['apr_1', 'apr_2'] } };
    expect([...endedByStart([asked, php, user])]).toEqual(['apr_1', 'apr_2']);
    // Only the approvals it names.
    const one: JsonObject = { ...php, harness_ended_by_start: { approval_ids: ['apr_2'] } };
    expect([...endedByStart([asked, one, user])]).toEqual(['apr_2']);
  });

  it("the round-11 per-entry shape (never released) is not read: nothing ends", () => {
    const old: JsonObject = { type: 'tool_result', tool_results: [entry('c1', { not_run: 'delegation_ended', approval_id: 'apr_1' })], tool_approval_responses: [] };
    expect([...endedByStart([asked, old, user])]).toEqual([]);
  });

  it("a mark that is not the shape, or not in the request's own turn, ends nothing", () => {
    const ran: JsonObject = { type: 'tool_result', tool_results: [entry('c1', { result: 'ok' })], tool_approval_responses: [] };
    const marked: JsonObject = { type: 'tool_result', tool_results: [entry('c1')], tool_approval_responses: [], harness_ended_by_start: { approval_ids: ['apr_1'] } };
    // Round 13 (G-387 item 1): ANY result row in the request's turn that
    // carries the mark naming the approval ends it, not only the first row
    // carrying its call id (only the harness writes the mark, and it marks
    // only calls with no result of their own).
    expect([...endedByStart([asked, ran, marked, user])]).toEqual(['apr_1']);
    // A later assistant step: the call had no result in its own turn.
    const later: JsonObject = { type: 'assistant', content: 'Hm.', tool_calls: [], tool_approval_requests: [] };
    expect([...endedByStart([asked, later, marked, user])]).toEqual([]);
    // A mark naming an approval asked for only after it.
    const after: JsonObject = { type: 'assistant', content: '', tool_calls: [{ id: 'c1', name: 'echo', arguments: {} }], tool_approval_requests: [{ approval_id: 'apr_9', tool_call_id: 'c1' }] };
    const names9: JsonObject = { ...marked, harness_ended_by_start: { approval_ids: ['apr_9'] } };
    expect([...endedByStart([asked, names9, user, after])]).toEqual([]);
    // Not the shape: a bare true, a string list, a list of non-strings.
    for (const mark of [true, ['apr_1'], { approval_ids: 'apr_1' }, { approval_ids: [1] }, {}]) {
      expect([...endedByStart([asked, { ...marked, harness_ended_by_start: mark as JsonObject }, user])]).toEqual([]);
    }
  });

  it('a tool result or arguments that carry the key cannot set the mark (the tool controls only its result text)', async () => {
    const forged = JSON.stringify({ harness_ended_by_start: { approval_ids: ['X'] } });
    const r = new Rig({ echo: () => forged });
    const [id] = await r.parkedUnanswerable([['echo', { harness_ended_by_start: { approval_ids: ['X'] } }]]);
    // Answerable again, approved and run by a host resume.
    const key = `${r.child().key()}:durable`;
    await r.durable.put(key, { ...((await r.durable.get(key)) ?? {}), child_depth: 1 });
    await recordApproval(r.child(), id!, true);
    r.text('Next done.');
    // Round 13 (DECISION B): a new message supersedes the step, so the
      // approved call is run by a resume.
      await r.runtime.send(r.child(), '');
    expect(r.ran.echo).toBe(1);

    const rows = await childRows(r);
    expect(rows.some((row) => Object.hasOwn(row, 'harness_ended_by_start'))).toBe(false);
    expect([...endedByStart(rows)]).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// Item 7 (TS run15 F2, probe S8) and item 4 (Py I2): a listener's error can
// never break a run, and every run event is emitted inside the run's try.
// ---------------------------------------------------------------------------

/** Values a listener may throw that String() cannot convert, or that carry text. */
const UNCONVERTIBLE: [string, () => unknown][] = [
  ['an object with no prototype', () => Object.create(null)],
  ['an object whose toString throws', () => ({ toString() { throw new Error('LISTENER-SECRET'); } })],
  ['an Error whose message getter throws', () => Object.defineProperty(new Error('x'), 'message', { get() { throw new Error('LISTENER-SECRET'); } })],
];

describe('item 7: a throwing run.started listener never breaks a run', () => {
  for (const [name, make] of UNCONVERTIBLE) {
    for (const via of ['runner', 'tool'] as const) {
      it(`S8 through the ${via}: the child's run.started listener throws ${name}; the start completes and the next delegation is not busy`, async () => {
        const r = new Rig();
        await r.session().usingMode('lead');
        vi.spyOn(process, 'emitWarning').mockImplementation((() => {}) as typeof process.emitWarning);
        const seen: string[] = [];
        const stop = r.events.listen((event: HarnessEvent) => {
          if (event.sessionKey !== r.child().key()) return;
          seen.push(event.type);
          if (event.type === 'run.started') throw make();
        });

        expect(await start(r, via, 'First')).toBe('completed');
        stop();

        expect(seen).toContain('run.started');
        expect(seen).toContain('run.finished');
        expect((await r.child().run())?.status).toBe('completed');
        expect(await r.child().childStartUnconfirmed()).toBe(false);
        expect(await start(r, via, 'Second')).toBe('completed');
      });
    }

    it(`a root run's run.started listener throws ${name}: the run completes, never left running`, async () => {
      const r = new Rig();
      vi.spyOn(process, 'emitWarning').mockImplementation((() => {}) as typeof process.emitWarning);
      r.events.listen((event: HarnessEvent) => { if (event.type === 'run.started') throw make(); });
      r.text('Hi.');

      expect((await r.runtime.send(r.session(), 'Hello')).text).toBe('Hi.');
      expect((await r.session().run())?.status).toBe('completed');
    });
  }

  it("a listener's warning names the class only: no listener text reaches the host's log", async () => {
    const r = new Rig();
    const warnings: string[] = [];
    vi.spyOn(process, 'emitWarning').mockImplementation(((warning: string | Error) => { warnings.push(String(warning)); }) as typeof process.emitWarning);
    const Named = class ListenerFault extends Error {};
    r.events.listen((event: HarnessEvent) => { if (event.type === 'run.started') throw new Named('LISTENER-SECRET /srv/app/x.ts'); });
    r.events.listen((event: HarnessEvent) => { if (event.type === 'run.started') throw 'LISTENER-SECRET'; });
    r.text('Hi.');
    await r.runtime.send(r.session(), 'Hello');

    expect(warnings).toHaveLength(2);
    expect(warnings[0]).toContain('ListenerFault');
    for (const warning of warnings) expect(warning).not.toMatch(/SECRET|srv|[\\/]/);
  });
});

/** A host's own events object whose emit throws `value` on run.started (it may pass any HarnessEvents). */
function throwingEvents(value: () => unknown, seen: string[]): HarnessEvents {
  return new (class extends HarnessEvents {
    override emit(event: HarnessEvent): void {
      seen.push(event.type);
      if (event.type === 'run.started') throw value();
      super.emit(event);
    }
  })();
}

describe('item 4 (Py I2) and item 7: a throw from emitting run.started fails the run, releases, and propagates', () => {
  for (const [name, make] of [['an object with no prototype', () => Object.create(null)], ['a string', () => 'EMIT-SECRET'], ['an Error', () => new Error('EMIT-SECRET')]] as [string, () => unknown][]) {
    it(`a ROOT run whose run.started emit throws ${name}: run.failed is emitted, the run row is failed with the class name only, the value reaches the caller, and the next turn runs`, async () => {
      const seen: string[] = [];
      const value = make();
      const r = new Rig({ events: throwingEvents(() => value, seen) });
      r.text('Hi.');

      const caught = await r.runtime.send(r.session(), 'Hello').then(() => 'resolved', (error: unknown) => error);

      expect(caught).toBe(value);
      expect(seen).toEqual(['run.started', 'run.failed']);
      const run = (await r.session().run())!;
      expect(run.status).toBe('failed');
      expect(JSON.stringify(run)).not.toMatch(/SECRET/);
      expect(r.requests).toHaveLength(0);
      // Not left locked: a next turn reaches the same emit again (not session_locked).
      seen.length = 0;
      expect(await r.runtime.send(r.session(), 'Again').then(() => 'resolved', (error: unknown) => error)).toBe(value);
      expect(seen).toEqual(['run.started', 'run.failed']);
    });
  }

  for (const via of ['runner', 'tool'] as const) {
    it(`a runner start through the ${via} whose run.started emit throws: failed, run.failed emitted, the start undone, and the next delegation is not busy`, async () => {
      const seen: string[] = [];
      let armed = true;
      const events = new (class extends HarnessEvents {
        override emit(event: HarnessEvent): void {
          seen.push(`${event.type}:${event.sessionKey.includes('::sub::') ? 'child' : 'parent'}`);
          if (armed && event.type === 'run.started' && event.sessionKey.includes('::sub::')) throw Object.create(null);
          super.emit(event);
        }
      })();
      const r = new Rig({ events });
      await r.session().usingMode('lead');

      expect(await start(r, via, 'First', () => {})).toBe('failed');

      expect(seen).toContain('run.started:child');
      expect(seen).toContain('run.failed:child');
      expect((await r.child().run())?.status ?? 'none').not.toBe('running');
      expect(await r.child().childTurnLive(Math.floor(Date.now() / 1000), 300)).toBe(false);
      expect(r.childRequests()).toBe(0);
      armed = false;
      expect(await start(r, via, 'Second')).toBe('completed');
    });
  }
});

// ---------------------------------------------------------------------------
// Item 4 (Py I5): holds are released BEFORE run.failed is emitted.
// ---------------------------------------------------------------------------

describe('item 4 (Py I5): a run.failed listener never observes a hold that is about to be released', () => {
  /** Every hold in the stores as they are at the moment of the call (a memory store reads synchronously). */
  function holdsNow(r: Rig, runId: string): Promise<string[]> {
    const reads = [...r.durable.keys].map(async (key) => [key, await r.durable.get(key)] as const);
    return Promise.all(reads).then((entries) => entries.flatMap(([key, value]) => {
      const holds = key.includes(':run-tree:') ? (value as { holds?: JsonObject } | null)?.holds
        : ((value as { child_steps?: { holds?: JsonObject } } | null)?.child_steps?.holds);
      return holds !== undefined && holds !== null && Object.hasOwn(holds, runId) ? [key] : [];
    }));
  }

  for (const via of ['runner', 'tool'] as const) {
    it(`a child run started through the ${via} fails after its task row: at run.failed its delegation hold and its tree hold are gone`, async () => {
      const r = new Rig({ childClient: async () => { throw new Error('MODEL-SECRET'); } });
      await r.session().usingMode('lead');
      let at: Promise<string[]> | null = null;
      r.events.listen((event: HarnessEvent) => {
        if (event.type === 'run.failed' && event.sessionKey === r.child().key()) at = holdsNow(r, event.runId);
      });

      expect(await start(r, via, 'Work', () => {})).toBe('failed');

      expect(at).not.toBeNull();
      expect(await at!).toEqual([]);
    });
  }

  it('a ROOT run fails: at run.failed its tree hold is gone', async () => {
    // The root delegates, so it holds a tree; then its own next model call fails.
    const r = new Rig({ failParentAfterChild: true });
    await r.session().usingMode('lead');
    r.calls([['a', { task: 'Work' }]]);
    r.text('Child done.');
    let at: Promise<string[]> | null = null;
    r.events.listen((event: HarnessEvent) => {
      if (event.type === 'run.failed' && event.sessionKey === r.session().key()) at = holdsNow(r, event.runId);
    });

    await expect(r.runtime.send(r.session(), 'Delegate', undefined, RunContext.root('root', new RunBudget(8)))).rejects.toThrow('MODEL-SECRET');

    expect(at).not.toBeNull();
    expect(await at!).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// Item 8 (TS run15 matrix additions made permanent) and new channels.
// ---------------------------------------------------------------------------

describe('item 8: a second start in the same step, and the plan across a failed and a completed start', () => {
  it('S5: two delegations to the same child in ONE parent step run in turn, each with its task row; the records are the second start\'s', async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    r.calls([['a', { task: 'One' }], ['a', { task: 'Two' }]]);
    r.text('Child one.');
    r.text('Child two.');
    r.text('Parent done.');
    expect((await r.runtime.send(r.session(), 'Delegate', undefined, RunContext.root('root', new RunBudget(8)))).text).toBe('Parent done.');

    const tasks = (await childRows(r)).filter((row) => row.type === 'user').map((row) => row.content);
    expect(tasks).toEqual(['One', 'Two']);
    const parentResults = (await r.session().thread().messages()).flatMap((entry) => entry.message.type === 'tool_result' ? entry.message.tool_results as JsonObject[] : []);
    expect(parentResults.map((entry) => entry.result)).toEqual(['Child one.', 'Child two.']);
    const child = r.child();
    expect(await child.childStartUnconfirmed()).toBe(false);
    expect(await child.childBinding()).not.toBeNull();
    expect(await child.childTurnLive(Math.floor(Date.now() / 1000), 300)).toBe(false);
    const runs = (await childRows(r)).length;
    expect(runs).toBeGreaterThan(0);
  });

  it("S6: a failed start leaves the plan untouched (no plan.updated); a completed one supersedes the held step's approval", async () => {
    const r = new Rig({ plans: true });
    await r.session().usingMode('lead');
    r.calls([['plan_draft', { goal: 'Do the work', steps: [{ title: 'Step 1', instruction: 'Echo once.', owner: 'agent' }] }]]);
    await r.run('Plan it');
    const draft = approvalsIn(await childRows(r)).pop()!;
    await recordApproval(r.child(), draft, true);
    r.text('Plan approved.');
    await r.runtime.send(r.child(), '');
    r.calls([['echo', { n: 1 }]]);
    await (await r.child().plan()).advance();
    const plan = async (): Promise<Record<string, any>> => ((await (await r.child().plan()).current())?.toJSON() ?? {}) as Record<string, any>;
    const echo = (await plan()).steps[0].approvalIds[0] as string;
    await recordApproval(r.child(), echo, true);
    const key = `${r.child().key()}:durable`;
    await r.durable.put(key, { ...((await r.durable.get(key)) ?? {}), child_depth: 1_000 });
    r.queue.length = 0;
    const before = JSON.stringify(await plan());
    const updates: string[] = [];
    r.events.listen((event: HarnessEvent) => { if (event.type === 'plan.updated') updates.push(event.type); });
    const resolve = ToolRegistry.prototype.resolveDetailed;
    const spy = vi.spyOn(ToolRegistry.prototype, 'resolveDetailed').mockImplementation(function (this: ToolRegistry, ...args: Parameters<ToolRegistry['resolveDetailed']>) {
      if (args[1]?.scope.includes('::sub::') === true) throw new Error('REGISTRY-SECRET');
      return resolve.apply(this, args);
    });

    expect((await r.run('Start over')).outcome).toBe('failed');
    spy.mockRestore();
    expect(JSON.stringify(await plan())).toBe(before);
    expect(updates).toEqual([]);

    r.text('Child done.');
    expect((await r.run('Start over')).outcome).toBe('completed');
    const step = (await plan()).steps[0];
    expect(step.approvedIds).not.toContain(echo);
    expect(step.supersededApprovalIds).toContain(echo);
    expect(r.ran.echo).toBeUndefined();
  });
});

describe('round 12: new channels considered', () => {
  it('a tool that throws a value String() cannot convert fails as a tool result naming no text, never as a broken run', async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    r.calls([['boom', {}]]);
    r.text('Child done.');
    expect((await r.run('Work')).outcome).toBe('completed');
    const result = (await childRows(r)).flatMap((row) => row.type === 'tool_result' ? row.tool_results as JsonObject[] : []).find((entry) => entry.tool_name === 'boom');
    expect(result?.result).toBe('The tool failed: Error');
  });

  it('a mark written on a USER row or an assistant row ends nothing (only a result row carries it)', () => {
    const asked: JsonObject = { type: 'assistant', content: '', tool_calls: [{ id: 'c1', name: 'echo', arguments: {} }], tool_approval_requests: [{ approval_id: 'apr_1', tool_call_id: 'c1' }] };
    const mark = { approval_ids: ['apr_1'] };
    expect([...endedByStart([asked, { type: 'user', content: 'x', harness_ended_by_start: mark }])]).toEqual([]);
    expect([...endedByStart([{ ...asked, harness_ended_by_start: mark }, { type: 'user', content: 'x' }])]).toEqual([]);
  });
});
