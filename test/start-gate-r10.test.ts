import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  HarnessError,
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
  assistantRow,
  recordApproval,
  type Durability,
  type HarnessEvent,
  type HarnessTool,
  type JsonObject,
  type LlmRequest,
  type LlmResponse,
  type ModeConfig,
  type RuntimeConfig,
} from '../src/index.js';
// G-387 round 10: the round-9 review findings, at the public seams (the
// subagent TOOL inside a parent turn, SubagentRunner.run, send, decide,
// recordApproval, a host ToolProvider, the plan). No network; every store is
// in memory.

const network = vi.fn(() => { throw new Error('A test tried to use the network.'); });
beforeEach(() => { vi.stubGlobal('fetch', network); });
afterEach(() => { expect(network).not.toHaveBeenCalled(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });

/** A memory store that logs every write and can run a hook before a lock is taken (a second worker). */
class LoggedStore extends MemorySessionStore {
  readonly writes: string[] = [];
  onLock: ((key: string) => Promise<void>) | null = null;

  constructor(private readonly kind: Durability) {
    super();
  }

  override durability(): Durability {
    return this.kind;
  }

  /** When it answers true, the write throws, as a store that is down would. */
  failPut: ((key: string, payload: JsonObject) => boolean) | null = null;

  override async put(key: string, payload: JsonObject, ttlSeconds: number | null = null): Promise<void> {
    if (this.failPut?.(key, payload) === true) throw new Error('STORE-SECRET');
    this.writes.push(key);
    this.#latest.set(key, payload);
    await super.put(key, payload, ttlSeconds);
  }

  readonly #latest = new Map<string, JsonObject>();

  /** The last payload written under `key`, read synchronously (null before any). */
  peek(key: string): JsonObject | null {
    return this.#latest.get(key) ?? null;
  }

  override async forget(key: string): Promise<void> {
    this.writes.push(key);
    await super.forget(key);
  }

  override async withLock<T>(key: string, callback: () => T | Promise<T>, ttlSeconds?: number, waitSeconds?: number): Promise<T> {
    const hook = this.onLock;
    if (hook !== null) await hook(key);

    return super.withLock(key, callback, ttlSeconds, waitSeconds);
  }
}

interface RigOptions {
  /** The child's mode turns agent plans on, and the harness is given a Plans. */
  plans?: boolean;
  declared?: Record<string, number>;
  config?: RuntimeConfig;
  client?: (request: LlmRequest) => Promise<LlmResponse | null>;
}

class Rig {
  readonly queue: LlmResponse[] = [];
  readonly requests: LlmRequest[] = [];
  readonly durable = new LoggedStore('durable');
  readonly ephemeral = new LoggedStore('volatile');
  readonly events = new HarnessEvents();
  readonly tools: Record<string, HarnessTool> = {};
  registry!: ToolRegistry;
  plans: Plans | null = null;
  readonly ran: Record<string, number> = {};
  runtime!: AgentRuntime;
  harness!: PrismHarness;
  modes!: ModeRegistry;
  /** Run once before the next model call is answered. */
  beforeRequest: (() => Promise<void>) | null = null;
  #call = 0;

  constructor(private readonly options: RigOptions = {}) {
    const declared = this.options.declared ?? { a: 2 };
    this.modes = new ModeRegistry({
      default: 'lead',
      modes: {
        lead: {
          system_prompt: 'Lead.', tools: ['echo', ...Object.keys(declared)], max_steps: 8,
          requires_approval: ['echo'],
          subagents: Object.fromEntries(Object.entries(declared).map(([name, steps]) => [name, { mode: 'work', max_steps: steps }])),
        } as ModeConfig,
        work: { system_prompt: 'Work.', tools: ['echo'], max_steps: 8, requires_approval: ['echo'], ...(this.options.plans === true ? { plans: true } : {}) } as ModeConfig,
      },
    });
    const tools = new ToolRegistry()
      .register({ name: 'echo', handle: () => { this.ran.echo = (this.ran.echo ?? 0) + 1; return 'ok'; } });
    this.registry = tools;
    if (this.options.plans === true) this.plans = new Plans({ modes: this.modes, config: { enabled: true }, events: this.events });
    this.runtime = new AgentRuntime({
      modes: this.modes, tools, events: this.events, ...(this.options.config === undefined ? {} : { config: this.options.config }),
      client: async (request) => {
        this.requests.push(request);
        const hook = this.beforeRequest;
        this.beforeRequest = null;
        if (hook !== null) await hook();
        const own = this.options.client === undefined ? null : await this.options.client(request);
        if (own !== null) return own;
        const next = this.queue.shift();
        if (next !== undefined) return next;

        return { text: '', finishReason: 'tool_calls', toolCalls: [{ id: `call_${++this.#call}`, name: 'echo', arguments: { n: this.#call } }] };
      },
    });
    const durable = this.durable, ephemeral = this.ephemeral;
    this.harness = new PrismHarness({
      drivers: { ephemeral: () => ephemeral, durable: () => durable },
      stores: { ephemeral: 'ephemeral', durable: 'durable' },
      runtime: this.runtime,
      ...(this.plans === null ? {} : { plans: this.plans }),
    });
    for (const name of Object.keys(declared)) {
      const tool = new SubagentRunner(this.harness, this.runtime)
        .tool(this.modes.resolve('lead').subagents[name]!, this.session(), RunContext.root('host', new RunBudget(100)), 'host_run');
      this.tools[name] = tool;
      tools.register(tool);
    }
  }

  session(scope = 'chat'): Session {
    return this.harness.for({ type: 'User', id: 1 }).session(scope);
  }

  child(name = 'a'): Session {
    return this.session(`chat::sub::${name}`);
  }

  calls(calls: [string, JsonObject][]): void {
    this.queue.push({ text: '', finishReason: 'tool_calls', toolCalls: calls.map(([name, args]) => ({ id: `call_${++this.#call}`, name, arguments: args })) });
  }

  text(text = 'Done.'): void {
    this.queue.push({ text, finishReason: 'stop' });
  }

  /** The runner starts subagent `name` directly under `context` (no parent turn). */
  async run(context: RunContext, task: unknown = 'Work.', name = 'a'): Promise<{ outcome: string; text: string }> {
    return new SubagentRunner(this.harness, this.runtime)
      .run(this.modes.resolve('lead').subagents[name]!, this.session(), context, 'root', task as string);
  }

  childRequests(): number {
    return this.requests.filter((request) => request.systemPrompt.startsWith('Work.')).length;
  }
}

/** The child's durable record, as stored text. */
async function durableOf(r: Rig, session: Session): Promise<string> {
  return JSON.stringify(await r.durable.get(`${session.key()}:durable`));
}

/** A child the runner never bound, parked on an approval nobody can answer (its binding is gone). */
async function deadChild(r: Rig): Promise<void> {
  const child = r.child();
  await child.rememberChildRefs({ tree: { session: r.session().key(), run: 'run_root' }, planLedger: null });
  await child.usingMode('work');
  await child.thread().record([{ type: 'user', content: 'Work.' }], 'run_child');
  const row = assistantRow('', [{ id: 'e1', name: 'echo', arguments: {} }], {}, [{ approval_id: 'apr_dead', tool_call_id: 'e1', arguments_format: 1 }]) as unknown as JsonObject;
  row.harness_tool_names = ['echo'];
  await child.thread().record([row], 'run_child');
  await expect(recordApproval(r.child(), 'apr_dead', true)).rejects.toMatchObject({ code: 'child_binding_missing' });
}

/** A child that ran once and finished, so it has a saved ceiling and a count. */
async function finishedChild(r: Rig): Promise<void> {
  r.text('Child done.');
  const result = await r.run(RunContext.root('root', new RunBudget(8)));
  expect(result.outcome).toBe('completed');
}

/** A child the runner started, parked on an approval a person can still answer. */
async function parkedChild(r: Rig): Promise<string> {
  await r.run(RunContext.root('root', new RunBudget(8)));
  const rows = (await r.child().thread().messages()).map((entry) => entry.message);
  const requests = rows.flatMap((row) => Array.isArray(row.tool_approval_requests) ? row.tool_approval_requests as JsonObject[] : []);

  return requests[requests.length - 1]!.approval_id as string;
}

/** Everything the parent keeps, as text: its thread, records and run state. */
async function parentText(r: Rig): Promise<string> {
  const key = r.session().key();

  return JSON.stringify([
    await r.durable.get(`${key}:thread`),
    await r.durable.get(`${key}:durable`),
    await r.ephemeral.get(`${key}:ephemeral`),
  ]);
}

// ---------------------------------------------------------------------------
// Round 9 helpers
// ---------------------------------------------------------------------------

const ROOT = (): RunContext => RunContext.root('root', new RunBudget(8));
const now = (): number => Math.floor(Date.now() / 1000);

/** The child's durable record (an object), as stored. */
async function childRecord(r: Rig): Promise<JsonObject> {
  return (await r.durable.get(`${r.child().key()}:durable`)) ?? {};
}

/** Rewrite the child's durable record, as a host (or a crash) would leave it. */
async function setChildRecord(r: Rig, change: (stored: JsonObject) => JsonObject): Promise<void> {
  await r.durable.put(`${r.child().key()}:durable`, change(await childRecord(r)));
}

/** The registry throws `error` for the child: a failure after commit (b), before the task row. */
function failChildRegistry(error: unknown = new Error('REGISTRY-SECRET')): void {
  const resolve = ToolRegistry.prototype.resolveDetailed;
  vi.spyOn(ToolRegistry.prototype, 'resolveDetailed').mockImplementation(function (this: ToolRegistry, ...args: Parameters<ToolRegistry['resolveDetailed']>) {
    if (args[1]?.scope.includes('::sub::') === true) throw error;
    return resolve.apply(this, args);
  });
}

/** The housekeeping undo of a failed start fails too (the same store outage). */
function failUndo(): void {
  vi.spyOn(Session.prototype, 'undoChildStart').mockRejectedValue(new Error('STORE-SECRET'));
}

/** Run `hook` once, on the child, right after its run begins: between commit (b) and the task row. */
function betweenCommitAndTask(hook: (child: Session) => Promise<void>): void {
  const beginRun = Session.prototype.beginRun;
  let fired = false;
  vi.spyOn(Session.prototype, 'beginRun').mockImplementation(async function (this: Session, ...args: Parameters<Session['beginRun']>) {
    const begun = await beginRun.apply(this, args);
    if (!fired && this.scope.includes('::sub::')) {
      fired = true;
      await hook(this);
    }
    return begun;
  });
}

/**
 * A start through the runner directly, or through the subagent tool inside a
 * parent turn; the framed text. `child` is what the child's model answers, in
 * order (queued after the parent's call to the tool).
 */
async function start(r: Rig, via: 'runner' | 'tool', task = 'Start over', child: string[] = []): Promise<string> {
  if (via === 'runner') {
    for (const text of child) r.text(text);
    const result = await r.run(ROOT(), task);
    return `${result.outcome}: ${result.text}`;
  }
  r.calls([['a', { task }]]);
  for (const text of child) r.text(text);
  r.text('Parent done.');
  const response = await r.runtime.send(r.session(), 'Delegate', undefined, ROOT());
  expect(response.text).toBe('Parent done.');
  const rows = (await r.session().thread().messages()).map((entry) => entry.message);
  const result = rows.filter((row) => row.type === 'tool_result').flatMap((row) => row.tool_results as JsonObject[]).pop();

  return String(result?.result ?? '');
}

const BINDING = { session: { provider: null, model: null }, declared: { provider: null, model: null }, parentMode: { provider: null, model: null } };

/** A dead child (no binding anywhere) that is given a binding again: a person can answer its approval. */
async function revivedChild(r: Rig): Promise<void> {
  await deadChild(r);
  await r.child().rememberChildBinding(BINDING, 1);
}

/** Every way a host or a person reaches the child outside the runner refuses, and no model step is taken. */
async function childUnreachable(r: Rig, approval: string): Promise<void> {
  const before = r.childRequests();
  const ran = r.ran.echo;
  await expect(recordApproval(r.child(), approval, true)).rejects.toMatchObject({ code: 'child_binding_missing' });
  await expect(r.runtime.decide(r.child(), approval, true)).rejects.toMatchObject({ code: 'child_binding_missing' });
  await expect(r.runtime.send(r.child(), '')).rejects.toMatchObject({ code: 'child_binding_missing' });
  await expect(r.runtime.send(r.child(), 'New message')).rejects.toMatchObject({ code: 'child_binding_missing' });
  await expect(r.runtime.send(r.child(), '', undefined, RunContext.root('host', new RunBudget(10)))).rejects.toMatchObject({ code: 'child_binding_missing' });
  await expect(r.runtime.send(r.child(), '', undefined, RunContext.metadata('turn_1', 'person'))).rejects.toMatchObject({ code: 'child_binding_missing' });
  await expect(r.runtime.sendStructured(r.child(), 'Structured', { type: 'object' })).rejects.toMatchObject({ code: 'child_binding_missing' });
  const stream = r.runtime.streamDecision(r.child(), [{ approvalId: approval, approved: true }]);
  await expect((async () => { for await (const _step of stream) { /* drain */ } })()).rejects.toMatchObject({ code: 'child_binding_missing' });
  expect(r.childRequests()).toBe(before);
  expect(r.ran.echo).toBe(ran);
}

const DECISION_ROW: JsonObject = { type: 'tool_result', tool_results: [], tool_approval_responses: [{ approval_id: 'apr_dead', approved: true, reason: null }] };

// ---------------------------------------------------------------------------
// Round 10 helpers
// ---------------------------------------------------------------------------

/** A child delegated through the subagent tool in a parent turn: saved in the parent run's tree. That tree. */
async function delegatedChild(r: Rig): Promise<{ session: string; run: string }> {
  // A parent turn with no run context opens a run tree of its own.
  r.calls([['a', { task: 'First task' }]]);
  r.text('Child done.');
  r.text('Parent done.');
  expect((await r.runtime.send(r.session(), 'Delegate')).text).toBe('Parent done.');
  const tree = (await r.child().childRefs())?.tree ?? null;
  expect(tree).not.toBeNull();
  expect(tree!.session).toBe(r.session().key());

  return tree!;
}

/** The run id of the child's newest user row. */
async function lastTaskRun(r: Rig): Promise<string> {
  const rows = await r.child().thread().messages();

  return rows.filter((entry) => entry.message.type === 'user').pop()!.runId!;
}

/** A direct start the runner makes outside any parent turn: by `SubagentRunner.run`, or by the tool wired with a host context. */
async function directStart(r: Rig, via: 'runner' | 'tool outside a turn', task: string): Promise<string> {
  if (via === 'runner') {
    const result = await r.run(ROOT(), task);
    return `${result.outcome}: ${result.text}`;
  }

  return String(await r.tools.a!.handle({ task }));
}

/** The child's thread and records, as text. */
async function childText(r: Rig): Promise<string> {
  return JSON.stringify([await durableOf(r, r.child()), await r.child().thread().messages()]);
}

// ---------------------------------------------------------------------------
// Item 1 (Py F1, TS I-3): an unconfirmed record reads as the last CONFIRMED
// tree and plan ledger, never as none.
// ---------------------------------------------------------------------------

describe('item 1: an unconfirmed child record keeps the tree it was confirmed in', () => {
  for (const via of ['runner', 'tool outside a turn'] as const) {
    it(`through the ${via}: after a failed start whose undo failed too, the next direct start keeps the root's tree, and a host turn on the child runs in it`, async () => {
      const r = new Rig();
      await r.session().usingMode('lead');
      const tree = await delegatedChild(r);

      failChildRegistry();
      failUndo();
      expect(await directStart(r, via, 'Retry')).toMatch(/failed/);
      vi.restoreAllMocks();
      // Provisional: no binding, but the tree it was last confirmed in.
      expect(await r.child().childBinding()).toBeNull();
      expect(await r.child().childStartUnconfirmed()).toBe(true);
      expect((await r.child().childRefs())?.tree).toEqual(tree);

      // Once the failed start's hold has expired, a direct start commits.
      vi.spyOn(Date, 'now').mockReturnValue(Date.now() + 301_000);
      r.text('Child done again.');
      expect(await directStart(r, via, 'Again')).not.toMatch(/failed|denied|exhausted/);
      expect((await r.child().childRefs())?.tree).toEqual(tree);

      // A host turn on the child continues the root's tree, never a new one.
      r.text('Hello back.');
      await r.runtime.send(r.child(), 'Hello.').catch(() => null);
      expect(await r.child().runTree(await lastTaskRun(r))).toEqual(tree);
    });
  }

  it("a host turn on such a child is bounded by the root's tree: with the tree spent, it takes no model step", async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    const tree = await delegatedChild(r);
    failChildRegistry();
    failUndo();
    await directStart(r, 'runner', 'Retry');
    vi.restoreAllMocks();
    vi.spyOn(Date, 'now').mockReturnValue(Date.now() + 301_000);
    r.text('Child done again.');
    await directStart(r, 'runner', 'Again');

    // The root's tree is spent: its record says every step is used.
    const key = `${tree.session}:run-tree:${tree.run}`;
    const record = await r.durable.get(key);
    expect(record).not.toBeNull();
    await r.durable.put(key, { ...record!, steps: 1_000 });
    const before = r.childRequests();
    r.text('Hello back.');
    await expect(r.runtime.send(r.child(), 'Hello.')).rejects.toMatchObject({ code: 'run_not_permitted' });
    expect(r.childRequests()).toBe(before);
  });

  it('a child with no confirmed tree before the failed start still reads as having none (fails closed)', async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    failChildRegistry();
    failUndo();
    expect((await r.run(ROOT(), 'First')).outcome).toBe('failed');
    vi.restoreAllMocks();
    expect(await r.child().childRefs()).toEqual({ tree: null, planLedger: null });
    expect(await r.child().childBinding()).toBeNull();
  });

  it('two failed starts in a row, each with a failed undo, still read the tree confirmed before the first', async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    const tree = await delegatedChild(r);
    const at = Date.now();
    for (const shift of [0, 301_000]) {
      vi.spyOn(Date, 'now').mockReturnValue(at + shift);
      failChildRegistry();
      failUndo();
      expect(await directStart(r, 'runner', 'Retry')).toMatch(/failed/);
      vi.restoreAllMocks();
    }
    expect((await r.child().childRefs())?.tree).toEqual(tree);
  });
});

// ---------------------------------------------------------------------------
// Item 2 (Py I3): a decision a person recorded blocks a re-delegation, as an
// answerable pending approval does, until a resume acts on it.
// ---------------------------------------------------------------------------

describe("item 2: a model cannot void a person's decision by delegating", () => {
  for (const via of ['runner', 'tool'] as const) {
    for (const approved of [true, false]) {
      it(`through the ${via}: a person ${approved ? 'approves' : 'denies'}, the parent re-delegates before any resume: refused child_waiting, nothing written; a host resume then acts on it once`, async () => {
        const r = new Rig();
        await r.session().usingMode('lead');
        const approval = await parkedChild(r);
        await recordApproval(r.child(), approval, approved);
        const before = await childText(r);
        const requests = r.childRequests();

        const framed = await start(r, via, 'Start over');

        expect(framed).toMatch(/waiting on a person/);
        expect(r.childRequests()).toBe(requests);
        expect(r.ran.echo).toBeUndefined();
        expect(await childText(r)).toBe(before);

        r.queue.length = 0;
        r.text('Child done.');
        await r.runtime.send(r.child(), '');
        expect(r.ran.echo).toBe(approved ? 1 : undefined);
        r.text('Again.');
        await r.runtime.send(r.child(), '').catch(() => null);
        expect(r.ran.echo).toBe(approved ? 1 : undefined);
      });
    }
  }

  it("(b)'s locked re-check: a person deciding between the gate read and (b) refuses the start; a host resume then runs the approved call once", async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    await finishedChild(r);
    const childKey = `${r.child().key()}:durable`;
    let after = '';
    r.durable.onLock = async (key) => {
      if (key !== childKey) return;
      r.durable.onLock = null;
      const parked = await r.runtime.send(r.child(), 'Host question');
      await recordApproval(r.child(), parked.pendingApprovals[0]!.id, true);
      after = await childText(r);
    };

    const result = await r.run(ROOT(), 'Start over');

    expect(after).not.toBe('');
    expect(result.outcome).toBe('denied');
    expect(result.text).toMatch(/waiting on a person/);
    expect(await childText(r)).toBe(after);
    expect(r.ran.echo).toBeUndefined();
    r.text('Child done.');
    await r.runtime.send(r.child(), '');
    expect(r.ran.echo).toBe(1);
  });

  it('a decided approval nobody can act on any more (its binding is gone) does not block, and is not run (P3 unchanged)', async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    await revivedChild(r);
    await recordApproval(r.child(), 'apr_dead', true);
    await setChildRecord(r, ({ child_binding: _gone, ...rest }) => rest);

    r.text('Child done.');
    expect((await r.run(ROOT(), 'New task')).outcome).toBe('completed');
    expect(r.ran.echo).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// Item 3 (Py I2): the mode (b) writes is provisional like its other records.
// ---------------------------------------------------------------------------

describe('item 3: the mode a runner start wrote reads as absent until its task row', () => {
  for (const prior of ['lead', null] as const) {
    it(`a provisional child reads ${prior === null ? 'no mode' : 'the mode it had before'}; once a start records its task row, its own`, async () => {
      const r = new Rig();
      await r.session().usingMode('lead');
      await deadChild(r);
      const key = `${r.child().key()}:ephemeral`;
      const { mode: _work, ...state } = (await r.ephemeral.get(key)) ?? {};
      await r.ephemeral.put(key, prior === null ? state : { ...state, mode: prior });
      failChildRegistry();
      failUndo();

      expect((await r.run(ROOT(), 'Start over')).outcome).toBe('failed');
      vi.restoreAllMocks();

      expect((await r.ephemeral.get(key))?.mode).toBe('work');
      expect(await r.child().mode()).toBe(prior);
      expect((await r.runtime.modeOf(r.child())).name).toBe(prior ?? 'lead');

      vi.spyOn(Date, 'now').mockReturnValue(Date.now() + 301_000);
      r.text('Child done.');
      expect((await r.run(ROOT(), 'Again')).outcome).toBe('completed');
      expect(await r.child().mode()).toBe('work');
    });
  }
});

// ---------------------------------------------------------------------------
// Item 4 (Py I4): a throw that is not an Error from host code.
// ---------------------------------------------------------------------------

describe('item 4: a thrown value that is not an Error', () => {
  const thrown: [string, () => unknown][] = [
    ['a string', () => 'SECRET string'],
    ['undefined', () => undefined],
    ['a number', () => 42],
    ['a plain object', () => ({ message: 'SECRET object' })],
  ];

  for (const [name, make] of thrown) {
    it(`${name} from host code between (b) and the task row: framed failed, run.failed emitted, no live hold, the dead approval refused`, async () => {
      const r = new Rig();
      await r.session().usingMode('lead');
      await deadChild(r);
      const failed: HarnessEvent[] = [];
      r.events.listen((event: HarnessEvent) => { if (event.type === 'run.failed' && event.sessionKey === r.child().key()) failed.push(event); });
      failChildRegistry(make());

      const result = await r.run(ROOT(), 'Start over');
      vi.restoreAllMocks();

      expect(result.outcome).toBe('failed');
      expect(result.text).not.toMatch(/SECRET/);
      expect(failed).toHaveLength(1);
      expect(JSON.stringify(failed)).not.toMatch(/SECRET/);
      expect(await r.child().childTurnLive(now(), 300)).toBe(false);
      await expect(recordApproval(r.child(), 'apr_dead', true)).rejects.toMatchObject({ code: 'child_binding_missing' });
    });

    it(`${name} from host code on a root turn still reaches the caller, and the run is failed and its lock free`, async () => {
      const r = new Rig();
      await r.session().usingMode('lead');
      const value = make();
      vi.spyOn(ToolRegistry.prototype, 'resolveDetailed').mockImplementation(() => { throw value; });

      const error = await r.runtime.send(r.session(), 'Hello').then(() => 'resolved', (caught: unknown) => caught);
      vi.restoreAllMocks();

      expect(error).toBe(value);
      expect((await r.session().run())?.status).toBe('failed');
      await expect(r.session().lock(() => 'free', 10, 0)).resolves.toBe('free');
    });
  }
});

// ---------------------------------------------------------------------------
// Item 5 (TS R13-1): confirming a provisional record is a READ, allowed in
// the read-only tool-resolution view.
// ---------------------------------------------------------------------------

describe("item 5: a host tool provider may read the child's session state while a runner start resolves its tools", () => {
  for (const via of ['runner', 'tool'] as const) {
    it(`through the ${via}: run, binding, refs, depth and ceiling read without throwing, the delegation completes, and no write gets through`, async () => {
      const r = new Rig();
      await r.session().usingMode('lead');
      const reads: unknown[] = [];
      const refused: string[] = [];
      let wrote = 0;
      r.registry.registerProvider(async (session) => {
        if (!session.scope.includes('::sub::')) return [];
        const run = await session.run();
        reads.push({ run: run?.id ?? null, binding: await session.childBinding(), refs: await session.childRefs(),
          depth: await session.childDepth(), ceiling: await session.childCeiling(), unconfirmed: await session.childStartUnconfirmed() });
        for (const write of [() => session.usingMode('lead'), () => session.thread().record([{ type: 'user', content: 'x' }]), () => session.usingCapability('x', {})]) {
          try {
            await write();
            wrote += 1;
          } catch (error) {
            refused.push((error as Error).message);
          }
        }

        return [];
      }, 'host', 'host_probe');

      const framed = await start(r, via, 'Start', ['Child done.']);

      expect(framed).not.toMatch(/failed|denied|exhausted/);
      expect(reads).toHaveLength(1);
      // The start is still provisional while it resolves its tools: the run
      // before it (none) and no binding.
      expect(reads[0]).toMatchObject({ run: null, binding: null, unconfirmed: true });
      expect(wrote).toBe(0);
      expect(refused).toEqual(Array(3).fill('Harness tool resolution is read-only.'));
    });
  }

  it("effectiveTools on a child left provisional by a failed start whose undo failed: a provider's reads work", async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    await deadChild(r);
    failChildRegistry();
    failUndo();
    await r.run(ROOT(), 'Start over');
    vi.restoreAllMocks();
    let read: unknown = 'not called';
    r.registry.registerProvider(async (session) => {
      if (session.scope.includes('::sub::')) read = [(await session.run())?.id ?? null, await session.childBinding()];
      return [];
    }, 'host', 'host_probe');

    await expect(r.runtime.effectiveTools(r.child())).resolves.toBeDefined();
    expect(read).toEqual(['run_child', null]);
  });
});

// ---------------------------------------------------------------------------
// Item 6 (TS I-4): tool_denied for the calls a start ends only once recorded.
// ---------------------------------------------------------------------------

describe('item 6: the events for the calls a runner start ends follow the write that records them', () => {
  it('a store failure on the task-row write emits no tool_denied', async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    await deadChild(r);
    const denied: HarnessEvent[] = [];
    r.events.listen((event: HarnessEvent) => { if (event.type === 'tool_denied' && event.sessionKey === r.child().key()) denied.push(event); });
    const threadKey = `${r.child().key()}:thread`;
    r.durable.failPut = (key, payload) => key === threadKey && JSON.stringify(payload).includes('Start over');

    const result = await r.run(ROOT(), 'Start over');
    r.durable.failPut = null;

    expect(result.outcome).toBe('failed');
    expect(denied).toHaveLength(0);
  });

  it('a start that records its task row emits one tool_denied per ended call, after the row is stored', async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    await deadChild(r);
    const threadKey = `${r.child().key()}:thread`;
    const stored: boolean[] = [];
    r.events.listen((event: HarnessEvent) => {
      if (event.type !== 'tool_denied' || event.sessionKey !== r.child().key()) return;
      // The thread write is synchronous in the memory store's map: read it as it is now.
      stored.push(JSON.stringify(r.durable.peek(threadKey)).includes('Start over'));
    });
    r.text('Child done.');

    expect((await r.run(ROOT(), 'Start over')).outcome).toBe('completed');
    expect(stored).toEqual([true]);
  });
});

// ---------------------------------------------------------------------------
// Item 7 (PHP run12 Low): a framed failure, run.failed and the run row name a
// NAMED class only: never a path, a file or a line (probe Z4).
// ---------------------------------------------------------------------------

describe('item 7: no class name that carries a path reaches the model, run.failed or the run row', () => {
  const forms: [string, () => unknown][] = [
    ['an anonymous class', () => new (class extends Error {})('SECRET')],
    ['a class named with a Windows path', () => {
      const E = class extends Error {};
      Object.defineProperty(E, 'name', { value: 'C:\\srv\\app\\tool.ts:12' });
      return new E('SECRET');
    }],
    ['a class named with a POSIX path', () => {
      const E = class extends Error {};
      Object.defineProperty(E, 'name', { value: '/srv/app/tool.js:12' });
      return new E('SECRET');
    }],
    ['a class whose static name getter answers a path', () => new (class extends Error { static override get name(): string { return 'srv/app/tool.py'; } })('SECRET')],
    ['an error with a forged own constructor', () => {
      const error = new Error('SECRET');
      Object.defineProperty(error, 'constructor', { value: { name: '/srv/app/tool.mjs' } });
      return error;
    }],
    ['a prototype whose constructor names a path', () => {
      class E extends Error {}
      Object.defineProperty(E.prototype, 'constructor', { value: { name: 'srv\\app\\tool.ts' } });
      return new E('SECRET');
    }],
    ['an error whose name carries a path', () => {
      const error = new TypeError('SECRET');
      error.name = '/srv/app/tool.ts:3';
      return error;
    }],
  ];
  const clean = (text: string): void => {
    expect(text).not.toMatch(/@anonymous/);
    expect(text).not.toMatch(/\.(php|ts|js|mjs|py)\b/);
    expect(text).not.toMatch(/srv|SECRET/);
  };

  for (const timing of ['before its task row', 'after its task row'] as const) {
    for (const [name, make] of forms) {
      it(`${name}, thrown ${timing}: the tool_result row, the next provider request, run.failed and the run row`, async () => {
        const error = make();
        const r = new Rig(timing === 'after its task row'
          ? { client: async (request) => { if (request.systemPrompt.startsWith('Work.')) throw error; return null; } }
          : {});
        await r.session().usingMode('lead');
        if (timing === 'before its task row') failChildRegistry(error);
        const failed: HarnessEvent[] = [];
        r.events.listen((event: HarnessEvent) => { if (event.type === 'run.failed') failed.push(event); });

        const framed = await start(r, 'tool', 'Work.');
        vi.restoreAllMocks();

        expect(framed).toMatch(/failed/);
        clean(framed);
        expect(framed).not.toMatch(/[\\/]/);
        const next = r.requests.filter((request) => request.systemPrompt.startsWith('Lead.')).pop()!;
        clean(JSON.stringify(next.messages));
        expect(failed.length).toBeGreaterThan(0);
        for (const event of failed) {
          clean(JSON.stringify(event));
          expect(String((event as { failure?: unknown }).failure)).not.toMatch(/[\\/]/);
        }
        const run = (await r.ephemeral.get(`${r.child().key()}:ephemeral`))?.run;
        clean(JSON.stringify(run ?? null));
        if (timing === 'after its task row') expect((run as JsonObject).failure).toMatch(/^The run failed \([A-Za-z]+\)\.$/);
      });
    }
  }
});

// ---------------------------------------------------------------------------
// Item 8 (PHP run12 Info 2): a call a runner start ended as not run never
// reads as approved.
// ---------------------------------------------------------------------------

describe('item 8: a call a runner start ended reads as ended, never approved', () => {
  it("the child's plan does not count an approval whose call the start ended as approved work", async () => {
    const r = new Rig({ plans: true, declared: { a: 8 } });
    await r.session().usingMode('lead');
    // The child drafts a plan, and a person approves it.
    r.calls([['plan_draft', { goal: 'Do the work', steps: [{ title: 'Step 1', instruction: 'Echo once.', owner: 'agent' }] }]]);
    await r.run(ROOT(), 'Plan it');
    const draft = (await r.child().thread().messages()).flatMap((entry) => Array.isArray(entry.message.tool_approval_requests) ? entry.message.tool_approval_requests as JsonObject[] : []).pop()!;
    await recordApproval(r.child(), draft.approval_id as string, true);
    r.text('Plan approved.');
    await r.runtime.send(r.child(), '');
    // Its step stages an echo that needs a person, and a person approves it.
    await (await r.child().plan()).advance();
    const plan = async (): Promise<Record<string, any>> => ((await (await r.child().plan()).current())?.toJSON() ?? {}) as Record<string, any>;
    const step = (await plan()).steps[0];
    expect(step.state).toBe('held');
    const echo = step.approvalIds[0] as string;
    await recordApproval(r.child(), echo, true);
    // The binding is lost: nobody can act on it; a re-delegation ends the call.
    await setChildRecord(r, ({ child_binding: _gone, ...rest }) => rest);
    r.queue.length = 0;
    r.text('Child done.');
    expect((await r.run(ROOT(), 'New task')).outcome).toBe('completed');
    expect(r.ran.echo).toBeUndefined();

    const after = (await plan()).steps[0];
    expect(after.approvedIds).not.toContain(echo);
  });
});
