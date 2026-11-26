import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  HarnessError,
  HarnessEvents,
  MemorySessionStore,
  ModeRegistry,
  PlanBudget,
  PlanBudgetInvalid,
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
// G-387 round 9: a runner start's child records are PROVISIONAL until its
// task row exists, so no second write is needed to make a failed start safe;
// and a runner start acts on no recorded decision. At the public seams (the
// subagent TOOL inside a parent turn, SubagentRunner.run, send, decide,
// streamDecision and recordApproval). No network; every store is in memory.

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

  override async put(key: string, payload: JsonObject, ttlSeconds: number | null = null): Promise<void> {
    this.writes.push(key);
    await super.put(key, payload, ttlSeconds);
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
        work: { system_prompt: 'Work.', tools: ['echo'], max_steps: 8, requires_approval: ['echo'] } as ModeConfig,
      },
    });
    const tools = new ToolRegistry()
      .register({ name: 'echo', handle: () => { this.ran.echo = (this.ran.echo ?? 0) + 1; return 'ok'; } });
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

describe('R12-1: a failed start whose undo fails too leaves nothing answerable', () => {
  for (const via of ['runner', 'tool'] as const) {
    it(`through the ${via}: the dead approval stays refused, the child takes no model step, a re-delegation runs once the hold expires`, async () => {
      const r = new Rig();
      await r.session().usingMode('lead');
      await deadChild(r);
      failChildRegistry();
      failUndo();

      const framed = await start(r, via);
      vi.restoreAllMocks();

      expect(framed).toMatch(/failed/);
      expect(framed).not.toMatch(/SECRET/);
      // (b) is still on the child (its undo failed): provisional, it reads as absent.
      expect((await childRecord(r)).child_binding).toBeDefined();
      expect(await r.child().childBinding()).toBeNull();
      expect(await r.child().childCeiling()).toBeNull();
      await childUnreachable(r, 'apr_dead');

      // Once the start's hold has expired, a re-delegation is not blocked.
      vi.spyOn(Date, 'now').mockReturnValue(Date.now() + 301_000);
      r.queue.length = 0;
      r.text('Child done.');
      const again = await r.run(ROOT(), 'Start over again');
      expect(again.outcome).toBe('completed');
      expect(r.ran.echo).toBeUndefined();
    });
  }
});

describe('R12-2 / Py F3: a runner start acts on no recorded decision before its task row', () => {
  for (const via of ['runner', 'tool'] as const) {
    it(`through the ${via}: a decision recorded between (b) and the task row is never acted on by this start`, async () => {
      const r = new Rig();
      await r.session().usingMode('lead');
      await deadChild(r);
      let accepted = false;
      betweenCommitAndTask(async (child) => {
        accepted = await recordApproval(child, 'apr_dead', true).then(() => true, () => false);
        // A host writing the decision row itself, bypassing every check.
        await child.thread().record([DECISION_ROW], null);
      });

      await start(r, via, 'Start over', ['Child done.']);
      vi.restoreAllMocks();

      expect(accepted).toBe(false);
      expect(r.ran.echo).toBeUndefined();
      // The task row ended the call: it has a result that says it did not run.
      const rows = (await r.child().thread().messages()).map((entry) => entry.message);
      const results = rows.filter((row) => row.type === 'tool_result').flatMap((row) => row.tool_results as JsonObject[]);
      expect(String(results.find((entry) => entry.tool_call_id === 'e1')?.result)).toMatch(/not run/i);
      // A host resume afterwards does not run it either.
      r.text('Resumed.');
      await r.runtime.send(r.child(), '').catch(() => null);
      expect(r.ran.echo).toBeUndefined();
    });
  }

  it('a decision recorded in the window of a start that then fails is acted on by neither the next delegation nor a host resume', async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    await deadChild(r);
    betweenCommitAndTask(async (child) => {
      await recordApproval(child, 'apr_dead', true).catch(() => null);
      await child.thread().record([DECISION_ROW], null);
    });
    failChildRegistry();

    expect((await r.run(ROOT())).outcome).toBe('failed');
    vi.restoreAllMocks();

    await expect(r.runtime.send(r.child(), '')).rejects.toMatchObject({ code: 'child_binding_missing' });
    r.text('Child done.');
    expect((await r.run(ROOT(), 'New task')).outcome).toBe('completed');
    expect(r.ran.echo).toBeUndefined();
  });

  it('P3: a person approves while the binding is alive, the binding is then lost: a bare resume is refused and a re-delegation runs nothing of it', async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    await revivedChild(r);
    await recordApproval(r.child(), 'apr_dead', true);
    await setChildRecord(r, ({ child_binding: _gone, ...rest }) => rest);

    await expect(r.runtime.send(r.child(), '')).rejects.toMatchObject({ code: 'child_binding_missing' });
    r.text('Child done.');
    expect((await r.run(ROOT(), 'New task')).outcome).toBe('completed');
    expect(r.ran.echo).toBeUndefined();
  });
});

describe('a provisional record whose task row exists is a confirmed one', () => {
  it('the normal path: a started child parks, its approval is answered, the resume runs the call', async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    const approval = await parkedChild(r);
    // Housekeeping cleared the marks once the task row was recorded.
    expect(Object.hasOwn(await childRecord(r), 'child_provisional')).toBe(false);
    expect(Object.hasOwn((await r.child().state()).run as JsonObject, 'provisional')).toBe(false);
    const before = r.childRequests();

    await recordApproval(r.child(), approval, true);
    r.text('Child done.');
    const resumed = await r.runtime.send(r.child(), '');

    expect(r.ran.echo).toBe(1);
    expect(resumed.text).toBe('Child done.');
    expect(r.childRequests()).toBe(before + 1);
  });

  it('a crash right after the task row (the housekeeping clear never ran) changes nothing', async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    const approval = await parkedChild(r);
    const task = (await r.child().thread().messages()).filter((entry) => entry.message.type === 'user').pop()!;
    await setChildRecord(r, (stored) => ({ ...stored, child_provisional: task.runId! }));

    expect(await r.child().childBinding()).not.toBeNull();
    await recordApproval(r.child(), approval, true);
    r.text('Child done.');
    await r.runtime.send(r.child(), '');
    expect(r.ran.echo).toBe(1);
  });

  it('the housekeeping clear failing changes nothing', async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    vi.spyOn(Session.prototype as unknown as { confirmChildStart: () => Promise<void> }, 'confirmChildStart').mockRejectedValue(new Error('STORE-SECRET'));
    const approval = await parkedChild(r);
    vi.restoreAllMocks();

    expect(Object.hasOwn(await childRecord(r), 'child_provisional')).toBe(true);
    await recordApproval(r.child(), approval, true);
    r.text('Child done.');
    await r.runtime.send(r.child(), '');
    expect(r.ran.echo).toBe(1);
  });

  for (const garbage of [42, '', null, { run: 'x' }, ['x']] as const) {
    it(`a garbage provisional value (${JSON.stringify(garbage)}) fails closed`, async () => {
      const r = new Rig();
      await r.session().usingMode('lead');
      const approval = await parkedChild(r);
      await setChildRecord(r, (stored) => ({ ...stored, child_provisional: garbage as never }));
      await childUnreachable(r, approval);
    });
  }

  it('a legacy record with no provisional field is unchanged', async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    const approval = await parkedChild(r);
    await setChildRecord(r, ({ child_provisional: _none, ...rest }) => rest);

    await recordApproval(r.child(), approval, true);
    r.text('Child done.');
    await r.runtime.send(r.child(), '');
    expect(r.ran.echo).toBe(1);
  });
});

describe('new channels considered', () => {
  it('a task row recorded under ANOTHER run id does not confirm the record', async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    await parkedChild(r);
    await r.child().thread().record([{ type: 'user', content: 'Forged.' }], 'run_other');
    await setChildRecord(r, (stored) => ({ ...stored, child_provisional: 'run_missing' }));

    // The user row ended the approval; every turn on the child is still refused.
    expect(await r.child().childBinding()).toBeNull();
    const before = r.childRequests();
    await expect(r.runtime.send(r.child(), '')).rejects.toMatchObject({ code: 'child_binding_missing' });
    await expect(r.runtime.send(r.child(), 'New message')).rejects.toMatchObject({ code: 'child_binding_missing' });
    expect(r.childRequests()).toBe(before);
  });

  it('a row under the provisional run id that is not a user row does not confirm it', async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    const approval = await parkedChild(r);
    await r.child().thread().record([{ type: 'tool_result', tool_results: [], tool_approval_responses: [] }], 'run_missing');
    await setChildRecord(r, (stored) => ({ ...stored, child_provisional: 'run_missing' }));

    await childUnreachable(r, approval);
  });

  it('a forged provisional id that matches an OLD task row grants no more than a record with no mark (a host owns its store)', async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    const approval = await parkedChild(r);
    const first = (await r.child().thread().messages()).find((entry) => entry.message.type === 'user')!;
    await setChildRecord(r, (stored) => ({ ...stored, child_provisional: first.runId! }));

    // Exactly the confirmed (legacy) behaviour, never more: the saved ceiling still bounds it.
    expect(await r.child().childCeiling()).toBe(2);
    await recordApproval(r.child(), approval, true);
    r.text('Child done.');
    await r.runtime.send(r.child(), '');
    expect(r.ran.echo).toBe(1);
  });

  it('a resume racing the task row write is refused and takes no model step', async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    await deadChild(r);
    r.text('Child done.');
    let raced: unknown = null;
    let stepped = -1;
    betweenCommitAndTask(async (child) => {
      const before = r.childRequests();
      raced = await r.runtime.send(child, '').then(() => 'ran', (error: unknown) => error);
      stepped = r.childRequests() - before;
    });

    expect((await r.run(ROOT())).outcome).toBe('completed');
    vi.restoreAllMocks();
    expect(raced).toMatchObject({ code: 'child_binding_missing' });
    expect(stepped).toBe(0);
    expect(r.ran.echo).toBeUndefined();
  });

  it('every reader of the child records reads a provisional record with no task row as absent', async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    await parkedChild(r);
    await setChildRecord(r, (stored) => ({ ...stored, child_provisional: 'run_missing' }));
    const child = r.child();

    expect(await child.childBinding()).toBeNull();
    expect(await child.childCeiling()).toBeNull();
    expect(await child.childDepth()).toBe(0);
    expect(await child.childRefs()).toEqual({ tree: null, planLedger: null });
    let claimed: number | null | 'unset' = 'unset';
    await child.claimChildSteps('run_probe', now(), 300, (count) => { claimed = count; return 0; });
    expect(claimed).toBeNull();
  });

  it('no read of the child records bypasses the one view that applies the rule (source scan)', async () => {
    const { readFile } = await import('node:fs/promises');
    const source = await readFile(new URL('../src/session.ts', import.meta.url), 'utf8');
    expect(source).toContain('#childRecord(');
    // Outside the writers, the child fields are read only from the view (`record.`), never from the raw store.
    const raw = source.split('\n').filter((line) => /\bstored\.child_(binding|depth|ceiling|refs)\b/.test(line));
    expect(raw).toEqual([]);
  });
});

describe('item 7: an attempted run with no task row is absent from run state and history', () => {
  it('after a failed start the child still names the run it had, and the attempted run has no tree entry', async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    await finishedChild(r);
    const had = await r.child().run();
    expect(had?.status).toBe('completed');
    const attempted: string[] = [];
    r.events.listen((event: HarnessEvent) => {
      if (event.type === 'run.started' && event.sessionKey === r.child().key()) attempted.push(event.runId);
    });
    failChildRegistry();

    expect((await r.run(ROOT(), 'Again.')).outcome).toBe('failed');
    vi.restoreAllMocks();

    expect((await r.child().run())?.id).toBe(had!.id);
    expect((await r.child().run())?.status).toBe('completed');
    expect(attempted).toHaveLength(1);
    expect(await r.child().runTree(attempted[0]!)).toBeNull();
  });

  it('a parked child whose binding was lost: a failed start leaves the parked run current, and answering follows the existing rules', async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    await deadChild(r);
    await r.child().beginRun('run_child', 'work', 'p', 'm');
    const parked = await r.child().run();
    failChildRegistry();

    expect((await r.run(ROOT(), 'Again.')).outcome).toBe('failed');
    vi.restoreAllMocks();

    expect((await r.child().run())?.id).toBe(parked!.id);
    expect((await r.child().run())?.status).toBe('running');
    await expect(recordApproval(r.child(), 'apr_dead', true)).rejects.toMatchObject({ code: 'child_binding_missing' });
  });
});

describe('item 4 / Py F2: a failure handler cleans up first and reads the error only guarded', () => {
  const malformed: [string, () => unknown][] = [
    ['a HarnessError subclass that never ran the base constructor', () => Object.create(HarnessError.prototype) as unknown],
    ['a HarnessError whose code and document getters throw', () => {
      const error = new HarnessError('session_locked', 'm');
      Object.defineProperty(error, 'code', { configurable: true, get: () => { throw new Error('GETTER-SECRET'); } });
      Object.defineProperty(error, 'document', { configurable: true, get: () => { throw new Error('GETTER-SECRET'); } });
      return error;
    }],
    ['an object whose prototype read throws', () => new Proxy({}, { getPrototypeOf: () => { throw new Error('PROTO-SECRET'); } })],
  ];

  for (const [name, make] of malformed) {
    it(`${name}: the run is failed, run.failed emitted, the dead approval still refused, no live hold`, async () => {
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
      await expect(recordApproval(r.child(), 'apr_dead', true)).rejects.toMatchObject({ code: 'child_binding_missing' });
      expect(await r.child().childTurnLive(now(), 300)).toBe(false);
    });
  }

  it('a run whose failRun write throws still emits run.failed', async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    await deadChild(r);
    const failed: HarnessEvent[] = [];
    r.events.listen((event: HarnessEvent) => { if (event.type === 'run.failed' && event.sessionKey === r.child().key()) failed.push(event); });
    failChildRegistry();
    vi.spyOn(Session.prototype, 'failRun').mockRejectedValue(new Error('STORE-SECRET'));

    const result = await r.run(ROOT(), 'Start over');
    vi.restoreAllMocks();

    expect(result.outcome).toBe('failed');
    expect(failed).toHaveLength(1);
    await expect(recordApproval(r.child(), 'apr_dead', true)).rejects.toMatchObject({ code: 'child_binding_missing' });
  });
});

describe("item 3 (TS I-1): only the harness's own codes count as harness codes", () => {
  it('a HarnessError with a code the harness does not define is reported by class name only', async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    await deadChild(r);
    failChildRegistry(new HarnessError('operator.note.db.password.hunter2' as never, 'm'));

    const result = await r.run(ROOT(), 'Start over');
    vi.restoreAllMocks();

    expect(result.outcome).toBe('failed');
    expect(result.text).not.toMatch(/hunter2|operator/);
    expect(result.text).toMatch(/HarnessError/);
    expect(JSON.stringify(await r.ephemeral.get(`${r.child().key()}:ephemeral`))).not.toMatch(/hunter2/);
  });

  it('a harness code is still reported', async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    await deadChild(r);
    failChildRegistry(HarnessError.runNotPermitted('SECRET words'));

    const result = await r.run(ROOT(), 'Start over');
    vi.restoreAllMocks();

    expect(result.text).toMatch(/HarnessError run_not_permitted/);
    expect(result.text).not.toMatch(/SECRET/);
  });
});

describe('item 8: a limit that is present but not a finite non-negative number is a config error', () => {
  const bad: unknown[] = ['NAN', 'INF', '-INF', '0,50', '60', '', NaN, Infinity, -Infinity, -1, true, {}, []];
  const label = (value: unknown): string => (typeof value === 'number' ? String(value) : JSON.stringify(value));

  for (const value of bad) {
    it(`RunBudget.fromConfig refuses ${label(value)} for max_cost_usd, max_seconds and max_steps`, () => {
      for (const key of ['max_cost_usd', 'max_seconds', 'max_steps']) {
        expect(() => RunBudget.fromConfig({ [key]: value }), key).toThrow(TypeError);
      }
    });
  }

  it('the runtime config refuses them for spend_ceiling, max_steps_ceiling and lock_ttl', () => {
    for (const value of bad) {
      for (const key of ['spend_ceiling', 'max_steps_ceiling', 'lock_ttl']) {
        expect(() => new AgentRuntime({
          modes: new ModeRegistry({ default: 'm', modes: { m: { system_prompt: '', tools: [] } as ModeConfig } }),
          tools: new ToolRegistry(),
          client: async () => ({ text: '', finishReason: 'stop' }),
          config: { [key]: value } as unknown as RuntimeConfig,
        }), `${key}=${label(value)}`).toThrow(TypeError);
      }
    }
  });

  it('a mode refuses them for max_steps', () => {
    for (const value of bad) {
      const modes = new ModeRegistry({ default: 'm', modes: { m: { system_prompt: '', tools: [], max_steps: value } as unknown as ModeConfig } });
      expect(() => modes.resolve('m'), label(value)).toThrow();
    }
  });

  it('a plan budget refuses them', () => {
    for (const value of bad) {
      expect(() => PlanBudget.settings({ mode: 'shared', max_steps: value }), label(value)).toThrow(PlanBudgetInvalid);
      expect(() => PlanBudget.settings({ mode: 'per_step', max_cost_usd: value }), label(value)).toThrow(PlanBudgetInvalid);
    }
  });
});
