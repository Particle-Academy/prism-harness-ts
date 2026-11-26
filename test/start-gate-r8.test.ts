import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  HarnessError,
  HarnessEvents,
  MemorySessionStore,
  ModeRegistry,
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
import { StartRefused } from '../src/run-starts.js';

// G-387 round 8: the round-7 review findings, at the public seams (the
// subagent TOOL inside a parent turn, SubagentRunner.run, send and
// recordApproval). No network; every store is in memory.

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

describe('item 1: a runner start is narrowed from what the gate READ', () => {
  for (const [where, spend] of [
    ['inside the child write (b)', 'lock'],
    ['while the run begins (c)', 'begin'],
  ] as const) {
    it(`steps the context spends after the read count (${where}): the child gets no more than the read allowed`, async () => {
      const r = new Rig();
      await r.session().usingMode('lead');
      const context = RunContext.root('root', new RunBudget(8));
      context.ledger.recordSteps(6);
      const childKey = `${r.child().key()}:durable`;
      if (spend === 'lock') {
        r.durable.onLock = async (key) => {
          if (key !== childKey) return;
          r.durable.onLock = null;
          context.ledger.recordSteps(2);
        };
      } else {
        const beginRun = Session.prototype.beginRun;
        vi.spyOn(Session.prototype, 'beginRun').mockImplementation(async function (this: Session, ...args: Parameters<Session['beginRun']>) {
          if (this.scope.includes('::sub::')) context.ledger.recordSteps(2);
          return beginRun.apply(this, args);
        });
      }

      const result = await r.run(context);

      expect(r.childRequests()).toBe(0);
      expect(context.ledger.steps).toBeLessThanOrEqual(8);
      expect(result.outcome).toBe('exhausted');
    });
  }
});

describe('item 2: commit (b) is undone by any failure before the task row', () => {
  const failures: [string, () => void][] = [
    ['PA: the tool registry throws for the child', () => {
      const resolve = ToolRegistry.prototype.resolveDetailed;
      vi.spyOn(ToolRegistry.prototype, 'resolveDetailed').mockImplementation(function (this: ToolRegistry, ...args: Parameters<ToolRegistry['resolveDetailed']>) {
        if (args[1]?.scope.includes('::sub::') === true) throw new Error('REGISTRY-SECRET');
        return resolve.apply(this, args);
      });
    }],
    ['PA2: the run-begin write throws for the child', () => {
      const beginRun = Session.prototype.beginRun;
      vi.spyOn(Session.prototype, 'beginRun').mockImplementation(function (this: Session, ...args: Parameters<Session['beginRun']>) {
        if (this.scope.includes('::sub::')) throw new Error('STORE-SECRET');
        return beginRun.apply(this, args);
      });
    }],
  ];

  for (const [name, fail] of failures) {
    it(`${name}: the dead approval stays refused, no live hold remains, the child's records are byte-identical`, async () => {
      const r = new Rig();
      await r.session().usingMode('lead');
      await deadChild(r);
      const before = await durableOf(r, r.child());
      const modeBefore = await r.child().mode();
      fail();

      const result = await r.run(RunContext.root('root', new RunBudget(8)), 'Start over');
      vi.restoreAllMocks();

      expect(result.outcome).toBe('failed');
      expect(result.text).not.toMatch(/SECRET/);
      expect(r.childRequests()).toBe(0);
      expect(await durableOf(r, r.child())).toBe(before);
      expect(await r.child().mode()).toBe(modeBefore);
      await expect(recordApproval(r.child(), 'apr_dead', true)).rejects.toMatchObject({ code: 'child_binding_missing' });
      expect(await r.child().childTurnLive(Math.floor(Date.now() / 1000), 300)).toBe(false);

      // Re-delegation is not refused as busy.
      r.text('Child done.');
      const again = await r.run(RunContext.root('root', new RunBudget(8)), 'Start over');
      expect(again.outcome).toBe('completed');
    });
  }

  it('a fresh child whose run cannot begin is left with no binding and no mode', async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    const before = await durableOf(r, r.child());
    const beginRun = Session.prototype.beginRun;
    vi.spyOn(Session.prototype, 'beginRun').mockImplementation(function (this: Session, ...args: Parameters<Session['beginRun']>) {
      if (this.scope.includes('::sub::')) throw new Error('STORE-SECRET');
      return beginRun.apply(this, args);
    });

    const result = await r.run(RunContext.root('root', new RunBudget(8)));
    vi.restoreAllMocks();

    expect(result.outcome).toBe('failed');
    expect(await durableOf(r, r.child())).toBe(before);
    expect(await r.child().mode()).toBeNull();
    expect(await r.child().childBinding()).toBeNull();
  });

  it('a later writer of the child record is never clobbered: only the start\'s own hold is settled', async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    await finishedChild(r);
    const childKey = `${r.child().key()}:durable`;
    const beginRun = Session.prototype.beginRun;
    vi.spyOn(Session.prototype, 'beginRun').mockImplementation(async function (this: Session, ...args: Parameters<Session['beginRun']>) {
      if (this.scope.includes('::sub::')) {
        // Another writer changes the child's record after (b).
        const stored = (await r.durable.get(childKey))!;
        await r.durable.put(childKey, { ...stored, capabilities: { later: { by: 'host' } } });
        throw new Error('STORE-SECRET');
      }
      return beginRun.apply(this, args);
    });

    const result = await r.run(RunContext.root('root', new RunBudget(8)), 'Again.');
    vi.restoreAllMocks();

    expect(result.outcome).toBe('failed');
    const stored = (await r.durable.get(childKey))!;
    expect(stored.capabilities).toEqual({ later: { by: 'host' } });
    expect(await r.child().childTurnLive(Math.floor(Date.now() / 1000), 300)).toBe(false);
  });
});

describe('item 3: pending approvals are re-checked under the child lock', () => {
  it('a host turn that parks an answerable approval between the gate read and (b) refuses the start; the approval stays pending', async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    await finishedChild(r);
    const childKey = `${r.child().key()}:durable`;
    let approval = '';
    let after = '';
    r.durable.onLock = async (key) => {
      if (key !== childKey) return;
      r.durable.onLock = null;
      // A host turn on the child runs and parks an approval a person can answer.
      const parked = await r.runtime.send(r.child(), 'Host question');
      approval = parked.pendingApprovals[0]!.id;
      after = JSON.stringify([await r.durable.get(childKey), await r.durable.get(`${r.child().key()}:thread`), await r.child().mode()]);
    };

    const result = await r.run(RunContext.root('root', new RunBudget(8)), 'Start over');

    expect(approval).not.toBe('');
    expect(result.outcome).toBe('denied');
    expect(result.text).toMatch(/waiting on a person/);
    expect(JSON.stringify([await r.durable.get(childKey), await r.durable.get(`${r.child().key()}:thread`), await r.child().mode()])).toBe(after);
    await expect(recordApproval(r.child(), approval, true)).resolves.toBeUndefined();
  });
});

describe('item 4: no foreign text reaches the parent model', () => {
  for (const timing of ['after the child run began', 'between the gate and the commit'] as const) {
    it(`a host cancel reason landing ${timing} is in neither the parent's tool result nor its next provider request`, async () => {
      const r = new Rig();
      await r.session().usingMode('lead');
      const context = RunContext.root('root', new RunBudget(20));
      const childKey = `${r.child().key()}:durable`;
      if (timing === 'after the child run began') {
        const beginRun = Session.prototype.beginRun;
        vi.spyOn(Session.prototype, 'beginRun').mockImplementation(async function (this: Session, ...args: Parameters<Session['beginRun']>) {
          const begun = await beginRun.apply(this, args);
          if (this.scope.includes('::sub::')) context.ledger.cancel('SECRET operator note');
          return begun;
        });
      } else {
        r.durable.onLock = async (key) => {
          if (key !== childKey) return;
          r.durable.onLock = null;
          context.ledger.cancel('SECRET operator note');
        };
      }
      r.calls([['a', { task: 'Work.' }]]);

      await r.runtime.send(r.session(), 'Go', undefined, context);
      vi.restoreAllMocks();
      expect(await parentText(r)).not.toContain('SECRET');

      const from = r.requests.length;
      r.text('Next done.');
      await r.runtime.send(r.session(), 'Next question');
      expect(r.requests.length).toBeGreaterThan(from);
      expect(r.requests.slice(from).map((request) => JSON.stringify(request)).join('\n')).not.toContain('SECRET');
    });
  }

  it('a forged StartRefused is mapped through the fixed outcomes and reasons', async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    vi.spyOn(Session.prototype, 'childTurnLive').mockRejectedValue(new StartRefused('SECRET' as never, 'completed' as never));

    const result = await r.run(RunContext.root('root', new RunBudget(8)));

    expect(result.outcome).toBe('failed');
    expect(result.text).not.toMatch(/SECRET|undefined/);
  });

  it("an error's code is read once, guarded: a changing getter cannot choose the text", async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    const error = new HarnessError('agent_spend_ceiling_exceeded', 'm');
    let reads = 0;
    Object.defineProperty(error, 'code', { configurable: true, get: () => (++reads === 1 ? 'agent_spend_ceiling_exceeded' : 'secret_operator_note') });
    vi.spyOn(Session.prototype, 'childTurnLive').mockRejectedValue(error);

    const result = await r.run(RunContext.root('root', new RunBudget(8)));

    expect(reads).toBeLessThanOrEqual(1);
    expect(result.text).not.toMatch(/secret/i);
  });

  it("a throwing code getter is framed as failed, never a rejection", async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    const error = new HarnessError('x', 'm');
    Object.defineProperty(error, 'code', { configurable: true, get: () => { throw new Error('GETTER-SECRET'); } });
    vi.spyOn(Session.prototype, 'childTurnLive').mockRejectedValue(error);

    const result = await r.run(RunContext.root('root', new RunBudget(8)));

    expect(result.outcome).toBe('failed');
    expect(result.text).not.toMatch(/SECRET/);
  });

  it('a spend ceiling or unpriced stop raised during the child run reads as fixed harness text', async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    vi.spyOn(Session.prototype, 'childTurnLive').mockRejectedValue(new HarnessError('budget_unpriced', 'SECRET host config words'));

    const result = await r.run(RunContext.root('root', new RunBudget(8)));

    expect(result.outcome).toBe('exhausted');
    expect(result.text).not.toMatch(/SECRET/);
  });
});

describe('item 5: a start releases only the hold it took', () => {
  it('a re-entrant start from a RunStarted listener that fails does not remove the first start\'s hold', async () => {
    const r = new Rig({ declared: { a: 2, b: 2 } });
    await r.session().usingMode('lead');
    const childA = r.child('a').key();
    let second: Promise<string> | null = null;
    let holdsDuringA: unknown = 'unread';
    const beginRun = Session.prototype.beginRun;
    vi.spyOn(Session.prototype, 'beginRun').mockImplementation(function (this: Session, ...args: Parameters<Session['beginRun']>) {
      if (this.scope.endsWith('::sub::b')) throw new Error('STORE-SECRET');
      return beginRun.apply(this, args);
    });
    r.events.listen((event: HarnessEvent) => {
      if (event.type !== 'run.started' || event.sessionKey !== childA || second !== null) return;
      second = r.tools.b!.handle({ task: 'Other.' });
      r.beforeRequest = async () => {
        await second;
        const key = r.durable.writes.find((written) => written.includes(':run-tree:'));
        holdsDuringA = key === undefined ? null : ((await r.durable.get(key)) as unknown as { holds: Record<string, unknown> }).holds;
      };
    });
    r.calls([['a', { task: 'Work.' }]]);
    r.text('Parent done.');

    await r.runtime.send(r.session(), 'Go');

    expect(second).not.toBeNull();
    expect(await second).toMatch(/it failed/);
    expect(holdsDuringA).not.toBe('unread');
    expect(holdsDuringA).not.toBeNull();
    expect(Object.keys(holdsDuringA as object)).toHaveLength(1);
  });
});

describe('item 6: the answerability read fails closed', () => {
  it('a store read inside the answerability check that throws another HarnessError refuses the start, nothing written', async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    const approval = await parkedChild(r);
    const before = await durableOf(r, r.child());
    const threadBefore = JSON.stringify(await r.durable.get(`${r.child().key()}:thread`));
    const childBinding = Session.prototype.childBinding;
    vi.spyOn(Session.prototype, 'childBinding').mockImplementation(function (this: Session) {
      if (new Error().stack?.includes('decisionsAnswerable') === true || new Error().stack?.includes('answerability') === true) {
        return Promise.reject(new HarnessError('session_locked', 'STORE-SECRET'));
      }
      return childBinding.apply(this);
    });

    const result = await r.run(RunContext.root('root', new RunBudget(8)), 'Start over');
    vi.restoreAllMocks();

    expect(result.outcome).toBe('failed');
    expect(result.text).not.toMatch(/SECRET/);
    expect(await durableOf(r, r.child())).toBe(before);
    expect(JSON.stringify(await r.durable.get(`${r.child().key()}:thread`))).toBe(threadBefore);
    await expect(recordApproval(r.child(), approval, true)).resolves.toBeUndefined();
  });
});

describe('item 7: non-finite limits fail closed', () => {
  for (const [name, make] of [
    ['max_cost_usd NaN', () => new RunBudget(8, Number.NaN)],
    ['max_cost_usd Infinity', () => new RunBudget(8, Number.POSITIVE_INFINITY)],
    ['max_cost_usd negative', () => new RunBudget(8, -1)],
    ['max_seconds NaN', () => new RunBudget(8, null, Number.NaN)],
    ['max_seconds Infinity', () => new RunBudget(8, null, Number.POSITIVE_INFINITY)],
    ['max_seconds negative', () => new RunBudget(8, null, -1)],
    ['config max_seconds NaN', () => RunBudget.fromConfig({ max_seconds: Number.NaN })],
    ['config max_cost_usd Infinity', () => RunBudget.fromConfig({ max_cost_usd: Number.POSITIVE_INFINITY })],
  ] as const) {
    it(`RunBudget refuses ${name}`, () => {
      expect(make).toThrow(TypeError);
    });
  }

  for (const [name, forge] of [
    ['max_seconds NaN', (b: RunBudget) => { (b as { maxSeconds: number | null }).maxSeconds = Number.NaN; }],
    ['max_seconds Infinity', (b: RunBudget) => { (b as { maxSeconds: number | null }).maxSeconds = Number.POSITIVE_INFINITY; }],
    ['max_cost_usd NaN', (b: RunBudget) => { (b as { maxCostUsd: number | null }).maxCostUsd = Number.NaN; }],
    ['max_cost_usd Infinity', (b: RunBudget) => { (b as { maxCostUsd: number | null }).maxCostUsd = Number.POSITIVE_INFINITY; }],
  ] as const) {
    it(`a forged ${name} on the runner context lets the child take no model step, nothing written`, async () => {
      const r = new Rig();
      await r.session().usingMode('lead');
      const before = await durableOf(r, r.child());
      const context = RunContext.root('root', new RunBudget(8, 1, 60));
      forge(context.budget);

      const result = await r.run(context);

      expect(r.childRequests()).toBe(0);
      expect(result.outcome).not.toBe('completed');
      expect(await durableOf(r, r.child())).toBe(before);
    });
  }
});

describe('item 8: exception text never reaches RunFailed or a run row', () => {
  it('a provider exception fails a root run: no exception text in the event, the run row or any run-state entry', async () => {
    const r = new Rig({ client: async () => { throw new Error('SECRET exception text'); } });
    const failed: HarnessEvent[] = [];
    r.events.listen((event) => { if (event.type === 'run.failed') failed.push(event); });

    await expect(r.runtime.send(r.session(), 'Go')).rejects.toThrow('SECRET exception text');

    expect(failed).toHaveLength(1);
    expect(JSON.stringify(failed)).not.toContain('SECRET');
    const run = await r.session().run();
    expect(run?.status).toBe('failed');
    expect(String(run?.failure)).toMatch(/Error/);
    expect(await parentText(r)).not.toContain('SECRET');
    expect(JSON.stringify(await r.durable.get(`${r.session().key()}:run-trees`))).not.toContain('SECRET');
  });

  it('a HarnessError keeps its code and loses its message', async () => {
    const r = new Rig({ client: async () => { throw new HarnessError('session_locked', 'SECRET exception text'); } });
    const failed: HarnessEvent[] = [];
    r.events.listen((event) => { if (event.type === 'run.failed') failed.push(event); });

    await expect(r.runtime.send(r.session(), 'Go')).rejects.toThrow();

    expect(JSON.stringify(failed)).not.toContain('SECRET');
    expect(JSON.stringify(failed)).toContain('session_locked');
    expect(String((await r.session().run())?.failure)).toContain('session_locked');
  });

  it('a provider exception in a child run leaves no exception text on the child run row', async () => {
    const r = new Rig({ client: async (request) => { if (request.systemPrompt.startsWith('Work.')) throw new Error('SECRET exception text'); return null; } });
    await r.session().usingMode('lead');
    const failed: HarnessEvent[] = [];
    r.events.listen((event) => { if (event.type === 'run.failed') failed.push(event); });

    const result = await r.run(RunContext.root('root', new RunBudget(8)));

    expect(result.outcome).toBe('failed');
    expect(failed).toHaveLength(1);
    expect(JSON.stringify(failed)).not.toContain('SECRET');
    expect(JSON.stringify(await r.ephemeral.get(`${r.child().key()}:ephemeral`))).not.toContain('SECRET');
  });
});
