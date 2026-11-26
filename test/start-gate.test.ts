import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  MAX_DEPTH,
  MemorySessionStore,
  ModeRegistry,
  PlanBudget,
  PrismHarness,
  RunBudget,
  RunContext,
  RunLedger,
  Session,
  SubagentRunner,
  ToolRegistry,
  assistantRow,
  recordApproval,
  type Durability,
  type JsonObject,
  type LlmRequest,
  type LlmResponse,
  type ModeConfig,
  type ModeRegistryConfig,
  type RuntimeConfig,
} from '../src/index.js';

// G-387 round 7: ONE start gate for a runner start. A runner start reads every
// stop condition first (writing nothing), asks one gate, and only on "start"
// commits: (a) the tree hold, (b) the child's records in one locked write,
// (c) the run. Driven through the public seams only: the subagent TOOL inside
// a parent turn, and SubagentRunner.run. Every refused start asserts (1) the
// framed outcome, (2) zero child model requests, (3) the child's durable
// records byte-identical, (4) no run state or run history for it, (5) the
// parent unchanged. No network; every store is in memory.

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
  lead?: number;
  declared?: Record<string, number>;
  config?: RuntimeConfig;
  /** The run context the host wires each subagent tool with. */
  toolContext?: () => RunContext;
}

class Rig {
  readonly queue: LlmResponse[] = [];
  readonly requests: LlmRequest[] = [];
  readonly ran: Record<string, number> = {};
  readonly durable = new LoggedStore('durable');
  readonly ephemeral = new LoggedStore('volatile');
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
          system_prompt: 'Lead.', tools: ['echo', ...Object.keys(declared)], max_steps: this.options.lead ?? 8,
          requires_approval: ['echo'],
          subagents: Object.fromEntries(Object.entries(declared).map(([name, steps]) => [name, { mode: 'work', max_steps: steps }])),
        } as ModeConfig,
        work: { system_prompt: 'Work.', tools: ['echo'], max_steps: 8, requires_approval: ['echo'] } as ModeConfig,
      },
    });
    const tools = new ToolRegistry()
      .register({ name: 'echo', handle: () => { this.ran.echo = (this.ran.echo ?? 0) + 1; return 'ok'; } });
    this.runtime = new AgentRuntime({
      modes: this.modes, tools, ...(this.options.config === undefined ? {} : { config: this.options.config }),
      client: async (request) => {
        this.requests.push(request);
        const hook = this.beforeRequest;
        this.beforeRequest = null;
        if (hook !== null) await hook();
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
      tools.register(new SubagentRunner(this.harness, this.runtime)
        .tool(this.modes.resolve('lead').subagents[name]!, this.session(), this.options.toolContext?.() ?? RunContext.root('host', new RunBudget(100)), 'host_run'));
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

  /** Every key written since `mark` (an index into both write logs). */
  writtenSince(mark: { durable: number; ephemeral: number }): string[] {
    return [...this.durable.writes.slice(mark.durable), ...this.ephemeral.writes.slice(mark.ephemeral)];
  }

  mark(): { durable: number; ephemeral: number } {
    return { durable: this.durable.writes.length, ephemeral: this.ephemeral.writes.length };
  }
}

/** Every record a session keeps, as stored text: durable, ephemeral (mode, run state), run history and thread. */
async function snapshot(r: Rig, session: Session): Promise<string> {
  const key = session.key();

  return JSON.stringify([
    await r.durable.get(`${key}:durable`),
    await r.ephemeral.get(`${key}:ephemeral`),
    await r.durable.get(`${key}:run-trees`),
    await r.durable.get(`${key}:thread`),
  ]);
}

/** Keys of `session`'s own records among `keys` (a child's address extends its parent's, so matched exactly). */
function ownKeys(keys: readonly string[], session: Session): string[] {
  const key = session.key();

  return keys.filter((written) => [':durable', ':ephemeral', ':run-trees', ':thread'].some((suffix) => written === `${key}${suffix}`));
}

function runTreeKeys(keys: readonly string[]): string[] {
  return [...new Set(keys.filter((written) => written.includes(':run-tree:')))];
}

/** Advance the clock every Date.now() reads by `seconds`, from now on. */
function advanceClock(seconds: number): void {
  const now = Date.now;
  const shifted = now() + seconds * 1000;
  const start = now();
  vi.spyOn(Date, 'now').mockImplementation(() => shifted + (now() - start));
}

/** What the parent's model was sent after its first request, as text. */
function sentAfter(r: Rig, from: number): string {
  return r.requests.slice(from).map((request) => JSON.stringify(request.messages)).join('\n');
}

/** A child the runner started, parked on an approval a person can still answer. */
async function parkedChild(r: Rig): Promise<string> {
  await r.run(RunContext.root('root', new RunBudget(8)));
  const rows = (await r.child().thread().messages()).map((entry) => entry.message);
  const requests = rows.flatMap((row) => Array.isArray(row.tool_approval_requests) ? row.tool_approval_requests as JsonObject[] : []);

  return requests[requests.length - 1]!.approval_id as string;
}

/** A child that ran once and finished, so it has a saved ceiling and a count. */
async function finishedChild(r: Rig): Promise<void> {
  r.text('Child done.');
  const result = await r.run(RunContext.root('root', new RunBudget(8)));
  expect(result.outcome).toBe('completed');
}

/**
 * The common assertions on a refused start: the framed outcome, zero child
 * requests, the child's records byte-identical and unwritten, no run state for
 * it, and (runner direct) the parent's records unwritten.
 */
async function refusedNothingWritten(r: Rig, before: string, mark: { durable: number; ephemeral: number }, childRequestsBefore: number, parentDirect: boolean): Promise<void> {
  expect(r.childRequests()).toBe(childRequestsBefore);
  expect(await snapshot(r, r.child())).toBe(before);
  expect(ownKeys(r.writtenSince(mark), r.child())).toEqual([]);
  if (parentDirect) expect(ownKeys(r.writtenSince(mark), r.session())).toEqual([]);
  expect(runTreeKeys(r.writtenSince(mark))).toEqual([]);
}

describe('each gate code refuses a runner start with nothing written, through SubagentRunner.run', () => {
  const cases: [string, string, string | RegExp, (r: Rig) => Promise<{ context: RunContext; task?: unknown }>][] = [
    ['task_invalid', 'denied', 'the task is not text', async () => ({ context: RunContext.root('root', new RunBudget(8)), task: 42 })],
    ['task_blank', 'denied', 'the task is empty', async () => ({ context: RunContext.root('root', new RunBudget(8)), task: '　​ \t' })],
    ['too_deep', 'denied', `maximum depth of ${MAX_DEPTH}`, async () => ({ context: new RunContext(RunLedger.start('deep'), new RunBudget(8), 'p', MAX_DEPTH - 1) })],
    ['cancelled', 'cancelled', /cancelled/, async () => { const context = RunContext.root('root', new RunBudget(8)); context.ledger.cancel('the host words'); return { context }; }],
    ['time', 'exhausted', /no time left/, async () => { const context = RunContext.root('root', new RunBudget(8, null, 60)); advanceClock(120); return { context }; }],
    ['steps', 'exhausted', /no steps left/, async () => { const context = RunContext.root('root', new RunBudget(2)); context.ledger.recordSteps(2); return { context }; }],
    ['cost', 'exhausted', /cost budget is spent/, async () => { const context = RunContext.root('root', new RunBudget(8, 1)); context.ledger.recordCost(1); return { context }; }],
    ['cost_unenforceable', 'exhausted', /cannot be enforced/, async () => { const context = RunContext.root('root', new RunBudget(8, 1)); context.ledger.recordCost(null); return { context }; }],
    ['child_busy', 'denied', 'a turn on it is still running', async (r) => {
      await finishedChild(r);
      const key = `${r.child().key()}:durable`;
      const stored = (await r.durable.get(key))!;
      await r.durable.put(key, { ...stored, child_steps: { steps: 0, holds: { run_live: { steps: 1, at: Math.floor(Date.now() / 1000) } }, charged: {} } });
      return { context: RunContext.root('root', new RunBudget(8)) };
    }],
    ['child_waiting', 'denied', 'waiting on a person', async (r) => {
      await parkedChild(r);
      return { context: RunContext.root('root', new RunBudget(8)) };
    }],
  ];

  for (const [code, outcome, reason, setup] of cases) {
    it(`${code}: ${outcome}, nothing written on the child or the parent`, async () => {
      const r = new Rig();
      await r.session().usingMode('lead');
      const { context, task } = await setup(r);
      const before = await snapshot(r, r.child());
      const parentBefore = await snapshot(r, r.session());
      const childRequests = r.childRequests();
      const mark = r.mark();

      const result = await r.run(context, task ?? 'Work.');

      expect(result.outcome).toBe(outcome);
      expect(result.text).toMatch(typeof reason === 'string' ? new RegExp(reason.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')) : reason);
      expect(result.text).not.toContain('the host words');
      await refusedNothingWritten(r, before, mark, childRequests, true);
      expect(await snapshot(r, r.session())).toBe(parentBefore);
    });
  }
});

describe('each gate code refuses a runner start with nothing written, through the subagent tool inside a parent turn', () => {
  const cases: [string, string, RegExp, RigOptions, JsonObject, ((r: Rig) => Promise<void>) | null][] = [
    ['task_invalid', 'denied', /the task is not text/, {}, { task: 42 }, null],
    ['task_blank', 'denied', /the task is empty/, {}, { task: ' ﻿' }, null],
    ['cancelled (host context)', 'cancelled', /cancelled/, { toolContext: () => { const c = RunContext.root('host', new RunBudget(100)); c.ledger.cancel('the host words'); return c; } }, { task: 'Work.' }, null],
    ['time (host context)', 'exhausted', /no time left/, { toolContext: () => RunContext.root('host', new RunBudget(100, null, 60)) }, { task: 'Work.' }, async () => { advanceClock(120); }],
    ['steps (host context)', 'exhausted', /no steps left/, { toolContext: () => RunContext.root('host', new RunBudget(0)) }, { task: 'Work.' }, null],
    ['cost (host context)', 'exhausted', /cost budget is spent/, { toolContext: () => { const c = RunContext.root('host', new RunBudget(100, 1)); c.ledger.recordCost(2); return c; } }, { task: 'Work.' }, null],
    ['cost_unenforceable (host context)', 'exhausted', /cannot be enforced/, { toolContext: () => { const c = RunContext.root('host', new RunBudget(100, 1)); c.ledger.recordCost(null); return c; } }, { task: 'Work.' }, null],
    ['child_busy', 'denied', /a turn on it is still running/, {}, { task: 'Work.' }, async (r) => {
      await finishedChild(r);
      const key = `${r.child().key()}:durable`;
      const stored = (await r.durable.get(key))!;
      await r.durable.put(key, { ...stored, child_steps: { steps: 0, holds: { run_live: { steps: 1, at: Math.floor(Date.now() / 1000) } }, charged: {} } });
    }],
    ['child_waiting', 'denied', /waiting on a person/, {}, { task: 'Work.' }, async (r) => { await parkedChild(r); }],
  ];

  for (const [code, outcome, reason, options, args, setup] of cases) {
    it(`${code}: the parent's model reads the framed ${outcome} result; nothing written on the child`, async () => {
      const r = new Rig(options);
      if (setup !== null) await setup(r);
      const before = await snapshot(r, r.child());
      const childRequests = r.childRequests();
      const from = r.requests.length;
      r.calls([['a', args]]);
      r.text('Parent done.');
      const mark = r.mark();

      const response = await r.runtime.send(r.session(), 'Go');

      expect(response.text).toBe('Parent done.');
      const sent = sentAfter(r, from + 1);
      expect(sent).toMatch(reason);
      expect(sent).toContain('subagent did not run');
      expect(sent).not.toContain('the host words');
      await refusedNothingWritten(r, before, mark, childRequests, false);
      // Outcome as the parent's run steps see it: a gate, never a child failure.
      expect(outcome).toMatch(/denied|cancelled|exhausted/);
    });
  }

  it('too_deep: a parent turn at the last depth delegates nothing', async () => {
    const r = new Rig();
    const before = await snapshot(r, r.child());
    r.calls([['a', { task: 'Work.' }]]);
    r.text('Parent done.');
    const mark = r.mark();

    const response = await r.runtime.send(r.session(), 'Go', undefined, new RunContext(RunLedger.start('deep'), new RunBudget(8), 'p', MAX_DEPTH - 1));

    expect(response.text).toBe('Parent done.');
    expect(sentAfter(r, 1)).toContain(`maximum depth of ${MAX_DEPTH}`);
    await refusedNothingWritten(r, before, mark, 0, false);
  });

  it('tree: a run tree spent before the delegation refuses it before the child is touched', async () => {
    const r = new Rig();
    const before = await snapshot(r, r.child());
    r.calls([['a', { task: 'Work.' }]]);
    r.text('Parent done.');
    // A sibling in another worker spent the tree the parent's run opened.
    let treeKey = '';
    r.beforeRequest = async () => {
      const run = (await r.session().run())!;
      treeKey = `${r.session().key()}:run-tree:${run.id}`;
      await r.durable.put(treeKey, { ...PlanBudget.open(run.id, { mode: 'shared', max_steps: 1, max_cost_micros: null }), steps: 1 } as unknown as JsonObject);
    };
    const mark = r.mark();

    const response = await r.runtime.send(r.session(), 'Go');

    expect(response.text).toBe('Parent done.');
    expect(sentAfter(r, 1)).toMatch(/subagent did not run: its run tree's budget is spent/);
    expect(r.childRequests()).toBe(0);
    expect(await snapshot(r, r.child())).toBe(before);
    expect(ownKeys(r.writtenSince(mark), r.child())).toEqual([]);
    // The tree record is the sibling's, untouched by the refused start.
    expect(r.durable.writes.slice(mark.durable).filter((key) => key === treeKey)).toHaveLength(1);
  });
});

/** A named-agent world: planner delegates to researcher, whose spend ceiling is 0.05 USD. */
function namedWorld(): { r: { requests: LlmRequest[]; durable: LoggedStore; ephemeral: LoggedStore }; harness: PrismHarness; runtime: AgentRuntime; parent: Session; child: () => Session; runner: SubagentRunner; context: RunContext; registry: ModeRegistry; tools: ToolRegistry } {
  const requests: LlmRequest[] = [];
  const durable = new LoggedStore('durable');
  const ephemeral = new LoggedStore('volatile');
  const modes: Record<string, ModeRegistryConfig> = {
    planner: { owner: 'app', modes: { plan: { system_prompt: 'Plan.', max_steps: 4, tools: ['research'], agents: ['researcher'], subagents: { research: { mode: 'research', agent: 'researcher' } } } } },
    researcher: { owner: 'app', modes: { research: { system_prompt: 'Research.', max_steps: 4, tools: [] } } },
  };
  const registry = new ModeRegistry({ modes: {}, agents: modes });
  const tools = new ToolRegistry();
  let call = 0;
  const runtime = new AgentRuntime({
    config: { provider: 'anthropic', model: 'global-model' },
    agents: { planner: {}, researcher: { spend_ceiling: 0.05 } },
    modes: registry,
    tools,
    client: async (request) => {
      const index = call++;
      requests.push(request);
      if (index === 0 && request.systemPrompt.startsWith('Plan.')) {
        return { text: '', finishReason: 'tool_calls', costUsd: 0.01, toolCalls: [{ id: 'delegate-call', name: 'research', arguments: { task: 't' } }] };
      }

      return { text: `answer ${index}`, finishReason: 'stop', costUsd: 0.01 };
    },
  });
  const harness = new PrismHarness({
    runtime,
    drivers: { memory: () => ephemeral, files: () => durable },
    stores: { ephemeral: 'memory', durable: 'files' },
  });
  const runner = new SubagentRunner(harness, runtime);
  const context = RunContext.root('root', new RunBudget(8));
  const parent = harness.pending({ type: 'User', id: 1 }, 'planner').session('support');
  const child = (): Session => harness.pending({ type: 'User', id: 1 }, 'researcher').session('support::sub::research');
  tools.forAgent('planner').register({
    name: 'research',
    handle: async (args) => await runner.tool(registry.forAgent('planner').resolve('plan').subagents.research!, parent, context, 'root').handle(args),
  }, 'app');

  return { r: { requests, durable, ephemeral }, harness, runtime, parent, child, runner, context, registry, tools };
}

async function namedSnapshot(world: ReturnType<typeof namedWorld>, session: Session): Promise<string> {
  const key = session.key();

  return JSON.stringify([
    await world.r.durable.get(`${key}:durable`),
    await world.r.ephemeral.get(`${key}:ephemeral`),
    await world.r.durable.get(`${key}:run-trees`),
    await world.r.durable.get(`${key}:thread`),
  ]);
}

describe("the child agent's own spend ceiling is a cost stop of the gate", () => {
  for (const [code, record] of [
    ['cost', (ledger: RunLedger) => { ledger.recordCost(0.05, 'researcher'); }],
    ['cost_unenforceable', (ledger: RunLedger) => { ledger.recordCost(null, 'researcher'); }],
  ] as const) {
    it(`${code}: through the tool inside a parent turn, the researcher is refused before it begins a run`, async () => {
      const world = namedWorld();
      await world.parent.usingMode('plan');
      record(world.context.ledger);
      const before = await namedSnapshot(world, world.child());
      const mark = world.r.durable.writes.length, markE = world.r.ephemeral.writes.length;

      const response = await world.runtime.send(world.parent, 'go', undefined, world.context);

      expect(response.text).toBe('answer 1');
      expect(world.r.requests.filter((request) => request.systemPrompt.startsWith('Research.'))).toHaveLength(0);
      expect(JSON.stringify(world.r.requests[1]!.messages)).toContain('subagent did not run');
      expect(await namedSnapshot(world, world.child())).toBe(before);
      const written = [...world.r.durable.writes.slice(mark), ...world.r.ephemeral.writes.slice(markE)];
      expect(written.filter((key) => key.startsWith(`${world.child().key()}:`))).toEqual([]);
    });

    it(`${code}: through SubagentRunner.run, exhausted with nothing written`, async () => {
      const world = namedWorld();
      await world.parent.usingMode('plan');
      record(world.context.ledger);
      const before = await namedSnapshot(world, world.child());
      const parentBefore = await namedSnapshot(world, world.parent);
      const mark = world.r.durable.writes.length, markE = world.r.ephemeral.writes.length;

      const result = await world.runner.run(world.registry.forAgent('planner').resolve('plan').subagents.research!, world.parent, world.context, 'root', 't');

      expect(result.outcome).toBe('exhausted');
      expect(world.r.requests).toHaveLength(0);
      expect(await namedSnapshot(world, world.child())).toBe(before);
      expect(await namedSnapshot(world, world.parent)).toBe(parentBefore);
      const written = [...world.r.durable.writes.slice(mark), ...world.r.ephemeral.writes.slice(markE)];
      expect(written).toEqual([]);
    });
  }
});

describe('the commit races (BRIEF-R7 step 3)', () => {
  it('a tree spent between the gate read and the hold refuses the start, with nothing written on the child', async () => {
    const r = new Rig();
    const before = await snapshot(r, r.child());
    r.calls([['a', { task: 'Work.' }]]);
    r.text('Parent done.');
    // The gate reads the tree with no lock; the hold is the first locked write
    // on it. A sibling spends the tree in between.
    let treeKey = '';
    r.durable.onLock = async (key) => {
      if (!key.includes(':run-tree:')) return;
      r.durable.onLock = null;
      treeKey = key;
      const run = key.slice(key.lastIndexOf(':run-tree:') + ':run-tree:'.length);
      await r.durable.put(key, { ...PlanBudget.open(run, { mode: 'shared', max_steps: 1, max_cost_micros: null }), steps: 1 } as unknown as JsonObject);
    };
    const mark = r.mark();

    const response = await r.runtime.send(r.session(), 'Go');

    expect(response.text).toBe('Parent done.');
    expect(treeKey).not.toBe('');
    expect(sentAfter(r, 1)).toMatch(/subagent did not run: its run tree's budget is spent/);
    expect(r.childRequests()).toBe(0);
    expect(await snapshot(r, r.child())).toBe(before);
    expect(ownKeys(r.writtenSince(mark), r.child())).toEqual([]);
    // Only the sibling wrote the record: the refused hold wrote nothing.
    expect(r.durable.writes.slice(mark.durable).filter((key) => key === treeKey)).toHaveLength(1);
    expect(((await r.durable.get(treeKey)) as unknown as { holds: object }).holds).toEqual({});
  });

  it('a resume claiming between the gate read and the child write refuses the start and releases the tree hold', async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    await finishedChild(r);
    r.calls([['a', { task: 'Again.' }]]);
    r.text('Parent done.');
    const childKey = `${r.child().key()}:durable`;
    let claimed = '';
    r.durable.onLock = async (key) => {
      if (key !== childKey) return;
      r.durable.onLock = null;
      // A turn on the child that the runner did not start claims its count.
      await r.child().claimChildSteps('run_resume', Math.floor(Date.now() / 1000), 300, () => 1);
      claimed = JSON.stringify(await r.durable.get(childKey));
    };
    const childRequests = r.childRequests();
    const mark = r.mark();

    const response = await r.runtime.send(r.session(), 'Go');

    expect(response.text).toBe('Parent done.');
    expect(claimed).not.toBe('');
    expect(sentAfter(r, childRequests + 1)).toContain('a turn on it is still running');
    expect(r.childRequests()).toBe(childRequests);
    // The child's records are exactly what the resume's claim left: the start wrote nothing.
    expect(JSON.stringify(await r.durable.get(childKey))).toBe(claimed);
    expect(JSON.parse(claimed).child_steps.holds).toHaveProperty('run_resume');
    // No run state for the refused start: the child's run is still the first one, completed.
    expect((await r.child().run())?.status).toBe('completed');
    // The tree hold the start took is released: the record it opened is gone.
    const trees = runTreeKeys(r.writtenSince(mark));
    expect(trees.length).toBeGreaterThan(0);
    for (const key of trees) expect(await r.durable.get(key)).toBeNull();
  });
});

describe('a committed start (BRIEF-R7 step 3 c)', () => {
  it('cancelled after it committed takes no model step and records its task', async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    // A child parked on an approval nobody can answer (its binding is gone).
    const child = r.child();
    await child.rememberChildRefs({ tree: { session: r.session().key(), run: 'run_root' }, planLedger: null });
    await child.usingMode('work');
    await child.thread().record([{ type: 'user', content: 'Work.' }], 'run_child');
    const row = assistantRow('', [{ id: 'e1', name: 'echo', arguments: {} }], {}, [{ approval_id: 'apr_dead', tool_call_id: 'e1', arguments_format: 1 }]) as unknown as JsonObject;
    row.harness_tool_names = ['echo'];
    await child.thread().record([row], 'run_child');
    await expect(recordApproval(r.child(), 'apr_dead', true)).rejects.toMatchObject({ code: 'child_binding_missing' });
    const context = RunContext.root('root', new RunBudget(8));
    // Cancelled once the start committed (the run is being begun).
    const beginRun = Session.prototype.beginRun;
    vi.spyOn(Session.prototype, 'beginRun').mockImplementation(async function (this: Session, ...args: Parameters<Session['beginRun']>) {
      if (this.scope.includes('::sub::')) context.ledger.cancel('stopped by host');
      return beginRun.apply(this, args);
    });

    const result = await r.run(context, 'Start over');
    vi.restoreAllMocks();

    expect(result.outcome).toBe('exhausted');
    expect(r.childRequests()).toBe(0);
    const rows = (await r.child().thread().messages()).map((entry) => entry.message);
    expect(rows.some((message) => message.type === 'user' && message.content === 'Start over')).toBe(true);
    expect((await r.child().run())?.status).toBe('completed');
    expect((await r.child().run())?.finish_reason).toBe('budget_exhausted');
  });
});

describe('new channels considered (round 7)', () => {
  it('a fact read that throws fails closed: failed, class name only, nothing written', async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    const before = await snapshot(r, r.child());
    const mark = r.mark();
    vi.spyOn(Session.prototype, 'childTurnLive').mockRejectedValue(new Error('STORE-SECRET-TEXT'));

    const result = await r.run(RunContext.root('root', new RunBudget(8)));

    expect(result.outcome).toBe('failed');
    expect(result.text).toContain('it failed (Error)');
    expect(result.text).not.toContain('STORE-SECRET-TEXT');
    await refusedNothingWritten(r, before, mark, 0, true);
  });

  it('a cancel together with an answerable approval refuses as cancelled and leaves the approval answerable', async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    const approval = await parkedChild(r);
    const before = await snapshot(r, r.child());
    const mark = r.mark();
    const context = RunContext.root('root', new RunBudget(8));
    context.ledger.cancel('stop');

    const result = await r.run(context, 'Start over');

    expect(result.outcome).toBe('cancelled');
    await refusedNothingWritten(r, before, mark, 1, true);
    await expect(recordApproval(r.child(), approval, true)).resolves.toBeUndefined();
  });

  it('a start whose run cannot begin releases its tree hold and fails as before', async () => {
    const r = new Rig();
    r.calls([['a', { task: 'Work.' }]]);
    r.text('Parent done.');
    const beginRun = Session.prototype.beginRun;
    vi.spyOn(Session.prototype, 'beginRun').mockImplementation(function (this: Session, ...args: Parameters<Session['beginRun']>) {
      if (this.scope.includes('::sub::')) throw new Error('STORE-SECRET-TEXT');
      return beginRun.apply(this, args);
    });
    const mark = r.mark();

    const response = await r.runtime.send(r.session(), 'Go');
    vi.restoreAllMocks();

    expect(response.text).toBe('Parent done.');
    expect(sentAfter(r, 1)).toContain('it failed (Error)');
    expect(r.childRequests()).toBe(0);
    const trees = runTreeKeys(r.writtenSince(mark));
    expect(trees.length).toBeGreaterThan(0);
    for (const key of trees) expect(await r.durable.get(key)).toBeNull();
  });

  it('a hold released after a race is never settled again: the next delegation opens the tree once and the parent settles it once', async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    await finishedChild(r);
    r.calls([['a', { task: 'Again.' }]]);       // refused by the race
    r.text('Child done.');                      // (unused by the refused start)
    const childKey = `${r.child().key()}:durable`;
    r.durable.onLock = async (key) => {
      if (key !== childKey) return;
      r.durable.onLock = null;
      await r.child().claimChildSteps('run_resume', Math.floor(Date.now() / 1000), 300, () => 1);
    };
    // The first queued answer is the parent's delegation; then the parent stops.
    r.queue.splice(1, 1);
    r.text('Parent done.');

    await r.runtime.send(r.session(), 'Go');

    const trees = runTreeKeys(r.durable.writes);
    for (const key of trees) expect(await r.durable.get(key)).toBeNull();
  });

  it('a runner start under a context that only names a runner turn still passes the one gate', async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    const before = await snapshot(r, r.child());
    const mark = r.mark();
    // A host-wired tool whose context's time is spent, used outside any turn.
    const context = RunContext.root('host', new RunBudget(100, null, 60));
    advanceClock(120);
    const tool = new SubagentRunner(r.harness, r.runtime).tool(r.modes.resolve('lead').subagents.a!, r.session(), context, 'host_run');

    const text = await tool.handle({ task: 'Work.' });

    expect(text).toMatch(/did not run: its run has no time left/);
    await refusedNothingWritten(r, before, mark, 0, true);
  });
});
