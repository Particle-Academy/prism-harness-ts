import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  HarnessEvents,
  MemorySessionStore,
  ModeRegistry,
  PlanMachine,
  Plans,
  PrismHarness,
  RunBudget,
  RunContext,
  RunLedger,
  ToolRegistry,
  recordApproval,
  recordDecisions,
  type Durability,
  type HarnessEvent,
  type JsonObject,
  type LlmRequest,
  type LlmResponse,
  type ModeConfig,
  type PlansConfig,
  type Session,
} from '../src/index.js';

// G-366: a plan's budget is the app's choice (Wish, q_fb920047): one shared
// budget for all of a plan's steps, the per-turn budget per step (the
// default), or no plan-level budget. Driven through the public API
// (runtime.send, recordApproval, session.plan().advance()) with a scripted
// model client; fetch is refused and every store is in memory.

beforeEach(() => {
  vi.stubGlobal('fetch', () => { throw new Error('A test tried to use the network.'); });
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

class DurableMemoryStore extends MemorySessionStore {
  /** When set, a write to a plan budget record fails, as a store that is down would. */
  failBudgetWrites = false;

  override durability(): Durability {
    return 'durable';
  }

  override async put(key: string, payload: JsonObject, ttlSeconds: number | null = null): Promise<void> {
    if (this.failBudgetWrites && key.endsWith(':plan-budget')) throw new Error('store down');
    await super.put(key, payload, ttlSeconds);
  }
}

type Call = [string, JsonObject];

interface RigOptions {
  config?: PlansConfig;
  mode?: Record<string, unknown>;
}

class Rig {
  readonly queue: LlmResponse[] = [];
  readonly requests: LlmRequest[] = [];
  readonly events: HarnessEvent[] = [];
  readonly durable = new DurableMemoryStore();
  readonly ephemeral = new MemorySessionStore();
  plans!: Plans;
  runtime!: AgentRuntime;
  harness!: PrismHarness;
  lookups = 0;
  weatherRan = 0;
  onLookup: (() => Promise<void>) | null = null;
  #call = 0;

  constructor(private readonly options: RigOptions = {}) {
    this.restart();
  }

  /** A new process: new Plans, runtime and harness over the same stores. */
  restart(): void {
    const config: PlansConfig = { enabled: true, ...this.options.config };
    const modes = new ModeRegistry({
      default: 'chat',
      modes: { chat: { system_prompt: 'You are a travel agent.', tools: ['weather', 'lookup'], requires_approval: ['weather'], plans: true, ...this.options.mode } as ModeConfig },
    });
    const tools = new ToolRegistry()
      .register({ name: 'weather', handle: () => { this.weatherRan += 1; return 'Sunny'; } })
      .register({ name: 'lookup', handle: async () => { this.lookups += 1; await this.onLookup?.(); return 'Found'; } });
    const events = new HarnessEvents();
    events.listen((event) => this.events.push(event));
    this.plans = new Plans({ modes, config, events });
    this.runtime = new AgentRuntime({
      modes,
      tools,
      events,
      client: async (request) => {
        this.requests.push(request);
        const next = this.queue.shift();
        if (next === undefined) throw new Error('The scripted model has no response left.');

        return next;
      },
    });
    this.harness = new PrismHarness({
      drivers: { memory: () => this.ephemeral, durable: () => this.durable },
      stores: { ephemeral: 'memory', durable: 'durable' },
      runtime: this.runtime,
      plans: this.plans,
    });
  }

  session(): Session {
    return this.harness.for({ type: 'User', id: 1 }).session('chat');
  }

  calls(calls: Call[]): void {
    this.queue.push({ text: '', finishReason: 'tool_calls', toolCalls: calls.map(([name, args]) => ({ id: `call_${++this.#call}`, name, arguments: args })) });
  }

  text(text = 'Done.'): void {
    this.queue.push({ text, finishReason: 'stop' });
  }

  /** One step's turn: `lookups` lookup calls, then the report, then text. */
  step(lookups: number, summary: string): void {
    for (let i = 0; i < lookups; i++) this.calls([['lookup', { q: `q${i}` }]]);
    this.calls([['plan_complete_step', { summary }]]);
    this.text('Step done.');
  }

  async active(count = 3): Promise<void> {
    this.calls([['plan_draft', { goal: 'Plan the trip', steps: steps(count) }]]);
    const pending = (await this.runtime.send(this.session(), 'Plan the trip')).pendingApprovals[0]!;
    this.text('Plan approved.');
    await recordApproval(this.session(), pending.id, true);
    await this.runtime.send(this.session(), '');
  }

  async handle() {
    return this.session().plan();
  }

  async plan(): Promise<Record<string, any>> {
    return ((await (await this.handle()).current())?.toJSON() ?? {}) as Record<string, any>;
  }
}

function steps(count: number): JsonObject[] {
  return Array.from({ length: count }, (_, i) => ({ title: `Step ${i + 1}`, instruction: `Do part ${i + 1} of the work.`, owner: 'agent' }));
}

const actor = (): RunContext => RunContext.metadata(null, 'user:ada');

describe('plan budgets (G-366)', () => {
  it('keeps per_step as the default: every step gets the whole per-turn budget and nothing is recorded', async () => {
    const r = new Rig();
    await r.active(2);
    const before = r.requests.length;
    r.step(3, 'one');
    expect((await (await r.handle()).advance()).status).toBe('ran');
    r.step(3, 'two');
    expect((await (await r.handle()).advance()).status).toBe('ran');
    expect((await (await r.handle()).advance()).status).toBe('done');

    expect(r.requests.length - before).toBe(10);
    expect(r.lookups).toBe(6);
    expect(await (await r.handle()).budget()).toMatchObject({ mode: 'per_step', steps: 0 });
    expect(await r.durable.get(`${r.session().key()}:plan-budget`)).toBeNull();
  });

  it('ignores configured limits under per_step: they apply only to a shared budget', async () => {
    const r = new Rig({ config: { budget: { mode: 'per_step', max_steps: 1 } } });
    await r.active(1);
    r.step(2, 'one');
    await (await r.handle()).advance();

    expect(r.lookups).toBe(2);
    expect((await (await r.handle()).advance()).status).toBe('done');
  });

  it('keeps the per-turn step limit and the plan advance limit under none', async () => {
    const r = new Rig({ config: { budget: { mode: 'none' }, max_advances: 2 }, mode: { max_steps: 2 } });
    await r.active(3);
    const before = r.requests.length;
    r.step(2, 'one');
    await (await r.handle()).advance();
    expect(r.requests.length - before).toBe(2);
    expect(r.lookups).toBe(2);

    r.queue.length = 0;
    r.step(0, 'two');
    await (await r.handle()).advance();
    const outcome = await (await r.handle()).advance();
    expect(outcome.status).toBe('blocked');
    expect(outcome.reason).toBe('budget');
  });

  it('stops a plan whose shared budget is spent across several steps, once, with plan_budget_exhausted', async () => {
    const r = new Rig({ config: { budget: { mode: 'shared', max_steps: 5 } } });
    await r.active(3);
    const before = r.requests.length;
    r.step(1, 'one');
    await (await r.handle()).advance();
    r.step(3, 'two');
    await (await r.handle()).advance();

    expect(r.requests.length - before).toBe(5);
    expect(r.lookups).toBe(3);
    expect(await (await r.handle()).budget()).toMatchObject({ mode: 'shared', max_steps: 5, steps: 5 });

    const outcome = await (await r.handle()).advance();
    expect(outcome.status).toBe('blocked');
    expect(outcome.reason).toBe('plan_budget_exhausted');
    const plan = await r.plan();
    expect(plan.status).toBe('blocked');
    expect(plan.blockedReason).toBe('plan_budget_exhausted');
    expect(plan.steps[0].state).toBe('done');
    const exhausted = r.events.filter((event) => event.type === 'plan.budget_exhausted');
    expect(exhausted).toHaveLength(1);
    expect(exhausted[0]).toMatchObject({ code: 'plan_budget_exhausted', limit: 'steps', steps: 5 });

    await expect((await r.handle()).resume(actor())).rejects.toMatchObject({ code: 'plan.not_resumable' });
    expect((await (await r.handle()).advance()).status).toBe('not_active');
    expect(r.requests.length - before).toBe(5);
    expect(r.events.filter((event) => event.type === 'plan.budget_exhausted')).toHaveLength(1);
  });

  it('continues one shared ledger across an approval pause and a restart, without counting a step twice', async () => {
    const r = new Rig({ config: { budget: { mode: 'shared', max_steps: 4 } } });
    await r.active(2);
    const before = r.requests.length;
    r.calls([['weather', { city: 'Paris' }]]);
    await (await r.handle()).advance();
    const held = (await r.plan()).steps[0];
    expect(held.state).toBe('held');
    expect((await (await r.handle()).budget())!.steps).toBe(1);

    r.restart();
    for (const q of ['a', 'b', 'c', 'd']) r.calls([['lookup', { q }]]);
    await recordApproval(r.session(), held.approvalIds[0], true);
    await r.runtime.send(r.session(), '', undefined, actor());

    expect(r.weatherRan).toBe(1);
    expect(r.lookups).toBe(3);
    expect(r.requests.length - before).toBe(4);
    expect((await (await r.handle()).budget())!.steps).toBe(4);

    r.queue.length = 0;
    const outcome = await (await r.handle()).advance();
    expect(outcome.status).toBe('blocked');
    expect(outcome.reason).toBe('plan_budget_exhausted');
    expect(r.requests.length - before).toBe(4);
  });

  it('lets only one of two concurrent advances run, and charges only the one that ran', async () => {
    const r = new Rig({ config: { budget: { mode: 'shared', max_steps: 4 } } });
    await r.active(2);
    let second: string | null = null;
    r.onLookup = async () => {
      try {
        await (await r.handle()).advance();
        second = 'ran';
      } catch (error) {
        second = (error as { code?: string }).code ?? 'other';
      }
    };
    r.step(1, 'one');
    await (await r.handle()).advance();

    expect(second).toBe('session_locked');
    expect((await (await r.handle()).budget())!.steps).toBe(3);
  });

  it('never grants two turns the same remainder of a shared budget', async () => {
    const r = new Rig({ config: { budget: { mode: 'shared', max_steps: 10 } } });
    await r.active(1);
    const session = r.session();
    const plan = (await r.plans.load(session))!;
    await r.plans.write(session, PlanMachine.claim(plan, plan.steps[0]!.id, 'worker', r.plans.now() + 300, r.plans.now()), plan.revision);
    const turn = (id: string) => r.plans.planTurn(session, id, new RunContext(RunLedger.start(id), new RunBudget(8)), undefined);

    expect((await turn('run_a')).budget.maxSteps).toBe(8);
    expect((await turn('run_b')).budget.maxSteps).toBe(2);
    await expect(turn('run_c')).rejects.toMatchObject({ code: 'plan_budget_exhausted' });
  });

  it('ignores budget fields the model sends and refuses a run tree handed to a shared plan step', async () => {
    const r = new Rig({ config: { budget: { mode: 'shared', max_steps: 3 } } });
    r.calls([['plan_draft', { goal: 'Plan', steps: steps(1), budget: { mode: 'none', max_steps: 999 } }]]);
    r.text('ok');
    await r.runtime.send(r.session(), 'Plan');
    expect(await (await r.handle()).current()).toBeNull();

    await r.active(2);
    const before = r.requests.length;
    r.text('unused');
    await expect((await r.handle()).advance(new RunContext(RunLedger.start('host'), new RunBudget(100))))
      .rejects.toMatchObject({ code: 'plan.budget_context' });
    expect(r.requests.length).toBe(before);
    expect(await (await r.handle()).budget()).toMatchObject({ mode: 'shared', max_steps: 3 });
  });

  it('lets host code choose a plan budget, fixes it on first use, and only lowers it after', async () => {
    const r = new Rig({ config: { budget: { mode: 'per_step', max_steps: 50 } } });
    await r.active(2);
    const handle = await r.handle();
    await handle.useBudget({ mode: 'shared', max_steps: 2 });
    expect(await handle.budget()).toMatchObject({ mode: 'shared', max_steps: 2 });
    await handle.useBudget({ mode: 'shared', max_steps: 40 });
    expect((await handle.budget())!.max_steps).toBe(2);
    await expect(handle.useBudget({ mode: 'none' })).rejects.toMatchObject({ code: 'plan.budget_fixed' });

    r.step(2, 'one');
    await (await r.handle()).advance();
    expect(r.lookups).toBe(2);
    expect((await (await r.handle()).advance()).reason).toBe('plan_budget_exhausted');
  });

  it.each([
    ['unknown mode', { mode: 'unlimited' }],
    ['shared without a limit', { mode: 'shared' }],
    ['zero steps', { mode: 'shared', max_steps: 0 }],
    ['steps as text', { mode: 'shared', max_steps: '5' }],
    ['negative cost', { mode: 'shared', max_steps: 5, max_cost_usd: -1 }],
    ['unknown key', { mode: 'shared', max_steps: 5, per_call: true }],
    ['not a map', 'shared'],
  ])('refuses an invalid plan budget in config before anything runs: %s', async (_name, budget) => {
    const r = new Rig({ config: { budget: budget as never } });
    await expect(r.session().plan()).rejects.toMatchObject({ name: 'PlanBudgetInvalid' });
  });

  // Only the plan's own turns spend from a shared budget: the turns advance()
  // runs, and the resume of an approval its held step asked for. A person
  // chatting while a step waits is not the plan, and is never refused for it.

  /** A shared plan whose first step paused on a weather approval after one model step. */
  async function held(maxSteps: number): Promise<{ r: Rig; id: string }> {
    const r = new Rig({ config: { budget: { mode: 'shared', max_steps: maxSteps } } });
    await r.active(2);
    r.calls([['weather', { city: 'Paris' }]]);
    await (await r.handle()).advance();
    const step = (await r.plan()).steps[0];
    expect(step.state).toBe('held');
    expect((await (await r.handle()).budget())!.steps).toBe(1);

    return { r, id: step.approvalIds[0] };
  }

  it('does not charge a shared plan for a chat turn while its step is held', async () => {
    const { r } = await held(10);
    r.calls([['lookup', { q: 'a' }]]);
    r.calls([['lookup', { q: 'b' }]]);
    r.text('Unrelated answer.');
    await r.runtime.send(r.session(), 'Unrelated question');

    expect(r.lookups).toBe(2);
    expect((await (await r.handle()).budget())!.steps).toBe(1);
  });

  it('lets the person chat when a shared budget is spent while a step is held', async () => {
    const { r } = await held(1);
    r.text('Unrelated answer.');

    expect((await r.runtime.send(r.session(), 'Unrelated question')).text).toBe('Unrelated answer.');
    expect((await (await r.handle()).budget())!.steps).toBe(1);
  });

  it('does not charge a shared plan for a turn while its step is claimed by no running advance', async () => {
    const r = new Rig({ config: { budget: { mode: 'shared', max_steps: 10 } } });
    await r.active(1);
    const session = r.session();
    // A worker claimed the step and died before its turn ran.
    const plan = (await r.plans.load(session))!;
    await r.plans.write(session, PlanMachine.claim(plan, plan.steps[0]!.id, 'worker', r.plans.now() + 300, r.plans.now()), plan.revision);
    r.calls([['lookup', { q: 'a' }]]);
    r.text('Unrelated answer.');
    await r.runtime.send(r.session(), 'Unrelated question');

    expect(r.lookups).toBe(1);
    expect((await (await r.handle()).budget())!.steps).toBe(0);
  });

  it('refuses an approval a spent shared budget cannot run, records nothing, and blocks the plan once', async () => {
    const { r, id } = await held(1);
    const before = r.requests.length;
    // How many result rows the thread holds: a recorded decision adds one.
    const results = async () => (await r.session().thread().messages()).filter((entry) => entry.message.type === 'tool_result').length;
    const recorded = await results();

    await expect(recordApproval(r.session(), id, true)).rejects.toMatchObject({ code: 'plan_budget_exhausted' });
    await expect(r.runtime.decide(r.session(), id, true)).rejects.toMatchObject({ code: 'plan_budget_exhausted' });
    await expect(recordDecisions(r.session(), [{ approvalId: id, approved: true }])).rejects.toMatchObject({ code: 'plan_budget_exhausted' });
    expect(await results()).toBe(recorded);
    expect(r.weatherRan).toBe(0);
    expect(r.requests.length).toBe(before);

    const outcome = await (await r.handle()).advance();
    expect(outcome.status).toBe('blocked');
    expect(outcome.reason).toBe('plan_budget_exhausted');
    expect(r.events.filter((event) => event.type === 'plan.budget_exhausted')).toHaveLength(1);
    expect((await (await r.handle()).advance()).status).toBe('not_active');
    expect(r.events.filter((event) => event.type === 'plan.budget_exhausted')).toHaveLength(1);

    // Still refused once the plan is blocked: the call never runs outside the budget.
    const stream = r.runtime.streamDecision(r.session(), [{ approvalId: id, approved: true }], actor());
    await expect((async () => { for await (const _ of stream) { /* drain */ } })()).rejects.toMatchObject({ code: 'plan_budget_exhausted' });
    expect(await results()).toBe(recorded);
    expect(r.weatherRan).toBe(0);
    expect(r.requests.length).toBe(before);
  });

  it('closes the run when a shared plan budget cannot be written', async () => {
    const r = new Rig({ config: { budget: { mode: 'shared', max_steps: 10 } } });
    await r.active(1);
    r.durable.failBudgetWrites = true;
    r.text('unused');

    await expect((await r.handle()).advance()).rejects.toThrow('store down');
    expect((await r.session().run())!.status).toBe('failed');
    expect(r.events.some((event) => event.type === 'run.failed')).toBe(true);
  });
});
