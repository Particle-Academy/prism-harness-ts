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
  SubagentRunner,
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
  /** A `research` subagent the host wires with its own run context, as SubagentRunner.tool() takes it. */
  hostSubagent?: RunContext;
  /** The research child delegates to a `dig` subagent of its own. */
  nested?: boolean;
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
      modes: {
        chat: { system_prompt: 'You are a travel agent.', tools: ['weather', 'lookup'], requires_approval: ['weather'], plans: true, ...this.options.mode } as ModeConfig,
        research: {
          system_prompt: 'You research.', tools: this.options.nested === true ? ['weather', 'lookup', 'dig'] : ['weather', 'lookup'], requires_approval: ['weather'], max_steps: 8,
          ...(this.options.nested === true ? { subagents: { dig: { mode: 'dig' } } } : {}),
        } as ModeConfig,
        dig: { system_prompt: 'You dig.', tools: [], max_steps: 4 } as ModeConfig,
      },
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
    if (this.options.hostSubagent !== undefined) {
      const subagent = modes.resolve('chat').subagents.research!;
      tools.register(new SubagentRunner(this.harness, this.runtime).tool(subagent, this.session(), this.options.hostSubagent, 'host_run'));
    }
    if (this.options.nested === true) {
      const dig = modes.resolve('research').subagents.dig!;
      tools.registerFactory('dig', (session) => new SubagentRunner(this.harness, this.runtime).tool(dig, session, RunContext.root('dig', new RunBudget(100)), 'dig_run'));
    }
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
    const turn = (id: string) => r.plans.planTurn(session, id, new RunContext(RunLedger.start(id), new RunBudget(8)), undefined, '', 8);
    // Each as the step's own turn, as advance() runs it.
    const serving = r.plans.servingStep(session);

    expect((await turn('run_a')).planLeft).toBe(8);
    expect((await turn('run_b')).planLeft).toBe(2);
    // Spent: a grant of no step, never a refusal (G-387).
    expect((await turn('run_c')).planLeft).toBe(0);
    serving();
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

  // G-387: acting on a recorded decision costs no model step, so a spent
  // shared STEP budget never refuses one; the resume runs the approved call
  // and takes no further step, and the plan then stops once.
  for (const [label, decide] of [
    ['recordApproval', (r: Rig, id: string) => recordApproval(r.session(), id, true)],
    ['decide', (r: Rig, id: string) => r.runtime.decide(r.session(), id, true)],
    ['recordDecisions', (r: Rig, id: string) => recordDecisions(r.session(), [{ approvalId: id, approved: true }])],
  ] as const) {
    it(`accepts a decision (${label}) on a step whose shared budget is spent, runs the call with no model step, and blocks the plan once`, async () => {
      const { r, id } = await held(1);
      const before = r.requests.length;

      await decide(r, id);
      const resumed = await r.runtime.send(r.session(), '', undefined, actor());

      expect(resumed.finishReason).toBe('budget_exhausted');
      expect(r.weatherRan).toBe(1);
      expect(r.requests.length).toBe(before);
      expect((await (await r.handle()).budget())!.steps).toBe(1);

      const outcome = await (await r.handle()).advance();
      expect(outcome.status).toBe('blocked');
      expect(outcome.reason).toBe('plan_budget_exhausted');
      expect(r.events.filter((event) => event.type === 'plan.budget_exhausted')).toHaveLength(1);
      expect((await (await r.handle()).advance()).status).toBe('not_active');
      expect(r.events.filter((event) => event.type === 'plan.budget_exhausted')).toHaveLength(1);
      expect(r.requests.length).toBe(before);
    });
  }

  it('closes the run when a shared plan budget cannot be written', async () => {
    const r = new Rig({ config: { budget: { mode: 'shared', max_steps: 10 } } });
    await r.active(1);
    r.durable.failBudgetWrites = true;
    r.text('unused');

    await expect((await r.handle()).advance()).rejects.toThrow('store down');
    expect((await r.session().run())!.status).toBe('failed');
    expect(r.events.some((event) => event.type === 'run.failed')).toBe(true);
  });

  // A subagent the step delegates to spends from the step's own turn, so from
  // the plan's shared budget. The run context the host wired the subagent
  // tool with may narrow that, never widen it or replace the ledger.

  const delegating = (maxSteps: number, nested = false) => new Rig({
    config: { budget: { mode: 'shared', max_steps: maxSteps } },
    mode: { tools: ['weather', 'lookup', 'research'], subagents: { research: { mode: 'research', max_steps: 8 } } },
    hostSubagent: RunContext.root('host', new RunBudget(100)),
    nested,
  });

  it('charges a subagent a shared plan step delegates to the plan budget', async () => {
    const r = delegating(10);
    await r.active(1);
    r.calls([['research', { task: 'Find the weather.' }]]);   // parent
    r.text('Child answer.');                                   // child
    r.calls([['plan_complete_step', { summary: 'Asked.' }]]);  // parent
    r.text('Step done.');                                      // parent
    await (await r.handle()).advance();

    expect(r.queue).toHaveLength(0);
    expect((await (await r.handle()).budget())!.steps).toBe(4);
  });

  it('never lets the host context a subagent tool was wired with widen a shared plan step', async () => {
    const r = delegating(3);
    await r.active(1);
    r.calls([['research', { task: 'Find the weather.' }]]);   // parent: step 1 of 3
    for (const q of ['a', 'b', 'c']) r.calls([['lookup', { q }]]);
    r.text('Child answer.');
    await (await r.handle()).advance();

    // The child had the two steps the plan had left, not the host's hundred.
    expect(r.requests.filter((request) => request.systemPrompt.startsWith('You research.'))).toHaveLength(2);
    expect(r.lookups).toBe(2);
    expect(r.queue).toHaveLength(2);
    expect((await (await r.handle()).budget())!.steps).toBe(3);
  });

  it('lets the host context a subagent tool was wired with narrow a turn', async () => {
    const r = new Rig({
      config: { budget: { mode: 'shared', max_steps: 10 } },
      mode: { tools: ['weather', 'lookup', 'research'], subagents: { research: { mode: 'research', max_steps: 8 } } },
      hostSubagent: RunContext.root('host', new RunBudget(1)),
    });
    await r.active(1);
    r.calls([['research', { task: 'Find the weather.' }]]);   // parent
    r.calls([['lookup', { q: 'a' }]]);                         // child: its one step
    r.text('Step done.');                                      // parent
    await (await r.handle()).advance();

    // The child stopped after the one step the host context allowed.
    expect(r.requests.filter((request) => request.systemPrompt.startsWith('You research.'))).toHaveLength(1);
    expect(r.lookups).toBe(1);
    expect(r.queue).toHaveLength(0);
  });

  // A subagent child that pauses on an approval is resumed later on its own
  // session, outside the runner. When the turn that started it spent from a
  // shared plan budget, the resumed child spends from the same one.

  /** A plan step that delegates; the child pauses on weather. */
  async function pausedChild(maxSteps: number, nested = false): Promise<{ r: Rig; child: Session; id: string }> {
    const r = delegating(maxSteps, nested);
    await r.active(1);
    r.calls([['research', { task: 'Find the weather.' }]]);   // parent
    r.calls([['weather', { city: 'Paris' }]]);                 // child: pauses
    r.calls([['plan_complete_step', { summary: 'Asked.' }]]);  // parent
    r.text('Step done.');                                      // parent
    await (await r.handle()).advance();

    const child = r.harness.for({ type: 'User', id: 1 }).session('chat::sub::research');
    const pending = (await child.thread().messages()).map((entry) => entry.message)
      .flatMap((row) => Array.isArray(row.tool_approval_requests) ? row.tool_approval_requests as JsonObject[] : [])
      .map((request) => request.approval_id as string);
    expect(pending).toHaveLength(1);
    expect((await (await r.handle()).budget())!.steps).toBe(4);

    return { r, child, id: pending[0]! };
  }

  it('charges a child resumed outside the runner to the shared plan budget it was started under', async () => {
    const { r, child, id } = await pausedChild(10);
    r.text('Child answer.');
    await recordApproval(child, id, true);
    await r.runtime.send(child, '');

    expect(r.weatherRan).toBe(1);
    expect((await (await r.handle()).budget())!.steps).toBe(5);
  });

  it('acts on an approval for a child whose shared plan budget is spent, with no model step', async () => {
    const { r, child, id } = await pausedChild(4);
    const before = r.requests.length;

    await recordApproval(child, id, true);
    const response = await r.runtime.send(child, '');

    expect(response.finishReason).toBe('budget_exhausted');
    expect(r.weatherRan).toBe(1);
    expect(r.requests.length).toBe(before);
    expect((await (await r.handle()).budget())!.steps).toBe(4);
  });

  const hostRoot = () => RunContext.root('host_root', new RunBudget(100));
  const drained = async (steps: AsyncGenerator<unknown, unknown, undefined>) => { for await (const _ of steps) { /* drained */ } };

  it('charges a host-supplied run context on a resumed child to the shared plan budget, never around it', async () => {
    const { r, child, id } = await pausedChild(10);
    r.text('Child answer.');
    await drained(r.runtime.streamDecision(child, [{ approvalId: id, approved: true }], hostRoot()));

    expect(r.weatherRan).toBe(1);
    expect((await (await r.handle()).budget())!.steps).toBe(5);
  });

  it('acts on a streamed decision under a host-supplied run context on a child whose shared plan budget is spent, with no model step', async () => {
    const { r, child, id } = await pausedChild(4);
    const before = r.requests.length;

    await drained(r.runtime.streamDecision(child, [{ approvalId: id, approved: true }], hostRoot()));

    expect(r.weatherRan).toBe(1);
    expect(r.requests.length).toBe(before);
    expect((await (await r.handle()).budget())!.steps).toBe(4);
  });

  it('charges a new message to a child session to the shared plan budget it was started under', async () => {
    const { r, child } = await pausedChild(10);
    r.text('Child answer.');
    await r.runtime.send(child, 'New question');

    expect((await (await r.handle()).budget())!.steps).toBe(5);
  });

  // G-381: the grandchild takes the CHILD's recorded membership through its
  // parent run, so it saves the tree the child continues and the plan budget
  // the child spends from, never one found through the host's ledger.
  it("saves the child's run tree and shared plan budget on a grandchild of a child resumed under a host context", async () => {
    const { r, child } = await pausedChild(20, true);
    const saved = await child.childRefs();
    expect(saved?.tree).not.toBeNull();
    expect(saved?.planLedger).not.toBeNull();

    r.calls([['dig', { task: 'Dig.' }]]);
    r.text('Grandchild answer.');
    r.text('Child answer.');
    await r.runtime.send(child, 'Dig deeper', undefined, hostRoot());

    expect(r.queue).toHaveLength(0);
    const grandchild = r.harness.for({ type: 'User', id: 1 }).session('chat::sub::research::sub::dig');
    expect(await grandchild.childRefs()).toEqual(saved);
  });
});
