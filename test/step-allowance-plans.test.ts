import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  MemorySessionStore,
  ModeRegistry,
  Plans,
  PrismHarness,
  RunContext,
  ToolRegistry,
  recordApproval,
  type AgentResponse,
  type Durability,
  type JsonObject,
  type LlmRequest,
  type LlmResponse,
  type ModeConfig,
  type Session,
} from '../src/index.js';

// G-387 on the plan path (D3, D4, D5): a plan step whose shared budget is
// spent on its last step still has its pending approval decided AND acted
// on, with no model step: never "accepted, then never acted on", and never
// refused for a step budget. The row it parks on says what it has left of
// its allowance, plan narrowing included. Driven through the public API
// (advance, recordApproval, send, streamDecision) with a scripted client.

const network = vi.fn(() => { throw new Error('A test tried to use the network.'); });
beforeEach(() => { vi.stubGlobal('fetch', network); });
afterEach(() => { expect(network).not.toHaveBeenCalled(); vi.unstubAllGlobals(); });

class DurableMemoryStore extends MemorySessionStore {
  override durability(): Durability {
    return 'durable';
  }
}

class Rig {
  readonly queue: LlmResponse[] = [];
  readonly requests: LlmRequest[] = [];
  readonly runtime: AgentRuntime;
  readonly harness: PrismHarness;
  weatherRan = 0;
  #call = 0;

  constructor(maxSteps: number, maxCostUsd: number | null = null) {
    const modes = new ModeRegistry({
      default: 'chat',
      modes: {
        chat: { system_prompt: 'You plan.', tools: ['weather', 'lookup'], requires_approval: ['weather'], plans: true, max_steps: 8 } as ModeConfig,
      },
    });
    const tools = new ToolRegistry()
      .register({ name: 'weather', handle: () => { this.weatherRan += 1; return 'Sunny'; } })
      .register({ name: 'lookup', handle: () => 'Found' });
    const plans = new Plans({ modes, config: { enabled: true, budget: { mode: 'shared', max_steps: maxSteps, ...(maxCostUsd === null ? {} : { max_cost_usd: maxCostUsd }) } } });
    this.runtime = new AgentRuntime({
      modes, tools,
      client: async (request) => {
        this.requests.push(request);
        const next = this.queue.shift();
        if (next !== undefined) return next;

        return { text: '', finishReason: 'tool_calls', toolCalls: [{ id: `call_${++this.#call}`, name: 'weather', arguments: { n: this.#call } }] };
      },
    });
    const durable = new DurableMemoryStore(), ephemeral = new MemorySessionStore();
    this.harness = new PrismHarness({
      drivers: { memory: () => ephemeral, durable: () => durable },
      stores: { ephemeral: 'memory', durable: 'durable' },
      runtime: this.runtime,
      plans,
    });
  }

  session(): Session {
    return this.harness.for({ type: 'User', id: 1 }).session('chat');
  }

  calls(calls: [string, JsonObject][]): void {
    this.queue.push({ text: '', finishReason: 'tool_calls', toolCalls: calls.map(([name, args]) => ({ id: `call_${++this.#call}`, name, arguments: args })) });
  }

  text(text = 'Done.'): void {
    this.queue.push({ text, finishReason: 'stop' });
  }

  /** An active plan of two agent steps; the first step's turn takes `lookups` steps, then parks on weather. */
  async held(lookups: number): Promise<string> {
    this.calls([['plan_draft', { goal: 'Plan the trip', steps: [1, 2].map((i) => ({ title: `Step ${i}`, instruction: `Do part ${i}.`, owner: 'agent' })) }]]);
    const draft = (await this.runtime.send(this.session(), 'Plan the trip')).pendingApprovals[0]!;
    this.text('Plan approved.');
    await recordApproval(this.session(), draft.id, true);
    await this.runtime.send(this.session(), '');
    for (let i = 0; i < lookups; i += 1) this.calls([['lookup', { q: i }]]);
    this.calls([['weather', { city: 'Paris' }]]);
    await (await this.session().plan()).advance();
    const step = ((await (await this.session().plan()).current())?.toJSON() as Record<string, any>).steps[0];
    expect(step.state).toBe('held');

    return step.approvalIds[0] as string;
  }

  async parkedRow(): Promise<JsonObject> {
    const rows = (await this.session().thread().messages()).map((entry) => entry.message);

    return [...rows].reverse().find((row) => Array.isArray(row.tool_approval_requests) && row.tool_approval_requests.length > 0)!;
  }
}

const drain = async (steps: AsyncGenerator<unknown, AgentResponse, undefined>): Promise<AgentResponse> => {
  for (;;) {
    const next = await steps.next();
    if (next.done === true) return next.value;
  }
};

describe('a plan step whose shared budget is spent on its last step', () => {
  const paths: [string, (r: Rig, id: string) => Promise<AgentResponse>][] = [
    ['recordApproval then send', async (r, id) => { await recordApproval(r.session(), id, true); return r.runtime.send(r.session(), '', undefined, RunContext.metadata(null, 'user:ada')); }],
    ['streamDecision', async (r, id) => drain(r.runtime.streamDecision(r.session(), [{ approvalId: id, approved: true }], RunContext.metadata(null, 'user:ada')))],
  ];

  for (const [label, resume] of paths) {
    it(`accepts the decision and runs the approved call (${label}), with no model step`, async () => {
      const r = new Rig(1);
      const id = await r.held(0);
      expect((await (await r.session().plan()).budget())!.steps).toBe(1);
      const before = r.requests.length;

      const response = await resume(r, id);

      expect(r.weatherRan).toBe(1);
      expect(r.requests.length).toBe(before);
      expect(response.finishReason).toBe('budget_exhausted');
      expect((await (await r.session().plan()).budget())!.steps).toBe(1);
    });
  }
});

describe('the row a plan step parks on', () => {
  it('says what the step has left of its allowance, plan narrowing included (D5)', async () => {
    const r = new Rig(3);
    await r.held(1);

    const row = await r.parkedRow();

    // Its own ceiling is 8; the plan's shared budget held 3 for it; it took 2.
    expect(row.harness_steps_used).toBe(2);
    expect(row.harness_steps_left).toBe(1);
  });
});

describe('a decision on a plan step whose shared COST budget is spent', () => {
  // Accepting decisions on a spent STEP budget must not open the cost
  // budget: the provider reported no cost, so the cost cap cannot be held
  // and the decision is refused with nothing recorded, as before.
  it('is still refused, recording nothing', async () => {
    const r = new Rig(10, 1);
    const id = await r.held(0);
    const rows = (await r.session().thread().messages()).length;

    await expect(recordApproval(r.session(), id, true)).rejects.toMatchObject({ code: 'plan_budget_exhausted' });
    expect((await r.session().thread().messages()).length).toBe(rows);
    expect(r.weatherRan).toBe(0);
  });
});
