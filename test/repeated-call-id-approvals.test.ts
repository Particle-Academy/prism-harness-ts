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
  SubagentRunner,
  ToolRegistry,
  recordApproval,
  type JsonObject,
  type LlmRequest,
  type LlmResponse,
  type LlmToolCall,
  type ModeConfig,
  type Session,
} from '../src/index.js';
import { ARGUMENTS_AS_SENT, assistantRow } from '../src/thread-rows.js';

class DurableMemory extends MemorySessionStore {
  override durability(): 'durable' {
    return 'durable';
  }
}

// G-392: each call in a step is decided by its OWN approval request, never by
// a map keyed by tool call id. A step whose calls (or whose approval requests)
// repeat a tool call id cannot be paired safely, so every gated call in it is
// denied and does not run, whatever was decided. Offline; every store is in
// memory.

const REPEATED = 'Tool call was not run because another call in this step has the same id';

const network = vi.fn(() => { throw new Error('A test tried to use the network.'); });
beforeEach(() => { vi.stubGlobal('fetch', network); });
afterEach(() => { expect(network).not.toHaveBeenCalled(); vi.unstubAllGlobals(); });

const modes = new ModeRegistry({
  default: 'guarded',
  modes: { guarded: { system_prompt: 'Careful.', tools: ['echo', 'free'], max_steps: 6, requires_approval: ['echo'] } },
});

interface Rig {
  session: Session;
  agent: AgentRuntime;
  requests: LlmRequest[];
  /** The `n` argument of every echo that ran, in order. */
  ran: number[];
  free: () => number;
}

function rig(calls: LlmToolCall[]): Rig {
  const harness = new PrismHarness({
    drivers: { memory: () => new MemorySessionStore(), durable: () => new DurableMemory() },
    stores: { ephemeral: 'memory', durable: 'durable' },
  });
  const session = harness.for({ type: 'User', id: 1 }).session('repeated');
  const ran: number[] = [];
  let free = 0;
  const tools = new ToolRegistry()
    .register({ name: 'echo', handle: (args: JsonObject) => { ran.push(Number(args.n)); return 'ok'; } })
    .register({ name: 'free', handle: () => { free += 1; return 'free ok'; } });
  const requests: LlmRequest[] = [];
  const client = async (request: LlmRequest): Promise<LlmResponse> => {
    requests.push(request);

    return requests.length === 1 && calls.length > 0
      ? { text: '', finishReason: 'tool_calls', toolCalls: calls }
      : { text: 'done', finishReason: 'stop' };
  };

  return { session, agent: new AgentRuntime({ client, modes, tools }), requests, ran, free: () => free };
}

const echo = (id: string, n: number): LlmToolCall => ({ id, name: 'echo', arguments: { n } });

async function rows(session: Session): Promise<JsonObject[]> {
  return (await session.thread().messages()).map((entry) => entry.message);
}

/** The result entries of the resolution row: the last tool result row. */
async function resolved(session: Session): Promise<JsonObject[]> {
  const results = (await rows(session)).filter((row) => row.type === 'tool_result' && (row.tool_results as JsonObject[]).length > 0);

  return (results.at(-1)?.tool_results ?? []) as JsonObject[];
}

describe('G-392: a step with a repeated tool call id runs none of its gated calls', () => {
  const orders: [string, boolean, boolean][] = [
    ['deny call 1, approve call 2', false, true],
    ['approve call 1, deny call 2', true, false],
    ['approve both', true, true],
  ];

  for (const [name, first, second] of orders) {
    it(`${name}: neither runs, and each gets its own denial`, async () => {
      const r = rig([echo('weather', 1), echo('weather', 2)]);
      const parked = await r.agent.send(r.session, 'go');
      expect(parked.finishReason).toBe('awaiting_approval');
      expect(parked.pendingApprovals).toHaveLength(2);
      const [one, two] = parked.pendingApprovals;
      expect(one!.id).not.toBe(two!.id);

      await recordApproval(r.session, one!.id, first, first ? undefined : 'no');
      await recordApproval(r.session, two!.id, second, second ? undefined : 'no');
      await r.agent.send(r.session, '');

      expect(r.ran).toEqual([]);
      const entries = await resolved(r.session);
      expect(entries.map((entry) => [entry.tool_call_id, (entry.args as JsonObject).n, entry.result])).toEqual([
        ['weather', 1, REPEATED],
        ['weather', 2, REPEATED],
      ]);
    });
  }

  it('no later decision or resume revives a call denied for a repeated id', async () => {
    const r = rig([echo('weather', 1), echo('weather', 2)]);
    const parked = await r.agent.send(r.session, 'go');
    const [one, two] = parked.pendingApprovals;
    await recordApproval(r.session, one!.id, false, 'no');
    await r.agent.send(r.session, '');
    expect(r.ran).toEqual([]);

    // The second approval was never answered; answering it now, and resuming
    // again, still runs nothing.
    await recordApproval(r.session, two!.id, true).catch(() => undefined);
    await r.agent.send(r.session, '');
    await r.agent.send(r.session, '');

    expect(r.ran).toEqual([]);
    const denials = (await rows(r.session)).filter((row) => row.type === 'tool_result')
      .flatMap((row) => row.tool_results as JsonObject[]).filter((entry) => entry.result === REPEATED);
    expect(denials).toHaveLength(2);
  });

  it('empty ids count as repeated', async () => {
    const r = rig([echo('', 1), echo('', 2)]);
    const parked = await r.agent.send(r.session, 'go');
    for (const approval of parked.pendingApprovals) await recordApproval(r.session, approval.id, true);
    await r.agent.send(r.session, '');

    expect(r.ran).toEqual([]);
    expect((await resolved(r.session)).map((entry) => entry.result)).toEqual([REPEATED, REPEATED]);
  });

  it('an ungated call sharing the id still runs at once; the gated calls do not run', async () => {
    const r = rig([{ id: 'dup', name: 'free', arguments: {} }, echo('dup', 1), echo('dup', 2)]);
    const parked = await r.agent.send(r.session, 'go');
    expect(r.free()).toBe(1);
    expect(parked.pendingApprovals).toHaveLength(2);
    for (const approval of parked.pendingApprovals) await recordApproval(r.session, approval.id, true);
    await r.agent.send(r.session, '');
    await r.agent.send(r.session, '');

    expect(r.ran).toEqual([]);
    expect(r.free()).toBe(1);
    const denials = (await rows(r.session)).filter((row) => row.type === 'tool_result')
      .flatMap((row) => row.tool_results as JsonObject[]).filter((entry) => entry.result === REPEATED);
    expect(denials.map((entry) => (entry.args as JsonObject).n)).toEqual([1, 2]);
  });

  it('approval requests that repeat a tool call id make the step ambiguous too (rows as stored)', async () => {
    const r = rig([]);
    // A step whose two requests both name the first call, as a row written by
    // another writer could hold it.
    await r.session.thread().record([
      { type: 'user', content: 'go' },
      {
        ...assistantRow('', [echo('c1', 1), echo('c2', 2)], {}, [
          { approval_id: 'apr_1', tool_call_id: 'c1', arguments_format: ARGUMENTS_AS_SENT },
          { approval_id: 'apr_2', tool_call_id: 'c1', arguments_format: ARGUMENTS_AS_SENT },
        ]),
        harness_tool_names: ['echo', 'free'],
      },
    ]);

    await recordApproval(r.session, 'apr_1', true);
    await recordApproval(r.session, 'apr_2', true);
    await r.agent.send(r.session, '');

    expect(r.ran).toEqual([]);
    expect((await resolved(r.session)).map((entry) => [entry.tool_call_id, entry.result])).toEqual([['c1', REPEATED], ['c2', REPEATED]]);
  });
});

describe('G-392 control: unique ids are decided per call, by their own approval', () => {
  it('deny call 1, approve call 2: only call 2 runs, and call 1 carries the denial', async () => {
    const r = rig([echo('c1', 1), echo('c2', 2)]);
    const [one, two] = (await r.agent.send(r.session, 'go')).pendingApprovals;
    await recordApproval(r.session, one!.id, false, 'no');
    await recordApproval(r.session, two!.id, true);
    await r.agent.send(r.session, '');

    expect(r.ran).toEqual([2]);
    expect((await resolved(r.session)).map((entry) => [entry.tool_call_id, entry.result])).toEqual([['c1', 'no'], ['c2', 'ok']]);
  });

  it('approve call 1, deny call 2: only call 1 runs', async () => {
    const r = rig([echo('c1', 1), echo('c2', 2)]);
    const [one, two] = (await r.agent.send(r.session, 'go')).pendingApprovals;
    await recordApproval(r.session, one!.id, true);
    await recordApproval(r.session, two!.id, false, 'no');
    await r.agent.send(r.session, '');

    expect(r.ran).toEqual([1]);
    expect((await resolved(r.session)).map((entry) => [entry.tool_call_id, entry.result])).toEqual([['c1', 'ok'], ['c2', 'no']]);
  });
});

// ---------------------------------------------------------------------------
// The start path, and the plan: a runner start that ends a parked child's
// calls marks each with its OWN approval.
// ---------------------------------------------------------------------------

class StartRig {
  readonly queue: LlmResponse[] = [];
  readonly durable = new DurableMemory();
  readonly ephemeral = new MemorySessionStore();
  readonly ran: number[] = [];
  readonly events = new HarnessEvents();
  readonly runtime: AgentRuntime;
  readonly harness: PrismHarness;
  readonly modes: ModeRegistry;

  constructor(private readonly id: (index: number) => string) {
    this.modes = new ModeRegistry({
      default: 'lead',
      modes: {
        lead: { system_prompt: 'Lead.', tools: ['echo'], max_steps: 8, requires_approval: ['echo'], subagents: { a: { mode: 'work', max_steps: 8 } } } as ModeConfig,
        work: { system_prompt: 'Work.', tools: ['echo'], max_steps: 8, requires_approval: ['echo'], plans: true } as ModeConfig,
      },
    });
    const registry = new ToolRegistry().register({ name: 'echo', handle: (args: JsonObject) => { this.ran.push(Number(args.n)); return 'ok'; } });
    this.runtime = new AgentRuntime({
      modes: this.modes, tools: registry, events: this.events,
      client: async () => this.queue.shift() ?? { text: 'Done.', finishReason: 'stop' },
    });
    const durable = this.durable, ephemeral = this.ephemeral;
    this.harness = new PrismHarness({
      drivers: { ephemeral: () => ephemeral, durable: () => durable },
      stores: { ephemeral: 'ephemeral', durable: 'durable' },
      runtime: this.runtime,
      plans: new Plans({ modes: this.modes, config: { enabled: true }, events: this.events }),
    });
  }

  session(): Session {
    return this.harness.for({ type: 'User', id: 1 }).session('chat');
  }

  child(): Session {
    return this.harness.for({ type: 'User', id: 1 }).session('chat::sub::a');
  }

  calls(calls: [string, JsonObject][]): void {
    this.queue.push({ text: '', finishReason: 'tool_calls', toolCalls: calls.map(([name, args], index) => ({ id: this.id(index), name, arguments: args })) });
  }

  async run(task: string): Promise<{ outcome: string }> {
    return new SubagentRunner(this.harness, this.runtime).run(this.modes.resolve('lead').subagents.a!, this.session(), RunContext.root('root', new RunBudget(8)), 'root', task);
  }

  async rows(): Promise<JsonObject[]> {
    return (await this.child().thread().messages()).map((entry) => entry.message);
  }

  async step(): Promise<Record<string, any>> {
    return (((await (await this.child().plan()).current())?.toJSON() ?? {}) as Record<string, any>).steps[0];
  }

  /** A child plan step held on two gated echo calls; both approved. Their approval ids. */
  async heldOnTwo(): Promise<string[]> {
    await this.session().usingMode('lead');
    this.calls([['plan_draft', { goal: 'Do the work', steps: [{ title: 'Step 1', instruction: 'Echo twice.', owner: 'agent' }] }]]);
    await this.run('Plan it');
    const draft = (await this.rows()).flatMap((row) => (row.tool_approval_requests ?? []) as JsonObject[]).pop()!;
    await recordApproval(this.child(), draft.approval_id as string, true);
    this.queue.push({ text: 'Plan approved.', finishReason: 'stop' });
    await this.runtime.send(this.child(), '');
    this.calls([['echo', { n: 1 }], ['echo', { n: 2 }]]);
    await (await this.child().plan()).advance();
    const step = await this.step();
    expect(step.state).toBe('held');
    const ids = step.approvalIds as string[];
    expect(ids).toHaveLength(2);
    for (const id of ids) await recordApproval(this.child(), id, true);

    return ids;
  }

  /** Nobody can act on the child any more, so a re-delegation ends its calls. */
  async endByStart(): Promise<void> {
    const key = `${this.child().key()}:durable`;
    await this.durable.put(key, { ...((await this.durable.get(key)) ?? {}), child_depth: 1_000 });
    this.queue.length = 0;
    this.queue.push({ text: 'Child done.', finishReason: 'stop' });
    expect((await this.run('New task')).outcome).toBe('completed');
  }
}

describe('G-392: ended entries carry their own approval id', () => {
  it('two calls sharing one id in a step a start ends: each ended entry names its own approval', async () => {
    const r = new StartRig(() => 'weather');
    const ids = await r.heldOnTwo();
    await r.endByStart();

    expect(r.ran).toEqual([]);
    // The row-level mark (G-387 round 12, the reference's shape): each call's own approval, in call order.
    const ended = (await r.rows()).filter((row) => row.type === 'tool_result' && Object.hasOwn(row, 'harness_ended_by_start'))
      .flatMap((row) => (row.harness_ended_by_start as { approval_ids: string[] }).approval_ids);
    expect(ended).toEqual(ids);
    const step = await r.step();
    // Round 13: both decisions were RECORDED as the repeated-id denial (as
    // the PHP reference records them), so the step never counted them
    // approved, before the start or after it.
    for (const id of ids) expect(step.approvedIds).not.toContain(id);
    expect(storedDecisionsOf(await r.rows(), ids)).toEqual(ids.map(() => [false, 'Tool call was not run because another call in this step has the same id']));
  });

  it('unique ids in a step a start ends: each ended entry names its own approval (control)', async () => {
    const r = new StartRig((index) => `call_${index + 1}`);
    const ids = await r.heldOnTwo();
    await r.endByStart();

    expect(r.ran).toEqual([]);
    // The row-level mark (G-387 round 12, the reference's shape): each call's own approval, in call order.
    const ended = (await r.rows()).filter((row) => row.type === 'tool_result' && Object.hasOwn(row, 'harness_ended_by_start'))
      .flatMap((row) => (row.harness_ended_by_start as { approval_ids: string[] }).approval_ids);
    expect(ended).toEqual(ids);
  });

  it('a plan step held on two calls sharing an id, both approved and resumed: nothing runs, and the plan never reads them as approved work', async () => {
    const r = new StartRig(() => 'weather');
    const ids = await r.heldOnTwo();
    await r.runtime.send(r.child(), '');

    expect(r.ran).toEqual([]);
    const step = await r.step();
    for (const id of ids) expect(step.approvedIds).not.toContain(id);
  });
});

/** The first stored decision for each id, as [approved, reason]. */
function storedDecisionsOf(rows: JsonObject[], ids: string[]): [unknown, unknown][] {
  const answers = rows.filter((row) => row.type === 'tool_result').flatMap((row) => (row.tool_approval_responses ?? []) as JsonObject[]);
  return ids.map((id) => { const answer = answers.find((entry) => entry.approval_id === id); return [answer?.approved, answer?.reason]; });
}
