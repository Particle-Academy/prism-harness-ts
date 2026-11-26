import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  RunBudget,
  RunContext,
  SubagentRunner,
  ToolRegistry,
  recordApproval,
  type Durability,
  type JsonObject,
  type LlmRequest,
  type LlmResponse,
  type ModeConfig,
  type Session,
} from '../src/index.js';

// A subagent child that stops on an approval is resumed later on its own
// session, outside the runner. That run continues the run tree it was started
// in: it spends what the tree has left, never a fresh root budget. A spent
// tree leaves it the approvals already decided and no model step (G-387):
// the approved call runs and nothing more.
// Driven through the public API (runtime.send, recordApproval) with a scripted
// model client; fetch is refused and every store is in memory.

beforeEach(() => {
  vi.stubGlobal('fetch', () => { throw new Error('A test tried to use the network.'); });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

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
  lookups = 0;
  weatherRan = 0;
  #call = 0;

  constructor(treeSteps: number) {
    const modes = new ModeRegistry({
      default: 'chat',
      modes: {
        chat: { system_prompt: 'You delegate.', tools: ['research'], max_steps: treeSteps, subagents: { research: { mode: 'research' } } } as ModeConfig,
        research: { system_prompt: 'Research.', tools: ['weather', 'lookup'], requires_approval: ['weather'], max_steps: 4 } as ModeConfig,
      },
    });
    const tools = new ToolRegistry()
      .register({ name: 'weather', handle: () => { this.weatherRan += 1; return 'Sunny'; } })
      .register({ name: 'lookup', handle: () => { this.lookups += 1; return 'Found'; } });
    this.runtime = new AgentRuntime({
      modes,
      tools,
      client: async (request) => {
        this.requests.push(request);
        const next = this.queue.shift();
        if (next === undefined) throw new Error('The scripted model has no response left.');

        return next;
      },
    });
    const durable = new DurableMemoryStore(), ephemeral = new MemorySessionStore();
    this.harness = new PrismHarness({
      drivers: { memory: () => ephemeral, durable: () => durable },
      stores: { ephemeral: 'memory', durable: 'durable' },
      runtime: this.runtime,
    });
    // The host wires the subagent tool with a context of its own; called by a
    // turn, the child runs in that turn's tree.
    tools.register(new SubagentRunner(this.harness, this.runtime)
      .tool(modes.resolve('chat').subagents.research!, this.session(), RunContext.root('host', new RunBudget(100)), 'host_run'));
  }

  session(scope = 'chat'): Session {
    return this.harness.for({ type: 'User', id: 1 }).session(scope);
  }

  calls(calls: [string, JsonObject][]): void {
    this.queue.push({ text: '', finishReason: 'tool_calls', toolCalls: calls.map(([name, args]) => ({ id: `call_${++this.#call}`, name, arguments: args })) });
  }

  text(text = 'Done.'): void {
    this.queue.push({ text, finishReason: 'stop' });
  }

  async pending(session: Session): Promise<string[]> {
    const rows = (await session.thread().messages()).map((entry) => entry.message);
    const answered = new Set(rows.flatMap((row) => Array.isArray(row.tool_approval_responses) ? row.tool_approval_responses as JsonObject[] : [])
      .map((response) => response.approval_id));

    return rows.flatMap((row) => Array.isArray(row.tool_approval_requests) ? row.tool_approval_requests as JsonObject[] : [])
      .map((request) => request.approval_id as string).filter((id) => !answered.has(id));
  }

  async results(session: Session): Promise<number> {
    return (await session.thread().messages()).filter((entry) => entry.message.type === 'tool_result').length;
  }
}

/** The parent turn (a tree of `treeSteps` model steps) delegates; the child pauses on weather after one step; the parent answers. Three tree steps. */
async function pausedChild(treeSteps: number): Promise<{ r: Rig; child: Session; id: string }> {
  const r = new Rig(treeSteps);
  r.calls([['research', { task: 'Find the weather.' }]]);   // parent
  r.calls([['weather', { city: 'Paris' }]]);                 // child: pauses
  r.text('Asked.');                                          // parent
  await r.runtime.send(r.session(), 'Go');

  const child = r.session('chat::sub::research');
  const pending = await r.pending(child);
  expect(pending).toHaveLength(1);

  return { r, child, id: pending[0]! };
}

describe('a subagent child resumed outside the runner', () => {
  it('resumes on what its run tree has left, not a fresh root budget', async () => {
    const { r, child, id } = await pausedChild(4);
    const before = r.requests.length;

    // The child would take three more steps; the tree has one left.
    r.calls([['lookup', { q: 'a' }]]);
    r.calls([['lookup', { q: 'b' }]]);
    r.text('Child answer.');
    await recordApproval(child, id, true);
    await r.runtime.send(child, '');

    expect(r.weatherRan).toBe(1);
    expect(r.requests.length - before).toBe(1);
  });

  it('acts on an approval for a child whose run tree is spent, with no model step', async () => {
    const { r, child, id } = await pausedChild(3);
    const before = r.requests.length;

    await recordApproval(child, id, true);
    const response = await r.runtime.send(child, '');

    expect(response.finishReason).toBe('budget_exhausted');
    expect(await r.pending(child)).toEqual([]);
    expect(r.weatherRan).toBe(1);
    expect(r.requests.length).toBe(before);
  });

  it('charges a resumed child to its run tree once, so a second resume gets only what is left', async () => {
    const { r, child, id } = await pausedChild(5);

    // Resumed: one step, which pauses on weather again (tree 4 of 5).
    r.calls([['weather', { city: 'Rome' }]]);
    await recordApproval(child, id, true);
    const second = (await r.runtime.send(child, '')).pendingApprovals[0]!.id;
    const before = r.requests.length;

    // One left: the child takes one of the three it would.
    r.calls([['lookup', { q: 'a' }]]);
    r.calls([['lookup', { q: 'b' }]]);
    r.text('Child answer.');
    await recordApproval(child, second, true);
    await r.runtime.send(child, '');

    expect(r.requests.length - before).toBe(1);
  });
});

// A turn on a child session the runner did not start never gets a fresh root
// budget by any route: not a new message instead of an answer, and not a run
// context the host hands in (which may only narrow what the tree has left).
describe('a turn on a subagent child session outside the runner', () => {
  const host = () => RunContext.root('host_root', new RunBudget(100));
  const drain = async (steps: AsyncGenerator<unknown, unknown, undefined>) => { for await (const _ of steps) { /* drained */ } };

  it('runs a new message to a child session on what its run tree has left', async () => {
    const { r, child } = await pausedChild(4);
    const before = r.requests.length;

    r.calls([['lookup', { q: 'a' }]]);
    r.calls([['lookup', { q: 'b' }]]);
    r.text('Child answer.');
    await r.runtime.send(child, 'New question');

    expect(r.requests.length - before).toBe(1);
  });

  it('takes no model step for a new message to a child session whose run tree is spent', async () => {
    const { r, child } = await pausedChild(3);
    const before = r.requests.length;

    const response = await r.runtime.send(child, 'New question');

    expect(response.finishReason).toBe('budget_exhausted');
    expect(r.requests.length).toBe(before);
  });

  it('narrows a host-supplied run context on a streamed child resume to what its run tree has left', async () => {
    const { r, child, id } = await pausedChild(4);
    const before = r.requests.length;

    r.calls([['lookup', { q: 'a' }]]);
    r.calls([['lookup', { q: 'b' }]]);
    r.text('Child answer.');
    await drain(r.runtime.streamDecision(child, [{ approvalId: id, approved: true }], host()));

    expect(r.weatherRan).toBe(1);
    expect(r.requests.length - before).toBe(1);
  });

  it('acts on a streamed decision under a host-supplied run context on a child whose run tree is spent, with no model step', async () => {
    const { r, child, id } = await pausedChild(3);
    const before = r.requests.length;

    await drain(r.runtime.streamDecision(child, [{ approvalId: id, approved: true }], host()));

    expect(await r.pending(child)).toEqual([]);
    expect(r.weatherRan).toBe(1);
    expect(r.requests.length).toBe(before);
  });

  it('narrows a host-supplied run context on a resume after the answer is recorded', async () => {
    const { r, child, id } = await pausedChild(4);
    const before = r.requests.length;

    r.calls([['lookup', { q: 'a' }]]);
    r.calls([['lookup', { q: 'b' }]]);
    r.text('Child answer.');
    await recordApproval(child, id, true);
    await r.runtime.send(child, '', undefined, host());

    expect(r.weatherRan).toBe(1);
    expect(r.requests.length - before).toBe(1);
  });
});
