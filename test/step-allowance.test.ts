import { mkdtemp } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  FileSessionStore,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  RunBudget,
  RunContext,
  SubagentRunner,
  ToolRegistry,
  assistantRow,
  toolResultEntry,
  toolResultRow,
  recordApproval,
  type AgentResponse,
  type Durability,
  type JsonObject,
  type LlmRequest,
  type LlmResponse,
  type ModeConfig,
  type RuntimeConfig,
  type Session,
  type SessionStore,
} from '../src/index.js';
import { RunTrees } from '../src/run-trees.js';

// G-387: a run's step allowance has ONE authority, decided when the run
// begins. Driven through the public seams only (send, streamDecision,
// sendStructured, recordApproval, SubagentRunner.run, a host RunContext),
// with a scripted model client that, once its script is spent, always asks
// for the approval-gated `echo` tool: every resume would take every step it
// is given. No network; every store is in memory or in a temp directory.

const network = vi.fn(() => { throw new Error('A test tried to use the network.'); });
beforeEach(() => { vi.stubGlobal('fetch', network); });
afterEach(() => { expect(network).not.toHaveBeenCalled(); vi.unstubAllGlobals(); });

class DurableMemoryStore extends MemorySessionStore {
  override durability(): Durability {
    return 'durable';
  }
}

interface RigOptions {
  /** The lead mode's max_steps. */
  lead?: number;
  /** Each subagent's declared max_steps (mode `work`, 8 steps). */
  declared?: Record<string, number>;
  /** Tools the lead mode gates, besides echo. */
  gated?: string[];
  config?: RuntimeConfig;
  /** A directory for file stores; in memory otherwise. */
  directory?: string;
}

class Rig {
  readonly queue: LlmResponse[] = [];
  readonly requests: LlmRequest[] = [];
  readonly ran: Record<string, number> = {};
  runtime!: AgentRuntime;
  harness!: PrismHarness;
  modes!: ModeRegistry;
  readonly #durable: SessionStore;
  readonly #ephemeral: SessionStore;
  #call = 0;

  constructor(private readonly options: RigOptions = {}) {
    this.#durable = options.directory === undefined ? new DurableMemoryStore() : new FileSessionStore(options.directory);
    this.#ephemeral = options.directory === undefined ? new MemorySessionStore() : new FileSessionStore(options.directory);
    this.restart();
  }

  /** A new process: a new runtime, registry and harness over the same stores. */
  restart(): void {
    const declared = this.options.declared ?? { a: 2 };
    this.modes = new ModeRegistry({
      default: 'lead',
      modes: {
        lead: {
          system_prompt: 'Lead.', tools: ['echo', 'lookup', ...Object.keys(declared)], max_steps: this.options.lead ?? 8,
          requires_approval: ['echo', ...(this.options.gated ?? [])],
          subagents: Object.fromEntries(Object.entries(declared).map(([name, steps]) => [name, { mode: 'work', max_steps: steps }])),
        } as ModeConfig,
        work: { system_prompt: 'Work.', tools: ['echo', 'lookup'], max_steps: 8, requires_approval: ['echo'] } as ModeConfig,
      },
    });
    const tools = new ToolRegistry()
      .register({ name: 'echo', handle: () => { this.ran.echo = (this.ran.echo ?? 0) + 1; return 'ok'; } })
      .register({ name: 'lookup', handle: () => { this.ran.lookup = (this.ran.lookup ?? 0) + 1; return 'found'; } });
    this.runtime = new AgentRuntime({
      modes: this.modes, tools, ...(this.options.config === undefined ? {} : { config: this.options.config }),
      client: async (request) => {
        this.requests.push(request);
        const next = this.queue.shift();
        if (next !== undefined) return next;

        return { text: '', finishReason: 'tool_calls', toolCalls: [{ id: `call_${++this.#call}`, name: 'echo', arguments: { n: this.#call } }] };
      },
    });
    const durable = this.#durable, ephemeral = this.#ephemeral;
    this.harness = new PrismHarness({
      drivers: { ephemeral: () => ephemeral, durable: () => durable },
      stores: { ephemeral: 'ephemeral', durable: 'durable' },
      runtime: this.runtime,
    });
    for (const name of Object.keys(declared)) {
      tools.register(new SubagentRunner(this.harness, this.runtime)
        .tool(this.modes.resolve('lead').subagents[name]!, this.session(), RunContext.root('host', new RunBudget(100)), 'host_run'));
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

  /** The runner starts subagent `name` directly, under a host root context of `steps` (no parent turn). */
  async runner(name = 'a', steps = 8, task = 'go'): Promise<void> {
    await this.session().usingMode('lead');
    await new SubagentRunner(this.harness, this.runtime)
      .run(this.modes.resolve('lead').subagents[name]!, this.session(), RunContext.root('root', new RunBudget(steps)), 'root', task);
  }

  async pending(session: Session): Promise<string[]> {
    const rows = (await session.thread().messages()).map((entry) => entry.message);
    const answered = new Set(rows.flatMap((row) => Array.isArray(row.tool_approval_responses) ? row.tool_approval_responses as JsonObject[] : [])
      .map((response) => response.approval_id));

    return rows.flatMap((row) => Array.isArray(row.tool_approval_requests) ? row.tool_approval_requests as JsonObject[] : [])
      .map((request) => request.approval_id as string).filter((id) => !answered.has(id));
  }

  /** The row the session last parked on. */
  async parkedRow(session: Session): Promise<JsonObject> {
    const rows = (await session.thread().messages()).map((entry) => entry.message);

    return [...rows].reverse().find((row) => Array.isArray(row.tool_approval_requests) && row.tool_approval_requests.length > 0)!;
  }

  /**
   * Approve whatever the session is parked on and resume it, up to `times`
   * times or until it stops parking; `fresh` restarts the rig before each
   * resume (a different worker, persisted state only).
   */
  async approveLoop(scope: () => Session, context: () => RunContext | undefined, times = 6, fresh = false): Promise<AgentResponse | null> {
    let last: AgentResponse | null = null;
    for (let index = 0; index < times; index += 1) {
      const pending = await this.pending(scope());
      if (pending.length === 0) break;
      if (fresh) this.restart();
      await recordApproval(scope(), pending[0]!, true);
      last = await this.runtime.send(scope(), '', undefined, context());
      if (last.finishReason !== 'awaiting_approval') break;
    }

    return last;
  }
}

const host = (steps: number) => () => RunContext.root('host_resume', new RunBudget(steps));
const none = () => undefined;
const metadata = () => RunContext.metadata('turn_1', 'user:ada');
const drain = async (steps: AsyncGenerator<unknown, AgentResponse, undefined>): Promise<AgentResponse> => {
  for (;;) {
    const next = await steps.next();
    if (next.done === true) return next.value;
  }
};

describe('D1: a declared subagent ceiling holds on turns the runner did not start', () => {
  for (const [label, context] of [['no context', none], ['a metadata context', metadata], ['a host context offering 10', host(10)]] as const) {
    it(`takes at most one more model step for a child declared 2 that parked after 1, resumed with ${label}`, async () => {
      const r = new Rig({ declared: { a: 2 } });
      await r.runner('a');
      expect(r.requests).toHaveLength(1);

      const last = await r.approveLoop(() => r.child(), context);

      expect(r.requests.length - 1).toBeLessThanOrEqual(1);
      expect(last?.finishReason).toBe('budget_exhausted');
    });
  }

  it('holds for a child a parent turn started, whose run tree has steps to spare', async () => {
    const r = new Rig({ declared: { a: 2 } });
    r.calls([['a', { task: 'Work.' }]]);   // parent
    r.calls([['echo', { n: 0 }]]);         // child: parks
    r.text('Asked.');                      // parent
    await r.runtime.send(r.session(), 'Go');
    const before = r.requests.length;

    await r.approveLoop(() => r.child(), none);

    expect(r.requests.length - before).toBeLessThanOrEqual(1);
  });

  it('is never re-derived from a context that only looks like the runner\'s', async () => {
    const r = new Rig({ declared: { a: 2 } });
    await r.runner('a');
    // A context the runner marked, derived again by the host: the host's.
    const marked = RunTrees.startingChild(RunContext.root('looks_like_runner', new RunBudget(50)), null, r.session().key());

    await r.approveLoop(() => r.child(), () => marked.withTurn('turn_2'));

    expect(r.requests.length - 1).toBeLessThanOrEqual(1);
  });
});

describe('a host-supplied context only lowers a saved ceiling', () => {
  it('never raises it: a child declared 3 that parked after 1 takes at most 2 more under a host context of 10', async () => {
    const r = new Rig({ declared: { a: 3 } });
    await r.runner('a');

    await r.approveLoop(() => r.child(), host(10));

    expect(r.requests.length - 1).toBeLessThanOrEqual(2);
  });

  it('lowers it: a host context of 1 gives the same child one step on that turn', async () => {
    const r = new Rig({ declared: { a: 3 } });
    await r.runner('a');
    await recordApproval(r.child(), (await r.pending(r.child()))[0]!, true);

    const response = await r.runtime.send(r.child(), '', undefined, host(1)());

    expect(r.requests.length - 1).toBe(1);
    expect(response.finishReason).toBe('awaiting_approval');
  });
});

describe('a sibling that spent the run tree', () => {
  it('leaves the other child its approved call, run, with no model step and no refusal', async () => {
    const r = new Rig({ lead: 5, declared: { a: 4, b: 4 } });
    r.calls([['a', { task: 'One.' }], ['b', { task: 'Two.' }]]);   // parent: both children
    r.calls([['echo', { n: 'a' }]]);                                // a: parks
    r.calls([['echo', { n: 'b' }]]);                                // b: parks
    r.text('Asked both.');                                          // parent (tree 4 of 5)
    await r.runtime.send(r.session(), 'Go');
    expect(await r.pending(r.child('a'))).toHaveLength(1);
    expect(await r.pending(r.child('b'))).toHaveLength(1);

    // a, resumed outside the runner, spends the tree's last step.
    let before = r.requests.length;
    await r.approveLoop(() => r.child('a'), none, 1);
    expect(r.requests.length - before).toBe(1);

    // b's decision is accepted and acted on, with no model step.
    before = r.requests.length;
    const echoes = r.ran.echo ?? 0;
    await recordApproval(r.child('b'), (await r.pending(r.child('b')))[0]!, true);
    const response = await r.runtime.send(r.child('b'), '');

    expect(r.ran.echo).toBe(echoes + 1);
    expect(r.requests.length).toBe(before);
    expect(response.finishReason).toBe('budget_exhausted');
  });
});

describe('a run approved again and again never passes its ceiling', () => {
  it('counts a root\'s children\'s steps: a root that spent 5 of 8 through a child gets at most 3 more, across fresh workers', async () => {
    const r = new Rig({ lead: 8, declared: { a: 8 }, directory: await mkdtemp(join(tmpdir(), 'g387-repark-')) });
    r.calls([['a', { task: 'Work.' }]]);   // parent
    r.calls([['lookup', { q: 1 }]]);       // child
    r.calls([['lookup', { q: 2 }]]);       // child
    r.text('Child done.');                 // child
    r.calls([['echo', { n: 0 }]]);         // parent: parks (5 of 8)
    await r.runtime.send(r.session(), 'Go');
    expect(r.requests).toHaveLength(5);
    expect((await r.parkedRow(r.session())).harness_steps_used).toBe(5);

    await r.approveLoop(() => r.session(), none, 6, true);

    expect(r.requests.length - 5).toBeLessThanOrEqual(3);
  });

  it('holds a child resumed outside the runner to its declared 3, across six approvals and fresh workers', async () => {
    const r = new Rig({ declared: { a: 3 }, directory: await mkdtemp(join(tmpdir(), 'g387-repark-')) });
    await r.runner('a');

    await r.approveLoop(() => r.child(), none, 6, true);

    expect(r.requests.length).toBeLessThanOrEqual(3);
  });
});

describe('the ceiling that binds, in turn', () => {
  it('the mode\'s max_steps for a root', async () => {
    const r = new Rig({ lead: 2 });
    await r.runtime.send(r.session(), 'Go');
    await r.approveLoop(() => r.session(), none);

    expect(r.requests).toHaveLength(2);
  });

  it('the declaration for a child, below its mode\'s', async () => {
    const r = new Rig({ declared: { a: 2 } });
    await r.runner('a');
    await r.approveLoop(() => r.child(), none);

    expect(r.requests).toHaveLength(2);
  });

  it('max_steps_ceiling, below the mode\'s, for a root', async () => {
    const r = new Rig({ lead: 8, config: { max_steps_ceiling: 2 } });
    await r.runtime.send(r.session(), 'Go');
    await r.approveLoop(() => r.session(), none);

    expect(r.requests).toHaveLength(2);
  });

  it('max_steps_ceiling, below the declaration, for a child resumed outside the runner', async () => {
    const r = new Rig({ declared: { a: 5 }, config: { max_steps_ceiling: 2 } });
    await r.runner('a');
    await r.approveLoop(() => r.child(), none);

    expect(r.requests).toHaveLength(2);
  });

  // D2: absolute on the run's whole count, whatever context the resume has.
  for (const [label, context] of [['no context', none], ['a host context offering 10', host(10)]] as const) {
    it(`max_steps_ceiling 3 holds a root resumed with ${label} to 3 in total`, async () => {
      const r = new Rig({ lead: 8, config: { max_steps_ceiling: 3 } });
      await r.runtime.send(r.session(), 'Go');
      await r.approveLoop(() => r.session(), context);

      expect(r.requests).toHaveLength(3);
    });
  }
});

describe('a run spent on its last step', () => {
  // Child declared 1, parked after its one step; its run tree has plenty left.
  async function spentChild(): Promise<Rig> {
    const r = new Rig({ declared: { a: 1 } });
    r.calls([['a', { task: 'Work.' }]]);   // parent
    r.calls([['echo', { n: 0 }]]);         // child: parks, its one step spent
    r.text('Asked.');                      // parent
    await r.runtime.send(r.session(), 'Go');
    expect(await r.pending(r.child())).toHaveLength(1);

    return r;
  }

  const paths: [string, (r: Rig, id: string) => Promise<AgentResponse>][] = [
    ['send', async (r, id) => { await recordApproval(r.child(), id, true); return r.runtime.send(r.child(), ''); }],
    ['streamDecision', async (r, id) => drain(r.runtime.streamDecision(r.child(), [{ approvalId: id, approved: true }]))],
    ['sendStructured', async (r, id) => {
      await recordApproval(r.child(), id, true);
      return r.runtime.sendStructured(r.child(), '', { type: 'object', properties: {} });
    }],
  ];

  for (const [label, resume] of paths) {
    it(`accepts the decision and runs the approved call on ${label}, with no model step`, async () => {
      const r = await spentChild();
      const before = r.requests.length;
      const echoes = r.ran.echo ?? 0;

      const response = await resume(r, (await r.pending(r.child()))[0]!);

      expect(r.ran.echo).toBe(echoes + 1);
      expect(r.requests.length).toBe(before);
      expect(response.finishReason).toBe('budget_exhausted');
    });
  }

  // Decided rule: the approval is acted on, and the delegation is refused.
  // The approved call runs, and its result is the runner's "did not run"
  // refusal, so its child takes no step.
  it('acts on an approved subagent call on a spent run: the call runs, and the runner refuses the delegation', async () => {
    const r = new Rig({ lead: 1, declared: { a: 4 }, gated: ['a'] });
    r.calls([['a', { task: 'Work.' }]]);   // parent: parks on the gated subagent
    await r.runtime.send(r.session(), 'Go');
    await recordApproval(r.session(), (await r.pending(r.session()))[0]!, true);

    const response = await r.runtime.send(r.session(), '');

    expect(r.requests).toHaveLength(1);
    expect(response.finishReason).toBe('budget_exhausted');
    expect(response.toolCalls).toEqual(['a']);
    const results = (await r.session().thread().messages()).map((entry) => JSON.stringify(entry.message)).filter((row) => row.includes('subagent did not run'));
    expect(results).toHaveLength(1);
    expect(await rowCountOf(r.child())).toBe(0);
  });
});

describe('the steps a parked run used, as its row records them', () => {
  /** A root session parked on one echo call, by hand: `assistants` model rows since the user row, the last asking. */
  async function forged(r: Rig, used: unknown, assistants = 1): Promise<string> {
    const thread = r.session().thread();
    await r.session().usingMode('lead');
    await thread.record([{ type: 'user', content: 'Go' }], 'run_forged');
    for (let index = 1; index < assistants; index += 1) {
      await thread.record([assistantRow('', [{ id: `l${index}`, name: 'lookup', arguments: {} }]) as unknown as JsonObject], 'run_forged');
      await thread.record([toolResultRow([toolResultEntry({ id: `l${index}`, name: 'lookup', arguments: {} }, 'found')]) as unknown as JsonObject], 'run_forged');
    }
    const row: JsonObject = assistantRow('', [{ id: 'e1', name: 'echo', arguments: {} }], {}, [{ approval_id: 'apr_forged', tool_call_id: 'e1', arguments_format: 1 }]) as unknown as JsonObject;
    row.harness_tool_names = ['echo', 'lookup'];
    if (used !== undefined) row.harness_steps_used = used as JsonObject[string];
    await thread.record([row], 'run_forged');

    return 'apr_forged';
  }

  for (const garbage of [-1, 1.5, '0', null, 2 ** 60, { steps: 0 }]) {
    it(`fails closed on ${JSON.stringify(garbage)}: the approved call runs and no model step is taken`, async () => {
      const r = new Rig({ lead: 3 });
      const id = await forged(r, garbage);
      await recordApproval(r.session(), id, true);

      const response = await r.runtime.send(r.session(), '');

      expect(r.ran.echo).toBe(1);
      expect(r.requests).toHaveLength(0);
      expect(response.finishReason).toBe('budget_exhausted');
    });
  }

  it('never counts below the rows the run recorded: a forged 0 on a run that took 3 of 3', async () => {
    const r = new Rig({ lead: 3 });
    const id = await forged(r, 0, 3);
    await recordApproval(r.session(), id, true);

    await r.runtime.send(r.session(), '');

    expect(r.requests).toHaveLength(0);
  });

  it('counts a legacy row (no harness_steps_used) from its rows, under a host context too', async () => {
    const r = new Rig({ lead: 8, config: { max_steps_ceiling: 3 } });
    const id = await forged(r, undefined, 3);
    await recordApproval(r.session(), id, true);

    await r.runtime.send(r.session(), '', undefined, host(10)());

    expect(r.requests).toHaveLength(0);
  });
});

describe('a new prompt with no step allowance', () => {
  // A NEW prompt resumes no recorded decisions, so an allowance of 0 is
  // refused before anything is recorded (run_not_permitted), as before
  // G-387. Only a resume with an allowance of 0 acts without a model step.
  async function refusedQuietly(r: Rig, session: Session): Promise<void> {
    const rows = (await session.thread().messages()).length;
    const before = r.requests.length;

    await expect(r.runtime.send(session, 'New question')).rejects.toMatchObject({ code: 'run_not_permitted' });

    expect((await session.thread().messages()).length).toBe(rows);
    expect(r.requests.length).toBe(before);
  }

  it('is refused on a child whose saved ceiling is spent', async () => {
    const r = new Rig({ declared: { a: 2 } });
    await r.runner('a');
    const binding = (await r.child().childBinding())!;
    await r.child().rememberChildBinding(binding, await r.child().childDepth(), 0);

    await refusedQuietly(r, r.child());
  });

  it('is refused on a child whose run tree a sibling spent', async () => {
    const r = new Rig({ lead: 5, declared: { a: 4, b: 4 } });
    r.calls([['a', { task: 'One.' }], ['b', { task: 'Two.' }]]);
    r.calls([['echo', { n: 'a' }]]);
    r.calls([['echo', { n: 'b' }]]);
    r.text('Asked both.');
    await r.runtime.send(r.session(), 'Go');
    await r.approveLoop(() => r.child('a'), none, 1);   // a spends the tree's last step

    await refusedQuietly(r, r.child('b'));
  });
});

describe('a parked row without a step count', () => {
  // Written by the same version that saves a child's ceiling: on a session
  // with a saved ceiling, an absent count is not a legacy row. Fails closed.
  it('fails closed on a child session that saved a ceiling', async () => {
    const r = new Rig({ declared: { a: 4 } });
    const legacy = vi.spyOn(RunTrees, 'allowanceOf').mockReturnValue(null);
    try {
      await r.runner('a');
    } finally {
      legacy.mockRestore();
    }
    expect((await r.parkedRow(r.child())).harness_steps_used).toBeUndefined();
    const echoes = r.ran.echo ?? 0;
    await recordApproval(r.child(), (await r.pending(r.child()))[0]!, true);

    const response = await r.runtime.send(r.child(), '');

    expect(r.ran.echo).toBe(echoes + 1);
    expect(r.requests).toHaveLength(1);
    expect(response.finishReason).toBe('budget_exhausted');
  });
});

async function rowCountOf(session: Session): Promise<number> {
  return (await session.thread().messages()).length;
}
