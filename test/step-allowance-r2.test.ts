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
  RunLedger,
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

// G-387 round 2 (the final review's findings). The rig is the one of
// step-allowance.test.ts. G-387: a run's step allowance has ONE authority, decided when the run
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
  /** Run once before the next model call is answered (a second worker, say). */
  beforeRequest: (() => Promise<void>) | null = null;
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
        const hook = this.beforeRequest;
        this.beforeRequest = null;
        if (hook !== null) await hook();
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

  get durable(): SessionStore {
    return this.#durable;
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

// G-387 round 2: the final-review findings, at the public seams.

/** Change a session's durable record as a store would hold it, bypassing the runtime's own writer. */
async function rewriteDurable(r: Rig, session: Session, change: (stored: JsonObject) => JsonObject): Promise<void> {
  const key = `${session.key()}:durable`;
  await r.durable.put(key, change((await r.durable.get(key)) ?? {}));
}

/** How many rows a session's thread holds. */
async function rowCount(session: Session): Promise<number> {
  return (await session.thread().messages()).length;
}

describe('a delegation\'s ceiling is a lifetime budget (item 1)', () => {
  it('holds two resumes of one child running at once to its ceiling between them', async () => {
    const r = new Rig({ declared: { a: 3 } });
    await r.runner('a');                          // one step, parks
    await recordApproval(r.child(), (await r.pending(r.child()))[0]!, true);
    for (let i = 0; i < 8; i += 1) r.calls([['lookup', { q: i }]]);
    let second: Promise<AgentResponse> | null = null;
    r.beforeRequest = async () => {
      // Inside the first resume's model call, a second worker resumes the same child.
      second = r.runtime.send(r.child(), '');
      await second.catch(() => null);
    };

    await r.runtime.send(r.child(), '');
    await (second as Promise<AgentResponse> | null)?.catch(() => null);

    expect(r.requests.length).toBeLessThanOrEqual(3);
  });

  it('holds three new messages to a child declared 2 to two model requests in total', async () => {
    const r = new Rig({ declared: { a: 2 } });
    r.text('First.');
    await r.runner('a');                          // one step
    for (let i = 0; i < 3; i += 1) {
      r.text(`Answer ${i}.`);
      await r.runtime.send(r.child(), `Question ${i}`).catch(() => null);
    }

    expect(r.requests.length).toBeLessThanOrEqual(2);
  });

  it('starts the count again when the runner starts the child again', async () => {
    const r = new Rig({ declared: { a: 2 } });
    r.text('First.');
    await r.runner('a');
    r.text('Second.');
    await r.runtime.send(r.child(), 'Question');   // the delegation's second step
    r.text('Again.');
    await r.runner('a', 8, 'Start over');          // a new delegation: a new budget

    expect(r.requests).toHaveLength(3);
  });

  it('fails closed on a step count that is not a count', async () => {
    const r = new Rig({ declared: { a: 3 } });
    await r.runner('a');
    await rewriteDurable(r, r.child(), (stored) => ({ ...stored, child_steps: 'lots' }));
    await recordApproval(r.child(), (await r.pending(r.child()))[0]!, true);

    const response = await r.runtime.send(r.child(), '');

    expect(r.requests).toHaveLength(1);
    expect(response.finishReason).toBe('budget_exhausted');
  });
});

describe('a linked child with no usable saved ceiling fails closed (item 2)', () => {
  const ceilings: [string, (stored: JsonObject) => JsonObject][] = [
    ['no ceiling', ({ child_ceiling: _gone, ...rest }) => rest],
    ['a string', (stored) => ({ ...stored, child_ceiling: '2' })],
    ['-1', (stored) => ({ ...stored, child_ceiling: -1 })],
    ['true', (stored) => ({ ...stored, child_ceiling: true })],
    ['1.5', (stored) => ({ ...stored, child_ceiling: 1.5 })],
    ['null', (stored) => ({ ...stored, child_ceiling: null })],
  ];

  for (const [label, change] of ceilings) {
    it(`acts on the approval with no model step, and refuses a new message, for ${label}`, async () => {
      const r = new Rig({ declared: { a: 2 } });
      await r.runner('a');
      await rewriteDurable(r, r.child(), change);
      const echoes = r.ran.echo ?? 0;
      await recordApproval(r.child(), (await r.pending(r.child()))[0]!, true);

      const response = await r.runtime.send(r.child(), '');

      expect(r.ran.echo).toBe(echoes + 1);
      expect(r.requests).toHaveLength(1);
      expect(response.finishReason).toBe('budget_exhausted');
      const rows = await rowCount(r.child());
      await expect(r.runtime.send(r.child(), 'New question')).rejects.toMatchObject({ code: 'run_not_permitted' });
      expect(await rowCount(r.child())).toBe(rows);
      expect(r.requests).toHaveLength(1);
    });
  }

  it('fails closed for a child linked only by the binding on its parked row (a 0.11 session)', async () => {
    const r = new Rig({ declared: { a: 2 } });
    await r.runner('a');
    await rewriteDurable(r, r.child(), ({ child_ceiling: _c, child_binding: _b, child_refs: _r, child_depth: _d, child_steps: _s, ...rest }) => rest);
    await recordApproval(r.child(), (await r.pending(r.child()))[0]!, true);

    const response = await r.runtime.send(r.child(), '');

    expect(r.requests).toHaveLength(1);
    expect(response.finishReason).toBe('budget_exhausted');
  });

  it('leaves a session that was never a child on its mode', async () => {
    const r = new Rig({ lead: 2 });
    await r.runtime.send(r.session(), 'Go');
    await r.approveLoop(() => r.session(), none);

    expect(r.requests).toHaveLength(2);
  });
});

describe('a budget that is not a number never removes the cap (item 3)', () => {
  for (const steps of [Number.NaN, Number.POSITIVE_INFINITY, undefined, -1]) {
    it(`refuses RunBudget(${String(steps)})`, () => {
      expect(() => new RunBudget(steps as number)).toThrow(TypeError);
    });
  }

  for (const steps of [Number.NaN, Number.POSITIVE_INFINITY, '8', undefined]) {
    it(`takes no model step under a host context whose budget says ${String(steps)}`, async () => {
      const r = new Rig({ declared: { a: 2 } });
      await r.runner('a');
      const budget = { maxSteps: steps, maxCostUsd: null, maxSeconds: null, maxOutputTokensPerCall: null } as unknown as RunBudget;
      const forged = new RunContext(RunLedger.start('forged'), budget);

      await r.approveLoop(() => r.child(), () => forged);

      expect(r.requests).toHaveLength(1);
    });
  }

  it('refuses a max_steps_ceiling that is not a whole number', () => {
    for (const ceiling of [Number.NaN, Number.POSITIVE_INFINITY, 1.5]) {
      expect(() => new AgentRuntime({ client: async () => ({ text: '', finishReason: 'stop' }), modes: new ModeRegistry({ default: 'm', modes: { m: { system_prompt: '', tools: [] } } }), tools: new ToolRegistry(), config: { max_steps_ceiling: ceiling } })).toThrow(TypeError);
    }
  });
});

describe('a decision that could not be acted on is refused before it is recorded (item 4)', () => {
  async function refusedBeforeRecording(r: Rig, session: Session, code: string, decide: (id: string) => Promise<unknown>): Promise<void> {
    const [id] = await r.pending(session);
    const rows = await rowCount(session);

    await expect(decide(id!)).rejects.toMatchObject({ code });

    expect(await rowCount(session)).toBe(rows);
    expect(await r.pending(session)).toEqual([id]);
  }

  it('on a child whose binding is missing', async () => {
    const r = new Rig({ declared: { a: 2 } });
    // A session the runner started (it saved run tree references), its
    // binding lost, parked on an approval.
    const child = r.child();
    await child.rememberChildRefs({ tree: { session: r.session().key(), run: 'run_root' }, planLedger: null });
    await child.usingMode('work');
    await child.thread().record([{ type: 'user', content: 'Work.' }], 'run_child');
    const row = assistantRow('', [{ id: 'e1', name: 'echo', arguments: {} }], {}, [{ approval_id: 'apr_child', tool_call_id: 'e1', arguments_format: 1 }]) as unknown as JsonObject;
    row.harness_tool_names = ['echo'];
    await child.thread().record([row], 'run_child');

    await refusedBeforeRecording(r, r.child(), 'child_binding_missing', (id) => recordApproval(r.child(), id, true));
  });

  it('on a child nested too deep', async () => {
    const r = new Rig({ declared: { a: 2 } });
    await r.runner('a');
    await rewriteDurable(r, r.child(), (stored) => ({ ...stored, child_depth: 4 }));

    await refusedBeforeRecording(r, r.child(), 'run_not_permitted', (id) => recordApproval(r.child(), id, true));
  });

  it('under a supplied context whose ledger was cancelled', async () => {
    const r = new Rig({ lead: 4 });
    await r.runtime.send(r.session(), 'Go');
    const cancelled = RunContext.root('host', new RunBudget(8));
    cancelled.ledger.cancel('stopped by host');

    await refusedBeforeRecording(r, r.session(), 'run_not_permitted', (id) => drain(r.runtime.streamDecision(r.session(), [{ approvalId: id, approved: true }], cancelled)));
  });

  it('under a supplied context whose time is spent', async () => {
    const r = new Rig({ lead: 4 });
    await r.runtime.send(r.session(), 'Go');
    const late = RunContext.root('host', new RunBudget(8, null, 0));

    await refusedBeforeRecording(r, r.session(), 'run_not_permitted', (id) => drain(r.runtime.streamDecision(r.session(), [{ approvalId: id, approved: true }], late)));
  });

  it('never runs an approved call on a resume whose run tree was cancelled', async () => {
    const r = new Rig({ lead: 4 });
    await r.runtime.send(r.session(), 'Go');
    await recordApproval(r.session(), (await r.pending(r.session()))[0]!, true);
    const cancelled = RunContext.root('host', new RunBudget(8));
    cancelled.ledger.cancel('stopped by host');

    const response = await r.runtime.send(r.session(), '', undefined, cancelled);

    expect(r.ran.echo ?? 0).toBe(0);
    expect(r.requests).toHaveLength(1);
    expect(response.stoppedBecause).toBe('stopped by host');
  });
});

describe('the runner refuses a delegation it must not make (item 5)', () => {
  for (const task of ['', '   ', '\n\t']) {
    it(`refuses an empty task (${JSON.stringify(task)}), writing nothing to the child`, async () => {
      const r = new Rig({ declared: { a: 2 } });
      await r.session().usingMode('lead');

      const result = await new SubagentRunner(r.harness, r.runtime)
        .run(r.modes.resolve('lead').subagents.a!, r.session(), RunContext.root('root', new RunBudget(8)), 'root', task);

      expect(result.outcome).toBe('denied');
      expect(r.requests).toHaveLength(0);
      expect(await rowCount(r.child())).toBe(0);
      expect(await r.child().childBinding()).toBeNull();
    });
  }

  it('refuses to delegate into a child a person still has to answer, leaving the approval pending', async () => {
    const r = new Rig({ declared: { a: 2 } });
    await r.runner('a');
    const pending = await r.pending(r.child());
    const rows = await rowCount(r.child());

    const result = await new SubagentRunner(r.harness, r.runtime)
      .run(r.modes.resolve('lead').subagents.a!, r.session(), RunContext.root('root', new RunBudget(8)), 'root', 'Something else');

    expect(result.outcome).toBe('denied');
    expect(await r.pending(r.child())).toEqual(pending);
    expect(await rowCount(r.child())).toBe(rows);
    expect(r.requests).toHaveLength(1);
  });
});

// The channels each fix could open, each with its test.
describe('new channels considered', () => {
  it('counts a live hold of a turn that never settled (a crashed worker) against the delegation', async () => {
    const r = new Rig({ declared: { a: 3 } });
    await r.runner('a');
    const now = Math.floor(Date.now() / 1000);
    await rewriteDurable(r, r.child(), (stored) => ({ ...stored, child_steps: { steps: 1, holds: { run_crashed: { steps: 2, at: now } } } }));
    await recordApproval(r.child(), (await r.pending(r.child()))[0]!, true);

    await r.runtime.send(r.child(), '');

    expect(r.requests).toHaveLength(1);
  });

  it('charges an expired hold in full, never returns it', async () => {
    const r = new Rig({ declared: { a: 3 } });
    await r.runner('a');
    await rewriteDurable(r, r.child(), (stored) => ({ ...stored, child_steps: { steps: 1, holds: { run_crashed: { steps: 2, at: 1 } } } }));
    await recordApproval(r.child(), (await r.pending(r.child()))[0]!, true);

    await r.runtime.send(r.child(), '');

    expect(r.requests).toHaveLength(1);
  });

  it('never counts below the parked row when the durable count is forged lower', async () => {
    const r = new Rig({ declared: { a: 2 } });
    r.calls([['lookup', { q: 1 }]]);
    await r.runner('a');                          // two steps: lookup, then echo parks
    await rewriteDurable(r, r.child(), (stored) => ({ ...stored, child_steps: { steps: 0, holds: {} } }));
    await recordApproval(r.child(), (await r.pending(r.child()))[0]!, true);

    await r.runtime.send(r.child(), '');

    expect(r.requests).toHaveLength(2);
  });

  it('leaves an approval refused under a cancelled context pending, to be answered later', async () => {
    const r = new Rig({ lead: 4 });
    await r.runtime.send(r.session(), 'Go');
    const [id] = await r.pending(r.session());
    const cancelled = RunContext.root('host', new RunBudget(8));
    cancelled.ledger.cancel('stopped by host');
    await expect(drain(r.runtime.streamDecision(r.session(), [{ approvalId: id!, approved: true }], cancelled))).rejects.toMatchObject({ code: 'run_not_permitted' });

    await recordApproval(r.session(), id!, true);
    await r.runtime.send(r.session(), '');

    expect(r.ran.echo).toBe(1);
  });

  it('runs an approval skipped under a cancelled context exactly once, on the next resume', async () => {
    const r = new Rig({ lead: 4 });
    await r.runtime.send(r.session(), 'Go');
    await recordApproval(r.session(), (await r.pending(r.session()))[0]!, true);
    const cancelled = RunContext.root('host', new RunBudget(8));
    cancelled.ledger.cancel('stopped by host');
    await r.runtime.send(r.session(), '', undefined, cancelled);
    expect(r.ran.echo ?? 0).toBe(0);

    await r.runtime.send(r.session(), '');
    await r.runtime.send(r.session(), 'Next').catch(() => null);

    expect(r.ran.echo).toBe(1);
  });

  it('refuses a task of Unicode whitespace only', async () => {
    const r = new Rig({ declared: { a: 2 } });
    await r.session().usingMode('lead');

    const result = await new SubagentRunner(r.harness, r.runtime)
      .run(r.modes.resolve('lead').subagents.a!, r.session(), RunContext.root('root', new RunBudget(8)), 'root', '　 ');

    expect(result.outcome).toBe('denied');
    expect(await rowCount(r.child())).toBe(0);
  });

  it('frames the refusal of a delegation into a parked child as a result the parent continues from', async () => {
    const r = new Rig({ declared: { a: 2 } });
    r.calls([['a', { task: 'One.' }]]);    // parent
    r.calls([['echo', { n: 0 }]]);         // child: parks
    r.calls([['a', { task: 'Two.' }]]);    // parent delegates again
    r.text('Parent done.');                // parent continues
    const response = await r.runtime.send(r.session(), 'Go');

    expect(response.finishReason).toBe('stop');
    expect(response.text).toBe('Parent done.');
    const refusals = (await r.session().thread().messages()).map((entry) => JSON.stringify(entry.message)).filter((row) => row.includes('waiting on a person'));
    expect(refusals).toHaveLength(1);
  });
});
