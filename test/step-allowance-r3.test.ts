import { readFileSync, writeFileSync } from 'node:fs';
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
  type SessionStore,
} from '../src/index.js';
import { HarnessError, Session } from '../src/index.js';
import { RunTrees } from '../src/run-trees.js';

// G-387 round 3 (the second final review's findings). The rig is the one of
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

// G-387 round 3: the second final review's findings, at the public seams.

const SECRET = 'STORE-SECRET-TEXT at /var/db/prism';

/** Every message of every provider request after the first `from`, as text. */
function sentAfter(r: Rig, from: number): string {
  return r.requests.slice(from).map((request) => JSON.stringify(request.messages)).join('\n');
}

/** Advance the clock every Date.now() reads by `seconds`, from now on. */
function advanceClock(seconds: number): void {
  const now = Date.now;
  const shifted = now() + seconds * 1000;
  const start = now();
  vi.spyOn(Date, 'now').mockImplementation(() => shifted + (now() - start));
}

afterEach(() => { vi.restoreAllMocks(); });

const runner = (r: Rig, task: unknown, name = 'a'): Promise<{ outcome: string; text: string }> =>
  new SubagentRunner(r.harness, r.runtime).run(r.modes.resolve('lead').subagents[name]!, r.session(), RunContext.root('root', new RunBudget(8)), 'root', task as string);

describe('the runner frames every failure around starting a child (item 1)', () => {
  it('frames a failure reading the child, inside a turn, with no exception text reaching the model', async () => {
    const r = new Rig({ declared: { a: 2 } });
    const thread = Session.prototype.thread;
    vi.spyOn(Session.prototype, 'thread').mockImplementation(function (this: Session) {
      if (this.scope.includes('::sub::')) throw new RangeError(SECRET);
      return thread.call(this);
    });
    r.calls([['a', { task: 'Work.' }]]);   // parent delegates
    r.text('Parent done.');                // parent continues
    const response = await r.runtime.send(r.session(), 'Go');

    expect(response.text).toBe('Parent done.');
    expect(sentAfter(r, 1)).not.toContain('STORE-SECRET-TEXT');
    expect(sentAfter(r, 1)).toContain('RangeError');
  });

  it('frames a failure while the child runs, with no exception text reaching the model', async () => {
    const r = new Rig({ declared: { a: 2 } });
    const beginRun = Session.prototype.beginRun;
    vi.spyOn(Session.prototype, 'beginRun').mockImplementation(function (this: Session, ...args: Parameters<Session['beginRun']>) {
      if (this.scope.includes('::sub::')) throw new SyntaxError(SECRET);
      return beginRun.apply(this, args);
    });
    r.calls([['a', { task: 'Work.' }]]);
    r.text('Parent done.');
    await r.runtime.send(r.session(), 'Go');

    expect(sentAfter(r, 1)).not.toContain('STORE-SECRET-TEXT');
    expect(sentAfter(r, 1)).toContain('SyntaxError');
  });

  for (const task of [null, undefined, 42, { text: 'go' }, ['go']]) {
    it(`denies a task that is not a string (${JSON.stringify(task) ?? 'undefined'})`, async () => {
      const r = new Rig({ declared: { a: 2 } });
      await r.session().usingMode('lead');

      const result = await runner(r, task);

      expect(result.outcome).toBe('denied');
      expect(r.requests).toHaveLength(0);
      expect(await rowCount(r.child())).toBe(0);
    });
  }
});

describe('a durable count that is not a count fails closed (item 2)', () => {
  const shapes: [string, (stored: JsonObject) => JsonObject][] = [
    ['no count', ({ child_steps: _gone, ...rest }) => rest],
    ['steps -1', (stored) => ({ ...stored, child_steps: { steps: -1, holds: {} } })],
    ['steps 1.5', (stored) => ({ ...stored, child_steps: { steps: 1.5, holds: {} } })],
    ['steps "1"', (stored) => ({ ...stored, child_steps: { steps: '1', holds: {} } })],
    ['no holds', (stored) => ({ ...stored, child_steps: { steps: 1 } })],
    ['holds a list', (stored) => ({ ...stored, child_steps: { steps: 1, holds: [] } })],
    ['a hold of 1.5 steps', (stored) => ({ ...stored, child_steps: { steps: 1, holds: { x: { steps: 1.5, at: 1 } } } })],
    ['a hold at 1.5', (stored) => ({ ...stored, child_steps: { steps: 1, holds: { x: { steps: 0, at: 1.5 } } } })],
    ['a hold at -1', (stored) => ({ ...stored, child_steps: { steps: 1, holds: { x: { steps: 0, at: -1 } } } })],
    ['a hold at "1"', (stored) => ({ ...stored, child_steps: { steps: 1, holds: { x: { steps: 0, at: '1' } } } })],
  ];

  for (const [label, change] of shapes) {
    it(`takes no model step, and never repairs the record, for ${label}`, async () => {
      const r = new Rig({ declared: { a: 4 } });
      await r.runner('a');
      await rewriteDurable(r, r.child(), change);
      const before = (await r.durable.get(`${r.child().key()}:durable`))!.child_steps ?? null;
      await recordApproval(r.child(), (await r.pending(r.child()))[0]!, true);

      const response = await r.runtime.send(r.child(), '');

      expect(r.requests).toHaveLength(1);
      expect(response.finishReason).toBe('budget_exhausted');
      expect((await r.durable.get(`${r.child().key()}:durable`))!.child_steps ?? null).toEqual(before);
    });
  }
});

describe('a child a released version parked (item 3)', () => {
  // Recorded from a parent turn on v0.11.2 (39229d6): a subagent declared
  // max_steps 2 in a mode of 8 parked on an approval. v0.11.2 saved no
  // child fields and no binding, tree or count on the row.
  async function released(): Promise<Rig> {
    const fixture = JSON.parse(readFileSync(new URL('./fixtures/legacy-child-0.11.2.json', import.meta.url), 'utf8')) as { files: Record<string, string> };
    const directory = await mkdtemp(join(tmpdir(), 'g387-released-'));
    for (const [name, content] of Object.entries(fixture.files)) writeFileSync(join(directory, name), content);

    return new Rig({ declared: { worker: 2 }, directory });
  }

  it('fails closed on the approval: the call runs, no model step', async () => {
    const r = await released();
    const child = r.child('worker');
    const [id] = await r.pending(child);
    expect(id).toBeDefined();
    const echoes = r.ran.echo ?? 0;

    await r.approveLoop(() => r.child('worker'), none);

    expect(r.ran.echo).toBe(echoes + 1);
    expect(r.requests).toHaveLength(0);
  });

  it('refuses a new message to it before anything is recorded', async () => {
    const r = await released();
    const rows = await rowCount(r.child('worker'));

    await expect(r.runtime.send(r.child('worker'), 'New question')).rejects.toMatchObject({ code: 'run_not_permitted' });

    expect(await rowCount(r.child('worker'))).toBe(rows);
    expect(r.requests).toHaveLength(0);
  });
});

describe('a turn that outlives the lock TTL is charged what it took (item 4)', () => {
  it('leaves 3 of 4 for the approved resume after a 400 s turn that took 1', async () => {
    const r = new Rig({ declared: { a: 4 } });
    r.beforeRequest = async () => { advanceClock(400); };
    await r.runner('a');                          // one step, 400 s, parks
    vi.restoreAllMocks();

    await r.approveLoop(() => r.child(), none);

    expect(r.requests).toHaveLength(4);
  });

  it('counts a hold that expired and then settled once', async () => {
    const r = new Rig({ declared: { a: 4 } });
    let refused: unknown = null;
    r.beforeRequest = async () => {
      advanceClock(400);
      // Another turn on the child finds the hold expired, charges it in full,
      // and is refused for want of a step.
      await r.runtime.send(r.child(), 'Meanwhile').catch((error: unknown) => { refused = error; });
    };
    await r.runner('a');
    vi.restoreAllMocks();
    expect(refused).toMatchObject({ code: 'run_not_permitted' });

    await r.approveLoop(() => r.child(), none);

    expect(r.requests).toHaveLength(4);
  });
});

describe('no delegation into a child while a turn on it runs (item 5)', () => {
  it('denies the runner while an outside turn on the child holds part of its count', async () => {
    const r = new Rig({ declared: { a: 3 } });
    await r.runner('a');
    await recordApproval(r.child(), (await r.pending(r.child()))[0]!, true);
    let inner: { outcome: string } | null = null;
    let during: unknown = null;
    r.beforeRequest = async () => {
      inner = await runner(r, 'Start over');
      during = (await r.durable.get(`${r.child().key()}:durable`))!.child_steps;
    };

    await r.runtime.send(r.child(), '');

    expect(inner!.outcome).toBe('denied');
    expect(Object.keys((during as { holds: JsonObject }).holds)).toHaveLength(1);
    expect(r.requests).toHaveLength(2);
  });

  it('refuses a claim for a run id that already holds', async () => {
    const r = new Rig({ declared: { a: 3 } });
    await r.runner('a');
    const parked = (await r.child().run())!.id;
    const now = Math.floor(Date.now() / 1000);
    await rewriteDurable(r, r.child(), (stored) => ({ ...stored, child_steps: { steps: 1, holds: { [parked]: { steps: 1, at: now } } } }));

    const response = await drain(r.runtime.streamDecision(r.child(), [{ approvalId: (await r.pending(r.child()))[0]!, approved: true }]));

    expect(r.requests).toHaveLength(1);
    expect(response.finishReason).toBe('budget_exhausted');
  });
});

describe('a blank task is one of an explicit set of codepoints (item 6)', () => {
  const BLANK = [0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x20, 0x85, 0xa0, 0x1680,
    0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a, 0x200b, 0x200c, 0x200d,
    0x2028, 0x2029, 0x202f, 0x205f, 0x2060, 0x3000, 0xfeff];

  for (const codepoint of BLANK) {
    it(`denies a task of U+${codepoint.toString(16).toUpperCase().padStart(4, '0')} alone`, async () => {
      const r = new Rig({ declared: { a: 2 } });
      await r.session().usingMode('lead');

      expect((await runner(r, String.fromCodePoint(codepoint))).outcome).toBe('denied');
      expect(await rowCount(r.child())).toBe(0);
    });
  }

  it('denies a mixed run of them', async () => {
    const r = new Rig({ declared: { a: 2 } });
    await r.session().usingMode('lead');

    expect((await runner(r, BLANK.map((c) => String.fromCodePoint(c)).join(''))).outcome).toBe('denied');
  });

  for (const task of ['a', '​go​', '᠎', '­', '.']) {
    it(`delegates a task with any other character (${JSON.stringify(task)})`, async () => {
      const r = new Rig({ declared: { a: 2 } });
      await r.session().usingMode('lead');
      r.text('Done.');

      expect((await runner(r, task)).outcome).toBe('completed');
    });
  }
});

describe('a pending approval blocks re-delegation only while a person could answer it (item 7)', () => {
  async function parkedChild(r: Rig, change: (stored: JsonObject) => JsonObject): Promise<void> {
    // A session the runner started, parked on an approval, by hand.
    const child = r.child();
    await child.rememberChildRefs({ tree: { session: r.session().key(), run: 'run_root' }, planLedger: null });
    await child.usingMode('work');
    await child.thread().record([{ type: 'user', content: 'Work.' }], 'run_child');
    const row = assistantRow('', [{ id: 'e1', name: 'echo', arguments: {} }], {}, [{ approval_id: 'apr_child', tool_call_id: 'e1', arguments_format: 1 }]) as unknown as JsonObject;
    row.harness_tool_names = ['echo'];
    await child.thread().record([row], 'run_child');
    await rewriteDurable(r, child, change);
  }

  it('lets the parent start again a child whose binding is missing', async () => {
    const r = new Rig({ declared: { a: 2 } });
    await r.session().usingMode('lead');
    await parkedChild(r, (stored) => stored);
    r.text('Done.');

    expect((await runner(r, 'Start over')).outcome).toBe('completed');
  });

  it('lets the parent start again a child saved too deep to answer', async () => {
    const r = new Rig({ declared: { a: 2 } });
    await r.session().usingMode('lead');
    await parkedChild(r, (stored) => ({ ...stored, child_binding: { session: { provider: null, model: null }, declared: { provider: null, model: null }, parent_mode: { provider: null, model: null } }, child_depth: 4 }));
    r.text('Done.');

    expect((await runner(r, 'Start over')).outcome).toBe('completed');
  });

  it('still denies it while the approval is one a person can answer', async () => {
    const r = new Rig({ declared: { a: 2 } });
    await r.runner('a');

    expect((await runner(r, 'Start over')).outcome).toBe('denied');
  });
});

describe('a step budget is a whole number (item 8)', () => {
  for (const steps of [1.5, 2 ** 60, 0.5]) {
    it(`refuses RunBudget(${steps}) and fromConfig({ max_steps: ${steps} })`, () => {
      expect(() => new RunBudget(steps)).toThrow(TypeError);
      expect(() => RunBudget.fromConfig({ max_steps: steps })).toThrow(TypeError);
    });
  }

  it('still accepts a whole number', () => {
    expect(new RunBudget(3).maxSteps).toBe(3);
    expect(RunBudget.fromConfig({ max_steps: 3 }).maxSteps).toBe(3);
  });
});

// The channels each round-3 fix could open, each with its test.
describe('new channels considered (round 3)', () => {
  it('carries a HarnessError\'s code, never its message, in a framed failure', async () => {
    const r = new Rig({ declared: { a: 2 } });
    const beginRun = Session.prototype.beginRun;
    vi.spyOn(Session.prototype, 'beginRun').mockImplementation(function (this: Session, ...args: Parameters<Session['beginRun']>) {
      if (this.scope.includes('::sub::')) throw new HarnessError('session_locked', SECRET);
      return beginRun.apply(this, args);
    });
    r.calls([['a', { task: 'Work.' }]]);
    r.text('Parent done.');
    await r.runtime.send(r.session(), 'Go');

    expect(sentAfter(r, 1)).not.toContain('STORE-SECRET-TEXT');
    expect(sentAfter(r, 1)).toContain('session_locked');
  });

  it('still reports a child stopped by its spend ceiling as exhausted, not failed', async () => {
    const r = new Rig({ declared: { a: 2 } });
    await r.session().usingMode('lead');
    const beginRun = Session.prototype.beginRun;
    vi.spyOn(Session.prototype, 'beginRun').mockImplementation(function (this: Session, ...args: Parameters<Session['beginRun']>) {
      if (this.scope.includes('::sub::')) throw new HarnessError('agent_spend_ceiling_exceeded', SECRET);
      return beginRun.apply(this, args);
    });

    const result = await runner(r, 'Work.');

    expect(result.outcome).toBe('exhausted');
    expect(result.text).not.toContain('STORE-SECRET-TEXT');
  });

  it('lets a runner start replace a count that is not a count, so a garbage record never wedges a child', async () => {
    const r = new Rig({ declared: { a: 2 } });
    r.text('First.');
    await r.runner('a');
    await rewriteDurable(r, r.child(), (stored) => ({ ...stored, child_steps: 'lots' }));
    await expect(r.runtime.send(r.child(), 'Question')).rejects.toMatchObject({ code: 'run_not_permitted' });

    r.text('Again.');
    expect((await runner(r, 'Start over')).outcome).toBe('completed');
  });

  it('fails a root session whose host-chosen scope looks like a child\'s closed, not open', async () => {
    const r = new Rig({ lead: 4 });
    const odd = r.session('chat::sub::odd');
    await odd.usingMode('lead');

    await expect(r.runtime.send(odd, 'Go')).rejects.toMatchObject({ code: 'run_not_permitted' });
    expect(r.requests).toHaveLength(0);
  });

  it('denies re-delegation while a crashed turn\'s hold is live, and allows it once the hold expired', async () => {
    const r = new Rig({ declared: { a: 2 } });
    r.text('First.');
    await r.runner('a');
    const now = Math.floor(Date.now() / 1000);
    await rewriteDurable(r, r.child(), (stored) => ({ ...stored, child_steps: { steps: 1, holds: { run_crashed: { steps: 1, at: now } }, charged: {} } }));
    expect((await runner(r, 'Start over')).outcome).toBe('denied');

    await rewriteDurable(r, r.child(), (stored) => ({ ...stored, child_steps: { steps: 1, holds: { run_crashed: { steps: 1, at: 1 } }, charged: {} } }));
    r.text('Again.');
    expect((await runner(r, 'Start over')).outcome).toBe('completed');
  });

  it('never subtracts another run\'s charge when a run settles', async () => {
    const r = new Rig({ declared: { a: 4 } });
    await r.runner('a');                          // one step, parks: count 1
    await rewriteDurable(r, r.child(), (stored) => ({ ...stored, child_steps: { steps: 3, holds: {}, charged: { run_other: 2 } } }));

    await r.approveLoop(() => r.child(), none);

    expect(r.requests).toHaveLength(2);           // 3 used: one more of 4
  });
});

// Item 9, set for all three ports: acting on a denial runs nothing, so a
// batch of denials only is not refused for a spent time limit of the
// supplied context; a cancelled context still refuses it, and a batch with
// an approval in it keeps the full check.
describe('a batch of denials and the supplied context\'s limits (item 9)', () => {
  async function parkedOnTwo(r: Rig): Promise<[string, string]> {
    r.calls([['echo', { n: 1 }], ['echo', { n: 2 }]]);   // one step asking for two approvals
    await r.runtime.send(r.session(), 'Go');
    const pending = await r.pending(r.session());
    expect(pending).toHaveLength(2);

    return [pending[0]!, pending[1]!];
  }

  it('records a batch of denials under a context whose time is spent, and runs nothing', async () => {
    const r = new Rig({ lead: 4 });
    const [a, b] = await parkedOnTwo(r);
    const late = RunContext.root('host', new RunBudget(8, null, 0));

    await drain(r.runtime.streamDecision(r.session(), [{ approvalId: a, approved: false }, { approvalId: b, approved: false }], late));

    expect(await r.pending(r.session())).toEqual([]);
    expect(r.ran.echo ?? 0).toBe(0);
    expect(r.requests).toHaveLength(1);
  });

  it('still refuses a batch of denials under a cancelled context, recording nothing', async () => {
    const r = new Rig({ lead: 4 });
    const [a, b] = await parkedOnTwo(r);
    const rows = await rowCount(r.session());
    const cancelled = RunContext.root('host', new RunBudget(8));
    cancelled.ledger.cancel('stopped by host');

    await expect(drain(r.runtime.streamDecision(r.session(), [{ approvalId: a, approved: false }, { approvalId: b, approved: false }], cancelled)))
      .rejects.toMatchObject({ code: 'run_not_permitted' });

    expect(await rowCount(r.session())).toBe(rows);
    expect(await r.pending(r.session())).toEqual([a, b]);
  });

  it('still refuses a batch with one approval in it under a context whose time is spent, recording nothing', async () => {
    const r = new Rig({ lead: 4 });
    const [a, b] = await parkedOnTwo(r);
    const rows = await rowCount(r.session());
    const late = RunContext.root('host', new RunBudget(8, null, 0));

    await expect(drain(r.runtime.streamDecision(r.session(), [{ approvalId: a, approved: false }, { approvalId: b, approved: true }], late)))
      .rejects.toMatchObject({ code: 'run_not_permitted' });

    expect(await rowCount(r.session())).toBe(rows);
    expect(await r.pending(r.session())).toEqual([a, b]);
    expect(r.ran.echo ?? 0).toBe(0);
  });
});
