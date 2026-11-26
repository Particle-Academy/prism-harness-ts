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

// G-387 round 5 (the fourth final review's findings). The rig is the one of
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
  /** The run context the host wires each subagent tool with. */
  toolContext?: () => RunContext;
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
  /** When set, a child's model call throws this. */
  failChild: Error | null = null;
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
        if (this.failChild !== null && request.systemPrompt.startsWith('Work.')) throw this.failChild;
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
        .tool(this.modes.resolve('lead').subagents[name]!, this.session(), this.options.toolContext?.() ?? RunContext.root('host', new RunBudget(100)), 'host_run'));
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

// G-387 round 4: the third final review's findings, at the public seams.

// G-387 round 5: the fourth final review's findings, at the public seams.

/** A child the runner started, parked on an approval, whose binding is gone: nobody can answer it. */
async function deadApproval(r: Rig): Promise<void> {
  const child = r.child();
  await child.rememberChildRefs({ tree: { session: r.session().key(), run: 'run_root' }, planLedger: null });
  await child.usingMode('work');
  await child.thread().record([{ type: 'user', content: 'Work.' }], 'run_child');
  const row = assistantRow('', [{ id: 'e1', name: 'echo', arguments: {} }], {}, [{ approval_id: 'apr_dead', tool_call_id: 'e1', arguments_format: 1 }]) as unknown as JsonObject;
  row.harness_tool_names = ['echo'];
  await child.thread().record([row], 'run_child');
}

describe('a refused runner start writes nothing durable (item 1)', () => {
  for (const [label, stop] of [
    ['cancelled', (context: RunContext) => { context.ledger.cancel('stopped by host'); }],
    ['out of time', (_context: RunContext) => { advanceClock(120); }],
  ] as const) {
    it(`leaves the child's durable record byte-identical when its run is ${label} between the pre-check and send`, async () => {
      const r = new Rig({ declared: { a: 2 } });
      await r.session().usingMode('lead');
      await deadApproval(r);
      await expect(recordApproval(r.child(), 'apr_dead', true)).rejects.toMatchObject({ code: 'child_binding_missing' });
      const key = `${r.child().key()}:durable`;
      const before = JSON.stringify(await r.durable.get(key));
      const context = RunContext.root('root', new RunBudget(8, null, 60));
      const usingMode = Session.prototype.usingMode;
      vi.spyOn(Session.prototype, 'usingMode').mockImplementation(async function (this: Session, ...args: Parameters<Session['usingMode']>) {
        if (this.scope.includes('::sub::')) stop(context);
        return usingMode.apply(this, args);
      });

      const result = await new SubagentRunner(r.harness, r.runtime).run(r.modes.resolve('lead').subagents.a!, r.session(), context, 'root', 'Start over');
      vi.restoreAllMocks();

      expect(result.outcome).toBe('failed');
      expect(JSON.stringify(await r.durable.get(key))).toBe(before);
      await expect(recordApproval(r.child(), 'apr_dead', true)).rejects.toMatchObject({ code: 'child_binding_missing' });
      expect(r.requests).toHaveLength(0);
    });
  }
});

describe('a host cancel reaches the subagent tool inside a turn (item 2)', () => {
  for (const [label, outcome, context] of [
    ['cancelled', 'cancelled', () => { const c = RunContext.root('host', new RunBudget(100)); c.ledger.cancel('stopped by operator'); return c; }],
    ['out of steps', 'exhausted', () => RunContext.root('host', new RunBudget(0))],
    ['with an unenforceable cost cap', 'exhausted', () => { const c = RunContext.root('host', new RunBudget(100, 1)); c.ledger.recordCost(null); return c; }],
  ] as const) {
    it(`returns the framed ${outcome} result for a host context ${label}, with no child model request`, async () => {
      const r = new Rig({ declared: { a: 2 }, toolContext: context });
      r.calls([['a', { task: 'Work.' }]]);   // parent delegates
      r.text('Parent done.');                // parent continues

      const response = await r.runtime.send(r.session(), 'Go');

      expect(response.text).toBe('Parent done.');
      expect(r.requests).toHaveLength(2);
      expect(r.requests.filter((request) => request.systemPrompt.startsWith('Work.'))).toHaveLength(0);
      expect(await rowCount(r.child())).toBe(0);
      expect(sentAfter(r, 1)).toContain('subagent did not run');
    });
  }
});

describe('a framed failure names a class, whatever was thrown (item 3)', () => {
  async function thrown(value: unknown): Promise<string> {
    const r = new Rig({ declared: { a: 2 } });
    const beginRun = Session.prototype.beginRun;
    vi.spyOn(Session.prototype, 'beginRun').mockImplementation(function (this: Session, ...args: Parameters<Session['beginRun']>) {
      if (this.scope.includes('::sub::')) throw value;
      return beginRun.apply(this, args);
    });
    r.calls([['a', { task: 'Work.' }]]);
    r.text('Parent done.');
    const response = await r.runtime.send(r.session(), 'Go');
    expect(response.text).toBe('Parent done.');

    return sentAfter(r, 1);
  }

  it('falls back to Error when the constructor getter throws', async () => {
    const error = new Error(SECRET);
    Object.defineProperty(error, 'constructor', { get: () => { throw new Error(`${SECRET} from the getter`); } });

    const sent = await thrown(error);

    expect(sent).not.toContain('STORE-SECRET-TEXT');
    expect(sent).toContain('it failed (Error)');
  });

  it('falls back to Error when the class name is arbitrary text', async () => {
    class Named extends Error {}
    Object.defineProperty(Named, 'name', { value: `${SECRET} name` });

    const sent = await thrown(new Named('x'));

    expect(sent).not.toContain('STORE-SECRET-TEXT');
    expect(sent).toContain('it failed (Error)');
  });

  it('falls back to Error for a thrown object whose constructor names itself freely', async () => {
    const sent = await thrown({ constructor: { name: `${SECRET} object` } });

    expect(sent).not.toContain('STORE-SECRET-TEXT');
  });

  it('keeps a plain class name', async () => {
    expect(await thrown(new TypeError('x'))).toContain('it failed (TypeError)');
  });
});

// The channels each round-5 fix could open, each with its test.
describe('new channels considered (round 5)', () => {
  it('leaves the delegation count untouched when a new prompt on a parked child is refused under a cancelled context', async () => {
    const r = new Rig({ declared: { a: 3 } });
    await r.runner('a');
    const key = `${r.child().key()}:durable`;
    const before = JSON.stringify(await r.durable.get(key));
    const cancelled = RunContext.root('host', new RunBudget(8));
    cancelled.ledger.cancel('stopped by host');

    await expect(r.runtime.send(r.child(), 'New question', undefined, cancelled)).rejects.toMatchObject({ code: 'run_not_permitted' });

    expect(JSON.stringify(await r.durable.get(key))).toBe(before);
  });

  it('still delegates under a host context with steps and time left', async () => {
    const r = new Rig({ declared: { a: 2 }, toolContext: () => RunContext.root('host', new RunBudget(100, null, 600)) });
    r.calls([['a', { task: 'Work.' }]]);
    r.text('Child done.');
    r.text('Parent done.');

    await r.runtime.send(r.session(), 'Go');

    expect(r.requests.filter((request) => request.systemPrompt.startsWith('Work.'))).toHaveLength(1);
  });

  it('keeps a harness error code with a dot, and drops one that is not a code', async () => {
    const r = new Rig({ declared: { a: 2 } });
    const beginRun = Session.prototype.beginRun;
    let code = 'plan.budget_context';
    vi.spyOn(Session.prototype, 'beginRun').mockImplementation(function (this: Session, ...args: Parameters<Session['beginRun']>) {
      if (this.scope.includes('::sub::')) throw new HarnessError(code as ConstructorParameters<typeof HarnessError>[0], SECRET);
      return beginRun.apply(this, args);
    });
    r.calls([['a', { task: 'Work.' }]]);
    r.text('Parent done.');
    await r.runtime.send(r.session(), 'Go');
    expect(sentAfter(r, 1)).toContain('HarnessError plan.budget_context');

    code = `${SECRET} code`;
    const from = r.requests.length;
    r.calls([['a', { task: 'Again.' }]]);
    r.text('Parent done.');
    await r.runtime.send(r.session(), 'Go again');
    expect(sentAfter(r, from)).not.toContain('STORE-SECRET-TEXT');
  });
});
