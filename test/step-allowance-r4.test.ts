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

// G-387 round 4 (the third final review's findings). The rig is the one of
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

describe('no exception text reaches the parent model, by any route (item 1)', () => {
  async function delegateOnce(r: Rig): Promise<string> {
    r.calls([['a', { task: 'Work.' }]]);   // parent delegates
    r.text('Parent done.');                // parent continues
    const response = await r.runtime.send(r.session(), 'Go');
    expect(response.text).toBe('Parent done.');

    return sentAfter(r, 1);
  }

  it('frames a failing store write on a child key', async () => {
    const r = new Rig({ declared: { a: 2 } });
    const put = MemorySessionStore.prototype.put;
    vi.spyOn(MemorySessionStore.prototype, 'put').mockImplementation(function (this: MemorySessionStore, key: string, ...rest: [JsonObject, (number | null)?]) {
      if (key.includes('::sub::')) throw new RangeError(SECRET);
      return put.call(this, key, ...rest);
    });

    const sent = await delegateOnce(r);

    expect(sent).not.toContain('STORE-SECRET-TEXT');
    expect(sent).toContain('RangeError');
  });

  it('frames a child provider client raising with a secret in its message', async () => {
    const r = new Rig({ declared: { a: 2 } });
    r.failChild = new EvalError(SECRET);

    const sent = await delegateOnce(r);

    expect(sent).not.toContain('STORE-SECRET-TEXT');
    expect(sent).toContain('EvalError');
  });

  it('frames a host run context whose ledger getter throws', async () => {
    const ledger = RunLedger.start('host');
    Object.defineProperty(ledger, 'steps', { get: () => { throw new URIError(SECRET); } });
    const r = new Rig({ declared: { a: 2 }, toolContext: () => new RunContext(ledger, new RunBudget(100)) });

    const sent = await delegateOnce(r);

    expect(sent).not.toContain('STORE-SECRET-TEXT');
    expect(sent).toContain('URIError');
  });

  it('frames a host run context with a forged max_steps of 1.5', async () => {
    const budget = { maxSteps: 1.5, maxCostUsd: null, maxSeconds: null, maxOutputTokensPerCall: null } as unknown as RunBudget;
    const r = new Rig({ declared: { a: 2 }, toolContext: () => new RunContext(RunLedger.start('host'), budget) });

    const sent = await delegateOnce(r);

    expect(sent).toContain('did not run');
    expect(sent).not.toContain('step budget must be');
  });
});

describe('the re-delegation refusal is atomic (item 2)', () => {
  it('refuses a runner start whose child a resume claimed between the check and the reset, within one ceiling', async () => {
    const r = new Rig({ declared: { a: 3 } });
    await r.runner('a');                          // one step, parks
    await recordApproval(r.child(), (await r.pending(r.child()))[0]!, true);
    for (let i = 0; i < 4; i += 1) r.calls([['lookup', { q: i }]]);
    let release: () => void = () => {};
    let resume: Promise<unknown> | null = null;
    // G-387 round 7: the start's gate reads the child, then commits its
    // records in one write that re-checks the count; the resume claims in
    // between.
    const remember = Session.prototype.rememberChildStart;
    vi.spyOn(Session.prototype, 'rememberChildStart').mockImplementation(async function (this: Session, ...args: Parameters<Session['rememberChildStart']>) {
      if (this.scope.includes('::sub::') && resume === null) {
        // Between the gate's read and the child write, a resume claims the
        // child and is mid model call.
        const started = new Promise<void>((resolve) => {
          r.beforeRequest = async () => { resolve(); await new Promise<void>((done) => { release = done; }); };
        });
        resume = r.runtime.send(r.child(), '').catch(() => null);
        await started;
      }
      return remember.apply(this, args);
    });

    const result = await runner(r, 'Start over');
    release();
    await resume;

    expect(result.outcome).toBe('denied');
    expect(result.text).toContain('a turn on it is still running');
    expect(r.requests.length).toBeLessThanOrEqual(3);
  });
});

describe('a ceiling with no count record fails closed (item 3, as PHP\'s missing delegation)', () => {
  it('takes no model step for a child that saved a ceiling and no count', async () => {
    const r = new Rig({ declared: { a: 4 } });
    await r.runner('a');
    await rewriteDurable(r, r.child(), ({ child_steps: _gone, ...rest }) => rest);
    await recordApproval(r.child(), (await r.pending(r.child()))[0]!, true);

    const response = await r.runtime.send(r.child(), '');

    expect(r.requests).toHaveLength(1);
    expect(response.finishReason).toBe('budget_exhausted');
  });
});

describe('a resume under a spent time limit acts on no recorded decision (item 4)', () => {
  const late = () => RunContext.root('host', new RunBudget(8, null, 0));

  it('runs no recorded approval on a resume under a context whose time is spent, and leaves it for a later resume', async () => {
    const r = new Rig({ lead: 4 });
    await r.runtime.send(r.session(), 'Go');
    await recordApproval(r.session(), (await r.pending(r.session()))[0]!, true);

    await r.runtime.send(r.session(), '', undefined, late());
    expect(r.ran.echo ?? 0).toBe(0);

    await r.runtime.send(r.session(), '');
    expect(r.ran.echo).toBe(1);
  });

  it('runs no earlier-recorded approval on the resume a denial-only batch starts under spent time', async () => {
    const r = new Rig({ lead: 4 });
    r.calls([['echo', { n: 1 }], ['echo', { n: 2 }]]);
    await r.runtime.send(r.session(), 'Go');
    const [a, b] = await r.pending(r.session());
    await recordApproval(r.session(), a!, true);

    await drain(r.runtime.streamDecision(r.session(), [{ approvalId: b!, approved: false }], late()));

    expect(r.ran.echo ?? 0).toBe(0);
    expect(r.requests).toHaveLength(1);
    await r.runtime.send(r.session(), '');
    expect(r.ran.echo).toBe(1);
  });
});

// The channels each round-4 fix could open, each with its test.
describe('new channels considered (round 4)', () => {
  // A resume that acts on nothing leaves the decisions recorded; a NEW
  // prompt under such a context would record a user row that ends the turn
  // those decisions belong to, so they would never be acted on. Refused
  // before anything is recorded instead.
  for (const [label, context] of [
    ['whose time is spent', () => RunContext.root('host', new RunBudget(8, null, 0))],
    ['that was cancelled', () => { const c = RunContext.root('host', new RunBudget(8)); c.ledger.cancel('stopped by host'); return c; }],
  ] as const) {
    it(`refuses a new prompt under a context ${label}, recording nothing, so a recorded approval is still acted on later`, async () => {
      const r = new Rig({ lead: 4 });
      await r.runtime.send(r.session(), 'Go');
      await recordApproval(r.session(), (await r.pending(r.session()))[0]!, true);
      const rows = await rowCount(r.session());

      await expect(r.runtime.send(r.session(), 'New question', undefined, context())).rejects.toMatchObject({ code: 'run_not_permitted' });
      expect(await rowCount(r.session())).toBe(rows);

      await r.runtime.send(r.session(), '');
      expect(r.ran.echo).toBe(1);
    });
  }

  it('lets the parent continue from a framed host-context failure', async () => {
    const ledger = RunLedger.start('host');
    Object.defineProperty(ledger, 'steps', { get: () => { throw new URIError(SECRET); } });
    const r = new Rig({ declared: { a: 2 }, toolContext: () => new RunContext(ledger, new RunBudget(100)) });
    r.calls([['a', { task: 'Work.' }]]);
    r.text('Parent done.');

    const response = await r.runtime.send(r.session(), 'Go');

    expect(response.finishReason).toBe('stop');
    expect(r.requests).toHaveLength(2);
  });
});
