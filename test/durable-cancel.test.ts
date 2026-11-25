import { randomUUID } from 'node:crypto';
import { mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  FileSessionStore,
  HarnessError,
  HarnessEvents,
  ModeRegistry,
  Plans,
  PrismHarness,
  RunBudget,
  RunContext,
  SubagentRunner,
  ToolRegistry,
  recordApproval,
  recordDecisions,
  type HarnessEvent,
  type JsonObject,
  type LlmRequest,
  type LlmResponse,
  type ModeConfig,
  type RunStep,
  type Session,
} from '../src/index.js';

// G-285: a durable, cross-process cancel. Every test drives the public API
// (runtime.send / streamSteps / sendStructured, the decision paths and
// session.plan().advance()) with a scripted model client, and fetch is refused
// before the first test. The run executes in the "worker" process; the cancel
// comes from a "web" process: a SEPARATE harness and runtime, built on its own
// FileSessionStore instances over the same temp directory, as a web request
// stopping a queued run would be. Nothing in memory is shared between them.

beforeEach(() => {
  vi.stubGlobal('fetch', () => { throw new Error('A test tried to use the network.'); });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

const REFUSAL = 'Not run: the run was cancelled.';
const WATCHED = new Set(['run.cancel_requested', 'run.cancelled', 'run.finished', 'run.failed']);

type Call = [string, JsonObject];
type Path = 'send' | 'streamSteps' | 'sendStructured';

interface RigOptions {
  requiresApproval?: string[];
  subagents?: boolean;
  plans?: boolean;
}

class Rig {
  readonly queue: LlmResponse[] = [];
  readonly requests: LlmRequest[] = [];
  readonly events: HarnessEvent[] = [];
  readonly ran = { search: 0, weather: 0 };
  readonly during: { search?: () => Promise<void> } = {};
  onRequest: ((request: number) => Promise<void>) | null = null;
  readonly runtime: AgentRuntime;
  readonly harness: PrismHarness;
  readonly modes: ModeRegistry;
  #call = 0;

  private constructor(readonly directory: string, readonly options: RigOptions) {
    this.modes = Rig.modes(options);
    const tools = new ToolRegistry()
      .register({
        name: 'search',
        handle: async () => {
          this.ran.search += 1;
          await this.during.search?.();

          return 'Found it';
        },
      })
      .register({ name: 'weather', handle: () => { this.ran.weather += 1; return 'Sunny'; } });
    const events = this.#listener();
    this.runtime = new AgentRuntime({
      modes: this.modes,
      tools,
      events,
      client: async (request) => {
        this.requests.push(request);
        await this.onRequest?.(this.requests.length);
        const next = this.queue.shift();
        if (next === undefined) throw new Error('The scripted model has no response left.');

        return next;
      },
    });
    const plans = options.plans === true ? new Plans({ modes: this.modes, config: { enabled: true }, events }) : undefined;
    this.harness = this.#harness(this.runtime, plans);
    if (options.subagents === true) {
      tools.register(new SubagentRunner(this.harness, this.runtime)
        .tool(this.modes.resolve('chat').subagents.run_op!, this.worker(), RunContext.root('host', new RunBudget(100)), 'host_run'));
    }
  }

  static async make(options: RigOptions = {}): Promise<Rig> {
    return new Rig(await mkdtemp(join(tmpdir(), 'prism-harness-g285-')), options);
  }

  static modes(options: RigOptions): ModeRegistry {
    return new ModeRegistry({
      default: 'chat',
      modes: options.subagents === true
        ? {
          chat: { system_prompt: 'You delegate.', tools: ['run_op'], max_steps: 6, subagents: { run_op: { mode: 'op_runner', max_steps: 3 } } } as ModeConfig,
          op_runner: { system_prompt: 'You run one Op.', tools: ['search'], max_steps: 4 } as ModeConfig,
        }
        : {
          chat: {
            system_prompt: 'You help.', tools: ['search', 'weather'], max_steps: 6,
            requires_approval: options.requiresApproval ?? [], ...(options.plans === true ? { plans: true } : {}),
          } as ModeConfig,
        },
    });
  }

  #listener(): HarnessEvents {
    const events = new HarnessEvents();
    events.listen((event) => { if (WATCHED.has(event.type)) this.events.push(event); });

    return events;
  }

  /** Each call builds NEW store instances over the shared directory. */
  #harness(runtime: AgentRuntime | undefined, plans?: Plans): PrismHarness {
    const directory = this.directory;

    return new PrismHarness({
      drivers: {
        eph: () => new FileSessionStore(join(directory, 'ephemeral')),
        dur: () => new FileSessionStore(join(directory, 'durable')),
      },
      stores: { ephemeral: 'eph', durable: 'dur' },
      ...(runtime === undefined ? {} : { runtime }),
      ...(plans === undefined ? {} : { plans }),
    });
  }

  /** The queue worker's session: the one that runs. */
  worker(name = 'Ada'): Session {
    return this.harness.for({ type: 'User', id: name }).session('chat');
  }

  /** Another process's session for the same address: its own harness, runtime and store instances. */
  web(name = 'Ada', runtime = true): Session {
    const own = runtime ? new AgentRuntime({
      modes: Rig.modes(this.options),
      tools: new ToolRegistry(),
      events: this.#listener(),
      client: async () => { throw new Error('The web process never calls a model.'); },
    }) : undefined;

    return this.#harness(own).for({ type: 'User', id: name }).session('chat');
  }

  /** The run id as another process reads it: from the store, never from the worker's memory. */
  async runId(name = 'Ada'): Promise<string> {
    const id = (await this.web(name).run())?.id;
    expect(typeof id).toBe('string');

    return id!;
  }

  calls(calls: Call[]): void {
    this.queue.push({ text: '', finishReason: 'tool_calls', toolCalls: calls.map(([name, args]) => ({ id: `call_g285_${++this.#call}`, name, arguments: args })) });
  }

  text(text: string): void {
    this.queue.push({ text, finishReason: 'stop', structured: { answer: text } });
  }

  /** Run the worker's turn on a path, returning the error it ended with (or null). */
  async turn(path: Path, prompt = 'go', name = 'Ada'): Promise<unknown> {
    try {
      if (path === 'send') await this.runtime.send(this.worker(name), prompt);
      else if (path === 'sendStructured') await this.runtime.sendStructured(this.worker(name), prompt, { type: 'object' });
      else await drain(this.runtime.streamSteps(this.worker(name), prompt));
    } catch (error) {
      return error;
    }

    return null;
  }

  async threadText(name = 'Ada'): Promise<string> {
    return JSON.stringify((await this.web(name).thread().messages()).map((entry) => entry.message));
  }

  /** The tool results the thread holds, by call id. */
  async outputs(): Promise<Record<string, string>> {
    const rows = (await this.web().thread().messages()).map((entry) => entry.message);

    return Object.fromEntries(rows.filter((row) => row.type === 'tool_result')
      .flatMap((row) => (row.tool_results as JsonObject[]).map((entry) => [entry.tool_call_id as string, entry.result as string])));
  }

  names(): string[] {
    return this.events.map((event) => event.type);
  }
}

async function drain(stream: AsyncGenerator<RunStep, unknown, undefined>): Promise<RunStep[]> {
  const out: RunStep[] = [];
  for (;;) {
    const next = await stream.next();
    if (next.done === true) return out;
    out.push(next.value);
  }
}

/** Red by assertion on a build without G-285, rather than by a missing method. */
async function cancel(session: Session, runId: string): Promise<string> {
  const candidate = session as unknown as { cancel?: (runId: string) => Promise<string> };
  expect(typeof candidate.cancel, 'Session.cancel() is G-285').toBe('function');

  return candidate.cancel!.call(session, runId);
}

async function approvalStatus(session: Session, approvalId: string): Promise<string> {
  const candidate = session as unknown as { approvalStatus?: (approvalId: string) => Promise<string> };
  expect(typeof candidate.approvalStatus, 'Session.approvalStatus() is G-285').toBe('function');

  return candidate.approvalStatus!.call(session, approvalId);
}

function code(error: unknown): string | null {
  return error instanceof HarnessError ? error.code : null;
}

function field(event: HarnessEvent | undefined, name: string): unknown {
  return (event as unknown as Record<string, unknown> | undefined)?.[name];
}

describe('durable cross-process cancel (G-285)', () => {
  it.each<Path>(['send', 'streamSteps', 'sendStructured'])('stops a run from another process at the next tool, and lets the running tool finish (%s)', async (path) => {
    const r = await Rig.make();
    let outcome: string | null = null;
    r.during.search = async () => { outcome = await cancel(r.web(), await r.runId()); };
    r.calls([['search', { query: 'tigers' }], ['weather', { city: 'Detroit' }]]);
    r.text('This step must never be asked for.');

    const failure = await r.turn(path);

    expect(outcome).toBe('ok');
    expect(code(failure)).toBe('run_cancelled');
    // The in-flight call finished; the next call in the same step did not run.
    expect(r.ran).toEqual({ search: 1, weather: 0 });
    // And no next model step was requested.
    expect(r.requests).toHaveLength(1);
    expect((await r.web().run())?.status).toBe('cancelled');
    expect(await r.threadText()).toContain(REFUSAL);
    expect(r.names()).toEqual(['run.cancel_requested', 'run.cancelled']);
    const [requested, cancelled] = r.events;
    const runId = (await r.web().run())!.id;
    expect(field(requested, 'runId')).toBe(field(cancelled, 'runId'));
    expect(field(requested, 'runId')).toBe(runId);
    expect(field(requested, 'rootRunId')).toBe(runId);
    expect(field(cancelled, 'rootRunId')).toBe(runId);
    expect(field(requested, 'parentRunId')).toBeNull();
    expect(field(requested, 'sessionKey')).toBe(r.web().key());
    expect(field(cancelled, 'sessionKey')).toBe(r.web().key());
    expect(field(requested, 'state')).toBe('running');
  });

  it.each<Path>(['send', 'streamSteps', 'sendStructured'])('stops between steps when the cancel lands while the model is thinking (%s)', async (path) => {
    const r = await Rig.make();
    let outcome: string | null = null;
    r.onRequest = async (request) => { if (request === 1) outcome = await cancel(r.web(), await r.runId()); };
    r.calls([['search', { query: 'tigers' }], ['weather', { city: 'Detroit' }]]);
    r.text('This step must never be asked for.');

    const failure = await r.turn(path);

    expect(outcome).toBe('ok');
    expect(code(failure)).toBe('run_cancelled');
    expect(r.ran).toEqual({ search: 0, weather: 0 });
    expect(r.requests).toHaveLength(1);
    expect((await r.web().run())?.status).toBe('cancelled');
    expect(r.names()).toEqual(['run.cancel_requested', 'run.cancelled']);
  });

  it.each<Path>(['send', 'streamSteps'])('ends a run cancelled when the cancel races its last step (%s)', async (path) => {
    const r = await Rig.make();
    let outcome: string | null = null;
    r.onRequest = async (request) => { if (request === 2) outcome = await cancel(r.web(), await r.runId()); };
    r.calls([['search', { query: 'tigers' }]]);
    r.text('All done.');

    const failure = await r.turn(path);

    // The last step had already been asked for, so its answer is kept; but a
    // cancel that returned ok is never followed by a run that reads completed.
    expect(outcome).toBe('ok');
    expect(code(failure)).toBe('run_cancelled');
    expect((await r.web().run())?.status).toBe('cancelled');
    // The STORED record says so, not only the read of it.
    const stored = (await r.web().state()).run as JsonObject;
    expect(stored.status).toBe('cancelled');
    expect(stored.ended_as).toBe('completed');
    expect(await r.threadText()).toContain('All done.');
    expect(r.requests).toHaveLength(2);
    expect(r.names()).toEqual(['run.cancel_requested', 'run.cancelled']);
  });

  it('refuses to cancel another session\'s run, and that run is untouched', async () => {
    const r = await Rig.make();
    // Bob has a run of his own, so an id that is not his reads as what it is.
    r.text('Bob is done.');
    await r.runtime.send(r.worker('Bob'), 'mine');
    r.events.length = 0;
    const outcome: string[] = [];
    r.during.search = async () => {
      outcome.push(await cancel(r.web('Bob'), await r.runId('Ada')));
      // A session with no run at all touches nothing either.
      outcome.push(await cancel(r.web('Cy'), await r.runId('Ada')));
    };
    r.calls([['search', { query: 'tigers' }], ['weather', { city: 'Detroit' }]]);
    r.text('Both done.');

    expect(await r.turn('send')).toBeNull();
    expect(outcome).toEqual(['refused', 'no-turn']);
    expect(r.ran).toEqual({ search: 1, weather: 1 });
    expect((await r.web('Ada').run())?.status).toBe('completed');
    expect((await r.web('Bob').run())?.status).toBe('completed');
    expect(r.names()).toEqual(['run.finished']);
  });

  it('refuses a forged or malformed run id, whatever the session holds', async () => {
    const r = await Rig.make();
    const outcome: string[] = [];
    r.during.search = async () => {
      const real = await r.runId();
      for (const forged of [randomUUID(), '', real.slice(0, 9), `../${real}`, real.toUpperCase(), `${real}\n`, `${real} `, ` ${real}`]) {
        outcome.push(await cancel(r.web(), forged));
      }
    };
    r.calls([['search', { query: 'tigers' }]]);
    r.text('Done.');

    expect(await r.turn('send')).toBeNull();
    expect(outcome).toHaveLength(8);
    expect([...new Set(outcome)]).toEqual(['refused']);
    expect((await r.web().run())?.status).toBe('completed');
    expect(r.names()).toEqual(['run.finished']);
  });

  it('answers no-turn, not-running and unsupported without changing anything', async () => {
    const r = await Rig.make();
    expect(await cancel(r.web(), randomUUID())).toBe('no-turn');
    // The order of decisions: the store first, then the id's format, then the run.
    expect(await cancel(r.web(), `${randomUUID()}\n`)).toBe('refused');
    expect(await cancel(r.web(), '')).toBe('refused');
    expect(await cancel(r.web('Ada', false), '')).toBe('unsupported');

    r.text('First.');
    r.text('Second.');
    await r.runtime.send(r.worker(), 'one');
    const first = await r.runId();
    await r.runtime.send(r.worker(), 'two');
    const second = await r.runId();
    const before = JSON.stringify(await r.web().state());

    expect(await cancel(r.web(), second)).toBe('not-running');
    // An earlier run of this session is its own, and over.
    expect(await cancel(r.web(), first)).toBe('not-running');
    expect(await cancel(r.web('Ada', false), second)).toBe('unsupported');
    expect((await r.web().run())?.status).toBe('completed');
    expect(JSON.stringify(await r.web().state())).toBe(before);
    expect(r.names()).toEqual(['run.finished', 'run.finished']);
  });

  it('cancels twice harmlessly: one request event, one end, and then not-running', async () => {
    const r = await Rig.make();
    const outcome: string[] = [];
    r.during.search = async () => {
      outcome.push(await cancel(r.web(), await r.runId()));
      outcome.push(await cancel(r.web(), await r.runId()));
    };
    r.calls([['search', { query: 'tigers' }]]);
    r.text('Never.');

    expect(code(await r.turn('send'))).toBe('run_cancelled');
    expect(outcome).toEqual(['ok', 'ok']);
    expect(await cancel(r.web(), await r.runId())).toBe('not-running');
    expect(r.names()).toEqual(['run.cancel_requested', 'run.cancelled']);
  });

  it('settles a run parked on an approval as cancelled, and it never resumes', async () => {
    const r = await Rig.make({ requiresApproval: ['weather'] });
    r.calls([['search', { query: 'tigers' }], ['weather', { city: 'Detroit' }]]);
    const pending = (await r.runtime.send(r.worker(), 'go')).pendingApprovals[0]!;
    const runId = await r.runId();
    expect(await approvalStatus(r.web(), pending.id)).toBe('pending');
    r.events.length = 0;

    expect(await cancel(r.web(), runId)).toBe('ok');
    expect(r.names()).toEqual(['run.cancel_requested', 'run.cancelled']);
    expect(field(r.events[0], 'state')).toBe('parked');
    expect(field(r.events[0], 'runId')).toBe(runId);
    expect(field(r.events[1], 'runId')).toBe(runId);
    expect(await approvalStatus(r.web(), pending.id)).toBe('cancelled');
    expect((await r.web().run())?.status).toBe('cancelled');
    expect(await cancel(r.web(), runId)).toBe('not-running');

    const requests = r.requests.length;
    const rows = await r.threadText();
    r.text('A resumed run must never be asked for.');
    const tries: (() => Promise<unknown>)[] = [
      () => recordApproval(r.worker(), pending.id, true),
      () => recordApproval(r.worker(), pending.id, false),
      () => recordDecisions(r.worker(), [{ approvalId: pending.id, approved: true }]),
      () => r.runtime.decide(r.worker(), pending.id, true),
      () => drain(r.runtime.streamDecision(r.worker(), [{ approvalId: pending.id, approved: true }])),
    ];
    for (const attempt of tries) {
      expect(code(await attempt().then(() => null, (error: unknown) => error))).toBe('approval_not_pending');
    }
    expect(code(await r.turn('send', ''))).toBe('run_cancelled');
    expect(code(await r.turn('streamSteps', ''))).toBe('run_cancelled');
    expect(r.requests).toHaveLength(requests);
    // Refused before anything began: no row written, no run started or ended.
    expect(await r.threadText()).toBe(rows);
    expect(r.names()).toEqual(['run.cancel_requested', 'run.cancelled']);
    expect(r.ran).toEqual({ search: 1, weather: 0 });
    expect(await approvalStatus(r.web(), pending.id)).toBe('cancelled');
    expect(await approvalStatus(r.web(), 'apr_never_issued')).toBe('unknown');
  });

  it('ends a run cancelled, not parked, when the cancel lands in the step that asks for approval', async () => {
    const r = await Rig.make({ requiresApproval: ['weather'] });
    let outcome: string | null = null;
    r.during.search = async () => { outcome = await cancel(r.web(), await r.runId()); };
    r.calls([['search', { query: 'tigers' }], ['weather', { city: 'Detroit' }]]);

    const failure = await r.turn('send');

    expect(outcome).toBe('ok');
    expect(code(failure)).toBe('run_cancelled');
    expect((await r.web().run())?.status).toBe('cancelled');
    expect(r.names()).toEqual(['run.cancel_requested', 'run.cancelled']);
    const approval = JSON.parse(await r.threadText()).flatMap((row: JsonObject) => (row.tool_approval_requests as JsonObject[] | undefined) ?? [])[0] as JsonObject;
    expect(await approvalStatus(r.web(), approval.approval_id as string)).toBe('cancelled');
    expect(code(await r.turn('send', ''))).toBe('run_cancelled');
    expect(r.ran).toEqual({ search: 1, weather: 0 });
    expect(r.requests).toHaveLength(1);
  });

  it('stops the whole run tree when the root is cancelled during a subagent', async () => {
    const r = await Rig.make({ subagents: true });
    let outcome: string | null = null;
    // The person stops the ROOT run, from another process, while the child works.
    r.during.search = async () => { outcome = await cancel(r.web(), await r.runId()); };
    r.calls([['run_op', { task: 'Run the billing Op' }]]);
    r.calls([['search', { query: 'billing' }]]);
    r.text('Never asked for.');

    expect(code(await r.turn('send'))).toBe('run_cancelled');
    expect(outcome).toBe('ok');
    expect(r.ran.search).toBe(1);
    // Parent step, child step; neither took another.
    expect(r.requests).toHaveLength(2);
    expect((await r.web().run())?.status).toBe('cancelled');
    expect(r.names()).toEqual(['run.cancel_requested', 'run.cancelled', 'run.cancelled']);
    // The child ended cancelled too, under its own id, in the root's tree.
    const [, child, root] = r.events;
    expect(field(child, 'runId')).not.toBe(field(root, 'runId'));
    expect(field(child, 'rootRunId')).toBe(field(root, 'runId'));
    expect(field(child, 'parentRunId')).toBe(field(root, 'runId'));
  });

  it('blocks a plan whose step turn was cancelled, and the next advance runs nothing', async () => {
    const r = await Rig.make({ plans: true });
    r.calls([['plan_draft', { goal: 'Plan the trip', steps: [1, 2].map((n) => ({ title: `Step ${n}`, instruction: `Do part ${n}.`, owner: 'agent' })) }]]);
    const pending = (await r.runtime.send(r.worker(), 'Plan the trip')).pendingApprovals[0]!;
    r.text('Plan approved.');
    await recordApproval(r.worker(), pending.id, true);
    await r.runtime.send(r.worker(), '');
    expect((await (await r.worker().plan()).current())?.toJSON().status).toBe('active');

    let outcome: string | null = null;
    const asked = r.requests.length + 1;
    r.onRequest = async (request) => { if (request === asked) outcome = await cancel(r.web(), await r.runId()); };
    r.calls([['plan_complete_step', { summary: 'Booked the flight' }]]);
    r.text('Never asked for.');

    const advanced = await (await r.worker().plan()).advance();
    const plan = (await (await r.worker().plan()).current())?.toJSON() as Record<string, any>;

    expect(outcome).toBe('ok');
    expect(advanced.status).toBe('blocked');
    expect(advanced.reason).toBe('run_cancelled');
    expect(plan.status).toBe('blocked');
    expect(plan.blockedReason).toBe('run_cancelled');
    expect(plan.steps[0].state).toBe('todo');
    expect(r.requests).toHaveLength(asked);

    // Blocked plans do not advance: nothing runs until the host resumes it.
    const again = await (await r.worker().plan()).advance();
    expect(again.ran()).toBe(false);
    expect(again.planStatus).toBe('blocked');
    expect(r.requests).toHaveLength(asked);
  });

  it('runs the next turn after a cancelled run normally, with or without a prompt', async () => {
    const r = await Rig.make();
    r.during.search = async () => { await cancel(r.web(), await r.runId()); };
    r.calls([['search', { query: 'tigers' }]]);
    expect(code(await r.turn('send'))).toBe('run_cancelled');
    delete r.during.search;
    r.events.length = 0;

    // The cancel was for THAT run: a later turn is not stopped by its flag.
    r.text('Carrying on.');
    expect(await r.turn('send', '')).toBeNull();
    r.text('And again.');
    expect(await r.turn('send', 'next')).toBeNull();
    expect((await r.web().run())?.status).toBe('completed');
    expect(r.names()).toEqual(['run.finished', 'run.finished']);
  });

  it('keeps the tool calls it guards identical in every other respect', async () => {
    const r = await Rig.make();
    r.calls([['search', { query: 'tigers' }], ['weather', { city: 'Detroit' }]]);
    r.text('Both done.');

    expect(await r.turn('send')).toBeNull();
    expect(r.ran).toEqual({ search: 1, weather: 1 });
    expect(await r.outputs()).toEqual({ call_g285_1: 'Found it', call_g285_2: 'Sunny' });
    expect((await r.web().run())?.status).toBe('completed');
    expect(r.names()).toEqual(['run.finished']);
  });
});
