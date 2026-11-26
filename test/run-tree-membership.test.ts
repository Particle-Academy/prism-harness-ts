import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  HarnessEvents,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  RunBudget,
  RunContext,
  RunLedger,
  SubagentRunner,
  Session,
  ToolRegistry,
  recordApproval,
  type AgentResponse,
  type Durability,
  type JsonObject,
  type LlmRequest,
  type LlmResponse,
  type ModeConfig,
  type ModelBinding,
  type Subagent,
} from '../src/index.js';
import { RunTrees } from '../src/run-trees.js';

// G-381: which run tree a run belongs to is decided ONCE, when the run begins,
// and every consumer (the tree a child saves, the budget it spends from, its
// depth) reads that one record. A host context is the host's whatever run ids
// it states; a child the runner starts belongs to the tree of the live run
// whose turn called the subagent tool, never to a tree found by a run id or a
// ledger the context carries.
//
// Driven through the public API (runtime.send, SubagentRunner) with a scripted
// model client; fetch is refused and every store is in memory.

beforeEach(() => {
  vi.stubGlobal('fetch', () => { throw new Error('A test tried to use the network.'); });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

class DurableMemoryStore extends MemorySessionStore {
  /** Every key written, in order. */
  readonly puts: string[] = [];

  override durability(): Durability {
    return 'durable';
  }

  override async put(key: string, payload: JsonObject, ttlSeconds: number | null = null): Promise<void> {
    this.puts.push(key);
    await super.put(key, payload, ttlSeconds);
  }
}

/** An ephemeral store a test can empty, as an expired TTL would. */
class ExpiringStore extends MemorySessionStore {
  readonly #keys = new Set<string>();

  override async put(key: string, payload: JsonObject, ttlSeconds: number | null = null): Promise<void> {
    this.#keys.add(key);
    await super.put(key, payload, ttlSeconds);
  }

  async expireAll(): Promise<void> {
    for (const key of this.#keys) await this.forget(key);
  }
}

class Rig {
  readonly ephemeral = new ExpiringStore();
  readonly durable = new DurableMemoryStore();
  weatherRan = 0;
  readonly queue: LlmResponse[] = [];
  readonly requests: LlmRequest[] = [];
  readonly runtime: AgentRuntime;
  readonly harness: PrismHarness;
  readonly research: Subagent;
  /** The depth of every run.started, in order. */
  readonly depths: number[] = [];
  /** Run once before the next model call is answered (a second worker, say). */
  beforeRequest: (() => Promise<void>) | null = null;
  /** What host code does inside the `hostcall` tool. */
  onHost: () => Promise<string> = async () => 'ok';
  #call = 0;

  constructor(chatSteps = 20, researchSteps = 8) {
    const modes = new ModeRegistry({
      default: 'chat',
      modes: {
        chat: {
          system_prompt: 'You delegate.', tools: ['research', 'scout', 'hostcall', 'weather'], requires_approval: ['weather'], max_steps: chatSteps,
          subagents: { research: { mode: 'research' }, scout: { mode: 'dig' } },
        } as ModeConfig,
        research: {
          system_prompt: 'Research.', tools: ['weather', 'dig', 'hostcall'], requires_approval: ['weather'], max_steps: researchSteps,
          subagents: { dig: { mode: 'dig' } },
        } as ModeConfig,
        dig: { system_prompt: 'Dig.', tools: [], max_steps: 4 } as ModeConfig,
        // A tool and a delegation that both need a person.
        gated: {
          system_prompt: 'Gated.', tools: ['weather', 'research'], requires_approval: ['weather', 'research'], max_steps: 4,
          subagents: { research: { mode: 'research' } },
        } as ModeConfig,
        // A delegation a person must approve first.
        guarded: {
          system_prompt: 'Guarded.', tools: ['research'], requires_approval: ['research'], max_steps: 3,
          subagents: { research: { mode: 'research' } },
        } as ModeConfig,
      },
    });
    const events = new HarnessEvents();
    events.listen((event) => {
      if (event.type === 'run.started') this.depths.push(event.depth);
    });
    const tools = new ToolRegistry()
      .register({ name: 'weather', handle: () => { this.weatherRan += 1; return 'Sunny'; } })
      .register({ name: 'hostcall', handle: () => this.onHost() });
    this.runtime = new AgentRuntime({
      // A second worker waiting on the session lock gives up quickly here.
      config: { lock_wait: 1 },
      modes,
      tools,
      events,
      client: async (request) => {
        this.requests.push(request);
        const hook = this.beforeRequest;
        this.beforeRequest = null;
        if (hook !== null) await hook();
        const next = this.queue.shift();
        if (next === undefined) throw new Error('The scripted model has no response left.');

        return next;
      },
    });
    const durable = this.durable, ephemeral = this.ephemeral;
    this.harness = new PrismHarness({
      drivers: { memory: () => ephemeral, durable: () => durable },
      stores: { ephemeral: 'memory', durable: 'durable' },
      runtime: this.runtime,
    });
    this.research = modes.resolve('chat').subagents.research!;
    const dig = modes.resolve('research').subagents.dig!;
    const scout = modes.resolve('chat').subagents.scout!;
    tools.registerFactory('scout', (session) => new SubagentRunner(this.harness, this.runtime)
      .tool(scout, session, RunContext.root('scout', new RunBudget(100)), 'scout_run'));
    // The host wires each subagent tool with a context of its own; called by
    // a turn, the child runs in that turn's tree.
    tools.registerFactory('research', (session) => new SubagentRunner(this.harness, this.runtime)
      .tool(this.research, session, RunContext.root('host', new RunBudget(100)), 'host_run'));
    tools.registerFactory('dig', (session) => new SubagentRunner(this.harness, this.runtime)
      .tool(dig, session, RunContext.root('dig', new RunBudget(100)), 'dig_run'));
  }

  session(user: number, scope = 'chat'): Session {
    return this.harness.for({ type: 'User', id: user }).session(scope);
  }

  calls(calls: [string, JsonObject][]): void {
    this.queue.push({ text: '', finishReason: 'tool_calls', toolCalls: calls.map(([name, args]) => ({ id: `call_${++this.#call}`, name, arguments: args })) });
  }

  text(text = 'Done.'): void {
    this.queue.push({ text, finishReason: 'stop' });
  }

  /** The approval ids `session` is waiting on. */
  async pending(session: Session): Promise<string[]> {
    const rows = (await session.thread().messages()).map((entry) => entry.message);
    const answered = new Set(rows.flatMap((row) => Array.isArray(row.tool_approval_responses) ? row.tool_approval_responses as JsonObject[] : [])
      .map((response) => response.approval_id));

    return rows.flatMap((row) => Array.isArray(row.tool_approval_requests) ? row.tool_approval_requests as JsonObject[] : [])
      .map((request) => request.approval_id as string).filter((id) => !answered.has(id));
  }

  /** How many model calls a mode's runs made. */
  made(systemPrompt: string): number {
    return this.requests.filter((request) => request.systemPrompt.startsWith(systemPrompt)).length;
  }

  /** A root turn on `user`'s chat that delegates to research once; the child answers in text. */
  async delegated(user: number, context?: RunContext): Promise<{ root: string; child: Session }> {
    this.calls([['research', { task: 'Find the weather.' }]]);
    this.text('Child answer.');
    this.text('Parent answer.');
    const root = (await this.runtime.send(this.session(user), 'Go', undefined, context)).runId;

    return { root, child: this.session(user, 'chat::sub::research') };
  }
}

const binding: ModelBinding = {
  session: { provider: null, model: null },
  declared: { provider: null, model: null },
  parentMode: { provider: null, model: null },
};

/** The durable tree record of `session`'s run `run`, or null when none was opened. */
async function treeRecord(session: Session, run: string): Promise<JsonObject | null> {
  return session.planStore().get(`${session.key()}:run-tree:${run}`);
}

describe('a host context never joins another run tree', () => {
  it('keeps a grandchild of a host context that names a live run out of that run tree', async () => {
    const r = new Rig();
    const ada = r.session(1), other = r.session(2);
    let live = '';
    r.onHost = async () => {
      // Host code, inside Ada's live turn, names Ada's run as the root and the
      // parent of a context it builds, and sends with it to another session.
      live = (await ada.run())!.id;
      await r.runtime.send(other, 'Delegate', undefined, new RunContext(RunLedger.start(live), new RunBudget(50), live, 0));

      return 'ok';
    };
    r.calls([['hostcall', {}]]);                         // Ada
    r.calls([['research', { task: 'Find the weather.' }]]); // the other session, under the host context
    r.text('Grandchild answer.');                        // its research child
    r.text('Other answer.');                             // the other session
    r.text('Ada answer.');                               // Ada
    await r.runtime.send(ada, 'Go');

    expect(r.queue).toHaveLength(0);
    const grandchild = r.session(2, 'chat::sub::research');
    expect((await grandchild.childRefs())?.tree).toBeNull();
    expect(await treeRecord(ada, live)).toBeNull();
  });

  it.each([['the live run\'s own session', 1], ['another session', 2]] as const)(
    'keeps a child a direct SubagentRunner.run() call starts with a forged parent run id out of the live tree, under %s',
    async (_label, parentUser) => {
      const r = new Rig();
      const ada = r.session(1);
      let live = '';
      r.onHost = async () => {
        live = (await ada.run())!.id;
        const result = await new SubagentRunner(r.harness, r.runtime)
          .run(r.research, r.session(parentUser), RunContext.root(live, new RunBudget(50)), live, 'Forged.');

        return result.text;
      };
      r.calls([['hostcall', {}]]);
      r.text('Forged child answer.');
      r.text('Ada answer.');
      await r.runtime.send(ada, 'Go');

      expect(r.queue).toHaveLength(0);
      const child = r.session(parentUser, 'chat::sub::research');
      expect(await child.childBinding()).not.toBeNull();
      expect((await child.childRefs())?.tree).toBeNull();
      expect(await treeRecord(ada, live)).toBeNull();
    },
  );

  it('gives each grandchild of one host context shared across two sessions its own parent\'s tree', async () => {
    const r = new Rig();
    const { child: adaChild } = await r.delegated(1);
    const { child: bobChild } = await r.delegated(2);
    const adaTree = (await adaChild.childRefs())!.tree, bobTree = (await bobChild.childRefs())!.tree;
    expect(adaTree).not.toBeNull();
    expect(bobTree).not.toBeNull();
    expect(bobTree).not.toEqual(adaTree);

    // One host context, handed to both children while Ada's child is live.
    const shared = RunContext.root('host_shared', new RunBudget(100));
    r.onHost = async () => {
      await r.runtime.send(bobChild, 'Dig', undefined, shared);

      return 'ok';
    };
    r.calls([['hostcall', {}]]);              // Ada's child
    r.calls([['dig', { task: 'Dig.' }]]);     // Bob's child
    r.text('Bob grandchild answer.');         // Bob's grandchild
    r.text('Bob child answer.');              // Bob's child
    r.text('Ada child answer.');              // Ada's child
    await r.runtime.send(adaChild, 'Go on', undefined, shared);

    expect(r.queue).toHaveLength(0);
    const bobGrandchild = r.session(2, 'chat::sub::research::sub::dig');
    expect((await bobGrandchild.childRefs())?.tree).toEqual(bobTree);
  });
});

describe('a child the runner starts belongs to its parent run\'s tree', () => {
  it('saves the root\'s tree on a child of a child, at depth 2', async () => {
    const r = new Rig();
    r.calls([['research', { task: 'Find the weather.' }]]); // root
    r.calls([['dig', { task: 'Dig.' }]]);                   // child
    r.text('Grandchild answer.');
    r.text('Child answer.');
    r.text('Root answer.');
    const root = (await r.runtime.send(r.session(1), 'Go')).runId;

    const tree = { session: r.session(1).key(), run: root };
    const grandchild = r.session(1, 'chat::sub::research::sub::dig');
    expect((await r.session(1, 'chat::sub::research').childRefs())?.tree).toEqual(tree);
    expect((await grandchild.childRefs())?.tree).toEqual(tree);
    expect(await grandchild.childDepth()).toBe(2);
    expect(r.depths).toEqual([0, 1, 2]);
  });

  it('starts a child again from a new tree at a shallower depth in the new tree, at the runner\'s depth', async () => {
    const r = new Rig();
    const child = r.session(1, 'chat::sub::research');
    await child.rememberChildBinding(binding, 3);
    await child.rememberChildRefs({ tree: { session: 'old-root', run: 'old-run' }, planLedger: null });

    const { root } = await r.delegated(1);

    expect((await child.childRefs())?.tree).toEqual({ session: r.session(1).key(), run: root });
    expect(await child.childDepth()).toBe(1);
    expect(r.depths).toEqual([0, 1]);
  });

  it('keeps the refs a child saved when it is started again under a host-built tree', async () => {
    const r = new Rig();
    const child = r.session(1, 'chat::sub::research');
    await child.rememberChildBinding(binding, 3);
    const saved = { tree: { session: 'old-root', run: 'old-run' }, planLedger: null };
    await child.rememberChildRefs(saved);

    await r.delegated(1, RunContext.root('host_root', new RunBudget(100)));

    expect(await child.childRefs()).toEqual(saved);
    expect(await child.childDepth()).toBe(1);
    expect(r.depths).toEqual([0, 1]);
  });
});

describe('the depth a turn the runner did not start runs at', () => {
  async function savedAt(r: Rig, depth: number): Promise<Session> {
    const child = r.session(1, 'chat::sub::research');
    await child.usingMode('research');
    await child.rememberChildBinding(binding, depth);

    return child;
  }

  it('runs a host context shallower than the saved depth at the saved depth, so delegation past the ceiling is refused', async () => {
    const r = new Rig();
    const child = await savedAt(r, 3);

    r.calls([['dig', { task: 'Dig.' }]]);
    r.text('Child answer.');
    await r.runtime.send(child, 'Dig', undefined, RunContext.root('host_root', new RunBudget(100)));

    expect(r.depths).toEqual([3]);
    expect(r.made('Dig.')).toBe(0);
    const results = (await child.thread().messages()).map((entry) => JSON.stringify(entry.message)).filter((row) => row.includes('maximum depth'));
    expect(results).toHaveLength(1);
  });

  it('runs a host context deeper than the saved depth at the context\'s depth', async () => {
    const r = new Rig();
    const child = await savedAt(r, 1);

    r.text('Child answer.');
    await r.runtime.send(child, 'Again', undefined, new RunContext(RunLedger.start('host_deep'), new RunBudget(100), 'host_parent', 2));

    expect(r.depths).toEqual([2]);
  });
});

describe('a root that resumes its parked run', () => {
  it('stays in the tree its parked run began in, so a new subagent it starts after the resume saves that tree', async () => {
    const r = new Rig();
    const root = r.session(1);
    r.calls([['research', { task: 'Find the weather.' }]]); // root
    r.calls([['weather', { city: 'Paris' }]]);              // research child: parks
    r.calls([['weather', { city: 'Rome' }]]);               // root: parks
    const first = await r.runtime.send(root, 'Go');
    const tree = { session: root.key(), run: first.runId };
    expect((await r.session(1, 'chat::sub::research').childRefs())?.tree).toEqual(tree);
    const [id] = await r.pending(root);
    expect(id).toBeDefined();

    r.calls([['scout', { task: 'Scout.' }]]);                // root, resumed
    r.text('Scout answer.');                                // scout child
    r.text('Root answer.');                                 // root
    await recordApproval(root, id!, true);
    const resumed = await r.runtime.send(root, '');

    expect(resumed.runId).not.toBe(first.runId);
    expect(r.queue).toHaveLength(0);
    expect((await r.session(1, 'chat::sub::scout').childRefs())?.tree).toEqual(tree);
  });
});

describe('a root that resumes its parked run after the ephemeral half expired', () => {
  it('still stays in the tree its parked run began in', async () => {
    const r = new Rig();
    const root = r.session(1);
    r.calls([['research', { task: 'Find the weather.' }]]); // root
    r.calls([['weather', { city: 'Paris' }]]);              // research child: parks
    r.calls([['weather', { city: 'Rome' }]]);               // root: parks
    const first = await r.runtime.send(root, 'Go');
    const [id] = await r.pending(root);
    await r.ephemeral.expireAll();

    r.calls([['scout', { task: 'Scout.' }]]);
    r.text('Scout answer.');
    r.text('Root answer.');
    await recordApproval(r.session(1), id!, true);
    await r.runtime.send(r.session(1), '');

    expect(r.queue).toHaveLength(0);
    expect((await r.session(1, 'chat::sub::scout').childRefs())?.tree).toEqual({ session: root.key(), run: first.runId });
  });
});

/**
 * The root delegates a child (it parks), then parks on its own approval; the
 * root is approved, resumes under a new run id, delegates a second subagent
 * (it completes) and parks again.
 */
async function resumedRootDelegatesAgain(chatSteps: number): Promise<{ r: Rig; child: Session; pending: string; firstRun: string }> {
  const r = new Rig(chatSteps);
  const root = r.session(1);
  r.calls([['research', { task: 'Find the weather.' }]]); // root
  r.calls([['weather', { city: 'Paris' }]]);              // research child: parks
  r.calls([['weather', { city: 'Rome' }]]);               // root: parks
  const firstRun = (await r.runtime.send(root, 'Go')).runId;
  const [first] = await r.pending(root);

  r.calls([['scout', { task: 'Scout.' }]]);               // root, resumed
  r.text('Scouted.');                                     // scout child
  r.calls([['weather', { city: 'Oslo' }]]);               // root: parks again
  await recordApproval(root, first!, true);
  await r.runtime.send(root, '');
  expect(await r.pending(root)).toHaveLength(1);

  const child = r.session(1, 'chat::sub::research');
  const [pending] = await r.pending(child);
  expect(pending).toBeDefined();

  return { r, child, pending: pending!, firstRun };
}

describe('a root that resumes its parked run and delegates again', () => {
  it('leaves a parked subagent what its tree has left after the root resumed, delegated and parked again', async () => {
    const { r, child, pending } = await resumedRootDelegatesAgain(20);
    const ran = r.weatherRan;

    r.text('Child done.');
    await recordApproval(child, pending, true);
    await r.runtime.send(child, '');

    expect(r.weatherRan).toBe(ran + 1);
    expect(await r.pending(child)).toEqual([]);
    expect((await child.run())?.status).toBe('completed');
  });

  it('holds a resumed root to its tree record, so the tree never spends past its allowance', async () => {
    const { r, child, pending, firstRun } = await resumedRootDelegatesAgain(6);
    const record = await treeRecord(r.session(1), firstRun);
    expect(record).not.toBeNull();
    expect(record!.steps).toBe(record!.max_steps);
    const ran = r.weatherRan, requests = r.requests.length;

    // Spent, the tree leaves the parked child its approved call and no model
    // step (G-387): the decision is accepted, the call runs, nothing more.
    await recordApproval(child, pending, true);
    const resumed = await r.runtime.send(child, '');

    expect(resumed.finishReason).toBe('budget_exhausted');
    expect(await r.pending(child)).toEqual([]);
    expect(r.weatherRan).toBe(ran + 1);
    expect(r.requests.length).toBe(requests);
    const after = (await treeRecord(r.session(1), firstRun))!;
    expect(after.steps).toBe(after.max_steps);
  });

  // A root resumed into a tree its sibling spent while the root waited still
  // has steps of its own, but the tree has none (G-387): its decision is
  // accepted and acted on, it takes no model step, and the tree is never
  // spent past its allowance. Approving it again is refused: the decision
  // was acted on, and nothing is pending.
  it('acts on the decision of a root whose sibling spent its tree, with no model step', async () => {
    const r = new Rig(6);
    const root = r.session(1);
    r.calls([['research', { task: 'Find the weather.' }]]); // root
    r.calls([['weather', { city: 'Paris' }]]);              // research child: parks
    r.calls([['weather', { city: 'Rome' }]]);               // root: parks, 3 of 6
    const first = await r.runtime.send(root, 'Go');
    const child = r.session(1, 'chat::sub::research');
    const [childId] = await r.pending(child);
    r.calls([['hostcall', {}]]);                            // the child spends the rest of the tree
    r.calls([['hostcall', {}]]);
    r.text('Child done.');
    await recordApproval(child, childId!, true);
    await r.runtime.send(child, '');
    const record = (await treeRecord(root, first.runId))!;
    expect(record.steps).toBe(record.max_steps);
    const [rootId] = await r.pending(root);
    const ran = r.weatherRan, requests = r.requests.length;

    await recordApproval(root, rootId!, true);
    const resumed = await r.runtime.send(r.session(1), '');

    expect(resumed.finishReason).toBe('budget_exhausted');
    expect(r.weatherRan).toBe(ran + 1);
    expect(r.requests.length).toBe(requests);
    expect(await r.pending(root)).toEqual([]);
    await expect(recordApproval(root, rootId!, true)).rejects.toMatchObject({ code: 'approval_not_pending' });
    const after = (await treeRecord(root, first.runId))!;
    expect(after.steps).toBe(after.max_steps);
  });
});

describe('where a run keeps its tree', () => {
  it('never rewrites the session durable record when a run starts', async () => {
    const r = new Rig();
    const session = r.session(1, 'plain');
    r.text('Done.');
    await r.runtime.send(session, 'go');

    expect(r.durable.puts.filter((key) => key === `${session.key()}:durable`)).toEqual([]);
    expect(r.durable.puts.filter((key) => key === `${session.key()}:run-trees`)).not.toEqual([]);
  });
});

describe('a root that resumes a parked run that started no child', () => {
  it('opens the parked run\'s tree for the child it starts, so that child can be resumed in it', async () => {
    const r = new Rig();
    const root = r.session(1);
    r.calls([['weather', { city: 'Rome' }]]);               // root: parks, no child yet
    const first = await r.runtime.send(root, 'Go');
    const [id] = await r.pending(root);

    r.calls([['research', { task: 'Find the weather.' }]]); // root, resumed
    r.calls([['weather', { city: 'Paris' }]]);              // research child: parks
    r.text('Root answer.');
    await recordApproval(root, id!, true);
    await r.runtime.send(root, '');
    const child = r.session(1, 'chat::sub::research');
    expect((await child.childRefs())?.tree).toEqual({ session: root.key(), run: first.runId });
    const [childId] = await r.pending(child);

    r.text('Child answer.');
    await recordApproval(child, childId!, true);
    await r.runtime.send(child, '');

    expect(r.queue).toHaveLength(0);
    expect(await r.pending(child)).toEqual([]);
  });
});

describe('a top-level session that ran on a binding the host supplied', () => {
  const hosted = (): RunContext => new RunContext(RunLedger.start('host_root'), new RunBudget(100), null, 0, {}, {
    session: { provider: null, model: null },
    declared: { provider: 'custom-cheap', model: 'cheap-model' },
    parentMode: { provider: null, model: null },
  });

  it('keeps answering its approvals after a resume that opened a tree of its own parks again', async () => {
    const r = new Rig();
    const plain = r.session(1, 'plain');
    r.calls([['weather', { city: 'Paris' }]]);
    await r.runtime.send(plain, 'Go', undefined, hosted());
    const [first] = await r.pending(plain);

    // The resume reuses the row's binding and opens a tree of its own; it parks again.
    r.calls([['weather', { city: 'Rome' }]]);
    await recordApproval(plain, first!, true);
    await r.runtime.send(plain, '');
    const [second] = await r.pending(plain);
    expect(second).toBeDefined();

    r.text('Done.');
    await recordApproval(plain, second!, true);
    await r.runtime.send(plain, '');
    r.text('Again.');
    await r.runtime.send(plain, 'New question');

    expect(r.queue).toHaveLength(0);
    expect(await r.pending(plain)).toEqual([]);
  });
});

// An approved subagent call runs when the session resumes, before the model is
// called again. It runs inside the turn: in the turn's tree, at the turn's
// depth and on the turn's budget, never on the context the host wired the
// subagent tool with.
describe('an approved subagent call, run on the resume', () => {
  const drain = async (steps: AsyncGenerator<unknown, unknown, undefined>) => { for await (const _ of steps) { /* drained */ } };
  const resumes = [['send', 'send'], ['streamDecision', 'streamDecision']] as const;

  async function approve(r: Rig, session: Session, how: 'send' | 'streamDecision'): Promise<void> {
    const [id] = await r.pending(session);
    expect(id).toBeDefined();
    if (how === 'send') {
      await recordApproval(session, id!, true);
      await r.runtime.send(session, '').catch(() => undefined);
    } else {
      await drain(r.runtime.streamDecision(session, [{ approvalId: id!, approved: true }])).catch(() => undefined);
    }
  }

  it.each(resumes)('runs at the turn\'s depth, so it is refused past the ceiling (%s)', async (_label, how) => {
    const r = new Rig();
    const child = r.session(1, 'chat::sub::guarded');
    await child.usingMode('guarded');
    await child.rememberChildBinding(binding, 3);
    r.calls([['research', { task: 'Find the weather.' }]]);   // parks on its approval
    await r.runtime.send(child, 'Go');

    r.text('Child answer.');
    await approve(r, child, how);

    expect(r.made('Research.')).toBe(0);
    const refusals = (await child.thread().messages()).map((entry) => JSON.stringify(entry.message)).filter((row) => row.includes('maximum depth'));
    expect(refusals).toHaveLength(1);
  });

  it.each(resumes)('spends from the turn\'s budget, never the host\'s (%s)', async (_label, how) => {
    const r = new Rig();
    const session = r.session(1, 'guarded');
    await session.usingMode('guarded');
    r.calls([['research', { task: 'Find the weather.' }]]);   // step 1 of 3: parks on its approval
    await r.runtime.send(session, 'Go');
    const before = r.requests.length;

    // The child would take four steps; the turn has two left.
    for (let i = 0; i < 3; i++) r.calls([['hostcall', {}]]);
    r.text('Child answer.');
    r.text('Answer.');
    await approve(r, session, how);

    expect(r.requests.length - before).toBe(2);
  });

  it('runs in the turn\'s tree, so the child saves it', async () => {
    const r = new Rig();
    const session = r.session(1, 'guarded');
    await session.usingMode('guarded');
    r.calls([['research', { task: 'Find the weather.' }]]);
    const first = await r.runtime.send(session, 'Go');

    r.text('Child answer.');
    r.text('Answer.');
    await approve(r, session, 'send');

    expect((await r.session(1, 'guarded::sub::research').childRefs())?.tree).toEqual({ session: session.key(), run: first.runId });
  });
});

// W-1: streamDecision resumes a parked root under the parked run's OWN id.
// Its tree's record holds the steps the tree spent while it was parked, and
// the resumed root is held to it there exactly as on a resume under a new id.
describe('a root resumed on the stream path under its own run id', () => {
  const drain = async (steps: AsyncGenerator<unknown, unknown, undefined>) => { for await (const _ of steps) { /* drained */ } };

  it('is charged to its tree record', async () => {
    const r = new Rig();
    const root = r.session(1);
    r.calls([['research', { task: 'Find the weather.' }]]); // root
    r.calls([['weather', { city: 'Paris' }]]);              // research child: parks
    r.calls([['weather', { city: 'Rome' }]]);               // root: parks
    const first = (await r.runtime.send(root, 'Go')).runId;
    const [rootId] = await r.pending(root);

    // The child is answered first, and spends from the tree.
    const child = r.session(1, 'chat::sub::research');
    const [childId] = await r.pending(child);
    r.text('Child done.');
    await recordApproval(child, childId!, true);
    await r.runtime.send(child, '');
    const before = (await treeRecord(root, first))!.steps as number;

    r.text('Root done.');                                   // one step
    await drain(r.runtime.streamDecision(r.session(1), [{ approvalId: rootId!, approved: true }]));

    const record = (await treeRecord(root, first))!;
    expect(r.queue).toHaveLength(0);
    expect(record.steps).toBe(before + 1);
    expect(record.steps as number).toBeLessThanOrEqual(record.max_steps as number);
  });
});

// Two resumes of one parked root that never delegated: the first opens the
// tree's record when it starts, so the second is held to it, and the tree
// never spends past its allowance.
describe('two concurrent resumes of a parked root that never delegated', () => {
  it.each([['no context', undefined], ['a null context', null]] as const)('never spend the tree past its allowance (%s)', async (_label, given) => {
    const context = given as unknown as RunContext | undefined;
    const r = new Rig(6);
    const root = r.session(1);
    r.calls([['weather', { city: 'Rome' }]]);               // root: parks, no child
    const first = (await r.runtime.send(root, 'Go')).runId;
    const [id] = await r.pending(root);
    await recordApproval(root, id!, true);

    // A second worker resumes the same session while the first is in its model call.
    r.beforeRequest = async () => {
      await r.runtime.send(r.session(1), '', undefined, context).catch(() => undefined);
    };
    r.calls([['research', { task: 'A.' }]]);                // the second resume
    r.text('Child A.');
    r.text('Second done.');
    r.calls([['research', { task: 'B.' }]]);                // the first resume
    r.text('Child B.');
    for (let i = 0; i < 5; i++) r.calls([['hostcall', {}]]);
    r.text('First done.');
    await r.runtime.send(root, '', undefined, context).catch(() => undefined);

    const record = await treeRecord(root, first);
    expect(record).not.toBeNull();
    expect(record!.steps as number).toBeLessThanOrEqual(record!.max_steps as number);
  });
});

// W-2: send() takes no session lock, so two overlapping turns on one session
// begin together; neither may lose the other's run tree.
describe('overlapping turns on one session', () => {
  it('keep both runs in the session\'s run trees', async () => {
    const r = new Rig();
    const session = r.session(1, 'plain');
    r.text('One.');
    r.text('Two.');
    const [one, two] = await Promise.all([r.runtime.send(session, 'one'), r.runtime.send(r.session(1, 'plain'), 'two')]);

    const stored = await r.durable.get(`${session.key()}:run-trees`);
    expect(Object.keys((stored?.runs ?? {}) as JsonObject).sort()).toEqual([one.runId, two.runId].sort());
  });
});

// Py round-3 F2: the runner saves a child's refs into the child session's
// durable record, which also holds the host's capabilities. A capability the
// host revokes while the parent delegates must stay revoked.
describe('a host revoke on a child session while the parent delegates', () => {
  it('is never undone by the runner saving the child\'s refs', async () => {
    const r = new Rig();
    await r.delegated(1);
    const child = r.session(1, 'chat::sub::research');
    await child.usingCapability('grant', { on: true });

    // Inside the refs save, right after its read, another worker revokes.
    const original = Session.prototype.rememberChildRefs;
    const get = r.durable.get.bind(r.durable);
    let armed = false, revoked: Promise<unknown> | null = null;
    const spyRefs = vi.spyOn(Session.prototype, 'rememberChildRefs').mockImplementation(async function (this: Session, refs) {
      armed = true;
      return original.call(this, refs);
    });
    const spyGet = vi.spyOn(r.durable, 'get').mockImplementation(async (key: string) => {
      const value = await get(key);
      if (armed && key === `${child.key()}:durable`) {
        armed = false;
        revoked = r.session(1, 'chat::sub::research').forgetCapability('grant');
        await new Promise((resolve) => setTimeout(resolve, 20));
      }

      return value;
    });
    try {
      await r.delegated(1);
      await revoked;
    } finally {
      spyRefs.mockRestore();
      spyGet.mockRestore();
    }

    expect(revoked).not.toBeNull();
    expect(await r.session(1, 'chat::sub::research').capability('grant')).toBeNull();
  });
});

// An accepted decision is always acted on. A resumed root whose own step
// budget is spent (by the measure its parked row records, a runner child's
// steps included) acts on its recorded decisions with no further model step
// and ends on its step budget: that costs no steps, so the decision is
// accepted even when its tree's record is spent too.
describe('a decision on a root whose own step budget is spent', () => {
  const drain = async (steps: AsyncGenerator<unknown, unknown, undefined>) => { for await (const _ of steps) { /* drained */ } };
  const paths = [['recordApproval', 'recordApproval'], ['decide', 'decide'], ['streamDecision', 'streamDecision']] as const;

  async function act(r: Rig, id: string, how: 'recordApproval' | 'decide' | 'streamDecision'): Promise<void> {
    const root = r.session(1);
    if (how === 'recordApproval') {
      await recordApproval(root, id, true);
      await r.runtime.send(root, '');
    } else if (how === 'decide') {
      await r.runtime.decide(root, id, true);
      await r.runtime.send(root, '');
    } else {
      await drain(r.runtime.streamDecision(root, [{ approvalId: id, approved: true }]));
    }
  }

  it.each(paths)('is acted on with no model step after the root delegated and parked on its last step (%s)', async (_label, how) => {
    const r = new Rig(3);
    const root = r.session(1);
    r.calls([['research', { task: 'Find the weather.' }]]); // root: step 1
    r.text('Child answer.');                                // child: step 2
    r.calls([['weather', { city: 'Rome' }]]);               // root: step 3, parks
    const first = (await r.runtime.send(root, 'Go')).runId;
    const record = (await treeRecord(root, first))!;
    expect(record.steps).toBe(record.max_steps);
    const [id] = await r.pending(root);
    const ran = r.weatherRan, requests = r.requests.length;

    await act(r, id!, how);

    expect(r.weatherRan).toBe(ran + 1);
    expect(r.requests.length).toBe(requests);
    expect(await r.pending(root)).toEqual([]);
    const after = (await treeRecord(root, first))!;
    expect(after.steps as number).toBeLessThanOrEqual(after.max_steps as number);
  });

  it.each(paths)('is acted on after a resumed root delegated and parked again on its last step (%s)', async (_label, how) => {
    const r = new Rig(4);
    const root = r.session(1);
    r.calls([['weather', { city: 'Rome' }]]);               // root: step 1, parks
    const first = (await r.runtime.send(root, 'Go')).runId;
    const [firstId] = await r.pending(root);
    r.calls([['research', { task: 'Find the weather.' }]]); // resumed root: step 2
    r.text('Child answer.');                                // child: step 3
    r.calls([['weather', { city: 'Oslo' }]]);               // resumed root: step 4, parks again
    await recordApproval(root, firstId!, true);
    await r.runtime.send(root, '');
    const record = (await treeRecord(root, first))!;
    expect(record.steps).toBe(record.max_steps);
    const [id] = await r.pending(root);
    expect(id).toBeDefined();
    const ran = r.weatherRan, requests = r.requests.length;

    await act(r, id!, how);

    expect(r.weatherRan).toBe(ran + 1);
    expect(r.requests.length).toBe(requests);
    expect(await r.pending(root)).toEqual([]);
    const after = (await treeRecord(root, first))!;
    expect(after.steps as number).toBeLessThanOrEqual(after.max_steps as number);
  });
});

// Review round 4, F-1: a root that delegated BEFORE its first pause. At its
// resume the ledger is rebuilt from the root's own rows, so the child's steps
// from before that pause are not on it; what the row it parks on again says
// is left is capped by the tree ceiling the resume holds, so the rule holds at
// every pause: spent, it acts on its decision with no model step.
describe('a root spent at its second pause, having delegated before its first', () => {
  const drain = async (steps: AsyncGenerator<unknown, unknown, undefined>) => { for await (const _ of steps) { /* drained */ } };

  it.each([['recordApproval'], ['decide'], ['streamDecision']] as const)('acts on its decision with no model step (%s)', async (how) => {
    const r = new Rig(4);
    const root = r.session(1);
    r.calls([['research', { task: 'Find the weather.' }]]); // root: step 1
    r.text('Child answer.');                                // child: step 2
    r.calls([['weather', { city: 'Paris' }]]);              // root: step 3, parks
    const first = (await r.runtime.send(root, 'Go')).runId;
    const [firstId] = await r.pending(root);
    r.calls([['weather', { city: 'Rome' }]]);               // resumed: step 4, parks again
    await recordApproval(root, firstId!, true);
    await r.runtime.send(root, '');
    const [id] = await r.pending(root);
    expect(id).toBeDefined();
    const record = (await treeRecord(root, first))!;
    expect(record.steps).toBe(record.max_steps);
    const ran = r.weatherRan, requests = r.requests.length;

    if (how === 'recordApproval') {
      await recordApproval(root, id!, true);
      await r.runtime.send(root, '');
    } else if (how === 'decide') {
      await r.runtime.decide(root, id!, true);
      await r.runtime.send(root, '');
    } else {
      await drain(r.runtime.streamDecision(root, [{ approvalId: id!, approved: true }]));
    }

    expect(r.weatherRan).toBe(ran + 1);
    expect(r.requests.length).toBe(requests);
    expect(await r.pending(root)).toEqual([]);
    const after = (await treeRecord(root, first))!;
    expect(after.steps as number).toBeLessThanOrEqual(after.max_steps as number);
  });
});

// Review round 4, P2: a spent root acts on the decisions in its parked batch,
// but takes nothing of its spent tree: an approved delegation in that batch is
// refused, and its child never calls the model.
describe('a spent root parked on a tool and a delegation', () => {
  it('runs the tool and refuses the delegation, with no child request', async () => {
    const r = new Rig();
    const root = r.session(1, 'gated');
    await root.usingMode('gated');
    r.calls([['research', { task: 'First.' }]]);            // step 1: parks on research
    const first = (await r.runtime.send(root, 'Go')).runId;
    let [id] = await r.pending(root);
    r.text('Child answer.');                                // child: step 2
    r.calls([['weather', { city: 'Paris' }]]);              // step 3: parks on weather
    await recordApproval(root, id!, true);
    await r.runtime.send(root, '');
    [id] = await r.pending(root);
    r.calls([['weather', { city: 'Rome' }], ['research', { task: 'Second.' }]]); // step 4: parks on both
    await recordApproval(root, id!, true);
    await r.runtime.send(root, '');
    const record = (await treeRecord(root, first))!;
    expect(record.steps).toBe(record.max_steps);
    const pending = await r.pending(root);
    expect(pending).toHaveLength(2);
    const ran = r.weatherRan, requests = r.requests.length, research = r.made('Research.');
    r.text('Should not run.');

    for (const approvalId of pending) await recordApproval(root, approvalId, true);
    await r.runtime.send(root, '');

    expect(r.weatherRan).toBe(ran + 1);
    expect(r.made('Research.')).toBe(research);
    expect(r.requests.length).toBe(requests);
    expect(await r.pending(root)).toEqual([]);
    const after = (await treeRecord(root, first))!;
    expect(after.steps as number).toBeLessThanOrEqual(after.max_steps as number);
  });
});

/** What the row `session` is parked on says its root has left of its own step budget. */
async function stepsLeftOnRow(session: Session): Promise<unknown> {
  const rows = (await session.thread().messages()).map((entry) => entry.message);

  return [...rows].reverse().find((row) => row.type === 'assistant' && Array.isArray(row.tool_approval_requests))?.harness_steps_left;
}

// G-387: the row a root parks on records the steps it used, a runner child's
// included. A sibling holding the rest of the tree while the root resumes
// leaves the root a tree grant of 0: it acts on its decision, takes no model
// step and finishes budget_exhausted, and the tree is never spent past its
// allowance.
describe('a root resumed while a sibling holds the rest of its tree', () => {
  it('acts on its decision with no model step', async () => {
    const r = new Rig(6, 3);
    const root = r.session(1);
    r.calls([['research', { task: 'Find the weather.' }]]); // root: step 1
    r.calls([['weather', { city: 'Paris' }]]);              // research child: step 2, parks
    r.calls([['weather', { city: 'Rome' }]]);               // root: step 3, parks
    const first = (await r.runtime.send(root, 'Go')).runId;
    expect(await stepsUsedOnRow(root)).toBe(3);
    expect(await stepsLeftOnRow(root)).toBe(3);
    const child = r.session(1, 'chat::sub::research');
    const [childId] = await r.pending(child);
    const [rootId] = await r.pending(root);
    await recordApproval(child, childId!, true);
    await recordApproval(root, rootId!, true);

    // The child resumes and holds the rest of the tree; inside its model call
    // the root resumes.
    let resumed: AgentResponse | null = null;
    const ran = r.weatherRan;
    r.beforeRequest = async () => {
      resumed = await r.runtime.send(r.session(1), '');
    };
    r.text('Child done.');                                  // the child
    await r.runtime.send(child, '');

    expect(resumed!.finishReason).toBe('budget_exhausted');
    expect(r.weatherRan).toBe(ran + 2);
    expect(r.queue).toHaveLength(0);
    expect(await r.pending(root)).toEqual([]);
    const record = (await treeRecord(root, first))!;
    expect(record.steps as number).toBeLessThanOrEqual(record.max_steps as number);
  });
});

/** The steps the row `session` is parked on says its run used. */
async function stepsUsedOnRow(session: Session): Promise<unknown> {
  const rows = (await session.thread().messages()).map((entry) => entry.message);

  return [...rows].reverse().find((row) => row.type === 'assistant' && Array.isArray(row.tool_approval_requests))?.harness_steps_used;
}

// A row written without harness_steps_used (by a harness before G-387) is
// counted from the root's own rows. On a spent tree its decision is accepted
// and acted on, with no model step.
describe('a legacy parked row with no step count', () => {
  it('is acted on with no model step when its tree is spent', async () => {
    const r = new Rig(3);
    const root = r.session(1);
    const legacy = vi.spyOn(RunTrees, 'allowanceOf').mockReturnValue(null);
    try {
      r.calls([['research', { task: 'Find the weather.' }]]); // root: step 1
      r.text('Child answer.');                                // child: step 2
      r.calls([['weather', { city: 'Rome' }]]);               // root: step 3, parks
      const first = (await r.runtime.send(root, 'Go')).runId;
      const record = (await treeRecord(root, first))!;
      expect(record.steps).toBe(record.max_steps);
    } finally {
      legacy.mockRestore();
    }
    expect(await stepsUsedOnRow(root)).toBeUndefined();
    const [id] = await r.pending(root);
    const requests = r.requests.length, ran = r.weatherRan;

    await recordApproval(root, id!, true);
    const resumed = await r.runtime.send(root, '');

    expect(resumed.finishReason).toBe('budget_exhausted');
    expect(r.weatherRan).toBe(ran + 1);
    expect(r.requests.length).toBe(requests);
  });
});
