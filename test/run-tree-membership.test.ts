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
  ToolRegistry,
  recordApproval,
  type Durability,
  type JsonObject,
  type LlmRequest,
  type LlmResponse,
  type ModeConfig,
  type ModelBinding,
  type Session,
  type Subagent,
} from '../src/index.js';

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
  override durability(): Durability {
    return 'durable';
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
  readonly queue: LlmResponse[] = [];
  readonly requests: LlmRequest[] = [];
  readonly runtime: AgentRuntime;
  readonly harness: PrismHarness;
  readonly research: Subagent;
  /** The depth of every run.started, in order. */
  readonly depths: number[] = [];
  /** What host code does inside the `hostcall` tool. */
  onHost: () => Promise<string> = async () => 'ok';
  #call = 0;

  constructor(chatSteps = 20) {
    const modes = new ModeRegistry({
      default: 'chat',
      modes: {
        chat: {
          system_prompt: 'You delegate.', tools: ['research', 'scout', 'hostcall', 'weather'], requires_approval: ['weather'], max_steps: chatSteps,
          subagents: { research: { mode: 'research' }, scout: { mode: 'dig' } },
        } as ModeConfig,
        research: {
          system_prompt: 'Research.', tools: ['weather', 'dig', 'hostcall'], requires_approval: ['weather'], max_steps: 8,
          subagents: { dig: { mode: 'dig' } },
        } as ModeConfig,
        dig: { system_prompt: 'Dig.', tools: [], max_steps: 4 } as ModeConfig,
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
      .register({ name: 'weather', handle: () => 'Sunny' })
      .register({ name: 'hostcall', handle: () => this.onHost() });
    this.runtime = new AgentRuntime({
      modes,
      tools,
      events,
      client: async (request) => {
        this.requests.push(request);
        const next = this.queue.shift();
        if (next === undefined) throw new Error('The scripted model has no response left.');

        return next;
      },
    });
    const durable = new DurableMemoryStore(), ephemeral = this.ephemeral;
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

describe('a root that resumes its parked run and delegates again', () => {
  it('leaves a parked subagent its tree budget after the root resumed, delegated and parked again', async () => {
    const r = new Rig(6);
    const root = r.session(1);
    r.calls([['research', { task: 'Find the weather.' }]]); // root
    r.calls([['weather', { city: 'Paris' }]]);              // research child: parks
    r.calls([['weather', { city: 'Rome' }]]);               // root: parks
    await r.runtime.send(root, 'Go');
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
    r.text('Child done.');
    const answered = await recordApproval(child, pending!, true).then(() => true, () => false);
    if (answered) await r.runtime.send(child, '').catch(() => undefined);

    expect(answered).toBe(true);
    expect(await r.pending(child)).toEqual([]);
    expect((await child.run())?.status).toBe('completed');
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
