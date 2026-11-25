import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  RunBudget,
  RunContext,
  RunLedger,
  SubagentRunner,
  ToolRegistry,
  type Durability,
  type JsonObject,
  type LlmRequest,
  type LlmResponse,
  type ModeConfig,
  type Session,
} from '../src/index.js';

// A subagent child keeps the provider and model the runner gave it on every
// later turn the runner does not start: a new message, an answer, a turn under
// a context the host built. The binding is kept with the child session's
// durable data, as the PHP reference keeps it, not only on the row a paused
// run stopped on. A child session with no saved binding is refused
// (child_binding_missing) rather than run on the root rules, and a context the
// host supplies never replaces what the child saved.
//
// Driven through runtime.send(); the assertions read the provider and model
// off the request the model client was actually sent. fetch is refused.

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
  readonly queue: LlmResponse[] = [];
  readonly requests: LlmRequest[] = [];
  readonly runtime: AgentRuntime;
  readonly harness: PrismHarness;
  readonly ephemeral = new ExpiringStore();
  #call = 0;

  constructor() {
    const modes = new ModeRegistry({
      default: 'chat',
      modes: {
        chat: {
          system_prompt: 'You delegate.', tools: ['research'], max_steps: 20,
          subagents: { research: { mode: 'research', provider: 'custom-cheap', model: 'cheap-model' } },
        } as ModeConfig,
        research: { system_prompt: 'Research.', tools: ['weather'], requires_approval: ['weather'], max_steps: 8 } as ModeConfig,
      },
    });
    const tools = new ToolRegistry().register({ name: 'weather', handle: () => 'Sunny' });
    this.runtime = new AgentRuntime({
      config: { provider: 'anthropic', model: 'root-model' },
      modes,
      tools,
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

  /** Provider and model of every request after the first `from`. */
  sent(from: number): { provider: string; model: string }[] {
    return this.requests.slice(from).map((request) => ({ provider: request.provider, model: request.model }));
  }
}

const cheap = { provider: 'custom-cheap', model: 'cheap-model' };

/** The parent delegates; the child answers in text (pause = false) or stops on an approval. */
async function startedChild(pause: boolean): Promise<{ r: Rig; child: Session }> {
  const r = new Rig();
  r.calls([['research', { task: 'Find the weather.' }]]);
  if (pause) r.calls([['weather', { city: 'Paris' }]]);
  else r.text('Child answer.');
  r.text('Parent answer.');
  await r.runtime.send(r.session(), 'Go');
  expect(r.sent(1)[0]).toEqual(cheap);

  return { r, child: r.session('chat::sub::research') };
}

describe('a subagent child keeps its saved binding outside the runner', () => {
  it('sends a new message to a finished child on the binding the runner gave it', async () => {
    const { r, child } = await startedChild(false);
    const before = r.requests.length;

    r.text('Again.');
    await r.runtime.send(child, 'New question');

    expect(r.sent(before)).toEqual([cheap]);
  });

  it('sends a new message to a child paused on an approval on its binding', async () => {
    const { r, child } = await startedChild(true);
    const before = r.requests.length;

    r.text('Again.');
    await r.runtime.send(child, 'New question');

    expect(r.sent(before)).toEqual([cheap]);
  });

  it('keeps the binding after the ephemeral half of the session is gone', async () => {
    const { r, child } = await startedChild(false);
    const before = r.requests.length;
    await r.ephemeral.expireAll();

    r.text('Again.');
    await r.runtime.send(child, 'New question');

    expect(r.sent(before)).toEqual([cheap]);
  });

  it('runs a turn under a host-built context on the child binding', async () => {
    const { r, child } = await startedChild(false);
    const before = r.requests.length;

    r.text('Again.');
    await r.runtime.send(child, 'New question', undefined, RunContext.root('host_root', new RunBudget(100)));

    expect(r.sent(before)).toEqual([cheap]);
  });

  it('never lets a host context carrying a binding replace the saved one or the saved run tree', async () => {
    const { r, child } = await startedChild(false);
    const refs = await child.childRefs();
    expect(refs?.tree).not.toBeNull();
    const before = r.requests.length;
    const forged = new RunContext(RunLedger.start('forged'), new RunBudget(100), null, 0, {}, {
      session: { provider: null, model: null },
      declared: { provider: 'forged-provider', model: 'forged-model' },
      parentMode: { provider: null, model: null },
    });

    r.text('Forged turn.');
    await r.runtime.send(child, 'Forged question', undefined, forged).catch(() => undefined);
    r.text('Again.');
    await r.runtime.send(child, 'New question');

    expect(r.sent(before).every((seen) => seen.provider === cheap.provider && seen.model === cheap.model)).toBe(true);
    expect(await child.childRefs()).toEqual(refs);
  });

  it('refuses a child session with no saved binding, before any request', async () => {
    const r = new Rig();
    const child = r.session('chat::sub::research');
    // A child written before the binding was kept on the session: its run
    // tree reference is saved, its binding is not.
    await child.rememberChildRefs({ tree: { session: 'some-root', run: 'some-run' }, planLedger: null });

    r.text('Should not run.');
    await expect(r.runtime.send(child, 'New question')).rejects.toMatchObject({ code: 'child_binding_missing' });
    expect(r.requests).toHaveLength(0);
  });
});

// The runner starts a child under a run tree the HOST built: the parent's turn
// was handed a RunContext the host made. The child's context is still the
// runner's own (it built it with forChild and bound it), so its binding is
// saved with the child session exactly as under a tree this runtime opened.
// Without that, the child's next turn would run on the default provider. The
// PHP reference does this (RunTrees::startingChild); the parity row is
// harness-plans hpl-0059.
describe('a child the runner starts under a run tree the host built', () => {
  async function hostTreeChild(pause: boolean): Promise<{ r: Rig; child: Session }> {
    const r = new Rig();
    r.calls([['research', { task: 'Find the weather.' }]]);
    if (pause) r.calls([['weather', { city: 'Paris' }]]);
    else r.text('Child answer.');
    r.text('Parent answer.');
    await r.runtime.send(r.session(), 'Go', undefined, RunContext.root('host_root', new RunBudget(100)));
    expect(r.sent(1)[0]).toEqual(cheap);

    return { r, child: r.session('chat::sub::research') };
  }

  it('saves the binding the runner gave it, so a new message runs on it', async () => {
    const { r, child } = await hostTreeChild(false);
    expect(await child.childBinding()).not.toBeNull();
    const before = r.requests.length;

    r.text('Again.');
    await r.runtime.send(child, 'New question');

    expect(r.sent(before)).toEqual([cheap]);
  });

  it('resumes the child paused on an approval on that binding', async () => {
    const { r, child } = await hostTreeChild(true);
    const before = r.requests.length;

    r.text('Again.');
    await r.runtime.send(child, 'New question');

    expect(r.sent(before)).toEqual([cheap]);
  });

  it('keeps that binding after the ephemeral half of the session is gone', async () => {
    const { r, child } = await hostTreeChild(false);
    const before = r.requests.length;
    await r.ephemeral.expireAll();

    r.text('Again.');
    await r.runtime.send(child, 'New question');

    expect(r.sent(before)).toEqual([cheap]);
  });

  it('never saves a binding the host put on a context it built itself', async () => {
    const r = new Rig();
    const session = r.session('plain');
    const hosted = new RunContext(RunLedger.start('host_root'), new RunBudget(100), null, 0, {}, {
      session: { provider: null, model: null },
      declared: { provider: 'custom-cheap', model: 'cheap-model' },
      parentMode: { provider: null, model: null },
    });

    r.text('Hosted.');
    await r.runtime.send(session, 'Go', undefined, hosted);
    r.text('Plain.');
    await r.runtime.send(session, 'Again');

    expect(r.sent(0)).toEqual([cheap, { provider: 'anthropic', model: 'root-model' }]);
    expect(await session.childBinding()).toBeNull();
  });
});
