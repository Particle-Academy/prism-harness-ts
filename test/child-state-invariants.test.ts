import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  HarnessEvents,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  RunBudget,
  RunContext,
  Session,
  SubagentRunner,
  ToolRegistry,
  recordApproval,
  type Durability,
  type JsonObject,
  type LlmRequest,
  type LlmResponse,
  type ModeConfig,
} from '../src/index.js';

/*
 * G-387 round 11, item 7: THE INVARIANT MATRIX.
 *
 * Every reader of a child's state, through every phase of a runner start,
 * from INSIDE the start's own run (the Session object the start runs on, and
 * the read-only view a tool provider is handed) and from OUTSIDE it (a fresh
 * Session object on the same child, as a host turn, a resume or a person's
 * decision builds one). The expected table is written FIRST, as data, from
 * the rules of BRIEF-R7..R11; the test then measures every cell.
 *
 * Every future fix must keep this table green. A cell may only change with
 * the rule that decides it.
 *
 * Tokens:
 *   PRIOR  the value the child had before this start (its last confirmed state)
 *   NEW    the value this start committed
 *   ABSENT no value (binding null, depth 0, ceiling null): fail closed
 *   BOUND  a binding is present (the start reuses the declaration, so the
 *          binding's VALUE is the same before and after; presence is the fact)
 *   approval cells hold the code recordApproval answers for the approval the
 *   child was parked on before the start (one nobody can answer: too deep).
 */

type Phase = 'before_gate' | 'after_b' | 'own_run' | 'after_task' | 'failed_undone' | 'failed_undo_failed';
type Reader = 'mode' | 'binding' | 'refs' | 'depth' | 'ceiling' | 'run' | 'offer';
type Cells = Record<Reader, string>;

const ALL = (token: string, binding = 'BOUND'): Cells => ({ mode: token, binding, refs: token, depth: token, ceiling: token, run: token, offer: token });
/** A start's records before its task row, read by anyone but the start itself (round 9 + round 10 items 1 and 3). */
const PROVISIONAL: Cells = { mode: 'PRIOR', binding: 'ABSENT', refs: 'PRIOR', depth: 'ABSENT', ceiling: 'ABSENT', run: 'PRIOR', offer: 'PRIOR' };
const TOO_DEEP = 'run_not_permitted';
const NO_BINDING = 'child_binding_missing';
const ENDED = 'approval_not_pending';

export const EXPECTED: Record<Phase, { inside: Cells; outside: Cells & { approval: string } }> = {
  // Nothing written yet: everyone reads the child as it was.
  before_gate: { inside: ALL('PRIOR'), outside: { ...ALL('PRIOR'), approval: TOO_DEEP } },
  // Commit (b) wrote the records provisionally. The start's own Session reads
  // what it committed (round 11 item 5); its run has not begun, so the
  // current run is still the one before it. Everyone else: provisional.
  after_b: { inside: { ...ALL('NEW'), run: 'PRIOR' }, outside: { ...PROVISIONAL, approval: NO_BINDING } },
  // Inside the start's own run, before its task row (tool resolution, the
  // plan context): its own state, its own run. Outside: still provisional.
  own_run: { inside: ALL('NEW'), outside: { ...PROVISIONAL, approval: NO_BINDING } },
  // The task row confirms the start for every reader; it ended the parked call.
  after_task: { inside: ALL('NEW'), outside: { ...ALL('NEW'), approval: ENDED } },
  // A failed start whose undo ran: every record is back, for every reader
  // (the start's own Session included: its view ends with its run).
  failed_undone: { inside: ALL('PRIOR'), outside: { ...ALL('PRIOR'), approval: TOO_DEEP } },
  // A failed start whose undo failed too: the records stay provisional for
  // good, so every reader (the start's own Session too) reads them absent,
  // keeping the last confirmed tree and mode (round 10 items 1 and 3).
  failed_undo_failed: { inside: PROVISIONAL, outside: { ...PROVISIONAL, approval: NO_BINDING } },
};

const network = vi.fn(() => { throw new Error('A test tried to use the network.'); });
beforeEach(() => { vi.stubGlobal('fetch', network); });
afterEach(() => { expect(network).not.toHaveBeenCalled(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });

class Store extends MemorySessionStore {
  constructor(private readonly kind: Durability) {
    super();
  }

  override durability(): Durability {
    return this.kind;
  }
}

class Rig {
  readonly queue: LlmResponse[] = [];
  readonly durable = new Store('durable');
  readonly ephemeral = new Store('volatile');
  readonly registry: ToolRegistry;
  readonly runtime: AgentRuntime;
  readonly harness: PrismHarness;
  /** Run once, before the child's first model request is answered. */
  beforeChildRequest: (() => Promise<void>) | null = null;
  /** Run once, the first time the start's tools are resolved (inside its run, before its task row). */
  onResolve: ((view: Session) => Promise<void>) | null = null;
  #call = 0;
  #resolving = false;

  constructor() {
    const modes = new ModeRegistry({
      default: 'lead',
      modes: {
        lead: { system_prompt: 'Lead.', tools: ['echo', 'a'], max_steps: 8, requires_approval: ['echo'], subagents: { a: { mode: 'work', max_steps: 8 } } } as ModeConfig,
        // The same delegation from a parent with fewer steps: a smaller ceiling.
        short: { system_prompt: 'Lead.', tools: ['echo', 'a'], max_steps: 4, requires_approval: ['echo'], subagents: { a: { mode: 'work', max_steps: 8 } } } as ModeConfig,
        work: { system_prompt: 'Work.', tools: ['*'], max_steps: 8, requires_approval: ['echo'] } as ModeConfig,
      },
    });
    this.registry = new ToolRegistry().register({ name: 'echo', handle: () => 'ok' });
    // A host provider that offers `probe` to every child, and `danger` only
    // to a session whose mode is not the child's own (`work`): a reader that
    // reports the wrong mode shows in the offer.
    this.registry.registerProvider(async (session) => {
      if (!session.scope.includes('::sub::')) return [];
      if (!this.#resolving && this.onResolve !== null) {
        const hook = this.onResolve;
        this.onResolve = null;
        this.#resolving = true;
        try { await hook(session); } finally { this.#resolving = false; }
      }
      const tools = [{ name: 'probe', handle: () => 'probed' }];
      return (await session.mode()) === 'work' ? tools : [...tools, { name: 'danger', handle: () => 'boom' }];
    }, 'host', 'host_probe');
    this.runtime = new AgentRuntime({
      modes, tools: this.registry, events: new HarnessEvents(),
      client: async (request: LlmRequest) => {
        if (request.systemPrompt.startsWith('Work.') && this.beforeChildRequest !== null) {
          const hook = this.beforeChildRequest;
          this.beforeChildRequest = null;
          await hook();
        }
        return this.queue.shift() ?? { text: 'Done.', finishReason: 'stop' };
      },
    });
    const durable = this.durable, ephemeral = this.ephemeral;
    this.harness = new PrismHarness({ drivers: { ephemeral: () => ephemeral, durable: () => durable }, stores: { ephemeral: 'ephemeral', durable: 'durable' }, runtime: this.runtime });
    this.registry.register(new SubagentRunner(this.harness, this.runtime)
      .tool(modes.resolve('lead').subagents.a!, this.session(), RunContext.root('host', new RunBudget(100)), 'host_run'));
  }

  session(scope = 'chat'): Session {
    return this.harness.for({ type: 'User', id: 1 }).session(scope);
  }

  child(): Session {
    return this.session('chat::sub::a');
  }

  calls(calls: [string, JsonObject][]): void {
    this.queue.push({ text: '', finishReason: 'tool_calls', toolCalls: calls.map(([name, args]) => ({ id: `call_${++this.#call}`, name, arguments: args })) });
  }

  text(text: string): void {
    this.queue.push({ text, finishReason: 'stop' });
  }

  /** The child's parked approval ids. */
  async approvals(): Promise<string[]> {
    return (await this.child().thread().messages()).flatMap((entry) => Array.isArray(entry.message.tool_approval_requests)
      ? (entry.message.tool_approval_requests as JsonObject[]).map((request) => request.approval_id as string) : []);
  }

  /** A delegation through the subagent tool in a parent turn under `context`. */
  async delegate(task: string, child: () => void, context?: RunContext): Promise<void> {
    this.calls([['a', { task }]]);
    child();
    this.text('Parent done.');
    expect((await this.runtime.send(this.session(), 'Delegate', undefined, context)).text).toBe('Parent done.');
  }
}

interface Values { mode: unknown; binding: unknown; refs: unknown; depth: unknown; ceiling: unknown; run: unknown; offer: unknown }

async function read(r: Rig, session: Session): Promise<Values> {
  return {
    mode: await session.mode(),
    binding: await session.childBinding(),
    refs: (await session.childRefs())?.tree ?? null,
    depth: await session.childDepth(),
    ceiling: await session.childCeiling(),
    run: (await session.run())?.id ?? null,
    offer: (await r.runtime.effectiveTools(session)).map((tool) => tool.name).sort(),
  };
}

async function approval(r: Rig, id: string): Promise<string> {
  return recordApproval(r.child(), id, true).then(() => 'accepted', (error: { code?: string; message?: string }) => error.code ?? String(error.message));
}

const ABSENT: Record<Reader, unknown> = { mode: null, binding: null, refs: null, depth: 0, ceiling: null, run: null, offer: [] };

function token(reader: Reader, value: unknown, prior: Values, next: Values | null): string {
  const same = (a: unknown, b: unknown): boolean => JSON.stringify(a) === JSON.stringify(b);
  if (reader === 'binding') return value === null ? 'ABSENT' : 'BOUND';
  if (next !== null && same(value, next[reader]) && !same(value, prior[reader])) return 'NEW';
  if (same(value, prior[reader])) return 'PRIOR';
  if (same(value, ABSENT[reader])) return 'ABSENT';
  return `?${JSON.stringify(value)}`;
}

function cells(values: Values, prior: Values, next: Values | null): Cells {
  const out = {} as Cells;
  for (const reader of Object.keys(ABSENT) as Reader[]) out[reader] = token(reader, values[reader], prior, next);
  return out;
}

/**
 * The child before the start: delegated once in a parent turn (tree T0),
 * parked on an echo approval, then left too deep to answer, with a mode the
 * host set (`lead`). Every reader's PRIOR value differs from the start's NEW one.
 */
async function prepared(): Promise<{ r: Rig; parked: string; prior: Values }> {
  const r = new Rig();
  await r.session().usingMode('lead');
  await r.delegate('First task', () => r.calls([['echo', { n: 1 }]]));
  const parked = (await r.approvals()).pop()!;
  const key = `${r.child().key()}:durable`;
  await r.durable.put(key, { ...((await r.durable.get(key)) ?? {}), child_depth: 1_000 });
  await r.child().usingMode('lead');
  const prior = await read(r, r.child());
  expect(prior.mode).toBe('lead');
  expect(prior.depth).toBe(1_000);
  expect(prior.binding).not.toBeNull();
  expect(prior.run).not.toBeNull();
  expect(await approval(r, parked)).toBe(TOO_DEEP);

  return { r, parked, prior };
}

/** The start under test: a re-delegation from a new parent turn (a new tree T1) in a mode that leaves the child 3 steps (ceiling 3). */
async function second(r: Rig, child: () => void): Promise<void> {
  await r.session().usingMode('short');
  await r.delegate('Second task', child);
}

/** Spy on `method` of the child's Session; `hook` runs on the start's own Session (`this`) once. */
function onChild<K extends 'childTurnLive' | 'rememberChildStart' | 'beginRun'>(method: K, after: boolean, hook: (own: Session) => Promise<void>): void {
  const original = Session.prototype[method] as (...args: unknown[]) => Promise<unknown>;
  let fired = false;
  vi.spyOn(Session.prototype, method).mockImplementation(async function (this: Session, ...args: unknown[]) {
    const mine = !fired && this.scope.includes('::sub::');
    if (mine) fired = true;
    if (mine && !after) await hook(this);
    const result = await original.apply(this, args);
    if (mine && after) await hook(this);
    return result;
  } as never);
}

interface Measured { inside: Cells; outside: Cells & { approval: string } }

describe('G-387 round 11 item 7: the child-state invariant matrix', () => {
  it('a start that completes: before the gate, after (b), during its own run, after its task row', async () => {
    const { r, parked, prior } = await prepared();
    const raw: Partial<Record<Phase, { inside: Values; outside: Values; approval: string }>> = {};
    const measure = async (phase: Phase, own: Session): Promise<void> => {
      raw[phase] = { inside: await read(r, own), outside: await read(r, r.child()), approval: await approval(r, parked) };
    };
    let own: Session | null = null;
    onChild('childTurnLive', false, (session) => measure('before_gate', session));
    onChild('rememberChildStart', true, async (session) => {
      await measure('after_b', session);
      // Armed only now: the readers above resolve tools too.
      r.onResolve = (view) => measure('own_run', view);
    });
    onChild('beginRun', false, async (session) => { own = session; });
    r.beforeChildRequest = () => measure('after_task', own!);

    await second(r, () => r.text('Child done.'));
    vi.restoreAllMocks();
    const next = await read(r, r.child());
    expect(next.mode).toBe('work');
    expect(next.depth).toBe(1);
    expect(next.ceiling).toBe(3);
    expect(next.refs).not.toEqual(prior.refs);

    const measured: Partial<Record<Phase, Measured>> = {};
    for (const phase of ['before_gate', 'after_b', 'own_run', 'after_task'] as const) {
      const got = raw[phase];
      expect(got, `${phase} was reached`).toBeDefined();
      measured[phase] = { inside: cells(got!.inside, prior, next), outside: { ...cells(got!.outside, prior, next), approval: got!.approval } };
    }
    expect(measured).toEqual({ before_gate: EXPECTED.before_gate, after_b: EXPECTED.after_b, own_run: EXPECTED.own_run, after_task: EXPECTED.after_task });
  });

  for (const phase of ['failed_undone', 'failed_undo_failed'] as const) {
    it(`a start that fails after (b), before its task row, ${phase === 'failed_undone' ? 'its undo done' : 'its undo failing too'}`, async () => {
      const { r, parked, prior } = await prepared();
      let own: Session | null = null;
      onChild('beginRun', false, async (session) => { own = session; });
      const resolve = ToolRegistry.prototype.resolveDetailed;
      vi.spyOn(ToolRegistry.prototype, 'resolveDetailed').mockImplementation(function (this: ToolRegistry, ...args: Parameters<ToolRegistry['resolveDetailed']>) {
        if (args[1]?.scope.includes('::sub::') === true) throw new Error('REGISTRY-SECRET');
        return resolve.apply(this, args);
      });
      if (phase === 'failed_undo_failed') vi.spyOn(Session.prototype, 'undoChildStart').mockRejectedValue(new Error('STORE-SECRET'));

      await second(r, () => {});
      vi.restoreAllMocks();
      expect(own).not.toBeNull();

      const measured: Measured = {
        inside: cells(await read(r, own!), prior, null),
        outside: { ...cells(await read(r, r.child()), prior, null), approval: await approval(r, parked) },
      };
      expect(measured).toEqual(EXPECTED[phase]);
    });
  }
});
