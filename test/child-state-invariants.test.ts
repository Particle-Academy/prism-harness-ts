import { AsyncLocalStorage } from 'node:async_hooks';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  HarnessEvents,
  MemorySessionStore,
  ModeRegistry,
  Plans,
  PrismHarness,
  RunBudget,
  RunContext,
  Session,
  SubagentRunner,
  ToolRegistry,
  recordApproval,
  type Durability,
  type HarnessEvent,
  type JsonObject,
  type LlmRequest,
  type LlmResponse,
  type ModeConfig,
} from '../src/index.js';
import { endedByStart, pausedApprovalIds } from '../src/thread-rows.js';
import { approvalDecisions, pendingApprovalCalls } from '../src/plans/thread.js';
import { activeTurn } from '../src/runtime.js';

/*
 * G-387 round 11 item 7, extended in round 12 (items 3 and 8): THE INVARIANT
 * MATRIX.
 *
 * Every reader of a child's state, through every phase of a runner start,
 * from INSIDE the start's own run (the Session object the start runs on, and
 * the read-only view a tool provider is handed) and from OUTSIDE it (a fresh
 * Session object on the same child, as a host turn, a resume or a person's
 * decision builds one), plus, in round 12, the start's OWN object used from
 * outside its run (a deferred registered in the test's own async context).
 * The expected table is written FIRST, as data, from the rules of
 * BRIEF-R7..R12; the test then measures every cell.
 *
 * Every future fix must keep this table green. A cell may only change with
 * the rule that decides it.
 *
 * Tokens:
 *   PRIOR  the value the child had before this start (its last confirmed state)
 *   NEW    the value this start committed
 *   ABSENT no value (binding null, depth 0, ceiling null, no tree): fail closed
 *   BOUND  a binding is present (the start reuses the declaration, so the
 *          binding's VALUE is the same before and after; presence is the fact)
 *   yes/no a yes-or-no reader
 *   START  plan_run names the start's own run; PRIOR: the parked run (a run
 *          parked on approvals stays `running`: send() returns
 *          awaiting_approval without closing it); NONE: no run in flight
 *   action cells hold what the action answered: a harness error code,
 *   `accepted`, or a runner outcome (`denied`: refused, and nothing written,
 *   no model request). An empty action cell is not measured in that phase
 *   (the action would succeed and change what the later phases measure).
 *
 * Readers (round 12 added provisional .. policy):
 *   mode, binding, refs (its tree), depth, ceiling, run (the current run),
 *   offer (effectiveTools with a host provider that adds `danger` whenever
 *   mode() is not `work`), provisional (childStartUnconfirmed), plans
 *   (Plans.enabledFor), plan_run (Plans.runId), paused (the thread tail is
 *   parked on approvals: the paused row), busy (childTurnLive, the gate's
 *   child_busy fact), resumed_tree (resumedRunTree), policy (the tool policy
 *   of the mode the session reads: its requires_approval list).
 * Actions:
 *   approval (recordApproval on a fresh Session), resume (send '' on a fresh
 *   Session), second_start (a second runner start onto the same child),
 *   decision_same (recordApproval on the start's OWN Session), own_resume /
 *   own_new (send '' / a new message on the start's own Session).
 *
 * Not columns in TS, and why: isLinkedChild has no reader of its own here
 * (the runtime folds scope, binding, refs and the paused row, each a column);
 * the delegation id is the count record, whose live hold is `busy`; cancel
 * reads no child state (RunContext's ledger).
 */

type Phase =
  | 'before_gate' | 'after_b' | 'run_started' | 'own_run' | 'tool_denied' | 'after_task'
  | 'failed_undone' | 'failed_undone_during' | 'failed_undo_failed' | 'failed_undo_failed_during';
type Reader = 'mode' | 'binding' | 'refs' | 'depth' | 'ceiling' | 'run' | 'offer'
  | 'provisional' | 'plans' | 'plan_run' | 'paused' | 'busy' | 'resumed_tree' | 'policy';
type Cells = Record<Reader, string>;
type InsideAction = 'decision_same' | 'own_resume' | 'own_new';
type OutsideAction = 'approval' | 'resume' | 'second_start';
interface Side { inside: Cells & Partial<Record<InsideAction, string>>; outside: Cells & Partial<Record<OutsideAction, string>>; own_outside?: Cells & Partial<Record<'decision_same' | 'own_resume', string>> }

const TOO_DEEP = 'run_not_permitted';
const NO_BINDING = 'child_binding_missing';
const ENDED = 'approval_not_pending';
const DENIED = 'denied';

/** Before the start, or after a start that was undone: the child as it was. */
const AS_BEFORE: Cells = {
  mode: 'PRIOR', binding: 'BOUND', refs: 'PRIOR', depth: 'PRIOR', ceiling: 'PRIOR', run: 'PRIOR', offer: 'PRIOR',
  provisional: 'no', plans: 'no', plan_run: 'PRIOR', paused: 'yes', busy: 'no', resumed_tree: 'PRIOR', policy: 'PRIOR',
};
/**
 * A start's records before its task row, read by anyone but the start itself
 * (round 9 + round 10 items 1 and 3): absent, keeping the last confirmed tree
 * and mode. The start's count hold is a live turn whatever its records say,
 * so `busy` reads yes (a second start is denied, fail closed).
 */
const PROVISIONAL: Cells = {
  mode: 'PRIOR', binding: 'ABSENT', refs: 'PRIOR', depth: 'ABSENT', ceiling: 'ABSENT', run: 'PRIOR', offer: 'PRIOR',
  provisional: 'yes', plans: 'no', plan_run: 'PRIOR', paused: 'yes', busy: 'yes', resumed_tree: 'PRIOR', policy: 'PRIOR',
};
/** The start's own view of what it committed (round 11 item 5), its run begun; the thread still parked. */
const OWN_VIEW: Cells = {
  mode: 'NEW', binding: 'BOUND', refs: 'NEW', depth: 'NEW', ceiling: 'NEW', run: 'NEW', offer: 'NEW',
  provisional: 'no', plans: 'yes', plan_run: 'START', paused: 'yes', busy: 'yes', resumed_tree: 'PRIOR', policy: 'NEW',
};
/** After the task row, for everyone: confirmed, the parked call ended, the start's turn still running. */
const CONFIRMED: Cells = { ...OWN_VIEW, paused: 'no', resumed_tree: 'ABSENT' };
/** Actions on the start's own object before its task row: each has a scope of its own, so the provisional rule. */
const OWN_REFUSED = { decision_same: NO_BINDING, own_resume: NO_BINDING, own_new: NO_BINDING };

export const EXPECTED: Record<Phase, Side> = {
  // Nothing written yet: everyone reads the child as it was.
  before_gate: { inside: AS_BEFORE, outside: { ...AS_BEFORE, approval: TOO_DEEP } },
  // Commit (b) wrote the records provisionally. The start's own Session reads
  // what it committed (round 11 item 5); its run has not begun, so the
  // current run is still the one before it. Everyone else, and the start's
  // own object used from outside its run: provisional.
  after_b: {
    inside: { ...OWN_VIEW, run: 'PRIOR', plan_run: 'PRIOR', ...OWN_REFUSED },
    outside: { ...PROVISIONAL, approval: NO_BINDING, resume: NO_BINDING, second_start: DENIED },
    own_outside: { ...PROVISIONAL, decision_same: NO_BINDING, own_resume: NO_BINDING },
  },
  // Inside a run.started listener (round 12, Py phase run_started): the run
  // has begun, the task row is not written.
  run_started: {
    inside: { ...OWN_VIEW, ...OWN_REFUSED },
    outside: { ...PROVISIONAL, approval: NO_BINDING, resume: NO_BINDING, second_start: DENIED },
  },
  // Inside the start's own run, before its task row (tool resolution): its
  // own state, its own run. Outside, and its own object outside its scope:
  // still provisional.
  own_run: {
    inside: { ...OWN_VIEW, ...OWN_REFUSED },
    outside: { ...PROVISIONAL, approval: NO_BINDING, resume: NO_BINDING, second_start: DENIED },
    own_outside: { ...PROVISIONAL, decision_same: NO_BINDING, own_resume: NO_BINDING },
  },
  // Inside a tool_denied listener for the call the task row ended (PHP
  // ToolDenied(A0)): announced only once the row is recorded, so confirmed.
  tool_denied: {
    inside: { ...CONFIRMED, decision_same: ENDED },
    outside: { ...CONFIRMED, approval: ENDED },
  },
  // The task row confirms the start for every reader; it ended the parked
  // call. Its turn still holds the count: a second start is denied.
  after_task: {
    inside: { ...CONFIRMED, decision_same: ENDED },
    outside: { ...CONFIRMED, approval: ENDED, second_start: DENIED },
  },
  // A failed start whose undo ran: every record is back, for every reader
  // (the start's own Session included: its view ends first on the failure
  // path), measured after the run and DURING the failure (a run.failed
  // listener), when every hold is already released (round 12 item 4).
  failed_undone: { inside: AS_BEFORE, outside: { ...AS_BEFORE, approval: TOO_DEEP } },
  failed_undone_during: { inside: { ...AS_BEFORE, decision_same: TOO_DEEP }, outside: { ...AS_BEFORE, approval: TOO_DEEP } },
  // A failed start whose undo failed too: the records stay provisional for
  // good, so every reader (the start's own Session too) reads them absent,
  // keeping the last confirmed tree and mode (round 10 items 1 and 3). The
  // count hold the undo would have settled stays until it expires (busy).
  failed_undo_failed: { inside: PROVISIONAL, outside: { ...PROVISIONAL, approval: NO_BINDING } },
  failed_undo_failed_during: { inside: { ...PROVISIONAL, decision_same: NO_BINDING }, outside: { ...PROVISIONAL, approval: NO_BINDING } },
};

const network = vi.fn(() => { throw new Error('A test tried to use the network.'); });
beforeEach(() => { vi.stubGlobal('fetch', network); });
afterEach(() => { expect(network).not.toHaveBeenCalled(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });

/** A store that mirrors what it holds, so a test can compare every record byte for byte. */
class Store extends MemorySessionStore {
  readonly mirror = new Map<string, string>();

  constructor(private readonly kind: Durability) {
    super();
  }

  override durability(): Durability {
    return this.kind;
  }

  override async put(key: string, payload: JsonObject, ttlSeconds: number | null = null): Promise<void> {
    await super.put(key, payload, ttlSeconds);
    this.mirror.set(key, JSON.stringify(payload));
  }

  override async forget(key: string): Promise<void> {
    await super.forget(key);
    this.mirror.delete(key);
  }
}

/** Set while the test itself measures: its reads never hold the run, and never fire a hook. */
const measuring = new AsyncLocalStorage<true>();

class Rig {
  readonly queue: LlmResponse[] = [];
  readonly requests: LlmRequest[] = [];
  readonly durable = new Store('durable');
  readonly ephemeral = new Store('volatile');
  readonly events = new HarnessEvents();
  readonly registry: ToolRegistry;
  readonly modes: ModeRegistry;
  readonly plans: Plans;
  readonly runtime: AgentRuntime;
  readonly harness: PrismHarness;
  /** Run once, before the child's first model request is answered. */
  beforeChildRequest: (() => Promise<void>) | null = null;
  /** Run once, the first time the start's tools are resolved (inside its run, before its task row). */
  onResolve: ((view: Session) => Promise<void>) | null = null;
  /** A measurement a listener started: the run waits for it at its next tool resolution or model request. */
  pending: Promise<void> | null = null;
  /** Every child model request (not a measurement's) throws: a start that fails AFTER its task row (round 13). */
  failChild = false;
  /** How many times `echo` ran. */
  echoed = 0;
  #call = 0;

  constructor() {
    this.modes = new ModeRegistry({
      default: 'lead',
      modes: {
        lead: { system_prompt: 'Lead.', tools: ['echo', 'a'], max_steps: 8, requires_approval: ['echo'], subagents: { a: { mode: 'work', max_steps: 8 } } } as ModeConfig,
        // The same delegation from a parent with fewer steps: a smaller ceiling.
        short: { system_prompt: 'Lead.', tools: ['echo', 'a'], max_steps: 4, requires_approval: ['echo'], subagents: { a: { mode: 'work', max_steps: 8 } } } as ModeConfig,
        // The child's own mode: plans on, and a tool policy of its own.
        work: { system_prompt: 'Work.', tools: ['*'], max_steps: 8, requires_approval: ['echo', 'danger'], plans: true } as ModeConfig,
      },
    });
    this.registry = new ToolRegistry().register({ name: 'echo', handle: () => { this.echoed += 1; return 'ok'; } });
    // A host provider that offers `probe` to every child, and `danger` only
    // to a session whose mode is not the child's own (`work`): a reader that
    // reports the wrong mode shows in the offer.
    this.registry.registerProvider(async (session) => {
      if (!session.scope.includes('::sub::')) return [];
      if (measuring.getStore() !== true) {
        await this.#wait();
        if (this.onResolve !== null) {
          const hook = this.onResolve;
          this.onResolve = null;
          await measuring.run(true, () => hook(session));
        }
      }
      const tools = [{ name: 'probe', handle: () => 'probed' }];
      return (await session.mode()) === 'work' ? tools : [...tools, { name: 'danger', handle: () => 'boom' }];
    }, 'host', 'host_probe');
    this.plans = new Plans({ modes: this.modes, config: { enabled: true }, events: this.events });
    this.runtime = new AgentRuntime({
      modes: this.modes, tools: this.registry, events: this.events,
      client: async (request: LlmRequest) => {
        if (measuring.getStore() !== true) {
          await this.#wait();
          if (request.systemPrompt.startsWith('Work.') && this.beforeChildRequest !== null) {
            const hook = this.beforeChildRequest;
            this.beforeChildRequest = null;
            await measuring.run(true, hook);
          }
        }
        this.requests.push(request);
        if (this.failChild && measuring.getStore() !== true && request.systemPrompt.startsWith('Work.')) throw new Error('MODEL-SECRET');
        // A measurement never takes the scripted answers (if it reaches the
        // model at all, the request count shows it).
        if (measuring.getStore() === true) return { text: 'Measured.', finishReason: 'stop' };
        return this.queue.shift() ?? { text: 'Done.', finishReason: 'stop' };
      },
    });
    const durable = this.durable, ephemeral = this.ephemeral;
    this.harness = new PrismHarness({ drivers: { ephemeral: () => ephemeral, durable: () => durable }, stores: { ephemeral: 'ephemeral', durable: 'durable' }, runtime: this.runtime, plans: this.plans });
    this.registry.register(new SubagentRunner(this.harness, this.runtime)
      .tool(this.modes.resolve('lead').subagents.a!, this.session(), RunContext.root('host', new RunBudget(100)), 'host_run'));
  }

  async #wait(): Promise<void> {
    while (this.pending !== null) {
      const pending = this.pending;
      await pending;
      if (this.pending === pending) this.pending = null;
    }
  }

  /** Start a measurement from a listener; the run waits for it. */
  measureFromListener(body: () => Promise<void>): void {
    const pending = measuring.run(true, body);
    this.pending = pending.catch(() => {});
    void pending.catch(() => {});
  }

  session(scope = 'chat'): Session {
    return this.harness.for({ type: 'User', id: 1 }).session(scope);
  }

  child(): Session {
    return this.session('chat::sub::a');
  }

  calls(calls: [string, JsonObject, string?][]): void {
    this.queue.push({ text: '', finishReason: 'tool_calls', toolCalls: calls.map(([name, args, id]) => ({ id: id ?? `call_${++this.#call}`, name, arguments: args })) });
  }

  text(text: string): void {
    this.queue.push({ text, finishReason: 'stop' });
  }

  childRequests(): number {
    return this.requests.filter((request) => request.systemPrompt.startsWith('Work.')).length;
  }

  /** The child's parked approval ids. */
  async approvals(): Promise<string[]> {
    return (await this.child().thread().messages()).flatMap((entry) => Array.isArray(entry.message.tool_approval_requests)
      ? (entry.message.tool_approval_requests as JsonObject[]).map((request) => request.approval_id as string) : []);
  }

  /** Every stored record of the child (durable, ephemeral, its thread), as bytes. */
  childRecords(): string {
    const out: [string, string][] = [];
    for (const store of [this.durable, this.ephemeral]) {
      for (const [key, value] of store.mirror) if (key.includes('::sub::')) out.push([key, value]);
    }
    return JSON.stringify(out.sort(([a], [b]) => a.localeCompare(b)));
  }

  /** A delegation through the subagent tool in a parent turn under `context`. */
  async delegate(task: string, child: () => void, context?: RunContext): Promise<void> {
    this.calls([['a', { task }]]);
    child();
    this.text('Parent done.');
    expect((await this.runtime.send(this.session(), 'Delegate', undefined, context)).text).toBe('Parent done.');
  }
}

interface Values { mode: unknown; binding: unknown; refs: unknown; depth: unknown; ceiling: unknown; run: unknown; offer: unknown; provisional: boolean; plans: boolean; plan_run: string | null; paused: boolean; busy: boolean; resumed_tree: unknown; policy: unknown }

async function read(r: Rig, session: Session): Promise<Values> {
  const now = Math.floor(Date.now() / 1000);
  return {
    mode: await session.mode(),
    binding: await session.childBinding(),
    refs: (await session.childRefs())?.tree ?? null,
    depth: await session.childDepth(),
    ceiling: await session.childCeiling(),
    run: (await session.run())?.id ?? null,
    offer: (await r.runtime.effectiveTools(session)).map((tool) => tool.name).sort(),
    provisional: await session.childStartUnconfirmed(),
    plans: await r.plans.enabledFor(session),
    plan_run: await r.plans.runId(session),
    // The thread is the child's, whichever object reads it (a provider's
    // read-only view refuses thread(): it is a write accessor).
    paused: pausedApprovalIds((await r.child().thread().messages()).map((entry) => entry.message)).length > 0,
    busy: await session.childTurnLive(now, r.runtime.holdTtl(session)),
    resumed_tree: await session.resumedRunTree(),
    policy: [...(await r.runtime.modeOf(session)).requiresApproval],
  };
}

const code = (error: { code?: string; message?: string }): string => error.code ?? `?${String(error.message)}`;

async function approve(session: Session, id: string): Promise<string> {
  return recordApproval(session, id, true).then(() => 'accepted', code);
}

async function turn(r: Rig, session: Session, prompt: string): Promise<string> {
  return r.runtime.send(session, prompt).then(() => 'ran', code);
}

/** A second runner start onto the same child: its outcome, `+wrote` if it wrote any child record or asked the model. */
async function secondStart(r: Rig): Promise<string> {
  const before = r.childRecords(), requests = r.childRequests();
  const result = await new SubagentRunner(r.harness, r.runtime)
    .run(r.modes.resolve('lead').subagents.a!, r.session(), RunContext.root('root2', new RunBudget(8)), 'root2', 'Second start');
  return before === r.childRecords() && requests === r.childRequests() ? result.outcome : `${result.outcome}+wrote`;
}

const ABSENT: Record<Reader, unknown> = { mode: null, binding: null, refs: null, depth: 0, ceiling: null, run: null, offer: [], provisional: false, plans: false, plan_run: null, paused: false, busy: false, resumed_tree: null, policy: [] };

function token(reader: Reader, value: unknown, prior: Values, next: Values | null, start: string | null): string {
  const same = (a: unknown, b: unknown): boolean => JSON.stringify(a) === JSON.stringify(b);
  if (reader === 'binding') return value === null ? 'ABSENT' : 'BOUND';
  if (reader === 'provisional' || reader === 'plans' || reader === 'paused' || reader === 'busy') return value === true ? 'yes' : value === false ? 'no' : `?${String(value)}`;
  if (reader === 'plan_run') return value === null ? 'NONE' : value === start ? 'START' : value === prior.run ? 'PRIOR' : `?${String(value)}`;
  if (reader === 'resumed_tree') return value === null ? 'ABSENT' : same(value, prior.resumed_tree) ? 'PRIOR' : `?${JSON.stringify(value)}`;
  if (next !== null && same(value, next[reader]) && !same(value, prior[reader])) return 'NEW';
  if (same(value, prior[reader])) return 'PRIOR';
  if (same(value, ABSENT[reader])) return 'ABSENT';
  return `?${JSON.stringify(value)}`;
}

function cells(values: Values, prior: Values, next: Values | null, start: string | null): Cells {
  const out = {} as Cells;
  for (const reader of Object.keys(ABSENT) as Reader[]) out[reader] = token(reader, values[reader], prior, next, start);
  return out;
}

/**
 * The child before the start: delegated once in a parent turn (tree T0),
 * parked on an echo approval, then left too deep to answer, with a mode the
 * host set (`lead`). Every reader's PRIOR value differs from the start's NEW one.
 */
async function prepared(step: [string, JsonObject, string?][] = [['echo', { n: 1 }]], decide = false): Promise<{ r: Rig; parked: string; prior: Values }> {
  const r = new Rig();
  await r.session().usingMode('lead');
  await r.delegate('First task', () => r.calls(step));
  const parked = (await r.approvals()).pop()!;
  // A person decides while the approval is answerable (round 13 scenarios).
  if (decide) expect(await approve(r.child(), parked)).toBe('accepted');
  const key = `${r.child().key()}:durable`;
  await r.durable.put(key, { ...((await r.durable.get(key)) ?? {}), child_depth: 1_000 });
  await r.child().usingMode('lead');
  const prior = await read(r, r.child());
  expect(prior.mode).toBe('lead');
  expect(prior.depth).toBe(1_000);
  expect(prior.binding).not.toBeNull();
  expect(prior.run).not.toBeNull();
  expect(prior.resumed_tree).not.toBeNull();
  expect(prior.paused).toBe(true);
  expect(await approve(r.child(), parked)).toBe(decide ? ENDED : TOO_DEEP);

  return { r, parked, prior };
}

/** The start under test: a re-delegation from a new parent turn (a new tree T1) in a mode that leaves the child 3 steps (ceiling 3). */
async function second(r: Rig, child: () => void): Promise<void> {
  await r.session().usingMode('short');
  await r.delegate('Second task', child);
}

/** Spy on `method` of the child's Session; `hook` runs on the start's own Session (`this`) once, inside its run's async context. */
function onChild<K extends 'childTurnLive' | 'rememberChildStart' | 'beginRun'>(method: K, after: boolean, hook: (own: Session, args: unknown[]) => Promise<void>): void {
  const original = Session.prototype[method] as (...args: unknown[]) => Promise<unknown>;
  let fired = false;
  vi.spyOn(Session.prototype, method).mockImplementation(async function (this: Session, ...args: unknown[]) {
    const mine = !fired && measuring.getStore() !== true && this.scope.includes('::sub::');
    if (mine) fired = true;
    if (mine && !after) await measuring.run(true, () => hook(this, args));
    const result = await original.apply(this, args);
    if (mine && after) await measuring.run(true, () => hook(this, args));
    return result;
  } as never);
}

/**
 * A deferred registered in the TEST's async context, outside any run: the
 * body runs there whenever the start triggers it, so the start's own object
 * is used from outside its run (TS review S4).
 */
function outsideDeferred(): { run: (body: () => Promise<void>) => Promise<void> } {
  let trigger!: (body: () => Promise<void>) => void;
  let ready = new Promise<() => Promise<void>>((resolve) => { trigger = resolve; });
  const loop = async (): Promise<void> => {
    for (;;) {
      const body = await ready;
      ready = new Promise((resolve) => { trigger = resolve; });
      await body();
    }
  };
  void loop();
  return {
    run: (body) => new Promise<void>((resolve, reject) => { trigger(() => body().then(resolve, reject)); }),
  };
}

type Measured = Partial<Record<Phase, Side>>;

describe('G-387 rounds 11-12: the child-state invariant matrix', () => {
  it('a start that completes: before the gate, after (b), at run.started, during its own run, at tool_denied, after its task row', async () => {
    const { r, parked, prior } = await prepared();
    const outside = outsideDeferred();
    let start: string | null = null;
    let own: Session | null = null;
    const raw: Partial<Record<Phase, { inside: Values; outside: Values; own_outside?: Values; actions: Record<string, string> }>> = {};
    const measure = async (phase: Phase, reader: Session, actions: { inside?: InsideAction[]; outside?: OutsideAction[]; ownOutside?: boolean } = {}): Promise<void> => {
      const done: Record<string, string> = {};
      const entry: { inside: Values; outside: Values; own_outside?: Values; actions: Record<string, string> } = { inside: await read(r, reader), outside: await read(r, r.child()), actions: done };
      if (actions.ownOutside === true) {
        const self = own!;
        await outside.run(async () => {
          entry.own_outside = await read(r, self);
          done['own_outside.decision_same'] = await approve(self, parked);
          done['own_outside.own_resume'] = await turn(r, self, '');
        });
      }
      for (const action of actions.outside ?? []) {
        done[`outside.${action}`] = action === 'approval' ? await approve(r.child(), parked)
          : action === 'resume' ? await turn(r, r.child(), '') : await secondStart(r);
      }
      for (const action of actions.inside ?? []) {
        done[`inside.${action}`] = action === 'decision_same' ? await approve(own!, parked)
          : await turn(r, own!, action === 'own_resume' ? '' : 'Injected');
      }
      raw[phase] = entry;
    };
    onChild('childTurnLive', false, (session) => measure('before_gate', session, { outside: ['approval'] }));
    onChild('rememberChildStart', true, async (session, args) => {
      own = session;
      start = (args[3] as { runId: string }).runId;
      await measure('after_b', session, { inside: ['decision_same', 'own_resume', 'own_new'], outside: ['approval', 'resume', 'second_start'], ownOutside: true });
      // Armed only now: the readers above resolve tools too.
      r.onResolve = (view) => measure('own_run', view, { inside: ['decision_same', 'own_resume', 'own_new'], outside: ['approval', 'resume', 'second_start'], ownOutside: true });
    });
    r.events.listen((event: HarnessEvent) => {
      if (event.type === 'run.started' && event.runId === start) r.measureFromListener(() => measure('run_started', own!, { inside: ['decision_same', 'own_resume', 'own_new'], outside: ['approval', 'resume', 'second_start'] }));
      if (event.type === 'tool_denied' && event.runId === start) r.measureFromListener(() => measure('tool_denied', own!, { inside: ['decision_same'], outside: ['approval'] }));
    });
    r.beforeChildRequest = () => measure('after_task', own!, { inside: ['decision_same'], outside: ['approval', 'second_start'] });

    await second(r, () => r.text('Child done.'));
    vi.restoreAllMocks();
    const next = await read(r, r.child());
    expect(next.mode).toBe('work');
    expect(next.depth).toBe(1);
    expect(next.ceiling).toBe(3);
    expect(next.refs).not.toEqual(prior.refs);
    // No action in any phase let the child take a model step beyond the start's own one.
    expect(r.childRequests()).toBe(2);

    const measured: Measured = {};
    for (const phase of ['before_gate', 'after_b', 'run_started', 'own_run', 'tool_denied', 'after_task'] as const) {
      const got = raw[phase];
      expect(got, `${phase} was reached`).toBeDefined();
      const side: Side = { inside: cells(got!.inside, prior, next, start), outside: cells(got!.outside, prior, next, start) };
      if (got!.own_outside !== undefined) side.own_outside = cells(got!.own_outside, prior, next, start);
      for (const [key, value] of Object.entries(got!.actions)) {
        const [where, action] = key.split('.') as ['inside' | 'outside' | 'own_outside', string];
        (side[where] as Record<string, string>)[action] = value;
      }
      measured[phase] = side;
    }
    const expected: Measured = {};
    for (const phase of ['before_gate', 'after_b', 'run_started', 'own_run', 'tool_denied', 'after_task'] as const) expected[phase] = EXPECTED[phase];
    expect(measured).toEqual(expected);
  });

  for (const phase of ['failed_undone', 'failed_undo_failed'] as const) {
    it(`a start that fails after (b), before its task row, ${phase === 'failed_undone' ? 'its undo done' : 'its undo failing too'}: measured during the failure (run.failed) and after`, async () => {
      const { r, parked, prior } = await prepared();
      let own: Session | null = null;
      let start: string | null = null;
      let during: Side | null = null;
      onChild('beginRun', false, async (session, args) => { own = session; start = args[0] as string; });
      r.events.listen((event: HarnessEvent) => {
        if (event.type !== 'run.failed' || event.runId !== start) return;
        r.measureFromListener(async () => {
          const inside = { ...cells(await read(r, own!), prior, null, start), decision_same: await approve(own!, parked) };
          const outside = { ...cells(await read(r, r.child()), prior, null, start), approval: await approve(r.child(), parked) };
          during = { inside, outside };
        });
      });
      const resolve = ToolRegistry.prototype.resolveDetailed;
      vi.spyOn(ToolRegistry.prototype, 'resolveDetailed').mockImplementation(function (this: ToolRegistry, ...args: Parameters<ToolRegistry['resolveDetailed']>) {
        if (args[1]?.scope.includes('::sub::') === true && measuring.getStore() !== true) throw new Error('REGISTRY-SECRET');
        return resolve.apply(this, args);
      });
      if (phase === 'failed_undo_failed') vi.spyOn(Session.prototype, 'undoChildStart').mockRejectedValue(new Error('STORE-SECRET'));

      await second(r, () => {});
      vi.restoreAllMocks();
      expect(own).not.toBeNull();

      const after: Side = {
        inside: cells(await read(r, own!), prior, null, start),
        outside: { ...cells(await read(r, r.child()), prior, null, start), approval: await approve(r.child(), parked) },
      };
      expect({ during, after }).toEqual({ during: EXPECTED[`${phase}_during`], after: EXPECTED[phase] });
    });
  }
});

/*
 * Round 13 (BRIEF-R13 items 1, 3 and 6): the matrix's new scenarios, phase
 * and checks, permanent. Every failure scenario before this round failed
 * BEFORE the task row; these fail after it, are cancelled, repeat call ids,
 * and keep running after a completed start.
 *
 * Tokens as above, plus:
 *   LATER  the current run is a turn that ran after the start (completes+)
 * Actions (outside, on a fresh Session): approval / approval1 (recordApproval
 * on the first / second parked approval), host_turn (a new message under a
 * context the host supplies), second_start (a later delegation:
 * `completed+wrote` is a start that ran and wrote its records).
 *
 * Not a column in TS: `cancel S`. A run here is cancelled through the
 * ledger of the context it runs under (RunContext.ledger.cancel), which
 * reads and writes no child state and is not addressed by run id; TS has no
 * durable tree cancel (G-285). The `cancelled` scenario covers it: a cancel
 * inside the start's run, before its first model step.
 */

/** A start's records once its task row is written and its run has ended, for every reader. */
const RECORDED: Cells = {
  mode: 'NEW', binding: 'BOUND', refs: 'NEW', depth: 'NEW', ceiling: 'NEW', run: 'NEW', offer: 'NEW',
  provisional: 'no', plans: 'yes', plan_run: 'NONE', paused: 'no', busy: 'no', resumed_tree: 'ABSENT', policy: 'NEW',
};
type R13Side = { inside?: Record<string, string>; outside: Record<string, string> };

export const EXPECTED_R13: Record<string, Record<string, R13Side>> = {
  // Phase at RunFinished(S), inside a run.finished listener: the start
  // stands, its run is completed and its holds are already released
  // (round 13: released BEFORE run.finished, as before run.failed).
  completes: {
    at_run_finished: { inside: { ...RECORDED, decision_same: ENDED }, outside: { ...RECORDED, approval: ENDED } },
    after: { outside: { ...RECORDED, approval: ENDED, second_start: 'completed+wrote' } },
  },
  // A completed start followed by more turns on the child: nothing about the
  // start is undone by them, or by any cleanup on the success path (item 5).
  'completes+': {
    after_more: { outside: { ...RECORDED, run: 'LATER', approval: ENDED, host_turn: 'ran', second_start: 'completed+wrote' } },
  },
  // The child's model request fails AFTER the task row: the start stands
  // (it is never undone once its task row exists), measured during the
  // failure (run.failed) and after it.
  fails_after_task: {
    failed_kept_during: { inside: { ...RECORDED, decision_same: ENDED }, outside: { ...RECORDED, approval: ENDED } },
    failed_kept: { outside: { ...RECORDED, approval: ENDED, host_turn: 'ran', second_start: 'completed+wrote' } },
  },
  // Cancelled inside its run, before its first model step: the task row is
  // still recorded (a stop never refuses a committed start), it takes no
  // model step, and its parked call is ended. No durable tree cancel in TS
  // (G-285): a later host turn under its own context runs.
  cancelled: {
    after: { outside: { ...RECORDED, approval: ENDED, host_turn: 'ran' } },
  },
  // Two gated calls under one id (G-392 composed with the start): both
  // approvals end, in call order, in one mark.
  repeated_ids: {
    after: { outside: { ...RECORDED, approval: ENDED, approval1: ENDED } },
  },
  // Round 13 item 1 (TS F1): a gated call and an UNGATED call share an id;
  // the ungated one ran; a person decided the gated one while it was
  // answerable. Expected: ended / not run / not offered again.
  mixed_repeated: {
    after: { outside: { ...RECORDED, approval: ENDED } },
  },
};

function cells13(values: Values, prior: Values, next: Values, start: string): Record<string, string> {
  const out: Record<string, string> = cells(values, prior, next, start);
  if (out.run?.startsWith('?') === true && typeof values.run === 'string') out.run = 'LATER';
  return out;
}

async function hostTurn(r: Rig, session: Session): Promise<string> {
  return r.runtime.send(session, 'More work.', undefined, RunContext.root(`host_${Math.random()}`, new RunBudget(8))).then(() => 'ran', code);
}

/** Capture the start's run id and its own Session object from commit (b). */
function captureStart(): { start: () => string | null; own: () => Session | null } {
  let start: string | null = null;
  let own: Session | null = null;
  const original = Session.prototype.rememberChildStart;
  vi.spyOn(Session.prototype, 'rememberChildStart').mockImplementation(async function (this: Session, ...args: Parameters<Session['rememberChildStart']>) {
    const result = await original.apply(this, args);
    if (start === null && measuring.getStore() !== true) { own = this; start = args[3].runId; }
    return result;
  });

  return { start: () => start, own: () => own };
}

/** The start under test from a parent turn under `context`, without asserting the parent's answer. */
async function secondUnder(r: Rig, child: () => void, context?: RunContext): Promise<void> {
  await r.session().usingMode('short');
  r.calls([['a', { task: 'Second task' }]]);
  child();
  r.text('Parent done.');
  await r.runtime.send(r.session(), 'Delegate', undefined, context);
}

/** The raw row the start's task row was written with (the row before the user row 'Second task'). */
async function endedRowOf(r: Rig): Promise<JsonObject | undefined> {
  const rows = (await r.child().thread().messages()).map((entry) => entry.message);
  const index = rows.findIndex((row) => row.type === 'user' && row.content === 'Second task');
  return index > 0 ? rows[index - 1] : undefined;
}

type Listened = { inside: Values; outside: Values; decision: string; approval: string };

/** Measure inside and outside from a listener of `type` for the start's run. */
function measureAt(r: Rig, type: 'run.finished' | 'run.failed', started: () => string | null, own: () => Session | null, parked: string): () => Promise<Listened> | null {
  let measured: Promise<Listened> | null = null;
  r.events.listen((event: HarnessEvent) => {
    if (event.type !== type || event.runId !== started()) return;
    measured = measuring.run(true, async () => ({
      inside: await read(r, own()!), outside: await read(r, r.child()),
      decision: await approve(own()!, parked), approval: await approve(r.child(), parked),
    }));
  });

  return () => measured;
}

describe('G-387 round 13: the matrix, extended (item 6) and the mixed repeated row (item 1)', () => {
  it('completes: at RunFinished(S) and after; and the RAW ended-mark payload, byte for byte', async () => {
    const { r, parked, prior } = await prepared();
    const callId = ((await r.child().thread().messages()).map((entry) => entry.message)
      .find((row) => row.type === 'assistant' && Array.isArray(row.tool_calls) && (row.tool_calls as JsonObject[]).length > 0)!.tool_calls as JsonObject[])[0]!.id as string;
    const s = captureStart();
    const at = measureAt(r, 'run.finished', s.start, s.own, parked);

    await second(r, () => r.text('Child done.'));
    vi.restoreAllMocks();
    const next = await read(r, r.child());
    expect(at()).not.toBeNull();
    const got = (await at())!;

    const measured = {
      at_run_finished: {
        inside: { ...cells13(got.inside, prior, next, s.start()!), decision_same: got.decision },
        outside: { ...cells13(got.outside, prior, next, s.start()!), approval: got.approval },
      },
      after: { outside: { ...cells13(await read(r, r.child()), prior, next, s.start()!), approval: await approve(r.child(), parked), second_start: await secondStart(r) } },
    };
    expect(measured).toEqual(EXPECTED_R13.completes);

    // The RAW stored mark: the exact bytes, not only its reading.
    const expected = '{"type":"tool_result","tool_results":[{"tool_call_id":' + JSON.stringify(callId)
      + ',"tool_name":"echo","args":{"n":1},"result":"Not run: the delegation this call was proposed under ended.","tool_call_result_id":null,"artifacts":[]}],'
      + '"tool_approval_responses":[],"harness_ended_by_start":{"approval_ids":[' + JSON.stringify(parked) + ']}}';
    expect(JSON.stringify(await endedRowOf(r))).toBe(expected);
  });

  it('completes+: a completed start followed by more turns', async () => {
    const { r, parked, prior } = await prepared();
    const s = captureStart();
    await second(r, () => r.text('Child done.'));
    vi.restoreAllMocks();
    const next = await read(r, r.child());
    const host_turn = await hostTurn(r, r.child());
    const measured = { after_more: { outside: { ...cells13(await read(r, r.child()), prior, next, s.start()!), approval: await approve(r.child(), parked), host_turn, second_start: await secondStart(r) } } };
    expect(measured).toEqual(EXPECTED_R13['completes+']);
  });

  it('fails AFTER the task row: the child model fails; measured during the failure (run.failed) and after', async () => {
    const { r, parked, prior } = await prepared();
    r.failChild = true;
    const s = captureStart();
    const during = measureAt(r, 'run.failed', s.start, s.own, parked);
    await secondUnder(r, () => {});
    vi.restoreAllMocks();
    r.failChild = false;
    const next = await read(r, r.child());
    expect((await r.child().run())?.status).toBe('failed');
    expect((await endedRowOf(r))?.harness_ended_by_start).toEqual({ approval_ids: [parked] });
    expect(during()).not.toBeNull();
    const got = (await during())!;

    const after = { ...cells13(await read(r, r.child()), prior, next, s.start()!), approval: await approve(r.child(), parked) };
    const measured = {
      failed_kept_during: {
        inside: { ...cells13(got.inside, prior, next, s.start()!), decision_same: got.decision },
        outside: { ...cells13(got.outside, prior, next, s.start()!), approval: got.approval },
      },
      failed_kept: { outside: { ...after, host_turn: await hostTurn(r, r.child()), second_start: await secondStart(r) } },
    };
    expect(measured).toEqual(EXPECTED_R13.fails_after_task);
  });

  it("cancelled: a cancel inside the start's run, before its first model step", async () => {
    const { r, parked, prior } = await prepared();
    const before = r.childRequests();
    const s = captureStart();
    // The ledger the start runs on is its parent turn's (the runner's context
    // is nested within it): cancelled from the start's run.started, which is
    // emitted inside the parent's tool call.
    r.events.listen((event: HarnessEvent) => {
      if (event.type === 'run.started' && event.runId === s.start()) activeTurn(r.session().key())!.run.ledger.cancel('SECRET operator note');
    });
    await secondUnder(r, () => r.text('Child done.'));
    vi.restoreAllMocks();
    const next = await read(r, r.child());
    // No model step, and the task row is recorded.
    expect(r.childRequests()).toBe(before);
    const rows = (await r.child().thread().messages()).map((entry) => entry.message);
    expect(rows.some((row) => row.type === 'user' && row.content === 'Second task')).toBe(true);
    expect(JSON.stringify(rows)).not.toContain('SECRET');

    const measured = { after: { outside: { ...cells13(await read(r, r.child()), prior, next, s.start()!), approval: await approve(r.child(), parked), host_turn: await hostTurn(r, r.child()) } } };
    expect(measured).toEqual(EXPECTED_R13.cancelled);
  });

  it('repeated ids: two gated calls under one id; the start ends both, in call order, in one mark', async () => {
    const { r, prior } = await prepared([['echo', { n: 1 }, 'dup'], ['echo', { n: 2 }, 'dup']]);
    const [a0, a1] = await r.approvals();
    const s = captureStart();
    await second(r, () => r.text('Child done.'));
    vi.restoreAllMocks();
    const next = await read(r, r.child());

    const measured = { after: { outside: { ...cells13(await read(r, r.child()), prior, next, s.start()!), approval: await approve(r.child(), a0!), approval1: await approve(r.child(), a1!) } } };
    expect(measured).toEqual(EXPECTED_R13.repeated_ids);
    expect((await endedRowOf(r))?.harness_ended_by_start).toEqual({ approval_ids: [a0, a1] });
    expect(r.echoed).toBe(0);
  });

  it('mixed repeated: a gated and an ungated call share an id; the person decided while answerable; then a start: ended / not run / not offered again', async () => {
    const { r, parked, prior } = await prepared([['echo', { n: 1 }, 'w'], ['probe', {}, 'w']], true);
    const s = captureStart();
    await second(r, () => r.text('Child done.'));
    vi.restoreAllMocks();
    const next = await read(r, r.child());
    const rows = (await r.child().thread().messages()).map((entry) => entry.message);

    const measured = { after: { outside: { ...cells13(await read(r, r.child()), prior, next, s.start()!), approval: await approve(r.child(), parked) } } };
    expect(measured).toEqual(EXPECTED_R13.mixed_repeated);
    // Ended, never approved work, not run, not offered again.
    expect([...endedByStart(rows)]).toContain(parked);
    expect(approvalDecisions(rows).has(parked)).toBe(false);
    expect(pendingApprovalCalls(rows)).toEqual([]);
    expect(r.echoed).toBe(0);
    expect((await endedRowOf(r))?.harness_ended_by_start).toEqual({ approval_ids: [parked] });
  });
});
