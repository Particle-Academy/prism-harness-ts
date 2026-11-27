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
  recordDecisions,
  type Durability,
  type HarnessEvent,
  type JsonObject,
  type LlmRequest,
  type LlmResponse,
  type ModeConfig,
} from '../src/index.js';
import { endedByStart, REPEATED_CALL_ID, REPEATED_CALL_ID_RESULT } from '../src/thread-rows.js';
import { approvalDecisions, pendingApprovalCalls } from '../src/plans/thread.js';

// G-387 round 13: the round-12 review findings, at the public seams (the
// subagent TOOL inside a parent turn, SubagentRunner.run, send,
// recordApproval / recordDecisions / decide / streamDecision, event
// listeners). No network; every store is in memory.

const network = vi.fn(() => { throw new Error('A test tried to use the network.'); });
beforeEach(() => { vi.stubGlobal('fetch', network); });
afterEach(() => { expect(network).not.toHaveBeenCalled(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });

/** A memory store whose reads can be made to throw, for the child's keys only. */
class Store extends MemorySessionStore {
  readonly keys = new Set<string>();
  /** Reads of a child key that throw: a count, or `always` until disarmed. */
  failReads: number | 'always' = 0;
  failed = 0;
  /** Writes of a child key that land, and then throw (a lost acknowledgement). */
  failAfterWrite = 0;

  constructor(private readonly kind: Durability) {
    super();
  }

  override durability(): Durability {
    return this.kind;
  }

  override async get(key: string): Promise<JsonObject | null> {
    if (key.includes('::sub::') && (this.failReads === 'always' || this.failReads > 0)) {
      if (this.failReads !== 'always') this.failReads -= 1;
      this.failed += 1;
      throw new Error('STORE-SECRET read failed');
    }
    return super.get(key);
  }

  override async put(key: string, payload: JsonObject, ttlSeconds: number | null = null): Promise<void> {
    this.keys.add(key);
    await super.put(key, payload, ttlSeconds);
    if (key.includes('::sub::') && this.failAfterWrite > 0) {
      this.failAfterWrite -= 1;
      throw new Error('STORE-SECRET acknowledgement lost');
    }
  }
}

type Call = [name: string, args: unknown, id?: string];

interface RigOptions {
  plans?: boolean;
  /** What the model call does for the child's requests after the first `n` (default: the scripted queue). */
  childFailsAfter?: number;
}

class Rig {
  readonly queue: LlmResponse[] = [];
  readonly requests: LlmRequest[] = [];
  readonly durable = new Store('durable');
  readonly ephemeral = new Store('volatile');
  readonly events = new HarnessEvents();
  readonly registry: ToolRegistry;
  readonly modes: ModeRegistry;
  readonly plans: Plans | null;
  readonly runtime: AgentRuntime;
  readonly harness: PrismHarness;
  readonly echoed: number[] = [];
  free = 0;
  #call = 0;

  constructor(options: RigOptions = {}) {
    this.modes = new ModeRegistry({
      default: 'lead',
      modes: {
        lead: { system_prompt: 'Lead.', tools: ['echo', 'a'], max_steps: 8, requires_approval: ['echo'], subagents: { a: { mode: 'work', max_steps: 8 } } } as ModeConfig,
        work: { system_prompt: 'Work.', tools: ['echo', 'free'], max_steps: 8, requires_approval: ['echo'], ...(options.plans === true ? { plans: true } : {}) } as ModeConfig,
      },
    });
    this.registry = new ToolRegistry()
      .register({ name: 'echo', handle: (args: JsonObject) => { this.echoed.push(Number(args.n)); return 'ok'; } })
      .register({ name: 'free', handle: () => { this.free += 1; return 'free ok'; } });
    this.plans = options.plans === true ? new Plans({ modes: this.modes, config: { enabled: true }, events: this.events }) : null;
    let childCalls = 0;
    this.runtime = new AgentRuntime({
      modes: this.modes, tools: this.registry, events: this.events,
      client: async (request) => {
        this.requests.push(request);
        if (request.systemPrompt.startsWith('Work.') && options.childFailsAfter !== undefined && ++childCalls > options.childFailsAfter) throw new Error('MODEL-SECRET');
        return this.queue.shift() ?? { text: 'Done.', finishReason: 'stop' };
      },
    });
    const durable = this.durable, ephemeral = this.ephemeral;
    this.harness = new PrismHarness({
      drivers: { ephemeral: () => ephemeral, durable: () => durable },
      stores: { ephemeral: 'ephemeral', durable: 'durable' },
      runtime: this.runtime,
      ...(this.plans === null ? {} : { plans: this.plans }),
    });
    this.registry.register(new SubagentRunner(this.harness, this.runtime)
      .tool(this.modes.resolve('lead').subagents.a!, this.session(), RunContext.root('host', new RunBudget(100)), 'host_run'));
  }

  session(scope = 'chat'): Session {
    return this.harness.for({ type: 'User', id: 1 }).session(scope);
  }

  child(): Session {
    return this.session('chat::sub::a');
  }

  /** One model step with `calls`, each under its own id or the one given (a provider may repeat ids). */
  calls(calls: Call[]): void {
    this.queue.push({ text: '', finishReason: 'tool_calls', toolCalls: calls.map(([name, args, id]) => ({ id: id ?? `call_${++this.#call}`, name, arguments: args as JsonObject })) });
  }

  text(text = 'Done.'): void {
    this.queue.push({ text, finishReason: 'stop' });
  }

  async run(task: string): Promise<{ outcome: string }> {
    return new SubagentRunner(this.harness, this.runtime).run(this.modes.resolve('lead').subagents.a!, this.session(), RunContext.root('root', new RunBudget(8)), 'root', task);
  }

  childRequests(): number {
    return this.requests.filter((request) => request.systemPrompt.startsWith('Work.')).length;
  }

  async rows(): Promise<JsonObject[]> {
    return (await this.child().thread().messages()).map((entry) => entry.message);
  }

  /** The child parked on one step of `calls` (a first delegation). Its approval ids. */
  async parked(calls: Call[]): Promise<string[]> {
    await this.session().usingMode('lead');
    this.calls(calls);
    await this.run('First');
    expect(approvalsIn(await this.rows()).length).toBeGreaterThan(0);

    return approvalsIn(await this.rows());
  }

  /** Nobody can answer the child any more, so a re-delegation ends its calls. */
  async unanswerable(): Promise<void> {
    const key = `${this.child().key()}:durable`;
    await this.durable.put(key, { ...((await this.durable.get(key)) ?? {}), child_depth: 1_000 });
  }

  /** The plan step a child plan holds on, the step's calls being `calls`. Its approval ids. */
  async planHeldOn(calls: Call[]): Promise<string[]> {
    await this.session().usingMode('lead');
    this.calls([['plan_draft', { goal: 'Do the work', steps: [{ title: 'Step 1', instruction: 'Echo once.', owner: 'agent' }] }]]);
    await this.run('Plan it');
    const draft = approvalsIn(await this.rows()).pop()!;
    await recordApproval(this.child(), draft, true);
    this.text('Plan approved.');
    await this.runtime.send(this.child(), '');
    this.calls(calls);
    await (await this.child().plan()).advance();
    const step = await this.step();
    expect(step.state).toBe('held');

    return step.approvalIds as string[];
  }

  async step(): Promise<Record<string, any>> {
    return (((await (await this.child().plan()).current())?.toJSON() ?? {}) as Record<string, any>).steps[0];
  }
}

/** A start through the runner directly, or through the subagent tool in a parent turn; the outcome word. */
async function start(r: Rig, via: 'runner' | 'tool', task: string, child: () => void = () => r.text('Child done.')): Promise<string> {
  if (via === 'runner') {
    child();
    return (await r.run(task)).outcome;
  }
  r.calls([['a', { task }]]);
  child();
  r.text('Parent done.');
  expect((await r.runtime.send(r.session(), 'Delegate', undefined, RunContext.root('root', new RunBudget(8)))).text).toBe('Parent done.');
  const rows = (await r.session().thread().messages()).map((entry) => entry.message);
  const framed = String(rows.filter((row) => row.type === 'tool_result').flatMap((row) => row.tool_results as JsonObject[]).pop()?.result ?? '');

  return /did not run: it (\w+)/.exec(framed)?.[1] ?? 'completed';
}

function approvalsIn(rows: JsonObject[]): string[] {
  return rows.flatMap((row) => Array.isArray(row.tool_approval_requests) ? (row.tool_approval_requests as JsonObject[]).map((request) => request.approval_id as string) : []);
}

/** Every decision as stored, by approval id (the first one stands). */
function storedDecisions(rows: JsonObject[]): Map<string, JsonObject> {
  const out = new Map<string, JsonObject>();
  for (const row of rows) {
    if (row.type !== 'tool_result') continue;
    for (const answer of (row.tool_approval_responses ?? []) as JsonObject[]) if (!out.has(answer.approval_id as string)) out.set(answer.approval_id as string, answer);
  }
  return out;
}

/** The approval reads ended: the plan's reading never has it as approved work, no decision is accepted, nothing runs. */
async function expectEnded(r: Rig, id: string): Promise<void> {
  const rows = await r.rows();
  expect([...endedByStart(rows)]).toContain(id);
  expect(approvalDecisions(rows).has(id)).toBe(false);
  await expect(recordApproval(r.child(), id, true)).rejects.toMatchObject({ code: 'approval_not_pending' });
  // A host resume on the child acts on nothing of it.
  r.text('Resumed.');
  await r.runtime.send(r.child(), '');
  expect(r.echoed).toEqual([]);
}

// ---------------------------------------------------------------------------
// Item 1 (TS F1): an approval reads ENDED when ANY result row after its own
// request row, before the next assistant row, carries the start's mark
// naming it. Not only the first row carrying its call id: in a step whose
// calls share an id, an earlier UNMARKED row (an ungated call, a call with
// unusable arguments) carries that id too.
// ---------------------------------------------------------------------------

describe('item 1: an approval a start ended reads ended however its step repeats ids', () => {
  for (const via of ['runner', 'tool'] as const) {
    it(`P1 through the ${via}: gated echo and ungated free share id w; free ran first; a person approved echo; a start ends it: ended, not run, not offered again`, async () => {
      const r = new Rig();
      const [echo] = await r.parked([['echo', { n: 1 }, 'w'], ['free', {}, 'w']]);
      expect(r.free).toBe(1);
      await recordApproval(r.child(), echo!, true);
      await r.unanswerable();

      expect(await start(r, via, 'Start over')).toBe('completed');

      const rows = await r.rows();
      const marks = rows.filter((row) => Object.hasOwn(row, 'harness_ended_by_start')).map((row) => row.harness_ended_by_start);
      expect(marks).toEqual([{ approval_ids: [echo] }]);
      await expectEnded(r, echo!);
    });
  }

  it("P1b: two echo calls share the id echo (a Gemini-style id); one's arguments are unusable and it got an immediate result; the other was approved; a start ends it", async () => {
    const r = new Rig();
    const [echo] = await r.parked([['echo', [1, 2], 'echo'], ['echo', { n: 1 }, 'echo']]);
    expect(echo).toBeDefined();
    await recordApproval(r.child(), echo!, true);
    await r.unanswerable();

    expect(await start(r, 'runner', 'Start over')).toBe('completed');

    await expectEnded(r, echo!);
  });

  it('P1 with plans on: the plan step never records the ended approval as approved work, and nothing runs', async () => {
    const r = new Rig({ plans: true });
    const [echo] = await r.planHeldOn([['echo', { n: 1 }, 'w'], ['free', {}, 'w']]);
    expect(echo).toBeDefined();
    await recordApproval(r.child(), echo!, true);
    await r.unanswerable();
    r.queue.length = 0;

    expect(await start(r, 'runner', 'Start over')).toBe('completed');

    const step = await r.step();
    expect(step.approvedIds).not.toContain(echo);
    expect(r.echoed).toEqual([]);
    expect(approvalDecisions(await r.rows()).has(echo!)).toBe(false);
  });

  // The read rule itself, on rows as a harness stores them.
  const asked: JsonObject = { type: 'assistant', content: '', tool_calls: [{ id: 'w', name: 'echo', arguments: { n: 1 } }, { id: 'w', name: 'free', arguments: {} }], tool_approval_requests: [{ approval_id: 'apr_1', tool_call_id: 'w' }] };
  const freeRan: JsonObject = { type: 'tool_result', tool_results: [{ tool_call_id: 'w', tool_name: 'free', args: {}, result: 'free ok', tool_call_result_id: null, artifacts: [] }], tool_approval_responses: [] };
  const decided: JsonObject = { type: 'tool_result', tool_results: [], tool_approval_responses: [{ approval_id: 'apr_1', approved: true, reason: null }] };
  const ended = (ids: string[]): JsonObject => ({ type: 'tool_result', tool_results: [{ tool_call_id: 'w', tool_name: 'echo', args: { n: 1 }, result: 'Not run: the delegation this call was proposed under ended.', tool_call_result_id: null, artifacts: [] }], tool_approval_responses: [], harness_ended_by_start: { approval_ids: ids } });
  const user: JsonObject = { type: 'user', content: 'Start over' };

  it('read rule: an earlier unmarked row carrying the id does not hide a later mark naming the approval', () => {
    const rows = [asked, freeRan, decided, ended(['apr_1']), user];
    expect([...endedByStart(rows)]).toEqual(['apr_1']);
    expect(approvalDecisions(rows).has('apr_1')).toBe(false);
  });

  it('read rule: a mark naming another approval, a mark after the next assistant row, or a mark before the request ends nothing', () => {
    expect([...endedByStart([asked, freeRan, decided, ended(['apr_9']), user])]).toEqual([]);
    const later: JsonObject = { type: 'assistant', content: 'Hm.', tool_calls: [], tool_approval_requests: [] };
    expect([...endedByStart([asked, freeRan, later, ended(['apr_1']), user])]).toEqual([]);
    expect([...endedByStart([ended(['apr_1']), asked, freeRan, decided, user])]).toEqual([]);
    // Not the shape: nothing ends.
    for (const mark of [true, ['apr_1'], { approval_ids: 'apr_1' }, { approval_ids: [1] }, {}]) {
      expect([...endedByStart([asked, freeRan, { ...ended([]), harness_ended_by_start: mark as JsonObject }, user])]).toEqual([]);
    }
  });

  it('read rule: an approval no mark names keeps its reading (approved stays approved)', () => {
    const rows = [asked, freeRan, decided];
    expect([...endedByStart(rows)]).toEqual([]);
    expect(approvalDecisions(rows).get('apr_1')).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Item 3 (Py F1, checked in TS): a gated call that shares its id with an
// UNGATED call. Every decision recorded on a step whose call ids repeat is a
// DENIAL with the G-392 text (as the PHP reference records it), and results
// are paired with calls by position, never by id.
// ---------------------------------------------------------------------------

describe('item 3: decisions on a repeated-id step are recorded as denials; results pair by position', () => {
  for (const id of ['dup', ''] as const) {
    for (const order of ['gated first', 'ungated first'] as const) {
      const steps = (): Call[] => order === 'gated first'
        ? [['echo', { n: 1 }, id], ['free', { n: 9 }, id]]
        : [['free', { n: 9 }, id], ['echo', { n: 1 }, id]];

      it(`P3 (id ${JSON.stringify(id)}, ${order}): the person's approval is recorded as the repeated-id denial; the plan's pending call is echo's; a start then ends it`, async () => {
        const r = new Rig();
        const [echo] = await r.parked(steps());
        expect(r.free).toBe(1);
        // The plan's view of the pending call names the call the approval was asked for.
        expect(pendingApprovalCalls(await r.rows()).map((call) => call.name)).toEqual(['echo']);

        await recordApproval(r.child(), echo!, true);

        const stored = storedDecisions(await r.rows()).get(echo!);
        expect(stored).toEqual({ approval_id: echo, approved: false, reason: REPEATED_CALL_ID_RESULT });
        expect(approvalDecisions(await r.rows()).get(echo!)).toBe(false);

        await r.unanswerable();
        expect(await start(r, 'runner', 'Start over')).toBe('completed');
        const rows = await r.rows();
        expect(rows.filter((row) => Object.hasOwn(row, 'harness_ended_by_start')).map((row) => row.harness_ended_by_start)).toEqual([{ approval_ids: [echo] }]);
        await expectEnded(r, echo!);
      });

      it(`P3 (id ${JSON.stringify(id)}, ${order}), resumed instead: the gated call gets the repeated-id denial and never runs`, async () => {
        const r = new Rig();
        const [echo] = await r.parked(steps());
        await recordApproval(r.child(), echo!, true);
        r.text('Resumed.');
        await r.runtime.send(r.child(), '');

        expect(r.echoed).toEqual([]);
        expect(r.free).toBe(1);
        const entries = (await r.rows()).filter((row) => row.type === 'tool_result').flatMap((row) => row.tool_results as JsonObject[]);
        const denied = entries.filter((entry) => entry[ 'not_run' ] === REPEATED_CALL_ID);
        expect(denied.map((entry) => [entry.tool_name, entry.approval_id])).toEqual([['echo', echo]]);
        // Denied for its repeated id: never approved work.
        expect(approvalDecisions(await r.rows()).get(echo!)).not.toBe(true);
        await expect(recordApproval(r.child(), echo!, true)).rejects.toMatchObject({ code: 'approval_not_pending' });
      });
    }
  }

  it('every decision path records the denial: recordDecisions, decide and streamDecision (its decision step reports the decision as given)', async () => {
    for (const path of ['recordDecisions', 'decide', 'streamDecision'] as const) {
      const r = new Rig();
      const ids = await r.parked([['echo', { n: 1 }, 'dup'], ['echo', { n: 2 }, 'dup']]);
      expect(ids).toHaveLength(2);
      const decisions = ids.map((approvalId) => ({ approvalId, approved: true, reason: 'Fine by me.' }));
      if (path === 'recordDecisions') await recordDecisions(r.child(), decisions);
      if (path === 'decide') for (const decision of decisions) await r.runtime.decide(r.child(), decision.approvalId, true, 'Fine by me.');
      if (path === 'streamDecision') {
        const steps: JsonObject[] = [];
        r.text('Resumed.');
        for await (const step of r.runtime.streamDecision(r.child(), decisions)) steps.push(step as unknown as JsonObject);
        const recorded = steps.filter((step) => step.type === 'decision_recorded');
        // The step reports the decision as given, as the PHP reference streams it; the row holds the denial.
        expect(recorded.map((step) => [step.verdict, step.reason]), path).toEqual(ids.map(() => ['approved', 'Fine by me.']));
      }
      const stored = storedDecisions(await r.rows());
      for (const approvalId of ids) expect(stored.get(approvalId), path).toEqual({ approval_id: approvalId, approved: false, reason: REPEATED_CALL_ID_RESULT });
      expect(r.echoed, path).toEqual([]);
    }
  });

  it('control: a step with distinct ids records the decision as given', async () => {
    const r = new Rig();
    const [echo] = await r.parked([['echo', { n: 1 }], ['free', {}]]);
    await recordApproval(r.child(), echo!, true, 'Yes.');
    expect(storedDecisions(await r.rows()).get(echo!)).toEqual({ approval_id: echo, approved: true, reason: 'Yes.' });
    r.text('Resumed.');
    await r.runtime.send(r.child(), '');
    expect(r.echoed).toEqual([1]);
  });
});

// ---------------------------------------------------------------------------
// Item 5 (PHP run15 Low, checked in TS): the success path NEVER undoes a
// start. A store read that throws during the cleanup of a completed start
// only ends the start's own view; its records stand, and the next host turn
// on the child runs.
// ---------------------------------------------------------------------------

describe('item 5: a completed start is never undone by its cleanup', () => {
  for (const how of ['once', 'always'] as const) {
    for (const via of ['runner', 'tool'] as const) {
      it(`ZB through the ${via}: ${how === 'once' ? 'one child read throws' : 'every child read throws'} once its run is completed: binding and mode kept, the next host turn runs`, async () => {
        const r = new Rig();
        await r.session().usingMode('lead');
        vi.spyOn(process, 'emitWarning').mockImplementation((() => {}) as typeof process.emitWarning);
        // Armed the moment the start's run is written completed: every read
        // after it (the hold release, the cleanup) is the success path's.
        const complete = Session.prototype.completeRun;
        let armed = true;
        vi.spyOn(Session.prototype, 'completeRun').mockImplementation(async function (this: Session, ...args: Parameters<Session['completeRun']>) {
          const result = await complete.apply(this, args);
          if (armed && this.scope.includes('::sub::')) {
            r.durable.failReads = how === 'once' ? 1 : 'always';
            r.ephemeral.failReads = how === 'once' ? 1 : 'always';
          }
          return result;
        });

        await start(r, via, 'Work');
        armed = false;
        r.durable.failReads = 0;
        r.ephemeral.failReads = 0;
        expect(r.durable.failed + r.ephemeral.failed).toBeGreaterThan(0);

        const child = r.child();
        expect(await child.childStartUnconfirmed()).toBe(false);
        expect(await child.childBinding()).not.toBeNull();
        expect(await child.mode()).toBe('work');
        expect(await child.childDepth()).toBe(1);
        const before = r.childRequests();
        // A host turn on the child, under a context the host supplies (the
        // root tree it was started in has finished with its root's turn).
        r.text('Next done.');
        const next = r.runtime.send(child, 'Next', undefined, RunContext.root('host2', new RunBudget(8)));
        if (how === 'once') {
          // One failed read: the settle is retried, nothing is over-charged.
          expect((await next).text).toBe('Next done.');
        } else {
          // The store failed for the whole cleanup: the start's count hold
          // could not be settled, so the delegation is over-charged (fail
          // closed: a budget refusal, never a lost binding), and a new
          // delegation, once that hold has expired, starts it again.
          await expect(next).rejects.toMatchObject({ code: 'run_not_permitted' });
          r.queue.length = 0;
          vi.spyOn(Date, 'now').mockReturnValue(Date.now() + 301_000);
          expect(await start(r, via, 'Again')).toBe('completed');
          r.text('Next done.');
          expect((await r.runtime.send(child, 'Next', undefined, RunContext.root('host3', new RunBudget(8)))).text).toBe('Next done.');
          return;
        }
        expect(r.childRequests()).toBe(before + 1);
      });
    }
  }

  it('ZB, fails AFTER the task row: the child model fails, and every child read in the cleanup throws: the start stands', async () => {
    const r = new Rig({ childFailsAfter: 0 });
    await r.session().usingMode('lead');
    vi.spyOn(process, 'emitWarning').mockImplementation((() => {}) as typeof process.emitWarning);
    // Armed once the run has failed after its task row: every child read in its cleanup throws.
    const original = Session.prototype.failRun;
    vi.spyOn(Session.prototype, 'failRun').mockImplementation(async function (this: Session, ...args: Parameters<Session['failRun']>) {
      if (this.scope.includes('::sub::')) { r.durable.failReads = 'always'; r.ephemeral.failReads = 'always'; }
      return original.apply(this, args);
    });

    expect(await start(r, 'runner', 'Work', () => {})).toBe('failed');
    r.durable.failReads = 0;
    r.ephemeral.failReads = 0;
    vi.restoreAllMocks();

    const child = r.child();
    expect(await child.childStartUnconfirmed()).toBe(false);
    expect(await child.childBinding()).not.toBeNull();
    expect(await child.mode()).toBe('work');
    expect(await child.childDepth()).toBe(1);
    // Its count hold could not be settled: it only over-charges, and a new
    // delegation once it has expired completes.
    vi.spyOn(Date, 'now').mockReturnValue(Date.now() + 301_000);
    expect(await start(r, 'runner', 'Again')).toBe('failed');
  });
});

// ---------------------------------------------------------------------------
// Item 4 (Py I2/I3, checked in TS) and the TS review's I-1: the start's own
// view ends when its run ends, whatever ended it; and the holds a run took
// are released BEFORE run.finished, as they are before run.failed.
// ---------------------------------------------------------------------------

describe("item 4: the start's own view ends with its run; holds release before run.finished", () => {
  it('P5: after a start whose undo AND failRun both failed, the object that committed it reads its records as absent and cannot act on them', async () => {
    const r = new Rig();
    await r.parked([['echo', { n: 1 }]]);
    await r.unanswerable();
    let own: Session | null = null;
    const remember = Session.prototype.rememberChildStart;
    vi.spyOn(Session.prototype, 'rememberChildStart').mockImplementation(async function (this: Session, ...args: Parameters<Session['rememberChildStart']>) {
      own = this;
      return remember.apply(this, args);
    });
    vi.spyOn(Session.prototype, 'undoChildStart').mockRejectedValue(new Error('STORE-SECRET'));
    vi.spyOn(Session.prototype, 'failRun').mockRejectedValue(new Error('STORE-SECRET'));
    const resolve = ToolRegistry.prototype.resolveDetailed;
    vi.spyOn(ToolRegistry.prototype, 'resolveDetailed').mockImplementation(function (this: ToolRegistry, ...args: Parameters<ToolRegistry['resolveDetailed']>) {
      if (args[1]?.scope.includes('::sub::') === true) throw new Error('REGISTRY-SECRET');
      return resolve.apply(this, args);
    });

    expect(await start(r, 'runner', 'Start over', () => {})).toBe('failed');
    vi.restoreAllMocks();

    expect(own).not.toBeNull();
    const self = own! as Session;
    expect(await self.childStartUnconfirmed()).toBe(true);
    expect(await self.childBinding()).toBeNull();
    expect(await self.childDepth()).toBe(0);
    await expect(r.runtime.send(self, '')).rejects.toMatchObject({ code: 'child_binding_missing' });
    expect(r.childRequests()).toBe(1);
  });

  for (const via of ['runner', 'tool'] as const) {
    it(`I-1 through the ${via}: at the child's run.finished its delegation hold and its tree hold are already released`, async () => {
      const r = new Rig();
      await r.session().usingMode('lead');
      let at: Promise<boolean> | null = null;
      r.events.listen((event: HarnessEvent) => {
        if (event.type === 'run.finished' && event.sessionKey === r.child().key()) {
          const runId = event.runId;
          // Every hold, read at the moment of the event (a memory store reads in microtasks).
          at = Promise.all([...r.durable.keys].map(async (key) => {
            const value = await r.durable.get(key);
            const holds = key.includes(':run-tree:') ? (value as { holds?: JsonObject } | null)?.holds
              : (value as { child_steps?: { holds?: JsonObject } } | null)?.child_steps?.holds;
            return holds != null && Object.hasOwn(holds, runId);
          })).then((held) => held.some(Boolean));
        }
      });

      expect(await start(r, via, 'Work')).toBe('completed');
      expect(at).not.toBeNull();
      expect(await at!).toBe(false);
    });
  }
});

// ---------------------------------------------------------------------------
// New channels the round-13 fixes could open, each pinned.
// ---------------------------------------------------------------------------

describe('round 13: new channels', () => {
  it('the settle is retried: a write that landed and then threw is not charged twice', async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    vi.spyOn(process, 'emitWarning').mockImplementation((() => {}) as typeof process.emitWarning);
    const settle = Session.prototype.settleChildSteps;
    vi.spyOn(Session.prototype, 'settleChildSteps').mockImplementation(async function (this: Session, ...args: Parameters<Session['settleChildSteps']>) {
      r.durable.failAfterWrite = 1;
      vi.mocked(Session.prototype.settleChildSteps).mockImplementation(settle);
      return settle.apply(this, args);
    });

    r.calls([['echo', { n: 1 }]]);
    await r.run('Work');
    vi.restoreAllMocks();

    const record = (await r.durable.get(`${r.child().key()}:durable`)) as { child_steps?: { steps: number; holds: JsonObject } };
    // One model step taken, charged once, and no hold left behind.
    expect(record.child_steps?.steps).toBe(1);
    expect(record.child_steps?.holds).toEqual({});
  });

  it("a start launched from the child's run.finished listener (holds now released before it) completes, and the child is consistent after", async () => {
    const r = new Rig();
    await r.session().usingMode('lead');
    let second: Promise<{ outcome: string }> | null = null;
    let fired = false;
    r.events.listen((event: HarnessEvent) => {
      if (event.type === 'run.finished' && event.sessionKey === r.child().key() && !fired) {
        fired = true;
        r.text('Second done.');
        second = r.run('Second');
      }
    });

    r.text('First done.');
    expect((await r.run('First')).outcome).toBe('completed');
    expect(second).not.toBeNull();
    expect((await second!).outcome).toBe('completed');

    const child = r.child();
    expect(await child.childStartUnconfirmed()).toBe(false);
    expect(await child.childBinding()).not.toBeNull();
    expect(await child.childTurnLive(Math.floor(Date.now() / 1000), 300)).toBe(false);
    const tasks = (await r.rows()).filter((row) => row.type === 'user').map((row) => row.content);
    expect(tasks).toEqual(['First', 'Second']);
  });
});

// ---------------------------------------------------------------------------
// BRIEF-R13 (added): a structured send whose FIRST step stops for approval
// returns the structured response, with its pending approvals and no
// document (`structured: null`), never a throw or a missing field.
// ---------------------------------------------------------------------------

describe('sendStructured that parks on its first step', () => {
  const schema: JsonObject = { type: 'object', properties: { ok: { type: 'boolean' } }, required: ['ok'] };

  it('returns awaiting_approval with the pending approvals and structured null; the resume returns the document', async () => {
    const r = new Rig();
    const root = r.session('structured');
    await root.usingMode('work');
    r.calls([['echo', { n: 1 }]]);
    const parked = await r.runtime.sendStructured(root, 'Go', schema);

    expect(parked.finishReason).toBe('awaiting_approval');
    expect(parked.pendingApprovals.map((pending) => pending.tool)).toEqual(['echo']);
    expect(Object.hasOwn(parked, 'structured')).toBe(true);
    expect(parked.structured).toBeNull();

    await recordApproval(root, parked.pendingApprovals[0]!.id, true);
    r.queue.push({ text: '{"ok":true}', finishReason: 'stop', structured: { ok: true } });
    const done = await r.runtime.sendStructured(root, '', schema);
    expect(done.structured).toEqual({ ok: true });
    expect(r.echoed).toEqual([1]);
  });

  it('a structured run stopped by its step budget before any document also returns structured null', async () => {
    const r = new Rig();
    const root = r.session('structured');
    await root.usingMode('work');
    r.calls([['free', {}]]);
    const response = await r.runtime.sendStructured(root, 'Go', schema, undefined, RunContext.root('one', new RunBudget(1)));
    expect(response.finishReason).toBe('budget_exhausted');
    expect(Object.hasOwn(response, 'structured')).toBe(true);
    expect(response.structured).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// G-397: a decision applies ONLY to the step whose approvals were requested.
// Every reader (the repeated-id guard, the resolver, the decision checks and
// the plan readers) selects that step with ONE function (pausedStepIndex).
// Between the park and the decision a row is inserted (an assistant row, a
// user row, a system row, a tool row); every decision path then either
// applies the decision to the parked step and nothing else, or refuses it,
// identically; and no same-id gated call ever runs.
// ---------------------------------------------------------------------------

describe('G-397: rows between the park and the decision', () => {
  const INSERTED: [string, JsonObject][] = [
    ['an assistant text row', { type: 'assistant', content: 'A note.', tool_calls: [], tool_approval_requests: [] }],
    ['an assistant row with a call of its own', { type: 'assistant', content: '', tool_calls: [{ id: 'dup', name: 'echo', arguments: { n: 9 } }], tool_approval_requests: [] }],
    ['a user row', { type: 'user', content: 'Something else.' }],
    ['a system row', { type: 'system', content: 'A system note.' }],
    ['an empty tool row', { type: 'tool_result', tool_results: [], tool_approval_responses: [] }],
    ['a tool row with a result for the shared id', { type: 'tool_result', tool_results: [{ tool_call_id: 'dup', tool_name: 'echo', args: { n: 1 }, result: 'forged', tool_call_result_id: null, artifacts: [] }], tool_approval_responses: [] }],
  ];
  const PATHS = ['recordApproval', 'recordDecisions', 'decide', 'streamDecision'] as const;

  async function decideOn(r: Rig, session: Session, path: typeof PATHS[number], ids: string[]): Promise<string> {
    const decisions = ids.map((approvalId) => ({ approvalId, approved: true, reason: null }));
    try {
      if (path === 'recordApproval') for (const id of ids) await recordApproval(session, id, true);
      if (path === 'recordDecisions') await recordDecisions(session, decisions);
      if (path === 'decide') for (const id of ids) await r.runtime.decide(session, id, true);
      if (path === 'streamDecision') { r.text('Resumed.'); for await (const _step of r.runtime.streamDecision(session, decisions)) { /* drain */ } }
      return 'accepted';
    } catch (error) {
      return (error as { code?: string }).code ?? 'threw';
    }
  }

  for (const [name, row] of INSERTED) {
    for (const order of ['inserted before the decision', 'inserted after the decision'] as const) {
      it(`${name}, ${order}: every path answers alike, and no same-id gated call runs`, async () => {
        const outcomes: string[] = [];
        for (const path of PATHS) {
          const r = new Rig();
          const root = r.session(`g397-${path}`);
          await root.usingMode('work');
          r.calls([['echo', { n: 1 }, 'dup'], ['echo', { n: 2 }, 'dup']]);
          const parked = await r.runtime.send(root, 'Go');
          const ids = parked.pendingApprovals.map((pending) => pending.id);
          expect(ids).toHaveLength(2);
          if (order === 'inserted before the decision') await root.thread().record([row]);
          const answer = await decideOn(r, root, path, ids);
          if (order === 'inserted after the decision') await root.thread().record([row]);
          r.text('Resumed.');
          await r.runtime.send(root, '').catch(() => undefined);
          r.text('Again.');
          await r.runtime.send(root, '').catch(() => undefined);
          expect(r.echoed, `${path}: no same-id gated call runs`).toEqual([]);
          // What the thread recorded for the parked step: nothing, or denials only.
          const decided = storedDecisions(await r.rows.call({ child: () => root } as unknown as Rig));
          expect([...decided.values()].every((answer) => answer.approved === false), path).toBe(true);
          outcomes.push(answer);
        }
        expect(new Set(outcomes).size, JSON.stringify(outcomes)).toBe(1);
      });
    }
  }

  for (const [name, row] of INSERTED) {
    it(`distinct ids, ${name} inserted before the decision: the decision applies to the parked step only, or is refused, on every path alike`, async () => {
      const outcomes: string[] = [];
      for (const path of PATHS) {
        const r = new Rig();
        const root = r.session(`g397d-${path}`);
        await root.usingMode('work');
        r.calls([['echo', { n: 1 }, 'a'], ['echo', { n: 2 }, 'b']]);
        const parked = await r.runtime.send(root, 'Go');
        const ids = parked.pendingApprovals.map((pending) => pending.id);
        await root.thread().record([row]);
        const answer = await decideOn(r, root, path, [ids[0]!]);
        r.text('Resumed.');
        await r.runtime.send(root, '').catch(() => undefined);
        // Only the approved call of the parked step can run, at most once; never the other, never a call of the inserted row.
        expect(r.echoed.every((n) => n === 1), path).toBe(true);
        expect(r.echoed.length, path).toBeLessThanOrEqual(1);
        outcomes.push(`${answer}:${r.echoed.length}`);
      }
      expect(new Set(outcomes).size, JSON.stringify(outcomes)).toBe(1);
    });
  }
});

// ---------------------------------------------------------------------------
// G-399: results and decisions for the parked step are read from EVERY
// tool_result row after it, up to the next assistant row. A row of another
// kind between them hides no result: an approved call runs exactly once
// however many resumes follow.
// ---------------------------------------------------------------------------

describe('G-399: a row between the results does not hide one', () => {
  const ROWS: [string, JsonObject][] = [
    ['a system row', { type: 'system', content: 'A system note.' }],
    ['an empty tool row', { type: 'tool_result', tool_results: [], tool_approval_responses: [] }],
    ['a row of an unknown kind', { type: 'note', content: 'Host bookkeeping.' }],
  ];
  // Each resume runs under a context with no model step left, so it acts on
  // the recorded decisions and adds NO assistant row: the parked step stays
  // the tail step, and only its results say the call already ran.
  const noStep = (resume: number): RunContext => RunContext.root(`g399_${resume}`, new RunBudget(0));

  for (const [name, row] of ROWS) {
    for (const where of ['between the decision and the first resume', 'after every resume'] as const) {
      it(`${name} ${where}: the approved call runs exactly once in three resumes`, async () => {
        const r = new Rig();
        const root = r.session('g399');
        await root.usingMode('work');
        r.calls([['echo', { n: 1 }]]);
        const parked = await r.runtime.send(root, 'Go');
        await recordApproval(root, parked.pendingApprovals[0]!.id, true);
        await root.thread().record([row]);
        for (let resume = 0; resume < 3; resume += 1) {
          await r.runtime.send(root, '', undefined, noStep(resume));
          if (where === 'after every resume') await root.thread().record([row]);
        }
        expect(r.echoed).toEqual([1]);
        // No model step was taken by any resume.
        expect(r.requests).toHaveLength(1);
      });
    }
  }

  it('stored rows: system rows around the decision, then a resume that takes no model step, three times: runs once', async () => {
    const r = new Rig();
    const root = r.session('g399-stored');
    await root.usingMode('work');
    await root.thread().record([
      { type: 'user', content: 'Go' },
      { ...assistantRowOf([{ id: 'c1', name: 'echo', arguments: { n: 1 } }], [{ approval_id: 'apr_1', tool_call_id: 'c1', arguments_format: 1 }]), harness_tool_names: ['echo', 'free'] },
      { type: 'system', content: 'Note 1.' },
      { type: 'tool_result', tool_results: [], tool_approval_responses: [{ approval_id: 'apr_1', approved: true, reason: null }] },
      { type: 'system', content: 'Note 2.' },
    ]);
    for (let resume = 0; resume < 3; resume += 1) {
      await r.runtime.send(root, '', undefined, noStep(resume));
      await root.thread().record([{ type: 'system', content: `After ${resume}.` }]);
    }
    expect(r.echoed).toEqual([1]);
  });
});

function assistantRowOf(calls: JsonObject[], requests: JsonObject[]): JsonObject {
  return { type: 'assistant', content: '', tool_calls: calls, additional_content: {}, tool_approval_requests: requests };
}
