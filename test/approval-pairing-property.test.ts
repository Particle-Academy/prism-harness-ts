import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  ToolRegistry,
  recordApproval,
  recordDecisions,
  type AgentResponse,
  type JsonObject,
  type LlmRequest,
  type LlmResponse,
  type RunStep,
  type Session,
} from '../src/index.js';
import { ARGUMENTS_AS_SENT, assistantRow, toolResultEntry, toolResultRow } from '../src/thread-rows.js';

// G-392 QC: property tests for the approval pairing invariants.
//
// Each example is a generated model step (1-6 tool calls, ids from a small
// pool so repeats and empty ids are common, gated and ungated tools), entered
// where a real caller enters: the runtime's send / sendStructured /
// streamSteps / streamDecision over a fake model, with decisions recorded by
// recordApproval / recordDecisions / streamDecision. A second source writes
// the step as stored rows, as another writer could hold them: requests
// shuffled, missing or repeated, prior results present. Then one or more
// resumes, with later decisions in between.
//
// The oracle is a small independent model of the rule (below, `expected`),
// never a call into the code under test:
//   a step whose call ids repeat (empty ids count), or whose approval requests
//   repeat a tool_call_id, runs none of its gated calls and gives each the
//   repeated-id denial; otherwise a gated call runs only if its OWN request
//   was approved; no decision is a denial; a call with a result never runs.
//
// Deterministic: a fixed base seed. PRISM_PROPERTY_EXAMPLES sets the count,
// PRISM_PROPERTY_BASE_SEED the base, PRISM_PROPERTY_SEED replays one example.
// A failure prints the seed, the shrunk case and the full case.

const REPEATED = 'Tool call was not run because another call in this step has the same id';
const NO_RESPONSE = 'No approval response provided';
const DENIED = 'User denied tool execution';

const EXAMPLES = positive(process.env.PRISM_PROPERTY_EXAMPLES, 150, 1);
const BASE_SEED = positive(process.env.PRISM_PROPERTY_BASE_SEED, 0x6392);
const REPLAY = process.env.PRISM_PROPERTY_SEED;

/** The variable's value, or `fallback` when it is not set. A value that is set but invalid FAILS (no silent default). */
function positive(value: string | undefined, fallback: number, min = 0): number {
  if (value === undefined) return fallback;
  if (!/^[0-9]+$/.test(value) || !Number.isSafeInteger(Number(value)) || Number(value) < min) {
    throw new Error(`A property-test setting must be a whole number >= ${min}, got ${JSON.stringify(value)}.`);
  }
  return Number(value);
}

// ---------------------------------------------------------------------------
// Seeded generator (mulberry32). No third-party dependency.
// ---------------------------------------------------------------------------

class Rng {
  #state: number;

  constructor(seed: number) {
    this.#state = seed >>> 0;
  }

  next(): number {
    this.#state = (this.#state + 0x6d2b79f5) >>> 0;
    let t = this.#state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  int(lo: number, hi: number): number {
    return lo + Math.floor(this.next() * (hi - lo + 1));
  }

  chance(p: number): boolean {
    return this.next() < p;
  }

  pick<T>(items: readonly T[]): T {
    return items[Math.floor(this.next() * items.length)]!;
  }

  shuffle<T>(items: readonly T[]): T[] {
    const out = [...items];
    for (let i = out.length - 1; i > 0; i -= 1) {
      const j = Math.floor(this.next() * (i + 1));
      [out[i], out[j]] = [out[j]!, out[i]!];
    }
    return out;
  }
}

function caseSeed(base: number, index: number): number {
  let h = (base ^ Math.imul(index + 1, 0x9e3779b1)) >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x85ebca6b) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35) >>> 0;
  return (h ^ (h >>> 16)) >>> 0;
}

// ---------------------------------------------------------------------------
// The generated case.
// ---------------------------------------------------------------------------

type Verdict = 'approve' | 'deny' | 'none';
type Path = 'send' | 'structured' | 'streamSteps';
type Recorder = 'recordApproval' | 'recordDecisions' | 'streamDecision';

interface Call {
  id: string;
  tool: 'echo' | 'free';
  /** Unique per call, carried in its arguments: tells calls apart when ids repeat. */
  token: number;
}

interface Request {
  /** Index into `calls` of the call this request names (by its id). */
  call: number;
  verdict: Verdict;
  reason: string | null;
}

interface Resume {
  /** Record an approval for every approval id before this resume (a later decision). */
  approveAll: boolean;
  path: Path;
}

interface Case {
  seed: number;
  /** model: the runtime issues the requests. stored: rows written as another writer could hold them. */
  source: 'model' | 'stored';
  calls: Call[];
  /** In call order for `model` (one per gated call); as written for `stored`. */
  requests: Request[];
  /** Indices of calls with a result already stored (stored source only). */
  prior: number[];
  /** Order in which the requests' decisions are recorded. */
  order: number[];
  initial: Path;
  recorder: Recorder;
  resume: Path;
  again: Resume[];
  /**
   * G-397 / G-399: a row recorded after the park, before the decision or
   * between the decision and the resume, or null. Drawn from its own
   * generator, so the rest of every seed's case is what it was before.
   */
  insert: Insert | null;
}

type InsertKind = 'assistant_text' | 'assistant_call' | 'user' | 'system' | 'tool_empty' | 'tool_result';

interface Insert {
  kind: InsertKind;
  when: 'before_decision' | 'after_decision';
  /** assistant_call: the step call whose id the inserted (ungated) call reuses; tool_result: the gated call the forged result names. */
  call: number;
}

/** A kind that makes the inserted row the tail step, or ends the turn: the parked step is superseded. */
const SUPERSEDES: readonly InsertKind[] = ['assistant_text', 'assistant_call', 'user'];

const ID_POOL = ['a', 'b', 'c', ''];
const PATHS: Path[] = ['send', 'structured', 'streamSteps'];

function generate(seed: number): Case {
  const rng = new Rng(seed);
  const source = rng.chance(0.6) ? 'model' : 'stored';
  const count = rng.int(1, 6);
  const calls: Call[] = [];
  for (let i = 0; i < count; i += 1) {
    calls.push({ id: rng.pick(ID_POOL), tool: rng.chance(0.7) ? 'echo' : 'free', token: 100 + i });
  }
  if (!calls.some((call) => call.tool === 'echo')) calls[rng.int(0, count - 1)]!.tool = 'echo';

  const verdict = (): Request => {
    const v = rng.pick<Verdict>(['approve', 'approve', 'deny', 'none']);
    return { call: -1, verdict: v, reason: v === 'deny' && rng.chance(0.5) ? `no-${rng.int(0, 999)}` : null };
  };
  const gated = calls.flatMap((call, index) => (call.tool === 'echo' ? [index] : []));
  let requests: Request[];
  let prior: number[] = [];
  if (source === 'model') {
    requests = gated.map((call) => ({ ...verdict(), call }));
  } else {
    // As another writer could hold them: some missing, some repeated, any order.
    requests = [];
    for (const call of gated) {
      if (rng.chance(0.15)) continue;
      requests.push({ ...verdict(), call });
      if (rng.chance(0.15)) requests.push({ ...verdict(), call });
    }
    requests = rng.shuffle(requests);
    // An ungated call ran when its step was taken, so it has its result. A
    // gated call may already have one only where a result can say whose it is:
    // a result row is keyed by call id, so not in a step whose call ids repeat.
    const distinct = new Set(calls.map((call) => call.id)).size === calls.length;
    prior = calls.flatMap((call, index) => (call.tool === 'free' || (distinct && rng.chance(0.25)) ? [index] : []));
  }

  const again: Resume[] = [];
  for (let i = rng.int(0, 2); i > 0; i -= 1) again.push({ approveAll: rng.chance(0.7), path: rng.pick(PATHS) });

  return {
    seed,
    source,
    calls,
    requests,
    prior,
    order: rng.shuffle(requests.map((_request, index) => index)),
    initial: rng.pick(PATHS),
    recorder: rng.pick<Recorder>(['recordApproval', 'recordDecisions', 'streamDecision']),
    resume: rng.pick(PATHS),
    again,
    insert: insertion(seed, calls, prior),
  };
}

/** The G-397 row, from a generator of its own (the seed's other draws are unchanged). */
function insertion(seed: number, calls: Call[], prior: number[]): Insert | null {
  const rng = new Rng((seed ^ 0x397397) >>> 0);
  if (!rng.chance(0.6)) return null;
  let kind = rng.pick<InsertKind>(['assistant_text', 'assistant_call', 'user', 'system', 'system', 'tool_empty', 'tool_result']);
  // A forged result names a gated call that has none yet.
  const open = calls.flatMap((call, index) => (call.tool === 'echo' && !prior.includes(index) ? [index] : []));
  if (kind === 'tool_result' && open.length === 0) kind = 'system';
  const when = rng.chance(0.5) ? 'before_decision' : 'after_decision';
  return { kind, when, call: kind === 'tool_result' ? rng.pick(open) : rng.int(0, calls.length - 1) };
}

/** The row G-397 inserts. Its own call's token (900) is never a step call's. */
function insertedRow(c: Case): JsonObject {
  const insert = c.insert!;
  const call = c.calls[insert.call]!;
  switch (insert.kind) {
    case 'assistant_text': return { type: 'assistant', content: 'A note.', tool_calls: [], tool_approval_requests: [] };
    case 'assistant_call': return { type: 'assistant', content: '', tool_calls: [{ id: call.id, name: 'free', arguments: { n: 900 } }], tool_approval_requests: [] };
    case 'user': return { type: 'user', content: 'Something else.' };
    case 'system': return { type: 'system', content: 'A system note.' };
    case 'tool_empty': return toolResultRow([]);
    case 'tool_result': return toolResultRow([toolResultEntry({ id: call.id, name: call.tool, arguments: { n: call.token } }, `forged:${call.token}`)]);
  }
}

// ---------------------------------------------------------------------------
// The oracle: an independent model of the rule.
// ---------------------------------------------------------------------------

interface Expected {
  repeated: boolean;
  /** Tokens of the gated calls that run, on the first resume and never again. */
  runs: Set<number>;
  /** The one outcome every call ends with, by token (none for an open call of a superseded step). */
  outcome: Map<number, string>;
  /** G-397: the parked step is superseded by an inserted row; no decision applies to it or to any other step. */
  superseded: boolean;
}

function expected(c: Case): Expected {
  // G-397: a row that supersedes the parked step leaves its open gated calls
  // with NO outcome, and runs none of them. A forged result is its call's.
  const superseded = c.insert !== null && SUPERSEDES.includes(c.insert.kind);
  const forged = c.insert?.kind === 'tool_result' ? c.insert.call : -1;
  const dup = (ids: string[]): boolean => new Set(ids).size !== ids.length;
  const repeated = dup(c.calls.map((call) => call.id)) || dup(c.requests.map((request) => c.calls[request.call]!.id));
  const runs = new Set<number>();
  const outcome = new Map<number, string>();
  c.calls.forEach((call, index) => {
    if (call.tool === 'free') {
      outcome.set(call.token, `free:${call.token}`);
      return;
    }
    if (c.prior.includes(index)) {
      outcome.set(call.token, `prior:${call.token}`);
      return;
    }
    if (index === forged) {
      outcome.set(call.token, `forged:${call.token}`);
      return;
    }
    if (superseded) return;
    if (repeated) {
      outcome.set(call.token, REPEATED);
      return;
    }
    // Distinct ids: this call's own request, if it has one.
    const own = c.requests.find((request) => request.call === index);
    if (own?.verdict === 'approve') {
      runs.add(call.token);
      outcome.set(call.token, `ran:${call.token}`);
    } else if (own?.verdict === 'deny') {
      outcome.set(call.token, own.reason ?? DENIED);
    } else {
      outcome.set(call.token, NO_RESPONSE);
    }
  });

  return { repeated, runs, outcome, superseded };
}

// ---------------------------------------------------------------------------
// The rig: the runtime over a fake model, every store in memory.
// ---------------------------------------------------------------------------

class DurableMemory extends MemorySessionStore {
  override durability(): 'durable' {
    return 'durable';
  }
}

const SCHEMA: JsonObject = { type: 'object', properties: { ok: { type: 'boolean' } }, required: ['ok'] };

const modes = new ModeRegistry({
  default: 'guarded',
  modes: { guarded: { system_prompt: 'Careful.', tools: ['echo', 'free'], max_steps: 6, requires_approval: ['echo'] } },
});

interface Rig {
  session: Session;
  agent: AgentRuntime;
  /** Token of every gated call that ran, in order, across the whole case. */
  ran: number[];
  freeRan: number[];
}

function rig(c: Case): Rig {
  const harness = new PrismHarness({
    drivers: { memory: () => new MemorySessionStore(), durable: () => new DurableMemory() },
    stores: { ephemeral: 'memory', durable: 'durable' },
  });
  const session = harness.for({ type: 'User', id: 1 }).session(`property-${c.seed}`);
  const ran: number[] = [];
  const freeRan: number[] = [];
  const tools = new ToolRegistry()
    .register({ name: 'echo', handle: (args: JsonObject) => { ran.push(Number(args.n)); return `ran:${String(args.n)}`; } })
    .register({ name: 'free', handle: (args: JsonObject) => { freeRan.push(Number(args.n)); return `free:${String(args.n)}`; } });
  let served = c.source === 'stored';
  const client = async (request: LlmRequest): Promise<LlmResponse> => {
    if (!served) {
      served = true;
      return { text: '', finishReason: 'tool_calls', toolCalls: c.calls.map((call) => ({ id: call.id, name: call.tool, arguments: { n: call.token } })) };
    }
    return request.schema === undefined
      ? { text: 'done', finishReason: 'stop' }
      : { text: '{"ok":true}', finishReason: 'stop', structured: { ok: true } };
  };

  return { session, agent: new AgentRuntime({ client, modes, tools }), ran, freeRan };
}

async function drain(stream: AsyncGenerator<RunStep, AgentResponse, undefined>): Promise<{ steps: RunStep[]; response: AgentResponse }> {
  const steps: RunStep[] = [];
  for (;;) {
    const next = await stream.next();
    if (next.done === true) return { steps, response: next.value };
    steps.push(next.value);
  }
}

async function turn(r: Rig, path: Path, prompt: string): Promise<{ steps: RunStep[] | null; response: AgentResponse }> {
  if (path === 'send') return { steps: null, response: await r.agent.send(r.session, prompt) };
  if (path === 'structured') return { steps: null, response: await r.agent.sendStructured(r.session, prompt, SCHEMA) };
  return await drain(r.agent.streamSteps(r.session, prompt));
}

async function rows(session: Session): Promise<JsonObject[]> {
  return (await session.thread().messages()).map((entry) => entry.message);
}

/** Every stored result entry after the step's assistant row, as [token, result]. */
async function outcomes(session: Session): Promise<[number, string][]> {
  const all = await rows(session);
  const step = all.findIndex((row) => row.type === 'assistant' && Array.isArray(row.tool_calls) && row.tool_calls.length > 0);
  // The step's own rows: up to the next assistant row.
  const after = all.slice(step + 1);
  const next = after.findIndex((row) => row.type === 'assistant');
  return (next === -1 ? after : after.slice(0, next)).filter((row) => row.type === 'tool_result')
    .flatMap((row) => (row.tool_results as JsonObject[]))
    .map((entry) => [Number((entry.args as JsonObject).n), String(entry.result)]);
}

// ---------------------------------------------------------------------------
// One example: drive it, then check every invariant against the oracle.
// ---------------------------------------------------------------------------

class Violation extends Error {}

function check(condition: boolean, invariant: string, detail: unknown = undefined): void {
  if (!condition) throw new Violation(`${invariant}${detail === undefined ? '' : `: ${JSON.stringify(detail)}`}`);
}

function sameSet(a: Iterable<number>, b: Iterable<number>): boolean {
  const x = [...new Set(a)].sort((p, q) => p - q), y = [...new Set(b)].sort((p, q) => p - q);
  return JSON.stringify(x) === JSON.stringify(y);
}

async function runCase(c: Case): Promise<void> {
  const oracle = expected(c);
  const r = rig(c);
  const gatedTokens = c.calls.filter((call) => call.tool === 'echo').map((call) => call.token);
  let approvalIds: string[];

  if (c.source === 'model') {
    const parked = await turn(r, c.initial, 'go');
    check(parked.response.finishReason === 'awaiting_approval', 'the step parks for a person', parked.response.finishReason);
    // Invariant 6: what the runtime reports as pending is one approval per
    // gated call, in call order, each its own id, each naming its own call.
    const pending = parked.response.pendingApprovals;
    const gated = c.calls.filter((call) => call.tool === 'echo');
    check(pending.length === gated.length, '6: one pending approval per gated call', pending.length);
    check(new Set(pending.map((p) => p.id)).size === pending.length, '6: pending approval ids are distinct');
    pending.forEach((p, i) => check(p.toolCallId === gated[i]!.id && Number(p.arguments.n) === gated[i]!.token && p.tool === 'echo',
      '6: pending approval names its own call', { p, call: gated[i] }));
    const stored = (await rows(r.session)).find((row) => row.type === 'assistant' && Array.isArray(row.tool_calls) && row.tool_calls.length > 0)!;
    check(JSON.stringify((stored.tool_approval_requests as JsonObject[]).map((q) => q.approval_id)) === JSON.stringify(pending.map((p) => p.id)),
      '6: the stored requests are the reported approvals');
    // Ungated calls ran at once, each exactly once.
    check(sameSet(r.freeRan, c.calls.filter((call) => call.tool === 'free').map((call) => call.token)) && r.freeRan.length === c.calls.filter((call) => call.tool === 'free').length,
      'ungated calls run once when the step is taken', r.freeRan);
    check(r.ran.length === 0, '1: no gated call runs before a decision', r.ran);
    approvalIds = pending.map((p) => p.id);
  } else {
    approvalIds = c.requests.map((_request, index) => `apr_${c.seed}_${index}`);
    const calls = c.calls.map((call) => ({ id: call.id, name: call.tool, arguments: { n: call.token } }));
    await r.session.thread().record([
      { type: 'user', content: 'go' },
      {
        ...assistantRow('', calls, {}, c.requests.map((request, index) => ({
          approval_id: approvalIds[index]!, tool_call_id: c.calls[request.call]!.id, arguments_format: ARGUMENTS_AS_SENT,
        }))),
        harness_tool_names: ['echo', 'free'],
      },
      ...(c.prior.length === 0 ? [] : [toolResultRow(c.prior.map((index) => {
        const call = c.calls[index]!;
        return toolResultEntry({ id: call.id, name: call.tool, arguments: { n: call.token } }, `${call.tool === 'free' ? 'free' : 'prior'}:${call.token}`);
      }))]),
    ]);
  }

  // G-397: a row recorded between the park and the decision.
  if (c.insert?.when === 'before_decision') await r.session.thread().record([insertedRow(c)]);

  // The decisions, in a shuffled order.
  const decisions = c.order.filter((index) => c.requests[index]!.verdict !== 'none').map((index) => ({
    approvalId: approvalIds[index]!,
    approved: c.requests[index]!.verdict === 'approve',
    reason: c.requests[index]!.reason,
  }));
  // A decision on a superseded step is refused (approval_not_pending) on
  // every path; nothing else is (G-397).
  const refusedBefore = oracle.superseded && c.insert?.when === 'before_decision';
  const refused = async (body: () => Promise<unknown>): Promise<boolean> => body().then(() => false, (error: unknown) => {
    if ((error as { code?: string }).code === 'approval_not_pending') return true;
    throw error;
  });
  let first: { steps: RunStep[] | null; response: AgentResponse } | null = null;
  if (c.recorder === 'streamDecision' && decisions.length > 0 && c.insert?.when !== 'after_decision') {
    const streamed: { value: { steps: RunStep[] | null; response: AgentResponse } | null } = { value: null };
    const no = await refused(async () => { streamed.value = await drain(r.agent.streamDecision(r.session, decisions)); });
    check(no === refusedBefore, 'G-397: a decision is refused exactly when its step is superseded', { no, refusedBefore });
    if (streamed.value !== null) {
      first = streamed.value;
      const recorded = first.steps!.filter((step) => step.type === 'decision_recorded');
      check(JSON.stringify(recorded.map((step) => [step.approvalId, step.verdict])) === JSON.stringify(decisions.map((d) => [d.approvalId, d.approved ? 'approved' : 'denied'])),
        '6: one decision_recorded step per decision, as given');
    }
  } else if (c.recorder === 'recordDecisions' && decisions.length > 0) {
    const no = await refused(() => recordDecisions(r.session, decisions));
    check(no === refusedBefore, 'G-397: a decision is refused exactly when its step is superseded', { no, refusedBefore });
  } else {
    // (streamDecision with a row after the decision: recorded one by one,
    // since a stream resumes at once and leaves no gap for the row.)
    for (const d of decisions) {
      const no = await refused(() => recordApproval(r.session, d.approvalId, d.approved, d.reason));
      check(no === refusedBefore, 'G-397: a decision is refused exactly when its step is superseded', { no, refusedBefore });
    }
  }
  // G-397 / G-399: a row recorded between the decision and the resume.
  if (c.insert?.when === 'after_decision') await r.session.thread().record([insertedRow(c)]);
  first ??= await turn(r, c.resume, '');

  // Invariants 1-4, then full equality with the oracle.
  check(r.ran.length === new Set(r.ran).size, '3: a call runs at most once', r.ran);
  const denied = new Set(c.requests.filter((q) => q.verdict === 'deny').map((q) => c.calls[q.call]!.token));
  check(r.ran.every((token) => !denied.has(token)), '1: a call a person denied never runs', r.ran);
  const approves = c.requests.filter((q) => q.verdict === 'approve');
  check(r.ran.every((token) => approves.some((q) => c.calls[q.call]!.token === token)), '2: each run call has its own approve', r.ran);
  check(r.ran.length <= approves.length, '2: one decision decides at most one call', { ran: r.ran.length, approves: approves.length });
  if (oracle.repeated) check(r.ran.length === 0, '4: a step with repeated ids runs none of its gated calls', r.ran);
  check(sameSet(r.ran, oracle.runs), 'the calls that ran are exactly the oracle\'s', { ran: r.ran, expected: [...oracle.runs] });

  // Invariant 6 on the resume itself: what it says it ran is what ran.
  check(first.response.toolCalls.filter((name) => name === 'echo').length === r.ran.length, '6: the resume reports the calls it ran', first.response.toolCalls);
  if (first.steps !== null) {
    const results = first.steps.filter((step) => step.type === 'tool_result');
    const ok = results.flatMap((step) => (step.ok ? [Number(step.result.replace(/^ran:/, ''))] : []));
    check(sameSet(ok, r.ran) && ok.length === r.ran.length, '6: the ok tool_result steps are the calls that ran', ok);
    // Open: no prior and no forged result; none at all once the step is superseded (G-397).
    const forgedToken = c.insert?.kind === 'tool_result' ? c.calls[c.insert.call]!.token : -1;
    const open = oracle.superseded ? [] : gatedTokens.filter((token) => token !== forgedToken && !c.prior.includes(c.calls.findIndex((call) => call.token === token)));
    check(results.length === open.length, '6: one tool_result step per open gated call', { steps: results.length, open: open.length });
  }

  await settledOutcomes(r, oracle, '5');

  // Repeat resumes, with later decisions in between: nothing runs twice and
  // no outcome changes.
  const ranBefore = [...r.ran];
  for (const again of c.again) {
    if (again.approveAll) {
      for (const id of approvalIds) await recordApproval(r.session, id, true).catch(() => undefined);
    }
    const next = await turn(r, again.path, '');
    check(next.response.toolCalls.filter((name) => name === 'echo').length === 0, '3/6: a repeat resume reports no run', next.response.toolCalls);
    if (next.steps !== null) check(!next.steps.some((step) => step.type === 'tool_result'), '3: a repeat resume answers no call again');
  }
  check(JSON.stringify(r.ran) === JSON.stringify(ranBefore), '3: a call runs at most once across any resume sequence', { before: ranBefore, after: r.ran });
  check(r.freeRan.length === c.calls.filter((call) => call.tool === 'free').length || c.source === 'stored', '3: an ungated call runs once', r.freeRan);
  if (c.source === 'stored') check(r.freeRan.length === 0, 'a stored ungated call with a result never runs', r.freeRan);
  await settledOutcomes(r, oracle, '5 (after repeat resumes)');
}

/** Invariants 5 and 6 on the thread rows: every call has exactly one outcome, the oracle's. */
async function settledOutcomes(r: Rig, oracle: Expected, label: string): Promise<void> {
  const seen = new Map<number, Set<string>>();
  for (const [token, result] of await outcomes(r.session)) {
    if (!seen.has(token)) seen.set(token, new Set());
    seen.get(token)!.add(result);
  }
  for (const [token, want] of oracle.outcome) {
    const got = [...(seen.get(token) ?? [])];
    check(got.length === 1, `${label}: every call gets exactly one recorded outcome`, { token, got });
    check(got[0] === want, `6: the recorded outcome matches what ran`, { token, got: got[0], want });
  }
  check([...seen.keys()].every((token) => oracle.outcome.has(token)), `${label}: no outcome for a call that is not in the step`, [...seen.keys()]);
}

// ---------------------------------------------------------------------------
// Shrinking: drop calls, requests and repeat resumes while it still fails.
// ---------------------------------------------------------------------------

function withoutCall(c: Case, drop: number): Case | null {
  if (c.calls.length <= 1) return null;
  const calls = c.calls.filter((_call, index) => index !== drop);
  if (!calls.some((call) => call.tool === 'echo')) return null;
  const remap = (index: number): number => (index > drop ? index - 1 : index);
  const keptRequests = c.requests.map((request, index) => ({ request, index })).filter(({ request }) => request.call !== drop);
  const position = new Map(keptRequests.map(({ index }, at) => [index, at]));
  const requests = keptRequests.map(({ request }) => ({ ...request, call: remap(request.call) }));
  return {
    ...c,
    calls,
    requests,
    prior: c.prior.filter((index) => index !== drop).map(remap),
    order: c.order.filter((index) => position.has(index)).map((index) => position.get(index)!),
  };
}

function withoutRequest(c: Case, drop: number): Case | null {
  if (c.source === 'model') return null;
  return {
    ...c,
    requests: c.requests.filter((_request, index) => index !== drop),
    order: c.order.filter((index) => index !== drop).map((index) => (index > drop ? index - 1 : index)),
  };
}

function smaller(c: Case): Case[] {
  const out: Case[] = [];
  if (c.again.length > 0) out.push({ ...c, again: c.again.slice(0, -1) });
  for (let i = 0; i < c.calls.length; i += 1) { const s = withoutCall(c, i); if (s !== null) out.push(s); }
  for (let i = 0; i < c.requests.length; i += 1) { const s = withoutRequest(c, i); if (s !== null) out.push(s); }
  return out;
}

async function failure(c: Case): Promise<string | null> {
  try {
    await runCase(c);
    return null;
  } catch (error) {
    return error instanceof Error ? `${error.constructor.name}: ${error.message}` : String(error);
  }
}

/** The invariant a failure names: its text up to the detail. */
function invariant(reason: string): string {
  return reason.split(': {')[0]!.split(': [')[0]!;
}

async function shrink(c: Case, reason: string): Promise<{ c: Case; reason: string }> {
  let best = { c, reason };
  for (let budget = 200; budget > 0;) {
    let improved = false;
    for (const candidate of smaller(best.c)) {
      budget -= 1;
      const why = await failure(candidate);
      // Only a candidate that breaks the SAME invariant: shrinking must not
      // trade a safety failure for a different, smaller one.
      if (why !== null && invariant(why) === invariant(reason)) { best = { c: candidate, reason: why }; improved = true; break; }
      if (budget <= 0) break;
    }
    if (!improved) break;
  }
  return best;
}

function describeCase(c: Case): string {
  return JSON.stringify(c);
}

// ---------------------------------------------------------------------------

const network = vi.fn(() => { throw new Error('A test tried to use the network.'); });
beforeEach(() => { vi.stubGlobal('fetch', network); });
afterEach(() => { expect(network).not.toHaveBeenCalled(); vi.unstubAllGlobals(); });

describe('G-392 property: approval pairing invariants over generated steps', () => {
  it(`holds for ${REPLAY === undefined ? `${EXAMPLES} examples from base seed ${BASE_SEED}` : `seed ${REPLAY}`}`, async () => {
    const seeds = REPLAY === undefined
      ? Array.from({ length: EXAMPLES }, (_unused, index) => caseSeed(BASE_SEED, index))
      : [Number(REPLAY) >>> 0];
    const started = Date.now();
    for (const seed of seeds) {
      const c = generate(seed);
      const why = await failure(c);
      if (why === null) continue;
      const small = await shrink(c, why);
      const report = [
        `Property failed at seed ${seed} (replay: PRISM_PROPERTY_SEED=${seed}).`,
        `Invariant: ${small.reason}`,
        `Shrunk case: ${describeCase(small.c)}`,
        `Original failure: ${why}`,
        `Full case: ${describeCase(c)}`,
      ].join('\n');
      console.error(report);
      throw new Error(report);
    }
    console.info(`G-392 property: ${seeds.length} examples passed in ${Date.now() - started} ms (base seed ${BASE_SEED}).`);
  }, Math.max(120_000, EXAMPLES * 100));

  // The three seeds that failed on main 63ccf38 (BRIEF-R13, the G-392
  // property test), kept as fixed regression cases in every language:
  // (a) a gated call got no outcome, (b) a call got an extra repeated-id
  // denial, (c) a request paired with an ungated call.
  it.each([2034523272, 232174035, 2060958372])('regression seed %i', async (seed) => {
    const why = await failure(generate(seed));
    expect(why).toBeNull();
  });

  it('the generator reaches every shape the brief asks for', () => {
    const cases = Array.from({ length: Math.max(EXAMPLES, 150) }, (_unused, index) => generate(caseSeed(BASE_SEED, index)));
    const has = (predicate: (c: Case) => boolean): boolean => cases.some(predicate);
    expect(has((c) => expected(c).repeated && c.source === 'model')).toBe(true);
    expect(has((c) => !expected(c).repeated && expected(c).runs.size > 0)).toBe(true);
    expect(has((c) => c.calls.filter((call) => call.id === '').length > 1)).toBe(true);
    expect(has((c) => c.source === 'stored' && new Set(c.requests.map((q) => q.call)).size < c.requests.length)).toBe(true);
    expect(has((c) => c.source === 'stored' && c.prior.some((index) => c.calls[index]!.tool === 'echo'))).toBe(true);
    expect(has((c) => c.again.length > 0 && c.again.some((a) => a.approveAll))).toBe(true);
    for (const path of PATHS) expect(has((c) => c.resume === path)).toBe(true);
    expect(has((c) => c.recorder === 'streamDecision')).toBe(true);
    // G-397 / G-399: every inserted row kind, before and after the decision.
    for (const kind of ['assistant_text', 'assistant_call', 'user', 'system', 'tool_empty', 'tool_result'] as const) {
      for (const when of ['before_decision', 'after_decision'] as const) expect(has((c) => c.insert?.kind === kind && c.insert.when === when), `${kind} ${when}`).toBe(true);
    }
    // G-399's shape: a call that runs, and a row that is neither user nor assistant between its decision and the resume.
    expect(has((c) => (c.insert?.kind === 'system' || c.insert?.kind === 'tool_empty') && c.insert.when === 'after_decision' && expected(c).runs.size > 0)).toBe(true);
  });
});
