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
  type JsonValue,
  type LlmRequest,
  type LlmResponse,
  type RunStep,
  type Session,
} from '../src/index.js';
import { ARGUMENTS_AS_SENT, threadView, toolResultEntry, toolResultRow } from '../src/thread-rows.js';

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
const REPLACED = 'Tool call was not run because a later step replaced it';
const DENIED = 'User denied tool execution';

/** The fixed-seed set normal CI runs; the coverage check below is asserted on exactly these seeds. */
const DEFAULT_EXAMPLES = 1000;
const FIXED_BASE_SEED = 0x6392;
const EXAMPLES = envInteger('PRISM_PROPERTY_EXAMPLES', 1, Number.MAX_SAFE_INTEGER, DEFAULT_EXAMPLES)!;
const BASE_SEED = envInteger('PRISM_PROPERTY_BASE_SEED', 0, Number.MAX_SAFE_INTEGER, FIXED_BASE_SEED)!;
const REPLAY = envInteger('PRISM_PROPERTY_SEED', 0, 0xffffffff, undefined);

/**
 * An integer setting from the environment. Unset means the default; a value
 * that is SET must be a plain decimal integer in [min, max], and anything else
 * (empty, 0 for a count, negative, non-numeric, fractional) is refused. A run
 * that silently checked nothing, or fell back to a default nobody asked for,
 * would report green for a property it never examined.
 */
function envInteger(name: string, min: number, max: number, fallback: number | undefined): number | undefined {
  const value = process.env[name];
  if (value === undefined) return fallback;
  const parsed = /^[0-9]+$/.test(value) ? Number(value) : NaN;
  if (!Number.isSafeInteger(parsed) || parsed < min || parsed > max) {
    throw new Error(`${name} must be an integer from ${min} to ${max} when set; got ${JSON.stringify(value)}.`);
  }
  return parsed;
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
  /**
   * Index of the request whose approval id this one carries (stored only):
   * its own index, or an earlier request's when requests repeat an approval
   * id (G-397). A request that repeats another's id records no decision.
   */
  approval?: number;
}

/** A later tool-call step another writer recorded after the park (G-397). */
interface LaterStep {
  calls: Call[];
  /** One per gated call, in call order, each with its own approval id. */
  requests: Request[];
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
   * G-397 / G-399: a row the host recorded (outside any run) after the park,
   * before the decision or between the decision and the resume, or null.
   * Drawn from a generator of its own (seed ^ 0x397397), the shape prism-ts
   * main uses, so the rest of every seed's case is unchanged by it.
   */
  insert: Insert | null;
  /** The model call after the first resume's resolution fails once, so the resume ends there. */
  failOnce: boolean;
  /** The run's steps are spent when it parks, so a resume resolves and stops with no model call (G-399). */
  spent: boolean;
  /** A later tool-call step after the park: it is the step decided, and the parked one is replaced. */
  later: LaterStep | null;
  /**
   * G-397: stored arguments in the form another writer may keep them: true
   * for a call whose arguments are stored as a JSON string (as the PHP
   * reference stores them), by call position. Stored rows only: the parked
   * step of a stored case, and a later step.
   */
  stringArgs: { parked: boolean[]; later: boolean[] };
  /**
   * G-397: an earlier answered step before the parked one, reusing its call
   * ids: each call names a parked call whose id it reuses, with that call's
   * own arguments (`same`) or other ones. Its calls ran; its results are not
   * outcomes of the parked step.
   */
  history: { call: number; same: boolean; string: boolean }[] | null;
  /**
   * G-397: indices of prior results (stored cases) recorded with arguments
   * that differ from their call's own (lossy: none), as a store that redacts
   * arguments would keep them. They still answer their own call.
   */
  lossy: number[];
  /**
   * G-408: after every decision, the host records an assistant row whose
   * tool_calls hold no objects (a string, a number, null, strings, or a
   * string that is not an array), keeping or dropping the requests; or a
   * MIXED array, whose one object (an ungated call) makes it a tool-call step.
   */
  junk: { kind: 'string' | 'number' | 'null' | 'strings' | 'scalar' | 'mixed'; requests: boolean } | null;
  /**
   * G-408, order B: right after the park, BEFORE the person decides, the host
   * records a copy of the parked step that keeps its approval requests but
   * holds a NEW call with the same id and other arguments, alone or with
   * non-objects, or beside a copy of the old call. Then the person decides
   * the request they were shown. Not combined with a later step, an
   * inserted row or a junk row: each is its own shape.
   */
  copy: { kind: CopyKind } | null;
}

type CopyKind = 'newOnly' | 'stringThenNew' | 'nullThenNew' | 'newThenJunk' | 'oldAndNewSameId';
const COPY_KINDS: readonly CopyKind[] = ['newOnly', 'stringThenNew', 'nullThenNew', 'newThenJunk', 'oldAndNewSameId'];

/** The copy's tool_calls: the new call (same id as `target`, arguments n 950), shaped by `kind`. */
function copyCalls(kind: CopyKind, target: Call): JsonValue[] {
  const call = (n: number): JsonObject => ({ id: target.id, name: 'echo', arguments: { n }, result_id: null, reasoning_id: null, reasoning_summary: null });
  switch (kind) {
    case 'newOnly': return [call(950)];
    case 'stringThenNew': return ['x', call(950)];
    case 'nullThenNew': return [null, call(950)];
    case 'newThenJunk': return [call(950), 'x', null];
    case 'oldAndNewSameId': return [call(target.token), call(950)];
  }
}

/** An assistant row's tool_calls in the junk shape: never an object, except one ungated call in a mixed array. */
function junkCalls(kind: NonNullable<Case['junk']>['kind'], id: string): JsonValue {
  switch (kind) {
    case 'string': return ['x'];
    case 'number': return [7];
    case 'null': return [null];
    case 'strings': return [[id, 'echo'], JSON.stringify({ id, name: 'echo', arguments: {} })];
    case 'scalar': return JSON.stringify({ id });
    case 'mixed': return ['x', { id, name: 'free', arguments: { n: 901 }, result_id: null, reasoning_id: null, reasoning_summary: null }, null, 3];
  }
}

/** Whether a stored row is a tool-call step: an assistant row with at least one tool call that is an object. */
function isStepRow(row: JsonObject): boolean {
  return row.type === 'assistant' && Array.isArray(row.tool_calls) && row.tool_calls.some((call) => typeof call === 'object' && call !== null && !Array.isArray(call));
}

type InsertKind = 'assistant_text' | 'assistant_call' | 'user' | 'system' | 'tool_empty' | 'tool_result';
const INSERT_KINDS: readonly InsertKind[] = ['assistant_text', 'assistant_call', 'user', 'system', 'tool_empty', 'tool_result'];

interface Insert {
  kind: InsertKind;
  when: 'before_decision' | 'after_decision';
  /** assistant_call: the step call whose id the inserted (ungated) call reuses; tool_result: the gated call the forged result names. */
  call: number;
  /**
   * With a later step: whose segment the row is recorded in, the parked
   * step's (before the later step) or the later step's (after it).
   */
  segment: 'earlier' | 'later';
  /** Which step `call` indexes: always one that exists when the row is recorded. */
  step: 'earlier' | 'later';
}

/**
 * Kinds that SUPERSEDE the parked step when the host records them outside
 * the run that owns it (G-397): an assistant row with tool calls always; an
 * assistant text row or a user row recorded outside that run. system and
 * tool rows never do.
 */
const SUPERSEDES: readonly InsertKind[] = ['assistant_text', 'assistant_call', 'user'];

/** The G-397 row, from a generator of its own (the seed's other draws are unchanged). */
function insertion(seed: number, calls: Call[], prior: number[], later: LaterStep | null): Insert | null {
  const rng = new Rng((seed ^ 0x397397) >>> 0);
  // A case with a later step usually gets a row too, so every combination is reached.
  if (!rng.chance(later === null ? 0.6 : 0.85)) return null;
  let kind = rng.pick<InsertKind>(['assistant_text', 'assistant_call', 'user', 'system', 'system', 'tool_empty', 'tool_result']);
  const when = rng.chance(0.5) ? 'before_decision' : 'after_decision';
  // With a later step: the row goes in either step's segment, and names a call
  // of a step that exists by then (draws made only here, so a case with no
  // later step is drawn exactly as prism-ts main draws it).
  const segment = later !== null && rng.chance(0.5) ? 'later' : 'earlier';
  const step = segment === 'later' && rng.chance(0.5) ? 'later' : 'earlier';
  const target = step === 'later' ? later!.calls : calls;
  // A forged result names a gated call that has none yet.
  const open = target.flatMap((call, index) => (call.tool === 'echo' && (step === 'later' || !prior.includes(index)) ? [index] : []));
  if (kind === 'tool_result' && open.length === 0) kind = 'system';
  return { kind, when, segment, step, call: kind === 'tool_result' ? rng.pick(open) : rng.int(0, target.length - 1) };
}

/** The row G-397 inserts. Its own call's token (900) is never a step call's. */
function insertedRow(c: Case): JsonObject {
  const insert = c.insert!;
  const call = (insert.step === 'later' ? c.later!.calls : c.calls)[insert.call]!;
  switch (insert.kind) {
    case 'assistant_text': return { type: 'assistant', content: 'A note.', tool_calls: [], tool_approval_requests: [] };
    case 'assistant_call': return { type: 'assistant', content: '', tool_calls: [{ id: call.id, name: 'free', arguments: { n: 900 } }], tool_approval_requests: [] };
    case 'user': return { type: 'user', content: 'Something else.' };
    case 'system': return { type: 'system', content: 'A system note.' };
    case 'tool_empty': return toolResultRow([]);
    case 'tool_result': return toolResultRow([toolResultEntry({ id: call.id, name: call.tool, arguments: { n: call.token } }, `forged:${call.token}`)]);
  }
}

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
    // Sometimes a request repeats an earlier request's APPROVAL id.
    requests = requests.map((request, index) => (index > 0 && rng.chance(0.1)
      ? { ...request, approval: rng.int(0, index - 1), verdict: 'none' as Verdict, reason: null }
      : request));
    // An ungated call ran when its step was taken, so it has its result. A
    // gated call may already have one only where a result can say whose it is:
    // a result row is keyed by call id, so not in a step whose call ids repeat.
    const distinct = new Set(calls.map((call) => call.id)).size === calls.length;
    prior = calls.flatMap((call, index) => (call.tool === 'free' || (distinct && rng.chance(0.25)) ? [index] : []));
  }

  const again: Resume[] = [];
  for (let i = rng.int(0, 2); i > 0; i -= 1) again.push({ approveAll: rng.chance(0.7), path: rng.pick(PATHS) });
  const failOnce = rng.chance(0.25);
  // Spent: the resume makes no model call, so there is nothing to fail.
  const spent = !failOnce && rng.chance(0.25);
  let later: LaterStep | null = null;
  if (source === 'model' && rng.chance(0.4)) {
    const laterCalls: Call[] = [];
    for (let i = rng.int(1, 3); i > 0; i -= 1) laterCalls.push({ id: rng.pick(ID_POOL), tool: rng.chance(0.7) ? 'echo' : 'free', token: 200 + laterCalls.length });
    if (!laterCalls.some((call) => call.tool === 'echo')) laterCalls[0]!.tool = 'echo';
    later = { calls: laterCalls, requests: laterCalls.flatMap((call, index) => (call.tool === 'echo' ? [{ ...verdict(), call: index }] : [])) };
  }
  // A resume that failed after its resolution is always followed by another.
  if (failOnce && again.length === 0) again.push({ approveAll: rng.chance(0.5), path: rng.pick(PATHS) });

  const c: Case = {
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
    insert: insertion(seed, calls, prior, later),
    failOnce,
    spent,
    later,
    ...storage(seed, calls.length, later?.calls.length ?? 0),
    lossy: [],
    junk: null,
    copy: null,
  };
  return sharing(c);
}

/**
 * G-397: identical arguments across calls, and lossy recorded results, from a
 * generator of their own: an ungated call may carry the same arguments as a
 * gated call of its step; a later step's call may reuse a parked call's id
 * and arguments; a stored prior result may record no arguments.
 */
function sharing(c: Case): Case {
  const rng = new Rng((c.seed ^ 0x5e6) >>> 0);
  const gated = c.calls.filter((call) => call.tool === 'echo');
  const calls = c.calls.map((call) => (call.tool === 'free' && gated.length > 0 && rng.chance(0.3) ? { ...call, token: rng.pick(gated).token } : call));
  const later = c.later === null ? null : {
    ...c.later,
    calls: (() => {
      // Each parked call's arguments reused at most once, so a later call that
      // ran is still told apart by its arguments.
      const used = new Set<number>();
      return c.later!.calls.map((call) => {
        const from = rng.pick(gated);
        if (!rng.chance(0.3) || used.has(from.token)) return call;
        used.add(from.token);
        return { ...call, id: from.id, token: from.token };
      });
    })(),
  };
  const lossy = c.prior.filter((index) => c.calls[index]!.tool === 'echo' && rng.chance(0.5));
  const junk = rng.chance(0.15) ? { kind: rng.pick(['string', 'number', 'null', 'strings', 'scalar', 'mixed'] as const), requests: rng.chance(0.5) } : null;
  const copyDraw = rng.next(), copyKind = rng.pick(COPY_KINDS);
  const copy = copyDraw < 0.25 && c.later === null && c.insert === null && junk === null ? { kind: copyKind } : null;
  // Two IDENTICAL gated calls in one step (same id, tool and arguments): the
  // clone gets its own request, as the runtime would issue one.
  const source = gated[0];
  if (source !== undefined && rng.chance(0.15) && c.prior.every((index) => c.calls[index]!.tool === 'free')) {
    const index = calls.length;
    calls.push({ ...calls[c.calls.indexOf(source)]!, token: source.token });
    const requests = [...c.requests, { call: index, verdict: rng.pick<Verdict>(['approve', 'deny', 'none']), reason: null }];
    return { ...c, calls, later, lossy, junk, copy, requests, order: [...c.order, requests.length - 1], stringArgs: { ...c.stringArgs, parked: [...c.stringArgs.parked, false] } };
  }
  return { ...c, calls, later, lossy, junk, copy };
}

/** G-397: argument storage forms and an earlier step reusing ids, from a generator of their own. */
function storage(seed: number, parked: number, later: number): Pick<Case, 'stringArgs' | 'history'> {
  const rng = new Rng((seed ^ 0x3a6f121) >>> 0);
  const stringArgs = { parked: Array.from({ length: parked }, () => rng.chance(0.4)), later: Array.from({ length: later }, () => rng.chance(0.4)) };
  const history = rng.chance(0.35)
    ? Array.from({ length: rng.int(1, Math.min(3, parked)) }, () => ({ call: rng.int(0, parked - 1), same: rng.chance(0.6), string: rng.chance(0.4) }))
    : null;
  return { stringArgs, history };
}

// ---------------------------------------------------------------------------
// The oracle: an independent model of the rule.
// ---------------------------------------------------------------------------

interface Expected {
  repeated: boolean;
  /** Keys (p<i> parked, l<i> later) of the gated calls that run, on the first resume and never again. */
  runs: Set<string>;
  /** The one outcome every call ends with, by key. */
  outcome: Map<string, string>;
  /** Keys of the decided step's gated calls a resume answers (no result yet): none when every step was superseded. */
  open: string[];
}

function expected(c: Case): Expected {
  // Each step is superseded or decided. The parked step is superseded by a
  // later tool-call step, or by a superseding row; a later step only by a
  // superseding row in its own segment (G-397). A result answers only a call
  // of the step whose segment holds it (absolute segment ownership): by exact
  // arguments, then id and name, then id. A superseded step's gated calls
  // with no result of their own are denied once as replaced; nothing of it
  // runs.
  const insert = c.insert;
  const supersedes = insert !== null && SUPERSEDES.includes(insert.kind);
  const steps = [{ key: 'p', calls: c.calls, requests: c.requests, prior: c.prior, superseded: c.later !== null || supersedes }];
  if (c.later !== null) steps.push({ key: 'l', calls: c.later.calls, requests: c.later.requests, prior: [], superseded: supersedes && insert!.segment === 'later' });
  // G-408: the host's junk row, recorded after every decision, is an
  // assistant row outside the run (or, mixed, a tool-call step): it
  // supersedes whichever step was still being decided.
  if (c.junk !== null) steps.at(-1)!.superseded = true;
  // G-408, order B: the copy is a later tool-call step, so the parked step is
  // replaced. Its requests repeat approval ids the parked step already asked
  // for, so they ask for nothing: its gated calls have no decision.
  let copySteps: { key: string; calls: Call[]; requests: Request[]; prior: number[]; superseded: boolean } | null = null;
  if (c.copy !== null) {
    steps[0]!.superseded = true;
    const target = c.calls.find((call) => call.tool === 'echo')!;
    const calls = copyCalls(c.copy.kind, target).filter((entry): entry is JsonObject => typeof entry === 'object' && entry !== null && !Array.isArray(entry))
      .map((entry, index) => ({ id: String(entry.id), tool: 'echo' as const, token: Number((entry.arguments as JsonObject).n) + index * 0 }));
    copySteps = { key: 'k', calls, requests: [], prior: [], superseded: false };
  }
  // The call a forged result answers: in its own segment's step.
  let forged: { step: number; call: number; text: string } | null = null;
  if (insert?.kind === 'tool_result') {
    const named = (insert.step === 'later' ? c.later!.calls : c.calls)[insert.call]!;
    const text = `forged:${named.token}`;
    if (insert.segment === 'earlier' || insert.step === 'later') forged = { step: insert.step === 'later' ? 1 : 0, call: insert.call, text };
    else {
      // Named an earlier call, recorded in the later step's segment: it answers
      // the later step's first unanswered call with that id and name, then id.
      const later = c.later!.calls;
      const owned = (index: number): boolean => later[index]!.tool === 'free';
      let at = later.findIndex((call, index) => call.id === named.id && call.token === named.token && call.tool === 'echo' && !owned(index));
      if (at === -1) at = later.findIndex((call, index) => call.id === named.id && call.tool === 'echo' && !owned(index));
      if (at === -1) at = later.findIndex((call, index) => call.id === named.id && !owned(index));
      if (at !== -1) forged = { step: 1, call: at, text };
    }
  }
  const out: Expected = { repeated: false, runs: new Set<string>(), outcome: new Map<string, string>(), open: [] };
  steps.forEach((step, at) => {
    const answered = forged?.step === at ? forged.call : -1;
    if (!step.superseded) {
      const decided = decideStep(step.key, step.calls, step.requests, [...step.prior, ...(answered === -1 ? [] : [answered])]);
      out.repeated = decided.repeated;
      out.runs = decided.runs;
      out.open = decided.open;
      for (const [key, outcome] of decided.outcome) out.outcome.set(key, outcome);
    }
    step.calls.forEach((call, index) => {
      const key = `${step.key}${index}`;
      if (index === answered) out.outcome.set(key, forged!.text);
      else if (step.superseded) {
        out.outcome.set(key, call.tool === 'free' ? `free:${call.token}` : step.prior.includes(index) ? `prior:${call.token}` : REPLACED);
      }
    });
  });
  if (copySteps !== null) {
    // The copy keeps the parked step's requests: they still count when the
    // step is checked for repeated ids, but none of them asks for anything.
    const dup = (ids: (string | number)[]): boolean => new Set(ids).size !== ids.length;
    const repeated = dup(copySteps.calls.map((call) => call.id)) || dup(c.requests.map((q) => c.calls[q.call]!.id))
      || dup(c.requests.map((q, index) => q.approval ?? index));
    out.repeated = repeated;
    out.runs = new Set();
    out.open = copySteps.calls.map((_call, index) => `k${index}`);
    copySteps.calls.forEach((_call, index) => out.outcome.set(`k${index}`, repeated ? REPEATED : NO_RESPONSE));
  }
  return out;
}

/** The order the case records its rows and decisions in, after the park. */
type Action = 'insert' | 'later' | 'old' | 'new';
function actions(c: Case): Action[] {
  const insert = c.insert;
  if (c.later === null) return insert === null ? ['old'] : insert.when === 'before_decision' ? ['insert', 'old'] : ['old', 'insert'];
  if (insert === null) return ['later', 'old', 'new'];
  if (insert.segment === 'earlier') return insert.when === 'before_decision' ? ['insert', 'later', 'old', 'new'] : ['old', 'insert', 'later', 'new'];
  return insert.when === 'before_decision' ? ['later', 'insert', 'old', 'new'] : ['later', 'old', 'new', 'insert'];
}

function decideStep(prefix: string, calls: Call[], requests: Request[], prior: number[]): Expected {
  const dup = (ids: (string | number)[]): boolean => new Set(ids).size !== ids.length;
  const repeated = dup(calls.map((call) => call.id)) || dup(requests.map((request) => calls[request.call]!.id))
    || dup(requests.map((request, index) => request.approval ?? index));
  const runs = new Set<string>();
  const outcome = new Map<string, string>();
  const open: string[] = [];
  calls.forEach((call, index) => {
    const key = `${prefix}${index}`;
    if (call.tool === 'free') {
      outcome.set(key, `free:${call.token}`);
      return;
    }
    if (prior.includes(index)) {
      outcome.set(key, `prior:${call.token}`);
      return;
    }
    open.push(key);
    if (repeated) {
      outcome.set(key, REPEATED);
      return;
    }
    // Distinct ids: this call's own request, if it has one.
    const own = requests.find((request) => request.call === index);
    if (own?.verdict === 'approve') {
      runs.add(key);
      outcome.set(key, `ran:${call.token}`);
    } else if (own?.verdict === 'deny') {
      outcome.set(key, own.reason ?? DENIED);
    } else {
      outcome.set(key, NO_RESPONSE);
    }
  });

  return { repeated, runs, outcome, open };
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

/** One step: the step that parks spends the run, so a resume resolves and stops. */
const spentModes = new ModeRegistry({
  default: 'guarded',
  modes: { guarded: { system_prompt: 'Careful.', tools: ['echo', 'free'], max_steps: 1, requires_approval: ['echo'] } },
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
  let failing = c.failOnce;
  const client = async (request: LlmRequest): Promise<LlmResponse> => {
    if (!served) {
      served = true;
      return { text: '', finishReason: 'tool_calls', toolCalls: c.calls.map((call) => ({ id: call.id, name: call.tool, arguments: { n: call.token } })) };
    }
    if (failing) {
      failing = false;
      throw new Error('The model is unavailable (scripted).');
    }
    return request.schema === undefined
      ? { text: 'done', finishReason: 'stop' }
      : { text: '{"ok":true}', finishReason: 'stop', structured: { ok: true } };
  };

  return { session, agent: new AgentRuntime({ client, modes: c.spent ? spentModes : modes, tools }), ran, freeRan };
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

/** A tool-call step as the case recorded it: its calls, each with its key and its arguments as an object. */
type RecordedStep = { calls: { key: string; id: string; name: string; args: JsonObject }[] };

/** JSON with object keys sorted. */
function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (typeof value === 'object' && value !== null) {
    const object = value as Record<string, unknown>;
    return `{${Object.keys(object).sort().map((key) => `${JSON.stringify(key)}:${canonical(object[key])}`).join(',')}}`;
  }
  return JSON.stringify(value);
}

/**
 * Every recorded result, given to the call it belongs to, by the rule under
 * test written out independently: a result answers a call of the step whose
 * segment holds it (a replaced-step row: the step its mark names), by exact
 * id, name and arguments, then id and name, then id, each the first call still
 * without a result; a copy of a result an earlier row of that segment already
 * holds answers nothing new. Returns the results by call key.
 */
async function attributed(session: Session, steps: RecordedStep[]): Promise<Map<string, Set<string>>> {
  const out = new Map<string, Set<string>>();
  const owned = steps.map(() => new Set<string>());
  const seen = steps.map(() => new Set<string>());
  const at = new Map<number, number>();
  let step = -1;
  for (const { position, message } of await session.thread().messages()) {
    if (isStepRow(message)) {
      step += 1;
      at.set(position, step);
      continue;
    }
    if (step < 0 || message.type !== 'tool_result' || !Array.isArray(message.tool_results)) continue;
    const mark = message.harness_replaced_step as JsonObject | undefined;
    const target = mark !== undefined && typeof mark.position === 'number' ? at.get(mark.position) ?? step : step;
    const calls = steps[target]?.calls ?? [];
    const prints: string[] = [];
    for (const entry of message.tool_results as JsonObject[]) {
      const print = canonical([entry.tool_call_id, entry.tool_name, entry.args, entry.result]);
      if (seen[target]!.has(print)) continue;
      prints.push(print);
      const give = (key: string): void => {
        owned[target]!.add(key);
        if (!out.has(key)) out.set(key, new Set());
        out.get(key)!.add(String(entry.result));
      };
      const same = (call: RecordedStep['calls'][number]): boolean => call.id === entry.tool_call_id && call.name === entry.tool_name && canonical(call.args) === canonical(entry.args);
      // A further result for a call that has one still counts as its outcome (so a second one is seen).
      const exact = calls.find((call) => same(call) && !owned[target]!.has(call.key)) ?? calls.find(same);
      if (exact !== undefined) { give(exact.key); continue; }
      const byName = calls.find((call) => call.id === entry.tool_call_id && call.name === entry.tool_name && !owned[target]!.has(call.key));
      if (byName !== undefined) { give(byName.key); continue; }
      const byId = calls.find((call) => call.id === entry.tool_call_id && !owned[target]!.has(call.key));
      if (byId !== undefined) give(byId.key);
    }
    for (const print of prints) seen[target]!.add(print);
  }
  return out;
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

/**
 * An assistant row as another writer stores it, arguments kept exactly as
 * given: a JSON string stays a string (assistantRow() would not keep one).
 */
function storedRow(calls: { id: string; name: string; arguments: JsonObject }[], requests: JsonObject[]): JsonObject {
  return {
    type: 'assistant',
    content: '',
    tool_calls: calls.map((call) => ({ id: call.id, name: call.name, arguments: call.arguments, result_id: null, reasoning_id: null, reasoning_summary: null })),
    additional_content: {},
    tool_approval_requests: requests,
  };
}

async function runCase(c: Case): Promise<void> {
  const oracle = expected(c);
  const r = rig(c);
  let approvalIds: string[];
  // The tool-call steps as recorded, in thread order, for attribution.
  const recorded: RecordedStep[] = [];
  const parkedStep: RecordedStep = { calls: c.calls.map((call, index) => ({ key: `p${index}`, id: call.id, name: call.tool, args: { n: call.token } })) };

  if (c.history !== null) {
    // An earlier step that reused the parked step's call ids, answered and
    // run before it: every call approved, each with its result.
    const history = c.history.map((h, index) => {
      const call = c.calls[h.call]!;
      const args: JsonObject = { n: h.same ? call.token : 300 + index };
      return { id: call.id, args, stored: h.string ? JSON.stringify(args) : args };
    });
    await r.session.thread().record([
      { type: 'user', content: 'earlier' },
      {
        type: 'assistant', content: '', additional_content: {}, harness_tool_names: ['echo', 'free'],
        tool_calls: history.map((h) => ({ id: h.id, name: 'echo', arguments: h.stored, result_id: null, reasoning_id: null, reasoning_summary: null })),
        tool_approval_requests: history.map((h, index) => ({ approval_id: `apr_hist_${c.seed}_${index}`, tool_call_id: h.id, arguments_format: ARGUMENTS_AS_SENT })),
      },
      toolResultRow(history.map((h) => toolResultEntry({ id: h.id, name: 'echo', arguments: h.args }, `old:${String(h.args.n)}`)),
        history.map((_h, index) => ({ approval_id: `apr_hist_${c.seed}_${index}`, approved: true, reason: null }))),
      { type: 'assistant', content: 'ok', tool_calls: [], tool_approval_requests: [] },
    ], 'run_history');
    recorded.push({ calls: history.map((h, index) => ({ key: `h${index}`, id: h.id, name: 'echo', args: h.args })) });
  }
  recorded.push(parkedStep);
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
    const stored = [...await rows(r.session)].reverse().find((row) => row.type === 'assistant' && Array.isArray(row.tool_calls) && row.tool_calls.length > 0)!;
    check(JSON.stringify((stored.tool_approval_requests as JsonObject[]).map((q) => q.approval_id)) === JSON.stringify(pending.map((p) => p.id)),
      '6: the stored requests are the reported approvals');
    // Ungated calls ran at once, each exactly once.
    check(sameSet(r.freeRan, c.calls.filter((call) => call.tool === 'free').map((call) => call.token)) && r.freeRan.length === c.calls.filter((call) => call.tool === 'free').length,
      'ungated calls run once when the step is taken', r.freeRan);
    check(r.ran.length === 0, '1: no gated call runs before a decision', r.ran);
    approvalIds = pending.map((p) => p.id);
  } else {
    approvalIds = c.requests.map((request, index) => `apr_${c.seed}_${request.approval ?? index}`);
    const calls = c.calls.map((call, index) => ({ id: call.id, name: call.tool, arguments: (c.stringArgs.parked[index] ? JSON.stringify({ n: call.token }) : { n: call.token }) as unknown as JsonObject }));
    await r.session.thread().record([
      { type: 'user', content: 'go' },
      {
        ...storedRow(calls, c.requests.map((request, index) => ({
          approval_id: approvalIds[index]!, tool_call_id: c.calls[request.call]!.id, arguments_format: ARGUMENTS_AS_SENT,
        }))),
        harness_tool_names: ['echo', 'free'],
      },
      ...(c.prior.length === 0 ? [] : [toolResultRow(c.prior.map((index) => {
        const call = c.calls[index]!;
        // A lossy store keeps no arguments with the result: it still answers its call.
        return toolResultEntry({ id: call.id, name: call.tool, arguments: c.lossy.includes(index) ? {} : { n: call.token } }, `${call.tool === 'free' ? 'free' : 'prior'}:${call.token}`);
      }))]),
    ]);
  }

  // The decisions, in a shuffled order: the parked step's, then a later step's.
  const oldDecisions = c.order.filter((index) => c.requests[index]!.verdict !== 'none').map((index) => ({
    approvalId: approvalIds[index]!,
    approved: c.requests[index]!.verdict === 'approve',
    reason: c.requests[index]!.reason,
  }));
  const newDecisions: { approvalId: string; approved: boolean; reason: string | null }[] = [];
  c.later?.requests.forEach((request, index) => {
    const id = `apr_later_${c.seed}_${index}`;
    approvalIds.push(id);
    if (request.verdict !== 'none') newDecisions.push({ approvalId: id, approved: request.verdict === 'approve', reason: request.reason });
  });
  const recordLater = async (): Promise<void> => {
    // The later step, as another writer could record it after the park.
    const later = c.later!;
    recorded.push({ calls: later.calls.map((call, index) => ({ key: `l${index}`, id: call.id, name: call.tool, args: { n: call.token } })) });
    const laterCalls = later.calls.map((call, index) => ({ id: call.id, name: call.tool, arguments: (c.stringArgs.later[index] ? JSON.stringify({ n: call.token }) : { n: call.token }) as unknown as JsonObject }));
    await r.session.thread().record([
      {
        ...storedRow(laterCalls, later.requests.map((request, index) => ({
          approval_id: `apr_later_${c.seed}_${index}`, tool_call_id: later.calls[request.call]!.id, arguments_format: ARGUMENTS_AS_SENT,
        }))),
        harness_tool_names: ['echo', 'free'],
      },
      ...(later.calls.some((call) => call.tool === 'free') ? [toolResultRow(later.calls.filter((call) => call.tool === 'free')
        .map((call) => toolResultEntry({ id: call.id, name: call.tool, arguments: { n: call.token } }, `free:${call.token}`)))] : []),
    ]);
  };
  // An independent model of which step is superseded when: a decision for a
  // step recorded after a row that superseded it is refused
  // (approval_not_pending), on every path; any other decision is accepted.
  const supersedes = c.insert !== null && SUPERSEDES.includes(c.insert.kind);
  let laterRecorded = false;
  let oldSuperseded = false;
  let newSuperseded = false;
  const refused = async (body: () => Promise<unknown>): Promise<boolean> => body().then(() => false, (error: unknown) => {
    if ((error as { code?: string }).code === 'approval_not_pending') return true;
    throw error;
  });
  // The first resume: null when the model call after its resolution failed (failOnce).
  type Turn = { steps: RunStep[] | null; response: AgentResponse };
  const resumed = async (resume: () => Promise<Turn>): Promise<Turn | null> => {
    if (!c.failOnce) return await resume();
    let threw = false;
    try { await resume(); } catch { threw = true; }
    check(threw, 'the scripted model failure reaches the caller');
    return null;
  };
  let first: Turn | null = null;
  let resumedYet = false;
  if (c.copy !== null) {
    // Order B: the copy lands before the person decides.
    const rowsNow = await rows(r.session);
    const parkedRow = [...rowsNow].reverse().find((row) => isStepRow(row))!;
    const target = c.calls.find((call) => call.tool === 'echo')!;
    const tool_calls = copyCalls(c.copy.kind, target);
    await r.session.thread().record([{ ...parkedRow, tool_calls }]);
    recorded.push({ calls: tool_calls.filter((entry): entry is JsonObject => typeof entry === 'object' && entry !== null && !Array.isArray(entry))
      .map((entry, index) => ({ key: `k${index}`, id: String(entry.id), name: 'echo', args: entry.arguments as JsonObject })) });
    oldSuperseded = true;
  }
  const plan = actions(c);
  for (const [at, action] of plan.entries()) {
    if (action === 'insert') {
      const row = insertedRow(c);
      await r.session.thread().record([row]);
      if (c.insert!.kind === 'assistant_call') recorded.push({ calls: (row.tool_calls as JsonObject[]).map((call) => ({ key: 'x0', id: String(call.id), name: 'free', args: { n: 900 } })) });
      if (supersedes) {
        if (laterRecorded) newSuperseded = true;
        else oldSuperseded = true;
      }
      continue;
    }
    if (action === 'later') {
      await recordLater();
      laterRecorded = true;
      oldSuperseded = true;
      continue;
    }
    const batch = action === 'old' ? oldDecisions : newDecisions;
    if (batch.length === 0) continue;
    const expectRefused = action === 'old' ? oldSuperseded : newSuperseded;
    // A stream records and resumes at once, so it is used only for the last
    // thing the case does before its resume.
    const last = plan.slice(at + 1).every((next) => (next === 'old' ? oldDecisions : next === 'new' ? newDecisions : [1]).length === 0);
    if (c.recorder === 'streamDecision' && last && !resumedYet && c.junk === null) {
      let run: Turn | null = null;
      // A refused stream fails before it records or resumes anything, so it is
      // not the resume a scripted model failure is expected on.
      const no = await refused(async () => {
        const stream = async (): Promise<Turn> => {
          run = await drain(r.agent.streamDecision(r.session, batch));
          return run;
        };
        first = expectRefused ? await stream() : await resumed(stream);
      });
      check(no === expectRefused, 'G-397: a decision is refused exactly when its step was superseded first', { action, no, expectRefused });
      if (!no) {
        resumedYet = true;
        if (run !== null) {
          const recorded = (run as Turn).steps!.filter((step) => step.type === 'decision_recorded');
          check(JSON.stringify(recorded.map((step) => [step.approvalId, step.verdict])) === JSON.stringify(batch.map((d) => [d.approvalId, d.approved ? 'approved' : 'denied'])),
            '6: one decision_recorded step per decision, as given');
        }
      }
    } else if (c.recorder === 'recordDecisions') {
      const no = await refused(() => recordDecisions(r.session, batch));
      check(no === expectRefused, 'G-397: a decision is refused exactly when its step was superseded first', { action, no, expectRefused });
    } else {
      for (const d of batch) {
        const no = await refused(() => recordApproval(r.session, d.approvalId, d.approved, d.reason));
        check(no === expectRefused, 'G-397: a decision is refused exactly when its step was superseded first', { action, no, expectRefused });
      }
    }
  }
  if (c.junk !== null && !resumedYet) {
    const named = c.later?.calls[0]?.id ?? c.calls[0]!.id;
    const requests = c.junk.requests
      ? (c.later === null ? c.requests.map((q, index) => ({ approval_id: approvalIds[index]!, tool_call_id: c.calls[q.call]!.id, arguments_format: ARGUMENTS_AS_SENT }))
        : c.later.requests.map((q, index) => ({ approval_id: `apr_later_${c.seed}_${index}`, tool_call_id: c.later!.calls[q.call]!.id, arguments_format: ARGUMENTS_AS_SENT })))
      : [];
    await r.session.thread().record([{ type: 'assistant', content: '', tool_calls: junkCalls(c.junk.kind, named), additional_content: {}, tool_approval_requests: requests, harness_tool_names: ['echo', 'free'] }]);
    if (c.junk.kind === 'mixed') recorded.push({ calls: [{ key: 'x1', id: named, name: 'free', args: { n: 901 } }] });
  }
  if (!resumedYet) first = await resumed(async () => await turn(r, c.resume, ''));

  // Invariants 1-4, then full equality with the oracle.
  // Which call ran: a gated call of the step that can still run, by its
  // arguments (unique among that step's gated calls).
  const decidedCalls = c.later !== null ? c.later.calls.map((call, index) => ({ call, key: `l${index}` })) : c.calls.map((call, index) => ({ call, key: `p${index}` }));
  const ranKeys = r.ran.map((token) => decidedCalls.find(({ call }) => call.tool === 'echo' && call.token === token)?.key ?? `?${token}`);
  check(ranKeys.length === new Set(ranKeys).size, '3: a call runs at most once', ranKeys);
  check(ranKeys.every((key) => !key.startsWith('?')), 'a replaced step\'s approvals never run anything', ranKeys);
  const all = [...c.requests.map((q) => ({ q, key: `p${q.call}` })), ...(c.later?.requests ?? []).map((q) => ({ q, key: `l${q.call}` }))];
  const denied = new Set(all.filter(({ q }) => q.verdict === 'deny').map(({ key }) => key));
  check(ranKeys.every((key) => !denied.has(key)), '1: a call a person denied never runs', ranKeys);
  const approves = all.filter(({ q }) => q.verdict === 'approve');
  check(ranKeys.every((key) => approves.some((a) => a.key === key)), '2: each run call has its own approve', ranKeys);
  check(ranKeys.length <= approves.length, '2: one decision decides at most one call', { ran: ranKeys.length, approves: approves.length });
  if (oracle.repeated) check(ranKeys.length === 0, '4: a step with repeated ids runs none of its gated calls', ranKeys);
  check(JSON.stringify([...ranKeys].sort()) === JSON.stringify([...oracle.runs].sort()), 'the calls that ran are exactly the oracle\'s', { ran: ranKeys, expected: [...oracle.runs] });

  // Invariant 6 on the resume itself: what it says it ran is what ran.
  const firstTurn = first as Turn | null;
  if (firstTurn !== null) check(firstTurn.response.toolCalls.filter((name) => name === 'echo').length === r.ran.length, '6: the resume reports the calls it ran', firstTurn.response.toolCalls);
  if (firstTurn !== null && firstTurn.steps !== null) {
    const results = firstTurn.steps.filter((step) => step.type === 'tool_result');
    const ok = results.flatMap((step) => (step.ok ? [Number(step.result.replace(/^ran:/, ''))] : []));
    check(sameSet(ok, r.ran) && ok.length === r.ran.length, '6: the ok tool_result steps are the calls that ran', ok);
    // One step per open gated call of the step that is decided (none when
    // every step was superseded): no result of its own yet.
    check(results.length === oracle.open.length, '6: one tool_result step per open gated call', { steps: results.length, open: oracle.open.length });
  }

  await settledOutcomes(r, recorded, oracle, '5');
  await replayKeepsEveryResult(r.session, '6');

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
  await settledOutcomes(r, recorded, oracle, '5 (after repeat resumes)');
  await replayKeepsEveryResult(r.session, '6 (after repeat resumes)');
}

/** Invariants 5 and 6 on the thread rows: every call has exactly one outcome, the oracle's. */
async function settledOutcomes(r: Rig, recorded: RecordedStep[], oracle: Expected, label: string): Promise<void> {
  const seen = await attributed(r.session, recorded);
  for (const [key, want] of oracle.outcome) {
    const got = [...(seen.get(key) ?? [])];
    check(got.length === 1, `${label}: every call gets exactly one recorded outcome`, { key, got });
    check(got[0] === want, `6: the recorded outcome matches what ran`, { key, got: got[0], want });
  }
  // An earlier answered step's calls and an inserted ungated call are not the case's.
  check([...seen.keys()].every((key) => oracle.outcome.has(key) || key.startsWith('h') || key.startsWith('x')), `${label}: no outcome for a call that is not in the step`, [...seen.keys()]);
}

/**
 * Invariant 6 on the replay the model is shown (G-402): in each step's
 * segment, every result the thread stores is still there in the replay view,
 * by its id, tool name and arguments, as often as it occurs in one row. A
 * later row replaces only the same result (same id, name and arguments,
 * occurrence by occurrence), never another call's that shares its id.
 */
async function replayKeepsEveryResult(session: Session, label: string): Promise<void> {
  const stored = await rows(session);
  const isStep = isStepRow;
  const segments = (list: JsonObject[], view: boolean): Map<string, number>[] => {
    const out: Map<string, number>[] = [];
    for (const row of list) {
      if (isStep(row)) { out.push(new Map()); continue; }
      if (out.length === 0 || row.type !== 'tool_result' || row.harness_replaced_step !== undefined) continue;
      // Per row as stored, the most often a result occurs in one row; per view row, as shown.
      const here = new Map<string, number>();
      for (const entry of (row.tool_results as JsonObject[] | undefined) ?? []) {
        const key = canonical([entry.tool_call_id, entry.tool_name, entry.args]);
        here.set(key, (here.get(key) ?? 0) + 1);
      }
      const segment = out.at(-1)!;
      for (const [key, count] of here) segment.set(key, view ? (segment.get(key) ?? 0) + count : Math.max(segment.get(key) ?? 0, count));
    }
    return out;
  };
  const want = segments(stored, false);
  const got = segments(threadView(stored), true);
  want.forEach((segment, at) => {
    for (const [key, count] of segment) {
      check((got[at]?.get(key) ?? 0) >= count, `${label}: the replay view keeps every result of a step, by id, name and arguments`, { step: at, key, stored: count, shown: got[at]?.get(key) ?? 0 });
    }
  });
  // And emits each exactly once: no result occurrence (id, name, arguments)
  // shown more often in a step's segment than it occurs in one stored row,
  // across every row of the replay (some providers refuse a repeated result).
  got.forEach((segment, at) => {
    for (const [key, count] of segment) {
      check(count <= (want[at]?.get(key) ?? 0), `${label}: the replay view emits each result of a step exactly once`, { step: at, key, stored: want[at]?.get(key) ?? 0, shown: count });
    }
  });
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
  const requests = keptRequests.map(({ request }) => ({ ...request, call: remap(request.call), approval: request.approval === undefined ? undefined : position.get(request.approval) }))
    .map((request, at) => (request.approval === undefined || request.approval >= at ? { ...request, approval: undefined } : request));
  return {
    ...c,
    calls,
    requests,
    prior: c.prior.filter((index) => index !== drop).map(remap),
    lossy: c.lossy.filter((index) => index !== drop).map(remap),
    stringArgs: { ...c.stringArgs, parked: c.stringArgs.parked.filter((_v, index) => index !== drop) },
    history: c.history === null ? null : (() => {
      const kept = c.history.filter((h) => h.call !== drop).map((h) => ({ ...h, call: remap(h.call) }));
      return kept.length === 0 ? null : kept;
    })(),
    insert: c.insert === null || c.insert.step === 'later' ? c.insert
      : c.insert.call === drop ? null : { ...c.insert, call: remap(c.insert.call) },
    order: c.order.filter((index) => position.has(index)).map((index) => position.get(index)!),
  };
}

function withoutRequest(c: Case, drop: number): Case | null {
  if (c.source === 'model') return null;
  return {
    ...c,
    requests: c.requests.filter((_request, index) => index !== drop).map((request) => (request.approval === undefined ? request
      : request.approval === drop ? { ...request, approval: undefined } : { ...request, approval: request.approval > drop ? request.approval - 1 : request.approval })),
    order: c.order.filter((index) => index !== drop).map((index) => (index > drop ? index - 1 : index)),
  };
}

function smaller(c: Case): Case[] {
  const out: Case[] = [];
  if (c.again.length > (c.failOnce ? 1 : 0)) out.push({ ...c, again: c.again.slice(0, -1) });
  if (c.insert !== null) out.push({ ...c, insert: null });
  if (c.history !== null) out.push({ ...c, history: null });
  if (c.junk !== null) out.push({ ...c, junk: null });
  if (c.copy !== null) out.push({ ...c, copy: null });
  if (c.failOnce) out.push({ ...c, failOnce: false });
  if (c.spent) out.push({ ...c, spent: false });
  if (c.later !== null) out.push({ ...c, later: null });
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
      : [REPLAY];
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

  it('a property setting that is set must be a valid integer: never a silent zero or a default', () => {
    const name = 'PRISM_PROPERTY_TEST_SETTING';
    const saved = process.env[name];
    try {
      delete process.env[name];
      expect(envInteger(name, 1, 100, 7)).toBe(7);
      for (const bad of ['', '0', '-1', '+3', '1.5', '1e3', ' 5', 'abc', '101', '0x10']) {
        process.env[name] = bad;
        expect(() => envInteger(name, 1, 100, 7), JSON.stringify(bad)).toThrow(`${name} must be an integer from 1 to 100 when set`);
      }
      process.env[name] = '42';
      expect(envInteger(name, 1, 100, 7)).toBe(42);
    } finally {
      if (saved === undefined) delete process.env[name]; else process.env[name] = saved;
    }
  });

  it('the generator reaches every shape the brief asks for', () => {
    const cases = Array.from({ length: DEFAULT_EXAMPLES }, (_unused, index) => generate(caseSeed(FIXED_BASE_SEED, index)));
    const has = (predicate: (c: Case) => boolean): boolean => cases.some(predicate);
    expect(has((c) => expected(c).repeated && c.source === 'model')).toBe(true);
    expect(has((c) => !expected(c).repeated && expected(c).runs.size > 0)).toBe(true);
    expect(has((c) => c.calls.filter((call) => call.id === '').length > 1)).toBe(true);
    expect(has((c) => c.source === 'stored' && new Set(c.requests.map((q) => q.call)).size < c.requests.length)).toBe(true);
    expect(has((c) => c.source === 'stored' && c.prior.some((index) => c.calls[index]!.tool === 'echo'))).toBe(true);
    expect(has((c) => c.again.length > 0 && c.again.some((a) => a.approveAll))).toBe(true);
    for (const path of PATHS) expect(has((c) => c.resume === path)).toBe(true);
    expect(has((c) => c.recorder === 'streamDecision')).toBe(true);
    // G-397 / G-399: every inserted row kind, before and after the decision (main's shape check).
    for (const kind of INSERT_KINDS) {
      for (const when of ['before_decision', 'after_decision'] as const) expect(has((c) => c.insert?.kind === kind && c.insert.when === when), `${kind} ${when}`).toBe(true);
      expect(has((c) => c.insert?.kind === kind && decideStep('p', c.calls, c.requests, c.prior).repeated && c.requests.some((q) => q.verdict === 'approve')), `${kind} in a repeated-id step`).toBe(true);
    }
    expect(has((c) => c.insert !== null && c.recorder === 'streamDecision'), 'insert + streamDecision').toBe(true);
    // G-399: a row that does not supersede, after the decision, a resume with no assistant row (steps spent), then repeat resumes.
    expect(has((c) => (c.insert?.kind === 'system' || c.insert?.kind === 'tool_empty') && c.insert.when === 'after_decision' && c.spent && expected(c).runs.size > 0 && c.again.length > 0), 'G-399 spent').toBe(true);
    // A superseding row after an approval: nothing runs.
    expect(has((c) => c.insert !== null && SUPERSEDES.includes(c.insert.kind) && c.insert.when === 'after_decision' && c.requests.some((q) => q.verdict === 'approve')), 'superseded after an approve').toBe(true);
    // G-397: a later step combined with every inserted row kind, before and after the decision, in either step's segment.
    for (const kind of INSERT_KINDS) {
      for (const when of ['before_decision', 'after_decision'] as const) {
        for (const segment of ['earlier', 'later'] as const) {
          expect(has((c) => c.later !== null && c.insert?.kind === kind && c.insert.when === when && c.insert.segment === segment), `later + ${kind} ${when} ${segment}`).toBe(true);
        }
      }
    }
    // Forged results: in the parked step's segment naming its call; in the later step's segment naming a call of either step.
    expect(has((c) => c.later !== null && c.insert?.kind === 'tool_result' && c.insert.segment === 'earlier' && c.insert.step === 'earlier'), 'forged, earlier segment').toBe(true);
    expect(has((c) => c.later !== null && c.insert?.kind === 'tool_result' && c.insert.segment === 'later' && c.insert.step === 'later'), 'forged, later segment, later call').toBe(true);
    expect(has((c) => c.later !== null && c.insert?.kind === 'tool_result' && c.insert.segment === 'later' && c.insert.step === 'earlier'), 'forged, later segment, earlier call').toBe(true);
    expect(has((c) => c.later !== null && c.insert?.kind === 'tool_result' && c.insert.step === 'later' && decideStep('l', c.later.calls, c.later.requests, []).repeated), 'forged in a later step that repeats ids').toBe(true);
    // G-408: rows whose tool_calls hold no objects, each shape, with and without requests, after an approval.
    for (const kind of ['string', 'number', 'null', 'strings', 'scalar', 'mixed'] as const) {
      for (const requests of [true, false]) {
        expect(has((c) => c.junk?.kind === kind && c.junk.requests === requests && expected(c).outcome.size > 0), `junk ${kind} requests ${requests}`).toBe(true);
      }
    }
    expect(has((c) => c.junk !== null && c.junk.requests && c.spent && c.requests.some((q) => q.verdict === 'approve')), 'junk with requests on a spent run after an approve').toBe(true);
    // G-408, order B: each copy shape, approved after the copy landed, on a spent run and with a model failure after the resolution.
    for (const kind of COPY_KINDS) {
      expect(has((c) => c.copy?.kind === kind && c.requests.some((q) => q.verdict === 'approve' && c.calls[q.call]!.tool === 'echo')), `copy ${kind} approved`).toBe(true);
    }
    expect(has((c) => c.copy !== null && c.spent && c.requests.some((q) => q.verdict === 'approve')), 'copy, spent').toBe(true);
    expect(has((c) => c.copy !== null && c.failOnce && c.requests.some((q) => q.verdict === 'approve')), 'copy, model fails after the resolution').toBe(true);
    // Two identical gated calls (same id, tool and arguments) in one step, with an approval, and on a spent run.
    expect(has((c) => c.calls.some((call, i) => call.tool === 'echo' && c.calls.some((other, j) => j > i && other.tool === 'echo' && other.id === call.id && other.token === call.token))
      && c.requests.some((q) => q.verdict === 'approve')), 'identical gated calls').toBe(true);
    expect(has((c) => c.spent && c.calls.some((call, i) => call.tool === 'echo' && c.calls.some((other, j) => j > i && other.tool === 'echo' && other.id === call.id && other.token === call.token))), 'identical gated calls, spent').toBe(true);
    // G-402: a step with repeated ids whose results carry different arguments, with a later result row in the same segment.
    expect(has((c) => c.calls.some((call, i) => c.calls.some((other, j) => j !== i && other.id === call.id && other.token !== call.token
      && (call.tool === 'free' || c.prior.includes(i))))), 'repeated ids, results with different arguments').toBe(true);
    // G-397, absolute segment ownership: results recorded with arguments that differ from their call's own; identical arguments across calls.
    expect(has((c) => c.lossy.length > 0 && c.requests.some((q) => c.lossy.includes(q.call) && q.verdict === 'approve')), 'lossy result for an approved call').toBe(true);
    expect(has((c) => c.calls.some((call) => call.tool === 'free' && c.calls.some((other) => other.tool === 'echo' && other.token === call.token))), 'identical arguments within a step').toBe(true);
    expect(has((c) => c.later !== null && c.later.calls.some((call) => c.calls.some((old) => old.id === call.id && old.token === call.token))), 'a later call with an earlier call\'s id and arguments').toBe(true);
    expect(has((c) => c.later !== null && c.insert?.kind === 'tool_result' && c.insert.segment === 'later' && c.insert.step === 'earlier'
      && c.later.calls.some((call) => call.id === c.calls[c.insert!.call]!.id && call.tool === 'echo')), 'a result naming an earlier call in the later segment, answering a later call').toBe(true);
    // G-397: arguments stored as JSON strings, and an earlier answered step reusing the parked step's ids.
    expect(has((c) => c.source === 'stored' && c.stringArgs.parked.some((v, i) => v && c.calls[i]!.tool === 'echo') && c.requests.some((q) => q.verdict === 'approve')), 'string-stored arguments, approved').toBe(true);
    expect(has((c) => c.later !== null && c.stringArgs.later.some((v, i) => v && c.later!.calls[i]!.tool === 'echo')), 'string-stored arguments in a later step').toBe(true);
    expect(has((c) => c.history !== null && c.history.some((h) => h.same) && expected(c).runs.size > 0), 'earlier step, same id and arguments, an approved call').toBe(true);
    expect(has((c) => c.history !== null && c.history.some((h) => !h.same)), 'earlier step, same id, other arguments').toBe(true);
    expect(has((c) => c.history !== null && c.source === 'stored' && c.history.some((h) => h.same && c.stringArgs.parked[h.call] && c.requests.some((q) => q.call === h.call && q.verdict === 'approve'))
      && (c.spent || c.failOnce)), 'the parked call stored as a string, an earlier step with its id and arguments, no assistant row after the resolution').toBe(true);
    // G-397: a later tool-call step replaces the parked one, with approvals for both.
    expect(has((c) => c.later !== null && c.requests.some((q) => q.verdict === 'approve') && c.later.requests.some((q) => q.verdict === 'approve')), 'later step').toBe(true);
    expect(has((c) => c.later !== null && decideStep('l', c.later.calls, c.later.requests, []).repeated), 'later step repeating ids').toBe(true);
    expect(has((c) => c.later !== null && c.later.calls.some((call) => c.calls.some((old) => old.id === call.id))), 'later step reusing an old id').toBe(true);
    // G-397: stored requests that repeat an APPROVAL id, with an approval recorded.
    expect(has((c) => c.requests.some((q) => q.approval !== undefined) && c.requests.some((q) => q.verdict === 'approve')), 'repeated approval id').toBe(true);
  });
});
