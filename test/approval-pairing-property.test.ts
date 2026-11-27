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
import { ARGUMENTS_AS_SENT, assistantRow, threadView, toolResultEntry, toolResultRow } from '../src/thread-rows.js';
import { canonicalJson } from '../src/tool-policy.js';

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
// The replaced-step denial: the PHP reference's text, as every main writes it
// (harness-approval-pairing apr-0016..0019).
const REPLACED = 'Tool call was not run because a later step replaced it';
const DENIED = 'User denied tool execution';

/** The fixed-seed set normal CI runs; the coverage check below is asserted on exactly these seeds. */
const DEFAULT_EXAMPLES = 500;
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
}

type InsertKind = 'assistant_text' | 'assistant_call' | 'user' | 'new_message' | 'system' | 'tool_empty' | 'tool_result';
const INSERT_KINDS: readonly InsertKind[] = ['assistant_text', 'assistant_call', 'user', 'new_message', 'system', 'tool_empty', 'tool_result'];

interface Insert {
  kind: InsertKind;
  when: 'before_decision' | 'after_decision';
  /** assistant_call: the step call whose id the inserted (ungated) call reuses; tool_result: the gated call the forged result names. */
  call: number;
  /**
   * With a later step (NO-BLIND-SPOT): `earlier` is recorded between the
   * park and the later step, and its call indexes the parked step's calls;
   * `later` is recorded after the later step, and indexes the later step's.
   */
  segment?: 'earlier' | 'later';
  /**
   * SEGMENT OWNERSHIP (round 13b): a forged result in the LATER segment that
   * names an EARLIER step's gated call (`call` indexes the parked step's
   * calls), with that call's id, tool and EXACT arguments. It is lossy for
   * any later call sharing the id: it answers the first such open call of
   * its own segment's step, or nothing, and never the earlier call.
   */
  cross?: boolean;
}

/**
 * Kinds that SUPERSEDE the parked step when the host records them outside
 * the run that owns it (G-397): an assistant row with tool calls always; an
 * assistant text row or a user row recorded outside that run. system and
 * tool rows never do.
 */
const SUPERSEDES: readonly InsertKind[] = ['assistant_text', 'assistant_call', 'user', 'new_message'];

/** The G-397 row, from a generator of its own (the seed's other draws are unchanged). */
function insertion(seed: number, calls: Call[], prior: number[], failOnce: boolean): Insert | null {
  const rng = new Rng((seed ^ 0x397397) >>> 0);
  if (!rng.chance(0.6)) return null;
  let kind = rng.pick<InsertKind>(['assistant_text', 'assistant_call', 'user', 'new_message', 'system', 'system', 'tool_empty', 'tool_result']);
  // A new message calls the model: never before a resume scripted to fail.
  if (kind === 'new_message' && failOnce) kind = 'user';
  // A forged result names a gated call that has none yet.
  const open = calls.flatMap((call, index) => (call.tool === 'echo' && !prior.includes(index) ? [index] : []));
  if (kind === 'tool_result' && open.length === 0) kind = 'system';
  const when = rng.chance(0.5) ? 'before_decision' : 'after_decision';
  return { kind, when, call: kind === 'tool_result' ? rng.pick(open) : rng.int(0, calls.length - 1) };
}

/**
 * NO-BLIND-SPOT: a row inserted with a LATER step, in the earlier segment
 * (between the park and the later step) or the later one (after it), from a
 * generator of its own (seed ^ 0x397398). A forged result names an open
 * gated call of its own segment's step.
 */
function laterInsertion(seed: number, calls: Call[], later: LaterStep, failOnce: boolean): Insert | null {
  const rng = new Rng((seed ^ 0x397398) >>> 0);
  if (!rng.chance(0.85)) return null;
  let kind = rng.pick<InsertKind>(['assistant_text', 'assistant_call', 'user', 'new_message', 'system', 'tool_empty', 'tool_result', 'tool_result']);
  if (kind === 'new_message' && failOnce) kind = 'user';
  const segment = rng.chance(0.5) ? 'earlier' : 'later';
  const own = segment === 'earlier' ? calls : later.calls;
  const gated = own.flatMap((call, index) => (call.tool === 'echo' ? [index] : []));
  const when = segment === 'earlier' ? 'before_decision' : rng.chance(0.5) ? 'before_decision' : 'after_decision';
  const insert: Insert = { kind, when, segment, call: kind === 'tool_result' ? rng.pick(gated) : rng.int(0, own.length - 1) };
  // SEGMENT OWNERSHIP (round 13b), from a generator of its own (seed ^
  // 0x13b13b): some later steps get, instead, a forged result in the LATER
  // segment that names an EARLIER gated call with its exact arguments,
  // half the time one whose id a later gated call shares.
  const crossRng = new Rng((seed ^ 0x13b13b) >>> 0);
  if (crossRng.chance(0.35)) {
    const earlier = calls.flatMap((call, index) => (call.tool === 'echo' ? [index] : []));
    const shared = earlier.filter((index) => later.calls.some((call) => call.tool === 'echo' && call.id === calls[index]!.id));
    const pool = shared.length > 0 && crossRng.chance(0.5) ? shared : earlier;
    return { kind: 'tool_result', when: crossRng.chance(0.5) ? 'before_decision' : 'after_decision', segment: 'later', cross: true, call: crossRng.pick(pool) };
  }
  return insert;
}

/** Record the inserted row as a host would, or send a new message through the harness. */
async function insertRow(r: Rig, c: Case): Promise<void> {
  if (c.insert!.kind === 'new_message') await r.agent.send(r.session, 'Something else.');
  else await r.session.thread().record([insertedRow(c)]);
}

/** The step whose calls an inserted row names: the later step's, for one recorded after it. */
function insertCalls(c: Case): Call[] {
  return c.insert?.segment === 'later' && c.insert.cross !== true ? c.later!.calls : c.calls;
}

/**
 * The call a forged result is the outcome of, by SEGMENT OWNERSHIP: a call
 * of the step whose segment it sits in, matched by exact arguments, then id
 * and tool, then id, among that step's calls with no result yet. Null for no
 * forged result, or one that answers nothing.
 */
function forgedTarget(c: Case): { step: 'earlier' | 'later'; index: number; text: string } | null {
  const insert = c.insert;
  if (insert?.kind !== 'tool_result') return null;
  const named = insertCalls(c)[insert.call]!;
  const text = `forged:${named.token}`;
  if (insert.cross !== true) return { step: insert.segment === 'later' ? 'later' : 'earlier', index: insert.call, text };
  // Exact arguments never match a later call (the tokens differ); the first
  // later call with the id and tool (its ungated calls already have results).
  const index = c.later!.calls.findIndex((call) => call.id === named.id && call.tool === named.tool);
  return index === -1 ? null : { step: 'later', index, text };
}

/** The row G-397 inserts. Its own call's token (900) is never a step call's. */
function insertedRow(c: Case): JsonObject {
  const insert = c.insert!;
  const call = insertCalls(c)[insert.call]!;
  switch (insert.kind) {
    case 'assistant_text': return { type: 'assistant', content: 'A note.', tool_calls: [], tool_approval_requests: [] };
    case 'assistant_call': return { type: 'assistant', content: '', tool_calls: [{ id: call.id, name: 'free', arguments: { n: 900 } }], tool_approval_requests: [] };
    case 'user': case 'new_message': return { type: 'user', content: 'Something else.' };
    case 'system': return { type: 'system', content: 'A system note.' };
    case 'tool_empty': return toolResultRow([]);
    case 'tool_result': return toolResultRow([toolResultEntry({ id: call.id, name: call.tool, arguments: { n: call.token } }, `forged:${call.token}`)]);
  }
}

/** The parked step is superseded: by the inserted row, or by a later tool-call step. */
function superseded(c: Case): boolean {
  return c.later !== null || (c.insert !== null && SUPERSEDES.includes(c.insert.kind));
}

/** The LATER step is superseded too: by a superseding row recorded after it. */
function laterSuperseded(c: Case): boolean {
  return c.later !== null && c.insert?.segment === 'later' && SUPERSEDES.includes(c.insert.kind);
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
  if (source === 'model' && rng.chance(0.25)) {
    const laterCalls: Call[] = [];
    for (let i = rng.int(1, 3); i > 0; i -= 1) laterCalls.push({ id: rng.pick(ID_POOL), tool: rng.chance(0.7) ? 'echo' : 'free', token: 200 + laterCalls.length });
    if (!laterCalls.some((call) => call.tool === 'echo')) laterCalls[0]!.tool = 'echo';
    later = { calls: laterCalls, requests: laterCalls.flatMap((call, index) => (call.tool === 'echo' ? [{ ...verdict(), call: index }] : [])) };
  }
  // A resume that failed after its resolution is always followed by another.
  if (failOnce && again.length === 0) again.push({ approveAll: rng.chance(0.5), path: rng.pick(PATHS) });

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
    // NO-BLIND-SPOT: a later step is combined with inserted rows too.
    insert: later === null ? insertion(seed, calls, prior, failOnce) : laterInsertion(seed, calls, later, failOnce),
    failOnce,
    spent,
    later,
  };
}

// ---------------------------------------------------------------------------
// The oracle: an independent model of the rule.
// ---------------------------------------------------------------------------

interface Expected {
  repeated: boolean;
  /** Tokens of the gated calls that run, on the first resume and never again. */
  runs: Set<number>;
  /** The one outcome every call ends with, by token. */
  outcome: Map<number, string>;
}

function expected(c: Case): Expected {
  // SEGMENT OWNERSHIP (round 13b): a forged result is the outcome of a call
  // of ITS OWN segment's step only (forgedTarget), never an earlier step's.
  const target = forgedTarget(c);
  const forged = target?.step === 'earlier' ? target.index : -1;
  const forgedLater = target?.step === 'later' ? target.index : -1;
  const forgedText = target?.text ?? '';
  const answered = [...c.prior, ...(forged === -1 ? [] : [forged])];
  if (!superseded(c)) {
    const decided = decideStep(c.calls, c.requests, answered);
    if (forged !== -1) decided.outcome.set(c.calls[forged]!.token, forgedText);
    return decided;
  }
  // Superseded (G-397, the release lines' rule): each gated call of the
  // parked step with no result of its own gets ONE explicit denial, and
  // nothing of it runs; a later tool-call step is decided by its own.
  // A later step superseded in its turn gives its open calls the same one
  // denial, and runs nothing.
  const decided = c.later === null
    ? { repeated: false, runs: new Set<number>(), outcome: new Map<number, string>() }
    : laterSuperseded(c)
      ? { repeated: false, runs: new Set<number>(), outcome: new Map(c.later.calls.map((call, index) => [call.token,
        call.tool === 'free' ? `free:${call.token}` : index === forgedLater ? forgedText : REPLACED])) }
      : decideStep(c.later.calls, c.later.requests, forgedLater === -1 ? [] : [forgedLater]);
  if (c.later !== null && forgedLater !== -1) decided.outcome.set(c.later.calls[forgedLater]!.token, forgedText);
  // The parked step's calls: an earlier call that a later-segment result
  // names still gets the replaced denial (segment ownership).
  c.calls.forEach((call, index) => decided.outcome.set(call.token, call.tool === 'free' ? `free:${call.token}`
    : c.prior.includes(index) ? `prior:${call.token}` : index === forged ? forgedText : REPLACED));
  return decided;
}

function decideStep(calls: Call[], requests: Request[], prior: number[]): Expected {
  const dup = (ids: (string | number)[]): boolean => new Set(ids).size !== ids.length;
  const repeated = dup(calls.map((call) => call.id)) || dup(requests.map((request) => calls[request.call]!.id))
    || dup(requests.map((request, index) => request.approval ?? index));
  const runs = new Set<number>();
  const outcome = new Map<number, string>();
  calls.forEach((call, index) => {
    if (call.tool === 'free') {
      outcome.set(call.token, `free:${call.token}`);
      return;
    }
    if (prior.includes(index)) {
      outcome.set(call.token, `prior:${call.token}`);
      return;
    }
    if (repeated) {
      outcome.set(call.token, REPEATED);
      return;
    }
    // Distinct ids: this call's own request, if it has one.
    const own = requests.find((request) => request.call === index);
    if (own?.verdict === 'approve') {
      runs.add(call.token);
      outcome.set(call.token, `ran:${call.token}`);
    } else if (own?.verdict === 'deny') {
      outcome.set(call.token, own.reason ?? DENIED);
    } else {
      outcome.set(call.token, NO_RESPONSE);
    }
  });

  return { repeated, runs, outcome };
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
      // G-405: an ANTHROPIC-shaped step keeps the model's text beside its
      // tool calls; on a structured turn that text is a bad-typed document
      // (and a client that parsed it hands it over), which a parked
      // structured response must never return.
      const toolCalls = c.calls.map((call) => ({ id: call.id, name: call.tool, arguments: { n: call.token } }));
      return request.schema === undefined
        ? { text: '', finishReason: 'tool_calls', toolCalls }
        : { text: '{"ok":"not a boolean"}', structured: { ok: 'not a boolean' }, finishReason: 'tool_calls', toolCalls };
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

/**
 * Every stored result entry after the first step, as [token, result], each
 * attributed BY SEGMENT OWNERSHIP (round 13b), independently of the code
 * under test: a result belongs to a call of the step whose segment it sits
 * in (after that step's assistant row, before the next assistant row with
 * tool calls), matched by exact arguments, then id and tool, then id, among
 * the calls with no result yet (an exact copy of one already read is the
 * same result). A result that matches no call is nobody's
 * outcome; so is one for the inserted row's own call (token 900).
 */
async function outcomes(session: Session): Promise<[number, string][]> {
  const out: [number, string][] = [];
  let calls: JsonObject[] = [];
  let has: boolean[] = [];
  const seen = new Set<string>();
  for (const row of await rows(session)) {
    if (row.type === 'assistant' && Array.isArray(row.tool_calls) && row.tool_calls.length > 0) {
      calls = row.tool_calls as JsonObject[];
      has = calls.map(() => false);
      seen.clear();
      continue;
    }
    if (row.type !== 'tool_result') continue;
    for (const entry of (row.tool_results ?? []) as JsonObject[]) {
      // The resolution row carries the step's earlier results with it: an
      // exact copy of a result already read is that result, not another.
      const copy = JSON.stringify([entry.tool_call_id, entry.tool_name, entry.args, entry.result]);
      if (seen.has(copy)) continue;
      seen.add(copy);
      const open = (at: number): boolean => !has[at] && calls[at]!.id === entry.tool_call_id;
      let at = calls.findIndex((call, index) => open(index) && call.name === entry.tool_name && JSON.stringify(call.arguments) === JSON.stringify(entry.args));
      if (at === -1) at = calls.findIndex((call, index) => open(index) && call.name === entry.tool_name);
      if (at === -1) at = calls.findIndex((_call, index) => open(index));
      if (at === -1) continue;
      has[at] = true;
      const token = Number((calls[at]!.arguments as JsonObject).n);
      if (token !== 900) out.push([token, String(entry.result)]);
    }
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

async function runCase(c: Case): Promise<void> {
  const oracle = expected(c);
  const r = rig(c);
  const gatedTokens = c.calls.filter((call) => call.tool === 'echo').map((call) => call.token);
  let approvalIds: string[];

  if (c.source === 'model') {
    const parked = await turn(r, c.initial, 'go');
    check(parked.response.finishReason === 'awaiting_approval', 'the step parks for a person', parked.response.finishReason);
    // G-405: a parked structured turn carries NO document, whatever the step's text held.
    if (c.initial === 'structured') check((parked.response as { structured?: unknown }).structured === null, 'G-405: a parked structured response has no document', (parked.response as { structured?: unknown }).structured);
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
    if (c.later !== null) {
      // NO-BLIND-SPOT: a row in the earlier segment, between the park and the later step.
      if (c.insert?.segment === 'earlier') await insertRow(r, c);
      // The later step, as another writer could record it after the park.
      const laterCalls = c.later.calls.map((call) => ({ id: call.id, name: call.tool, arguments: { n: call.token } }));
      await r.session.thread().record([
        {
          ...assistantRow('', laterCalls, {}, c.later.requests.map((request, index) => ({
            approval_id: `apr_later_${c.seed}_${index}`, tool_call_id: c.later!.calls[request.call]!.id, arguments_format: ARGUMENTS_AS_SENT,
          }))),
          harness_tool_names: ['echo', 'free'],
        },
        ...(c.later.calls.some((call) => call.tool === 'free') ? [toolResultRow(c.later.calls.filter((call) => call.tool === 'free')
          .map((call) => toolResultEntry({ id: call.id, name: call.tool, arguments: { n: call.token } }, `free:${call.token}`)))] : []),
      ]);
    }
  } else {
    approvalIds = c.requests.map((request, index) => `apr_${c.seed}_${request.approval ?? index}`);
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

  // The decisions, in a shuffled order.
  const decisions = c.order.filter((index) => c.requests[index]!.verdict !== 'none').map((index) => ({
    approvalId: approvalIds[index]!,
    approved: c.requests[index]!.verdict === 'approve',
    reason: c.requests[index]!.reason,
  }));
  // A later step's decisions follow; the replaced step's decisions are still recorded.
  c.later?.requests.forEach((request, index) => {
    const id = `apr_later_${c.seed}_${index}`;
    approvalIds.push(id);
    if (request.verdict !== 'none') decisions.push({ approvalId: id, approved: request.verdict === 'approve', reason: request.reason });
  });
  // G-397: a row the host records between the park and the decision.
  if (c.insert?.when === 'before_decision' && c.insert.segment !== 'earlier') await insertRow(r, c);
  // A decision for the parked step after a row that superseded it is refused
  // (approval_not_pending), on every path; a later step's decisions are not.
  const oldIds = new Set(approvalIds.slice(0, c.requests.length));
  const refusedOld = superseded(c) && (c.later !== null || c.insert?.when === 'before_decision');
  const refused = async (body: () => Promise<unknown>): Promise<boolean> => body().then(() => false, (error: unknown) => {
    if ((error as { code?: string }).code === 'approval_not_pending') return true;
    throw error;
  });
  // The later step's decisions are refused when a row superseded it first.
  const refusedLater = laterSuperseded(c) && c.insert?.when === 'before_decision';
  // A refused batch records none of its decisions (all-or-nothing), so the
  // old step's and the later step's decisions go in batches of their own.
  const batches = refusedOld || c.later !== null
    ? [decisions.filter((d) => oldIds.has(d.approvalId)), decisions.filter((d) => !oldIds.has(d.approvalId))]
    : [decisions];
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
  for (const batch of batches.filter((b) => b.length > 0)) {
    const isOld = batch.every((d) => oldIds.has(d.approvalId));
    const expectRefused = isOld ? refusedOld : refusedLater;
    if (c.recorder === 'streamDecision' && c.insert?.when !== 'after_decision' && !resumedYet) {
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
      check(no === expectRefused, 'G-397: a decision is refused exactly when its step was superseded first', { no, expectRefused });
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
      check(no === expectRefused, 'G-397: a decision is refused exactly when its step was superseded first', { no, expectRefused });
    } else {
      for (const d of batch) {
        const no = await refused(() => recordApproval(r.session, d.approvalId, d.approved, d.reason));
        check(no === expectRefused, 'G-397: a decision is refused exactly when its step was superseded first', { no, expectRefused });
      }
    }
  }
  // G-397 / G-399: a row recorded between the decision and the resume.
  if (c.insert?.when === 'after_decision') await insertRow(r, c);
  if (!resumedYet) first = await resumed(async () => await turn(r, c.resume, ''));

  // Invariants 1-4, then full equality with the oracle.
  check(r.ran.length === new Set(r.ran).size, '3: a call runs at most once', r.ran);
  const all = [...c.requests.map((q) => ({ q, call: c.calls[q.call]! })), ...(c.later?.requests ?? []).map((q) => ({ q, call: c.later!.calls[q.call]! }))];
  const denied = new Set(all.filter(({ q }) => q.verdict === 'deny').map(({ call }) => call.token));
  check(r.ran.every((token) => !denied.has(token)), '1: a call a person denied never runs', r.ran);
  const approves = all.filter(({ q }) => q.verdict === 'approve');
  check(r.ran.every((token) => approves.some(({ call }) => call.token === token)), '2: each run call has its own approve', r.ran);
  check(r.ran.length <= approves.length, '2: one decision decides at most one call', { ran: r.ran.length, approves: approves.length });
  if (c.later !== null) check(r.ran.every((token) => token >= 200), 'a replaced step\'s approvals never run anything', r.ran);
  if (oracle.repeated) check(r.ran.length === 0, '4: a step with repeated ids runs none of its gated calls', r.ran);
  check(sameSet(r.ran, oracle.runs), 'the calls that ran are exactly the oracle\'s', { ran: r.ran, expected: [...oracle.runs] });

  // Invariant 6 on the resume itself: what it says it ran is what ran.
  const firstTurn = first as Turn | null;
  if (firstTurn !== null) check(firstTurn.response.toolCalls.filter((name) => name === 'echo').length === r.ran.length, '6: the resume reports the calls it ran', firstTurn.response.toolCalls);
  if (firstTurn !== null && firstTurn.steps !== null) {
    const results = firstTurn.steps.filter((step) => step.type === 'tool_result');
    const ok = results.flatMap((step) => (step.ok ? [Number(step.result.replace(/^ran:/, ''))] : []));
    check(sameSet(ok, r.ran) && ok.length === r.ran.length, '6: the ok tool_result steps are the calls that ran', ok);
    const hit = forgedTarget(c);
    const forgedToken = hit === null ? -1 : (hit.step === 'later' ? c.later!.calls : c.calls)[hit.index]!.token;
    const open = c.later !== null ? (laterSuperseded(c) ? [] : c.later.calls.filter((call) => call.tool === 'echo' && call.token !== forgedToken).map((call) => call.token))
      : superseded(c) ? []
        : gatedTokens.filter((token) => token !== forgedToken && !c.prior.includes(c.calls.findIndex((call) => call.token === token)));
    check(results.length === open.length, '6: one tool_result step per open gated call', { steps: results.length, open: open.length });
  }

  await settledOutcomes(r, c, oracle, '5');
  await replayShowsEachResultOnce(r.session, '7');

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
  await settledOutcomes(r, c, oracle, '5 (after repeat resumes)');
  await replayShowsEachResultOnce(r.session, '7 (after repeat resumes)');
}

/**
 * Invariant 7 on the replay the model is shown (G-403, BRIEF-R14 items 2 and
 * 5): in each step's segment, every result the thread stores is in the
 * replay view (by its id, tool name and canonical arguments, as often as it
 * occurs in one stored row: TS 0.11.x #83's replayKeepsEveryResult), and
 * NONE is emitted more often than that across the whole replay: a result
 * stored in two non-consecutive rows of one segment (a carried copy after a
 * row of another kind, a resolution row after a new message) is shown once,
 * in its own segment.
 */
async function replayShowsEachResultOnce(session: Session, label: string): Promise<void> {
  const stored = await rows(session);
  const isStep = (row: JsonObject): boolean => row.type === 'assistant' && Array.isArray(row.tool_calls) && row.tool_calls.length > 0;
  const segments = (list: JsonObject[], view: boolean): Map<string, number>[] => {
    const out: Map<string, number>[] = [new Map()];
    for (const row of list) {
      if (isStep(row)) { out.push(new Map()); continue; }
      if (row.type !== 'tool_result' || row.harness_replaced_step !== undefined) continue;
      // Per stored row, the most often a result occurs in one row; per view row, as shown.
      const here = new Map<string, number>();
      for (const entry of (row.tool_results as JsonObject[] | undefined) ?? []) {
        const key = canonicalJson([entry.tool_call_id, entry.tool_name, entry.args]);
        here.set(key, (here.get(key) ?? 0) + 1);
      }
      const segment = out.at(-1)!;
      for (const [key, count] of here) segment.set(key, view ? (segment.get(key) ?? 0) + count : Math.max(segment.get(key) ?? 0, count));
    }
    return out;
  };
  const want = segments(stored, false);
  const got = segments(threadView(stored), true);
  check(got.length === want.length, `${label}: the replay view keeps every step`, { stored: want.length, shown: got.length });
  want.forEach((segment, at) => {
    for (const [key, count] of segment) {
      const shown = got[at]?.get(key) ?? 0;
      check(shown >= count, `${label}: the replay view keeps every result of a step, by id, name and arguments`, { step: at, key, stored: count, shown });
      check(shown <= count, `${label}: the replay view shows a result once, in its own segment`, { step: at, key, stored: count, shown });
    }
    for (const key of (got[at] ?? new Map<string, number>()).keys()) check(segment.has(key), `${label}: the replay view shows no result outside its own segment`, { step: at, key });
  });
}

/** Invariants 5 and 6 on the thread rows: every call has exactly one outcome, the oracle's. */
async function settledOutcomes(r: Rig, _c: Case, oracle: Expected, label: string): Promise<void> {
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
  const requests = keptRequests.map(({ request }) => ({ ...request, call: remap(request.call), approval: request.approval === undefined ? undefined : position.get(request.approval) }))
    .map((request, at) => (request.approval === undefined || request.approval >= at ? { ...request, approval: undefined } : request));
  return {
    ...c,
    calls,
    requests,
    prior: c.prior.filter((index) => index !== drop).map(remap),
    insert: c.insert === null ? null : c.insert.segment === 'later' && c.insert.cross !== true ? c.insert : c.insert.call === drop ? null : { ...c.insert, call: remap(c.insert.call) },
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

  // The three seeds that failed on main 63ccf38 (BRIEF-R13, the G-392
  // property test), kept as fixed regression cases in every language:
  // (a) a gated call got no outcome, (b) a call got an extra repeated-id
  // denial, (c) a request paired with an ungated call.
  // 2483776628: the round-13 TS review's replay seed (G-403: a result dropped from the replay).
  it.each([2034523272, 232174035, 2060958372, 2483776628])('regression seed %i', async (seed) => {
    const why = await failure(generate(seed));
    expect(why).toBeNull();
  });

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
      expect(has((c) => c.insert?.kind === kind && decideStep(c.calls, c.requests, c.prior).repeated && c.requests.some((q) => q.verdict === 'approve')), `${kind} in a repeated-id step`).toBe(true);
    }
    expect(has((c) => c.insert !== null && c.recorder === 'streamDecision'), 'insert + streamDecision').toBe(true);
    // G-399: a row that does not supersede, after the decision, a resume with no assistant row (steps spent), then repeat resumes.
    expect(has((c) => (c.insert?.kind === 'system' || c.insert?.kind === 'tool_empty') && c.insert.when === 'after_decision' && c.spent && expected(c).runs.size > 0 && c.again.length > 0), 'G-399 spent').toBe(true);
    // A superseding row after an approval: nothing runs.
    expect(has((c) => c.insert !== null && SUPERSEDES.includes(c.insert.kind) && c.insert.when === 'after_decision' && c.requests.some((q) => q.verdict === 'approve')), 'superseded after an approve').toBe(true);
    // G-397: a later tool-call step replaces the parked one, with approvals for both.
    expect(has((c) => c.later !== null && c.requests.some((q) => q.verdict === 'approve') && c.later.requests.some((q) => q.verdict === 'approve')), 'later step').toBe(true);
    expect(has((c) => c.later !== null && decideStep(c.later.calls, c.later.requests, []).repeated), 'later step repeating ids').toBe(true);
    expect(has((c) => c.later !== null && c.later.calls.some((call) => c.calls.some((old) => old.id === call.id))), 'later step reusing an old id').toBe(true);
    // NO-BLIND-SPOT: a later step WITH every inserted kind, in both segments,
    // and a forged result in BOTH the earlier and the later segment.
    for (const kind of INSERT_KINDS) {
      for (const segment of ['earlier', 'later'] as const) expect(has((c) => c.later !== null && c.insert?.kind === kind && c.insert.segment === segment), `later step + ${kind} (${segment})`).toBe(true);
    }
    for (const when of ['before_decision', 'after_decision'] as const) {
      expect(has((c) => laterSuperseded(c) && c.insert!.when === when && (c.later?.requests.some((q) => q.verdict === 'approve') ?? false)), `later step superseded ${when}, approved`).toBe(true);
      expect(has((c) => c.later !== null && c.insert?.kind === 'tool_result' && c.insert.segment === 'later' && c.insert.when === when), `forged in the later segment ${when}`).toBe(true);
    }
    // SEGMENT OWNERSHIP (round 13b): a later-segment forged result naming an
    // EARLIER call with its exact arguments, both when a later gated call
    // shares its id (it answers that call) and when none does (it answers
    // nothing), before and after the decision, and answering an APPROVED call.
    for (const when of ['before_decision', 'after_decision'] as const) {
      expect(has((c) => c.insert?.cross === true && c.insert.when === when && forgedTarget(c) !== null), `cross-segment forged result answering a later call ${when}`).toBe(true);
      expect(has((c) => c.insert?.cross === true && c.insert.when === when && forgedTarget(c) === null), `cross-segment forged result answering nothing ${when}`).toBe(true);
    }
    expect(has((c) => c.insert?.cross === true && forgedTarget(c) !== null && c.later!.requests.some((q) => q.call === forgedTarget(c)!.index && q.verdict === 'approve')), 'cross-segment forged result answering an APPROVED later call').toBe(true);
    // G-403: a step with repeated ids whose results carry different arguments (TS 0.11.x #83's coverage line).
    expect(has((c) => c.calls.some((call, i) => c.calls.some((other, j) => j !== i && other.id === call.id && other.token !== call.token
      && (call.tool === 'free' || c.prior.includes(i))))), 'repeated ids, results with different arguments').toBe(true);
    // Item 5: a row of another kind between an ungated call's result and the resolution row that carries it.
    expect(has((c) => c.source === 'model' && c.calls.some((call) => call.tool === 'free') && (c.insert?.kind === 'system' || c.insert?.kind === 'tool_empty')
      && c.insert.when === 'after_decision' && expected(c).runs.size > 0), 'item 5: a carried copy after a row of another kind').toBe(true);
    // G-405: a structured turn that parks (its step's text a bad-typed document).
    expect(has((c) => c.source === 'model' && c.initial === 'structured'), 'G-405: a parked structured turn').toBe(true);
    // G-397: stored requests that repeat an APPROVAL id, with an approval recorded.
    expect(has((c) => c.requests.some((q) => q.approval !== undefined) && c.requests.some((q) => q.verdict === 'approve')), 'repeated approval id').toBe(true);
  });
});
