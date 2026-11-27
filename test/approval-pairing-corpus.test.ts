import { readFileSync } from 'node:fs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  HarnessError,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  ToolRegistry,
  recordDecisions,
  type JsonObject,
  type LlmResponse,
} from '../src/index.js';
import { ARGUMENTS_AS_SENT, assistantRow, toolResultEntry, toolResultRow } from '../src/thread-rows.js';

/**
 * The cross-language approval-pairing corpus (harness-approval-pairing, G-392).
 *
 * Every row's `expected` was written by hand before any language ran. This
 * runner drives the real runtime offline with a fake model client: the first
 * reply is the row's calls, each under the row's id, and every later reply is
 * the text `done`. `echo` is gated by mode and `free` is not. Each round
 * decides the listed approvals with one recordDecisions() call, then resumes
 * once with an empty prompt; a decision the harness refuses is swallowed.
 * `decided` (0.2.0) is the decision the thread recorded for each approval.
 * `outcomes` / `later_outcomes` are read BY SEGMENT (round 13b): a step's
 * results are the tool_result rows after it and before the next assistant
 * row with tool calls.
 */
interface PairingCase {
  id: string;
  title: string;
  calls: { id: string; name: string; args: JsonObject }[];
  rounds: [number, boolean][][];
  /** DECISION B (0.2.0): what supersedes the parked step, after the park and before the rounds. */
  supersede?: 'assistant_tool_step' | 'assistant_text' | 'user' | 'user_owning_run_id' | 'new_message' | 'new_message_failed';
  /**
   * SEGMENT OWNERSHIP (0.2.0, round 13b): a later step a host records after
   * the park, each echo call asking its own approval (`apr_later_<k>`, k
   * counting the step's echo calls), then ONE host-recorded result row right
   * after it; its `rounds` are decided before the row's, then `resumes`
   * empty-prompt resumes.
   */
  later?: {
    calls: { id: string; name: string; args: JsonObject }[];
    result: { id: string; name: string; args: JsonObject; result: string; approved: number[] };
    rounds: [number, boolean][][];
    resumes: number;
  };
  expected: unknown;
}

const corpus = JSON.parse(
  readFileSync(new URL('./fixtures/harness-approval-pairing.json', import.meta.url), 'utf8'),
) as { cases: PairingCase[] };

/** An in-memory store that declares itself durable, for an offline run. */
class DurableMemory extends MemorySessionStore {
  override durability(): 'durable' {
    return 'durable';
  }
}

const network = vi.fn(() => { throw new Error('A test tried to use the network.'); });
beforeEach(() => { vi.stubGlobal('fetch', network); });
afterEach(() => { expect(network).not.toHaveBeenCalled(); vi.unstubAllGlobals(); });

async function record(row: PairingCase): Promise<unknown> {
  const ran: number[] = [];
  let free = 0;
  const tools = new ToolRegistry()
    .register({ name: 'echo', handle: (args: JsonObject) => { ran.push(Number(args.n)); return 'ok'; } })
    .register({ name: 'free', handle: () => { free += 1; return 'free ok'; } });
  const modes = new ModeRegistry({
    default: 'guarded',
    modes: { guarded: { system_prompt: 'Careful.', tools: ['echo', 'free'], max_steps: 6, requires_approval: ['echo'] } },
  });
  let step = 0;
  const client = async (): Promise<LlmResponse> => {
    step += 1;
    // apr-0023 (0.3.0): the provider fails the new message's request.
    if (step === 2 && row.supersede === 'new_message_failed') throw new Error('The provider is unavailable (scripted).');
    return step === 1
      ? { text: '', finishReason: 'tool_calls', toolCalls: row.calls.map((call) => ({ id: call.id, name: call.name, arguments: call.args })) }
      : { text: 'done', finishReason: 'stop' };
  };
  const harness = new PrismHarness({
    drivers: { memory: () => new MemorySessionStore(), durable: () => new DurableMemory() },
    stores: { ephemeral: 'memory', durable: 'durable' },
  });
  const session = harness.for({ type: 'User', id: 1 }).session('corpus');
  const runtime = new AgentRuntime({ client, modes, tools });

  const parked = await runtime.send(session, 'go');
  const pending = parked.pendingApprovals;
  // DECISION B (0.2.0): a row that supersedes the parked step, recorded by a
  // host (outside the run that owns the step), or a new message.
  if (row.supersede === 'assistant_tool_step') await session.thread().record([{ type: 'assistant', content: '', tool_calls: [{ id: 'later', name: 'free', arguments: {} }], tool_approval_requests: [] }]);
  if (row.supersede === 'assistant_text') await session.thread().record([{ type: 'assistant', content: 'A later answer.', tool_calls: [], tool_approval_requests: [] }]);
  if (row.supersede === 'user') await session.thread().record([{ type: 'user', content: 'A later message.' }]);
  // A caller passing the parked step's OWN run id: a run id a caller passes is always outside (UPDATE 3).
  if (row.supersede === 'user_owning_run_id') await session.thread().record([{ type: 'user', content: 'A later message.' }], parked.runId);
  if (row.supersede === 'new_message') await runtime.send(session, 'Something else.');
  // A new message whose turn then FAILS (G-404): it still supersedes, its message and denials one write before the request.
  if (row.supersede === 'new_message_failed') await runtime.send(session, 'Something else.').catch(() => undefined);
  const later = row.later;
  const laterIds: string[] = [];
  if (later !== undefined) {
    const requests = later.calls.flatMap((call) => {
      if (call.name !== 'echo') return [];
      const id = `apr_later_${laterIds.length}`;
      laterIds.push(id);
      return [{ approval_id: id, tool_call_id: call.id, arguments_format: ARGUMENTS_AS_SENT }];
    });
    await session.thread().record([assistantRow('', later.calls.map((call) => ({ id: call.id, name: call.name, arguments: call.args })), {}, requests)]);
    const result = later.result;
    await session.thread().record([toolResultRow(
      [toolResultEntry({ id: result.id, name: result.name, arguments: result.args }, result.result)],
      result.approved.map((k) => ({ approval_id: laterIds[k]!, approved: true, reason: null })),
    )]);
  }
  // The code each round was refused with, or null (0.2.0, rows with
  // `supersede` or `later`): the later step's rounds first, then the row's.
  const refusals: (string | null)[] = [];
  const rounds: [[number, boolean][], string[]][] = [
    ...(later?.rounds ?? []).map((round): [[number, boolean][], string[]] => [round, laterIds]),
    ...row.rounds.map((round): [[number, boolean][], string[]] => [round, pending.map((approval) => approval.id)]),
  ];
  for (const [round, ids] of rounds) {
    try {
      await recordDecisions(session, round.map(([index, approved]) => ({ approvalId: ids[index]!, approved })));
      refusals.push(null);
    } catch (error) {
      // A decision the harness refuses is swallowed; the round still resumes.
      if (!(error instanceof HarnessError && error.code === 'approval_not_pending')) throw error;
      refusals.push(error.code);
    }
    await runtime.send(session, '');
  }
  for (let resume = 0; resume < (later?.resumes ?? 0); resume += 1) await runtime.send(session, '');

  // The decision the thread RECORDED for each approval the first step issued,
  // in issue order (0.2.0): the first answer, or null when none was recorded.
  const recorded = new Map<string, boolean>();
  for (const { message } of await session.thread().messages()) {
    if (message.type !== 'tool_result') continue;
    for (const answer of (message.tool_approval_responses ?? []) as JsonObject[]) {
      if (typeof answer.approval_id === 'string' && !recorded.has(answer.approval_id)) recorded.set(answer.approval_id, answer.approved === true);
    }
  }

  const outcome: Record<string, unknown> = { approvals: pending.length, free, ran: [...ran].sort((a, b) => a - b), decided: pending.map((approval) => recorded.get(approval.id) ?? null) };
  if (row.supersede !== undefined || later !== undefined) {
    // The result each call of a step ended with, in call order: its FIRST
    // result IN ITS STEP'S SEGMENT (round 13b), or null.
    const steps = segments((await session.thread().messages()).map((entry) => entry.message));
    outcome.outcomes = segmentOutcomes(row.calls, steps[0] ?? []);
    if (later !== undefined) outcome.later_outcomes = segmentOutcomes(later.calls, steps[1] ?? []);
    outcome.refusals = refusals;
  }

  return outcome;
}

/**
 * The tool_results entries of each step's segment, oldest first: one list
 * per assistant row with tool calls (the rows after it, before the next
 * assistant row with tool calls), in thread order.
 */
function segments(rows: readonly JsonObject[]): JsonObject[][] {
  const out: JsonObject[][] = [];
  for (const row of rows) {
    if (row.type === 'assistant' && Array.isArray(row.tool_calls) && row.tool_calls.length > 0) out.push([]);
    else if (out.length > 0 && row.type === 'tool_result') out.at(-1)!.push(...((row.tool_results ?? []) as JsonObject[]));
  }
  return out;
}

type CorpusCall = PairingCase['calls'][number];

/**
 * The first result each call got in its segment, in call order, or null. An
 * entry is a call's by exact arguments, then id and tool name, then id: the
 * first call so matched with no result yet; an entry whose match already has
 * one is a repeat, and an entry that matches no call counts for nothing.
 */
function segmentOutcomes(calls: readonly CorpusCall[], entries: readonly JsonObject[]): (string | null)[] {
  const first: (string | null)[] = calls.map(() => null);
  for (const entry of entries) {
    const args = typeof entry.args === 'object' && entry.args !== null && !Array.isArray(entry.args) ? entry.args : {};
    const rules: ((call: CorpusCall) => boolean)[] = [
      (call) => call.id === entry.tool_call_id && call.name === entry.tool_name && sameJson(call.args, args),
      (call) => call.id === entry.tool_call_id && call.name === entry.tool_name,
      (call) => call.id === entry.tool_call_id,
    ];
    for (const matches of rules) {
      const matched = calls.flatMap((call, at) => (matches(call) ? [at] : []));
      if (matched.length === 0) continue;
      const open = matched.find((at) => first[at] === null);
      if (open !== undefined) first[open] = String(entry.result);
      break;
    }
  }
  return first;
}

/** JSON equality with object keys in any order (as PHP compares arrays with `==`). */
function sameJson(a: unknown, b: unknown): boolean {
  const canonical = (value: unknown): unknown => (Array.isArray(value) ? value.map(canonical)
    : typeof value === 'object' && value !== null
      ? Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonical((value as Record<string, unknown>)[key])]))
      : value);
  return JSON.stringify(canonical(a)) === JSON.stringify(canonical(b));
}

/** Set to an absolute path to write what this port measures for every row (for the parity recorder). */
const RECORD = process.env.HARNESS_APPROVAL_PAIRING_RECORD;

describe('harness-approval-pairing corpus', () => {
  if (RECORD !== undefined && RECORD !== '') {
    it("records this port's outcomes", async () => {
      const { writeFileSync } = await import('node:fs');
      const out: Record<string, unknown> = {};
      for (const row of corpus.cases) out[row.id] = await record(row);
      writeFileSync(RECORD, JSON.stringify(out, null, 2));
    });
  }

  it('has rows', () => {
    expect(corpus.cases.length).toBeGreaterThan(0);
  });

  for (const row of corpus.cases) {
    it(`${row.id}: ${row.title}`, async () => {
      expect(await record(row)).toEqual(row.expected);
    });
  }
});
