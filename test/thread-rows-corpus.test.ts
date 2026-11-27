import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { assistantRow, threadView, toolResultRow, type JsonObject } from '../src/index.js';
import { answeringStep, endedByStart, endedByStartRow, newResults, pairStepApprovals, pendingApprovalIds, replayView, standingDecisions, startedRow, stepAnswers, withBoundRequests } from '../src/thread-rows.js';
import { readerRows } from '../src/thread.js';

/**
 * The cross-language thread-rows corpus from `prism-parity`.
 *
 * The rows this port stores for a run, and what it replays them as. A PHP app
 * and a TypeScript agent can share a session; if the rows differed, one would
 * resume the other's approval against the wrong call, or replay a conversation
 * a provider refuses.
 */
interface RowsCase {
  id: string;
  input?: JsonObject & { write: 'assistant' | 'tool_result' | 'ended_by_start' | 'started' | 'answer' };
  fold?: JsonObject[];
  /** A PLACE case (0.5.1): rows as stored (`harness: true` on a row the harness wrote), placed then folded as the replay sends them. */
  place?: JsonObject[];
  /** A READ case (0.4.0): what the harness reads from `stored` rows for each id in `approvals`. */
  read?: 'approval_states' | 'call_outcomes' | 'approval_reads';
  stored?: JsonObject[];
  /** call_outcomes (0.6.0): the thread position of the step read. */
  step?: number;
  approvals?: string[];
  rows: { php: string; ts: string; py: string };
  agrees: boolean;
}

const corpus = JSON.parse(readFileSync(new URL('./fixtures/harness-thread-rows.json', import.meta.url), 'utf8')) as {
  cases: RowsCase[];
};

/**
 * Corpus rows as the harness's readers take them (0.6.0, ONE TRUST SIGNAL):
 * a row carrying `"harness": true` is one the runtime wrote; any other row
 * is a host's, its marks read as absent (readerRows). A `run_id` key is read
 * as nothing.
 */
function asStored(rows: readonly JsonObject[]): JsonObject[] {
  return readerRows(rows.map((row) => {
    const { harness, run_id: _runId, ...message } = row;
    return { message, harness: harness === true };
  }));
}

/** The same conversion prism-parity's recorder makes. */
function rowsFor(testCase: RowsCase): JsonObject[] | boolean[] {
  if (testCase.fold !== undefined) return threadView(testCase.fold);

  // Round 15: the rows as stored, placed (started rows dropped, a marked row
  // moved to its own step, interrupted / replaced results given, standing
  // decisions, void requests) and then folded: what the runtime sends.
  if (testCase.place !== undefined) return replayView(asStored(testCase.place));

  if (testCase.read === 'approval_states') {
    // Which approvals a runner start ended (G-387 round 13), by the harness's
    // own reader over the rows as stored: one entry per id, in order.
    const ended = endedByStart(asStored(testCase.stored ?? []));
    return (testCase.approvals ?? []).map((id) => ({ approval_id: id, ended_by_start: ended.has(id) }));
  }

  if (testCase.read === 'call_outcomes') {
    // 0.6.0: which calls of the step at `step` have an outcome (a result, or a started row).
    return stepAnswers(asStored(testCase.stored ?? []), (testCase.step ?? 0) - 1).map((answer) => answer !== null);
  }

  if (testCase.read === 'approval_reads') {
    // 0.6.0: per approval, whether it is pending and the decision that stands (null: none).
    const stored = asStored(testCase.stored ?? []);
    const pending = new Set(pendingApprovalIds(stored));
    const standing = standingDecisions(stored);
    return (testCase.approvals ?? []).map((id) => ({ approval_id: id, pending: pending.has(id), decision: standing.get(id)?.approved ?? null }));
  }

  if (testCase.input?.write === 'answer') {
    // 0.6.0: the answer row a resolution writes: only the results its step
    // does not hold yet, and the mark when one is kept (the runtime's build).
    const answer = testCase.input as JsonObject;
    const stored = asStored(answer.stored as JsonObject[]);
    const position = answer.position as number;
    const own = newResults(stored, position - 1, answer.results as JsonObject[]);
    const decisions = answer.decisions as JsonObject[];
    if (own.length === 0 && decisions.length === 0) return [];
    const row = toolResultRow(own as never, decisions as never) as unknown as JsonObject;
    return [own.length === 0 ? row : answeringStep(row, position)];
  }

  const input = testCase.input as unknown as Record<string, never>;

  if (testCase.input?.write === 'ended_by_start') {
    // The row a runner start writes when it ends the calls its child was
    // parked on (G-387): the harness's own builder, each call paired with
    // its OWN approval request by position (G-392), as the runtime pairs them.
    const calls = input.tool_calls as JsonObject[];
    const requests = input.approval_requests as JsonObject[];
    return [
      endedByStartRow(
        calls.map((call) => ({ id: call.id as string, name: call.name as string, arguments: call.arguments as JsonObject, resultId: (call.result_id as string | undefined) ?? null })),
        pairStepApprovals(calls, requests).approvalIds,
      ) as unknown as JsonObject,
    ];
  }

  if (testCase.input?.write === 'started') {
    // The write-ahead started row (round 15), the harness's own builder.
    return [startedRow({ id: input.tool_call_id as string, name: input.tool_name as string }, input.approval_id as string, input.position as number)];
  }

  if (testCase.input?.write === 'assistant') {
    // `bind`: the requests as Thread.record() stores them, each bound to its own call (round 15).
    const bind = (row: JsonObject): JsonObject => (testCase.input?.bind === true ? withBoundRequests(row) : row);
    return [
      bind(assistantRow(
        input.content,
        (input.tool_calls as JsonObject[]).map((call) => ({
          id: call.id as string,
          name: call.name as string,
          arguments: call.arguments as JsonObject,
          resultId: (call.result_id as string | undefined) ?? null,
          reasoningId: (call.reasoning_id as string | undefined) ?? null,
          reasoningSummary: (call.reasoning_summary as JsonObject[] | undefined) ?? null,
        })),
        input.additional_content,
        input.approval_requests,
      )),
    ];
  }

  return [
    toolResultRow(
      (input.results as JsonObject[]).map((result) => ({
        tool_call_id: result.tool_call_id as string,
        tool_name: result.tool_name as string,
        args: result.args as JsonObject,
        result: result.result as string,
        tool_call_result_id: (result.tool_call_result_id as string | undefined) ?? null,
        artifacts: [],
      })),
      input.decisions,
    ),
  ];
}

/** Set to an absolute path to write what this port produces for every row (for the parity recorder). */
const RECORD = process.env.HARNESS_THREAD_ROWS_RECORD;

describe('harness-thread-rows corpus', () => {
  if (RECORD !== undefined && RECORD !== '') {
    it("records this port's rows", async () => {
      const { writeFileSync } = await import('node:fs');
      writeFileSync(RECORD, JSON.stringify(Object.fromEntries(corpus.cases.map((testCase) => [testCase.id, JSON.stringify(rowsFor(testCase))])), null, 2));
    });
  }

  it('is the whole suite, not a subset someone trimmed to green', () => {
    expect(corpus.cases).toHaveLength(40);
  });

  it.each(corpus.cases.map((testCase) => [testCase.id, testCase] as const))('%s stores and replays the rows the corpus records for this port', (_id, testCase) => {
    expect(JSON.stringify(rowsFor(testCase))).toBe(testCase.rows.ts);
  });

  it('agrees with the reference and the Python port on every row', () => {
    for (const testCase of corpus.cases) {
      expect([testCase.rows.ts, testCase.rows.py], testCase.id).toEqual([testCase.rows.php, testCase.rows.php]);
      expect(testCase.agrees, testCase.id).toBe(true);
    }
  });
});
