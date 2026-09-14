import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { assistantRow, threadView, toolResultRow, type JsonObject } from '../src/index.js';

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
  input?: JsonObject & { write: 'assistant' | 'tool_result' };
  fold?: JsonObject[];
  rows: { php: string; ts: string; py: string };
  agrees: boolean;
}

const corpus = JSON.parse(readFileSync(new URL('./fixtures/harness-thread-rows.json', import.meta.url), 'utf8')) as {
  cases: RowsCase[];
};

/** The same conversion prism-parity's recorder makes. */
function rowsFor(testCase: RowsCase): JsonObject[] {
  if (testCase.fold !== undefined) return threadView(testCase.fold);

  const input = testCase.input as unknown as Record<string, never>;

  if (testCase.input?.write === 'assistant') {
    return [
      assistantRow(
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
      ),
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

describe('harness-thread-rows corpus', () => {
  it('is the whole suite, not a subset someone trimmed to green', () => {
    expect(corpus.cases).toHaveLength(10);
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
