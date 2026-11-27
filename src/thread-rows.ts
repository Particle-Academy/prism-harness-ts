import type { JsonObject, JsonValue } from './json.js';
import { createHash } from 'node:crypto';
import { usableArguments } from './tool-arguments.js';
import { canonicalJson } from './tool-policy.js';

/**
 * The rows a run writes to a thread, in the shape the PHP reference stores.
 *
 * prism-harness stores Prism's own messages: an assistant turn carries its tool
 * calls (arguments and provider ids included), its provider state and its
 * approval requests; a tool result turn carries every result of a step and any
 * approval decisions. These builders write exactly that, with the row's `type`
 * beside it, so a thread written here reads the same as one written there.
 * prism-parity's `harness-thread-rows` corpus pins it.
 */

// Type aliases rather than interfaces: an alias is assignable to JsonObject's
// index signature, and an interface is not.

export type ToolCallRow = {
  id: string;
  name: string;
  arguments: JsonObject;
  result_id: string | null;
  reasoning_id: string | null;
  reasoning_summary: JsonValue[] | null;
};

export type ToolResultEntry = {
  tool_call_id: string;
  tool_name: string;
  args: JsonObject;
  result: string;
  tool_call_result_id: string | null;
  artifacts: JsonValue[];
};

export type ApprovalRequestEntry = {
  approval_id: string;
  tool_call_id: string;
  /** Present when the call's arguments were stored as the model sent them. */
  arguments_format?: number;
  /** The call this approval was asked for (round 15, CALL_BINDING_KEY): its LAST key. */
  harness_call?: { tool_name: string; args_digest: string };
};

/**
 * The key of the binding on every approval request the runtime writes when
 * a step parks (round 15, the reference's shape): `{tool_name, args_digest}`,
 * the request's LAST key; `args_digest` is `sha256:` and the hex sha256 of
 * the call's arguments as canonical JSON (callArgsDigest). A call runs only
 * when its id (the request's pairing), its tool name and that digest all
 * match its request's; otherwise it is denied. A request stored before
 * bindings were kept has none, and binds to its own step's call paired by
 * position (the request is its FIRST asker's, so that call is the one the
 * person was shown).
 *
 * @internal
 */
export const CALL_BINDING_KEY = 'harness_call';

/** The digest a binding holds for a call's arguments, as stored (a JSON string decoded first). @internal */
export function callArgsDigest(args: JsonValue | undefined): string {
  return `sha256:${createHash('sha256').update(canonicalJson(usableArguments(args) ?? {}), 'utf8').digest('hex')}`;
}

/**
 * The outcome of an approved call the resolver may not start (round 15, the
 * reference's text): its request does not bind it (item 7), or it is no
 * longer open. Written only for a call that has no outcome yet.
 *
 * @internal
 */
export const UNBOUND_APPROVAL_RESULT = 'Not run: this call is not the one that was approved, or it is no longer open.';

/** `request` bound to `call` (CALL_BINDING_KEY last). @internal */
export function boundRequest(request: ApprovalRequestEntry, call: { name: string; arguments: JsonValue | undefined }): ApprovalRequestEntry {
  const { harness_call: _old, ...rest } = request;
  return { ...rest, harness_call: { tool_name: call.name, args_digest: callArgsDigest(call.arguments) } };
}

/**
 * `row` (an assistant step) with every approval request bound to its OWN
 * call (CALL_BINDING_KEY last): the first call with the request's
 * tool_call_id not already taken by an earlier request of the step. A
 * request with no call of its own is left unbound. Thread.record() writes it
 * on every stored step with requests, a host's or the runtime's (round 15).
 *
 * @internal
 */
export function withBoundRequests(row: JsonObject): JsonObject {
  const calls = stepCalls(row);
  const requests = row.tool_approval_requests;
  if (calls.length === 0 || !Array.isArray(requests) || requests.length === 0) return row;
  const taken = new Set<number>();
  const bound = requests.map((request) => {
    if (!isObject(request)) return request;
    const at = calls.findIndex((call, index) => !taken.has(index) && text(call.id) === text(request.tool_call_id));
    if (at === -1) return request;
    taken.add(at);
    const call = calls[at]!;
    const { [CALL_BINDING_KEY]: _old, ...rest } = request;
    return { ...rest, [CALL_BINDING_KEY]: { tool_name: text(call.name), args_digest: callArgsDigest(call.arguments) } };
  });
  return { ...row, tool_approval_requests: bound };
}

/** Whether `request` may decide `call`: its binding names the call's tool and arguments digest; a request with none binds by position. @internal */
export function bindsCall(request: JsonObject | undefined, call: JsonObject): boolean {
  if (request === undefined || !Object.hasOwn(request, CALL_BINDING_KEY)) return true;
  const binding = request[CALL_BINDING_KEY];
  return isObject(binding) && binding.tool_name === call.name && binding.args_digest === callArgsDigest(call.arguments);
}

/**
 * The marker on an approval request whose call was stored exactly as the model
 * sent it. 0.11.0 and earlier stored a call whose arguments were a bare number
 * or boolean as `{}`, the same as a real call with no arguments, and wrote no
 * marker; such an approval expires instead of running.
 */
export const ARGUMENTS_AS_SENT = 1;

export type ApprovalDecisionEntry = {
  approval_id: string;
  approved: boolean;
  reason: string | null;
};

export type AssistantRow = {
  type: 'assistant';
  content: string;
  tool_calls: ToolCallRow[];
  additional_content: JsonObject;
  tool_approval_requests: ApprovalRequestEntry[];
};

export type ToolResultRow = {
  type: 'tool_result';
  tool_results: ToolResultEntry[];
  tool_approval_responses: ApprovalDecisionEntry[];
};

export interface ToolCallInput {
  id: string;
  name: string;
  arguments: JsonObject;
  resultId?: string | null;
  reasoningId?: string | null;
  reasoningSummary?: readonly JsonValue[] | null;
}

export function assistantRow(
  content: string,
  toolCalls: readonly ToolCallInput[],
  additionalContent: Readonly<JsonObject> = {},
  approvalRequests: readonly ApprovalRequestEntry[] = [],
): AssistantRow {
  return {
    type: 'assistant',
    content,
    tool_calls: toolCalls.map((call) => ({
      id: call.id,
      name: call.name,
      arguments: { ...call.arguments },
      result_id: call.resultId ?? null,
      reasoning_id: call.reasoningId ?? null,
      reasoning_summary: call.reasoningSummary ? [...call.reasoningSummary] : null,
    })),
    additional_content: { ...additionalContent },
    tool_approval_requests: approvalRequests.map((request) => ({ ...request })),
  };
}

export function toolResultEntry(call: ToolCallInput, result: string): ToolResultEntry {
  return {
    tool_call_id: call.id,
    tool_name: call.name,
    args: { ...call.arguments },
    result,
    tool_call_result_id: call.resultId ?? null,
    artifacts: [],
  };
}

export function toolResultRow(
  results: readonly ToolResultEntry[],
  decisions: readonly ApprovalDecisionEntry[] = [],
): ToolResultRow {
  return {
    type: 'tool_result',
    tool_results: results.map((entry) => ({ ...entry })),
    tool_approval_responses: decisions.map((decision) => ({ ...decision })),
  };
}

/**
 * The thread as the model is shown it: rows written by 0.3.0 and earlier read
 * in today's shape, and consecutive tool result rows folded into one.
 *
 * An approval leaves several tool result rows in a row: the results of the
 * calls a stopped step did run, the decisions, and the results written when the
 * run resumes. Rows are never rewritten, so all of them stay stored; shown to a
 * provider separately, a tool output would go twice. Folded, a result is keyed
 * by its call id, tool name and canonical arguments, occurrence by occurrence
 * (G-403, the reference's fold), never by call id alone, and a decision by its
 * approval id; a later one replaces an earlier one in the earlier one's
 * position.
 *
 * EACH RESULT ONCE, IN ITS OWN SEGMENT (BRIEF-R14 item 5, the reference's
 * rule, pinned by thr-0017): a step's segment is every row after it up to
 * the next assistant row with tool calls, and its result rows need not be
 * consecutive. A resolution recorded after a row of another kind (a host's
 * system row) carries the step's earlier results again. A later run of the
 * segment therefore leaves out every result (same key, same occurrence) and
 * decision an earlier run of it already replayed; the first one stands where
 * it was. A run left with nothing is not replayed at all (a run that was
 * empty to begin with is replayed as before).
 *
 * This is the REPLAY view, pinned by the harness-thread-rows corpus. What a
 * decision means for running a call is read from the stored rows instead, by
 * `turnDecisions()`, where the first answer to an approval is the answer.
 */
export function threadView(rows: readonly JsonObject[]): JsonObject[] {
  const view: JsonObject[] = [];
  // The consecutive tool result rows not yet replayed, and what the current
  // segment has replayed: by result key, how many; by approval id.
  let run: ToolResultRow[] = [];
  let replayed = new Map<string, number>();
  let decided = new Set<string>();
  const flush = (): void => {
    if (run.length === 0) return;
    const merged = unreplayed(run.reduce((earlier, later) => fold(earlier, later)), replayed, decided);
    if (merged !== null) view.push(merged);
    run = [];
  };

  for (const row of rows) {
    const normalised = normalise(row, view);

    if (normalised === null) continue;

    if (normalised.type === 'tool_result') {
      run.push(normalised as ToolResultRow);
      continue;
    }

    flush();
    if (normalised.type === 'assistant' && list(normalised.tool_calls).length > 0) {
      // A new step: a new segment.
      replayed = new Map();
      decided = new Set();
    }
    view.push(normalised);
  }
  flush();

  return view;
}

/**
 * What a provider is sent (round 15): the stored rows PLACED (placedRows),
 * then folded (threadView). The runtime's replay; the reference's
 * Thread::placedRows() then its fold, pinned by the corpus `place` rows.
 *
 * @internal
 */
export function replayView(rows: readonly JsonObject[]): JsonObject[] {
  return threadView(placedRows(rows));
}

/**
 * The stored rows as the replay places them (round 15, the reference's
 * Thread::placedRows): in the order replayOrder() gives; every decision
 * entry rewritten to the decision that STANDS for its approval
 * (standingDecisions(): the one that stood when its call was resolved,
 * never one recorded after its call had an outcome, which is left out,
 * round 16); and a step's
 * request that repeats an approval id an EARLIER step asked for dropped (it
 * asks for nothing, G-408 / #92).
 *
 * @internal
 */
export function placedRows(rows: readonly JsonObject[]): JsonObject[] {
  const standing = standingDecisions(rows);
  const asked = new Set<string>();
  return replayOrder(rows).map((row) => {
    if (isToolStep(row) && Array.isArray(row.tool_approval_requests)) {
      const own = row.tool_approval_requests.filter((request) => !isObject(request) || !asked.has(text(request.approval_id)));
      for (const request of stepRequests(row)) asked.add(text(request.approval_id));
      return own.length === row.tool_approval_requests.length ? row : { ...row, tool_approval_requests: own };
    }
    if (row.type === 'tool_result' && Array.isArray(row.tool_approval_responses)) {
      // A decision that never stood (recorded once its call had an outcome)
      // is never shown (round 16, item 10).
      const answers = row.tool_approval_responses.flatMap((answer): JsonValue[] => (!isObject(answer) ? [answer] : standing.has(text(answer.approval_id)) ? [standing.get(text(answer.approval_id))!] : []));
      return { ...row, tool_approval_responses: answers as JsonValue[] };
    }
    if (row.type === 'tool_approval_response') {
      const stands = standing.get(text(row.approval_id));
      return stands === undefined ? row : { ...row, approved: stands.approved, reason: stands.reason };
    }
    return row;
  });
}

/**
 * The rows in the order the replay shows them (G-387 round 15): a harness
 * STARTED row is never replayed; a harness row whose honoured mark
 * (markedOwner) names an earlier step is replayed in THAT step's result run
 * (directly after the step and the result and system rows that follow it),
 * not where it sits; and a started call with no result its step owns gets
 * ONE result there, INTERRUPTED_RESULT, so a provider sees an answer for
 * every call. Unmarked threads are returned as they are.
 */
function replayOrder(rows: readonly JsonObject[]): readonly JsonObject[] {
  const marked = (row: JsonObject): boolean => row.harness === true && row.type === 'tool_result'
    && (Object.hasOwn(row, ANSWERS_STEP_KEY) || Object.hasOwn(row, STARTED_KEY));
  const paused = pausedStepIndex(rows);
  const anyMarked = rows.some(marked);
  // What each step's replay is given (round 15): its started calls with no
  // result read INTERRUPTED; a step that is NOT the paused one, a requested
  // call of which has no outcome (a thread an older version wrote, item 9),
  // reads REPLACED; and ANY other call of such a step with no stored result
  // reads INTERRUPTED (round 16, item 3: an ungated call whose result never
  // landed), so a provider request answers every call. One row per step, in
  // call order.
  const synthesised = new Map<number, JsonObject>();
  for (const [index, row] of rows.entries()) {
    if (!isToolStep(row)) continue;
    const requests = ownRequests(rows, index);
    const answers = stepAnswers(rows, index);
    if (index === paused && !answers.includes('started')) continue;
    if (index !== paused && !answers.includes(null) && !answers.includes('started')) continue;
    const calls = stepCalls(row);
    const requested = pairStepApprovals(calls, requests, (_call, position) => answers[position] === null).approvalIds;
    const entries = answers.flatMap((answer, position) => {
      const result = answer === 'started' ? INTERRUPTED_RESULT
        : answer === null && index !== paused ? (requested[position] !== undefined ? SUPERSEDED_RESULT : INTERRUPTED_RESULT) : null;
      if (result === null) return [];
      const call = calls[position]!;
      return [{
        tool_call_id: text(call.id),
        tool_name: text(call.name),
        args: usableArguments(call.arguments) ?? {},
        result,
        tool_call_result_id: nullableText(call.result_id),
        artifacts: [],
      }];
    });
    if (entries.length > 0) synthesised.set(index, toolResultRow(entries) as unknown as JsonObject);
  }
  if (synthesised.size === 0 && !anyMarked) return rows;
  const moved = new Map<number, JsonObject[]>();
  const kept: { row: JsonObject; index: number }[] = [];
  let segment = -1;
  for (const [index, row] of rows.entries()) {
    if (isToolStep(row)) segment = index;
    if (row.harness === true && isStartedRow(row)) continue;
    const owner = markedOwner(rows, index);
    if (owner !== undefined && owner !== segment) {
      moved.set(owner, [...(moved.get(owner) ?? []), row]);
      continue;
    }
    kept.push({ row, index });
  }
  const out: JsonObject[] = [];
  for (let at = 0; at < kept.length; at += 1) {
    const { row, index } = kept[at]!;
    out.push(row);
    if (!isToolStep(row)) continue;
    // The step's own result run: the result and system rows right after it.
    while (at + 1 < kept.length && (kept[at + 1]!.row.type === 'tool_result' || kept[at + 1]!.row.type === 'system')) out.push(kept[++at]!.row);
    out.push(...(moved.get(index) ?? []));
    const extra = synthesised.get(index);
    if (extra !== undefined) out.push(extra);
  }

  return out;
}

/**
 * A later row's result replaces only the SAME result (resultKey), occurrence
 * by occurrence: the k-th with a key in the later row replaces the k-th with
 * that key so far, in its position; any other is added after (TS 0.11.x #83).
 */
function fold(earlier: ToolResultRow, later: ToolResultRow): ToolResultRow {
  const results = [...earlier.tool_results];
  const keys = results.map(resultKey);
  const seen = new Map<string, number>();
  for (const entry of later.tool_results) {
    const key = resultKey(entry);
    const occurrence = seen.get(key) ?? 0;
    seen.set(key, occurrence + 1);
    let at = -1;
    for (let index = 0, found = 0; index < keys.length; index += 1) {
      if (keys[index] !== key) continue;
      if (found === occurrence) { at = index; break; }
      found += 1;
    }
    if (at === -1) { results.push(entry); keys.push(key); } else results[at] = entry;
  }
  const decisions = new Map<string, ApprovalDecisionEntry>();
  for (const entry of [...earlier.tool_approval_responses, ...later.tool_approval_responses]) decisions.set(entry.approval_id, entry);

  return toolResultRow(results, [...decisions.values()]);
}

/**
 * `row` without what its segment already replayed, or null when that leaves
 * nothing of a row that held something. Records what it replays now.
 */
function unreplayed(row: ToolResultRow, replayed: Map<string, number>, decided: Set<string>): ToolResultRow | null {
  if (row.tool_results.length === 0 && row.tool_approval_responses.length === 0) return row;
  const seen = new Map<string, number>();
  const results: ToolResultEntry[] = [];
  for (const entry of row.tool_results) {
    const key = resultKey(entry);
    seen.set(key, (seen.get(key) ?? 0) + 1);
    if (seen.get(key)! > (replayed.get(key) ?? 0)) results.push(entry);
  }
  for (const [key, count] of seen) replayed.set(key, Math.max(replayed.get(key) ?? 0, count));
  const decisions: ApprovalDecisionEntry[] = [];
  for (const entry of row.tool_approval_responses) {
    if (decided.has(entry.approval_id)) continue;
    decided.add(entry.approval_id);
    decisions.push(entry);
  }
  if (results.length === row.tool_results.length && decisions.length === row.tool_approval_responses.length) return row;

  return results.length === 0 && decisions.length === 0 ? null : toolResultRow(results, decisions);
}

/**
 * Which result an entry is, for folding (G-403, as the reference and TS
 * 0.11.x fold): its call id and tool name, each length-prefixed, and its
 * arguments as canonical JSON. Two calls that share an id but not their tool
 * or their arguments are two results.
 */
function resultKey(entry: ToolResultEntry): string {
  const known = KEYS.get(entry);
  if (known !== undefined) return known;
  const key = `${entry.tool_call_id.length}:${entry.tool_call_id}${entry.tool_name.length}:${entry.tool_name}${sortedJson(entry.args)}`;
  KEYS.set(entry, key);

  return key;
}

/** Each entry's key, computed once: arguments can be large, and a fold asks more than once. */
const KEYS = new WeakMap<ToolResultEntry, string>();

/**
 * JSON with every object's keys in sorted order (UTF-16 code units, as
 * canonical JSON sorts them), by the engine's own serialiser: the same
 * equality as canonical JSON for stored arguments, in one pass and without
 * an intermediate string per element (a call's arguments may be very large).
 */
function sortedJson(value: unknown): string {
  return JSON.stringify(value, (_key, item: unknown) => (typeof item === 'object' && item !== null && !Array.isArray(item)
    ? Object.fromEntries(Object.keys(item).sort().map((key) => [key, (item as Record<string, unknown>)[key]]))
    : item)) ?? 'null';
}

/**
 * Whether a stored row is a TOOL-CALL STEP (G-408, the reference's rule and
 * TS 0.11.x #92's): an assistant row whose `tool_calls` is a JSON ARRAY
 * holding AT LEAST ONE OBJECT; its calls are exactly those objects, in order
 * (stepCalls). Anything else (a JSON string, a number, null, a JSON object,
 * an array of strings, an array of JSON-string calls) is not a step for ANY
 * reader: not the paused step's calls, not a request row, not a mark's
 * target, and it replays as an assistant text row with no tool calls. The
 * ONE rule every reader takes the calls and the requests by.
 *
 * @internal
 */
export function isToolStep(row: JsonObject | undefined): boolean {
  return row?.type === 'assistant' && list(row.tool_calls).length > 0;
}

/** The calls of a tool-call step (isToolStep): its object calls, in order; none for any other row. @internal */
export function stepCalls(row: JsonObject | undefined): JsonObject[] {
  return isToolStep(row) ? list(row!.tool_calls) : [];
}

/** The approval requests of a tool-call step (isToolStep); none for any other row. @internal */
export function stepRequests(row: JsonObject | undefined): JsonObject[] {
  return isToolStep(row) ? list(row!.tool_approval_requests) : [];
}

/**
 * The approval ids an EARLIER tool-call step than `index` asked for: an
 * approval belongs ONLY to the step that first asked for it (G-408, TS
 * 0.11.x #92), so a later row repeating its id (a copy a host recorded,
 * keeping the old request beside a new call) asks for nothing.
 *
 * @internal
 */
export function askedBefore(rows: readonly JsonObject[], index: number): Set<string> {
  const asked = new Set<string>();
  for (const [id, step] of rowIndex(rows).firstAsker) if (step < index) asked.add(id);
  return asked;
}

/**
 * The approval requests the tool-call step at `index` OWNS: its own
 * requests (stepRequests) less any an earlier step already asked for
 * (askedBefore). Every reader of a step's requests takes them from here.
 *
 * @internal
 */
export function ownRequests(rows: readonly JsonObject[], index: number): JsonObject[] {
  const requests = stepRequests(rows[index]);
  if (requests.length === 0) return requests;
  const first = rowIndex(rows).firstAsker;
  return requests.filter((request) => typeof request.approval_id !== 'string' || (first.get(request.approval_id) ?? index) >= index);
}

/**
 * The key of the mark on every tool_result row the runtime writes that
 * carries tool results, and on its started rows (G-387 round 15, the
 * reference's shape): `{position}`, the 1-based thread position of the step
 * those results answer, the LAST key of the row. Such a row answers that step
 * wherever it lands (a tool-call step another writer records while a call
 * runs never takes the call's result). Honoured only on a row the harness
 * wrote (the reader view carries `harness: true` for it, see readerRows) and
 * only when it names an EARLIER tool-call step holding a call with the same
 * id and tool name as every result in the row; otherwise the row is its
 * segment's, which fails closed. Never replayed.
 *
 * @internal
 */
export const ANSWERS_STEP_KEY = 'harness_answers_step';

/**
 * The key of a STARTED row (G-387 round 15, option (i)): written by the
 * resolver under the private in-run token immediately BEFORE it runs one
 * approved call, in the call's own segment:
 * `{"type":"tool_result","tool_results":[],"tool_approval_responses":[],
 * "harness_started":{"tool_call_id","tool_name","approval_id"},
 * "harness_answers_step":{"position"}}`. A call with a started row and no
 * result is ANSWERED IN PROGRESS: a superseding write gives it no replaced
 * denial, it is not pending, and it never runs again. One that never gets a
 * result reads INTERRUPTED_RESULT. No TTL. Never replayed.
 *
 * @internal
 */
export const STARTED_KEY = 'harness_started';

/** The outcome of a started call that never got a result (no final period; the reference's text). @internal */
export const INTERRUPTED_RESULT = 'Tool call was interrupted, it may have run';

/**
 * The started row for `call` of the step at `position`, run under
 * `approvalId` (STARTED_KEY); null for a call that needs nobody's approval
 * (round 16: an UNGATED call gets its started row too, before its batch
 * runs, so a result that never lands reads interrupted, never "not run").
 * @internal
 */
export function startedRow(call: { id: string; name: string }, approvalId: string | null, position: number): JsonObject {
  return {
    type: 'tool_result',
    tool_results: [],
    tool_approval_responses: [],
    [STARTED_KEY]: { tool_call_id: call.id, tool_name: call.name, approval_id: approvalId },
    [ANSWERS_STEP_KEY]: { position },
  };
}

/** `row` marked as answering the step at `position` (ANSWERS_STEP_KEY), the mark its last key. @internal */
export function answeringStep<T extends JsonObject | ToolResultRow>(row: T, position: number): T {
  const { [ANSWERS_STEP_KEY]: _old, ...rest } = row as JsonObject;

  return { ...rest, [ANSWERS_STEP_KEY]: { position } } as unknown as T;
}

/** Whether `row` is a started row (by its shape alone; markedOwner decides whether it is honoured). @internal */
export function isStartedRow(row: JsonObject): boolean {
  const mark = row[STARTED_KEY];

  return row.type === 'tool_result' && isObject(mark) && typeof mark.tool_call_id === 'string' && typeof mark.tool_name === 'string';
}

/**
 * The index in `rows` of the step a harness row's honoured mark names, or
 * undefined (no mark, a host row, or a mark that fails validation: then the
 * row is its segment's). `rows` is the whole thread in order, so the row at
 * index i is at thread position i + 1 (Thread assigns positions so).
 *
 * @internal
 */
export function markedOwner(rows: readonly JsonObject[], index: number): number | undefined {
  return rowIndex(rows).owners[index];
}

/** markedOwner(), computed (rowIndex() keeps it). */
function ownerOf(rows: readonly JsonObject[], row: JsonObject, index: number): number | undefined {
  if (row.type !== 'tool_result' || row.harness !== true) return undefined;
  const mark = row[ANSWERS_STEP_KEY];
  if (!isObject(mark) || typeof mark.position !== 'number' || !Number.isSafeInteger(mark.position)) return undefined;
  const step = mark.position - 1;
  if (step < 0 || step >= index || !isToolStep(rows[step])) return undefined;
  const calls = stepCalls(rows[step]);
  const named = isStartedRow(row) ? [row[STARTED_KEY] as JsonObject] : list(row.tool_results);
  const known = (entry: JsonObject): boolean => calls.some((call) => text(call.id) === text(entry.tool_call_id) && text(call.name) === text(entry.tool_name));

  return named.every(known) ? step : undefined;
}

/**
 * What each call of the tool-call step at `index` has (G-387 round 15):
 * `result` (a result its step owns), `started` (a started row its step owns
 * and no result: answered in progress, or interrupted once no run is
 * resolving it), or null (open). A row's owner is the step its honoured
 * mark names (markedOwner); any other row is the segment's that holds it:
 * the rows after the step up to the next assistant row (G-399). A row owned
 * by another step counts for nothing where it sits. Results pair BY
 * POSITION (round 13), never by tool call id: each result with the FIRST
 * call without an outcome that has its id, its tool name and its arguments;
 * then its id and name; then its id (a started call last: it is still open
 * to its own result).
 *
 * @internal
 */
export function stepAnswers(rows: readonly JsonObject[], index: number): Answer[] {
  const known = rowIndex(rows).steps.get(index);
  return known === undefined ? stepCalls(rows[index]).map(() => null) : [...known.has];
}

type Answer = 'result' | 'started' | null;

/** One tool-call step's answers as the index's pass reads them, row by row (stepAnswers' rule). */
class StepAnswers {
  readonly calls: JsonObject[];
  readonly has: Answer[];
  readonly #args: string[];
  /** A result an EARLIER row of the step already holds (a copy: the same id, tool, arguments and result) answers nothing new (TS I-1). */
  readonly #prints = new Set<string>();

  constructor(row: JsonObject) {
    this.calls = stepCalls(row);
    this.has = this.calls.map(() => null);
    this.#args = this.calls.map((call) => JSON.stringify(usableArguments(call.arguments) ?? {}));
  }

  /** Apply one row the step owns; `honoured`: its mark named this step. */
  feed(row: JsonObject, honoured: boolean): void {
    const { calls, has } = this;
    if (honoured && isStartedRow(row)) {
      const started = row[STARTED_KEY] as JsonObject;
      const call = calls.findIndex((candidate, position) => has[position] === null && text(candidate.id) === text(started.tool_call_id) && text(candidate.name) === text(started.tool_name));
      if (call !== -1) has[call] = 'started';
      return;
    }
    const entries = Array.isArray(row.tool_results) ? list(row.tool_results)
      : Object.hasOwn(row, 'tool_call_id') ? [{ tool_call_id: row.tool_call_id ?? null, tool_name: row.name ?? null }] : [];
    const added: string[] = [];
    for (const entry of entries) {
      const id = text(entry.tool_call_id);
      const print = sortedJson([id, text(entry.tool_name), isObject(entry.args) ? entry.args : {}, text(entry.result)]);
      if (this.#prints.has(print)) continue;
      added.push(print);
      // A started call is still open to ITS result (the real one, wherever it lands).
      const open = (position: number): boolean => has[position] !== 'result' && text(calls[position]!.id) === id;
      const first = (test: (call: JsonObject, position: number) => boolean): number => {
        const unstarted = calls.findIndex((call, position) => has[position] === null && open(position) && test(call, position));
        return unstarted !== -1 ? unstarted : calls.findIndex((call, position) => open(position) && test(call, position));
      };
      let found = first((call, position) => call.name === entry.tool_name && this.#args[position] === JSON.stringify(isObject(entry.args) ? entry.args : {}));
      if (found === -1) found = first((call) => call.name === entry.tool_name);
      if (found === -1) found = first(() => true);
      if (found !== -1) has[found] = 'result';
    }
    for (const print of added) this.#prints.add(print);
  }
}

/**
 * Everything the readers need from one stored thread, read in ONE pass
 * (BRIEF-R16 item 7): no reader walks the thread once per step, per
 * approval or per row. Kept per rows array (rowIndex), so every reader of
 * one read of the thread shares it.
 */
interface RowIndex {
  /** markedOwner() of each row. */
  owners: (number | undefined)[];
  /** The tool-call step that FIRST asked for each approval id (G-408): askedBefore()'s rule. */
  firstAsker: Map<string, number>;
  /** Each tool-call step's answers (stepAnswers()). */
  steps: Map<number, StepAnswers>;
  /** The approvals each harness mark names within its own request's turn (markedApprovals()). */
  marked: Map<string, Set<string>>;
  /**
   * The decision that STANDS for each approval (round 16, BRIEF-R16 items 1
   * and 10) and the row it was recorded in: the first decision recorded
   * while its call had NO outcome (no started row, no result), except that
   * a later denial recorded while it still had none revokes an approval
   * (prevails(), G-401). A decision recorded once its call has an outcome is
   * never the answer, whoever wrote it.
   */
  standing: Map<string, { entry: ApprovalDecisionEntry; at: number }>;
}

const INDEXES = new WeakMap<readonly JsonObject[], { length: number; index: RowIndex }>();

/** The one-pass index of `rows` (RowIndex), built once per rows array. */
function rowIndex(rows: readonly JsonObject[]): RowIndex {
  const known = INDEXES.get(rows);
  if (known !== undefined && known.length === rows.length) return known.index;
  const index = buildIndex(rows);
  INDEXES.set(rows, { length: rows.length, index });
  return index;
}

function buildIndex(rows: readonly JsonObject[]): RowIndex {
  const owners: (number | undefined)[] = [];
  const firstAsker = new Map<string, number>();
  // The requests markedApprovals() reads: the first assistant row asking with a call id.
  const markAsker = new Map<string, number>();
  const steps = new Map<number, StepAnswers>();
  const marked = new Map<string, Set<string>>([[ENDED_BY_START_KEY, new Set()], [SUPERSEDED_KEY, new Set()]]);
  const standing = new Map<string, { entry: ApprovalDecisionEntry; at: number }>();
  // Each step's own requests (firstAsker's), to pair a decision with its call.
  const ownOf = new Map<number, JsonObject[]>();
  let segment: number | null = null;
  let lastAssistant = -1;

  const decide = (answer: JsonObject, at: number): void => {
    const entry = decision(answer);
    const step = firstAsker.get(entry.approval_id);
    if (step !== undefined) {
      // Counted only while its call (paired by position with the step's
      // calls that have no outcome yet) has none.
      const state = steps.get(step)!;
      const paired = pairStepApprovals(state.calls, ownOf.get(step) ?? [], (_call, position) => state.has[position] === null).approvalIds;
      const position = paired.indexOf(entry.approval_id);
      if (position === -1 || state.has[position] !== null) return;
    }
    if (prevails(standing.get(entry.approval_id)?.entry, entry.approved)) standing.set(entry.approval_id, { entry, at });
  };

  for (const [at, row] of rows.entries()) {
    const owner = ownerOf(rows, row, at);
    owners.push(owner);
    if (row.type === 'assistant') {
      lastAssistant = at;
      if (!isToolStep(row)) { segment = null; continue; }
      segment = at;
      steps.set(at, new StepAnswers(row));
      const own: JsonObject[] = [];
      for (const request of stepRequests(row)) {
        const id = request.approval_id;
        if (typeof id !== 'string') { own.push(request); continue; }
        if (!firstAsker.has(id)) firstAsker.set(id, at);
        if (firstAsker.get(id) === at) own.push(request);
        if (typeof request.tool_call_id === 'string' && !markAsker.has(id)) markAsker.set(id, at);
      }
      ownOf.set(at, own);
      continue;
    }
    if (row.type === 'tool_approval_response') { decide(row, at); continue; }
    if (row.type !== 'tool_result') continue;
    // The row's decisions first, against what the calls had BEFORE it.
    for (const answer of list(row.tool_approval_responses)) decide(answer, at);
    for (const [key, set] of marked) {
      const mark = row[key];
      if (!isObject(mark) || !Array.isArray(mark.approval_ids)) continue;
      for (const id of mark.approval_ids) if (typeof id === 'string' && markAsker.get(id) === lastAssistant) set.add(id);
    }
    // A 0.11.x replaced-step answer is an EARLIER step's (it names that
    // step's position): never this step's results.
    if (isReplacedStepRow(row)) continue;
    const target = owner ?? segment;
    if (target !== null) steps.get(target)?.feed(row, owner !== undefined);
  }

  return { owners, firstAsker, steps, marked, standing };
}

/**
 * THE step a decision applies to (G-397): the index of the assistant row the
 * run paused at the transcript tail, or null when a user row comes after the
 * last assistant row (a new turn ended every approval). The ONE selector:
 * the repeated-id guard, the resolver, the decision checks and the plan
 * readers all take the step from here, so a row recorded after the park (an
 * assistant row, a user row, a system or tool row) can never make one of
 * them read a different step from another. A step whose row asked nothing
 * has nothing to decide: each reader checks the requests on THIS row.
 *
 * @internal
 */
export function pausedStepIndex(rows: readonly JsonObject[]): number | null {
  for (let index = rows.length - 1; index >= 0; index -= 1) {
    const type = rows[index]!.type;
    if (type === 'user') return null;
    if (type === 'assistant') return index;
  }

  return null;
}

/**
 * The decisions recorded for the turn being resumed (the paused step,
 * pausedStepIndex()), by approval id, read from the stored rows.
 *
 * An approval has ONE answer, the one that STANDS (standingDecisions()):
 * the first one recorded while its call had no outcome, except that a later
 * DENIAL, recorded while the call still had none (no started row, no
 * result), revokes an approval (prevails(), G-401). Any other decision for
 * the same id (two reviewers, a retry racing the original, a row written by
 * hand, one recorded after the call started or ran) is never the one that
 * counts, here, in the plan's reading of the thread or in the replay, so
 * what ran and what the thread says was decided cannot disagree.
 */
export function turnDecisions(rows: readonly JsonObject[]): Map<string, ApprovalDecisionEntry> {
  const decisions = new Map<string, ApprovalDecisionEntry>();
  // The paused step (G-397): no decision applies to any other.
  const step = pausedStepIndex(rows);
  if (step === null || !isToolStep(rows[step])) return decisions;
  for (const [id, { entry, at }] of rowIndex(rows).standing) if (at > step) decisions.set(id, entry);

  return decisions;
}

/**
 * The decision that STANDS for every approval in the thread (round 16): see
 * turnDecisions(). One pass, shared by every reader (rowIndex).
 *
 * @internal
 */
export function standingDecisions(rows: readonly JsonObject[]): Map<string, ApprovalDecisionEntry> {
  return new Map([...rowIndex(rows).standing].map(([id, { entry }]) => [id, entry]));
}

/**
 * Whether a decision recorded after `earlier` is the answer (G-401, the
 * recommended option until the owner decides): the FIRST answer stands,
 * except that a later DENIAL revokes an earlier approval of a call that has
 * not started (the decision path accepts it only then, revocableApprovalIds).
 * An approval never overrides anything.
 *
 * @internal
 */
export function prevails(earlier: { approved: boolean } | boolean | undefined, approved: boolean): boolean {
  if (earlier === undefined) return true;
  const was = typeof earlier === 'boolean' ? earlier : earlier.approved;
  return was && !approved;
}

/**
 * The approval ids a person APPROVED whose call has not started (no started
 * row, no result): a denial may still revoke each (G-401's recommended
 * option). Nothing else may be decided again.
 *
 * @internal
 */
export function revocableApprovalIds(rows: readonly JsonObject[]): string[] {
  const decisions = turnDecisions(rows);
  return unresolvedApprovalIds(rows).filter((id) => decisions.get(id)?.approved === true);
}

/** Stored approval authority, kept separate from the provider replay view. */
export function approvalToolScope(rows: readonly JsonObject[]): { names: string[]; legacy: boolean } | null {
  // The paused step (G-397), as stored, and the same row in the view (which
  // folds a 0.3.0 request row into the assistant row before it).
  const step = pausedStepIndex(rows);
  if (step === null || !isToolStep(rows[step])) return null;
  const ordinal = rows.slice(0, step + 1).filter((candidate) => candidate.type === 'assistant').length;
  const row = threadView(rows).filter((candidate) => candidate.type === 'assistant')[ordinal - 1];
  if (row === undefined) return null;
  const asked = askedBefore(rows, step);
  const requests = list(row.tool_approval_requests).filter((request) => typeof request.approval_id !== 'string' || !asked.has(request.approval_id));
  if (requests.length === 0) return null;
  const stored = rows[step]!;
  const scope = stored.harness_tool_names;
  if (Array.isArray(scope) && scope.every((name): name is string => typeof name === 'string' && name !== '' && name !== '*')) {
    return {names: [...new Set(scope)], legacy: false};
  }
  // A step that repeats a call id approves nothing (G-392).
  if (pairStepApprovals(list(row.tool_calls), requests).repeated) return {names: [], legacy: true};
  const decisions = turnDecisions(rows);
  const approvedCalls = new Set(requests.filter(request => typeof request.approval_id === 'string' && decisions.get(request.approval_id)?.approved === true)
    .map(request => request.tool_call_id));
  const names = list(row.tool_calls).filter(call => approvedCalls.has(call.id) && typeof call.name === 'string' && call.name !== '' && call.name !== '*')
    .map(call => call.name as string);
  return {names: [...new Set(names)], legacy: true};
}

/**
 * The approval ids the run paused at the transcript tail is still waiting on:
 * issued by the last assistant step and not yet answered (G-354).
 *
 * Read from the stored rows, newest first. A user turn, or an assistant step
 * that asked nothing, ends the scope: nothing before it is pending. An id
 * answered once is no longer pending, and neither is one a harness mark
 * answered (a replaced step's denial, a start's ended row).
 */
export function pendingApprovalIds(rows: readonly JsonObject[]): string[] {
  // The paused step (G-397); its answers are read from the rows after it.
  const step = pausedStepIndex(rows);
  if (step === null) return [];
  // An approval a harness mark answered is not pending (G-404, BRIEF-R14):
  // a replaced step's denial or a start's ended row, each read within its
  // own request's turn, as every other reader reads them.
  const answered = new Set<string>([...supersededApprovals(rows), ...endedByStart(rows)]);
  for (const later of rows.slice(step + 1)) {
    if (later.type !== 'tool_result') continue;
    for (const response of list(later.tool_approval_responses)) {
      if (typeof response.approval_id === 'string') answered.add(response.approval_id);
    }
  }
  // Nor is one whose call already has an OUTCOME (round 16, TS run19 F2): a
  // result its step owns, or a started row. By the SAME pairing as
  // unresolvedApprovalIds() and the replay, so the two never disagree.
  const open = new Set(unresolvedApprovalIds(rows));
  const pending = ownRequests(rows, step)
    .map(request => request.approval_id)
    .filter((id): id is string => typeof id === 'string' && id !== '' && !answered.has(id) && open.has(id));

  return [...new Set(pending)];
}

/**
 * The approval ids the run paused at the transcript tail asked for whose
 * call has no result yet, pending OR decided (G-387 round 10): a resume would
 * act on each of them. Read newest first, as pendingApprovalIds() is; a user
 * turn ends the scope.
 */
export function unresolvedApprovalIds(rows: readonly JsonObject[]): string[] {
  // The paused step (G-397); its results are read from the rows after it.
  const step = pausedStepIndex(rows);
  if (step === null) return [];
  const row = rows[step]!;
  const requests = ownRequests(rows, step);
  const ids = (entries: JsonObject[]): string[] => [...new Set(entries
    .map((request) => request.approval_id)
    .filter((id): id is string => typeof id === 'string' && id !== ''))];

  // A step that repeats a call id: a result by id cannot say whose it is
  // (G-392), so every request is unresolved until the step is settled.
  if (pairStepApprovals(stepCalls(row), requests).repeated) return repeatedStepSettled(rows, step) ? [] : ids(requests);

  // By position, from the rows the step OWNS (round 15): a result its
  // honoured mark sends elsewhere is not this step's, and a started call is
  // answered in progress (a resume never acts on it again).
  const calls = stepCalls(row);
  const answered = stepResultsByPosition(rows, step);
  const open = (id: string): boolean => calls.some((call, position) => !answered[position] && text(call.id) === id);

  return ids(requests.filter((request) => typeof request.tool_call_id !== 'string' || open(request.tool_call_id)));
}

/**
 * The result a runner start's task row records for each call its child was
 * parked on (G-387 round 9): not run, whatever decision is recorded on it.
 * One fixed text in all three languages (round 12: the reference's), so a
 * thread reads the same whichever language wrote it. The text is for the
 * model; the harness recognises such a row only by its mark
 * (ENDED_BY_START_KEY), never by these words.
 *
 * @internal
 */
export function endedByStartResult(): string {
  return 'Not run: the delegation this call was proposed under ended.';
}

/**
 * The key of the mark a runner start puts on the result row it ends its
 * child's parked calls with (G-387 rounds 11-12): `{approval_ids: [...]}`,
 * the approvals those calls were asked under, in call order. The PHP
 * reference's shape and placement: a top-level key of the row, AFTER
 * `tool_approval_responses`, beside (never inside) the results. Only the
 * harness writes it: a tool returns a `result` string and nothing else of
 * the row, and the replay view rebuilds rows from their known keys, so the
 * model never sees it either. Pinned by harness-thread-rows thr-0014.
 *
 * @internal
 */
export const ENDED_BY_START_KEY = 'harness_ended_by_start';

export type EndedByStartRow = ToolResultRow & { harness_ended_by_start: { approval_ids: string[] } };

/**
 * The row a runner start records with its task row when it ends `calls`,
 * the calls of its child's parked step that have no result (G-387): each
 * with the one fixed result, never run, and the row marked with the
 * approvals they were asked under (`approvalIds`, one per call, undefined
 * for a call asked under none), in call order, each once.
 *
 * @internal Written by the runtime; the harness-thread-rows corpus drives it.
 */
export function endedByStartRow(calls: readonly ToolCallInput[], approvalIds: readonly (string | undefined)[]): EndedByStartRow {
  const approvals: string[] = [];
  for (const id of approvalIds) if (typeof id === 'string' && !approvals.includes(id)) approvals.push(id);

  return {
    ...toolResultRow(calls.map((call) => toolResultEntry(call, endedByStartResult()))),
    [ENDED_BY_START_KEY]: { approval_ids: approvals },
  };
}

/**
 * The key of the mark on the row that denies a superseded step's calls
 * (DECISION B, G-387 round 13): `{approval_ids: [...]}`, the approvals of the
 * denied calls in call order, the LAST key of the row, as the ended mark is.
 * Only the harness writes it (Thread.record()). The PHP reference's key.
 *
 * @internal
 */
export const SUPERSEDED_KEY = 'harness_superseded';

/** The result each open call of a superseded step is given; the same in every language (DECISION B). @internal */
export const SUPERSEDED_RESULT = 'Tool call was not run because a later step replaced it';

/**
 * Whether `message` supersedes the paused step (DECISION B, refined): any
 * assistant row WITH tool calls, always; an assistant text row or a user row
 * only when it is recorded OUTSIDE the run that owns the step.
 *
 * @internal
 */
export function supersedes(message: JsonObject, outside: boolean): boolean {
  if (isToolStep(message)) return true;

  return outside && (message.type === 'assistant' || message.type === 'user');
}

/**
 * The row that denies the paused step's open calls when it is superseded
 * (DECISION B), or null when nothing is paused or nothing of it is open.
 * `alsoResults` are result rows about to be recorded before the superseding
 * row (in the same batch). A call is open when no result is paired with it
 * by position: a call that ran has one, so the rest were waiting on a person.
 * Each open call gets ONE result, SUPERSEDED_RESULT, in call order; the row
 * is marked with the approvals they were asked under.
 *
 * @internal Written by Thread.record(); the harness-approval-pairing corpus pins it.
 */
export function supersededRow(rows: readonly JsonObject[], alsoResults: readonly JsonObject[] = []): (ToolResultRow & { harness_superseded: { approval_ids: string[] } }) | null {
  const step = pausedStepIndex(rows);
  if (step === null) return null;
  const calls = stepCalls(rows[step]);
  if (calls.length === 0) return null;
  const answered = stepResultsByPosition([...rows, ...alsoResults], step);
  const open = calls.map((_call, position) => position).filter((position) => !answered[position]);
  if (open.length === 0) return null;
  const paired = pairStepApprovals(calls, ownRequests(rows, step), (_call, position) => !answered[position]).approvalIds;
  const approvals: string[] = [];
  for (const position of open) {
    const id = paired[position];
    if (typeof id === 'string' && !approvals.includes(id)) approvals.push(id);
  }

  return {
    ...toolResultRow(open.map((position) => {
      const call = calls[position]!;
      return {
        tool_call_id: text(call.id),
        tool_name: text(call.name),
        args: usableArguments(call.arguments) ?? {},
        result: SUPERSEDED_RESULT,
        tool_call_result_id: nullableText(call.result_id),
        artifacts: [],
      };
    })),
    [SUPERSEDED_KEY]: { approval_ids: approvals },
  };
}

/**
 * The key of the mark the 0.11.x release line puts on the row that answers
 * a REPLACED step's calls: `{position}`, the thread position of the step it
 * answers, stored AFTER the later step. Main writes its own rows with
 * SUPERSEDED_KEY, before the superseding row; it READS both, so a thread a
 * 0.11.x harness wrote resumes with every call answered once. It also writes
 * this shape for an older step a 0.11.x thread left superseded and
 * unanswered (answerReplacedSteps), where only an after-the-fact row fits.
 *
 * @internal
 */
export const REPLACED_STEP_KEY = 'harness_replaced_step';

/**
 * The approval ids of the calls a 0.11.x-shape replaced-step answer denied
 * (REPLACED_STEP_KEY): each request of the step it names (the row at index
 * position - 1 of the whole thread) whose call the row answers. Never acted
 * on, so never approved work.
 *
 * @internal
 */
export function replacedStepApprovals(rows: readonly JsonObject[]): Set<string> {
  const out = new Set<string>();
  for (const row of rows) {
    if (!isReplacedStepRow(row)) continue;
    const step = ((row[REPLACED_STEP_KEY] as JsonObject).position as number) - 1;
    if (!Number.isSafeInteger(step) || step < 0 || !isToolStep(rows[step])) continue;
    const answered = new Set(list(row.tool_results).map((entry) => text(entry.tool_call_id)));
    for (const request of ownRequests(rows, step)) {
      if (typeof request.approval_id === 'string' && answered.has(text(request.tool_call_id))) out.add(request.approval_id);
    }
  }
  return out;
}

/** Whether a stored row answers a replaced step (the 0.11.x shape). @internal */
export function isReplacedStepRow(row: JsonObject): boolean {
  const mark = row[REPLACED_STEP_KEY];
  return row.type === 'tool_result' && isObject(mark) && typeof mark.position === 'number';
}

/** A stored row with its thread position and the run that recorded it. @internal */
export type PositionedRow = { position: number; message: JsonObject; runId: string | null };

/**
 * Every tool-call step BEFORE the paused one that a later row superseded
 * (by the same rule as Thread.record: a later assistant row with tool calls,
 * or an assistant or user row outside the runs that own it) and that still
 * has a gated call with no answer: no result of its own before the
 * superseding row, no harness_superseded denial, no replaced-step answer
 * naming its position. Such a step exists only in a thread an older
 * harness wrote; each is answered once, with one replaced-step row.
 *
 * @internal
 */
export function unansweredReplacedSteps(rows: readonly PositionedRow[], gated: (name: string) => boolean): { position: number; calls: JsonObject[] }[] {
  const messages = rows.map((row) => row.message);
  const paused = pausedStepIndex(messages);
  const answeredAt = new Set(messages.filter((row) => isReplacedStepRow(row)).map((row) => (row[REPLACED_STEP_KEY] as JsonObject).position as number));
  const out: { position: number; calls: JsonObject[] }[] = [];
  for (const [index, step] of messages.entries()) {
    if (index === paused || !isToolStep(step)) continue;
    if (answeredAt.has(rows[index]!.position)) continue;
    // ONE TRUST SIGNAL (round 16): only rows the runtime wrote make a run an
    // owner (a reader row the harness wrote carries `harness: true`).
    const owners = new Set<string>();
    if (rows[index]!.runId !== null && step.harness === true) owners.add(rows[index]!.runId!);
    let end = -1;
    for (let at = index + 1; at < rows.length; at += 1) {
      const { message, runId } = rows[at]!;
      if (isToolStep(message)) { end = at; break; }
      // A result row a harness mark gives to another step is not this one's (round 15).
      const owner = markedOwner(messages, at);
      if (message.type === 'tool_result' && message.harness === true && runId !== null && list(message.tool_results).length > 0 && (owner === undefined || owner === index)) owners.add(runId);
      if ((message.type === 'user' || message.type === 'assistant') && (runId === null || !owners.has(runId))) { end = at; break; }
    }
    if (end === -1) continue;
    // Its own results: its segment's, and any a harness row names it for
    // (round 15), read over the whole thread (positions from 1).
    const answered = stepResultsByPosition(messages, index);
    const calls = stepCalls(step);
    const open = calls.filter((call, at) => !answered[at] && gated(text(call.name)));
    if (open.length > 0) out.push({ position: rows[index]!.position, calls: open });
  }

  return out;
}

/** The 0.11.x replaced-step answer row, byte for byte as that line writes it. @internal */
export function replacedStepRow(position: number, calls: readonly JsonObject[]): JsonObject {
  return {
    ...toolResultRow(calls.map((call) => toolResultEntry({
      id: typeof call.id === 'string' ? call.id : '',
      name: typeof call.name === 'string' ? call.name : '',
      arguments: isObject(call.arguments) ? call.arguments : {},
      resultId: typeof call.result_id === 'string' ? call.result_id : null,
    }, SUPERSEDED_RESULT))),
    [REPLACED_STEP_KEY]: { position },
  } as JsonObject;
}

/**
 * The approval ids a later step superseded with their call open (DECISION
 * B): read by the same rule as endedByStart(), from the harness's own mark.
 *
 * @internal
 */
export function supersededApprovals(rows: readonly JsonObject[]): Set<string> {
  return markedApprovals(rows, SUPERSEDED_KEY);
}

/**
 * The approval ids whose call a runner start's task row ended as not run
 * (G-387 round 10): their decision, if any, was never acted on, so it reads
 * as ended, never as approved.
 *
 * The rule (round 13, the same in every language): for each approval, its
 * request row; the approval is ended when ANY result row after that row and
 * before the next assistant row carries the mark and the mark names this
 * approval. Not only the first row carrying its call id (round 12): in a
 * step whose calls share an id, an ungated call's result, or the immediate
 * result of a call with unusable arguments, carries that id too, unmarked,
 * and must not hide the start's mark. Only the harness writes the mark, so
 * this is no looser. Never by the result text (a tool can return any text),
 * and never by tool_call id across the thread (providers reuse call ids: one
 * per function name, or numbered per response). Pinned by the
 * harness-thread-rows corpus.
 *
 * @internal
 */
export function endedByStart(rows: readonly JsonObject[]): Set<string> {
  return markedApprovals(rows, ENDED_BY_START_KEY);
}

/** The approvals a harness mark under `key` names, each within its own request's turn (the endedByStart() rule). */
function markedApprovals(rows: readonly JsonObject[], key: string): Set<string> {
  // The step that FIRST asked for an approval owns it (G-408); read in the
  // index's one pass (rowIndex).
  return new Set(rowIndex(rows).marked.get(key));
}

/**
 * Every approval id the run paused at the transcript tail asked for, answered
 * or not; empty when the tail is a user turn or an assistant step that asked
 * nothing. A turn with no new message continues exactly these calls (G-366).
 */
export function pausedApprovalIds(rows: readonly JsonObject[]): string[] {
  const step = pausedStepIndex(rows);
  if (step === null) return [];

  return [...new Set(ownRequests(rows, step)
    .map(request => request.approval_id)
    .filter((id): id is string => typeof id === 'string' && id !== ''))];
}

/**
 * The model steps the session's own rows show the run paused at the
 * transcript tail has taken, or null when nothing at the tail is waiting on
 * an approval: one assistant row per model step since the turn's user row.
 *
 * NOT the run's step count (G-387): a subagent child's steps are on the
 * child's rows, never here. That count is `harness_steps_used` on the row the
 * run parked on. This is the floor a resume never counts below, and what a
 * row written before that field existed is counted by.
 */
export function approvalRunSteps(rows: readonly JsonObject[]): number | null {
  if (approvalToolScope(rows) === null) return null;
  let steps = 0;
  for (let index = rows.length - 1; index >= 0; index -= 1) {
    const type = rows[index]!.type;
    if (type === 'user') break;
    if (type === 'assistant') steps += 1;
  }
  return steps;
}

/** What a named agent had spent in its run tree when the run paused for a person. */
export type PausedAgentSpend = { cost_usd: number; unmetered_runs: number };

/** A subagent child run's model binding, as stored on the row it paused on. */
export type PausedBinding = {
  session: { provider: string | null; model: string | null };
  declared: { provider: string | null; model: string | null };
  parent_mode: { provider: string | null; model: string | null };
};

/**
 * What the run paused at the transcript tail carries into its resume, or null
 * when nothing at the tail is waiting on an approval: its named agent's spend
 * (`harness_agent_spend`) and, for a subagent's child run, the provider and
 * model binding it ran under (`harness_binding`). Both are written on the
 * assistant row that asked for the approvals.
 *
 * A spend that is present but malformed is read as one call that reported no
 * cost, so a spend ceiling refuses (`budget_unpriced`) rather than starting
 * again from nothing. A malformed binding is read as none. A row written
 * before these fields existed carries neither.
 */
export function pausedRunState(rows: readonly JsonObject[]): {
  agentSpend: PausedAgentSpend | null;
  binding: PausedBinding | null;
  tree: { session: string; run: string } | null;
  planLedger: { session: string; plan: string } | null;
  /**
   * The steps the parked run had used, its subagent children's included
   * (`harness_steps_used`, G-387): null on a row written before the field
   * existed, `invalid` when it is present and not a non-negative safe
   * integer (the resume then fails closed).
   */
  stepsUsed: number | null | 'invalid';
} | null {
  if (approvalToolScope(rows) === null) return null;
  const step = pausedStepIndex(rows);
  if (step === null) return null;
  const paused = rows[step]!;

  let agentSpend: PausedAgentSpend | null = null;
  if (paused.harness_agent_spend !== undefined) {
    const spend = paused.harness_agent_spend;
    const cost = isObject(spend) ? spend.cost_usd : undefined;
    const unmetered = isObject(spend) ? spend.unmetered_runs : undefined;
    agentSpend = typeof cost === 'number' && Number.isFinite(cost) && cost >= 0
      && typeof unmetered === 'number' && Number.isSafeInteger(unmetered) && unmetered >= 0
      ? { cost_usd: cost, unmetered_runs: unmetered }
      : { cost_usd: 0, unmetered_runs: 1 };
  }

  return {
    agentSpend,
    binding: storedBinding(paused.harness_binding),
    tree: storedPair(paused.harness_tree, 'session', 'run') as { session: string; run: string } | null,
    planLedger: storedPair(paused.harness_plan_ledger, 'session', 'plan') as { session: string; plan: string } | null,
    stepsUsed: !Object.hasOwn(paused, 'harness_steps_used') ? null
      : (typeof paused.harness_steps_used === 'number' && Number.isSafeInteger(paused.harness_steps_used) && paused.harness_steps_used >= 0 ? paused.harness_steps_used : 'invalid'),
  };
}

/** A subagent child's run tree or shared plan budget reference, as stored on the row it paused on; null when absent or malformed. */
function storedPair(value: JsonValue | undefined, first: string, second: string): Record<string, string> | null {
  if (!isObject(value)) return null;
  const a = value[first], b = value[second];

  return typeof a === 'string' && a !== '' && typeof b === 'string' && b !== '' ? { [first]: a, [second]: b } : null;
}

/** A stored model binding (`{session, declared, parent_mode}`), or null when absent or malformed. @internal */
export function storedBinding(value: JsonValue | undefined): PausedBinding | null {
  if (!isObject(value)) return null;
  const choice = (entry: JsonValue | undefined): { provider: string | null; model: string | null } | null => {
    if (!isObject(entry)) return null;
    const valid = (field: JsonValue | undefined): field is string | null => field === null || (typeof field === 'string' && field !== '');
    return valid(entry.provider) && valid(entry.model) ? { provider: entry.provider, model: entry.model } : null;
  };
  const session = choice(value.session), declared = choice(value.declared), parentMode = choice(value.parent_mode);

  return session === null || declared === null || parentMode === null ? null : { session, declared, parent_mode: parentMode };
}

/**
 * The ids of stored calls whose arguments cannot be used. Read from the rows as
 * written, because the thread view has already replaced such arguments with
 * none, and an approved call must not run on an empty set nobody sent.
 *
 * Read from the one turn being resumed, the paused step (pausedStepIndex),
 * never the whole thread: providers that number calls per response reuse
 * ids, and an older turn's call must not decide a newer one.
 */
export function unusableStoredCalls(rows: readonly JsonObject[]): Set<string> {
  const unusable = new Set<string>();
  // The paused step (G-397), never another turn's.
  const step = pausedStepIndex(rows);
  const turn = step === null ? undefined : rows[step];

  for (const call of list(turn?.tool_calls)) {
    if (usableArguments(call.arguments) === null) unusable.add(text(call.id));
  }

  return unusable;
}

/**
 * The ids of the calls in one assistant turn whose approval expires rather
 * than runs: a request for the call has no marker, and the call is stored as
 * no arguments, which is also how 0.11.0 and earlier stored a call whose
 * arguments were a bare number or boolean.
 *
 * Read from the one turn being resumed, never the whole thread: providers that
 * number calls per response reuse ids, and an older turn's request must not
 * decide a newer one. Pass the turn as the thread view gives it; a call whose
 * stored arguments cannot be used is refused before this is asked.
 */
export function ambiguousCalls(assistant: AssistantRow): Set<string> {
  const unmarked = new Set(assistant.tool_approval_requests
    .filter(request => request.arguments_format !== ARGUMENTS_AS_SENT)
    .map(request => request.tool_call_id));

  return new Set(assistant.tool_calls
    .filter(call => unmarked.has(call.id) && Object.keys(call.arguments).length === 0)
    .map(call => call.id));
}

/**
 * The result a call gets when its step repeats a tool call id (G-392): not
 * run, whatever was decided. The harness recognises such an entry only by
 * its mark (REPEATED_CALL_ID under NOT_RUN_KEY), never by this text.
 *
 * @internal
 */
export const REPEATED_CALL_ID_RESULT = 'Tool call was not run because another call in this step has the same id';
/** @internal */
export const REPEATED_CALL_ID = 'repeated_call_id';
/**
 * The key of the per-entry mark on a call denied because its step repeats
 * a call id (G-392), beside `approval_id`. Not the ended-by-start mark,
 * which is the row-level ENDED_BY_START_KEY (G-387 round 12).
 *
 * @internal
 */
export const NOT_RUN_KEY = 'not_run';

/** Every call counts as gated: the pairing when the caller cannot tell (a stored row read alone). */
const EVERY_CALL = (): boolean => true;

/**
 * Each call of one assistant step paired with its OWN approval request, by
 * position (G-392), and whether the step repeats a tool call id.
 *
 * Requests are written in call order, one per gated call, so the n-th request
 * belongs to the next unpaired call it names. Never a map keyed by call id:
 * providers repeat ids within one response (Gemini names a call by its
 * function), and one key would let one approval decide two calls.
 *
 * `gated`, when given, says which calls needed approval: a request is paired
 * with such a call first.
 *
 * `repeated` is true when a call id repeats among the step's calls (an empty
 * id is an id like any other) or among its requests, or an approval id
 * repeats among its requests. Such a step cannot be
 * paired safely, and none of its gated calls may run.
 *
 * @internal
 */
export function pairStepApprovals(
  calls: readonly JsonObject[],
  requests: readonly JsonObject[],
  gated: (call: JsonObject, position: number) => boolean = EVERY_CALL,
): { approvalIds: (string | undefined)[]; repeated: boolean } {
  const ids = calls.map((call) => text(call.id));
  const asked = requests.map((request) => text(request.tool_call_id));
  const approvalIds: (string | undefined)[] = ids.map(() => undefined);
  let cursor = 0;
  for (const [index, request] of requests.entries()) {
    const want = asked[index]!;
    // A call that needs approval first, so an ungated call sharing the id is
    // not taken for the gated one; then any call that names it.
    const free = (position: number, id: string): boolean => approvalIds[position] === undefined && id === want;
    let at = ids.findIndex((id, position) => position >= cursor && free(position, id) && gated(calls[position]!, position));
    if (at === -1) at = ids.findIndex((id, position) => free(position, id) && gated(calls[position]!, position));
    // A request pairs only with a GATED call (round 13): when the caller says
    // which calls are gated, an ungated call never takes an approval.
    if (at === -1 && gated === EVERY_CALL) at = ids.findIndex((id, position) => position >= cursor && free(position, id));
    if (at === -1 && gated === EVERY_CALL) at = ids.findIndex((id, position) => free(position, id));
    if (at === -1) continue;
    approvalIds[at] = text(request.approval_id);
    cursor = at + 1;
  }

  // A repeated APPROVAL id across requests (two calls' requests carrying
  // one approval id) cannot say which call a decision belongs to either: the
  // step fails closed like one that repeats a call id (G-397, round 13).
  const approvals = requests.map((request) => text(request.approval_id));
  return { approvalIds, repeated: new Set(ids).size !== ids.length || new Set(asked).size !== asked.length || new Set(approvals).size !== approvals.length };
}

/**
 * The entries of `entries` that are NEW for the step at `index` (round 15:
 * exactly one stored outcome per call): by the fold's key (call id, tool
 * name, canonical arguments), an entry is new only beyond the most any one
 * row the step OWNS (stepAnswers' rule) holds of that key.
 *
 * @internal
 */
/** The fold's key of a stored result entry, read defensively: call id, tool name, canonical arguments. */
function foldKey(entry: JsonObject): string {
  return sortedJson([text(entry.tool_call_id), text(entry.tool_name), isObject(entry.args) ? entry.args : {}]);
}

export function newResults<T extends JsonObject | ToolResultEntry>(rows: readonly JsonObject[], index: number, entries: readonly T[]): T[] {
  const most = new Map<string, number>();
  let inSegment = true;
  for (let at = index + 1; at < rows.length; at += 1) {
    const row = rows[at]!;
    if (row.type === 'assistant') inSegment = false;
    if (row.type !== 'tool_result' || isReplacedStepRow(row)) continue;
    const owner = markedOwner(rows, at);
    if (owner === undefined ? !inSegment : owner !== index) continue;
    const here = new Map<string, number>();
    for (const entry of list(row.tool_results)) {
      const key = foldKey(entry);
      here.set(key, (here.get(key) ?? 0) + 1);
    }
    for (const [key, count] of here) most.set(key, Math.max(most.get(key) ?? 0, count));
  }
  const seen = new Map<string, number>();
  return entries.filter((entry) => {
    const key = foldKey(entry as JsonObject);
    seen.set(key, (seen.get(key) ?? 0) + 1);
    return seen.get(key)! > (most.get(key) ?? 0);
  });
}

/**
 * Which calls of the stored assistant step at `index` are answered: a result
 * its step owns, or a started row (answered in progress). stepAnswers()
 * says which.
 *
 * @internal
 */
export function stepResultsByPosition(rows: readonly JsonObject[], index: number): boolean[] {
  return stepAnswers(rows, index).map((answer) => answer !== null);
}

/**
 * Whether the step the run paused at the transcript tail (the last assistant
 * row, unless a user row comes after it) repeats a call id, among its calls
 * or its approval requests (G-392). Every decision recorded for such a step
 * is recorded as a denial (round 13, as the PHP reference records it).
 *
 * @internal
 */
export function pausedStepRepeatsCallIds(rows: readonly JsonObject[]): boolean {
  const step = pausedStepIndex(rows);

  return step !== null && pairStepApprovals(stepCalls(rows[step]), ownRequests(rows, step)).repeated;
}

/**
 * True when the step at `index` (an assistant row of `rows`, as stored) that
 * repeats a call id has had its gated calls answered: the harness wrote its
 * not-run entries for them (G-392), a runner start ended them, or (a step a
 * resolved before this check existed) every call already has a result.
 *
 * @internal
 */
export function repeatedStepSettled(rows: readonly JsonObject[], index: number): boolean {
  const calls = stepCalls(rows[index]).map((call) => text(call.id));
  const counts = new Map<string, number>();
  for (const row of rows.slice(index + 1)) {
    if (row.type !== 'tool_result' || isReplacedStepRow(row)) continue;
    const entries = list(row.tool_results);
    if (isObject(row[ENDED_BY_START_KEY]) || isObject(row[SUPERSEDED_KEY]) || entries.some((entry) => entry[NOT_RUN_KEY] === REPEATED_CALL_ID)) return true;
    const here = new Map<string, number>();
    for (const entry of entries) here.set(text(entry.tool_call_id), (here.get(text(entry.tool_call_id)) ?? 0) + 1);
    for (const [id, count] of here) counts.set(id, Math.max(counts.get(id) ?? 0, count));
  }
  const needed = new Map<string, number>();
  for (const id of calls) needed.set(id, (needed.get(id) ?? 0) + 1);

  return calls.length > 0 && [...needed].every(([id, count]) => (counts.get(id) ?? 0) >= count);
}

/**
 * The approval ids whose call the harness denied because its step repeats a
 * tool call id (G-392): never approved work, whatever was decided. Only for
 * the approval the mark names, asked for by a request row before it for the
 * same call id, as for endedByStart().
 *
 * @internal
 */
export function repeatedIdDenials(rows: readonly JsonObject[]): Set<string> {
  const asked = new Map<string, string>();
  const denied = new Set<string>();
  for (const row of rows) {
    if (row.type === 'assistant') {
      for (const request of stepRequests(row)) {
        if (typeof request.approval_id === 'string' && typeof request.tool_call_id === 'string' && !asked.has(request.approval_id)) asked.set(request.approval_id, request.tool_call_id);
      }
      continue;
    }
    if (row.type !== 'tool_result') continue;
    for (const result of list(row.tool_results)) {
      const approval = result.approval_id;
      if (result[NOT_RUN_KEY] === REPEATED_CALL_ID && typeof approval === 'string'
        && asked.has(approval) && asked.get(approval) === result.tool_call_id) denied.add(approval);
    }
  }

  return denied;
}

function normalise(row: JsonObject, view: JsonObject[]): JsonObject | null {
  switch (row.type) {
    case 'assistant': {
      const requests = stepRequests(row);
      return assistantRow(
        typeof row.content === 'string' ? row.content : '',
        stepCalls(row).map((call) => ({
          id: text(call.id),
          name: text(call.name),
          arguments: usableArguments(call.arguments) ?? {},
          resultId: nullableText(call.result_id),
          reasoningId: nullableText(call.reasoning_id),
          reasoningSummary: Array.isArray(call.reasoning_summary) ? call.reasoning_summary : null,
        })),
        isObject(row.additional_content) ? row.additional_content : {},
        requests.map((request) => ({
          approval_id: text(request.approval_id),
          tool_call_id: text(request.tool_call_id),
          ...(request.arguments_format === ARGUMENTS_AS_SENT ? { arguments_format: ARGUMENTS_AS_SENT } : {}),
        })),
      );
    }

    case 'tool_result':
      // A 0.11.x replaced-step answer is stored after the step that replaced
      // the step it answers: never shown as that later step's results.
      if (isReplacedStepRow(row)) return null;
      if (Array.isArray(row.tool_results) || Array.isArray(row.tool_approval_responses)) {
        return toolResultRow(
          list(row.tool_results).map((entry) => ({
            tool_call_id: text(entry.tool_call_id),
            tool_name: text(entry.tool_name),
            args: isObject(entry.args) ? entry.args : {},
            result: text(entry.result),
            tool_call_result_id: nullableText(entry.tool_call_result_id),
            artifacts: Array.isArray(entry.artifacts) ? entry.artifacts : [],
          })),
          list(row.tool_approval_responses).map(decision),
        );
      }

      // 0.3.0 and earlier: one row per call, with no arguments on it.
      return toolResultRow([
        {
          tool_call_id: text(row.tool_call_id),
          tool_name: text(row.name),
          args: {},
          result: text(row.result),
          tool_call_result_id: null,
          artifacts: [],
        },
      ]);

    case 'tool_approval_request': {
      // 0.3.0 and earlier kept the request in its own row, keyed by the CALL id.
      // It belongs to the assistant turn before it.
      const assistant = [...view].reverse().find((candidate) => candidate.type === 'assistant') as AssistantRow | undefined;

      for (const approval of list(row.approvals)) {
        assistant?.tool_approval_requests.push({ approval_id: text(approval.id), tool_call_id: text(approval.id) });
      }

      return null;
    }

    case 'tool_approval_response':
      return toolResultRow([], [decision(row)]);

    default: {
      // The reader view's trust flag (readerRows) is never sent anywhere.
      if (!Object.hasOwn(row, 'harness')) return row;
      const { harness: _harness, ...rest } = row;
      return rest;
    }
  }
}

function decision(value: JsonObject): ApprovalDecisionEntry {
  return { approval_id: text(value.approval_id), approved: value.approved === true, reason: nullableText(value.reason) };
}

function list(value: JsonValue | undefined): JsonObject[] {
  return Array.isArray(value) ? value.filter(isObject) : [];
}

function isObject(value: unknown): value is JsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function text(value: JsonValue | undefined): string {
  return typeof value === 'string' ? value : value === undefined || value === null ? '' : JSON.stringify(value);
}

function nullableText(value: JsonValue | undefined): string | null {
  return typeof value === 'string' ? value : null;
}
