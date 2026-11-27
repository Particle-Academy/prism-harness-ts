import type { JsonObject, JsonValue } from './json.js';
import { usableArguments } from './tool-arguments.js';

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
};

/**
 * The marker on an approval request whose call was stored exactly as the model
 * sent it. 0.11.0 stored a call whose arguments were a bare number or boolean
 * as `{}`, the same as a real call with no arguments, and 0.11.1 and earlier
 * wrote no marker; such an approval expires instead of running.
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
 * by its tool call id and a decision by its approval id, and a later one
 * replaces an earlier one in the earlier one's position.
 */
export function threadView(rows: readonly JsonObject[]): JsonObject[] {
  const view: JsonObject[] = [];

  for (const row of rows) {
    const normalised = normalise(row, view);

    if (normalised === null) continue;

    const previous = view.at(-1);

    if (normalised.type === 'tool_result' && previous?.type === 'tool_result') {
      view[view.length - 1] = fold(previous as ToolResultRow, normalised as ToolResultRow);
    } else {
      view.push(normalised);
    }
  }

  return onceEach(view);
}

/**
 * Each result of a step EXACTLY ONCE in its step's segment (G-405), across
 * result rows that are not consecutive (a host row, a new message between
 * them): a later occurrence of the same result (resultKey, occurrence by
 * occurrence) replaces the earlier one in its position and is not shown
 * again. A result row left with nothing to show is dropped.
 */
function onceEach(view: JsonObject[]): JsonObject[] {
  let shown = new Map<string, { row: number; entry: number }[]>();
  const removed = new Set<string>();
  view.forEach((row, at) => {
    if (isToolCallStep(row)) { shown = new Map(); return; }
    if (row.type !== 'tool_result') return;
    const results = (row as ToolResultRow).tool_results;
    const here = new Map<string, number>();
    results.forEach((entry, index) => {
      const key = resultKey(entry);
      const occurrence = here.get(key) ?? 0;
      here.set(key, occurrence + 1);
      const earlier = shown.get(key) ?? [];
      const target = earlier[occurrence];
      if (target === undefined) {
        earlier.push({ row: at, entry: index });
        shown.set(key, earlier);
        return;
      }
      // The same result again: it replaces the one already shown.
      (view[target.row] as ToolResultRow).tool_results[target.entry] = entry;
      removed.add(`${at}:${index}`);
    });
  });
  if (removed.size === 0) return view;
  return view.flatMap((row, at) => {
    if (row.type !== 'tool_result') return [row];
    const result = row as ToolResultRow;
    const kept = result.tool_results.filter((_entry, index) => !removed.has(`${at}:${index}`));
    if (kept.length === result.tool_results.length) return [row];
    if (kept.length === 0 && result.tool_approval_responses.length === 0) return [];
    return [toolResultRow(kept, result.tool_approval_responses)];
  });
}

/** Stored approval authority, kept separate from the provider replay view. */
export function approvalToolScope(rows: readonly JsonObject[]): { names: string[]; legacy: boolean } | null {
  const decisions = new Map<string, boolean>();
  for (const row of threadView(rows).reverse()) {
    if (row.type === 'tool_result') {
      for (const response of list(row.tool_approval_responses)) {
        if (typeof response.approval_id === 'string' && !decisions.has(response.approval_id)) decisions.set(response.approval_id, response.approved === true);
      }
      continue;
    }
    if (row.type === 'user') return null;
    if (row.type !== 'assistant') continue;
    const requests = list(row.tool_approval_requests);
    if (requests.length === 0) return null;
    const ids = new Set(requests.map(request => request.approval_id));
    const stored = [...rows].reverse().find(candidate => candidate.type === 'assistant'
      && list(candidate.tool_approval_requests).length === requests.length
      && list(candidate.tool_approval_requests).every(request => ids.has(request.approval_id)));
    const scope = stored?.harness_tool_names;
    if (Array.isArray(scope) && scope.every((name): name is string => typeof name === 'string' && name !== '' && name !== '*')) {
      return {names: [...new Set(scope)], legacy: false};
    }
    // A step that repeats a call id approves nothing (G-392).
    if (repeatsCallIds(list(row.tool_calls), requests)) return {names: [], legacy: true};
    const approvedCalls = new Set(requests.filter(request => typeof request.approval_id === 'string' && decisions.get(request.approval_id) === true)
      .map(request => request.tool_call_id));
    const names = list(row.tool_calls).filter(call => approvedCalls.has(call.id) && typeof call.name === 'string' && call.name !== '' && call.name !== '*')
      .map(call => call.name as string);
    return {names: [...new Set(names)], legacy: true};
  }
  return null;
}

/**
 * The model steps the run paused at the transcript tail has already taken, or
 * null when nothing at the tail is waiting on an approval.
 *
 * A run that pauses for a person and is resumed is the SAME run, so its step
 * limit continues rather than starting again. Counted from the stored rows
 * (one assistant row per model step since the turn's user row), so the count
 * survives a reload, a different worker and a deploy, with no extra state.
 */
/**
 * The model steps the turn at the transcript tail has taken so far, since
 * the turn's last user row, whether the turn is paused, finished or ran out of
 * steps: each assistant row the harness recorded (one per model step), and
 * each tool-call step another writer recorded. An assistant text row a host
 * recorded is not a model step. An empty send continues that turn, so it
 * starts from this count and never reopens the turn's budget.
 *
 * @internal
 */
export function turnStepsSoFar(rows: readonly StoredRow[]): number {
  let steps = 0;
  for (let index = rows.length - 1; index >= 0; index -= 1) {
    const { message, harness } = rows[index]!;
    if (message.type === 'user') break;
    if (message.type === 'assistant' && (harness === true || isToolCallStep(message))) steps += 1;
  }
  return steps;
}

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

/**
 * The ids of stored calls whose arguments cannot be used. Read from the rows as
 * written, because the thread view has already replaced such arguments with
 * none, and an approved call must not run on an empty set nobody sent.
 */
export function unusableStoredCalls(rows: readonly JsonObject[]): Set<string> {
  const unusable = new Set<string>();

  for (const row of rows) {
    if (row.type !== 'assistant') continue;
    for (const call of list(row.tool_calls)) {
      if (usableArguments(call.arguments) === null) unusable.add(text(call.id));
    }
  }

  return unusable;
}

/**
 * The ids of the calls in one assistant turn whose approval expires rather
 * than runs: a request for the call has no marker, and the call is stored as
 * no arguments, which is also how 0.11.0 stored a call whose arguments were a
 * bare number or boolean.
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

function normalise(row: JsonObject, view: JsonObject[]): JsonObject | null {
  switch (row.type) {
    case 'assistant':
      return assistantRow(
        typeof row.content === 'string' ? row.content : '',
        list(row.tool_calls).map((call) => ({
          id: text(call.id),
          name: text(call.name),
          arguments: usableArguments(call.arguments) ?? {},
          resultId: nullableText(call.result_id),
          reasoningId: nullableText(call.reasoning_id),
          reasoningSummary: Array.isArray(call.reasoning_summary) ? call.reasoning_summary : null,
        })),
        isObject(row.additional_content) ? row.additional_content : {},
        list(row.tool_approval_requests).map((request) => ({
          approval_id: text(request.approval_id),
          tool_call_id: text(request.tool_call_id),
          ...(request.arguments_format === ARGUMENTS_AS_SENT ? { arguments_format: ARGUMENTS_AS_SENT } : {}),
        })),
      );

    case 'tool_result':
      // The answer to a replaced step is stored after the step that replaced
      // it: never shown as that step's results (G-397).
      if (isReplacedStepRow(row) || isStartedRow(row)) return null;
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

    default:
      return row;
  }
}

/**
 * Which result an entry is, for folding (G-402): its call id and tool name,
 * each length-prefixed, and its arguments as canonical JSON. Two calls that
 * share an id but not their arguments are two results.
 */
function resultKey(entry: ToolResultEntry): string {
  return `${entry.tool_call_id.length}:${entry.tool_call_id}${entry.tool_name.length}:${entry.tool_name}${canonicalJson(entry.args)}`;
}

function fold(earlier: ToolResultRow, later: ToolResultRow): ToolResultRow {
  // A later result replaces only the same result, occurrence by occurrence:
  // the k-th with a key in the later row replaces the k-th with that key so
  // far, in its position; any other is added after.
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

function decision(value: JsonObject): ApprovalDecisionEntry {
  return { approval_id: text(value.approval_id), approved: value.approved === true, reason: nullableText(value.reason) };
}

/**
 * The result a gated call gets when its step repeats a tool call id (G-392):
 * not run, whatever was decided.
 *
 * @internal
 */
export const REPEATED_CALL_ID_RESULT = 'Tool call was not run because another call in this step has the same id';

/**
 * Whether one assistant step repeats a tool call id, among its calls (an
 * empty id is an id like any other) or among its approval requests, or
 * whose approval requests repeat an APPROVAL id. A decision reaches a call
 * through the approval id and the call id its request names, so such a step
 * cannot say which call a decision belongs to (G-392, G-397).
 *
 * @internal
 */
export function repeatsCallIds(calls: readonly JsonObject[], requests: readonly JsonObject[]): boolean {
  const ids = calls.map((call) => text(call.id));
  const asked = requests.map((request) => text(request.tool_call_id));
  const approvals = requests.map((request) => text(request.approval_id));

  return new Set(ids).size !== ids.length || new Set(asked).size !== asked.length || new Set(approvals).size !== approvals.length;
}

/**
 * The result each gated call of a parked step gets when a later tool-call
 * step follows it before the parked step was answered (G-397): not run,
 * whatever was decided for it.
 *
 * @internal
 */
export const REPLACED_STEP_RESULT = 'Tool call was not run because a later step replaced it';

/**
 * The key of the mark on the row that answers a replaced step's calls:
 * `{position}`, the thread position of the step it answers. The row is
 * stored after the later step, so it is kept out of the replay view and out
 * of the later step's results: its entries name the replaced step's calls,
 * whose ids the later step may reuse.
 *
 * @internal
 */
export const REPLACED_STEP_KEY = 'harness_replaced_step';

/**
 * The key of the mark on every result row the harness writes for a step's
 * calls: `{position}`, the 1-based thread position of the step it answers, an
 * integer (G-407). Stored last, after `tool_approval_responses`:
 * `{"type":"tool_result","tool_results":[...],"tool_approval_responses":[...],
 * "harness_answers_step":{"position":2}}`. Such a row answers that step
 * wherever it lands: a tool-call step another writer records while a call
 * runs never takes the call's result. Honoured only on rows the harness
 * recorded, and only for a result whose id and tool name a call of an
 * earlier tool-call step at that position has; otherwise the result is its
 * segment's. Kept out of the replay view, which rebuilds rows from their
 * known keys.
 *
 * @internal
 */
export const ANSWERS_STEP_KEY = 'harness_answers_step';

/** `row` marked as answering the step at `position` (unchanged when it is unknown). @internal */
export function answeringStep<T extends JsonObject | ToolResultRow>(row: T, position: number | undefined): T {
  return position === undefined ? row : { ...row, [ANSWERS_STEP_KEY]: { position } } as T;
}

/** The step position a harness result row names, if it names one. */
function answeredStep(row: JsonObject): number | undefined {
  const mark = row[ANSWERS_STEP_KEY] ?? (isReplacedStepRow(row) ? row[REPLACED_STEP_KEY] : undefined);
  return isObject(mark) && typeof mark.position === 'number' ? mark.position : undefined;
}

/**
 * G-410: the result a started call with no result reads: it may have run, and
 * it is never run again.
 *
 * @internal
 */
export const INTERRUPTED_RESULT = 'Tool call was interrupted, it may have run';

/** G-410: an approved call that was no longer open when the resolver came to start it. @internal */
export const NOT_OPEN_RESULT = 'Not run: this call is not the one that was approved, or it is no longer open.';

/**
 * G-410: the key of the started row the harness records BEFORE it runs one
 * approved call: `{tool_call_id, tool_name, approval_id}`, followed by the
 * row's `harness_answers_step {position}`. The shared bytes (harness-thread-rows
 * thr-0022): {"type":"tool_result","tool_results":[],"tool_approval_responses":[],
 * "harness_started":{"tool_call_id":"call_paris","tool_name":"weather","approval_id":"apr_1"},
 * "harness_answers_step":{"position":2}}. Kept out of the replay view.
 *
 * @internal
 */
export const STARTED_KEY = 'harness_started';

/** The started row for `call` of the step at `position`, run under `approvalId`. @internal */
export function startedRow(call: { id: string; name: string }, approvalId: string, position: number): JsonObject {
  return {
    ...toolResultRow([]),
    [STARTED_KEY]: { tool_call_id: call.id, tool_name: call.name, approval_id: approvalId },
    [ANSWERS_STEP_KEY]: { position },
  };
}

/** Whether a stored row carries a started mark. @internal */
export function isStartedRow(row: JsonObject): boolean {
  return row.type === 'tool_result' && isObject(row[STARTED_KEY]);
}

/**
 * The positions, in the step at `index`, of the calls a started row names:
 * only a row the harness recorded, naming this step by position, and a call
 * of it with the started row's id and tool name (the first such call not
 * already named, occurrence by occurrence). Any other started mark is read as
 * absent.
 *
 * @internal
 */
export function startedCalls(messages: readonly StoredRow[], index: number): Set<number> {
  const position = messages[index]?.position;
  const calls = list(messages[index]?.message.tool_calls);
  const started = new Set<number>();
  for (const { message, harness } of messages.slice(index + 1)) {
    if (harness !== true || !isStartedRow(message)) continue;
    const answers = message[ANSWERS_STEP_KEY];
    if (!isObject(answers) || answers.position !== position) continue;
    const mark = message[STARTED_KEY] as JsonObject;
    const at = calls.findIndex((call, place) => !started.has(place) && text(call.id) === text(mark.tool_call_id) && text(call.name) === text(mark.tool_name));
    if (at !== -1) started.add(at);
  }
  return started;
}

/**
 * For each call of the step at `index` that a started row names, the decision
 * its approval had when the call started (G-411): the last one in the replay
 * view of the rows between the step and the started row. A decision written
 * after a call started never becomes the decision recorded beside its outcome.
 * Read by the same rule as `startedCalls()`.
 *
 * @internal
 */
export function startedDecisions(messages: readonly StoredRow[], index: number): Map<string, ApprovalDecisionEntry> {
  const position = messages[index]?.position;
  const calls = list(messages[index]?.message.tool_calls);
  const named = new Set<number>();
  const standing = new Map<string, ApprovalDecisionEntry>();
  for (let at = index + 1; at < messages.length; at += 1) {
    const { message, harness } = messages[at]!;
    if (harness !== true || !isStartedRow(message)) continue;
    const answers = message[ANSWERS_STEP_KEY];
    if (!isObject(answers) || answers.position !== position) continue;
    const mark = message[STARTED_KEY] as JsonObject;
    const place = calls.findIndex((call, where) => !named.has(where) && text(call.id) === text(mark.tool_call_id) && text(call.name) === text(mark.tool_name));
    if (place === -1) continue;
    named.add(place);
    const approvalId = text(mark.approval_id);
    let found: ApprovalDecisionEntry | undefined;
    for (const row of threadView(messages.slice(index + 1, at).map((entry) => entry.message))) {
      if (row.type !== 'tool_result') continue;
      for (const entry of list(row.tool_approval_responses)) if (text(entry.approval_id) === approvalId) found = decision(entry);
    }
    if (found !== undefined && !standing.has(approvalId)) standing.set(approvalId, found);
  }
  return standing;
}

/** Whether a stored row is the answer to a replaced step (G-397). @internal */
export function isReplacedStepRow(row: JsonObject): boolean {
  const mark = row[REPLACED_STEP_KEY];
  return row.type === 'tool_result' && isObject(mark) && typeof mark.position === 'number';
}

/**
 * THE rule for which rows are tool-call steps, for every reader (G-408): an
 * assistant row with at least one tool call that is an object. A row whose
 * tool_calls hold no objects (strings, numbers, null, arrays) is an assistant
 * row without tool calls.
 *
 * @internal
 */
export function isToolCallStep(row: JsonObject | undefined): boolean {
  return row !== undefined && row.type === 'assistant' && list(row.tool_calls).length > 0;
}

/** A stored thread row and the run that recorded it. @internal */
export type StoredRow = { position: number; message: JsonObject; runId: string | null; harness?: true };

/**
 * The index of the row that SUPERSEDES the tool-call step at `index` of the
 * stored rows (G-397), or -1 while it is not superseded:
 *
 * - any later assistant row WITH tool calls, always;
 * - a later assistant row without tool calls, or a user row, recorded
 *   OUTSIDE the runs that own the step: the run the harness recorded it in
 *   and the runs the harness recorded its results in. Rows the runtime
 *   writes inside those runs (a resume's answer, a structured turn's
 *   document) never supersede; a row recorded through the public
 *   Thread.record does, whatever run id it was given.
 *
 * Rows of any other kind (a system row, a tool result row) never supersede.
 * This only decides how a step's open calls are labelled and whether a late
 * decision is accepted: what can run is decided by the one step selector.
 *
 * @internal
 */
export function supersededAt(rows: readonly StoredRow[], index: number): number {
  // Only rows the harness itself recorded name a run that owns the step: a
  // run id given through the public Thread.record counts as outside.
  const inside = (row: StoredRow | undefined): string | null => (row?.harness === true && typeof row.runId === 'string' ? row.runId : null);
  const owners = new Set<string>();
  const own = inside(rows[index]);
  if (own !== null) owners.add(own);
  for (let at = index + 1; at < rows.length; at += 1) {
    const row = rows[at]!;
    const { message } = row;
    const run = inside(row);
    if (isToolCallStep(message)) return at;
    if (message.type === 'tool_result' && run !== null && list(message.tool_results).length > 0) owners.add(run);
    if ((message.type === 'user' || message.type === 'assistant') && (run === null || !owners.has(run))) return at;
  }
  return -1;
}

/**
 * Which call every stored result belongs to, as the calls each tool-call step
 * has an outcome for (by the step's index in `rows`, then the call's
 * position in it). ABSOLUTE SEGMENT OWNERSHIP (G-397): a result answers only
 * a call of the step whose segment holds its row (after that step, before
 * the next tool-call step); the rows a replaced step is answered with answer
 * the step their mark names. Inside that step, by exact id, name and parsed
 * arguments (usableArguments, keys in a fixed order), then id and name, then
 * id, each the first call still without a result. A result an earlier row of
 * that segment already holds (a copy) answers nothing new, and neither does a
 * further result for a call that has one. A result never answers a call of
 * another step.
 *
 * @internal
 */
export function resultOwners(messages: readonly StoredRow[]): Map<number, Set<number>> {
  const owned = new Map<number, Set<number>>();
  const seen = new Map<number, Set<string>>();
  const byPosition = new Map<number, number>();
  let segment = -1;
  const args = (value: JsonValue | undefined): string => canonicalJson(usableArguments(value) ?? {});
  for (const [index, { position, message: row, harness }] of messages.entries()) {
    if (isToolCallStep(row)) {
      segment = index;
      owned.set(index, new Set());
      seen.set(index, new Set());
      byPosition.set(position, index);
      continue;
    }
    if (segment === -1) continue;
    const replaced = isReplacedStepRow(row);
    // A harness row answers the step it names (G-407); any other row, the step
    // whose segment holds it.
    // Only a mark the harness itself recorded is honoured.
    // It must name an EARLIER tool-call step (one already read) holding a call
    // with the result's id and tool name; otherwise the result is its
    // segment's, which fails closed: that call is answered.
    const named = harness === true ? answeredStep(row) : undefined;
    const step = named === undefined ? undefined : byPosition.get(named);
    // Each row read on its own: folding rows would merge results sharing an id.
    const entries = replaced ? list(row.tool_results)
      : threadView([row]).filter((view) => view.type === 'tool_result').flatMap((view) => list(view.tool_results));
    const added = new Map<number, string[]>();
    for (const entry of entries) {
      const id = text(entry.tool_call_id), name = text(entry.tool_name), recorded = args(entry.args);
      const target = step !== undefined && list(messages[step]!.message.tool_calls).some((call) => text(call.id) === id && text(call.name) === name) ? step : segment;
      const calls = list(messages[target]!.message.tool_calls);
      const mine = owned.get(target)!;
      const prints = seen.get(target)!;
      const print = canonicalJson([id, name, recorded, text(entry.result)]);
      if (prints.has(print)) continue;
      added.set(target, [...(added.get(target) ?? []), print]);
      const exact = (call: JsonObject): boolean => text(call.id) === id && text(call.name) === name && args(call.arguments) === recorded;
      if (calls.some((call, at) => exact(call) && mine.has(at)) && !calls.some((call, at) => exact(call) && !mine.has(at))) continue;
      let at = calls.findIndex((call, position) => exact(call) && !mine.has(position));
      if (at === -1) at = calls.findIndex((call, position) => text(call.id) === id && text(call.name) === name && !mine.has(position));
      if (at === -1) at = calls.findIndex((call, position) => text(call.id) === id && !mine.has(position));
      if (at !== -1) mine.add(at);
    }
    for (const [target, prints] of added) for (const print of prints) seen.get(target)!.add(print);
  }
  return owned;
}

/**
 * Every tool-call step that was SUPERSEDED (G-397) and still has gated calls
 * with no outcome anywhere (resultOwners), by the step's thread position,
 * with those calls. A step is superseded as supersededAt() says. A replaced
 * denial is never given to a call that already has an outcome, so it is
 * given once.
 *
 * `gated` says which calls needed approval; an ungated call counts only when
 * a request names its id and no gated call shares that id.
 *
 * @internal
 */
export function replacedStepCalls(
  messages: readonly StoredRow[],
  gated: (name: string) => boolean,
): { position: number; calls: JsonObject[]; started: boolean[] }[] {
  const rows = messages.map(({ message }) => message);
  const owners = resultOwners(messages);
  const replaced: { position: number; calls: JsonObject[]; started: boolean[] }[] = [];
  for (const [index, step] of rows.entries()) {
    if (!isToolCallStep(step)) continue;
    if (supersededAt(messages, index) === -1) continue;
    const done = owners.get(index)!;
    const calls = list(step.tool_calls);
    const requested = new Set(list(step.tool_approval_requests).map((request) => text(request.tool_call_id)));
    // A started call with no result is answered as interrupted (G-410), never
    // denied as replaced.
    const started = startedCalls(messages, index);
    const open = calls.map((call, at) => ({ call, at })).filter(({ call, at }) => !done.has(at) && (started.has(at) || gated(text(call.name))
      || (requested.has(text(call.id)) && !calls.some((other) => other.id === call.id && gated(text(other.name))))));
    if (open.length > 0) replaced.push({ position: messages[index]!.position, calls: open.map(({ call }) => call), started: open.map(({ at }) => started.has(at)) });
  }

  return replaced;
}

/**
 * Whether a decision for `approvalId` can no longer be acted on because the
 * step that asked for it was superseded (G-397): a decision recorded after
 * the superseding row never runs anything, so it is refused.
 *
 * @internal
 */
export function approvalSuperseded(messages: readonly StoredRow[], approvalId: string): boolean {
  return messages.some(({ message }, index) => message.type === 'assistant'
    && list(message.tool_approval_requests).some((request) => request.approval_id === approvalId)
    && supersededAt(messages, index) !== -1);
}

/** JSON with object keys sorted, so equal values compare equal. */
function canonicalJson(value: JsonValue): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  if (isObject(value)) return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key]!)}`).join(',')}}`;
  return JSON.stringify(value);
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
