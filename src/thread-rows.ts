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

  return view;
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

    default:
      return row;
  }
}

function fold(earlier: ToolResultRow, later: ToolResultRow): ToolResultRow {
  const results = new Map<string, ToolResultEntry>();
  const decisions = new Map<string, ApprovalDecisionEntry>();

  for (const entry of [...earlier.tool_results, ...later.tool_results]) results.set(entry.tool_call_id, entry);
  for (const entry of [...earlier.tool_approval_responses, ...later.tool_approval_responses]) decisions.set(entry.approval_id, entry);

  return toolResultRow([...results.values()], [...decisions.values()]);
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

/** Whether a stored row is the answer to a replaced step (G-397). @internal */
export function isReplacedStepRow(row: JsonObject): boolean {
  const mark = row[REPLACED_STEP_KEY];
  return row.type === 'tool_result' && isObject(mark) && typeof mark.position === 'number';
}

/** A stored thread row and the run that recorded it. @internal */
export type StoredRow = { position: number; message: JsonObject; runId: string | null };

/**
 * The index of the row that SUPERSEDES the tool-call step at `index` of the
 * stored rows (G-397), or -1 while it is not superseded:
 *
 * - any later assistant row WITH tool calls, always;
 * - a later assistant row without tool calls, or a user row, recorded
 *   OUTSIDE the runs that own the step: the run that recorded it and the
 *   runs that recorded its results. Rows the runtime writes inside those
 *   runs (a resume's answer, a structured turn's document) never supersede;
 *   a row recorded with no run id, or under another run, does.
 *
 * Rows of any other kind (a system row, a tool result row) never supersede.
 * This only decides how a step's open calls are labelled and whether a late
 * decision is accepted: what can run is decided by the one step selector.
 *
 * @internal
 */
export function supersededAt(rows: readonly StoredRow[], index: number): number {
  const owners = new Set<string>();
  const own = rows[index]?.runId;
  if (typeof own === 'string') owners.add(own);
  for (let at = index + 1; at < rows.length; at += 1) {
    const { message, runId } = rows[at]!;
    if (message.type === 'assistant' && list(message.tool_calls).length > 0) return at;
    if (message.type === 'tool_result' && typeof runId === 'string' && list(message.tool_results).length > 0) owners.add(runId);
    if ((message.type === 'user' || message.type === 'assistant') && (typeof runId !== 'string' || !owners.has(runId))) return at;
  }
  return -1;
}

/**
 * Which call every stored result belongs to, as the calls each tool-call step
 * has an outcome for (by the step's index in `rows`, then the call's position
 * in it). One owner per result, so every call has at most one outcome:
 *
 * - a result belongs to the call with its id AND the arguments it records
 *   (the harness copies a call's parsed arguments into its result; both are
 *   compared parsed), in the step whose segment holds the row (after it,
 *   before the next tool-call step) or, failing that, an earlier step's call
 *   that has no result yet, the latest first. A copy of a result of the
 *   segment's step belongs to the same call again; it never answers a
 *   second one;
 * - a result whose arguments match no call with its id belongs to the last
 *   call with its id, of the step whose segment holds it, still without one.
 *
 * A result never answers a call of a step recorded after it. The rows a
 * replaced step is answered with count too, so it is answered once.
 *
 * @internal
 */
export function resultOwners(rows: readonly JsonObject[]): Map<number, Set<number>> {
  const owned = new Map<number, Set<number>>();
  const steps: number[] = [];
  // Both sides in the form the harness uses a call's arguments, parsed
  // (usableArguments), whatever form a writer stored them in: a JSON string
  // (as the PHP reference stores them) and the same object compare equal.
  const args = (value: JsonValue | undefined): string => canonicalJson(usableArguments(value) ?? {});
  for (const [index, row] of rows.entries()) {
    if (row.type === 'assistant' && list(row.tool_calls).length > 0) {
      steps.push(index);
      owned.set(index, new Set());
      continue;
    }
    if (steps.length === 0) continue;
    // Each row read on its own: folding rows would merge results sharing an id.
    const entries = isReplacedStepRow(row) ? list(row.tool_results)
      : threadView([row]).filter((view) => view.type === 'tool_result').flatMap((view) => list(view.tool_results));
    for (const entry of entries) {
      const id = text(entry.tool_call_id);
      let owner: [number, number] | undefined;
      let copy = false;
      for (const step of [...steps].reverse()) {
        const calls = list(rows[step]!.tool_calls);
        for (const [at, call] of calls.entries()) {
          if (text(call.id) !== id || args(call.arguments) !== args(entry.args)) continue;
          if (!owned.get(step)!.has(at)) { owner = [step, at]; break; }
          // A copy of a result of the segment's own step answers nothing new.
          // An exact match on an EARLIER call that already has its result is
          // not a copy of it: the result stays in this segment (below).
          if (step === steps.at(-1)) copy = true;
        }
        if (owner !== undefined || copy) break;
      }
      if (owner === undefined && !copy) {
        const segment = steps.at(-1)!;
        const calls = list(rows[segment]!.tool_calls);
        const at = calls.map((call, position) => ({ call, position }))
          .filter(({ call, position }) => text(call.id) === id && !owned.get(segment)!.has(position)).at(-1)?.position;
        if (at !== undefined) owner = [segment, at];
      }
      if (owner !== undefined) owned.get(owner[0])!.add(owner[1]);
    }
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
): { position: number; calls: JsonObject[] }[] {
  const rows = messages.map(({ message }) => message);
  const owners = resultOwners(rows);
  const replaced: { position: number; calls: JsonObject[] }[] = [];
  for (const [index, step] of rows.entries()) {
    if (step.type !== 'assistant' || list(step.tool_calls).length === 0) continue;
    if (supersededAt(messages, index) === -1) continue;
    const done = owners.get(index)!;
    const calls = list(step.tool_calls);
    const requested = new Set(list(step.tool_approval_requests).map((request) => text(request.tool_call_id)));
    const open = calls.filter((call, at) => !done.has(at) && (gated(text(call.name))
      || (requested.has(text(call.id)) && !calls.some((other) => other.id === call.id && gated(text(other.name))))));
    if (open.length > 0) replaced.push({ position: messages[index]!.position, calls: open });
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
