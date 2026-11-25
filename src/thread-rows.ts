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
 * by its tool call id and a decision by its approval id, and a later one
 * replaces an earlier one in the earlier one's position.
 *
 * This is the REPLAY view, pinned by the harness-thread-rows corpus. What a
 * decision means for running a call is read from the stored rows instead, by
 * `turnDecisions()`, where the first answer to an approval is the answer.
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

/**
 * The decisions recorded for the turn being resumed (the last assistant row
 * with tool calls), by approval id, read from the stored rows.
 *
 * An approval has ONE answer: the first one recorded. A second decision for
 * the same id (two reviewers, a retry racing the original, a row written by
 * hand) is never the one that counts, here or in the plan's reading of the
 * thread, so what runs and what the plan records cannot disagree.
 */
export function turnDecisions(rows: readonly JsonObject[]): Map<string, ApprovalDecisionEntry> {
  const decisions = new Map<string, ApprovalDecisionEntry>();
  let start = rows.length;
  while (start > 0 && !(rows[start - 1]!.type === 'assistant' && list(rows[start - 1]!.tool_calls).length > 0)) start -= 1;
  if (start === 0) return decisions;

  for (const row of rows.slice(start)) {
    const answers = row.type === 'tool_result' ? list(row.tool_approval_responses)
      : row.type === 'tool_approval_response' ? [row] : [];
    for (const answer of answers) {
      const entry = decision(answer);
      if (!decisions.has(entry.approval_id)) decisions.set(entry.approval_id, entry);
    }
  }

  return decisions;
}

/** Stored approval authority, kept separate from the provider replay view. */
export function approvalToolScope(rows: readonly JsonObject[]): { names: string[]; legacy: boolean } | null {
  for (const row of threadView(rows).reverse()) {
    if (row.type === 'tool_result') continue;
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
    const decisions = turnDecisions(rows);
    const approvedCalls = new Set(requests.filter(request => typeof request.approval_id === 'string' && decisions.get(request.approval_id)?.approved === true)
      .map(request => request.tool_call_id));
    const names = list(row.tool_calls).filter(call => approvedCalls.has(call.id) && typeof call.name === 'string' && call.name !== '' && call.name !== '*')
      .map(call => call.name as string);
    return {names: [...new Set(names)], legacy: true};
  }
  return null;
}

/**
 * The approval ids the run paused at the transcript tail is still waiting on:
 * issued by the last assistant step and not yet answered (G-354).
 *
 * Read from the stored rows, newest first. A user turn, or an assistant step
 * that asked nothing, ends the scope: nothing before it is pending. An id
 * answered once is no longer pending.
 */
export function pendingApprovalIds(rows: readonly JsonObject[]): string[] {
  const answered = new Set<string>();
  for (let index = rows.length - 1; index >= 0; index -= 1) {
    const row = rows[index]!;
    if (row.type === 'tool_result') {
      for (const response of list(row.tool_approval_responses)) {
        if (typeof response.approval_id === 'string') answered.add(response.approval_id);
      }
      continue;
    }
    if (row.type === 'user') return [];
    if (row.type !== 'assistant') continue;

    const pending = list(row.tool_approval_requests)
      .map(request => request.approval_id)
      .filter((id): id is string => typeof id === 'string' && id !== '' && !answered.has(id));

    return [...new Set(pending)];
  }

  return [];
}

/**
 * Every approval id the run paused at the transcript tail asked for, answered
 * or not; empty when the tail is a user turn or an assistant step that asked
 * nothing. A turn with no new message continues exactly these calls (G-366).
 */
export function pausedApprovalIds(rows: readonly JsonObject[]): string[] {
  for (let index = rows.length - 1; index >= 0; index -= 1) {
    const row = rows[index]!;
    if (row.type === 'user') return [];
    if (row.type !== 'assistant') continue;

    return [...new Set(list(row.tool_approval_requests)
      .map(request => request.approval_id)
      .filter((id): id is string => typeof id === 'string' && id !== ''))];
  }

  return [];
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
export function pausedRunState(rows: readonly JsonObject[]): { agentSpend: PausedAgentSpend | null; binding: PausedBinding | null } | null {
  if (approvalToolScope(rows) === null) return null;
  const paused = [...rows].reverse().find((row) => row.type === 'assistant');
  if (paused === undefined) return null;

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

  return { agentSpend, binding: storedBinding(paused.harness_binding) };
}

function storedBinding(value: JsonValue | undefined): PausedBinding | null {
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
 * Read from the one turn being resumed, the last assistant row with tool
 * calls, never the whole thread: providers that number calls per response
 * reuse ids, and an older turn's call must not decide a newer one.
 */
export function unusableStoredCalls(rows: readonly JsonObject[]): Set<string> {
  const unusable = new Set<string>();
  const turn = [...rows].reverse().find((row) => row.type === 'assistant' && list(row.tool_calls).length > 0);

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
