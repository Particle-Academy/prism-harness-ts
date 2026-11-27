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
 * An approval has ONE answer: the first one recorded. A second decision for
 * the same id (two reviewers, a retry racing the original, a row written by
 * hand) is never the one that counts, here or in the plan's reading of the
 * thread, so what runs and what the plan records cannot disagree.
 */
export function turnDecisions(rows: readonly JsonObject[]): Map<string, ApprovalDecisionEntry> {
  const decisions = new Map<string, ApprovalDecisionEntry>();
  // The paused step (G-397): no decision applies to any other.
  const step = pausedStepIndex(rows);
  if (step === null || list(rows[step]!.tool_calls).length === 0) return decisions;

  for (const row of rows.slice(step + 1)) {
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
  // The paused step (G-397), as stored, and the same row in the view (which
  // folds a 0.3.0 request row into the assistant row before it).
  const step = pausedStepIndex(rows);
  if (step === null) return null;
  const ordinal = rows.slice(0, step + 1).filter((candidate) => candidate.type === 'assistant').length;
  const row = threadView(rows).filter((candidate) => candidate.type === 'assistant')[ordinal - 1];
  if (row === undefined) return null;
  const requests = list(row.tool_approval_requests);
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
  const pending = list(rows[step]!.tool_approval_requests)
    .map(request => request.approval_id)
    .filter((id): id is string => typeof id === 'string' && id !== '' && !answered.has(id));

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
  const requests = list(row.tool_approval_requests);
  const ids = (entries: JsonObject[]): string[] => [...new Set(entries
    .map((request) => request.approval_id)
    .filter((id): id is string => typeof id === 'string' && id !== ''))];

  // A step that repeats a call id: a result by id cannot say whose it is
  // (G-392), so every request is unresolved until the step is settled.
  if (pairStepApprovals(list(row.tool_calls), requests).repeated) return repeatedStepSettled(rows, step) ? [] : ids(requests);

  const finished = new Set<string>();
  for (const later of rows.slice(step + 1)) {
    if (later.type !== 'tool_result' || isReplacedStepRow(later)) continue;
    for (const result of list(later.tool_results)) {
      if (typeof result.tool_call_id === 'string') finished.add(result.tool_call_id);
    }
  }

  return ids(requests.filter((request) => typeof request.tool_call_id !== 'string' || !finished.has(request.tool_call_id)));
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
  if (message.type === 'assistant' && list(message.tool_calls).length > 0) return true;

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
  const calls = list(rows[step]!.tool_calls);
  if (calls.length === 0) return null;
  const answered = stepResultsByPosition([...rows, ...alsoResults], step);
  const open = calls.map((_call, position) => position).filter((position) => !answered[position]);
  if (open.length === 0) return null;
  const paired = pairStepApprovals(calls, list(rows[step]!.tool_approval_requests), (_call, position) => !answered[position]).approvalIds;
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
    if (index === paused || step.type !== 'assistant' || list(step.tool_calls).length === 0) continue;
    if (answeredAt.has(rows[index]!.position)) continue;
    const owners = new Set<string>();
    if (rows[index]!.runId !== null) owners.add(rows[index]!.runId!);
    let end = -1;
    for (let at = index + 1; at < rows.length; at += 1) {
      const { message, runId } = rows[at]!;
      if (message.type === 'assistant' && list(message.tool_calls).length > 0) { end = at; break; }
      if (message.type === 'tool_result' && runId !== null && list(message.tool_results).length > 0) owners.add(runId);
      if ((message.type === 'user' || message.type === 'assistant') && (runId === null || !owners.has(runId))) { end = at; break; }
    }
    if (end === -1) continue;
    const own = messages.slice(index, end);
    const answered = stepResultsByPosition(own, 0);
    const calls = list(step.tool_calls);
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
  const requests = new Map<string, number>();
  for (const [index, row] of rows.entries()) {
    if (row.type !== 'assistant') continue;
    for (const request of list(row.tool_approval_requests)) {
      if (typeof request.approval_id === 'string' && typeof request.tool_call_id === 'string') requests.set(request.approval_id, index);
    }
  }
  const ended = new Set<string>();
  for (const [approval, index] of requests) {
    for (const row of rows.slice(index + 1)) {
      if (row.type === 'assistant') break;
      if (row.type !== 'tool_result') continue;
      const mark = row[key];
      if (isObject(mark) && Array.isArray(mark.approval_ids) && mark.approval_ids.includes(approval)) {
        ended.add(approval);
        break;
      }
    }
  }

  return ended;
}

/**
 * Every approval id the run paused at the transcript tail asked for, answered
 * or not; empty when the tail is a user turn or an assistant step that asked
 * nothing. A turn with no new message continues exactly these calls (G-366).
 */
export function pausedApprovalIds(rows: readonly JsonObject[]): string[] {
  const step = pausedStepIndex(rows);
  if (step === null) return [];

  return [...new Set(list(rows[step]!.tool_approval_requests)
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
 * Which calls of the stored assistant step at `index` have a result, paired
 * BY POSITION (G-387 round 13), never by tool call id: providers repeat ids
 * within one step, and an ungated call's result (or the immediate result of
 * a call with unusable arguments) must not make a gated call sharing its id
 * look finished. Read from EVERY result row after the step, up to the next
 * assistant row (G-399), as stored (the replay view folds copies away).
 * Each result pairs with the first call without one that has its id, its
 * tool name and its arguments; then its id and name; then its id.
 *
 * @internal
 */
export function stepResultsByPosition(rows: readonly JsonObject[], index: number): boolean[] {
  const calls = list(rows[index]?.tool_calls);
  const has = calls.map(() => false);
  const args = calls.map((call) => JSON.stringify(usableArguments(call.arguments) ?? {}));
  // A result an EARLIER row of the step already holds (a copy: the same id,
  // tool, arguments and result) answers nothing new (TS I-1, as 0.11.x reads
  // it): the resolution row carries the step's earlier results, and a copy
  // must never claim another open call that shares its id.
  const prints = new Set<string>();
  // EVERY tool_result row after the step, up to the next assistant row
  // (G-399): a row of another kind between them hides nothing.
  for (const row of rows.slice(index + 1)) {
    if (row.type === 'assistant') break;
    // A 0.11.x replaced-step answer is an EARLIER step's (it names that
    // step's position): never this step's results.
    if (row.type !== 'tool_result' || isReplacedStepRow(row)) continue;
    const entries = Array.isArray(row.tool_results) ? list(row.tool_results)
      : Object.hasOwn(row, 'tool_call_id') ? [{ tool_call_id: row.tool_call_id ?? null, tool_name: row.name ?? null }] : [];
    const added: string[] = [];
    for (const entry of entries) {
      const id = text(entry.tool_call_id);
      const print = sortedJson([id, text(entry.tool_name), isObject(entry.args) ? entry.args : {}, text(entry.result)]);
      if (prints.has(print)) continue;
      added.push(print);
      const open = (position: number): boolean => !has[position] && text(calls[position]!.id) === id;
      let at = calls.findIndex((call, position) => open(position) && call.name === entry.tool_name && args[position] === JSON.stringify(isObject(entry.args) ? entry.args : {}));
      if (at === -1) at = calls.findIndex((call, position) => open(position) && call.name === entry.tool_name);
      if (at === -1) at = calls.findIndex((_call, position) => open(position));
      if (at !== -1) has[at] = true;
    }
    for (const print of added) prints.add(print);
  }

  return has;
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

  return step !== null && pairStepApprovals(list(rows[step]!.tool_calls), list(rows[step]!.tool_approval_requests)).repeated;
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
  const calls = list(rows[index]?.tool_calls).map((call) => text(call.id));
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
      for (const request of list(row.tool_approval_requests)) {
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

    default:
      return row;
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
