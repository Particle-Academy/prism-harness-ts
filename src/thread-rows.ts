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
    // A step that repeats a call id approves nothing (G-392).
    if (pairStepApprovals(list(row.tool_calls), requests).repeated) return {names: [], legacy: true};
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
 * The approval ids the run paused at the transcript tail asked for whose
 * call has no result yet, pending OR decided (G-387 round 10): a resume would
 * act on each of them. Read newest first, as pendingApprovalIds() is; a user
 * turn ends the scope.
 */
export function unresolvedApprovalIds(rows: readonly JsonObject[]): string[] {
  const finished = new Set<string>();
  for (let index = rows.length - 1; index >= 0; index -= 1) {
    const row = rows[index]!;
    if (row.type === 'tool_result') {
      for (const result of list(row.tool_results)) {
        if (typeof result.tool_call_id === 'string') finished.add(result.tool_call_id);
      }
      continue;
    }
    if (row.type === 'user') return [];
    if (row.type !== 'assistant') continue;

    // A step that repeats a call id: a result by id cannot say whose it is
    // (G-392), so every request is unresolved until the step is settled.
    if (pairStepApprovals(list(row.tool_calls), list(row.tool_approval_requests)).repeated) {
      if (repeatedStepSettled(rows, index)) return [];

      return [...new Set(list(row.tool_approval_requests)
        .map((request) => request.approval_id)
        .filter((id): id is string => typeof id === 'string' && id !== ''))];
    }

    return [...new Set(list(row.tool_approval_requests)
      .filter((request) => typeof request.tool_call_id !== 'string' || !finished.has(request.tool_call_id))
      .map((request) => request.approval_id)
      .filter((id): id is string => typeof id === 'string' && id !== ''))];
  }

  return [];
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
      const mark = row[ENDED_BY_START_KEY];
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
 * id is an id like any other) or among its requests. Such a step cannot be
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

  return { approvalIds, repeated: new Set(ids).size !== ids.length || new Set(asked).size !== asked.length };
}

/**
 * Which calls of the stored assistant step at `index` have a result, paired
 * BY POSITION (G-387 round 13), never by tool call id: providers repeat ids
 * within one step, and an ungated call's result (or the immediate result of
 * a call with unusable arguments) must not make a gated call sharing its id
 * look finished. Read from the result rows after the step, before the next
 * assistant or user row, as stored (the replay view folds results by id).
 * Each result pairs with the first call without one that has its id, its
 * tool name and its arguments; then its id and name; then its id.
 *
 * @internal
 */
export function stepResultsByPosition(rows: readonly JsonObject[], index: number): boolean[] {
  const calls = list(rows[index]?.tool_calls);
  const has = calls.map(() => false);
  const args = calls.map((call) => JSON.stringify(usableArguments(call.arguments) ?? {}));
  for (const row of rows.slice(index + 1)) {
    if (row.type === 'assistant' || row.type === 'user') break;
    if (row.type !== 'tool_result') continue;
    const entries = Array.isArray(row.tool_results) ? list(row.tool_results)
      : Object.hasOwn(row, 'tool_call_id') ? [{ tool_call_id: row.tool_call_id ?? null, tool_name: row.name ?? null }] : [];
    for (const entry of entries) {
      const id = text(entry.tool_call_id);
      const open = (position: number): boolean => !has[position] && text(calls[position]!.id) === id;
      let at = calls.findIndex((call, position) => open(position) && call.name === entry.tool_name && args[position] === JSON.stringify(isObject(entry.args) ? entry.args : {}));
      if (at === -1) at = calls.findIndex((call, position) => open(position) && call.name === entry.tool_name);
      if (at === -1) at = calls.findIndex((_call, position) => open(position));
      if (at !== -1) has[at] = true;
    }
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
  for (let index = rows.length - 1; index >= 0; index -= 1) {
    const row = rows[index]!;
    if (row.type === 'user') return false;
    if (row.type !== 'assistant') continue;

    return pairStepApprovals(list(row.tool_calls), list(row.tool_approval_requests)).repeated;
  }

  return false;
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
    if (row.type !== 'tool_result') continue;
    const entries = list(row.tool_results);
    if (isObject(row[ENDED_BY_START_KEY]) || entries.some((entry) => entry[NOT_RUN_KEY] === REPEATED_CALL_ID)) return true;
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
