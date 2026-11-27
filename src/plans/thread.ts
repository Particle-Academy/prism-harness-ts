import type { JsonObject, JsonValue } from '../json.js';
import { endedByStart, pairStepApprovals, pausedStepIndex, pendingApprovalIds, repeatedIdDenials, stepResultsByPosition, supersededApprovals, isReplacedStepRow } from '../thread-rows.js';
import { argsDigest } from '../tool-policy.js';
import type { PendingCall } from './machine.js';

// What a plan reads from the thread (G-360). Read from the persisted rows
// only: a plan binds to these ids and to no other, and never to an id from a
// tool argument or model text.

function list(value: JsonValue | undefined): JsonObject[] {
  return Array.isArray(value) ? value.filter((entry): entry is JsonObject => typeof entry === 'object' && entry !== null && !Array.isArray(entry)) : [];
}

function digestOf(args: JsonValue | undefined): string {
  if (typeof args === 'string') {
    try {
      return argsDigest(JSON.parse(args) ?? {});
    } catch {
      return argsDigest({});
    }
  }

  return argsDigest(args ?? {});
}

/** Every pending approval at the transcript tail, with the tool it was asked for and the digest of its stored arguments. */
export function pendingApprovalCalls(rows: readonly JsonObject[]): PendingCall[] {
  const ids = pendingApprovalIds(rows);
  // The paused step (G-397), by the one selector: the pending ids are its own.
  const index = pausedStepIndex(rows);
  if (ids.length === 0 || index === null) return [];
  const row = rows[index]!;
  // Each approval's call, paired by position (G-392), with a call that has
  // no result of its own first (round 13): a call that already has one ran
  // ungated, so it is not the one asked about.
  const stepCalls = list(row.tool_calls);
  const answered = stepResultsByPosition(rows, index);
  const paired = pairStepApprovals(stepCalls, list(row.tool_approval_requests), (_call, position) => !answered[position]).approvalIds;
  const calls: PendingCall[] = [];
  for (const id of ids) {
    const call = stepCalls[paired.indexOf(id)];
    if (call !== undefined && typeof call.name === 'string') calls.push({ id, name: call.name, digest: digestOf(call.arguments) });
  }

  return calls;
}

/**
 * Every recorded decision, by approval id: true when approved. The first
 * decision for an id stands. A decision on a call a runner start's task row
 * ended as not run is left out (G-387 round 10): it was never acted on, so
 * the plan reads that approval as ended, never as approved work.
 */
export function approvalDecisions(rows: readonly JsonObject[]): Map<string, boolean> {
  const decided = new Map<string, boolean>();
  // Calls denied because their step repeats a call id (G-392) were never
  // acted on either: never approved work.
  // Superseded by a later step with the call open (DECISION B): never acted on.
  const ended = new Set([...endedByStart(rows), ...repeatedIdDenials(rows), ...supersededApprovals(rows)]);
  for (const row of rows) {
    if (row.type !== 'tool_result') continue;
    for (const response of list(row.tool_approval_responses)) {
      if (typeof response.approval_id === 'string' && !decided.has(response.approval_id) && !ended.has(response.approval_id)) {
        decided.set(response.approval_id, response.approved === true);
      }
    }
  }

  return decided;
}

/**
 * How many calls of the turn paused at the transcript tail a person approved,
 * for this tool and these arguments, that have no result yet. The evidence a
 * self-gated tool requires before it applies anything.
 */
export function approvedCallsAwaitingResult(rows: readonly JsonObject[], name: string, digest: string): number {
  // The paused step (G-397), by the one selector every decision reader uses.
  const step = pausedStepIndex(rows);
  if (step === null) return 0;
  const row = rows[step]!;
  // A step that repeats a call id runs none of its gated calls (G-392).
  if (pairStepApprovals(list(row.tool_calls), list(row.tool_approval_requests)).repeated) return 0;
  // The FIRST decision recorded for an id is the one answer that counts, as
  // for the resume and the plan.
  const decided = new Map<string, boolean>();
  const finished = new Set<string>();
  for (const later of rows.slice(step + 1)) {
    if (later.type !== 'tool_result' || isReplacedStepRow(later)) continue;
    for (const result of list(later.tool_results)) {
      if (typeof result.tool_call_id === 'string') finished.add(result.tool_call_id);
    }
    for (const response of list(later.tool_approval_responses)) {
      if (typeof response.approval_id === 'string' && !decided.has(response.approval_id)) decided.set(response.approval_id, response.approved === true);
    }
  }

  const approvedCalls = new Set(list(row.tool_approval_requests)
    .filter((request) => typeof request.approval_id === 'string' && decided.get(request.approval_id) === true && typeof request.tool_call_id === 'string')
    .map((request) => request.tool_call_id as string));

  return list(row.tool_calls).filter((call) => typeof call.id === 'string' && approvedCalls.has(call.id)
    && !finished.has(call.id) && call.name === name && digestOf(call.arguments) === digest).length;
}
