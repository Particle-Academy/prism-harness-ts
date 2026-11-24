import type { JsonObject, JsonValue } from '../json.js';
import { pendingApprovalIds } from '../thread-rows.js';
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
  if (ids.length === 0) return [];
  const calls: PendingCall[] = [];
  for (const id of ids) {
    for (let index = rows.length - 1; index >= 0; index -= 1) {
      const row = rows[index]!;
      if (row.type !== 'assistant') continue;
      const request = list(row.tool_approval_requests).find((entry) => entry.approval_id === id);
      if (request === undefined) continue;
      const call = list(row.tool_calls).find((entry) => entry.id === request.tool_call_id);
      if (call !== undefined && typeof call.name === 'string') calls.push({ id, name: call.name, digest: digestOf(call.arguments) });
      break;
    }
  }

  return calls;
}

/** Every recorded decision, by approval id: true when approved. The first decision for an id stands. */
export function approvalDecisions(rows: readonly JsonObject[]): Map<string, boolean> {
  const decided = new Map<string, boolean>();
  for (const row of rows) {
    if (row.type !== 'tool_result') continue;
    for (const response of list(row.tool_approval_responses)) {
      if (typeof response.approval_id === 'string' && !decided.has(response.approval_id)) {
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
  // Read newest first, so the last write per id is the FIRST decision
  // recorded: the one answer that counts, as for the resume and the plan.
  const decided = new Map<string, boolean>();
  const finished = new Set<string>();
  for (let index = rows.length - 1; index >= 0; index -= 1) {
    const row = rows[index]!;
    if (row.type === 'tool_result') {
      for (const result of list(row.tool_results)) {
        if (typeof result.tool_call_id === 'string') finished.add(result.tool_call_id);
      }
      for (const response of list(row.tool_approval_responses).reverse()) {
        if (typeof response.approval_id === 'string') decided.set(response.approval_id, response.approved === true);
      }
      continue;
    }
    if (row.type !== 'assistant') return 0;

    const approvedCalls = new Set(list(row.tool_approval_requests)
      .filter((request) => typeof request.approval_id === 'string' && decided.get(request.approval_id) === true && typeof request.tool_call_id === 'string')
      .map((request) => request.tool_call_id as string));

    return list(row.tool_calls).filter((call) => typeof call.id === 'string' && approvedCalls.has(call.id)
      && !finished.has(call.id) && call.name === name && digestOf(call.arguments) === digest).length;
  }

  return 0;
}
