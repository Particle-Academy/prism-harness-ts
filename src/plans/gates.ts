import type { JsonObject } from '../json.js';
import type { HarnessTool } from '../tools.js';

/**
 * A plan tool decides for itself, per call, whether a person must approve it
 * (G-360). `authorized` is the host call policy's verdict on this call, asked
 * by the runtime BEFORE the gate, so a call the host forbids records nothing.
 */
export type ApprovalGate = (args: JsonObject, authorized: boolean) => Promise<boolean>;

/**
 * The plan tools' own approval gates, keyed by the tool object the harness
 * built. Only a tool built by `Plans`, or the authorizer's wrapper around one,
 * is in this map, so no registered tool can claim to gate itself and slip out
 * from under its mode's approval list.
 */
const GATES = new WeakMap<HarnessTool, ApprovalGate>();

/** @internal */
export function setPlanApprovalGate(tool: HarnessTool, gate: ApprovalGate): void {
  GATES.set(tool, gate);
}

/**
 * The approval gate of a plan tool, or null for every other tool.
 *
 * The runtime consults this INSTEAD of the mode's approval list and the host
 * tool policy, so neither a mode gating `*` nor a permissive policy can wrap
 * or bypass a plan tool's approval.
 *
 * @internal
 */
export function planApprovalGate(tool: HarnessTool | undefined): ApprovalGate | null {
  return tool === undefined ? null : (GATES.get(tool) ?? null);
}

/**
 * A wrapper around a plan tool keeps the plan tool's gate. Without this, the
 * authorizer's call-time wrapper hid the gate, and with authorization on a
 * plan tool was neither gated by itself nor able to record a proposal.
 *
 * @internal
 */
export function carryPlanApprovalGate(from: HarnessTool, to: HarnessTool): void {
  const gate = GATES.get(from);
  if (gate !== undefined) GATES.set(to, gate);
}
