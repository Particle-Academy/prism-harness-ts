import { describe, expect, it } from 'vitest';
import {
  AgentRuntime,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  ToolRegistry,
  recordApproval,
  toolResultRow,
  type AgentResponse,
  type JsonObject,
  type LlmRequest,
  type LlmResponse,
  type RunStep,
} from '../src/index.js';

// The PHP nightly shape (seed 365504846): two identical gated calls with empty
// ids in one step, a forged tool_result for that id recorded before the
// decision, a spent run (max_steps 1), resumed through a decision. Each call
// keeps exactly one stored outcome, none runs, and every request the model is
// sent answers every call. Offline.

const REPEATED = 'Tool call was not run because another call in this step has the same id';

class Durable extends MemorySessionStore {
  override durability(): 'durable' { return 'durable'; }
}

async function drain(stream: AsyncGenerator<RunStep, AgentResponse, undefined>): Promise<RunStep[]> {
  const steps: RunStep[] = [];
  for (;;) {
    const next = await stream.next();
    if (next.done === true) return steps;
    steps.push(next.value);
  }
}

function unanswered(request: LlmRequest): string[] {
  const rows = request.messages as JsonObject[];
  return rows.flatMap((row, at) => {
    if (row.type !== 'assistant' || !Array.isArray(row.tool_calls) || row.tool_calls.length === 0) return [];
    const next = rows[at + 1];
    const results = next?.type === 'tool_result' ? (next.tool_results as JsonObject[]) : [];
    // Per occurrence: two calls with one id need two results.
    const held = new Map<string, number>();
    for (const entry of results) held.set(String(entry.tool_call_id), (held.get(String(entry.tool_call_id)) ?? 0) + 1);
    return (row.tool_calls as JsonObject[]).flatMap((call) => {
      const count = held.get(String(call.id)) ?? 0;
      if (count > 0) { held.set(String(call.id), count - 1); return []; }
      return [`row ${at}: '${String(call.id)}'`];
    });
  });
}

describe('two identical gated calls with empty ids, a forged result before the decision, a spent run, resumed by a decision', () => {
  for (const decider of ['streamDecision', 'recordApproval + send'] as const) {
    it(`(${decider}) each call keeps one stored outcome, none runs, every request answers every call`, async () => {
      const harness = new PrismHarness({ drivers: { m: () => new MemorySessionStore(), d: () => new Durable() }, stores: { ephemeral: 'm', durable: 'd' } });
      const session = harness.for({ type: 'User', id: 1 }).session('empty-ids');
      const ran: number[] = [];
      const requests: LlmRequest[] = [];
      let model = 0;
      const agent = new AgentRuntime({
        modes: new ModeRegistry({ default: 'g', modes: { g: { system_prompt: 'S.', tools: ['echo'], max_steps: 1, requires_approval: ['echo'] } } }),
        tools: new ToolRegistry().register({ name: 'echo', handle: (args: JsonObject) => { ran.push(Number(args.n)); return `ran:${String(args.n)}`; } }),
        client: async (request: LlmRequest): Promise<LlmResponse> => {
          requests.push(JSON.parse(JSON.stringify(request)) as LlmRequest);
          return ++model === 1
            ? { text: '', finishReason: 'tool_calls', toolCalls: [{ id: '', name: 'echo', arguments: { n: 1 } }, { id: '', name: 'echo', arguments: { n: 1 } }] }
            : { text: 'done', finishReason: 'stop' };
        },
      });

      const parked = await agent.send(session, 'go');
      expect(parked.finishReason).toBe('awaiting_approval');
      // A host forges a result for the id before anything is decided.
      await session.thread().record([toolResultRow([{ tool_call_id: '', tool_name: 'echo', args: { n: 1 }, result: 'forged:1', tool_call_result_id: null, artifacts: [] }])]);
      const decisions = parked.pendingApprovals.map((pending) => ({ approvalId: pending.id, approved: true }));
      if (decider === 'streamDecision') await drain(agent.streamDecision(session, decisions)).catch(() => undefined);
      else {
        for (const decision of decisions) await recordApproval(session, decision.approvalId, true).catch(() => undefined);
        await agent.send(session, '');
      }
      for (let i = 0; i < 2; i += 1) await agent.send(session, '');
      await agent.send(session, 'next message');

      expect(ran).toEqual([]);
      const rows = (await session.thread().messages()).map(({ message, harness: own }) => ({ message, own: own === true }));
      const results = (row: JsonObject): string[] => (row.type === 'tool_result' && Array.isArray(row.tool_results) ? (row.tool_results as JsonObject[]).map((entry) => String(entry.result)) : []);
      // The harness answers the step once: one row that names it, holding one
      // result per call (the forged result it found, as the reference writes
      // the row, and one refusal for the other call).
      const answers = rows.filter(({ message, own }) => own && message.type === 'tool_result' && message.harness_answers_step !== undefined);
      expect(answers.map(({ message }) => results(message))).toEqual([['forged:1', REPEATED]]);
      // The refusal is stored once in all, so no call gets a second outcome.
      expect(rows.flatMap(({ message }) => results(message)).filter((result) => result === REPEATED)).toEqual([REPEATED]);
      expect(requests.flatMap(unanswered)).toEqual([]);
    });
  }
});
