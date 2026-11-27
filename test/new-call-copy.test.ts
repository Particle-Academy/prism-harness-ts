import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  ToolRegistry,
  recordApproval,
  type JsonObject,
  type JsonValue,
  type LlmResponse,
  type Session,
} from '../src/index.js';

// G-408, order B: the runtime parks on a gated call; BEFORE the person
// decides, the host records a copy of the parked step that keeps its
// approval request but holds a NEW call with the same id and other
// arguments; then the person approves the request they were shown. The copy
// replaces the parked step, so the approval is refused; the copy's requests
// repeat an approval id the parked step already asked for, so they ask for
// nothing. Nothing runs, the old call and every new call end with one
// outcome each. Offline; every store is in memory.

const REPLACED = 'Tool call was not run because a later step replaced it';
const REPEATED = 'Tool call was not run because another call in this step has the same id';
const NO_RESPONSE = 'No approval response provided';

const network = vi.fn(() => { throw new Error('A test tried to use the network.'); });
beforeEach(() => { vi.stubGlobal('fetch', network); });
afterEach(() => { expect(network).not.toHaveBeenCalled(); vi.unstubAllGlobals(); });

class DurableMemory extends MemorySessionStore {
  override durability(): 'durable' {
    return 'durable';
  }
}

const newCall = (old: JsonObject): JsonObject => ({ ...old, arguments: { n: 2 } });
const VARIANTS: [string, (old: JsonObject) => JsonValue[], string[]][] = [
  ['newOnly', (old) => [newCall(old)], [NO_RESPONSE]],
  ['stringThenNew', (old) => ['x', newCall(old)], [NO_RESPONSE]],
  ['nullThenNew', (old) => [null, newCall(old)], [NO_RESPONSE]],
  ['newThenJunk', (old) => [newCall(old), 'x', null], [NO_RESPONSE]],
  ['oldAndNewSameId', (old) => [{ ...old }, newCall(old)], [REPEATED, REPEATED]],
];

async function run(variant: (old: JsonObject) => JsonValue[], options: { maxSteps: number; failAfter: boolean }): Promise<{ ran: number[]; refused: unknown; byN: Record<number, string[]> }> {
  const harness = new PrismHarness({
    drivers: { memory: () => new MemorySessionStore(), durable: () => new DurableMemory() },
    stores: { ephemeral: 'memory', durable: 'durable' },
  });
  const session: Session = harness.for({ type: 'User', id: 1 }).session('new-call-copy');
  const ran: number[] = [];
  let model = 0;
  const agent = new AgentRuntime({
    modes: new ModeRegistry({ default: 'guarded', modes: { guarded: { system_prompt: 'Careful.', tools: ['echo'], max_steps: options.maxSteps, requires_approval: ['echo'] } } }),
    tools: new ToolRegistry().register({ name: 'echo', handle: (args: JsonObject) => { ran.push(Number(args.n)); return `ran:${String(args.n)}`; } }),
    client: async (): Promise<LlmResponse> => {
      model += 1;
      if (model === 1) return { text: '', finishReason: 'tool_calls', toolCalls: [{ id: 'c1', name: 'echo', arguments: { n: 1 } }] };
      if (options.failAfter) throw new Error('The provider is down (scripted).');
      return { text: 'done', finishReason: 'stop' };
    },
  });
  const [pending] = (await agent.send(session, 'go')).pendingApprovals;
  const parked = (await session.thread().messages()).map((entry) => entry.message).find((row) => row.type === 'assistant')!;
  await session.thread().record([{ ...parked, tool_calls: variant((parked.tool_calls as JsonObject[])[0]!) }]);
  const refused = await recordApproval(session, pending!.id, true).then(() => 'accepted', (error: unknown) => (error as { code?: string }).code);
  for (let i = 0; i < 3; i += 1) await agent.send(session, '').catch(() => undefined);
  const byN: Record<number, string[]> = {};
  for (const row of (await session.thread().messages()).map((entry) => entry.message)) {
    if (row.type !== 'tool_result' || !Array.isArray(row.tool_results)) continue;
    for (const entry of row.tool_results as JsonObject[]) (byN[Number((entry.args as JsonObject).n)] ??= []).push(String(entry.result));
  }
  return { ran, refused, byN };
}

describe('G-408 order B: a copy with a new call recorded before the person decides', () => {
  for (const [name, variant, fresh] of VARIANTS) {
    for (const [label, options] of [['the run\'s steps spent', { maxSteps: 1, failAfter: false }], ['the model failing after each resolution', { maxSteps: 6, failAfter: true }]] as [string, { maxSteps: number; failAfter: boolean }][]) {
      it(`${name}, ${label}: the approval is refused, nothing runs, and each call has one outcome`, async () => {
        const { ran, refused, byN } = await run(variant, options);

        expect(refused).toBe('approval_not_pending');
        expect(ran).toEqual([]);
        // The old call (n 1) is replaced once; the copy's calls are answered
        // once each (a copy of the old call shares n 1 with it).
        expect([...new Set(byN[2])]).toEqual([fresh.at(-1)]);
        if (name === 'oldAndNewSameId') expect([...new Set(byN[1])].sort()).toEqual([REPEATED, REPLACED].sort());
        else expect([...new Set(byN[1])]).toEqual([REPLACED]);
      });
    }
  }
});
