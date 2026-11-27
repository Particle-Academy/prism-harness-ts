import { readFileSync } from 'node:fs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  HarnessError,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  ToolRegistry,
  recordDecisions,
  type JsonObject,
  type LlmResponse,
} from '../src/index.js';

/**
 * The cross-language approval-pairing corpus (harness-approval-pairing, G-392).
 *
 * Every row's `expected` was written by hand before any language ran. This
 * runner drives the real runtime offline with a fake model client: the first
 * reply is the row's calls, each under the row's id, and every later reply is
 * the text `done`. `echo` is gated by mode and `free` is not. Each round
 * decides the listed approvals with one recordDecisions() call, then resumes
 * once with an empty prompt; a decision the harness refuses is swallowed.
 * `decided` (0.2.0) is the decision the thread recorded for each approval.
 */
interface PairingCase {
  id: string;
  title: string;
  calls: { id: string; name: string; args: JsonObject }[];
  rounds: [number, boolean][][];
  expected: unknown;
}

const corpus = JSON.parse(
  readFileSync(new URL('./fixtures/harness-approval-pairing.json', import.meta.url), 'utf8'),
) as { cases: PairingCase[] };

/** An in-memory store that declares itself durable, for an offline run. */
class DurableMemory extends MemorySessionStore {
  override durability(): 'durable' {
    return 'durable';
  }
}

const network = vi.fn(() => { throw new Error('A test tried to use the network.'); });
beforeEach(() => { vi.stubGlobal('fetch', network); });
afterEach(() => { expect(network).not.toHaveBeenCalled(); vi.unstubAllGlobals(); });

async function record(row: PairingCase): Promise<unknown> {
  const ran: number[] = [];
  let free = 0;
  const tools = new ToolRegistry()
    .register({ name: 'echo', handle: (args: JsonObject) => { ran.push(Number(args.n)); return 'ok'; } })
    .register({ name: 'free', handle: () => { free += 1; return 'free ok'; } });
  const modes = new ModeRegistry({
    default: 'guarded',
    modes: { guarded: { system_prompt: 'Careful.', tools: ['echo', 'free'], max_steps: 6, requires_approval: ['echo'] } },
  });
  let step = 0;
  const client = async (): Promise<LlmResponse> => {
    step += 1;
    return step === 1
      ? { text: '', finishReason: 'tool_calls', toolCalls: row.calls.map((call) => ({ id: call.id, name: call.name, arguments: call.args })) }
      : { text: 'done', finishReason: 'stop' };
  };
  const harness = new PrismHarness({
    drivers: { memory: () => new MemorySessionStore(), durable: () => new DurableMemory() },
    stores: { ephemeral: 'memory', durable: 'durable' },
  });
  const session = harness.for({ type: 'User', id: 1 }).session('corpus');
  const runtime = new AgentRuntime({ client, modes, tools });

  const pending = (await runtime.send(session, 'go')).pendingApprovals;
  for (const round of row.rounds) {
    try {
      await recordDecisions(session, round.map(([index, approved]) => ({ approvalId: pending[index]!.id, approved })));
    } catch (error) {
      // A decision the harness refuses is swallowed; the round still resumes.
      if (!(error instanceof HarnessError && error.code === 'approval_not_pending')) throw error;
    }
    await runtime.send(session, '');
  }

  // The decision the thread RECORDED for each approval the first step issued,
  // in issue order (0.2.0): the first answer, or null when none was recorded.
  const recorded = new Map<string, boolean>();
  for (const { message } of await session.thread().messages()) {
    if (message.type !== 'tool_result') continue;
    for (const answer of (message.tool_approval_responses ?? []) as JsonObject[]) {
      if (typeof answer.approval_id === 'string' && !recorded.has(answer.approval_id)) recorded.set(answer.approval_id, answer.approved === true);
    }
  }

  return { approvals: pending.length, free, ran: [...ran].sort((a, b) => a - b), decided: pending.map((approval) => recorded.get(approval.id) ?? null) };
}

/** Set to an absolute path to write what this port measures for every row (for the parity recorder). */
const RECORD = process.env.HARNESS_APPROVAL_PAIRING_RECORD;

describe('harness-approval-pairing corpus', () => {
  if (RECORD !== undefined && RECORD !== '') {
    it("records this port's outcomes", async () => {
      const { writeFileSync } = await import('node:fs');
      const out: Record<string, unknown> = {};
      for (const row of corpus.cases) out[row.id] = await record(row);
      writeFileSync(RECORD, JSON.stringify(out, null, 2));
    });
  }

  it('has rows', () => {
    expect(corpus.cases.length).toBeGreaterThan(0);
  });

  for (const row of corpus.cases) {
    it(`${row.id}: ${row.title}`, async () => {
      expect(await record(row)).toEqual(row.expected);
    });
  }
});
