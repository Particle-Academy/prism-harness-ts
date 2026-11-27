import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  ToolRegistry,
  recordApproval,
  recordDecisions,
  type AgentResponse,
  type ApprovalDecision,
  type JsonObject,
  type LlmRequest,
  type LlmResponse,
  type LlmToolCall,
  type RunStep,
  type Session,
} from '../src/index.js';

// G-392 / G-397: the step whose repeated ids are checked must be exactly the
// step whose decisions are resolved, whatever rows were recorded between the
// park and the decision. Here an assistant text row, a user row, a system row,
// an unrelated tool result row or an empty tool result row is written into
// the thread after the step parked (before or after the decisions), and the
// decisions go through recordApproval, recordDecisions and streamDecision.
// Offline; every store is in memory.

const REPEATED = 'Tool call was not run because another call in this step has the same id';

const network = vi.fn(() => { throw new Error('A test tried to use the network.'); });
beforeEach(() => { vi.stubGlobal('fetch', network); });
afterEach(() => { expect(network).not.toHaveBeenCalled(); vi.unstubAllGlobals(); });

class DurableMemory extends MemorySessionStore {
  override durability(): 'durable' {
    return 'durable';
  }
}

const modes = new ModeRegistry({
  default: 'guarded',
  modes: { guarded: { system_prompt: 'Careful.', tools: ['echo', 'free'], max_steps: 6, requires_approval: ['echo'] } },
});

/** The rows a writer could record between the park and the decision. */
const INTERPOSED: Record<string, JsonObject> = {
  'an assistant text row': { type: 'assistant', content: 'Still thinking.', tool_calls: [], tool_approval_requests: [] },
  'a user row': { type: 'user', content: 'Any news?' },
  'a system row': { type: 'system', content: 'A note from the host.' },
  'an unrelated tool result row': {
    type: 'tool_result',
    tool_results: [{ tool_call_id: 'elsewhere', tool_name: 'free', args: { n: 0 }, result: 'unrelated', tool_call_result_id: null, artifacts: [] }],
    tool_approval_responses: [],
  },
  'an empty tool result row': { type: 'tool_result', tool_results: [], tool_approval_responses: [] },
};

interface Rig {
  session: Session;
  agent: AgentRuntime;
  ran: number[];
}

function rig(calls: LlmToolCall[]): Rig {
  const harness = new PrismHarness({
    drivers: { memory: () => new MemorySessionStore(), durable: () => new DurableMemory() },
    stores: { ephemeral: 'memory', durable: 'durable' },
  });
  const session = harness.for({ type: 'User', id: 1 }).session('interposed');
  const ran: number[] = [];
  const tools = new ToolRegistry()
    .register({ name: 'echo', handle: (args: JsonObject) => { ran.push(Number(args.n)); return `ran:${String(args.n)}`; } })
    .register({ name: 'free', handle: () => 'free ok' });
  let served = false;
  const client = async (_request: LlmRequest): Promise<LlmResponse> => {
    if (served) return { text: 'done', finishReason: 'stop' };
    served = true;
    return { text: '', finishReason: 'tool_calls', toolCalls: calls };
  };

  return { session, agent: new AgentRuntime({ client, modes, tools }), ran };
}

const echo = (id: string, n: number): LlmToolCall => ({ id, name: 'echo', arguments: { n } });

async function drain(stream: AsyncGenerator<RunStep, AgentResponse, undefined>): Promise<AgentResponse> {
  for (;;) {
    const next = await stream.next();
    if (next.done === true) return next.value;
  }
}

/** Every result recorded for the echo call carrying `n`, across the thread. */
async function outcomesOf(session: Session, n: number): Promise<string[]> {
  const rows = (await session.thread().messages()).map((entry) => entry.message);
  return [...new Set(rows.filter((row) => row.type === 'tool_result')
    .flatMap((row) => row.tool_results as JsonObject[])
    .filter((entry) => entry.tool_name === 'echo' && (entry.args as JsonObject).n === n)
    .map((entry) => String(entry.result)))];
}

type Decide = 'recordApproval' | 'recordDecisions' | 'streamDecision';

async function decideAndResume(r: Rig, decisions: ApprovalDecision[], how: Decide, row: JsonObject, where: 'before' | 'after'): Promise<void> {
  if (where === 'before') await r.session.thread().record([row]);
  if (how === 'streamDecision') {
    if (where === 'after') throw new Error('streamDecision records and resumes in one call');
    await drain(r.agent.streamDecision(r.session, decisions));
    return;
  }
  if (how === 'recordDecisions') await recordDecisions(r.session, decisions);
  else for (const d of decisions) await recordApproval(r.session, d.approvalId, d.approved, d.reason ?? null);
  if (where === 'after') await r.session.thread().record([row]);
  await r.agent.send(r.session, '');
}

const PLACES: [Decide, 'before' | 'after'][] = [
  ['recordApproval', 'before'], ['recordApproval', 'after'],
  ['recordDecisions', 'before'], ['recordDecisions', 'after'],
  ['streamDecision', 'before'],
];

describe('G-397: a row between the park and the decision never makes a repeated-id step runnable', () => {
  for (const [label, row] of Object.entries(INTERPOSED)) {
    for (const [how, where] of PLACES) {
      for (const verdicts of [[true, true], [false, true], [true, false]]) {
        it(`${label} ${where} the decisions (${how}, ${verdicts.map((v) => (v ? 'approve' : 'deny')).join('/')}): neither same-id call runs, each is denied once`, async () => {
          const r = rig([echo('weather', 1), echo('weather', 2)]);
          const parked = await r.agent.send(r.session, 'go');
          expect(parked.finishReason).toBe('awaiting_approval');
          expect(parked.pendingApprovals).toHaveLength(2);
          const decisions = parked.pendingApprovals.map((p, i) => ({ approvalId: p.id, approved: verdicts[i]!, reason: verdicts[i] ? null : 'no' }));

          await decideAndResume(r, decisions, how, row, where);
          // A later approval and repeat resumes revive nothing.
          for (const p of parked.pendingApprovals) await recordApproval(r.session, p.id, true).catch(() => undefined);
          await r.agent.send(r.session, '');
          await drain(r.agent.streamSteps(r.session, ''));

          expect(r.ran).toEqual([]);
          expect(await outcomesOf(r.session, 1)).toEqual([REPEATED]);
          expect(await outcomesOf(r.session, 2)).toEqual([REPEATED]);
        });
      }
    }
  }

  it('empty ids, with an ungated call sharing the id, and a user row between: no gated call runs', async () => {
    const r = rig([{ id: '', name: 'free', arguments: {} }, echo('', 1), echo('', 2)]);
    const parked = await r.agent.send(r.session, 'go');
    await r.session.thread().record([INTERPOSED['a user row']!]);
    await recordDecisions(r.session, parked.pendingApprovals.map((p) => ({ approvalId: p.id, approved: true })));
    await r.agent.send(r.session, '');

    expect(r.ran).toEqual([]);
    expect(await outcomesOf(r.session, 1)).toEqual([REPEATED]);
    expect(await outcomesOf(r.session, 2)).toEqual([REPEATED]);
  });
});

describe('G-397 control: distinct ids with a row between are decided per call, never beyond their own approval', () => {
  for (const [label, row] of Object.entries(INTERPOSED)) {
    for (const [how, where] of PLACES) {
      it(`${label} ${where} the decisions (${how}): a denied call never runs, an approved one runs at most once`, async () => {
        const r = rig([echo('c1', 1), echo('c2', 2)]);
        const parked = await r.agent.send(r.session, 'go');
        const [one, two] = parked.pendingApprovals;
        await decideAndResume(r, [{ approvalId: one!.id, approved: false, reason: 'no' }, { approvalId: two!.id, approved: true }], how, row, where);
        await recordApproval(r.session, one!.id, true).catch(() => undefined);
        await r.agent.send(r.session, '');

        expect(r.ran).not.toContain(1);
        expect(r.ran.filter((n) => n === 2).length).toBeLessThanOrEqual(1);
        // Each call has exactly one outcome, and a call that did not run is not reported as run.
        const first = await outcomesOf(r.session, 1);
        const second = await outcomesOf(r.session, 2);
        expect(first).toHaveLength(1);
        expect(first[0]).not.toMatch(/^ran:/);
        expect(second).toHaveLength(1);
        expect(second[0] === 'ran:2').toBe(r.ran.includes(2));
      });
    }
  }
});

describe('G-397: a resolved call is never run again because a row sits between its decisions and its results', () => {
  for (const [label, row] of Object.entries(INTERPOSED)) {
    for (const how of ['recordApproval', 'recordDecisions'] as const) {
      it(`${label} after the decisions (${how}), and the turn after the resolution fails: the approved call ran once, and the next resume runs nothing`, async () => {
        const harness = new PrismHarness({
          drivers: { memory: () => new MemorySessionStore(), durable: () => new DurableMemory() },
          stores: { ephemeral: 'memory', durable: 'durable' },
        });
        const session = harness.for({ type: 'User', id: 1 }).session('interposed-failed');
        const ran: number[] = [];
        const tools = new ToolRegistry()
          .register({ name: 'echo', handle: (args: JsonObject) => { ran.push(Number(args.n)); return `ran:${String(args.n)}`; } })
          .register({ name: 'free', handle: () => 'free ok' });
        let model = 0;
        const agent = new AgentRuntime({
          modes, tools, client: async (): Promise<LlmResponse> => {
            model += 1;
            if (model === 1) return { text: '', finishReason: 'tool_calls', toolCalls: [echo('c1', 1), echo('c2', 2)] };
            if (model === 2) throw new Error('The model is unavailable (scripted).');
            return { text: 'done', finishReason: 'stop' };
          },
        });
        const [one, two] = (await agent.send(session, 'go')).pendingApprovals;
        const decisions = [{ approvalId: one!.id, approved: false, reason: 'no' }, { approvalId: two!.id, approved: true }];
        if (how === 'recordDecisions') await recordDecisions(session, decisions);
        else for (const d of decisions) await recordApproval(session, d.approvalId, d.approved, d.reason ?? null);
        await session.thread().record([row]);

        await expect(agent.send(session, '')).rejects.toThrow('scripted');
        // A row that moves the tail (a user or assistant text row) may make the
        // approved call fail closed; otherwise it ran. Either way, once.
        expect(ran.length).toBeLessThanOrEqual(1);
        const once = [...ran];
        const again = await agent.send(session, '');
        await drain(agent.streamSteps(session, ''));

        expect(ran).toEqual(once);
        expect(again.toolCalls).toEqual([]);
        expect(await outcomesOf(session, 1)).toEqual(['no']);
        const second = await outcomesOf(session, 2);
        expect(second).toHaveLength(1);
        expect(second[0] === 'ran:2').toBe(ran.includes(2));
      });
    }
  }
});
