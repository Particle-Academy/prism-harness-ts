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
  type LlmResponse,
  type LlmToolCall,
  type RunStep,
  type Session,
} from '../src/index.js';
import { ARGUMENTS_AS_SENT, assistantRow, threadView, toolResultEntry, toolResultRow } from '../src/thread-rows.js';

// G-392 / G-397, two rules for every resolver:
//  - a parked step followed by a later tool-call step is REPLACED: each of its
//    gated calls with no result of its own gets one denial, and a decision for
//    it never runs anything; the later step is decided by its own approvals;
//  - a step whose approval requests repeat an APPROVAL id runs none of its
//    gated calls, and each gets the repeated-id denial.
// Offline; every store is in memory.

const REPLACED = 'Tool call was not run because a later step replaced it';
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

interface Rig {
  session: Session;
  agent: AgentRuntime;
  ran: number[];
  requests: number;
}

function rig(calls: LlmToolCall[]): Rig {
  const harness = new PrismHarness({
    drivers: { memory: () => new MemorySessionStore(), durable: () => new DurableMemory() },
    stores: { ephemeral: 'memory', durable: 'durable' },
  });
  const session = harness.for({ type: 'User', id: 1 }).session('replaced');
  const r: Rig = { session, agent: undefined as unknown as AgentRuntime, ran: [], requests: 0 };
  const tools = new ToolRegistry()
    .register({ name: 'echo', handle: (args: JsonObject) => { r.ran.push(Number(args.n)); return `ran:${String(args.n)}`; } })
    .register({ name: 'free', handle: (args: JsonObject) => `free:${String(args.n)}` });
  r.agent = new AgentRuntime({
    modes, tools, client: async (): Promise<LlmResponse> => {
      r.requests += 1;
      return r.requests === 1 && calls.length > 0 ? { text: '', finishReason: 'tool_calls', toolCalls: calls } : { text: 'done', finishReason: 'stop' };
    },
  });
  return r;
}

const echo = (id: string, n: number): LlmToolCall => ({ id, name: 'echo', arguments: { n } });

/** A later tool-call step, recorded onto the thread as another writer can: [call, approval id] pairs. */
async function recordStep(session: Session, calls: [LlmToolCall, string | null][]): Promise<void> {
  await session.thread().record([{
    ...assistantRow('', calls.map(([call]) => call), {}, calls.flatMap(([call, approval]) => (approval === null ? [] : [{
      approval_id: approval, tool_call_id: call.id, arguments_format: ARGUMENTS_AS_SENT,
    }]))),
    harness_tool_names: ['echo', 'free'],
  }]);
}

async function drain(stream: AsyncGenerator<RunStep, AgentResponse, undefined>): Promise<AgentResponse> {
  for (;;) {
    const next = await stream.next();
    if (next.done === true) return next.value;
  }
}

/** Every distinct result recorded for the call carrying `n`. */
async function outcomesOf(session: Session, n: number): Promise<string[]> {
  const rows = (await session.thread().messages()).map((entry) => entry.message);
  return [...new Set(rows.filter((row) => row.type === 'tool_result')
    .flatMap((row) => row.tool_results as JsonObject[])
    .filter((entry) => (entry.args as JsonObject).n === n)
    .map((entry) => String(entry.result)))];
}

type Decide = 'recordApproval' | 'recordDecisions' | 'streamDecision';

async function decide(r: Rig, how: Decide, decisions: ApprovalDecision[]): Promise<void> {
  if (how === 'streamDecision') {
    await drain(r.agent.streamDecision(r.session, decisions));
    return;
  }
  if (how === 'recordDecisions') await recordDecisions(r.session, decisions);
  else for (const d of decisions) await recordApproval(r.session, d.approvalId, d.approved, d.reason ?? null);
  await r.agent.send(r.session, '');
}

const HOW: Decide[] = ['recordApproval', 'recordDecisions', 'streamDecision'];

describe('G-397: a parked step a later tool-call step replaced', () => {
  for (const how of HOW) {
    it(`(${how}) each of its gated calls is denied once, its approvals run nothing, and the later step is decided by its own`, async () => {
      const r = rig([echo('c1', 1), echo('c2', 2)]);
      const [one, two] = (await r.agent.send(r.session, 'go')).pendingApprovals;
      await recordStep(r.session, [[echo('c3', 3), 'apr_later']]);

      await decide(r, how, [
        { approvalId: one!.id, approved: true },
        { approvalId: two!.id, approved: true },
        { approvalId: 'apr_later', approved: true },
      ]);
      // Approving the old step again, and resuming again, still runs nothing of it.
      await recordApproval(r.session, one!.id, true);
      await r.agent.send(r.session, '');
      await drain(r.agent.streamSteps(r.session, ''));

      expect(r.ran).toEqual([3]);
      expect(await outcomesOf(r.session, 1)).toEqual([REPLACED]);
      expect(await outcomesOf(r.session, 2)).toEqual([REPLACED]);
      expect(await outcomesOf(r.session, 3)).toEqual(['ran:3']);
    });
  }

  it('the later step reuses the old call ids: the old calls are denied as replaced, the later calls are decided by their own approvals', async () => {
    const r = rig([echo('a', 1), echo('b', 2)]);
    const [one, two] = (await r.agent.send(r.session, 'go')).pendingApprovals;
    await recordStep(r.session, [[echo('a', 3), 'apr_a'], [echo('b', 4), 'apr_b']]);
    await recordDecisions(r.session, [
      { approvalId: one!.id, approved: true },
      { approvalId: two!.id, approved: true },
      { approvalId: 'apr_a', approved: false, reason: 'not this one' },
      { approvalId: 'apr_b', approved: true },
    ]);
    await r.agent.send(r.session, '');
    await r.agent.send(r.session, '');

    expect(r.ran).toEqual([4]);
    expect(await outcomesOf(r.session, 1)).toEqual([REPLACED]);
    expect(await outcomesOf(r.session, 2)).toEqual([REPLACED]);
    expect(await outcomesOf(r.session, 3)).toEqual(['not this one']);
    expect(await outcomesOf(r.session, 4)).toEqual(['ran:4']);
  });

  it('a call of the old step that has its own result keeps it; an ungated call is never denied', async () => {
    const r = rig([{ id: 'f', name: 'free', arguments: { n: 1 } }, echo('e', 2)]);
    const [only] = (await r.agent.send(r.session, 'go')).pendingApprovals;
    await recordStep(r.session, [[echo('x', 3), 'apr_x']]);
    await recordDecisions(r.session, [{ approvalId: only!.id, approved: true }, { approvalId: 'apr_x', approved: false }]);
    await r.agent.send(r.session, '');

    expect(r.ran).toEqual([]);
    expect(await outcomesOf(r.session, 1)).toEqual(['free:1']);
    expect(await outcomesOf(r.session, 2)).toEqual([REPLACED]);
    expect(await outcomesOf(r.session, 3)).toEqual(['User denied tool execution']);
  });

  it('the later step needs nobody: the old calls are still denied once', async () => {
    const r = rig([echo('c1', 1)]);
    const [only] = (await r.agent.send(r.session, 'go')).pendingApprovals;
    await recordStep(r.session, [[{ id: 'f', name: 'free', arguments: { n: 9 } }, null]]);
    await r.session.thread().record([toolResultRow([toolResultEntry({ id: 'f', name: 'free', arguments: { n: 9 } }, 'free:9')])]);
    await recordApproval(r.session, only!.id, true);
    await r.agent.send(r.session, '');
    await r.agent.send(r.session, '');

    expect(r.ran).toEqual([]);
    expect(await outcomesOf(r.session, 1)).toEqual([REPLACED]);
  });

  it('the replaced denials never reach the provider in place of the later step\'s results', async () => {
    const r = rig([echo('a', 1)]);
    const [only] = (await r.agent.send(r.session, 'go')).pendingApprovals;
    await recordStep(r.session, [[echo('a', 3), 'apr_a']]);
    await recordDecisions(r.session, [{ approvalId: only!.id, approved: true }, { approvalId: 'apr_a', approved: true }]);
    await r.agent.send(r.session, '');

    expect(r.ran).toEqual([3]);
    const view = threadView((await r.session.thread().messages()).map((entry) => entry.message));
    const results = view.filter((row) => row.type === 'tool_result').flatMap((row) => row.tool_results as JsonObject[]);
    expect(results.filter((entry) => entry.tool_call_id === 'a').map((entry) => entry.result)).toEqual(['ran:3']);
  });
});

describe('G-397: a step whose approval requests repeat an approval id', () => {
  for (const how of HOW) {
    it(`(${how}) runs none of its gated calls, each gets the repeated-id denial`, async () => {
      const r = rig([]);
      await r.session.thread().record([{ type: 'user', content: 'go' }]);
      await recordStep(r.session, [[echo('o1', 1), 'apr_same'], [echo('l1', 2), 'apr_same']]);

      await decide(r, how, [{ approvalId: 'apr_same', approved: true }]);
      await r.agent.send(r.session, '');

      expect(r.ran).toEqual([]);
      expect(await outcomesOf(r.session, 1)).toEqual([REPEATED]);
      expect(await outcomesOf(r.session, 2)).toEqual([REPEATED]);
    });
  }
});

describe('G-398 does not apply to this runtime', () => {
  it('an approved call runs before the provider send, and its result is recorded first: a retry after a failed send runs nothing again', async () => {
    const harness = new PrismHarness({
      drivers: { memory: () => new MemorySessionStore(), durable: () => new DurableMemory() },
      stores: { ephemeral: 'memory', durable: 'durable' },
    });
    const session = harness.for({ type: 'User', id: 1 }).session('retry');
    const ran: number[] = [];
    let model = 0;
    const agent = new AgentRuntime({
      modes,
      tools: new ToolRegistry().register({ name: 'echo', handle: (args: JsonObject) => { ran.push(Number(args.n)); return 'ok'; } }).register({ name: 'free', handle: () => 'free' }),
      client: async (): Promise<LlmResponse> => {
        model += 1;
        if (model === 1) return { text: '', finishReason: 'tool_calls', toolCalls: [echo('c1', 1)] };
        if (model <= 3) throw new Error('The provider send failed (scripted).');
        return { text: 'done', finishReason: 'stop' };
      },
    });
    const [only] = (await agent.send(session, 'go')).pendingApprovals;
    await recordApproval(session, only!.id, true);
    await expect(agent.send(session, '')).rejects.toThrow('scripted');
    await expect(agent.send(session, '')).rejects.toThrow('scripted');
    await agent.send(session, '');

    expect(ran).toEqual([1]);
  });
});
