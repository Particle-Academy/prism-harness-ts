import { mkdtemp, readdir, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  FileSessionStore,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  ToolRegistry,
  recordApproval,
  type JsonObject,
  type LlmRequest,
  type LlmResponse,
  type LlmToolCall,
  type Session,
} from '../src/index.js';

// What the resume reads about a stored call comes from the ONE turn it is
// resuming. Providers that number calls per response reuse ids, so an older
// turn must never decide a newer call with the same id.

const modes = new ModeRegistry({
  default: 'guarded',
  modes: { guarded: { system_prompt: 'Careful.', tools: ['echo'], max_steps: 4, requires_approval: ['echo'] } },
});

beforeEach(() => {
  vi.stubGlobal('fetch', () => { throw new Error('network blocked in tests'); });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

async function rig(call: LlmToolCall): Promise<{ session: Session; agent: AgentRuntime; directory: string; ran: () => number }> {
  const directory = await mkdtemp(join(tmpdir(), 'prism-harness-stored-scope-'));
  const harness = new PrismHarness({
    drivers: { memory: () => new MemorySessionStore(), files: () => new FileSessionStore(directory) },
    stores: { ephemeral: 'memory', durable: 'files' },
  });
  const session = harness.for({ type: 'User', id: 1 }).session('support');
  await session.usingProvider('anthropic');
  await session.usingModel('claude-sonnet-4-5');
  let runs = 0;
  let turn = 0;
  const tools = new ToolRegistry().register({ name: 'echo', handle: () => { runs += 1; return 'echoed'; } });
  const client = async (_request: LlmRequest): Promise<LlmResponse> => {
    turn += 1;

    return turn === 1 ? { text: '', finishReason: 'tool_calls', toolCalls: [call] } : { text: 'done', finishReason: 'stop' };
  };

  return { session, agent: new AgentRuntime({ client, modes, tools }), directory, ran: () => runs };
}

async function pendingId(session: Session): Promise<string> {
  const rows = (await session.thread().messages()).map((entry) => entry.message);
  const asked = [...rows].reverse().find((row) => row.type === 'assistant' && Array.isArray(row.tool_approval_requests) && row.tool_approval_requests.length > 0)!;

  return ((asked.tool_approval_requests as JsonObject[])[0]!).approval_id as string;
}

describe('the stored-call check reads only the turn being resumed', () => {
  it('runs an approved call whose id an OLDER turn used for arguments that were not an object', async () => {
    const { session, agent, ran } = await rig({ id: 'c1', name: 'echo', arguments: { value: 'x' } });
    // An earlier, finished turn: call c1 stored with arguments that are a list.
    await session.thread().record([
      { type: 'user', content: 'earlier' },
      { type: 'assistant', content: '', tool_calls: [{ id: 'c1', name: 'echo', arguments: [1, 2] as unknown as JsonObject, result_id: null, reasoning_id: null, reasoning_summary: null }], additional_content: {}, tool_approval_requests: [] },
      { type: 'tool_result', tool_results: [{ tool_call_id: 'c1', tool_name: 'echo', args: {}, result: 'Not run.', tool_call_result_id: null, artifacts: [] }], tool_approval_responses: [] },
      { type: 'assistant', content: 'ok', tool_calls: [], additional_content: {}, tool_approval_requests: [] },
    ]);

    await agent.send(session, 'go');
    await recordApproval(session, await pendingId(session), true);
    await agent.send(session, '');

    expect(ran()).toBe(1);
  });
});

describe('the arguments_format marker as stored', () => {
  it('reads a stored 1.0 as no marker, as the PHP and Python harnesses do', async () => {
    const { session, agent, directory, ran } = await rig({ id: 'c1', name: 'echo', arguments: {} });
    await agent.send(session, 'go');
    const id = await pendingId(session);

    // Edit the stored request by hand: the same number, written as 1.0.
    let edited = 0;
    for (const name of await readdir(directory, { recursive: true })) {
      const path = join(directory, name);
      const text = await readFile(path, 'utf8').catch(() => null);
      if (text === null || !text.includes('"arguments_format":1')) continue;
      await writeFile(path, text.replaceAll('"arguments_format":1', '"arguments_format":1.0'));
      edited += 1;
    }
    expect(edited).toBe(1);

    await recordApproval(session, id, true);
    await agent.send(session, '');

    // Unmarked, with no arguments: the approval expires instead of running.
    expect(ran()).toBe(0);
  });
});
