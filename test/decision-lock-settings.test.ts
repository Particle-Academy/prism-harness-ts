import {mkdtemp} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';
import {AgentRuntime, FileSessionStore, MemorySessionStore, ModeRegistry, PrismHarness, ToolRegistry,
  type LlmResponse, type RuntimeConfig} from '../src/index.js';

/**
 * Recording a decision takes the session lock. Through the runtime it waits as
 * long as the host's `lock_wait` says, as a streamed decision does, not a fixed
 * thirty seconds.
 */
async function parked(config: RuntimeConfig) {
  const directory = await mkdtemp(join(tmpdir(), 'prism-decision-lock-'));
  const harness = new PrismHarness({drivers: {memory: () => new MemorySessionStore(), files: () => new FileSessionStore(directory)}, stores: {ephemeral: 'memory', durable: 'files'}});
  const session = harness.for({type: 'User', id: 1}).session('decision-lock');
  await session.usingMode('chat'); await session.usingProvider('synthetic'); await session.usingModel('offline');

  const ran: unknown[] = [];
  const responses: LlmResponse[] = [
    {text: '', finishReason: 'tool_calls', toolCalls: [{id: 'call-1', name: 'lookup', arguments: {city: 'Paris'}}]},
    {text: 'done', finishReason: 'stop'},
  ];
  const tools = new ToolRegistry();
  tools.register({name: 'lookup', handle: (input: unknown) => {ran.push(input); return 'ran';}});
  const runtime = new AgentRuntime({
    client: async () => {
      const response = responses.shift();
      if (!response) throw new Error('Unexpected extra model call');
      return response;
    },
    tools,
    config,
    modes: new ModeRegistry({default: 'chat', modes: {chat: {tools: ['lookup'], max_steps: 4, requires_approval: ['lookup']}}}),
  });

  const response = await runtime.send(session, 'Look it up.');
  const holder = harness.for({type: 'User', id: 1}).session('decision-lock');

  return {session, holder, runtime, ran, approvalId: response.pendingApprovals[0]!.id};
}

describe('a decision recorded through the runtime', () => {
  const network = vi.fn(() => { throw new Error('A test tried to use the network.'); });

  beforeEach(() => {
    network.mockClear();
    vi.stubGlobal('fetch', network);
  });

  afterEach(() => {
    expect(network).not.toHaveBeenCalled();
    vi.unstubAllGlobals();
  });

  it('honours lock_wait: refused at once with session_locked while another holder has the lock, recording nothing', async () => {
    const {session, holder, runtime, ran, approvalId} = await parked({lock_wait: 0});
    const before = await session.thread().count();
    const started = Date.now();

    await holder.lock(async () => {
      await expect(runtime.decide(session, approvalId, true)).rejects.toMatchObject({code: 'session_locked'});
    }, 30, 0);

    expect(Date.now() - started).toBeLessThan(5000);
    expect(await session.thread().count()).toBe(before);
    expect(ran).toEqual([]);
  });

  it('records the decision once the lock is free, and the next send() acts on it', async () => {
    const {session, runtime, ran, approvalId} = await parked({lock_wait: 0});

    await runtime.decide(session, approvalId, true);
    await expect(runtime.decide(session, approvalId, false)).rejects.toMatchObject({code: 'approval_not_pending'});

    const resumed = await runtime.send(session, 'Go on.');
    expect(resumed.finishReason).toBe('stop');
    expect(ran).toEqual([{city: 'Paris'}]);
  });
});
