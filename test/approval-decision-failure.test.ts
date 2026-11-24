import {mkdtemp} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';
import {AgentRuntime, FileSessionStore, HarnessEvents, MemorySessionStore, ModeRegistry, PrismHarness, ToolAuthorizer,
  ToolRegistry, recordApproval, type LlmResponse} from '../src/index.js';

/**
 * Whether a call may run is decided before it runs. When that decision cannot
 * be made, because the check itself fails, the call does not run.
 */
async function world(options: {authorizer?: ToolAuthorizer; events?: HarnessEvents; requiresApproval?: string[]}) {
  const directory = await mkdtemp(join(tmpdir(), 'prism-decision-failure-'));
  const harness = new PrismHarness({drivers: {memory: () => new MemorySessionStore(), files: () => new FileSessionStore(directory)}, stores: {ephemeral: 'memory', durable: 'files'}});
  const session = harness.for({type: 'User', id: 1}).session('decision');
  await session.usingMode('chat'); await session.usingProvider('synthetic'); await session.usingModel('offline');

  const ran: unknown[] = [];
  const responses: LlmResponse[] = [
    {text: '', finishReason: 'tool_calls', toolCalls: [{id: 'call-1', name: 'lookup', arguments: {city: 'Paris'}}]},
    {text: 'done', finishReason: 'stop'},
  ];
  const tools = new ToolRegistry();
  tools.register({name: 'lookup', handle: (input: unknown) => {ran.push(input); return 'ran';}});
  const runtime = new AgentRuntime({
    client: async () => responses.shift() ?? {text: 'done', finishReason: 'stop'},
    tools,
    ...(options.authorizer ? {authorizer: options.authorizer} : {}),
    ...(options.events ? {events: options.events} : {}),
    modes: new ModeRegistry({default: 'chat', modes: {chat: {tools: ['lookup'], max_steps: 4, requires_approval: options.requiresApproval ?? []}}}),
  });

  return {runtime, session, ran};
}

describe('a failing decision never runs the call', () => {
  const network = vi.fn(() => { throw new Error('A test tried to use the network.'); });

  beforeEach(() => {
    network.mockClear();
    vi.stubGlobal('fetch', network);
  });

  afterEach(() => {
    expect(network).not.toHaveBeenCalled();
    vi.unstubAllGlobals();
  });

  it('stops the run when the call policy throws', async () => {
    const authorizer = new ToolAuthorizer({enabled: true, call: () => { throw new Error('check failed'); }});
    const w = await world({authorizer});

    await w.runtime.send(w.session, 'go').catch((error: unknown) => error);
    expect(w.ran).toEqual([]);
  });

  it('stops an approved call when the call policy throws at resume', async () => {
    let failing = false;
    const authorizer = new ToolAuthorizer({enabled: true, call: () => { if (failing) throw new Error('check failed'); return true; }});
    const w = await world({authorizer, requiresApproval: ['lookup']});

    const paused = await w.runtime.send(w.session, 'go');
    expect(paused.finishReason).toBe('awaiting_approval');
    await recordApproval(w.session, paused.pendingApprovals[0]!.id, true);
    failing = true;

    await w.runtime.send(w.session, '').catch((error: unknown) => error);
    expect(w.ran).toEqual([]);
  });

  it('still pauses for a person when an event listener throws', async () => {
    const events = new HarnessEvents();
    events.listen(() => { throw new Error('listener failed'); });
    const warning = vi.spyOn(process, 'emitWarning').mockImplementation(() => undefined);
    const w = await world({events, requiresApproval: ['lookup']});
    try {
      const response = await w.runtime.send(w.session, 'go');
      expect(response.finishReason).toBe('awaiting_approval');
    } finally {
      warning.mockRestore();
    }

    expect(w.ran).toEqual([]);
  });
});
