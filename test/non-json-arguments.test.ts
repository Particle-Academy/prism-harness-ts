import {mkdtemp} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';
import {AgentRuntime, FileSessionStore, HarnessEvents, MemorySessionStore, ModeRegistry, PrismHarness, ToolRegistry,
  type HarnessEvent, type JsonObject, type LlmRequest, type LlmResponse} from '../src/index.js';

/**
 * A host client maps a provider response into JS objects, and can leave values
 * in the arguments that JSON has no form for. Every decision about a gated call
 * carries the digest of its arguments, so arguments that cannot be digested are
 * unusable: the call is answered as unusable, asks nobody and runs nothing.
 */
async function run(args: unknown) {
  const directory = await mkdtemp(join(tmpdir(), 'prism-non-json-'));
  const harness = new PrismHarness({drivers: {memory: () => new MemorySessionStore(), files: () => new FileSessionStore(directory)}, stores: {ephemeral: 'memory', durable: 'files'}});
  const session = harness.for({type: 'User', id: 1}).session('non-json');
  await session.usingMode('chat'); await session.usingProvider('synthetic'); await session.usingModel('offline');

  const received: unknown[] = [];
  const requests: LlmRequest[] = [];
  const emitted: HarnessEvent[] = [];
  const events = new HarnessEvents();
  events.listen((event) => emitted.push(event));
  const responses: LlmResponse[] = [
    {text: '', finishReason: 'tool_calls', toolCalls: [{id: 'call-1', name: 'lookup', arguments: args as JsonObject}]},
    {text: 'done', finishReason: 'stop'},
  ];
  const tools = new ToolRegistry();
  tools.register({name: 'lookup', handle: (input: unknown) => {received.push(input); return 'ran';}});
  const runtime = new AgentRuntime({
    client: async (request) => {
      requests.push(request);
      const response = responses.shift();
      if (!response) throw new Error('Unexpected extra model call');
      return response;
    },
    tools,
    events,
    modes: new ModeRegistry({default: 'chat', modes: {chat: {tools: ['lookup'], max_steps: 4, requires_approval: ['lookup']}}}),
  });

  const response = await runtime.send(session, 'Look it up.');

  return {received, requests, response, emitted};
}

describe('tool arguments JSON has no form for', () => {
  const network = vi.fn(() => { throw new Error('A test tried to use the network.'); });

  beforeEach(() => {
    network.mockClear();
    vi.stubGlobal('fetch', network);
  });

  afterEach(() => {
    expect(network).not.toHaveBeenCalled();
    vi.unstubAllGlobals();
  });

  it.each([
    ['undefined in a list', {a: [undefined]}],
    ['a function', {a: () => 1}],
    ['a nested function', {a: {b: [() => 1]}}],
    ['a symbol', {a: Symbol('s')}],
    ['a bigint', {a: 1n}],
    ['a Date', {a: new Date(0)}],
    ['a Map', {a: new Map([['k', 'v']])}],
  ])('answer %s as unusable: send() completes, nobody is asked, nothing runs', async (_name, args) => {
    const {received, requests, response, emitted} = await run(args);

    expect(response.finishReason).toBe('stop');
    expect(response.pendingApprovals).toEqual([]);
    expect(received).toEqual([]);
    expect(emitted.filter((event) => event.type === 'tool_approval_asked')).toEqual([]);
    expect(JSON.stringify(requests[1]!.messages)).toContain('Not run: the arguments for lookup are not a JSON object.');
  });

  it('still asks about a call whose only extra is an absent (undefined) property', async () => {
    const {received, response, emitted} = await run({city: 'Paris', unit: undefined});

    expect(response.finishReason).toBe('awaiting_approval');
    expect(response.pendingApprovals).toHaveLength(1);
    expect(received).toEqual([]);
    expect(emitted.filter((event) => event.type === 'tool_approval_asked')).toHaveLength(1);
  });
});
