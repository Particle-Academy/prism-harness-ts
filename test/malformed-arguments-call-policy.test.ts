import {mkdtemp} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';
import {AgentRuntime, FileSessionStore, MemorySessionStore, ModeRegistry, PrismHarness, ToolAuthorizer, ToolRegistry,
  type JsonObject, type LlmResponse} from '../src/index.js';

/**
 * A host call policy is asked with the arguments of each call before it runs.
 * Arguments that are not a usable object must never reach it as if they were
 * real; arguments that mean "none" reach it as an empty object.
 */
async function run(args: unknown) {
  const directory = await mkdtemp(join(tmpdir(), 'prism-call-policy-'));
  const harness = new PrismHarness({drivers: {memory: () => new MemorySessionStore(), files: () => new FileSessionStore(directory)}, stores: {ephemeral: 'memory', durable: 'files'}});
  const session = harness.for({type: 'User', id: 1}).session('policy');
  await session.usingMode('chat'); await session.usingProvider('synthetic'); await session.usingModel('offline');

  const asked: unknown[] = [];
  const received: unknown[] = [];
  const responses: LlmResponse[] = [
    {text: '', finishReason: 'tool_calls', toolCalls: [{id: 'call-1', name: 'lookup', arguments: args as JsonObject}]},
    {text: 'done', finishReason: 'stop'},
  ];
  const tools = new ToolRegistry();
  tools.register({name: 'lookup', handle: (input: unknown) => {received.push(input); return 'ran';}});
  const runtime = new AgentRuntime({
    client: async () => {
      const response = responses.shift();
      if (!response) throw new Error('Unexpected extra model call');
      return response;
    },
    tools,
    authorizer: new ToolAuthorizer({enabled: true, call: (_session, _tool, callArgs) => {asked.push(callArgs); return true;}}),
    modes: new ModeRegistry({default: 'chat', modes: {chat: {tools: ['lookup'], max_steps: 4}}}),
  });

  const response = await runtime.send(session, 'Look it up.');

  return {asked, received, response};
}

describe('a call policy and tool arguments', () => {
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
    ['unparsed text', '{"path": "/tmp/'],
    ['a list', ['/tmp']],
    ['a number', 5],
    ['an integer key', {0: '/tmp'}],
    ['a number past the float range', '{"amount":1e999}'],
  ])('is never asked about %s, and the tool does not run', async (_name, args) => {
    const {asked, received, response} = await run(args);

    expect(asked).toEqual([]);
    expect(received).toEqual([]);
    expect(response.finishReason).toBe('stop');
  });

  it.each([
    ['an empty string', ''],
    ['null', null],
    ['an empty object', {}],
  ])('is asked about %s as no arguments, and the tool runs', async (_name, args) => {
    const {asked, received} = await run(args);

    expect(asked).toEqual([{}]);
    expect(received).toEqual([{}]);
  });

  it('is asked about JSON text as the values it names', async () => {
    const {asked, received} = await run('{"path":"/tmp"}');

    expect(asked).toEqual([{path: '/tmp'}]);
    expect(received).toEqual([{path: '/tmp'}]);
  });
});
