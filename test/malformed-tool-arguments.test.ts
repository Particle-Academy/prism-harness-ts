import {mkdtemp} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';
import {AgentRuntime, FileSessionStore, MemorySessionStore, ModeRegistry, PrismHarness, recordApproval, ToolRegistry,
  type JsonObject, type LlmRequest, type LlmResponse} from '../src/index.js';

async function run(args: unknown, requiresApproval: string[] = []) {
  const directory = await mkdtemp(join(tmpdir(), 'prism-malformed-'));
  const harness = new PrismHarness({drivers: {memory: () => new MemorySessionStore(), files: () => new FileSessionStore(directory)}, stores: {ephemeral: 'memory', durable: 'files'}});
  const session = harness.for({type: 'User', id: 1}).session('malformed');
  await session.usingMode('chat'); await session.usingProvider('synthetic'); await session.usingModel('offline');

  const received: unknown[] = [];
  const requests: LlmRequest[] = [];
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
    modes: new ModeRegistry({default: 'chat', modes: {chat: {tools: ['lookup'], max_steps: 4, requires_approval: requiresApproval}}}),
  });

  const response = await runtime.send(session, 'Look it up.');
  const rows = (await session.thread().messages()).map(entry => entry.message);

  return {received, requests, response, rows};
}

describe('a tool call whose arguments are not an object', () => {
  it.each([
    ['unparsed text', '{"city": '],
    ['a list', [1, 2]],
    ['a number', 5],
  ])('reports %s as a tool error without running the tool', async (_name, args) => {
    const {received, requests, response, rows} = await run(args);

    expect(received).toEqual([]);
    expect(response.finishReason).toBe('stop');
    expect(requests).toHaveLength(2);
    expect(JSON.stringify(requests[1]!.messages)).toContain('Not run: the arguments for lookup are not a JSON object.');

    const assistant = rows.find(row => row.type === 'assistant') as {tool_calls: {arguments: unknown}[]};
    expect(assistant.tool_calls[0]!.arguments).toEqual({});
  });

  it('does not ask a person to approve it', async () => {
    const {received, response} = await run('{"city": ', ['lookup']);

    expect(response.pendingApprovals).toEqual([]);
    expect(response.finishReason).toBe('stop');
    expect(received).toEqual([]);
  });

  it('runs a well-formed call as before', async () => {
    const {received, response} = await run({city: 'Paris'});

    expect(received).toEqual([{city: 'Paris'}]);
    expect(response.finishReason).toBe('stop');
  });
});

describe('arguments as the loop reads them', () => {
  const realRequests = vi.fn(() => {
    throw new Error('A test tried to use the network.');
  });

  beforeEach(() => {
    realRequests.mockClear();
    vi.stubGlobal('fetch', realRequests);
  });

  afterEach(() => {
    expect(realRequests).not.toHaveBeenCalled();
    vi.unstubAllGlobals();
  });

  it.each([
    ['an empty string', ''],
    ['the text 0', '0'],
    ['the text null', 'null'],
    ['the text of an empty list', '[]'],
    ['the text of an empty object', '{}'],
    ['null', null],
    ['an empty list', []],
    ['an empty object', {}],
  ])('runs %s as no arguments', async (_name, args) => {
    const {received, requests} = await run(args);

    expect(requests).toHaveLength(2);
    expect(received).toEqual([{}]);
  });

  it('reads JSON text naming values as those values', async () => {
    const {received} = await run('{"city":"Paris"}');

    expect(received).toEqual([{city: 'Paris'}]);
  });

  it.each([
    ['an integer key', {0: 'x'}],
    ['an integer key in text', '{"0":"x"}'],
    ['a number past the float range', '{"amount":1e999}'],
    ['an integer past the float range', `{"amount":1${'0'.repeat(400)}}`],
    ['a key starting with NUL', '{"amount":"5","\\u0000a":1}'],
    ['a nested key starting with NUL', {a: {'\u0000b': 1}}],
    ['a control character inside a number', '{"amount":1\u000100}'],
    ['a control character inside a literal', 'nu\u0001ll'],
    ['true', true],
    ['zero', 0],
    ['false', false],
    ['501 levels of nesting', `${'{"a":'.repeat(501)}1${'}'.repeat(501)}`],
  ])('reports %s without running the tool', async (_name, args) => {
    const {received, requests, rows} = await run(args);

    expect(received).toEqual([]);
    expect(requests).toHaveLength(2);
    expect(JSON.stringify(requests[1]!.messages)).toContain('Not run: the arguments for lookup are not a JSON object.');

    const assistant = rows.find(row => row.type === 'assistant') as {tool_calls: {arguments: unknown}[]};
    expect(assistant.tool_calls[0]!.arguments).toEqual({});
  });

  it('keeps a control character inside a string and accepts a long digit key', async () => {
    const key = '7'.repeat(5000);
    const {received} = await run(`{"note":"a\u0001b","${key}":1}`);

    expect(received).toEqual([{note: 'a\u0001b', [key]: 1}]);
  });

  async function resumeStored(stored: unknown, approved = true) {
    const directory = await mkdtemp(join(tmpdir(), 'prism-stored-'));
    const harness = new PrismHarness({drivers: {memory: () => new MemorySessionStore(), files: () => new FileSessionStore(directory)}, stores: {ephemeral: 'memory', durable: 'files'}});
    const session = harness.for({type: 'User', id: 1}).session('stored');
    await session.usingMode('chat'); await session.usingProvider('synthetic'); await session.usingModel('offline');
    await session.thread().record([
      {type: 'user', content: 'Look it up.', additional_content: [], additional_attributes: {}},
      {
        type: 'assistant',
        content: '',
        tool_calls: [{id: 'call-1', name: 'lookup', arguments: stored as JsonObject, result_id: null, reasoning_id: null, reasoning_summary: null}],
        additional_content: {},
        tool_approval_requests: [{approval_id: 'apr-1', tool_call_id: 'call-1', arguments_format: 1}],
      },
    ]);

    const received: unknown[] = [];
    const requests: LlmRequest[] = [];
    const tools = new ToolRegistry();
    tools.register({name: 'lookup', handle: (input: unknown) => {received.push(input); return 'ran';}});
    const runtime = new AgentRuntime({
      client: async (request) => {
        requests.push(request);
        return {text: 'done', finishReason: 'stop'};
      },
      tools,
      modes: new ModeRegistry({default: 'chat', modes: {chat: {tools: ['lookup'], max_steps: 4, requires_approval: ['lookup']}}}),
    });

    await recordApproval(session, 'apr-1', approved);
    await runtime.send(session, '');

    return {received, requests};
  }

  it.each([
    ['text split into characters', Object.fromEntries([...'{"city":"Par'].map((char, index) => [String(index), char]))],
    ['unparsed text', '{"city":"Par'],
    ['a number', 5],
    ['true', true],
    ['a list', [1, 2]],
    ['an integer key', {0: 'x'}],
  ])('refuses an approved stored call holding %s', async (_name, stored) => {
    const {received, requests} = await resumeStored(stored);

    expect(received).toEqual([]);
    expect(JSON.stringify(requests[0]!.messages)).toContain('Not run: the arguments for lookup are not a JSON object.');
  });

  it.each([
    ['named values', {city: 'Paris'}, {city: 'Paris'}],
    ['none', {}, {}],
    ['null', null, {}],
  ])('still runs an approved stored call holding %s', async (_name, stored, expected) => {
    const {received} = await resumeStored(stored);

    expect(received).toEqual([expected]);
  });
});

describe('argument keys that could bind by position', () => {
  it.each(['{"0":"prod"}', '["prod"]', '{"0":"prod","note":"x"}', '{"note":"x","1":"prod"}', '{"-1":"prod"}'])('runs nothing for %s', async (args) => {
    const network = vi.fn(() => { throw new Error('A test tried to use the network.'); });
    vi.stubGlobal('fetch', network);
    const {received, response} = await run(args, ['lookup']);

    expect(received).toEqual([]);
    expect(response.pendingApprovals).toEqual([]);
    expect(network).not.toHaveBeenCalled();
    vi.unstubAllGlobals();
  });

  it('runs nothing for a mixed map handed over as an object', async () => {
    const {received} = await run({0: 'prod', note: 'x'});

    expect(received).toEqual([]);
  });

  it.each(['{"00":"prod"}', '{"1.0":"prod"}', '{"-0":"prod"}', '{"+1":"prod"}', '{" 1":"prod"}'])('hands %s over under its own name', async (args) => {
    const {received} = await run(args);

    expect(received).toEqual([JSON.parse(args)]);
  });
});
