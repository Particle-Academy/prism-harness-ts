import {mkdtemp} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {describe, expect, it} from 'vitest';
import {AgentRuntime, FileSessionStore, MemorySessionStore, ModeRegistry, PrismHarness, ToolRegistry,
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
