import {readFileSync} from 'node:fs';
import {mkdtemp} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';
import {AgentRuntime, FileSessionStore, HarnessEvents, MemorySessionStore, ModeRegistry, PrismHarness, recordApproval, ToolRegistry,
  type HarnessEvent, type JsonObject, type LlmRequest, type LlmResponse} from '../src/index.js';

/*
 * 0.11.0 stored a gated call whose arguments arrived as a bare number or
 * boolean as `{}`, the same as a real call with no arguments. An approval
 * without a marker on such a call is expired at resume: the call does not
 * run, and the model is told to call again.
 *
 * The rows in fixtures/legacy-approvals-0.11.0.json were written by the
 * released @particle-academy/prism-harness 0.11.0, through its public API.
 */

const legacy = JSON.parse(readFileSync(new URL('./fixtures/legacy-approvals-0.11.0.json', import.meta.url), 'utf8')) as
  Record<string, {approval_id: string; rows: JsonObject[]}>;

const EXPIRED = 'has expired';

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn(() => { throw new Error('A test tried to use the network.'); }));
});

afterEach(() => {
  vi.unstubAllGlobals();
});

async function session() {
  const directory = await mkdtemp(join(tmpdir(), 'prism-g351-'));
  const harness = new PrismHarness({drivers: {memory: () => new MemorySessionStore(), files: () => new FileSessionStore(directory)}, stores: {ephemeral: 'memory', durable: 'files'}});
  const live = harness.for({type: 'User', id: 1}).session('stored');
  await live.usingMode('chat'); await live.usingProvider('synthetic'); await live.usingModel('offline');
  return live;
}

function runtime(responses: LlmResponse[]) {
  const received: unknown[] = [];
  const requests: LlmRequest[] = [];
  const seen: HarnessEvent[] = [];
  const events = new HarnessEvents();
  events.listen(event => seen.push(event));
  const tools = new ToolRegistry();
  tools.register({name: 'lookup', handle: (input: unknown) => {received.push(input); return 'ran';}});
  const agent = new AgentRuntime({
    client: async (request) => {
      requests.push(request);
      return responses.shift() ?? {text: 'done', finishReason: 'stop'};
    },
    tools,
    events,
    modes: new ModeRegistry({default: 'chat', modes: {chat: {tools: ['lookup'], max_steps: 4, requires_approval: ['lookup']}}}),
  });

  return {agent, received, requests, seen};
}

async function resume(rows: JsonObject[], approvalId: string, approved = true, responses: LlmResponse[] = []) {
  const live = await session();
  await live.thread().record(rows);
  const run = runtime(responses);
  await recordApproval(live, approvalId, approved);
  const response = await run.agent.send(live, '');
  const stored = (await live.thread().messages()).map(entry => entry.message);

  return {...run, response, stored};
}

function storedRow(args: unknown, request: JsonObject): JsonObject[] {
  return [
    {type: 'user', content: 'Look it up.'},
    {
      type: 'assistant',
      content: '',
      tool_calls: [{id: 'call-1', name: 'lookup', arguments: args as JsonObject, result_id: null, reasoning_id: null, reasoning_summary: null}],
      additional_content: {},
      tool_approval_requests: [{approval_id: 'apr-1', tool_call_id: 'call-1', ...request}],
      harness_tool_names: ['lookup'],
    },
  ];
}

describe('an approval stored without a marker on a call stored as no arguments', () => {
  it.each(['wire_number', 'wire_true', 'empty_object'])('expires the row 0.11.0 wrote for %s instead of running it', async (form) => {
    const {approval_id, rows} = legacy[form]!;
    const {received, requests, response, stored, seen} = await resume(rows, approval_id);

    expect(received).toEqual([]);
    expect(response.finishReason).toBe('stop');
    expect(JSON.stringify(requests[0]!.messages)).toContain(EXPIRED);
    expect(JSON.stringify(requests[0]!.messages)).toContain('Call the tool again');

    const results = stored.filter(row => row.type === 'tool_result').flatMap(row => (row.tool_results as JsonObject[]) ?? []);
    expect(results).toHaveLength(1);
    expect(results[0]!.args).toEqual({});
    expect(String(results[0]!.result)).toContain(EXPIRED);

    expect(seen.map(event => event.type)).not.toContain('tool_approved');
    const expired = seen.filter(event => event.type === 'tool_approval_expired');
    expect(expired).toHaveLength(1);
    expect(Object.keys(expired[0]!)).not.toContain('arguments');
    expect(expired[0]).toMatchObject({tool: 'lookup', decidedBy: 'policy'});
  });

  it('asks the person again when the model calls the tool again', async () => {
    const {approval_id, rows} = legacy.wire_number!;
    const {received, response, stored} = await resume(rows, approval_id, true, [
      {text: '', finishReason: 'tool_calls', toolCalls: [{id: 'call-2', name: 'lookup', arguments: {}}]},
    ]);

    expect(received).toEqual([]);
    expect(response.finishReason).toBe('awaiting_approval');
    expect(response.pendingApprovals).toHaveLength(1);

    const assistant = stored.filter(row => row.type === 'assistant').at(-1)!;
    expect(assistant.tool_approval_requests).toEqual([expect.objectContaining({tool_call_id: 'call-2', arguments_format: 1})]);
  });

  it.each([
    ['a string', '1'],
    ['zero', 0],
    ['a future version', 99],
    ['true', true],
    ['null', null],
  ])('treats a marker it does not recognise (%s) as none', async (_name, format) => {
    const {received, requests} = await resume(storedRow({}, {arguments_format: format}), 'apr-1');

    expect(received).toEqual([]);
    expect(JSON.stringify(requests[0]!.messages)).toContain(EXPIRED);
  });

  it('leaves a denied one denied', async () => {
    const {approval_id, rows} = legacy.wire_true!;
    const {received, requests, seen} = await resume(rows, approval_id, false);

    expect(received).toEqual([]);
    expect(JSON.stringify(requests[0]!.messages)).not.toContain(EXPIRED);
    expect(seen.map(event => event.type)).toContain('tool_denied');
  });
});

describe('approvals that are not ambiguous', () => {
  it('runs a marked approval on a call with no arguments, issued and resumed here', async () => {
    const live = await session();
    const first = runtime([{text: '', finishReason: 'tool_calls', toolCalls: [{id: 'call-1', name: 'lookup', arguments: {}}]}]);
    const paused = await first.agent.send(live, 'Look it up.');
    const assistant = (await live.thread().messages()).map(entry => entry.message).find(row => row.type === 'assistant')!;

    expect(assistant.tool_approval_requests).toEqual([expect.objectContaining({arguments_format: 1})]);

    const second = runtime([]);
    await recordApproval(live, paused.pendingApprovals[0]!.id, true);
    await second.agent.send(live, '');

    expect(second.received).toEqual([{}]);
    expect(second.seen.map(event => event.type)).toContain('tool_approved');
  });

  it('runs an unmarked approval on a call with named arguments', async () => {
    const {received} = await resume(storedRow({city: 'Paris'}, {}), 'apr-1');

    expect(received).toEqual([{city: 'Paris'}]);
  });
});
