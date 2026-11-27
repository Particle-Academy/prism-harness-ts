import { describe, expect, it } from 'vitest';
import {
  AgentRuntime,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  ToolRegistry,
  type JsonObject,
  type LlmRequest,
  type LlmResponse,
} from '../src/index.js';

// G-415: an assistant row a host records with Thread.record that holds NO tool
// calls, but carries provider-native content copied from a stored step, must
// not send a tool call to the model again. A client built on prism-ts replays
// `additional_content.output_items` (OpenAI Responses) and
// `additional_content.content_blocks` (Anthropic) from an assistant message
// as given, so a tool item there would reach the provider with no result. The
// request the model is sent carries such a row without its tool items; the
// stored row is unchanged, and the harness's own rows are untouched. Checked
// at the request the client receives. Offline.

class Durable extends MemorySessionStore {
  override durability(): 'durable' { return 'durable'; }
}

const OUTPUT_ITEMS = [
  { type: 'reasoning', id: 'rs_1', summary: [] },
  { type: 'function_call', id: 'fc_1', call_id: 'call_1', name: 'lookup', arguments: '{"q":"x"}' },
  { type: 'function_call_output', call_id: 'call_1', output: 'found' },
  { type: 'message', role: 'assistant', content: [{ type: 'output_text', text: 'Here it is.' }] },
];

const CONTENT_BLOCKS = [
  { type: 'text', text: 'Looking it up.' },
  { type: 'tool_use', id: 'toolu_1', name: 'lookup', input: { q: 'x' } },
  { type: 'fallback', reason: 'copied' },
  { type: 'tool_use', id: 'toolu_2', name: 'lookup', input: { q: 'y' } },
  { type: 'text', text: 'Here it is.' },
];

function rig() {
  const harness = new PrismHarness({ drivers: { m: () => new MemorySessionStore(), d: () => new Durable() }, stores: { ephemeral: 'm', durable: 'd' } });
  const session = harness.for({ type: 'User', id: 1 }).session('host-content');
  const requests: LlmRequest[] = [];
  const agent = new AgentRuntime({
    modes: new ModeRegistry({ default: 'm', modes: { m: { system_prompt: 'S.', tools: ['lookup'], max_steps: 4 } } }),
    tools: new ToolRegistry().register({ name: 'lookup', handle: () => 'found' }),
    client: async (request: LlmRequest): Promise<LlmResponse> => {
      requests.push(JSON.parse(JSON.stringify(request)) as LlmRequest);
      return { text: 'ok', finishReason: 'stop' };
    },
  });
  return { session, agent, requests };
}

function hostRow(additional: JsonObject, toolCalls: unknown = []): JsonObject {
  return { type: 'assistant', content: 'Here it is.', tool_calls: toolCalls as JsonObject[], additional_content: additional, tool_approval_requests: [] };
}

/** The sent assistant row whose content is the host's. */
function sentHostRow(request: LlmRequest): JsonObject {
  return (request.messages as JsonObject[]).find((row) => row.type === 'assistant' && row.content === 'Here it is.')!;
}

describe('G-415: a host assistant row with no tool calls sends no tool item', () => {
  it('OpenAI output_items: function_call and function_call_output are left out of the request; reasoning and the message are kept', async () => {
    const r = rig();
    await r.session.thread().record([{ type: 'user', content: 'q' }, hostRow({ output_items: OUTPUT_ITEMS, keep: 'me' })]);
    await r.agent.send(r.session, 'next');

    const sent = sentHostRow(r.requests.at(-1)!);
    expect((sent.additional_content as JsonObject).output_items).toEqual([OUTPUT_ITEMS[0], OUTPUT_ITEMS[3]]);
    expect((sent.additional_content as JsonObject).keep).toBe('me');
    // Stored as recorded.
    const stored = (await r.session.thread().messages()).map((entry) => entry.message).find((row) => row.type === 'assistant')!;
    expect((stored.additional_content as JsonObject).output_items).toEqual(OUTPUT_ITEMS);
  });

  it('Anthropic content_blocks: every tool_use is left out of the request, including one after a fallback block', async () => {
    const r = rig();
    await r.session.thread().record([{ type: 'user', content: 'q' }, hostRow({ content_blocks: CONTENT_BLOCKS })]);
    await r.agent.send(r.session, 'next');

    const sent = sentHostRow(r.requests.at(-1)!);
    expect((sent.additional_content as JsonObject).content_blocks).toEqual([CONTENT_BLOCKS[0], CONTENT_BLOCKS[2], CONTENT_BLOCKS[4]]);
  });

  it('a host row whose tool_calls hold no objects counts as one with none: its tool items are left out too', async () => {
    const r = rig();
    await r.session.thread().record([{ type: 'user', content: 'q' }, hostRow({ output_items: OUTPUT_ITEMS, content_blocks: CONTENT_BLOCKS }, ['{"id":"call_1"}'])]);
    await r.agent.send(r.session, 'next');

    const sent = sentHostRow(r.requests.at(-1)!);
    expect(JSON.stringify(sent.additional_content)).not.toContain('function_call');
    expect(JSON.stringify(sent.additional_content)).not.toContain('tool_use');
  });

  it('the harness\'s own rows are sent as recorded: a model step with tool calls keeps its function_call item', async () => {
    const harness = new PrismHarness({ drivers: { m: () => new MemorySessionStore(), d: () => new Durable() }, stores: { ephemeral: 'm', durable: 'd' } });
    const session = harness.for({ type: 'User', id: 1 }).session('own-rows');
    const requests: LlmRequest[] = [];
    let model = 0;
    const agent = new AgentRuntime({
      modes: new ModeRegistry({ default: 'm', modes: { m: { system_prompt: 'S.', tools: ['lookup'], max_steps: 4 } } }),
      tools: new ToolRegistry().register({ name: 'lookup', handle: () => 'found' }),
      client: async (request: LlmRequest): Promise<LlmResponse> => {
        requests.push(JSON.parse(JSON.stringify(request)) as LlmRequest);
        return ++model === 1
          ? { text: '', finishReason: 'tool_calls', toolCalls: [{ id: 'call_1', name: 'lookup', arguments: { q: 'x' } }], additionalContent: { output_items: [OUTPUT_ITEMS[1]!] } }
          : { text: 'done', finishReason: 'stop' };
      },
    });
    await agent.send(session, 'go');

    const step = (requests.at(-1)!.messages as JsonObject[]).find((row) => row.type === 'assistant' && Array.isArray(row.tool_calls) && row.tool_calls.length > 0)!;
    expect((step.additional_content as JsonObject).output_items).toEqual([OUTPUT_ITEMS[1]]);
  });
});
