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

// G-415: a client built on prism-ts replays an assistant message's
// provider-native content as given: `additional_content.output_items`
// (OpenAI Responses) and `additional_content.content_blocks` (Anthropic),
// even after a 'fallback' block. An assistant row a host records with
// Thread.record can carry content copied from another step. Sent as is, it
// can send a tool call with no result, drop the row's own text for the copied
// text, or send a reasoning item twice.
//
// The rule: a host row's provider content is sent ONLY when its tool items
// match the row's own tool_calls exactly (ids and names, as a multiset) AND
// its text matches the row's content. Otherwise it is dropped, and the client
// maps the row from content and tool_calls. Rows the harness writes from
// provider responses are sent byte for byte as recorded. The stored row never
// changes. Checked at the request the client receives. Offline.

class Durable extends MemorySessionStore {
  override durability(): 'durable' { return 'durable'; }
}

const NOTE = 'My note.';

function rig(responses: LlmResponse[] = []) {
  const harness = new PrismHarness({ drivers: { m: () => new MemorySessionStore(), d: () => new Durable() }, stores: { ephemeral: 'm', durable: 'd' } });
  const session = harness.for({ type: 'User', id: 1 }).session('host-content');
  const requests: LlmRequest[] = [];
  const agent = new AgentRuntime({
    modes: new ModeRegistry({ default: 'm', modes: { m: { system_prompt: 'S.', tools: ['lookup'], max_steps: 4 } } }),
    tools: new ToolRegistry().register({ name: 'lookup', handle: () => 'found' }),
    client: async (request: LlmRequest): Promise<LlmResponse> => {
      requests.push(JSON.parse(JSON.stringify(request)) as LlmRequest);
      return responses.shift() ?? { text: 'ok', finishReason: 'stop' };
    },
  });
  return { session, agent, requests };
}

function call(id: string, name = 'lookup'): JsonObject {
  return { id, name, arguments: { q: 'x' }, result_id: null, reasoning_id: null, reasoning_summary: null };
}

function hostRow(content: string, additional: JsonObject, toolCalls: unknown[] = []): JsonObject {
  return { type: 'assistant', content, tool_calls: toolCalls as JsonObject[], additional_content: additional, tool_approval_requests: [] };
}

/** Record `row` as a host would, after a user row, then send; return the sent copy of it and the stored one. */
async function sendWith(row: JsonObject): Promise<{ sent: JsonObject; stored: JsonObject }> {
  const r = rig();
  await r.session.thread().record([{ type: 'user', content: 'q' }, row]);
  // A host row with tool calls is a step: answer it, as a host would.
  const calls = (row.tool_calls as JsonObject[]).filter((entry) => typeof entry === 'object' && entry !== null);
  if (calls.length > 0) {
    await r.session.thread().record([{ type: 'tool_result', tool_approval_responses: [], tool_results: calls.map((entry) => ({ tool_call_id: entry.id, tool_name: entry.name, args: { q: 'x' }, result: 'found', tool_call_result_id: null, artifacts: [] })) }]);
  }
  await r.agent.send(r.session, 'next');
  const assistants = (rows: JsonObject[]) => rows.filter((entry) => entry.type === 'assistant' && entry.content === row.content);
  return {
    sent: assistants(r.requests.at(-1)!.messages as JsonObject[])[0]!,
    stored: assistants((await r.session.thread().messages()).map((entry) => entry.message))[0]!,
  };
}

const additional = (row: JsonObject): JsonObject => row.additional_content as JsonObject;

const openaiCall = (id: string) => ({ type: 'function_call', id, call_id: `c_${id}`, name: 'lookup', arguments: '{"q":"x"}' });
const message = (text: string) => ({ type: 'message', role: 'assistant', content: [{ type: 'output_text', text }] });
const toolUse = (id: string) => ({ type: 'tool_use', id, name: 'lookup', input: { q: 'x' } });

describe('G-415: a host row\'s provider content is sent only when it is that row\'s own', () => {
  it('OpenAI, a note with no tool calls carrying a copied function_call: output_items are dropped; content and other keys are kept; stored unchanged', async () => {
    const items = [{ type: 'reasoning', id: 'rs_1', summary: [] }, openaiCall('fc_1'), { type: 'function_call_output', call_id: 'c_fc_1', output: 'found' }, message(NOTE)];
    const { sent, stored } = await sendWith(hostRow(NOTE, { output_items: items, keep: 'me' }));
    expect(additional(sent)).toEqual({ keep: 'me' });
    expect(sent.content).toBe(NOTE);
    expect(additional(stored).output_items).toEqual(items);
  });

  it('Anthropic, a note with no tool calls carrying a copied tool_use: content_blocks are dropped', async () => {
    const { sent } = await sendWith(hostRow(NOTE, { content_blocks: [{ type: 'text', text: NOTE }, toolUse('toolu_1')] }));
    expect(additional(sent)).toEqual({});
  });

  it('A1 OpenAI: a host row WITH tool calls carrying another step\'s function_call: output_items are dropped', async () => {
    const { sent } = await sendWith(hostRow('', { output_items: [openaiCall('fc_other')] }, [call('fc_own')]));
    expect(additional(sent)).toEqual({});
    expect((sent.tool_calls as JsonObject[]).map((entry) => entry.id)).toEqual(['fc_own']);
  });

  it('A1 Anthropic: a host row WITH tool calls carrying another step\'s tool_use, also after a fallback block: content_blocks are dropped', async () => {
    for (const blocks of [[toolUse('toolu_other')], [{ type: 'fallback' }, toolUse('toolu_own'), toolUse('toolu_other')]]) {
      const { sent } = await sendWith(hostRow('', { content_blocks: blocks }, [call('toolu_own')]));
      expect(additional(sent)).toEqual({});
    }
  });

  it('A2 Anthropic: a note whose copied text block is not its own text: content_blocks are dropped, so the note\'s text is what is sent (with and without a fallback block)', async () => {
    for (const blocks of [[{ type: 'text', text: 'Copied text.' }], [{ type: 'fallback' }, { type: 'text', text: 'Copied text.' }]]) {
      const { sent } = await sendWith(hostRow(NOTE, { content_blocks: blocks }));
      expect(additional(sent)).toEqual({});
      expect(sent.content).toBe(NOTE);
    }
  });

  it('A3 OpenAI: a note carrying a copied reasoning item and no message: output_items are dropped', async () => {
    const { sent } = await sendWith(hostRow(NOTE, { output_items: [{ type: 'reasoning', id: 'rs_copied', summary: [] }] }));
    expect(additional(sent)).toEqual({});
  });

  it('a host row whose provider content IS its own (same calls, same text, no reasoning) is sent as recorded, on both providers', async () => {
    const own = { output_items: [message('Looking.'), openaiCall('fc_1')], content_blocks: [{ type: 'text', text: 'Looking.' }, toolUse('fc_1')] };
    // OpenAI's text is the first part of the last output item: here the call, so empty.
    const openai = await sendWith(hostRow('', { output_items: [message('Looking.'), openaiCall('fc_1')] }, [call('fc_1')]));
    expect(additional(openai.sent).output_items).toEqual(own.output_items);
    const anthropic = await sendWith(hostRow('Looking.', { content_blocks: own.content_blocks }, [call('fc_1')]));
    expect(additional(anthropic.sent).content_blocks).toEqual(own.content_blocks);
  });

  it('reasoning on a host row cannot be checked, so the row\'s provider content is dropped: OpenAI reasoning items, even when calls and text match', async () => {
    for (const items of [
      [{ type: 'reasoning', id: 'rs_copied', summary: [] }, message(NOTE)],
      [{ type: 'reasoning', id: 'rs_copied', summary: [{ type: 'summary_text', text: 'thought' }], encrypted_content: 'x' }, openaiCall('fc_1')],
    ]) {
      const calls = items.some((item) => item.type === 'function_call') ? [call('fc_1')] : [];
      const { sent, stored } = await sendWith(hostRow(calls.length > 0 ? '' : NOTE, { output_items: items, keep: 'me' }, calls));
      expect(additional(sent)).toEqual({ keep: 'me' });
      expect(additional(stored).output_items).toEqual(items);
    }
  });

  it('reasoning on a host row cannot be checked, so the row\'s provider content is dropped: Anthropic thinking, thinking_signature, thinking_blocks and thinking blocks', async () => {
    const thinking = { type: 'thinking', thinking: 'copied thought', signature: 'sig' };
    for (const extra of [
      { thinking: 'copied thought', thinking_signature: 'sig' },
      { thinking_blocks: [thinking] },
      { content_blocks: [thinking, { type: 'text', text: NOTE }] },
      { content_blocks: [{ type: 'redacted_thinking', data: 'x' }, { type: 'text', text: NOTE }] },
      { content_blocks: [{ type: 'fallback' }, thinking, { type: 'text', text: NOTE }] },
      { thinking: 'copied thought', thinking_signature: 'sig', content_blocks: [{ type: 'text', text: NOTE }] },
    ]) {
      const { sent } = await sendWith(hostRow(NOTE, { ...extra, keep: 'me' }));
      expect(additional(sent)).toEqual({ keep: 'me' });
      expect(sent.content).toBe(NOTE);
    }
  });

  it('a host row with a fallback block in content_blocks is not sent with its provider content, even when its blocks would match (prism-ts sends a different set of blocks from a fallback block on)', async () => {
    for (const [content, blocks] of [
      // The row's own call ahead of the fallback block: prism-ts would leave it out.
      ['Looking.', [toolUse('toolu_own'), { type: 'fallback' }, { type: 'text', text: 'Looking.' }]],
      // The row's own text ahead of the fallback block, and the call after it.
      ['Looking.', [{ type: 'text', text: 'Looking.' }, { type: 'fallback' }, toolUse('toolu_own')]],
      // Everything after the fallback block.
      ['Looking.', [{ type: 'fallback' }, { type: 'text', text: 'Looking.' }, toolUse('toolu_own')]],
    ] as const) {
      const { sent } = await sendWith(hostRow(content, { content_blocks: [...blocks] as JsonObject[], keep: 'me' }, [call('toolu_own')]));
      expect(additional(sent)).toEqual({ keep: 'me' });
      expect((sent.tool_calls as JsonObject[]).map((entry) => entry.id)).toEqual(['toolu_own']);
    }
  });

  it('a host row\'s tool calls are sent without a copied reasoning id or summary (a client would rebuild a reasoning item from them); stored unchanged', async () => {
    const copied = { ...call('fc_1'), reasoning_id: 'rs_copied', reasoning_summary: [{ type: 'summary_text', text: 'copied thought' }] };
    const { sent, stored } = await sendWith(hostRow('', {}, [copied]));
    expect((sent.tool_calls as JsonObject[]).map((entry) => [entry.id, entry.reasoning_id, entry.reasoning_summary])).toEqual([['fc_1', null, null]]);
    expect((stored.tool_calls as JsonObject[])[0]!.reasoning_id).toBe('rs_copied');
  });

  it('the harness\'s own tool calls keep their reasoning id and summary', async () => {
    const r = rig([
      { text: '', finishReason: 'tool_calls', toolCalls: [{ id: 'fc_9', name: 'lookup', arguments: { q: 'x' }, reasoningId: 'rs_9', reasoningSummary: [{ type: 'summary_text', text: 'own thought' }] }] },
      { text: 'done', finishReason: 'stop' },
    ]);
    await r.agent.send(r.session, 'go');
    const step = (r.requests.at(-1)!.messages as JsonObject[]).find((row) => row.type === 'assistant' && Array.isArray(row.tool_calls) && row.tool_calls.length > 0)!;
    expect((step.tool_calls as JsonObject[]).map((entry) => [entry.reasoning_id, entry.reasoning_summary])).toEqual([['rs_9', [{ type: 'summary_text', text: 'own thought' }]]]);
  });

  it('a host row whose tool_calls hold no objects counts as one with none', async () => {
    const { sent } = await sendWith(hostRow(NOTE, { output_items: [openaiCall('fc_1'), message(NOTE)], content_blocks: [{ type: 'text', text: NOTE }, toolUse('t1')] }, ['{"id":"fc_1"}']));
    expect(additional(sent)).toEqual({});
  });

  it('rows the harness writes from a provider response are sent byte for byte, even when their content would not match', async () => {
    const returned = {
      output_items: [{ type: 'reasoning', id: 'rs_9', summary: [] }, openaiCall('fc_9'), message('different text')],
      content_blocks: [{ type: 'fallback' }, toolUse('toolu_9'), { type: 'text', text: 'other' }],
      thinking: 'kept',
    };
    const r = rig([
      { text: '', finishReason: 'tool_calls', toolCalls: [{ id: 'call_1', name: 'lookup', arguments: { q: 'x' } }], additionalContent: returned },
      { text: 'done', finishReason: 'stop' },
    ]);
    await r.agent.send(r.session, 'go');
    const step = (r.requests.at(-1)!.messages as JsonObject[]).find((row) => row.type === 'assistant' && Array.isArray(row.tool_calls) && row.tool_calls.length > 0)!;
    expect(JSON.stringify(step.additional_content)).toBe(JSON.stringify(returned));
  });
});
