import { mkdtemp } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { describe, expect, it } from 'vitest';
import {
  AgentRuntime,
  FileSessionStore,
  HarnessError,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  ToolRegistry,
  type JsonObject,
  type LlmRequest,
  type LlmResponse,
  type Session,
} from '../src/index.js';

const modes = new ModeRegistry({
  default: 'chat',
  modes: {
    chat: { system_prompt: 'Be brief.', max_steps: 4 },
    thinking: { system_prompt: 'Think.', max_steps: 4, provider_options: { thinking: { enabled: true, budgetTokens: 4000 } } },
  },
});

async function aSession(mode = 'chat'): Promise<Session> {
  const directory = await mkdtemp(join(tmpdir(), 'prism-harness-attachments-'));
  const harness = new PrismHarness({
    drivers: { memory: () => new MemorySessionStore(), files: () => new FileSessionStore(directory) },
    stores: { ephemeral: 'memory', durable: 'files' },
  });
  const session = harness.for({ type: 'User', id: 1 }).session('support');
  await session.usingMode(mode);

  return session;
}

/** A model that remembers what it was asked. */
function recording(): { client: (request: LlmRequest) => Promise<LlmResponse>; requests: LlmRequest[] } {
  const requests: LlmRequest[] = [];

  return {
    requests,
    client: async (request) => {
      requests.push(request);

      return { text: 'Seen.', finishReason: 'stop' };
    },
  };
}

const image: JsonObject = { kind: 'image', url: null, base64: 'UE5HQllURVM=', mime_type: 'image/png', file_id: null, filename: null };

describe('attachments on a turn', () => {
  it('sends them with the prompt and stores them in the shape prism-ts rebuilds', async () => {
    const session = await aSession();
    const model = recording();

    await new AgentRuntime({ client: model.client, modes, tools: new ToolRegistry() }).send(session, 'What is this?', undefined, undefined, [image]);

    const turn = model.requests[0]!.messages[0]!;
    expect(turn).toEqual({
      type: 'user',
      content: 'What is this?',
      additional_content: [image, { text: 'What is this?' }],
      additional_attributes: {},
    });
    expect((await session.thread().messages())[0]!.message).toEqual(turn);
  });

  it('keeps a turn without attachments in its existing shape', async () => {
    const session = await aSession();
    const model = recording();

    await new AgentRuntime({ client: model.client, modes, tools: new ToolRegistry() }).send(session, 'Hi');

    expect(model.requests[0]!.messages[0]).toEqual({ type: 'user', content: 'Hi' });
  });

  it('asks a media object where it came from before serializing it', async () => {
    // A prism-ts media built from a local path serializes as bytes with no path,
    // so only the object itself can say it came from a file.
    const fromFile = { toObject: () => image, isUrl: () => false, isFile: () => true };
    const session = await aSession();
    const model = recording();

    await expect(
      new AgentRuntime({ client: model.client, modes, tools: new ToolRegistry() }).send(session, 'Look', undefined, undefined, [fromFile]),
    ).rejects.toMatchObject({ code: 'attachment_by_reference' });

    expect(model.requests).toHaveLength(0);
  });

  const refusals: Array<[string, unknown, string, string]> = [
    ['a url', { ...image, base64: null, url: 'http://169.254.169.254/latest/meta-data/' }, 'attachment_by_reference', 'Look'],
    ['a url that was fetched', { ...image, url: 'https://example.com/a.png' }, 'attachment_by_reference', 'Look'],
    ['a stored local path', { ...image, local_path: '/etc/passwd' }, 'attachment_by_reference', 'Look'],
    ['a url media object', { toObject: () => image, isUrl: () => true }, 'attachment_by_reference', 'Look'],
    ['a string', 'UE5HQllURVM=', 'attachment_not_media', 'Look'],
    ['a text part', { text: 'hello' }, 'attachment_not_media', 'Look'],
    ['an unknown kind', { ...image, kind: 'spreadsheet' }, 'attachment_not_media', 'Look'],
    ['empty base64', { ...image, base64: '' }, 'attachment_empty', 'Look'],
    ['nothing at all', { kind: 'image' }, 'attachment_empty', 'Look'],
    ['no chunks', { kind: 'document', chunks: [], document_title: 'Empty' }, 'attachment_empty', 'Look'],
    ['an empty prompt', image, 'attachment_without_prompt', ''],
  ];

  for (const [label, attachment, code, prompt] of refusals) {
    it(`refuses ${label} with ${code}, before a run or a request exists`, async () => {
      const session = await aSession();
      const model = recording();
      const agent = new AgentRuntime({ client: model.client, modes, tools: new ToolRegistry() });

      const refused = await agent.send(session, prompt, undefined, undefined, [attachment]).catch((error: unknown) => error);

      expect(refused).toBeInstanceOf(HarnessError);
      expect((refused as HarnessError).code).toBe(code);
      expect(model.requests).toHaveLength(0);
      expect(await session.thread().messages()).toHaveLength(0);
      expect(await session.run()).toBeNull();
    });
  }

  it('admits a provider file id and document chunks', async () => {
    const session = await aSession();
    const model = recording();

    await new AgentRuntime({ client: model.client, modes, tools: new ToolRegistry() }).send(session, 'Read these', undefined, undefined, [
      { kind: 'document', url: null, base64: null, mime_type: null, file_id: 'file_123', filename: null, document_title: 'Report', chunks: null },
      { kind: 'document', url: null, base64: null, mime_type: null, file_id: null, filename: null, document_title: 'Chunked', chunks: ['One.', 'Two.'] },
    ]);

    expect(model.requests).toHaveLength(1);
  });
});

describe('provider options per mode', () => {
  it('hands the mode\'s provider options to the model on every step', async () => {
    const session = await aSession('thinking');
    const model = recording();

    await new AgentRuntime({ client: model.client, modes, tools: new ToolRegistry() }).send(session, 'Think about it');

    expect(model.requests[0]!.providerOptions).toEqual({ thinking: { enabled: true, budgetTokens: 4000 } });
  });

  it('hands an empty map when a mode declares none', async () => {
    const session = await aSession();
    const model = recording();

    await new AgentRuntime({ client: model.client, modes, tools: new ToolRegistry() }).send(session, 'Hi');

    expect(model.requests[0]!.providerOptions).toEqual({});
  });

  it('refuses provider options that are not a map', () => {
    const broken = new ModeRegistry({ modes: { chat: { provider_options: ['thinking'] as unknown as Record<string, unknown> } } });

    expect(() => broken.resolve('chat')).toThrow(expect.objectContaining({ code: 'mode_malformed' }));
  });
});
