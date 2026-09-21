import { mkdtemp } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { describe, expect, it } from 'vitest';
import {
  AgentRuntime,
  FileSessionStore,
  HarnessError,
  HarnessEvents,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  ToolRegistry,
  schemaProblems,
  type HarnessEvent,
  type HarnessTool,
  type JsonObject,
  type LlmRequest,
  type LlmResponse,
  type Session,
} from '../src/index.js';

/*
 * A turn whose answer is a document. The PHP reference shipped this as
 * Session::sendStructured() (prism-harness#13); these tests mirror its suite,
 * because the decisions worth porting are the RECORDED SHAPE and the refusal,
 * not the plumbing.
 */

const modes = new ModeRegistry({
  default: 'chat',
  modes: { chat: { system_prompt: 'Be brief.', tools: ['echo'], max_steps: 4 } },
});

const planSchema: JsonObject = {
  name: 'plan',
  type: 'object',
  properties: {
    title: { type: 'string' },
    steps: { type: 'array', items: { type: 'string' } },
    confidence: { type: 'number' },
  },
  required: ['title', 'steps'],
};

const planDocument = { title: 'Ship it', steps: ['write', 'test'], confidence: 0.8 };

function echoTool(): HarnessTool {
  return { name: 'echo', handle: (args) => `echoed:${String(args.value ?? '')}` };
}

function scripted(responses: LlmResponse[]): { client: (request: LlmRequest) => Promise<LlmResponse>; seen: LlmRequest[] } {
  const seen: LlmRequest[] = [];
  let call = 0;

  return {
    seen,
    client: async (request) => {
      seen.push(request);

      return responses[Math.min(call++, responses.length - 1)]!;
    },
  };
}

async function aSession(): Promise<Session> {
  const directory = await mkdtemp(join(tmpdir(), 'prism-harness-structured-'));
  const harness = new PrismHarness({
    drivers: { memory: () => new MemorySessionStore(), files: () => new FileSessionStore(directory) },
    stores: { ephemeral: 'memory', durable: 'files' },
  });
  const session = harness.for({ type: 'User', id: 1 }).session('support');
  await session.usingMode('chat');
  await session.usingProvider('anthropic');
  await session.usingModel('claude-sonnet-4-5');

  return session;
}

function runtime(client: (request: LlmRequest) => Promise<LlmResponse>, events?: HarnessEvents): AgentRuntime {
  return new AgentRuntime({ client, modes, tools: new ToolRegistry().register(echoTool()), events });
}

async function rows(session: Session): Promise<JsonObject[]> {
  return (await session.thread().messages()).map((entry) => entry.message as JsonObject);
}

describe('a structured turn', () => {
  it('returns the parsed document and the text it was read from', async () => {
    const session = await aSession();
    const { client } = scripted([
      { text: JSON.stringify(planDocument), finishReason: 'stop', structured: planDocument },
    ]);

    const response = await runtime(client).sendStructured(session, 'Plan the release', planSchema);

    expect(response.structured).toEqual(planDocument);
    expect(response.text).toBe(JSON.stringify(planDocument));
    expect(response.finishReason).toBe('stop');
  });

  it('hands the schema to the client, and nothing else changes about the request', async () => {
    const session = await aSession();
    const { client, seen } = scripted([
      { text: JSON.stringify(planDocument), finishReason: 'stop', structured: planDocument },
    ]);

    await runtime(client).sendStructured(session, 'Plan the release', planSchema);

    expect(seen[0]!.schema).toEqual(planSchema);
    expect(seen[0]!.systemPrompt).toBe('Be brief.');
    expect(seen[0]!.tools.map((tool) => tool.name)).toEqual(['echo']);
  });

  it('records the document as TEXT, with the parsed object beside it', async () => {
    // The thread stays readable by everything that reads text today. A
    // transcript that differs by the SHAPE of the request that produced it is
    // the same defect as one that differs when streamed.
    const session = await aSession();
    const { client } = scripted([
      { text: JSON.stringify(planDocument), finishReason: 'stop', structured: planDocument },
    ]);

    await runtime(client).sendStructured(session, 'Plan the release', planSchema);

    const stored = await rows(session);

    expect(stored.map((row) => row.type)).toEqual(['user', 'assistant']);
    expect(stored[1]!.content).toBe(JSON.stringify(planDocument));
    expect((stored[1]!.additional_content as JsonObject).structured).toEqual(planDocument);
  });

  it('replays to a later turn as the text the model wrote', async () => {
    const session = await aSession();
    const { client, seen } = scripted([
      { text: JSON.stringify(planDocument), finishReason: 'stop', structured: planDocument },
      { text: 'Yes, two steps.', finishReason: 'stop' },
    ]);
    const agent = runtime(client);

    await agent.sendStructured(session, 'Plan the release', planSchema);
    await agent.send(session, 'Is that all?');

    const replayed = seen[1]!.messages.filter((message) => message.type === 'assistant');

    expect(replayed.map((message) => message.content)).toEqual([JSON.stringify(planDocument)]);
    expect(seen[1]!.schema).toBeUndefined();
  });

  it('REFUSES a document that misses the schema, and says every way it missed', async () => {
    const session = await aSession();
    const wrong = { title: 'Ship it', confidence: 'very' };
    const { client } = scripted([{ text: JSON.stringify(wrong), finishReason: 'stop', structured: wrong }]);

    const thrown = await runtime(client)
      .sendStructured(session, 'Plan the release', planSchema)
      .then(() => null)
      .catch((error: unknown) => error as HarnessError);

    expect(thrown).toBeInstanceOf(HarnessError);
    expect(thrown?.code).toBe('structured_schema_violation');
    expect(thrown?.document).toBe(JSON.stringify(wrong));
    expect(thrown?.problems).toEqual([
      'plan.steps is required and missing.',
      'plan.confidence is the string "very", and the schema asks for \'number\'.',
    ]);
  });

  it('REFUSES text that holds no document at all, under its own code', async () => {
    const session = await aSession();
    const { client } = scripted([
      { text: 'I am afraid I cannot help with that.', finishReason: 'stop', structured: null },
    ]);

    const thrown = await runtime(client)
      .sendStructured(session, 'Plan the release', planSchema)
      .then(() => null)
      .catch((error: unknown) => error as HarnessError);

    expect(thrown?.code).toBe('structured_unreadable');
    expect(thrown?.document).toBe('I am afraid I cannot help with that.');
    expect(thrown?.problems).toEqual([]);
  });

  it('records the refused document too, and fails the run', async () => {
    // The exchange happened. A thread that omits the answer it did not like
    // cannot explain the retry sitting next to it.
    const session = await aSession();
    const wrong = { title: 'Ship it' };
    const { client } = scripted([{ text: JSON.stringify(wrong), finishReason: 'stop', structured: wrong }]);
    const seen: HarnessEvent[] = [];
    const events = new HarnessEvents();
    events.listen((event) => seen.push(event));

    await expect(runtime(client, events).sendStructured(session, 'Plan the release', planSchema)).rejects.toThrow(
      HarnessError,
    );

    expect((await rows(session)).map((row) => row.type)).toEqual(['user', 'assistant']);
    expect(seen.map((event) => event.type)).toContain('run.failed');
    expect(seen.map((event) => event.type)).not.toContain('run.finished');
  });

  it('keeps the model’s words out of the event, and names the code instead', async () => {
    // The document is in the thread, in full, where it is read deliberately.
    // An event carrying pieces of it would ship model output to every listener
    // by default — the same reason tool arguments are names-only here.
    const session = await aSession();
    const wrong = { title: 'Ship it', steps: 'do-not-log' };
    const { client } = scripted([{ text: JSON.stringify(wrong), finishReason: 'stop', structured: wrong }]);
    const seen: HarnessEvent[] = [];
    const events = new HarnessEvents();
    events.listen((event) => seen.push(event));

    await expect(runtime(client, events).sendStructured(session, 'Plan the release', planSchema)).rejects.toThrow(
      HarnessError,
    );

    expect(JSON.stringify(seen)).not.toContain('do-not-log');
    expect(seen.find((event) => event.type === 'run.failed')).toMatchObject({
      failure: 'structured_schema_violation',
    });
    expect(await session.run()).toMatchObject({ status: 'failed', failure: 'structured_schema_violation' });
  });

  it('records the tool rounds behind the answer', async () => {
    // A thread holding the document and forgetting the tool calls behind it
    // shows a later turn an agent that knew something for no reason.
    const session = await aSession();
    const { client } = scripted([
      { text: '', finishReason: 'tool_calls', toolCalls: [{ id: 'c1', name: 'echo', arguments: { value: 'notes' } }] },
      { text: JSON.stringify(planDocument), finishReason: 'stop', structured: planDocument },
    ]);

    const response = await runtime(client).sendStructured(session, 'Plan the release', planSchema);

    expect(response.toolCalls).toEqual(['echo']);
    expect((await rows(session)).map((row) => row.type)).toEqual([
      'user',
      'assistant',
      'tool_result',
      'assistant',
    ]);
  });
});

describe('the schema check', () => {
  /*
   * The same rules as the PHP reference's SchemaCheck, message for message. It
   * reads a JSON Schema rather than any object model, so a hand-written schema
   * is held to the same terms. What it cannot read it passes, and THAT is the
   * part worth pinning: a validator silently reporting nothing for a constraint
   * it did not check is the failure mode worth naming.
   */

  it('passes a document that satisfies its schema', () => {
    expect(schemaProblems(planSchema, planDocument, 'plan')).toEqual([]);
  });

  it('names a required field that is missing, by path', () => {
    expect(schemaProblems(planSchema, { steps: [] }, 'plan')).toEqual(['plan.title is required and missing.']);
  });

  it('reports EVERY problem, not the first', () => {
    expect(schemaProblems(planSchema, {}, 'plan')).toHaveLength(2);
  });

  it('checks the type of a value that is present', () => {
    expect(schemaProblems(planSchema, { title: 'Ship', steps: [], confidence: 'very' }, 'plan')).toEqual([
      "plan.confidence is the string \"very\", and the schema asks for 'number'.",
    ]);
  });

  it('accepts an integer where a number is asked for, and refuses a float where an integer is', () => {
    const number: JsonObject = { type: 'object', properties: { value: { type: 'number' } } };
    const integer: JsonObject = { type: 'object', properties: { value: { type: 'integer' } } };

    expect(schemaProblems(number, { value: 2 })).toEqual([]);
    expect(schemaProblems(number, { value: 2.5 })).toEqual([]);
    expect(schemaProblems(integer, { value: 2 })).toEqual([]);
    expect(schemaProblems(integer, { value: 2.5 })).toHaveLength(1);
  });

  it('checks inside an array, item by item', () => {
    const schema: JsonObject = {
      type: 'object',
      properties: {
        steps: {
          type: 'array',
          items: { type: 'object', properties: { do: { type: 'string' } }, required: ['do'], additionalProperties: false },
        },
      },
    };

    expect(schemaProblems(schema, { steps: [{ do: 'write' }, { note: 'oops' }] }, 'plan')).toEqual([
      'plan.steps[1].do is required and missing.',
      'plan.steps[1].note was returned, and the schema declares no such property.',
    ]);
  });

  it('refuses a value outside an enum, and names the members', () => {
    const schema: JsonObject = { type: 'object', properties: { mode: { enum: ['fast', 'careful'] } } };

    expect(schemaProblems(schema, { mode: 'reckless' }, 'plan')).toEqual([
      "plan.mode is the string \"reckless\", which is not one of 'fast', 'careful'.",
    ]);
  });

  it('accepts null only where the schema says so', () => {
    const strict: JsonObject = { type: 'object', properties: { title: { type: 'string' } } };
    const nullable: JsonObject = { type: 'object', properties: { title: { type: ['string', 'null'] } } };

    expect(schemaProblems(strict, { title: null })).toHaveLength(1);
    expect(schemaProblems(nullable, { title: null })).toEqual([]);
  });

  it('reports a key nobody declared ONLY when the schema closed itself', () => {
    const closed: JsonObject = { type: 'object', properties: { title: { type: 'string' } }, additionalProperties: false };
    const open: JsonObject = { type: 'object', properties: { title: { type: 'string' } } };

    expect(schemaProblems(closed, { title: 'Ship', extra: 1 })).toHaveLength(1);
    expect(schemaProblems(open, { title: 'Ship', extra: 1 })).toEqual([]);
  });

  it('tells an object from a list', () => {
    const object: JsonObject = { type: 'object', properties: {} };
    const array: JsonObject = { type: 'array', items: { type: 'string' } };

    expect(schemaProblems(object, ['a', 'b'])).toHaveLength(1);
    expect(schemaProblems(array, { title: 'Ship' })).toHaveLength(1);
    expect(schemaProblems(array, [])).toEqual([]);
  });

  it('passes what it cannot read rather than guessing', () => {
    expect(schemaProblems({ $ref: '#/definitions/Thing' }, { anything: true })).toEqual([]);
    expect(schemaProblems({ type: 'integer', minimum: 10 }, 1)).toEqual([]);
  });

  it('takes any branch of an anyOf', () => {
    const schema: JsonObject = { anyOf: [{ type: 'string' }, { type: 'number' }] };

    expect(schemaProblems(schema, 'text')).toEqual([]);
    expect(schemaProblems(schema, 3)).toEqual([]);
    expect(schemaProblems(schema, true)).toHaveLength(1);
  });
});
