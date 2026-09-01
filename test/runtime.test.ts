import { mkdtemp } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { describe, expect, it } from 'vitest';
import {
  AgentRuntime,
  MAX_DEPTH,
  FileSessionStore,
  HarnessEvents,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  RunBudget,
  RunContext,
  Subagent,
  ToolAuthorizer,
  ToolRegistry,
  recordApproval,
  type HarnessEvent,
  type HarnessTool,
  type LlmResponse,
  type Session,
} from '../src/index.js';

const modes = new ModeRegistry({
  default: 'chat',
  modes: {
    chat: { system_prompt: 'Be brief.', tools: ['echo'], max_steps: 4 },
    guarded: { system_prompt: 'Careful.', tools: ['echo'], max_steps: 4, requires_approval: ['echo'] },
  },
});

function echoTool(): HarnessTool {
  return { name: 'echo', handle: (args) => `echoed:${String(args.value ?? '')}` };
}

function failingTool(): HarnessTool {
  return {
    name: 'echo',
    handle: () => {
      throw new Error('the tool exploded');
    },
  };
}

/** A scripted model: each call returns the next response, then repeats the last. */
function scripted(responses: LlmResponse[]): { client: () => Promise<LlmResponse> } {
  let call = 0;

  return {
    client: async () => responses[Math.min(call++, responses.length - 1)]!,
  };
}

async function aSession(mode = 'chat'): Promise<Session> {
  const directory = await mkdtemp(join(tmpdir(), 'prism-harness-runtime-'));
  const harness = new PrismHarness({
    drivers: { memory: () => new MemorySessionStore(), files: () => new FileSessionStore(directory) },
    stores: { ephemeral: 'memory', durable: 'files' },
  });
  const session = harness.for({ type: 'User', id: 1 }).session('support');
  await session.usingMode(mode);
  await session.usingProvider('anthropic');
  await session.usingModel('claude-sonnet-4-5');

  return session;
}

function runtime(client: () => Promise<LlmResponse>, extra: Partial<{ tools: ToolRegistry; authorizer: ToolAuthorizer; events: HarnessEvents }> = {}) {
  return new AgentRuntime({
    client,
    modes,
    tools: extra.tools ?? new ToolRegistry().register(echoTool()),
    authorizer: extra.authorizer,
    events: extra.events,
  });
}

describe('a plain turn', () => {
  it('returns the text and records both messages in the thread', async () => {
    const session = await aSession();
    const { client } = scripted([{ text: 'Hello.', finishReason: 'stop' }]);

    const response = await runtime(client).send(session, 'Hi');

    expect(response.text).toBe('Hello.');
    expect(response.steps).toBe(1);
    expect(response.finishReason).toBe('stop');
    expect((await session.thread().messages()).map((m) => m.message.type)).toEqual([
      'user',
      'assistant',
    ]);
  });

  it('marks the run completed, with the tools it reached for', async () => {
    const session = await aSession();
    const { client } = scripted([
      { text: '', finishReason: 'tool_calls', toolCalls: [{ id: 'c1', name: 'echo', arguments: { value: 'x' } }] },
      { text: 'Done.', finishReason: 'stop' },
    ]);

    const response = await runtime(client).send(session, 'Use the tool');

    expect(response.toolCalls).toEqual(['echo']);
    expect(await session.run()).toMatchObject({ status: 'completed', tool_calls: ['echo'] });
  });

  it('does NOT record a user message for an empty prompt', async () => {
    // An empty prompt is how a run resumes after an approval: the conversation
    // already holds the request and the decision, and a new prompt there would
    // be a second instruction competing with the one the tool call came from.
    const session = await aSession();
    const { client } = scripted([{ text: 'ok', finishReason: 'stop' }]);

    await runtime(client).send(session, '');

    expect((await session.thread().messages()).map((m) => m.message.type)).toEqual(['assistant']);
  });
});

describe('budgets', () => {
  it('stops BEFORE taking a step it cannot afford', async () => {
    // Checking afterwards means the step that broke the limit has already been
    // paid for, which makes a budget a report rather than a control.
    const session = await aSession();
    let calls = 0;
    const client = async (): Promise<LlmResponse> => {
      calls += 1;

      return { text: 'again', finishReason: 'tool_calls', toolCalls: [{ id: `c${calls}`, name: 'echo', arguments: {} }] };
    };

    const context = RunContext.root('root', new RunBudget(2));
    const response = await runtime(client).send(session, 'go', undefined, context);

    expect(calls).toBe(2);
    expect(response.stoppedBecause).toMatch(/step budget exhausted/);
    expect(response.finishReason).toBe('budget_exhausted');
  });

  it('reports a cancellation as the reason it stopped', async () => {
    const session = await aSession();
    const context = RunContext.root('root', new RunBudget(4));
    context.ledger.cancel('the user closed the tab');

    const { client } = scripted([{ text: 'never', finishReason: 'stop' }]);
    const response = await runtime(client).send(session, 'go', undefined, context);

    expect(response.stoppedBecause).toBe('the user closed the tab');
  });

  it('refuses a run nested past the depth ceiling', async () => {
    const session = await aSession();
    let context = RunContext.root('root', new RunBudget(8));
    const child = new Subagent('r', '', 'chat', new RunBudget(8));
    for (let depth = 0; depth < MAX_DEPTH; depth += 1) {
      context = context.forChild(child, 'root');
    }

    const { client } = scripted([{ text: 'x', finishReason: 'stop' }]);

    await expect(runtime(client).send(session, 'go', undefined, context)).rejects.toMatchObject({
      code: 'run_not_permitted',
    });
  });
});

describe('approvals', () => {
  it('STOPS and does not run a gated tool that has no approval', async () => {
    // Failing closed is the only safe direction: an unanswered approval that
    // executed anyway is exactly what the mechanism exists to prevent.
    const session = await aSession('guarded');
    let handled = 0;
    const tools = new ToolRegistry().register({
      name: 'echo',
      handle: () => {
        handled += 1;

        return 'ran';
      },
    });
    const { client } = scripted([
      { text: '', finishReason: 'tool_calls', toolCalls: [{ id: 'c1', name: 'echo', arguments: { value: 'x' } }] },
    ]);

    const response = await runtime(client, { tools }).send(session, 'go');

    expect(handled).toBe(0);
    expect(response.finishReason).toBe('awaiting_approval');
    expect(response.pendingApprovals).toEqual([{ id: 'c1', tool: 'echo', arguments: { value: 'x' } }]);
  });

  it('writes the request to the THREAD, so another process can resume it', async () => {
    const session = await aSession('guarded');
    const { client } = scripted([
      { text: '', finishReason: 'tool_calls', toolCalls: [{ id: 'c1', name: 'echo', arguments: {} }] },
    ]);

    await runtime(client).send(session, 'go');

    const types = (await session.thread().messages()).map((m) => m.message.type);
    expect(types).toContain('tool_approval_request');
  });

  it('runs the tool once the approval is recorded, on a RESUMED turn', async () => {
    // The approval a person grants this morning is a durable row, so the worker
    // that resumes tonight — a different process, possibly after a deploy —
    // reads the same answer.
    const session = await aSession('guarded');
    let handled = 0;
    const tools = new ToolRegistry().register({
      name: 'echo',
      handle: () => {
        handled += 1;

        return 'ran';
      },
    });

    let turn = 0;
    const client = async (): Promise<LlmResponse> => {
      turn += 1;

      return turn <= 2
        ? { text: '', finishReason: 'tool_calls', toolCalls: [{ id: 'c1', name: 'echo', arguments: {} }] }
        : { text: 'Finished.', finishReason: 'stop' };
    };

    const agent = runtime(client, { tools });

    await agent.send(session, 'go');
    expect(handled).toBe(0);

    await recordApproval(session, 'c1', true);
    const resumed = await agent.send(session, '');

    expect(handled).toBe(1);
    expect(resumed.text).toBe('Finished.');
  });

  it('does not ask twice once an approval is answered', async () => {
    const session = await aSession('guarded');
    const { client } = scripted([
      { text: '', finishReason: 'tool_calls', toolCalls: [{ id: 'c1', name: 'echo', arguments: {} }] },
      { text: 'done', finishReason: 'stop' },
    ]);

    await runtime(client).send(session, 'go');
    await recordApproval(session, 'c1', true);
    const resumed = await runtime(client).send(session, '');

    expect(resumed.pendingApprovals).toEqual([]);
  });
});

describe('tools', () => {
  it('records a failed tool as a RESULT rather than crashing the run', async () => {
    // The model can often recover, and losing the whole turn to one bad call is
    // worse than telling it what happened.
    const session = await aSession();
    const tools = new ToolRegistry().register(failingTool());
    const { client } = scripted([
      { text: '', finishReason: 'tool_calls', toolCalls: [{ id: 'c1', name: 'echo', arguments: {} }] },
      { text: 'Recovered.', finishReason: 'stop' },
    ]);

    const response = await runtime(client, { tools }).send(session, 'go');

    expect(response.text).toBe('Recovered.');
    const result = (await session.thread().messages()).find((m) => m.message.type === 'tool_result');
    expect(result?.message.failed).toBe(true);
    expect(String(result?.message.result)).toContain('exploded');
  });

  it('lets a REFUSED call propagate rather than feeding it back to the model', async () => {
    const session = await aSession();
    const tools = new ToolRegistry().register(echoTool());
    const authorizer = new ToolAuthorizer({ enabled: true, call: () => false });
    const { client } = scripted([
      { text: '', finishReason: 'tool_calls', toolCalls: [{ id: 'c1', name: 'echo', arguments: {} }] },
    ]);

    await expect(runtime(client, { tools, authorizer }).send(session, 'go')).rejects.toMatchObject({
      code: 'call_not_authorized',
    });
  });
});

describe('events and failures', () => {
  it('emits started and finished, with tool NAMES only', async () => {
    const session = await aSession();
    const events = new HarnessEvents();
    const seen: HarnessEvent[] = [];
    events.listen((event) => seen.push(event));

    const { client } = scripted([
      { text: '', finishReason: 'tool_calls', toolCalls: [{ id: 'c1', name: 'echo', arguments: { secret: 'do-not-log' } }] },
      { text: 'done', finishReason: 'stop', costUsd: 0.01 },
    ]);

    await runtime(client, { events }).send(session, 'go');

    expect(seen.map((event) => event.type)).toEqual(['run.started', 'run.finished']);
    expect(JSON.stringify(seen)).not.toContain('do-not-log');
  });

  it('reports a null cost rather than pretending the tree spent nothing', async () => {
    const session = await aSession();
    const events = new HarnessEvents();
    const seen: HarnessEvent[] = [];
    events.listen((event) => seen.push(event));

    const { client } = scripted([{ text: 'done', finishReason: 'stop' }]);
    await runtime(client, { events }).send(session, 'go');

    const finished = seen.find((event) => event.type === 'run.finished');
    expect(finished).toMatchObject({ costUsd: null });
  });

  it('marks the run failed and emits, when the model throws', async () => {
    const session = await aSession();
    const events = new HarnessEvents();
    const seen: HarnessEvent[] = [];
    events.listen((event) => seen.push(event));

    const client = async (): Promise<LlmResponse> => {
      throw new Error('the provider is down');
    };

    await expect(runtime(client, { events }).send(session, 'go')).rejects.toThrow('provider is down');

    expect(await session.run()).toMatchObject({ status: 'failed' });
    expect(seen.map((event) => event.type)).toEqual(['run.started', 'run.failed']);
  });
});
