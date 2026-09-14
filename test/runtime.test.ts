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
  type LlmRequest,
  type LlmToolCall,
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

function runtime(client: (request: LlmRequest) => Promise<LlmResponse>, extra: Partial<{ tools: ToolRegistry; authorizer: ToolAuthorizer; events: HarnessEvents }> = {}) {
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

describe('what the next step is sent back (G-58)', () => {
  it("records each call's arguments, provider ids and the turn's provider state, and sends them on the next step", async () => {
    // The next request is built from the thread. Recorded with an id and a name
    // only, a client had no input to send for the tool_use it was replaying, and
    // nowhere to find the thinking signature Anthropic requires with it.
    const session = await aSession();
    const requests: LlmRequest[] = [];
    const responses: LlmResponse[] = [
      {
        text: 'Checking.',
        finishReason: 'tool_calls',
        toolCalls: [{ id: 'fc_1', name: 'echo', arguments: { value: 'x' }, resultId: 'call_1', reasoningId: 'rs_1' }],
        additionalContent: { thinking: 'Use the tool.', thinking_signature: 'sig-1' },
      },
      { text: 'Done.', finishReason: 'stop' },
    ];
    const client = async (request: LlmRequest): Promise<LlmResponse> => {
      requests.push(request);

      return responses[requests.length - 1]!;
    };

    await runtime(client).send(session, 'Use the tool');

    expect(requests[1]?.messages.slice(1)).toEqual([
      {
        type: 'assistant',
        content: 'Checking.',
        tool_calls: [
          { id: 'fc_1', name: 'echo', arguments: { value: 'x' }, result_id: 'call_1', reasoning_id: 'rs_1', reasoning_summary: null },
        ],
        additional_content: { thinking: 'Use the tool.', thinking_signature: 'sig-1' },
        tool_approval_requests: [],
      },
      {
        type: 'tool_result',
        tool_results: [
          { tool_call_id: 'fc_1', tool_name: 'echo', args: { value: 'x' }, result: 'echoed:x', tool_call_result_id: 'call_1', artifacts: [] },
        ],
        tool_approval_responses: [],
      },
    ]);
  });

  it('records empty provider state when the client reports none', async () => {
    const session = await aSession();
    const { client } = scripted([{ text: 'Hello.', finishReason: 'stop' }]);

    await runtime(client).send(session, 'Hi');

    expect((await session.thread().messages()).at(-1)?.message).toEqual({
      type: 'assistant',
      content: 'Hello.',
      tool_calls: [],
      additional_content: {},
      tool_approval_requests: [],
    });
  });

  it('records all of a step’s results as ONE row, as the reference does', async () => {
    const session = await aSession();
    const { client } = scripted([
      {
        text: '',
        finishReason: 'tool_calls',
        toolCalls: [
          { id: 'c1', name: 'echo', arguments: { value: 'a' } },
          { id: 'c2', name: 'echo', arguments: { value: 'b' } },
        ],
      },
      { text: 'Done.', finishReason: 'stop' },
    ]);

    await runtime(client).send(session, 'go');

    const rows = (await session.thread().messages()).map((m) => m.message);

    expect(rows.map((row) => row.type)).toEqual(['user', 'assistant', 'tool_result', 'assistant']);
    expect((rows[2]?.tool_results as { result: string }[]).map((entry) => entry.result)).toEqual(['echoed:a', 'echoed:b']);
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
  function counting(counts: Record<string, number>) {
    return new ToolRegistry()
      .register({ name: 'echo', handle: (args) => { counts.echo = (counts.echo ?? 0) + 1; return `echoed:${String(args.value ?? '')}`; } })
      .register({ name: 'shout', handle: (args) => { counts.shout = (counts.shout ?? 0) + 1; return `SHOUTED:${String(args.value ?? '')}`; } });
  }

  const guardedModes = new ModeRegistry({
    default: 'guarded',
    modes: { guarded: { system_prompt: 'Careful.', tools: ['echo', 'shout'], max_steps: 4, requires_approval: ['echo'] } },
  });

  function guardedRuntime(client: (request: LlmRequest) => Promise<LlmResponse>, counts: Record<string, number>) {
    return new AgentRuntime({ client, modes: guardedModes, tools: counting(counts) });
  }

  /** A model that asks for the calls ONCE and then answers; a real provider does not re-issue a call under the same id. */
  function once(toolCalls: LlmToolCall[], requests: LlmRequest[] = []) {
    return async (request: LlmRequest): Promise<LlmResponse> => {
      requests.push(request);

      return requests.length === 1
        ? { text: '', finishReason: 'tool_calls', toolCalls }
        : { text: `Finished after ${requests.length - 1}.`, finishReason: 'stop' };
    };
  }

  it('STOPS and does not run a gated tool that has no approval', async () => {
    // Failing closed is the only safe direction: an unanswered approval that
    // executed anyway is exactly what the mechanism exists to prevent.
    const session = await aSession('guarded');
    const counts: Record<string, number> = {};

    const response = await guardedRuntime(once([{ id: 'c1', name: 'echo', arguments: { value: 'x' } }]), counts).send(session, 'go');

    expect(counts).toEqual({});
    expect(response.finishReason).toBe('awaiting_approval');
    expect(response.pendingApprovals).toEqual([
      { id: expect.stringMatching(/^apr_[0-9a-f]{32}$/), toolCallId: 'c1', tool: 'echo', arguments: { value: 'x' } },
    ]);
  });

  it('writes the request onto the assistant row, so another process can resume it', async () => {
    const session = await aSession('guarded');

    const response = await guardedRuntime(once([{ id: 'c1', name: 'echo', arguments: {} }]), {}).send(session, 'go');

    const assistant = (await session.thread().messages()).find((m) => m.message.type === 'assistant')?.message;
    expect(assistant?.tool_approval_requests).toEqual([{ approval_id: response.pendingApprovals[0]!.id, tool_call_id: 'c1' }]);
  });

  it('runs the calls that need nobody, and records their results, before stopping for the rest', async () => {
    const session = await aSession('guarded');
    const counts: Record<string, number> = {};

    const response = await guardedRuntime(
      once([
        { id: 'c1', name: 'echo', arguments: { value: 'x' } },
        { id: 'c2', name: 'shout', arguments: { value: 'y' } },
      ]),
      counts,
    ).send(session, 'go');

    expect(counts).toEqual({ shout: 1 });
    expect(response.pendingApprovals.map((pending) => pending.toolCallId)).toEqual(['c1']);
    expect((await session.thread().messages()).map((m) => m.message.type)).toEqual(['user', 'assistant', 'tool_result']);
  });

  it('runs an approved call ONCE on the resumed turn, without asking the model again', async () => {
    // A real provider asked again issues the call afresh under a new id, so a
    // decision recorded against the old id would never match. The call that
    // stopped the run is answered where it is.
    const session = await aSession('guarded');
    const counts: Record<string, number> = {};
    const requests: LlmRequest[] = [];
    const agent = guardedRuntime(once([{ id: 'c1', name: 'echo', arguments: { value: 'x' } }], requests), counts);

    const first = await agent.send(session, 'go');
    await recordApproval(session, first.pendingApprovals[0]!.id, true);
    const resumed = await agent.send(session, '');

    expect(counts).toEqual({ echo: 1 });
    expect(resumed.text).toBe('Finished after 1.');
    expect(resumed.toolCalls).toEqual(['echo']);
    // The model saw one tool result turn holding the result and the decision.
    expect(requests[1]?.messages.at(-1)).toMatchObject({
      type: 'tool_result',
      tool_results: [{ tool_call_id: 'c1', result: 'echoed:x' }],
      tool_approval_responses: [{ approved: true }],
    });
  });

  it('sends a denied call the reason, and a call with no decision a refusal, and runs neither', async () => {
    const session = await aSession('guarded');
    const counts: Record<string, number> = {};
    const requests: LlmRequest[] = [];
    const agent = new AgentRuntime({
      client: once(
        [
          { id: 'c1', name: 'echo', arguments: { value: 'a' } },
          { id: 'c2', name: 'echo', arguments: { value: 'b' } },
        ],
        requests,
      ),
      modes: guardedModes,
      tools: counting(counts),
    });

    const first = await agent.send(session, 'go');
    await recordApproval(session, first.pendingApprovals[0]!.id, false, 'not today');
    await agent.send(session, '');

    expect(counts).toEqual({});
    expect((requests[1]?.messages.at(-1)?.tool_results as { tool_call_id: string; result: string }[])).toEqual([
      expect.objectContaining({ tool_call_id: 'c1', result: 'not today' }),
      expect.objectContaining({ tool_call_id: 'c2', result: 'No approval response provided' }),
    ]);
  });

  it('runs every approved call when all decisions are recorded before resuming', async () => {
    const session = await aSession('guarded');
    const counts: Record<string, number> = {};
    const agent = guardedRuntime(
      once([
        { id: 'c1', name: 'echo', arguments: { value: 'a' } },
        { id: 'c2', name: 'echo', arguments: { value: 'b' } },
      ]),
      counts,
    );

    const first = await agent.send(session, 'go');
    for (const pending of first.pendingApprovals) await recordApproval(session, pending.id, true);
    await agent.send(session, '');

    expect(counts).toEqual({ echo: 2 });
  });

  it('never runs an approved call again once it has a result', async () => {
    // The decision stays in the thread, and every later turn reads it again.
    const session = await aSession('guarded');
    const counts: Record<string, number> = {};
    const agent = guardedRuntime(once([{ id: 'c1', name: 'echo', arguments: { value: 'x' } }]), counts);

    const first = await agent.send(session, 'go');
    await recordApproval(session, first.pendingApprovals[0]!.id, true);
    await agent.send(session, '');
    await agent.send(session, 'And again?');
    await agent.send(session, '');

    expect(counts).toEqual({ echo: 1 });
  });

  it('resumes an approval recorded by 0.3.0, in the rows it wrote', async () => {
    // 0.3.0 kept the request in its own row, keyed by the CALL id, and answered
    // it in a tool_approval_response row.
    const session = await aSession('guarded');
    const counts: Record<string, number> = {};

    await session.thread().record([
      { type: 'user', content: 'go' },
      { type: 'assistant', content: '', tool_calls: [{ id: 'c1', name: 'echo', arguments: { value: 'x' } }], additional_content: {} },
      { type: 'tool_approval_request', approvals: [{ id: 'c1', tool: 'echo', arguments: { value: 'x' } }] },
      { type: 'tool_approval_response', approval_id: 'c1', approved: true, reason: null },
    ]);

    const resumed = await guardedRuntime(async () => ({ text: 'Finished.', finishReason: 'stop' }), counts).send(session, '');

    expect(counts).toEqual({ echo: 1 });
    expect(resumed.text).toBe('Finished.');
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
    expect(String((result?.message.tool_results as { result: string }[])[0]?.result)).toContain('exploded');
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
