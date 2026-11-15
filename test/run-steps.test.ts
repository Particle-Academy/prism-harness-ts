import { describe, expect, it } from 'vitest';
import {
  AgentRuntime,
  HarnessError,
  HarnessEvents,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  RUN_STEP_CONTRACT_VERSION,
  RunBudget,
  RunContext,
  Subagent,
  SubagentRunner,
  ToolAuthorizer,
  ToolRegistry,
  agentCompletionTool,
  recordApproval,
  type AgentResponse,
  type AgentTask,
  type AgentTaskSource,
  type Durability,
  type HarnessEvent,
  type HarnessTool,
  type JsonObject,
  type LlmRequest,
  type LlmResponse,
  type LlmToolCall,
  type RunStep,
  type RuntimeConfig,
  type Session,
} from '../src/index.js';
import { markRefused } from '../src/refusals.js';

// G-282 A and A+ (G-284): the tagged run step stream, ported from the PHP
// reference's tests/RunStepStreamTest.php. The field names are an agreed
// contract with Fancy (contractVersion g282.1); these tests pin the shape, the
// sequence, the timing, refused-versus-failed, the session lock, and that the
// streamed approval continuation records every decision before any tool runs.

const ISO_MICROS = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$/;
const ENVELOPE = ['contractVersion', 'runId', 'parentRunId', 'rootRunId', 'turnId', 'seq', 'at', 'type'];

const modes = new ModeRegistry({
  default: 'chat',
  modes: {
    chat: { system_prompt: 'Be brief.', tools: ['tick', 'probe'], max_steps: 4 },
    guarded: { system_prompt: 'Careful.', tools: ['tick', 'probe'], max_steps: 4, requires_approval: ['tick'] },
  },
});

/** Lives as long as the test, and says so: the durable slot refuses a volatile store. */
class DurableMemoryStore extends MemorySessionStore {
  override durability(): Durability {
    return 'durable';
  }
}

/** One harness, so two sessions at one address share the same stores and lock. */
function harness(durable: MemorySessionStore = new DurableMemoryStore()): PrismHarness {
  const ephemeral = new MemorySessionStore();

  return new PrismHarness({
    drivers: { memory: () => ephemeral, durable: () => durable },
    stores: { ephemeral: 'memory', durable: 'durable' },
  });
}

async function sessionOf(h: PrismHarness, scope = 'chat', mode = 'chat'): Promise<Session> {
  const session = h.for({ type: 'User', id: 1 }).session(scope);
  await session.usingMode(mode);
  await session.usingProvider('anthropic');
  await session.usingModel('m');

  return session;
}

function scripted(responses: LlmResponse[], calls: LlmRequest[] = []) {
  return async (request: LlmRequest): Promise<LlmResponse> => {
    calls.push(request);

    return responses[Math.min(calls.length - 1, responses.length - 1)]!;
  };
}

function toolStep(calls: [string, string][], text = 'working'): LlmResponse {
  return { text, finishReason: 'tool_calls', toolCalls: calls.map(([id, name]): LlmToolCall => ({ id, name, arguments: {} })) };
}

const FINAL: LlmResponse = { text: 'all done', finishReason: 'stop' };

function tools(handlers: Partial<Record<'tick' | 'probe', HarnessTool['handle']>> = {}): ToolRegistry {
  return new ToolRegistry()
    .register({ name: 'tick', handle: handlers.tick ?? (() => 'ticked') })
    .register({ name: 'probe', handle: handlers.probe ?? (() => 'probed') });
}

function runtimeWith(
  client: (request: LlmRequest) => Promise<LlmResponse>,
  options: { tools?: ToolRegistry; events?: HarnessEvents; stepEvents?: boolean; config?: RuntimeConfig; authorizer?: ToolAuthorizer } = {},
): AgentRuntime {
  return new AgentRuntime({ client, modes, tools: options.tools ?? tools(), events: options.events, stepEvents: options.stepEvents, config: options.config, authorizer: options.authorizer });
}

async function collect(stream: AsyncGenerator<RunStep, AgentResponse, undefined>): Promise<{ steps: RunStep[]; response: AgentResponse }> {
  const steps: RunStep[] = [];
  for (;;) {
    const next = await stream.next();
    if (next.done === true) return { steps, response: next.value };
    steps.push(next.value);
  }
}

function ofType<T extends RunStep['type']>(steps: readonly RunStep[], type: T): Extract<RunStep, { type: T }>[] {
  return steps.filter((step) => step.type === type) as Extract<RunStep, { type: T }>[];
}

function recording(): { events: HarnessEvents; seen: HarnessEvent[] } {
  const events = new HarnessEvents();
  const seen: HarnessEvent[] = [];
  events.listen((event) => { seen.push(event); });

  return { events, seen };
}

const tick = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 1));

describe('the run step envelope', () => {
  it('tags every step with the agreed envelope and contract version, in wire order', async () => {
    const h = harness();
    const { steps } = await collect(runtimeWith(scripted([{ text: 'hello there', finishReason: 'stop' }])).streamSteps(await sessionOf(h), 'go'));

    expect(steps.map((step) => step.type)).toEqual(['step_start', 'text', 'step_finish']);
    for (const step of steps) {
      // JSON order IS the contract: what Fancy parses is the serialised bytes.
      expect(Object.keys(JSON.parse(JSON.stringify(step))).slice(0, 8)).toEqual(ENVELOPE);
      expect(step.contractVersion).toBe('g282.1');
      expect(RUN_STEP_CONTRACT_VERSION).toBe('g282.1');
      expect(step.parentRunId).toBeNull();
      expect(step.rootRunId).toBe(step.runId);
      expect(step.turnId).toBeNull();
      expect(step.at).toMatch(ISO_MICROS);
    }
    expect(Object.keys(steps[1]!)).toEqual([...ENVELOPE, 'delta']);
    expect((steps[1] as { delta: string }).delta).toBe('hello there');
    expect(Object.keys(steps[2]!)).toEqual([...ENVELOPE, 'text']);
    expect((steps[2] as { text: string }).text).toBe('hello there');
  });

  it('numbers steps from 1 with no gaps, and emits each as run.stepped in the same order when opted in', async () => {
    const h = harness();
    const { events, seen } = recording();
    const session = await sessionOf(h);
    const { steps } = await collect(runtimeWith(scripted([toolStep([['c1', 'tick']]), FINAL]), { events, stepEvents: true }).streamSteps(session, 'go'));
    const stepped = seen.filter((event) => event.type === 'run.stepped');

    expect(steps.length).toBeGreaterThan(4);
    expect(steps.map((step) => step.seq)).toEqual(steps.map((_, index) => index + 1));
    expect(stepped.map((event) => event.step)).toEqual(steps);
    expect(stepped[0]).toMatchObject({ runId: steps[0]!.runId, rootRunId: steps[0]!.runId, parentRunId: null, sessionKey: session.key() });
  });

  it('freezes a step, so a listener cannot alter what the reader or the tool receives', async () => {
    const h = harness();
    const { events } = recording();
    const received: JsonObject[] = [];
    events.listen((event) => {
      if (event.type === 'run.stepped' && event.step.type === 'tool_call') (event.step.args as JsonObject).path = '/etc/passwd';
    });
    const rt = runtimeWith(async () => ({ text: '', finishReason: 'tool_calls', toolCalls: [{ id: 'c1', name: 'tick', arguments: { path: '/tmp/x' } }] }), {
      events, stepEvents: true, tools: tools({ tick: (args) => { received.push(args); return 'ok'; } }),
    });
    const quiet = process.emitWarning;
    process.emitWarning = () => {};
    try {
      const { steps } = await collect(rt.streamSteps(await sessionOf(h), 'go', undefined, RunContext.root('r', new RunBudget(1))));
      expect(ofType(steps, 'tool_call')[0]!.args).toEqual({ path: '/tmp/x' });
    } finally {
      process.emitWarning = quiet;
    }
    expect(received).toEqual([{ path: '/tmp/x' }]);
  });
});

describe('send() is unchanged', () => {
  it('emits only run.started and run.finished by default, and records the same thread with or without step events', async () => {
    const shapes: unknown[] = [];
    for (const stepEvents of [false, true]) {
      const h = harness();
      const { events, seen } = recording();
      const session = await sessionOf(h);
      const response = await runtimeWith(scripted([toolStep([['c1', 'tick']]), FINAL]), { events, stepEvents }).send(session, 'go');
      const rows = (await session.thread().messages()).map((entry) => entry.message);
      shapes.push({ response: { ...response, runId: 'x' }, rows, run: { ...(await session.run()), id: 'x', started_at: 'x', finished_at: 'x', last_seq: 'x' } });
      const types = seen.map((event) => event.type);
      if (stepEvents) {
        expect(types[0]).toBe('run.started');
        expect(types.at(-1)).toBe('run.finished');
        expect(types.filter((type) => type === 'run.stepped').length).toBeGreaterThan(0);
      } else {
        expect(types).toEqual(['run.started', 'run.finished']);
        // Nothing step-shaped is written to a run that did not ask for steps.
        expect((await session.run())?.last_seq).toBeUndefined();
      }
    }
    expect(shapes[1]).toEqual(shapes[0]);
  });

  it('records the same thread from streamSteps() as from send()', async () => {
    const rows: unknown[] = [];
    for (const streamed of [false, true]) {
      const session = await sessionOf(harness());
      const rt = runtimeWith(scripted([toolStep([['c1', 'tick'], ['c2', 'probe']]), FINAL]));
      if (streamed) await collect(rt.streamSteps(session, 'go'));
      else await rt.send(session, 'go');
      rows.push((await session.thread().messages()).map((entry) => entry.message));
    }
    expect(rows[1]).toEqual(rows[0]);
  });
});

describe('lineage and timing', () => {
  it.each(['plain', 'tree'] as const)('carries the run lineage the %s path has, and remembers where its sequence ended', async (path) => {
    const session = await sessionOf(harness());
    const context = path === 'plain' ? undefined : RunContext.root('tree-root', new RunBudget(10)).withTurn('turn-7');
    const { steps } = await collect(runtimeWith(scripted([toolStep([['call-1', 'tick']]), FINAL])).streamSteps(session, 'go', undefined, context));

    expect(new Set(steps.map((step) => step.runId)).size).toBe(1);
    expect(steps[0]!.rootRunId).toBe(path === 'plain' ? steps[0]!.runId : 'tree-root');
    expect(steps[0]!.turnId).toBe(path === 'plain' ? null : 'turn-7');
    expect((await session.run())?.status).toBe('completed');
    expect((await session.run())?.last_seq).toBe(steps.length);
    const call = ofType(steps, 'tool_call')[0]!;
    expect(call.toolCallId).toBe('call-1');
    expect(JSON.stringify(call.args)).toBe('{}');
    expect(ofType(steps, 'tool_result')[0]).toMatchObject({ toolCallId: 'call-1', ok: true, result: 'ticked', refused: false });
  });

  it('measures a tool where it ran, from its call to its result', async () => {
    const session = await sessionOf(harness());
    const rt = runtimeWith(scripted([toolStep([['c1', 'tick']]), FINAL]), {
      tools: tools({ tick: async () => { await new Promise((resolve) => setTimeout(resolve, 40)); return 'ticked'; } }),
    });
    const { steps } = await collect(rt.streamSteps(session, 'go'));
    const call = ofType(steps, 'tool_call')[0]!;
    const result = ofType(steps, 'tool_result')[0]!;

    expect(Object.keys(call).slice(8)).toEqual(['toolCallId', 'name', 'args', 'startedAt']);
    expect(call.name).toBe('tick');
    expect(call.startedAt).toMatch(ISO_MICROS);
    expect(Object.keys(result).slice(8)).toEqual(['toolCallId', 'ok', 'result', 'refused', 'durationMs', 'durationExact']);
    expect(result).toMatchObject({ toolCallId: 'c1', ok: true, result: 'ticked', refused: false, durationExact: true });
    expect(Number.isInteger(result.durationMs)).toBe(true);
    // A timer may fire a hair early; 35 still proves it was measured, not zero.
    expect(result.durationMs).toBeGreaterThanOrEqual(35);
    expect(result.seq).toBeGreaterThan(call.seq);
    // core's order: calls, then results, then the approvals waiting on a person.
    expect(steps.map((step) => step.type).slice(0, 5)).toEqual(['step_start', 'text', 'tool_call', 'tool_result', 'step_finish']);
  });

  it('marks a duration exact for a lone call and inexact when siblings ran in the same step', async () => {
    const session = await sessionOf(harness());
    const rt = runtimeWith(scripted([toolStep([['call-1', 'tick'], ['call-2', 'probe']]), toolStep([['call-3', 'tick']]), FINAL]));
    const { steps } = await collect(rt.streamSteps(session, 'go'));

    expect(Object.fromEntries(ofType(steps, 'tool_result').map((step) => [step.toolCallId, step.durationExact])))
      .toEqual({ 'call-1': false, 'call-2': false, 'call-3': true });
  });
});

describe('refused versus failed', () => {
  it('reports a tool that threw as failed, not refused', async () => {
    const session = await sessionOf(harness());
    const rt = runtimeWith(scripted([toolStep([['c1', 'tick']]), FINAL]), { tools: tools({ tick: () => { throw new Error('disk on fire'); } }) });
    const result = ofType((await collect(rt.streamSteps(session, 'go'))).steps, 'tool_result')[0]!;

    expect(Object.keys(result).slice(8)).toEqual(['toolCallId', 'ok', 'error', 'refused', 'durationMs', 'durationExact']);
    expect(result).toMatchObject({ ok: false, refused: false, error: { code: 'tool_failed', message: 'The tool failed: disk on fire' } });
  });

  it('does not take a tool that merely returns a refusal-shaped string for a refusal', async () => {
    const session = await sessionOf(harness());
    const imitation = 'The [tick] subagent did not run: the tree has nothing left to spend.';
    const rt = runtimeWith(scripted([toolStep([['c1', 'tick']]), FINAL]), { tools: tools({ tick: () => imitation }) });
    const result = ofType((await collect(rt.streamSteps(session, 'go'))).steps, 'tool_result')[0]!;

    expect(result).toMatchObject({ ok: true, result: imitation, refused: false });
  });

  it('reports a subagent the tree refused to run as refused, though the model got an ordinary result', async () => {
    const h = harness();
    const session = await sessionOf(h);
    const childRuntime = runtimeWith(async () => FINAL);
    const exhausted = RunContext.root('tree', new RunBudget(1));
    exhausted.ledger.recordSteps(1);
    const subagent = new SubagentRunner(h, childRuntime).tool(new Subagent('tick', 'Child', 'chat', new RunBudget(1)), session, exhausted, 'tree');
    const rt = runtimeWith(scripted([toolStep([['c1', 'tick']]), FINAL]), { tools: new ToolRegistry().register(subagent).register({ name: 'probe', handle: () => 'probed' }) });
    const result = ofType((await collect(rt.streamSteps(session, 'go'))).steps, 'tool_result')[0]!;

    expect(result).toMatchObject({ ok: false, refused: true, error: { code: 'tool_call_refused' } });
    if (!result.ok) expect(result.error.message).toMatch(/^The \[tick\] subagent did not run: /);
  });

  it('reports a task completion gate as refused', async () => {
    const session = await sessionOf(harness());
    const task: AgentTask = { id: () => 't-1', instruction: () => 'do it', state: () => 'claimed' } as unknown as AgentTask;
    const source = { find: async () => null } as unknown as AgentTaskSource;
    const registry = tools().registerFactory('complete_task', agentCompletionTool({ source, task, worker: 'w-1', authorizer: new ToolAuthorizer({ enabled: true }), name: 'complete_task' }));
    const gatedModes = new ModeRegistry({ default: 'chat', modes: { chat: { system_prompt: '', tools: ['complete_task'], max_steps: 4 } } });
    const rt = new AgentRuntime({
      client: scripted([{ text: '', finishReason: 'tool_calls', toolCalls: [{ id: 'c1', name: 'complete_task', arguments: { outcome: 'done' } }] }, FINAL]),
      modes: gatedModes, tools: registry, authorizer: new ToolAuthorizer({ enabled: true }),
    });
    const result = ofType((await collect(rt.streamSteps(session, 'go'))).steps, 'tool_result')[0]!;

    expect(result).toMatchObject({ ok: false, refused: true, error: { code: 'tool_call_refused' } });
  });

  it('fails the run, as send() does, when the model calls a tool the run does not offer', async () => {
    // A divergence from the reference, where core returns this as a result: the
    // TS loop refuses the call by failing the run, and the stream says so.
    const session = await sessionOf(harness());
    const stream = runtimeWith(scripted([toolStep([['c1', 'shell_exec']]), FINAL])).streamSteps(session, 'go');
    const seen: RunStep[] = [];
    await expect((async () => { for await (const step of stream) seen.push(step); })()).rejects.toMatchObject({ code: 'tool_not_available' });
    expect(seen.map((step) => step.type)).toEqual(['step_start', 'text', 'tool_call']);
    expect((await session.run())?.status).toBe('failed');
  });

  it('reports an error step with a FIXED message, never the provider text', async () => {
    const session = await sessionOf(harness());
    const stream = runtimeWith(async () => ({ text: '', finishReason: 'error', additionalContent: { error: 'https://api.example/key=sk-secret' } })).streamSteps(session, 'go');
    const seen: RunStep[] = [];
    await expect((async () => { for await (const step of stream) seen.push(step); })()).rejects.toMatchObject({ code: 'model_step_failed' });
    const error = ofType(seen, 'error')[0]!;

    expect(Object.keys(error).slice(8)).toEqual(['error']);
    expect(error.error).toEqual({ code: 'model_step_failed', message: 'The model stream reported an error.' });
    expect(JSON.stringify(seen)).not.toContain('sk-secret');
  });

  it('keeps the run going when a run.stepped listener throws', async () => {
    const h = harness();
    const events = new HarnessEvents();
    events.listen(() => { throw new Error('a broken renderer'); });
    const warnings: string[] = [];
    const quiet = process.emitWarning;
    process.emitWarning = ((warning: string) => { warnings.push(String(warning)); }) as typeof process.emitWarning;
    try {
      const session = await sessionOf(h);
      const rt = runtimeWith(scripted([{ text: 'still answered', finishReason: 'stop' }]), { events, stepEvents: true });
      const { steps } = await collect(rt.streamSteps(session, 'go'));
      const second = await sessionOf(h, 'second');
      await rt.send(second, 'go');

      expect(ofType(steps, 'text')[0]!.delta).toBe('still answered');
      expect((await session.run())?.status).toBe('completed');
      expect((await second.run())?.status).toBe('completed');
    } finally {
      process.emitWarning = quiet;
    }
    // Isolated, not swallowed: the failure is still reported.
    expect(warnings.some((warning) => warning.includes('a broken renderer'))).toBe(true);
  });

  it('emits no run.stepped by default, so content never reaches apps that record every event', async () => {
    const session = await sessionOf(harness());
    const { events, seen } = recording();
    const rt = runtimeWith(scripted([{ text: 'private answer', finishReason: 'stop' }]), { events });
    await rt.send(session, 'go');
    const { steps } = await collect(rt.streamSteps(session, 'again'));

    expect(seen.filter((event) => event.type === 'run.stepped')).toEqual([]);
    expect(JSON.stringify(seen)).not.toContain('private answer');
    expect(steps.length).toBeGreaterThan(0);
  });
});

describe('refusals are scoped to one call', () => {
  it('ignores a refusal marked outside any tool call', () => {
    expect(() => markRefused()).not.toThrow();
  });

  it('keeps two concurrent runs with an identical (tool, result) apart', async () => {
    // Run A's subagent is refused; while A's call is still in flight, run B (a
    // different session) gets the SAME string from a `tick` that really ran.
    const h = harness();
    const refusal = 'The [tick] subagent did not run: nothing left.';
    let inner: RunStep[] = [];
    const sessionB = await sessionOf(h, 'run-b');
    const runB = runtimeWith(scripted([toolStep([['call-b1', 'tick']]), FINAL]), { tools: tools({ tick: () => refusal }) });
    const runA = runtimeWith(scripted([toolStep([['call-a1', 'tick'], ['call-a2', 'probe']]), FINAL]), {
      tools: tools({
        tick: async () => {
          const pending = collect(runB.streamSteps(sessionB, 'go'));
          await tick();
          markRefused();
          inner = (await pending).steps;
          return refusal;
        },
      }),
    });
    const outer = (await collect(runA.streamSteps(await sessionOf(h, 'run-a'), 'go'))).steps;
    const results = Object.fromEntries(ofType(outer, 'tool_result').map((step) => [step.toolCallId, step]));

    expect(ofType(inner, 'tool_result')[0]).toMatchObject({ toolCallId: 'call-b1', ok: true, result: refusal, refused: false });
    expect(results['call-a1']).toMatchObject({ ok: false, refused: true, error: { code: 'tool_call_refused', message: refusal } });
    expect(results['call-a2']).toMatchObject({ ok: true, result: 'probed', refused: false });
  });

  it('runs two refusing and non-refusing runs in parallel without either relabelling the other', async () => {
    const h = harness();
    const same = 'same result';
    const make = (refuse: boolean) => runtimeWith(scripted([toolStep([['c1', 'tick']]), FINAL]), {
      tools: tools({ tick: async () => { await tick(); if (refuse) markRefused(); await tick(); return same; } }),
    });
    const [refused, plain] = await Promise.all([
      collect(make(true).streamSteps(await sessionOf(h, 'p-a'), 'go')),
      collect(make(false).streamSteps(await sessionOf(h, 'p-b'), 'go')),
    ]);

    expect(ofType(refused.steps, 'tool_result')[0]).toMatchObject({ refused: true });
    expect(ofType(plain.steps, 'tool_result')[0]).toMatchObject({ ok: true, refused: false });
  });

  it('leaves nothing behind a finished or failed run for a later identical result', async () => {
    const h = harness();
    const refusing = tools({ tick: () => { markRefused(); return 'same'; } });
    await runtimeWith(scripted([toolStep([['c1', 'tick']]), FINAL]), { tools: refusing }).send(await sessionOf(h, 'first'), 'go');
    const failing = await sessionOf(h, 'failing');
    await expect(collect(runtimeWith(scripted([toolStep([['c1', 'tick']]), { text: '', finishReason: 'error' }]), { tools: refusing }).streamSteps(failing, 'go')))
      .rejects.toMatchObject({ code: 'model_step_failed' });
    const later = await collect(runtimeWith(scripted([toolStep([['c2', 'tick']]), FINAL]), { tools: tools({ tick: () => 'same' }) }).streamSteps(await sessionOf(h, 'later'), 'go'));

    expect((await failing.run())?.status).toBe('failed');
    expect(ofType(later.steps, 'tool_result')[0]).toMatchObject({ ok: true, refused: false, result: 'same' });
  });
});

describe('metadata-only context', () => {
  it('takes the same path and budget as no context; a run tree does change it', async () => {
    const calls: LlmRequest[] = [];
    const looping = async (request: LlmRequest): Promise<LlmResponse> => { calls.push(request); return toolStep([[`c${calls.length}`, 'probe']]); };
    const config: RuntimeConfig = { max_output_tokens_per_call: 64 };
    const counts: number[] = [];
    const runs: [string, RunContext | undefined, boolean][] = [
      ['bare', undefined, false],
      ['meta-stream', RunContext.metadata('turn-1', 'user:1'), true],
      ['meta-send', RunContext.metadata(null, 'user:1'), false],
      ['bare-stream', undefined, true],
      ['tree', RunContext.root('tree', new RunBudget(1)), false],
    ];
    let tagged: RunStep[] = [];
    for (const [scope, context, streamed] of runs) {
      const before = calls.length;
      const session = await sessionOf(harness(), scope);
      const rt = runtimeWith(looping, { config });
      if (streamed) {
        const { steps } = await collect(rt.streamSteps(session, 'go', undefined, context));
        if (scope === 'meta-stream') tagged = steps;
      } else await rt.send(session, 'go', undefined, context);
      counts.push(calls.length - before);
    }

    expect(counts).toEqual([4, 4, 4, 4, 1]);
    expect(new Set(calls.map((request) => request.maxOutputTokens))).toEqual(new Set([64]));
    expect(new Set(tagged.map((step) => step.turnId))).toEqual(new Set(['turn-1']));
    expect(tagged[0]!.rootRunId).toBe(tagged[0]!.runId);
    const meta = RunContext.metadata('t').withActor('a');
    expect([meta.metadataOnly, meta.turnId, meta.actor]).toEqual([true, 't', 'a']);
    expect(() => new RunContext(meta.ledger, meta.budget, null, 0, { actor: 7 as unknown as string })).toThrow(TypeError);
  });
});

// ---------------------------------------------------------------------------
// A+ (G-284): streamed approval continuation
// ---------------------------------------------------------------------------

interface Parked {
  h: PrismHarness;
  session: Session;
  parked: RunStep[];
  ran: { tick: number };
  rt: AgentRuntime;
  events: HarnessEvents;
  seen: HarnessEvent[];
  journal: string[];
}

/** A session parked on an approval for `tick`, streamed, with every durable write journalled. */
async function parkOnApproval(finals = 1, context: RunContext | undefined = RunContext.root('approval-root', new RunBudget(10))): Promise<Parked> {
  const journal: string[] = [];
  const durable = new DurableMemoryStore();
  const put = durable.put.bind(durable);
  durable.put = async (key, payload, ttl) => {
    const messages = Array.isArray(payload.messages) ? payload.messages as JsonObject[] : [];
    const last = messages.at(-1)?.message as JsonObject | undefined;
    if (last?.type === 'tool_result' && Array.isArray(last.tool_approval_responses) && last.tool_approval_responses.length > 0 && last.tool_results instanceof Array && last.tool_results.length === 0) {
      journal.push('decision-written');
    }
    await put(key, payload, ttl);
  };
  const h = harness(durable);
  const session = await sessionOf(h, 'approval', 'guarded');
  const ran = { tick: 0 };
  const { events, seen } = recording();
  events.listen((event) => { if (event.type === 'run.stepped' && event.step.type === 'decision_recorded') journal.push('decision-step'); });
  const responses: LlmResponse[] = [toolStep([['call-1', 'tick']]), ...Array.from({ length: finals }, () => FINAL)];
  const rt = runtimeWith(scripted(responses), {
    events, stepEvents: true,
    tools: tools({ tick: () => { ran.tick += 1; journal.push('tool-ran'); return 'ticked'; } }),
  });
  const { steps: parked } = await collect(rt.streamSteps(session, 'go', undefined, context));

  return { h, session, parked, ran, rt, events, seen, journal };
}

describe('streamDecision', () => {
  it('records every decision before any tool runs, reports it first, and continues the parked run', async () => {
    const { h, parked, ran, rt, journal } = await parkOnApproval();
    const request = ofType(parked, 'approval_request')[0]!;
    const parkedRunId = parked[0]!.runId;
    expect(Object.keys(request).slice(8)).toEqual(['approvalId', 'toolCallId']);
    expect(parked.map((step) => step.type)).toEqual(['step_start', 'text', 'tool_call', 'approval_request', 'step_finish']);
    expect(ran.tick).toBe(0);

    const context = RunContext.root('approval-root', new RunBudget(10)).withActor('user:42').withTurn('turn-3');
    const fresh = await sessionOf(h, 'approval', 'guarded');
    const { steps: rest } = await collect(rt.streamDecision(fresh, [{ approvalId: request.approvalId, approved: true, reason: 'looks fine' }], context));

    // The durable write, then the step that reports it, then the tool.
    expect(journal).toEqual(['decision-written', 'decision-step', 'tool-ran']);
    expect(JSON.parse(JSON.stringify(rest[0]))).toEqual({
      contractVersion: 'g282.1',
      runId: parkedRunId,
      parentRunId: null,
      rootRunId: 'approval-root',
      turnId: 'turn-3',
      seq: parked.length + 1,
      at: rest[0]!.at,
      type: 'decision_recorded',
      approvalId: request.approvalId,
      verdict: 'approved',
      reason: 'looks fine',
      actor: 'user:42',
    });
    const result = ofType(rest, 'tool_result')[0]!;
    expect(ran.tick).toBe(1);
    expect(rest.map((step) => step.seq)).toEqual(rest.map((_, index) => parked.length + 1 + index));
    expect(new Set(rest.map((step) => step.runId))).toEqual(new Set([parkedRunId]));
    expect(result).toMatchObject({ ok: true, result: 'ticked', refused: false, durationExact: false });
    expect(result.durationMs).toBeGreaterThanOrEqual(0);
    expect((await fresh.run())?.status).toBe('completed');
  });

  it('records the decisions in the same row shape as recordApproval()', async () => {
    const viaStream = await parkOnApproval();
    const viaSync = await parkOnApproval();
    const idOf = (p: Parked) => ofType(p.parked, 'approval_request')[0]!.approvalId;
    await collect(viaStream.rt.streamDecision(viaStream.session, [{ approvalId: idOf(viaStream), approved: false, reason: 'no' }]));
    await recordApproval(viaSync.session, idOf(viaSync), false, 'no');
    const decisionRow = async (p: Parked) => (await p.session.thread().messages())
      .map((entry) => entry.message).find((row) => row.type === 'tool_result' && (row.tool_results as unknown[]).length === 0);

    expect({ ...(await decisionRow(viaStream)), tool_approval_responses: [] }).toEqual({ ...(await decisionRow(viaSync)), tool_approval_responses: [] });
    expect((await decisionRow(viaStream))?.tool_approval_responses).toEqual([{ approval_id: idOf(viaStream), approved: false, reason: 'no' }]);
  });

  it('runs nothing twice when the same decision is streamed again', async () => {
    const { session, parked, ran, rt } = await parkOnApproval(2);
    const approvalId = ofType(parked, 'approval_request')[0]!.approvalId;

    await collect(rt.streamDecision(session, [{ approvalId, approved: true }]));
    const { steps: again } = await collect(rt.streamDecision(session, [{ approvalId, approved: true }]));

    expect(ran.tick).toBe(1);
    expect(again[0]!.type).toBe('decision_recorded');
    expect(ofType(again, 'tool_result')).toEqual([]);
  });

  it('reports a denied decision, and the call it denied as refused', async () => {
    const { session, parked, ran, rt } = await parkOnApproval();
    const approvalId = ofType(parked, 'approval_request')[0]!.approvalId;
    const { steps } = await collect(rt.streamDecision(session, [{ approvalId, approved: false, reason: 'not today' }]));
    const result = ofType(steps, 'tool_result')[0]!;

    expect(ran.tick).toBe(0);
    expect(Object.fromEntries(Object.entries(steps[0]!).slice(8))).toEqual({ approvalId, verdict: 'denied', reason: 'not today', actor: null });
    expect(result).toMatchObject({ ok: false, refused: true, error: { code: 'approval_denied', message: 'not today' } });
  });

  it('reports an approved call whose tool is no longer offered as refused, not run', async () => {
    const { session, parked, ran } = await parkOnApproval();
    const approvalId = ofType(parked, 'approval_request')[0]!.approvalId;
    const narrowed = new AgentRuntime({
      client: async () => FINAL,
      modes: new ModeRegistry({ default: 'guarded', modes: { guarded: { system_prompt: '', tools: ['probe'], max_steps: 4, requires_approval: ['tick'] } } }),
      tools: tools(),
    });
    const { steps } = await collect(narrowed.streamDecision(session, [{ approvalId, approved: true }]));

    expect(ran.tick).toBe(0);
    expect(ofType(steps, 'tool_result')[0]).toMatchObject({ ok: false, refused: true, error: { code: 'tool_not_available' } });
  });

  it('refuses a malformed or empty decision list when called, before recording anything', async () => {
    const { session, rt } = await parkOnApproval();
    const before = await session.thread().count();

    expect(() => rt.streamDecision(session, [])).toThrow(TypeError);
    expect(() => rt.streamDecision(session, [{ approvalId: '', approved: true }])).toThrow(TypeError);
    expect(() => rt.streamDecision(session, [{ approvalId: 'a', approved: 'yes' as unknown as boolean }])).toThrow(TypeError);
    expect(await session.thread().count()).toBe(before);
  });

  it('carries the actor from a metadata-only context onto a streamed decision', async () => {
    const { session, parked, ran, rt } = await parkOnApproval();
    const approvalId = ofType(parked, 'approval_request')[0]!.approvalId;
    const { steps } = await collect(rt.streamDecision(session, [{ approvalId, approved: true }], RunContext.metadata('turn-9', 'user:7')));

    expect(steps[0]).toMatchObject({ type: 'decision_recorded', actor: 'user:7', turnId: 'turn-9' });
    expect(ran.tick).toBe(1);
  });

  it('emits run.resumed, not a second run.started, when a parked run continues', async () => {
    const { session, parked, rt, seen } = await parkOnApproval();
    const approvalId = ofType(parked, 'approval_request')[0]!.approvalId;
    const parkedRunId = parked[0]!.runId;
    seen.length = 0;

    await collect(rt.streamDecision(session, [{ approvalId, approved: true }], RunContext.root('approval-root', new RunBudget(10))));
    const runEvents = seen.filter((event) => event.type !== 'run.stepped');

    expect(runEvents.map((event) => event.type)).toEqual(['run.resumed', 'run.finished']);
    expect(runEvents[0]).toMatchObject({ runId: parkedRunId, rootRunId: 'approval-root' });
    expect(runEvents[1]).toMatchObject({ runId: parkedRunId });
  });
});

describe('the session lock', () => {
  const noWait: RuntimeConfig = { lock_wait: 0 };

  async function heldElsewhere<T>(h: PrismHarness, scope: string, mode: string, during: () => Promise<T>): Promise<T> {
    const holder = await sessionOf(h, scope, mode);
    return holder.lock(during, 30, 0);
  }

  it('refuses streamSteps() while another holder has the session lock, calling no provider', async () => {
    const h = harness();
    const session = await sessionOf(h, 'locked');
    const calls: LlmRequest[] = [];
    const rt = runtimeWith(scripted([FINAL], calls), { config: noWait });

    await heldElsewhere(h, 'locked', 'chat', async () => {
      await expect(collect(rt.streamSteps(session, 'go'))).rejects.toMatchObject({ code: 'session_locked' });
    });

    expect(calls).toHaveLength(0);
    expect(await session.thread().count()).toBe(0);
    expect(await session.run()).toBeNull();
  });

  it('refuses streamDecision() while another holder has the lock, recording nothing and running nothing', async () => {
    const { h, session, parked, ran, rt } = await parkOnApproval();
    const approvalId = ofType(parked, 'approval_request')[0]!.approvalId;
    const before = await session.thread().count();
    const locked = new AgentRuntime({ client: async () => { throw new Error('no provider call may happen'); }, modes, tools: tools({ tick: () => { ran.tick += 1; return 'ticked'; } }), config: noWait });

    await heldElsewhere(h, 'approval', 'guarded', async () => {
      await expect(collect(locked.streamDecision(session, [{ approvalId, approved: true }]))).rejects.toMatchObject({ code: 'session_locked' });
    });

    expect(ran.tick).toBe(0);
    expect(await session.thread().count()).toBe(before);
    // The decision was never written, so it can still be made.
    expect((await collect(rt.streamDecision(session, [{ approvalId, approved: true }]))).steps[0]!.type).toBe('decision_recorded');
    expect(ran.tick).toBe(1);
  });

  it('holds the lock for the whole streamed turn, and releases it when the turn ends', async () => {
    const h = harness();
    const session = await sessionOf(h, 'held');
    let release!: () => void;
    const gate = new Promise<void>((resolve) => { release = resolve; });
    let observed: string | null = null;
    const rt = runtimeWith(scripted([toolStep([['c1', 'tick']]), FINAL]), {
      config: noWait,
      tools: tools({
        tick: async () => {
          // Mid-turn, another holder cannot get in.
          observed = await (await sessionOf(h, 'held')).lock(() => 'got in', 30, 0).catch((error: HarnessError) => error.code);
          release();
          return 'ticked';
        },
      }),
    });

    await collect(rt.streamSteps(session, 'go'));
    await gate;

    expect(observed).toBe('session_locked');
    expect(await (await sessionOf(h, 'held')).lock(() => 'free', 30, 0)).toBe('free');
  });

  it('releases the lock when the turn throws, and when the reader stops early', async () => {
    const h = harness();
    const failing = await sessionOf(h, 'throws');
    await expect(collect(runtimeWith(async () => ({ text: '', finishReason: 'error' }), { config: noWait }).streamSteps(failing, 'go')))
      .rejects.toBeInstanceOf(HarnessError);
    expect(await (await sessionOf(h, 'throws')).lock(() => 'free', 30, 0)).toBe('free');

    const early = await sessionOf(h, 'early');
    let finished!: () => void;
    const done = new Promise<void>((resolve) => { finished = resolve; });
    const events = new HarnessEvents();
    events.listen((event) => { if (event.type === 'run.finished') finished(); });
    const rt = runtimeWith(scripted([toolStep([['c1', 'tick']]), FINAL]), { config: noWait, events });
    for await (const step of rt.streamSteps(early, 'go')) {
      if (step.type === 'step_start') break;
    }
    // The run is not cancelled by a reader leaving (that is G-285): it reaches
    // its end, closes itself, and lets go of the lock.
    await done;
    await tick();
    expect((await early.run())?.status).toBe('completed');
    expect(await (await sessionOf(h, 'early')).lock(() => 'free', 30, 0)).toBe('free');
  });
});
