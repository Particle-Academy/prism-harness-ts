import { mkdtemp } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  FileSessionStore,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  ToolRegistry,
  recordApproval,
  type LlmResponse,
  type Session,
} from '../src/index.js';

/**
 * The step limit belongs to the RUN. A run that pauses for a person and is
 * resumed is the same run, so every resume continues the same count rather
 * than starting a new one. Entered through the public send()/recordApproval()
 * and streamDecision() surfaces, with a fake provider that always asks for a
 * gated tool.
 */
const MAX_STEPS = 3;
const APPROVALS = 6;

const modes = new ModeRegistry({
  default: 'guarded',
  modes: {
    guarded: { system_prompt: 'Careful.', tools: ['echo'], max_steps: MAX_STEPS, requires_approval: ['echo'] },
  },
});

function counting(): { calls: number; client: () => Promise<LlmResponse> } {
  const state = {
    calls: 0,
    client: async (): Promise<LlmResponse> => {
      state.calls += 1;
      return { text: '', finishReason: 'tool_calls', toolCalls: [{ id: `c${state.calls}`, name: 'echo', arguments: { n: state.calls } }] };
    },
  };
  return state;
}

function runtime(client: () => Promise<LlmResponse>): AgentRuntime {
  return new AgentRuntime({ client, modes, tools: new ToolRegistry().register({ name: 'echo', handle: () => 'ok' }) });
}

async function persisted(): Promise<() => Promise<Session>> {
  const directory = await mkdtemp(join(tmpdir(), 'prism-harness-step-budget-'));
  return async () => {
    // A fresh harness and store objects every time: only what is on disk survives.
    const harness = new PrismHarness({
      drivers: { files: () => new FileSessionStore(directory) },
      stores: { ephemeral: 'files', durable: 'files' },
    });
    const session = harness.for({ type: 'User', id: 1 }).session('support');
    await session.usingProvider('anthropic');
    await session.usingModel('claude-sonnet-4-5');
    return session;
  };
}

/** One harness and one session object for the whole test: the same worker resuming. */
async function inMemory(): Promise<() => Promise<Session>> {
  const directory = await mkdtemp(join(tmpdir(), 'prism-harness-step-budget-'));
  const harness = new PrismHarness({
    drivers: { memory: () => new MemorySessionStore(), files: () => new FileSessionStore(directory) },
    stores: { ephemeral: 'memory', durable: 'files' },
  });
  const session = harness.for({ type: 'User', id: 1 }).session('support');
  await session.usingProvider('anthropic');
  await session.usingModel('claude-sonnet-4-5');
  return async () => session;
}

describe('the step limit holds across approval pauses', () => {
  const network = vi.fn(() => { throw new Error('A test tried to use the network.'); });

  beforeEach(() => {
    network.mockClear();
    vi.stubGlobal('fetch', network);
  });

  afterEach(() => {
    expect(network).not.toHaveBeenCalled();
    vi.unstubAllGlobals();
  });

  it(`never takes more than max_steps model steps across ${APPROVALS} approvals in one runtime`, async () => {
    const open = await inMemory();
    const model = counting();
    const agent = runtime(model.client);

    let response = await agent.send(await open(), 'go');
    for (let index = 0; index < APPROVALS && response.finishReason === 'awaiting_approval'; index += 1) {
      await recordApproval(await open(), response.pendingApprovals[0]!.id, true);
      response = await agent.send(await open(), '');
    }

    expect(model.calls).toBe(MAX_STEPS);
    expect(response.finishReason).toBe('budget_exhausted');
    expect(response.steps).toBe(MAX_STEPS);
  });

  it(`never takes more than max_steps model steps across ${APPROVALS} approvals resumed from persisted state`, async () => {
    const open = await persisted();
    const model = counting();

    let response = await runtime(model.client).send(await open(), 'go');
    for (let index = 0; index < APPROVALS && response.finishReason === 'awaiting_approval'; index += 1) {
      await recordApproval(await open(), response.pendingApprovals[0]!.id, true);
      // A new runtime and a reloaded session: a different worker resuming.
      response = await runtime(model.client).send(await open(), '');
    }

    expect(model.calls).toBe(MAX_STEPS);
    expect(response.finishReason).toBe('budget_exhausted');
  });

  it(`holds through streamDecision() resumes too`, async () => {
    const open = await persisted();
    const model = counting();

    let response = await runtime(model.client).send(await open(), 'go');
    for (let index = 0; index < APPROVALS && response.finishReason === 'awaiting_approval'; index += 1) {
      const stream = runtime(model.client).streamDecision(await open(), [{ approvalId: response.pendingApprovals[0]!.id, approved: true }]);
      for (;;) {
        const next = await stream.next();
        if (next.done === true) { response = next.value; break; }
      }
    }

    expect(model.calls).toBe(MAX_STEPS);
    expect(response.finishReason).toBe('budget_exhausted');
  });

  it('still runs the last approved call when the budget is spent, and takes no further step', async () => {
    const open = await inMemory();
    const model = counting();
    const ran: number[] = [];
    const agent = new AgentRuntime({
      client: model.client,
      modes,
      tools: new ToolRegistry().register({ name: 'echo', handle: (args) => { ran.push(Number(args.n)); return 'ok'; } }),
    });

    let response = await agent.send(await open(), 'go');
    for (let index = 0; index < APPROVALS && response.finishReason === 'awaiting_approval'; index += 1) {
      await recordApproval(await open(), response.pendingApprovals[0]!.id, true);
      response = await agent.send(await open(), '');
    }

    expect(ran).toEqual([1, 2, 3]);
    expect(model.calls).toBe(MAX_STEPS);
  });

  it('gives a new prompt after a finished run a fresh budget', async () => {
    const open = await inMemory();
    const model = counting();
    const agent = runtime(model.client);

    let response = await agent.send(await open(), 'go');
    for (let index = 0; index < APPROVALS && response.finishReason === 'awaiting_approval'; index += 1) {
      await recordApproval(await open(), response.pendingApprovals[0]!.id, true);
      response = await agent.send(await open(), '');
    }
    expect(response.finishReason).toBe('budget_exhausted');

    const next = await agent.send(await open(), 'Another task');
    expect(next.finishReason).toBe('awaiting_approval');
    expect(model.calls).toBe(MAX_STEPS + 1);
  });
});
