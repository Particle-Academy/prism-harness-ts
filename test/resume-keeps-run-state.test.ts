import { mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AgentRuntime,
  FileSessionStore,
  HarnessError,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  RunBudget,
  RunContext,
  SubagentRunner,
  ToolRegistry,
  recordApproval,
  type JsonObject,
  type LlmRequest,
  type LlmResponse,
  type ModeRegistryConfig,
  type RuntimeConfig,
  type Session,
} from '../src/index.js';

// A run that pauses for a person and is resumed is the SAME run. Its step
// count already carries across the pause; so must what its agent has spent
// (G-321 E's spend ceiling), and, for a subagent's child run resumed outside
// the runner, the provider and model it was bound to. A resume that started
// these again would let a model reset its own allowance by asking for an
// approval.

beforeEach(() => {
  vi.stubGlobal('fetch', () => { throw new Error('network blocked in tests'); });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

interface World {
  runtime: AgentRuntime;
  harness: PrismHarness;
  registry: ModeRegistry;
  requests: { provider: string; model: string; systemPrompt: string }[];
}

/**
 * The fake model answers by mode (the system prompt is the mode's name): the
 * first call in `plan` delegates to `research`, the first call in any mode
 * listing `echo` calls it, and every other call answers in text.
 */
async function world(global: RuntimeConfig, agents: Record<string, RuntimeConfig & { modes: ModeRegistryConfig['modes'] }>, costs: readonly number[] = []): Promise<World> {
  const directory = await mkdtemp(join(tmpdir(), 'prism-harness-resume-state-'));
  const requests: World['requests'] = [];
  const agentModes: Record<string, ModeRegistryConfig> = {};
  const settings: Record<string, RuntimeConfig> = {};
  for (const [agent, { modes, ...rest }] of Object.entries(agents)) {
    agentModes[agent] = {
      owner: 'app',
      modes: Object.fromEntries(Object.entries(modes ?? {}).map(([name, mode]) => [name, {
        system_prompt: name, max_steps: 6, ...mode, tools: [...(mode.tools ?? []), ...Object.keys(mode.subagents ?? {})],
      }])),
    };
    settings[agent] = rest;
  }
  const registry = new ModeRegistry({ modes: {}, agents: agentModes });
  const tools = new ToolRegistry();
  const seen = new Map<string, number>();
  let call = 0;
  const runtime = new AgentRuntime({
    config: global,
    agents: settings,
    modes: registry,
    tools,
    client: async (request: LlmRequest): Promise<LlmResponse> => {
      requests.push({ provider: request.provider, model: request.model, systemPrompt: request.systemPrompt });
      const costUsd = costs[call++] ?? 0.001;
      const turn = seen.get(request.systemPrompt) ?? 0;
      seen.set(request.systemPrompt, turn + 1);
      if (turn === 0 && request.systemPrompt === 'plan') {
        return { text: '', finishReason: 'tool_calls', costUsd, toolCalls: [{ id: 'delegate', name: 'research', arguments: { task: 't' } }] };
      }
      if (turn === 0 && request.tools.some((tool) => tool.name === 'echo')) {
        return { text: '', finishReason: 'tool_calls', costUsd, toolCalls: [{ id: 'e1', name: 'echo', arguments: { value: 'x' } }] };
      }

      return { text: `answer ${call}`, finishReason: 'stop', costUsd };
    },
  });
  const harness = new PrismHarness({
    runtime,
    drivers: { memory: () => new MemorySessionStore(), files: () => new FileSessionStore(directory) },
    stores: { ephemeral: 'memory', durable: 'files' },
  });
  for (const agent of Object.keys(agents)) {
    tools.forAgent(agent).register({ name: 'echo', handle: () => 'echoed' }, 'app');
  }

  return { runtime, harness, registry, requests };
}

async function pendingId(session: Session): Promise<string> {
  const rows = (await session.thread().messages()).map((entry) => entry.message);
  const asked = [...rows].reverse().find((row) => row.type === 'assistant' && Array.isArray(row.tool_approval_requests) && row.tool_approval_requests.length > 0)!;

  return ((asked.tool_approval_requests as JsonObject[])[0]!).approval_id as string;
}

describe('an agent spend ceiling across an approval pause', () => {
  it('still holds after the person approves and the run resumes', async () => {
    const w = await world({ provider: 'anthropic', model: 'global-model' }, {
      planner: { spend_ceiling: 0.02, modes: { chat: { tools: ['echo'], requires_approval: ['echo'] } } },
    }, [0.03]);
    const session = w.harness.pending({ type: 'User', id: 1 }, 'planner').session('support');
    await session.usingMode('chat');

    // Under the ceiling when it starts: the call lands, is charged 0.03, and asks.
    const paused = await w.runtime.send(session, 'go');
    expect(paused.finishReason).toBe('awaiting_approval');

    await recordApproval(session, await pendingId(session), true);
    const failure = await w.runtime.send(session, '').then(() => null, (error: unknown) => error);

    expect((failure as HarnessError).code).toBe('agent_spend_ceiling_exceeded');
    expect(w.requests).toHaveLength(1);
  });
});

describe('a subagent child run resumed outside the runner', () => {
  async function pausedChild(global: RuntimeConfig, researcher: RuntimeConfig, costs: readonly number[] = []) {
    const w = await world(global, {
      planner: { modes: { plan: { agents: ['researcher'], subagents: { research: { mode: 'research', agent: 'researcher', provider: 'custom-cheap', model: 'cheap-model' } } } } },
      researcher: { ...researcher, model: 'researcher-default', modes: { research: { tools: ['echo'], requires_approval: ['echo'] } } },
    }, costs);
    const parent = w.harness.pending({ type: 'User', id: 1 }, 'planner').session('support');
    await parent.usingMode('plan');
    const runner = new SubagentRunner(w.harness, w.runtime);
    const context = RunContext.root('root', new RunBudget(8));
    const subagent = w.registry.forAgent('planner').resolve('plan').subagents.research!;

    await runner.run(subagent, parent, context, 'root', 'look it up');
    const child = w.harness.pending(parent.participant, 'researcher', parent.actor).session(subagent.scopeUnder(parent.scope));

    return { w, child };
  }

  it('keeps the provider and model it was bound to', async () => {
    const { w, child } = await pausedChild({ provider: 'anthropic', model: 'global-model' }, {});
    expect(w.requests.map((seen) => [seen.provider, seen.model])).toEqual([['custom-cheap', 'cheap-model']]);

    await recordApproval(child, await pendingId(child), true);
    await w.runtime.send(child, '');

    expect(w.requests.map((seen) => [seen.provider, seen.model])).toEqual([
      ['custom-cheap', 'cheap-model'],
      ['custom-cheap', 'cheap-model'],
    ]);
  });

  it('keeps its agent spend ceiling', async () => {
    const { w, child } = await pausedChild({ provider: 'anthropic', model: 'global-model' }, { spend_ceiling: 0.02 }, [0.03]);

    await recordApproval(child, await pendingId(child), true);
    const failure = await w.runtime.send(child, '').then(() => null, (error: unknown) => error);

    expect((failure as HarnessError).code).toBe('agent_spend_ceiling_exceeded');
    expect(w.requests).toHaveLength(1);
  });
});
