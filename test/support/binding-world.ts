import { mkdtemp } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import {
  AgentRuntime,
  FileSessionStore,
  HarnessEvents,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  RunBudget,
  RunContext,
  SubagentRunner,
  ToolRegistry,
  type HarnessEvent,
  type LlmRequest,
  type LlmResponse,
  type ModeRegistryConfig,
  type RuntimeConfig,
  type Session,
} from '../../src/index.js';

export interface BindingSpec {
  global: RuntimeConfig;
  /** Legacy-namespace modes. */
  modes?: ModeRegistryConfig['modes'];
  /** Per agent: runtime settings plus that agent's modes. */
  agents?: Record<string, RuntimeConfig & { modes?: ModeRegistryConfig['modes'] }>;
}

export interface RunSpec {
  agent: string | null;
  mode: string;
  session_provider?: string | null;
  session_model?: string | null;
  /** The subagent the parent's first step calls, if any. */
  delegate?: string | null;
  /** What each request reports as its cost, by call index; default 0.01. */
  costs?: readonly (number | null)[];
}

export interface Seen {
  provider: string;
  model: string;
  systemPrompt: string;
}

/**
 * A real harness, runtime and runner, wired the way an application wires a
 * subagent in this port: the mode lists the subagent's name as a tool, and the
 * tool is the runner's, bound to the run's root context.
 *
 * The fake client RECORDS what it was asked for. Its first answer calls the
 * `delegate` subagent when one is set; every other call answers in text.
 */
export async function bindingWorld(spec: BindingSpec, run: RunSpec): Promise<{
  requests: Seen[];
  events: HarnessEvent[];
  send: () => Promise<{ text: string }>;
  parent: Session;
  context: RunContext;
}> {
  const directory = await mkdtemp(join(tmpdir(), 'prism-harness-binding-'));
  const requests: Seen[] = [];
  const events: HarnessEvent[] = [];
  const bus = new HarnessEvents();
  bus.listen((event) => events.push(event));

  const withPrompts = (modes: ModeRegistryConfig['modes'] = {}): ModeRegistryConfig['modes'] =>
    Object.fromEntries(Object.entries(modes).map(([name, mode]) => [name, {
      system_prompt: name, max_steps: 4, ...mode,
      // The subagent is offered as a TOOL of the same name, as an app wires it.
      tools: [...(mode.tools ?? []), ...Object.keys(mode.subagents ?? {})],
    }]));

  const agentModes: Record<string, ModeRegistryConfig> = {};
  const agentSettings: Record<string, RuntimeConfig> = {};
  for (const [agent, { modes, ...settings }] of Object.entries(spec.agents ?? {})) {
    agentModes[agent] = { owner: 'app', modes: withPrompts(modes) };
    agentSettings[agent] = settings;
  }

  const registry = new ModeRegistry({ modes: withPrompts(spec.modes), agents: agentModes });
  const tools = new ToolRegistry();
  let call = 0;
  const runtime = new AgentRuntime({
    config: spec.global,
    agents: agentSettings,
    modes: registry,
    tools,
    events: bus,
    client: async (request: LlmRequest): Promise<LlmResponse> => {
      const index = call++;
      requests.push({ provider: request.provider, model: request.model, systemPrompt: request.systemPrompt });
      const costUsd = run.costs === undefined ? 0.01 : (run.costs[index] ?? run.costs[run.costs.length - 1] ?? null);
      if (index === 0 && run.delegate) {
        return {
          text: '', finishReason: 'tool_calls', costUsd,
          toolCalls: [{ id: 'delegate-call', name: run.delegate, arguments: { task: 't' } }],
        };
      }

      return { text: `answer ${index}`, finishReason: 'stop', costUsd };
    },
  });
  const harness = new PrismHarness({
    runtime,
    drivers: { memory: () => new MemorySessionStore(), files: () => new FileSessionStore(directory) },
    stores: { ephemeral: 'memory', durable: 'files' },
  });
  const runner = new SubagentRunner(harness, runtime);
  const context = RunContext.root('root', new RunBudget(8));
  const parent = harness.pending({ type: 'User', id: 1 }, run.agent).session('support');
  await parent.usingMode(run.mode);
  if (run.session_provider) await parent.usingProvider(run.session_provider);
  if (run.session_model) await parent.usingModel(run.session_model);

  if (run.delegate) {
    const delegate = run.delegate;
    // Resolved when the model calls it, so a refused declaration surfaces
    // from send() itself, as it would for any application.
    tools.forAgent(run.agent).register({
      name: delegate,
      handle: async (args) => {
        const subagent = registry.forAgent(run.agent).resolve(run.mode).subagents[delegate]!;

        return await runner.tool(subagent, parent, context, 'root').handle(args);
      },
    }, 'app');
  }

  return {
    requests,
    events,
    parent,
    context,
    send: async () => await runtime.send(parent, 'go', undefined, context),
  };
}
