import { readFileSync } from 'node:fs';
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
  ToolPolicy,
  ToolRegistry,
  type HarnessEvent,
  type HarnessTool,
  type JsonObject,
  type LlmResponse,
  type ToolDecisionEvent,
  type ToolTier,
} from '../src/index.js';

/**
 * The cross-language tool-policy corpus (G-326).
 *
 * Every row's `expected` was written by hand before any language recorded.
 * This runner drives the real runtime with a fake model that asks for the
 * row's calls, then answers. Every tool needs approval by mode; a tool with
 * `provider` is supplied at resolve time by a provider registered under that
 * name, as an MCP bridge supplies one, and `server` on it is only its claim.
 */
interface PolicyCase {
  id: string;
  title: string;
  policy: unknown;
  tools: { name: string; server?: string; provider?: string; tier?: ToolTier }[];
  calls: { name: string; args: JsonObject }[];
  then: { policy: unknown } | null;
  expected: unknown;
}

const corpus = JSON.parse(
  readFileSync(new URL('./fixtures/harness-tool-policy.json', import.meta.url), 'utf8'),
) as { cases: PolicyCase[] };

async function record(row: PolicyCase): Promise<unknown> {
  const ran: string[] = [];
  const events: HarnessEvent[] = [];
  const registry = new ToolRegistry();
  const provided = new Map<string, HarnessTool[]>();
  for (const { provider, ...spec } of row.tools) {
    const tool: HarnessTool = { ...spec, handle: () => { ran.push(spec.name); return 'ok'; } };
    if (provider === undefined) registry.register(tool);
    else provided.set(provider, [...(provided.get(provider) ?? []), tool]);
  }
  for (const [name, tools] of provided) registry.registerProvider(() => tools, 'mcp', name);
  const modes = new ModeRegistry({
    default: 'guarded',
    modes: { guarded: { system_prompt: 'Careful.', tools: row.tools.map((tool) => tool.name), max_steps: 6, requires_approval: ['*'] } },
  });
  let step = 0;
  const client = async (): Promise<LlmResponse> => {
    step += 1;
    return step === 1
      ? { text: '', finishReason: 'tool_calls', toolCalls: row.calls.map((call, index) => ({ id: `c${index}`, name: call.name, arguments: call.args })) }
      : { text: 'done', finishReason: 'stop' };
  };
  const bus = new HarnessEvents();
  bus.listen((event) => { events.push(event); });

  let runtime: AgentRuntime;
  try {
    runtime = new AgentRuntime({ client, modes, tools: registry, events: bus, toolPolicy: ToolPolicy.fromJson(row.policy) });
  } catch (error) {
    if (error instanceof HarnessError && error.code === 'tool_policy_refused') return { refused: error.code, ran: [], asked: [], events: [] };
    throw error;
  }

  const directory = await mkdtemp(join(tmpdir(), 'prism-harness-tool-policy-corpus-'));
  const harness = new PrismHarness({
    drivers: { memory: () => new MemorySessionStore(), files: () => new FileSessionStore(directory) },
    stores: { ephemeral: 'memory', durable: 'files' },
  });
  const session = harness.for({ type: 'User', id: 1 }).session('corpus');
  await session.usingProvider('anthropic');
  await session.usingModel('fake');

  let response = await runtime.send(session, 'go');
  if (row.then !== null) {
    runtime.setToolPolicy(ToolPolicy.fromJson(row.then.policy));
    response = await runtime.send(session, '');
  }

  return {
    refused: null,
    ran,
    asked: response.pendingApprovals.map((pending) => pending.tool),
    events: events
      .filter((event): event is HarnessEvent & ToolDecisionEvent => event.type.startsWith('tool_'))
      .map((event) => ({
        type: event.type,
        server: event.server,
        tool: event.tool,
        args_digest: event.argsDigest,
        decided_by: event.decidedBy,
        policy_revision: event.policyRevision,
      })),
  };
}

describe('harness-tool-policy corpus', () => {
  it('has rows', () => {
    expect(corpus.cases.length).toBeGreaterThan(0);
  });

  for (const row of corpus.cases) {
    it(`${row.id}: ${row.title}`, async () => {
      expect(await record(row)).toEqual(row.expected);
    });
  }
});
