import { appendFileSync, readFileSync } from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { expect, it } from 'vitest';
import { AgentRuntime, FileSessionStore, HarnessError, ModeRegistry, PrismHarness, ToolAuthorizer, ToolRegistry } from '../src/index.js';

interface Observation { entry: string; requests: string[][]; error: { code: string } | null }
interface Case {
  id: string; title: string; names: string[] | null; expected: string[] | null;
  named: boolean; authorize: boolean; wildcardMode: boolean;
  result: { ts?: Record<string, Observation> };
}
const corpus = JSON.parse(readFileSync(new URL('./fixtures/harness-tool-list-default.json', import.meta.url), 'utf8')) as { cases: Case[] };
it('vendors all eleven shared authority cases', () => expect(corpus.cases).toHaveLength(11));

for (const entry of ['send', 'structured'] as const) it.each(corpus.cases)('$id native ' + entry + ': $title', async scenario => {
  const directory = await mkdtemp(join(tmpdir(), 'harness-tool-corpus-'));
  try {
    const store = new FileSessionStore(directory);
    const harness = new PrismHarness({ drivers: { owned: () => store }, stores: { ephemeral: 'owned', durable: 'owned' } });
    const session = (scenario.named ? harness.forAgent('support') : harness).for({ type: 'test', id: 'corpus' }).session('probe');
    const tools = new ToolRegistry();
    let handled = 0;
    for (const name of ['inside', 'second', 'outside']) {
      const tool = () => ({ name, handle: () => { handled++; throw new Error('Offering probes must not execute tools'); } });
      tools.register(tool()); tools.forAgent('support').register(tool(), 'support');
    }
    const ceiling = scenario.wildcardMode ? ['*'] : ['inside', 'second'];
    const modes = new ModeRegistry({ default: 'chat', modes: { chat: { tools: ceiling } } });
    modes.forAgent('support').register('chat', { tools: ceiling }, 'support').setDefault('chat');
    const requests: string[][] = [];
    const runtime = new AgentRuntime({ tools, modes,
      authorizer: scenario.authorize ? new ToolAuthorizer({ enabled: true, offer: (_session, tool) => tool.name === 'inside', call: () => true }) : undefined,
      client: async request => { requests.push(request.tools.map(tool => tool.name)); return { text: 'done', finishReason: 'stop', structured: { answer: 'done' }, costUsd: 0 }; },
    });
    let error: Observation['error'] = null;
    try {
      if (entry === 'structured') await runtime.sendStructured(session, 'probe', { type: 'object' }, scenario.names ?? undefined);
      else await runtime.send(session, 'probe', scenario.names ?? undefined);
    } catch (failure) {
      if (!(failure instanceof HarnessError)) throw failure;
      error = { code: failure.code };
    }
    const actual: Observation = { entry, requests, error };
    if (process.env.PRISM_TOOL_LIST_RECORDING) appendFileSync(process.env.PRISM_TOOL_LIST_RECORDING, JSON.stringify({ id: scenario.id, ...actual }) + '\n');
    else expect(actual).toEqual(scenario.result.ts?.[entry]);
    if (scenario.expected === null) {
      expect(error).toEqual({ code: 'tool_not_available' }); expect(requests).toEqual([]);
    } else {
      expect(error).toBeNull(); expect(requests).toEqual([scenario.expected]);
    }
    expect(handled).toBe(0);
  } finally { await rm(directory, { recursive: true, force: true }); }
});
