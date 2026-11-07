import { appendFileSync, readFileSync } from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { expect, it } from 'vitest';
import { AgentRuntime, FileSessionStore, HarnessError, ModeRegistry, PrismHarness, ToolAuthorizer, ToolRegistry, recordApproval, type LlmResponse } from '../src/index.js';

interface Observation { entry: string; requests: string[][]; error: { code: string } | null; handlers?: {inside: number; second: number} }
interface Case {
  id: string; title: string; names: string[] | null; expected: string[] | null;
  named: boolean; authorize: boolean; wildcardMode: boolean;
  resumeApproval?: boolean; expectedHandlers?: {inside: number; second: number};
  result: { ts?: Record<string, Observation> };
}
const corpus = JSON.parse(readFileSync(new URL('./fixtures/harness-tool-list-default.json', import.meta.url), 'utf8')) as { cases: Case[] };
it('vendors all fourteen shared authority cases', () => expect(corpus.cases).toHaveLength(14));

for (const entry of ['send', 'structured'] as const) it.each(corpus.cases)('$id native ' + entry + ': $title', async scenario => {
  if (scenario.resumeApproval !== undefined) {
    await approvalResume(scenario, entry);
    return;
  }
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

async function approvalResume(scenario: Case, entry: 'send' | 'structured') {
  const directory = await mkdtemp(join(tmpdir(), 'harness-approval-corpus-'));
  try {
    const store = new FileSessionStore(directory);
    const harness = new PrismHarness({drivers: {owned: () => store}, stores: {ephemeral: 'owned', durable: 'owned'}});
    const open = () => harness.for({type: 'test', id: 'corpus'}).session('probe');
    const session = open();
    const handlers = {inside: 0, second: 0};
    const tools = new ToolRegistry();
    for (const name of ['inside', 'second'] as const) tools.register({name, handle: () => {handlers[name]++; return 'done';}});
    const modes = new ModeRegistry({default: 'chat', modes: {chat: {tools: ['inside', 'second'], requires_approval: ['inside'], max_steps: 4}}});
    const requests: string[][] = [];
    const responses: LlmResponse[] = [
      {text: '', finishReason: 'tool_calls', toolCalls: [{id: 'inside-call', name: 'inside', arguments: {}}]},
      {text: '', finishReason: 'tool_calls', toolCalls: [{id: 'second-call', name: 'second', arguments: {}}]},
      {text: '{"answer":"done"}', finishReason: 'stop', structured: {answer: 'done'}},
    ];
    const runtime = () => new AgentRuntime({tools, modes, client: async request => {
      requests.push(request.tools.map(tool => tool.name));
      expect(JSON.stringify(request.messages)).not.toContain('harness_tool_names');
      const response = responses.shift();
      if (!response) throw new Error('Unexpected extra native model call');
      return response;
    }});
    const first = await runtime().send(session, 'probe', scenario.names ?? undefined);
    expect(first.pendingApprovals).toHaveLength(1);
    expect(handlers).toEqual({inside: 0, second: 0});
    const fresh = open();
    expect(fresh).not.toBe(session); expect(fresh.key()).toBe(session.key());
    await recordApproval(fresh, first.pendingApprovals[0]!.id, scenario.resumeApproval!);
    let error: Observation['error'] = null;
    try {
      if (entry === 'structured') await runtime().sendStructured(fresh, '', {type: 'object'});
      else await runtime().send(fresh, '');
    } catch (failure) {
      if (!(failure instanceof HarnessError)) throw failure;
      error = {code: failure.code};
    }
    expect(handlers).toEqual(scenario.expectedHandlers);
    expect(requests.length).toBeGreaterThanOrEqual(2);
    for (const offer of requests) expect(offer).toEqual(scenario.expected);
    expect(error).toEqual(scenario.expected?.includes('second') ? null : {code: 'tool_not_available'});
    const actual: Observation = {entry, requests, error, handlers};
    if (process.env.PRISM_TOOL_LIST_RECORDING) appendFileSync(process.env.PRISM_TOOL_LIST_RECORDING, JSON.stringify({id: scenario.id, ...actual}) + '\n');
    else expect(actual).toEqual(scenario.result.ts?.[entry]);
  } finally {
    expect(directory.startsWith(join(tmpdir(), 'harness-approval-corpus-'))).toBe(true);
    await rm(directory, {recursive: true, force: true});
  }
}
