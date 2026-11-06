import { mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { expect, it } from 'vitest';
import { AgentRuntime, FileSessionStore, ModeRegistry, PrismHarness, ToolAuthorizer, ToolRegistry } from '../src/index.js';

interface Scenario {
  title: string;
  names?: readonly string[];
  expected: string[] | null;
  flags?: Record<string, unknown>;
  named?: boolean;
  authorize?: boolean;
  wildcardMode?: boolean;
}

const cases: Scenario[] = [
  { title: 'default refuses a registered tool outside the mode', names: ['outside'], expected: null },
  { title: 'default refuses a mixed allowed and foreign list', names: ['inside', 'outside'], expected: null },
  { title: 'a subset narrows', names: ['inside'], expected: ['inside'] },
  { title: 'omitting names offers the full mode', expected: ['inside', 'second'] },
  { title: 'an empty list offers none', names: [], expected: [] },
  { title: 'requested wildcard retains the ceiling', names: ['*'], expected: ['inside', 'second'] },
  { title: 'wildcard plus foreign name cannot widen', names: ['*', 'outside'], expected: ['inside', 'second'] },
  { title: 'an unrestricted mode can select a registered tool', names: ['outside'], wildcardMode: true, expected: ['outside'] },
  { title: 'explicit widening opt-out restores replacement', names: ['outside'], flags: { allowLegacyToolListWidening: true }, expected: ['outside'] },
  { title: 'authorizer filters the default mode list', authorize: true, expected: ['inside'] },
  { title: 'authorizer still filters with the widening opt-out', names: ['outside'], flags: { allowLegacyToolListWidening: true }, authorize: true, expected: [] },
  { title: 'named agents ignore the legacy widening opt-out', names: ['outside'], flags: { allowLegacyToolListWidening: true }, named: true, expected: null },
  { title: 'named subsets remain valid with the legacy option', names: ['inside'], flags: { allowLegacyToolListWidening: true }, named: true, expected: ['inside'] },
];

for (const shape of ['send', 'structured'] as const) {
  it.each(cases)('$title (' + shape + ')', async scenario => {
    const store = new FileSessionStore(await mkdtemp(join(tmpdir(), 'harness-default-tools-')));
    const harness = new PrismHarness({ drivers: { owned: () => store }, stores: { ephemeral: 'owned', durable: 'owned' } });
    const session = (scenario.named ? harness.forAgent('support') : harness).for({ type: 'test', id: 'default-tools' }).session('probe');
    let handled = 0;
    const tool = (name: string) => ({ name, handle: () => { handled++; throw new Error('No tool should execute in an offering probe'); } });
    const tools = new ToolRegistry();
    for (const name of ['inside', 'second', 'outside']) {
      tools.register(tool(name));
      tools.forAgent('support').register(tool(name), 'support');
    }
    const modeTools = scenario.wildcardMode ? ['*'] : ['inside', 'second'];
    const modes = new ModeRegistry({ default: 'chat', modes: { chat: { tools: modeTools } } });
    modes.forAgent('support').register('chat', { tools: modeTools }, 'support').setDefault('chat');
    const requests: string[][] = [];
    const options = {
      tools, modes, ...scenario.flags,
      authorizer: scenario.authorize ? new ToolAuthorizer({ enabled: true, offer: (_session, offered) => offered.name === 'inside', call: () => true }) : undefined,
      client: async (request: { tools: readonly { name: string }[] }) => {
        requests.push(request.tools.map(offered => offered.name));
        return { text: 'done', finishReason: 'stop', structured: { answer: 'done' }, costUsd: 0 };
      },
    };
    const runtime = new AgentRuntime(options);
    const turn = shape === 'structured'
      ? runtime.sendStructured(session, 'probe', { type: 'object' }, scenario.names)
      : runtime.send(session, 'probe', scenario.names);
    if (scenario.expected === null) {
      await expect(turn).rejects.toMatchObject({ code: 'tool_not_available' });
      expect(requests).toEqual([]);
    } else {
      await turn;
      expect(requests).toEqual([scenario.expected]);
    }
    expect(handled).toBe(0);
  });
}

for (const flags of [
  { narrowRequestedTools: false },
  { narrowRequestedTools: false, allowLegacyToolListWidening: true },
  { narrowRequestedTools: true, allowLegacyToolListWidening: true },
]) it('rejects ambiguous/obsolete authority configuration ' + JSON.stringify(flags), () => {
  expect(() => new AgentRuntime({ modes: new ModeRegistry({}), tools: new ToolRegistry(), client: async () => ({ text: 'done', finishReason: 'stop' }), ...flags })).toThrow(/allowLegacyToolListWidening/);
});

for (const value of [null, 1, 'true', [], {}]) it('refuses a nonboolean widening option ' + JSON.stringify(value), () => {
  const options = { modes: new ModeRegistry({}), tools: new ToolRegistry(), client: async () => ({ text: 'done', finishReason: 'stop' }), ...{ allowLegacyToolListWidening: value } };
  // @ts-expect-error Untrusted runtime configuration intentionally violates the public boolean type.
  expect(() => new AgentRuntime(options)).toThrow(/allowLegacyToolListWidening/);
});
