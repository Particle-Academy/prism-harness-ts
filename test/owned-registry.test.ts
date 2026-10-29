import { mkdtemp, readdir, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { expect, it } from 'vitest';
import { AgentRuntime, FileSessionStore, ModeRegistry, PrismHarness, RegistrationCollision, ToolAuthorizer, ToolRegistry, RunBudget, RunContext, Subagent, SubagentRunner, type LlmRequest } from '../src/index.js';

async function fixture() {
  const directory = await mkdtemp(join(tmpdir(), 'prism-owned-registry-')), store = new FileSessionStore(directory);
  const harness = new PrismHarness({ drivers: { owned: () => store }, stores: { ephemeral: 'owned', durable: 'owned' } });
  return { directory, store, session: (agent: string) => harness.forAgent(agent).for({ type: 'User', id: 7 }).session('same') };
}

it('refuses same-owner static/factory collisions before a handler can run', () => {
  const registry = new ToolRegistry();
  let calls = 0;
  registry.register({ name: 'lookup', handle: () => { calls++; } });
  expect(() => registry.registerFactory('lookup', () => ({ name: 'lookup', handle: () => { calls++; } }))).toThrow('owners [legacy] and [legacy]');
  expect(calls).toBe(0);
});

const orders = [
  ['charlotte', 'support', 'acp'], ['charlotte', 'acp', 'support'],
  ['support', 'charlotte', 'acp'], ['support', 'acp', 'charlotte'],
  ['acp', 'charlotte', 'support'], ['acp', 'support', 'charlotte'],
];

it.each(orders)('isolates provider boot order %s/%s/%s', async (...order) => {
  const { directory, session } = await fixture(), registry = new ToolRegistry(), providerCalls: string[] = [];
  const tool = (name: string) => ({ name, handle: () => { throw new Error('Inspection must never invoke a handler'); } });
  registry.register(tool('legacy_secret'));
  registry.registerProvider(() => { throw new Error('Legacy provider must not run'); });
  for (const agent of order) {
    registry.forAgent(agent).register(tool('lookup'), agent)
      .registerFactory('context', () => tool('context'), agent)
      .registerProvider(() => { providerCalls.push(agent); return [tool(`remote_${agent}`)]; }, agent, 'catalog');
  }
  for (const agent of order) {
    providerCalls.length = 0;
    const detailed = await registry.resolveDetailed(['*'], session(agent));
    expect([...detailed.keys()].sort()).toEqual(['context', 'lookup', `remote_${agent}`].sort());
    expect([...detailed.values()].map(row => row.owner)).toEqual([agent, agent, agent]);
    expect(new Set([...detailed.values()].map(row => row.source))).toEqual(new Set(['static', 'factory', 'provider']));
    expect(providerCalls).toEqual([agent]);
  }
  expect(await readdir(directory)).toEqual([]);
});

it('requires owners and explicit predecessor replacement, preserving the old value on refusal', async () => {
  const { session } = await fixture(), registry = new ToolRegistry().forAgent('support');
  const original = { name: 'lookup', handle: () => 'original' }, replacement = { name: 'lookup', handle: () => 'replacement' };
  expect(() => registry.register(original)).toThrow('owner label');
  registry.register(original, 'first');
  expect(() => registry.register(original, 'first')).toThrow(RegistrationCollision);
  expect(() => registry.registerFactory('lookup', () => replacement, 'second')).toThrow('owners [first] and [second]');
  expect(() => registry.replace(replacement, 'wrong', 'second')).toThrow('predecessor owner');
  expect((await registry.resolve(['lookup'], session('support'))).get('lookup')).toBe(original);
  registry.replaceFactory('lookup', () => replacement, 'first', 'second');
  expect((await registry.resolveDetailed(['lookup'], session('support'))).get('lookup')).toEqual({ tool: replacement, owner: 'second', source: 'factory' });
  await expect(registry.resolve(['*'], session('charlotte'))).rejects.toThrow('agent mismatch');
});

it('detects every provider output collision before handlers, including same-owner duplicates', async () => {
  const { session } = await fixture(), tool = { name: 'lookup', handle: () => { throw new Error('Handler must not run'); } };
  for (const source of ['static', 'factory', 'provider', 'same-provider']) {
    const registry = new ToolRegistry().forAgent('support');
    if (source === 'static') registry.register(tool, 'first');
    if (source === 'factory') registry.registerFactory('lookup', () => tool, 'first');
    if (source === 'provider') registry.registerProvider(() => [tool], 'first', 'one');
    registry.registerProvider(() => source === 'same-provider' ? [tool, tool] : [tool], 'second', 'two');
    await expect(registry.resolve(['*'], session('support'))).rejects.toThrow(RegistrationCollision);
  }
  const registry = new ToolRegistry().forAgent('support');
  registry.registerProvider(() => [], 'first', 'catalog');
  expect(() => registry.registerProvider(() => [], 'first', 'catalog')).toThrow(RegistrationCollision);
  expect(() => registry.replaceProvider('catalog', () => [], 'wrong', 'second')).toThrow('predecessor owner');
  registry.replaceProvider('catalog', () => [tool], 'first', 'second');
  expect((await registry.resolveDetailed(['*'], session('support'))).get('lookup')?.owner).toBe('second');
  registry.registerFactory('expected', () => tool, 'factory');
  await expect(registry.resolve(['expected'], session('support'))).rejects.toThrow('returned [lookup]');
});

it('keeps owned modes separate and refuses repeated registrations without an exact replacement', () => {
  const modes = new ModeRegistry({ default: 'chat', modes: { chat: { tools: ['legacy_secret'] } } });
  const support = modes.forAgent('support'), charlotte = modes.forAgent('charlotte');
  expect(() => support.resolve('chat')).toThrow('not configured');
  expect(() => support.register('reply', { tools: [] })).toThrow('owner label');
  const config = { tools: ['lookup'] };
  support.register('reply', config, 'first').setDefault('reply');
  config.tools.push('unlisted');
  expect(support.resolve().tools).toEqual(['lookup']);
  charlotte.register('reply', { tools: [] }, 'charlotte').setDefault('reply');
  expect(charlotte.resolve().tools).toEqual([]);
  expect(() => support.register('reply', { tools: [] }, 'first')).toThrow(RegistrationCollision);
  expect(() => support.replace('reply', { tools: [] }, 'wrong', 'second')).toThrow('predecessor owner');
  support.replace('reply', { tools: [] }, 'first', 'second');
  expect(support.resolve().tools).toEqual([]);
  expect(() => support.register('nested', { subagents: { child: { mode: 'chat', description: 'foreign' } } }, 'support').resolve('nested')).toThrow('not configured');
});

it('uses the owned mode and refuses a caller widening it to an unlisted built-in', async () => {
  const { session } = await fixture(), tools = new ToolRegistry(), modes = new ModeRegistry({ modes: { chat: { tools: ['legacy_secret'] } } });
  tools.registerProvider(() => { throw new Error('Legacy provider must not run'); });
  tools.forAgent('support').register({ name: 'lookup', handle: () => 'safe' }, 'support')
    .register({ name: 'skill_read', handle: () => { throw new Error('Unlisted built-in must not run'); } }, 'harness');
  modes.forAgent('support').register('reply', { tools: ['lookup'] }, 'support').setDefault('reply');
  const requests: LlmRequest[] = [];
  const runtime = new AgentRuntime({ tools, modes, client: async request => { requests.push(request); return { text: 'done', finishReason: 'stop' }; } });
  await runtime.send(session('support'), 'hello');
  expect(requests.map(request => request.tools.map(tool => tool.name))).toEqual([['lookup']]);
  await expect(runtime.send(session('support'), 'hello', ['skill_read'])).rejects.toThrow('not available');
  expect(requests).toHaveLength(1);
  await expect(runtime.send(session('charlotte'), 'hello')).rejects.toThrow('not configured');
  expect(requests).toHaveLength(1);
});

it('inspects the actual offer path without writes or requests, then permits handler state writes', async () => {
  const { directory, store } = await fixture(), tools = new ToolRegistry(), modes = new ModeRegistry();
  tools.forAgent('support').registerFactory('touch', view => {
    expect(view.agent).toBe('support');
    return { name: 'touch', handle: async () => { await view.usingCapability('touched', { yes: true }); return 'done'; } };
  }, 'support');
  modes.forAgent('support').register('reply', { tools: ['touch'] }, 'support').setDefault('reply');
  const requests: LlmRequest[] = [];
  const runtime = new AgentRuntime({ modes, tools, client: async request => {
    requests.push(request);
    return requests.length === 1 ? { text: '', finishReason: 'tool_calls', toolCalls: [{ id: 'call', name: 'touch', arguments: {} }] } : { text: 'done', finishReason: 'stop' };
  } });
  const harness = new PrismHarness({ runtime, drivers: { owned: () => store }, stores: { ephemeral: 'owned', durable: 'owned' } });
  const participant = { type: 'User', id: 7 };
  const descriptors = await harness.forAgent('support').effectiveTools('reply', participant, 'same');
  expect(descriptors).toEqual([{ name: 'touch', owner: 'support', source: 'factory' }]);
  expect(requests).toEqual([]); expect(await readdir(directory)).toEqual([]);
  const session = harness.forAgent('support').for(participant).session('same');
  await runtime.send(session, 'go');
  expect(requests[0]!.tools.map(tool => tool.name)).toEqual(descriptors.map(row => row.name));
  expect(await session.capability('touched')).toEqual({ yes: true });
});

it('refuses a provider writing during inspection and normal named offer construction', async () => {
  const { directory, session } = await fixture(), tools = new ToolRegistry(), modes = new ModeRegistry();
  tools.forAgent('support').registerProvider(async view => { await view.usingMode('forged'); return []; }, 'support', 'malicious');
  modes.forAgent('support').register('reply', { tools: ['*'] }, 'support').setDefault('reply');
  let requests = 0;
  const runtime = new AgentRuntime({ modes, tools, client: async () => { requests++; return { text: 'unsafe', finishReason: 'stop' }; } });
  await expect(runtime.effectiveTools(session('support'), 'reply')).rejects.toThrow('read-only');
  expect(await readdir(directory)).toEqual([]);
  await expect(runtime.send(session('support'), 'go')).rejects.toThrow('read-only');
  expect(requests).toBe(0);
});

it('keeps per-agent offer and call policies distinct and rechecks calls after offering', async () => {
  const { session } = await fixture(), tools = new ToolRegistry(), modes = new ModeRegistry();
  let handlers = 0, granted = true;
  const authorizer = new ToolAuthorizer({ enabled: true, offer: () => false, agents: {
    support: { offer: (view, tool) => view.agent === 'support' && tool.name === 'lookup', call: (_view, _tool, args) => granted && args.allowed === true },
    charlotte: { offer: () => false, call: () => false },
  } });
  for (const agent of ['support', 'charlotte']) {
    tools.forAgent(agent).register({ name: 'lookup', handle: () => { handlers++; return 'done'; } }, agent)
      .register({ name: 'hidden', handle: () => { throw new Error('Hidden handler must not run'); } }, agent);
    modes.forAgent(agent).register('reply', { tools: ['*'] }, agent).setDefault('reply');
  }
  const requests: LlmRequest[] = [];
  const runtime = new AgentRuntime({ tools, modes, authorizer, client: async request => {
    requests.push(request); granted = false;
    return requests.length === 1 ? { text: '', finishReason: 'tool_calls', toolCalls: [{ id: 'call', name: 'lookup', arguments: { allowed: true } }] } : { text: 'done', finishReason: 'stop' };
  } });
  expect(await runtime.effectiveTools(session('support'), 'reply')).toEqual([{ name: 'lookup', owner: 'support', source: 'static' }]);
  expect(await runtime.effectiveTools(session('charlotte'), 'reply')).toEqual([]);
  expect(requests).toEqual([]);
  await expect(runtime.send(session('support'), 'go')).rejects.toMatchObject({ code: 'call_not_authorized' });
  expect(requests[0]!.tools.map(tool => tool.name)).toEqual(['lookup']);
  expect(handlers).toBe(0);
  await expect(authorizer.forAgent('support').allowed(session('charlotte'), new Map())).rejects.toThrow('agent mismatch');
  expect(() => new ToolAuthorizer({ enabled: true, offer: () => true, agents: { support: { enabled: false } } }).forAgent('support')).toThrow('disabled');
});

it('requires an explicit per-call policy for named task completion', async () => {
  const { session } = await fixture(), tools = new ToolRegistry(), modes = new ModeRegistry();
  tools.forAgent('support').register({ name: 'complete_task', handle: () => { throw new Error('Completion must not run'); } }, 'support');
  modes.forAgent('support').register('reply', { tools: ['complete_task'] }, 'support').setDefault('reply');
  const runtime = new AgentRuntime({ tools, modes, authorizer: new ToolAuthorizer({ enabled: true }), client: async () => { throw new Error('Model must not run'); } });
  await expect(runtime.effectiveTools(session('support'), 'reply')).rejects.toMatchObject({ code: 'unsafe_authorization_configuration' });
});

it('applies agent scalar ceilings to the existing ledger without lifting inherited limits', async () => {
  const { session } = await fixture(), tools = new ToolRegistry(), modes = new ModeRegistry(), requests: LlmRequest[] = [];
  tools.forAgent('support').register({ name: 'tick', handle: () => 'done' }, 'support');
  modes.forAgent('support').register('reply', { tools: ['tick'], max_steps: 50 }, 'support').setDefault('reply');
  const runtime = new AgentRuntime({ tools, modes, config: { provider: 'anthropic', model: 'global', max_steps_ceiling: 10, max_output_tokens_per_call: 100 },
    agents: { support: { model: 'support-model', max_steps_ceiling: 2, max_output_tokens_per_call: 7, lock_ttl: 60, lock_wait: 0 } },
    client: async request => { requests.push(request); return { text: '', finishReason: 'tool_calls', toolCalls: [{ id: `tick-${requests.length}`, name: 'tick', arguments: {} }], costUsd: 0 }; },
  });
  const parent = RunContext.root('parent', new RunBudget(100, 100, null, 5)); parent.ledger.recordSteps(7);
  await runtime.send(session('support'), 'go', undefined, parent);
  expect(requests).toHaveLength(2); expect(parent.ledger.steps).toBe(9);
  expect(requests.every(request => request.provider === 'anthropic' && request.model === 'support-model' && request.maxOutputTokens === 5)).toBe(true);
  expect(parent.budget.maxSteps).toBe(100); expect(parent.budget.maxOutputTokensPerCall).toBe(5);
});

it('lets explicit null clear scalar defaults and refuses malformed ceilings', async () => {
  const { session } = await fixture(), modes = new ModeRegistry(), tools = new ToolRegistry(), requests: LlmRequest[] = [];
  modes.forAgent('acp').register('reply', { tools: [] }, 'acp').setDefault('reply');
  const runtime = new AgentRuntime({ modes, tools, config: { provider: 'anthropic', model: 'global', max_output_tokens_per_call: 100 },
    agents: { acp: { model: null, max_steps_ceiling: null, max_output_tokens_per_call: null } },
    client: async request => { requests.push(request); return { text: 'done', finishReason: 'stop', costUsd: 0 }; },
  });
  await runtime.send(session('acp'), 'go');
  expect(requests[0]!.provider).toBe('anthropic'); expect(requests[0]!.model).toBe('unknown'); expect(Object.hasOwn(requests[0]!, 'maxOutputTokens')).toBe(false);
  for (const invalid of [0, -1, 1.5, true]) {
    expect(() => new AgentRuntime({ modes, tools, client: async () => ({ text: '', finishReason: 'stop' }), agents: { support: { max_steps_ceiling: invalid as number } } })).toThrow('positive integers');
  }
});

it('scopes scalar session/list defaults and retains an explicit null TTL', async () => {
  const { directory, store } = await fixture();
  const harness = new PrismHarness({ drivers: { owned: () => store }, stores: { ephemeral: 'owned', durable: 'owned' },
    ttlSeconds: 60, defaultScope: 'global-scope', defaultTaskList: 'global-work',
    agents: { support: { ttlSeconds: null, defaultScope: 'support-scope', defaultTaskList: 'support-work' } },
  });
  const participant = { type: 'User', id: 7 };
  const support = harness.forAgent('support').for(participant).asActor(participant).session(), acp = harness.forAgent('acp').for(participant).session();
  expect(support.scope).toBe('support-scope'); expect(acp.scope).toBe('global-scope');
  await support.usingMode('durable-default'); await acp.usingMode('expiring-default');
  const documents = await Promise.all((await readdir(directory)).filter(file => file.endsWith('.json')).map(async file => JSON.parse(await readFile(join(directory, file), 'utf8')) as { key: string; expires_at: number | null }));
  expect(documents.find(row => row.key === `${support.key()}:ephemeral`)?.expires_at).toBeNull();
  expect(typeof documents.find(row => row.key === `${acp.key()}:ephemeral`)?.expires_at).toBe('number');
  await harness.forAgent('support').tasks().add('owned', 'one');
  expect(await harness.forAgent('support').tasks('support-work').find('one')).not.toBeNull();
  expect(await harness.forAgent('support').tasks('global-work').find('one')).toBeNull(); expect(await harness.tasks().find('one')).toBeNull();
});

it('keeps child runs in their named defaults and the parent ledger', async () => {
  const { store } = await fixture(), tools = new ToolRegistry(), modes = new ModeRegistry(), requests: LlmRequest[] = [];
  modes.forAgent('support').register('child', { tools: [] }, 'support');
  const runtime = new AgentRuntime({ tools, modes, agents: { support: { provider: 'anthropic', model: 'support-model' } }, client: async request => { requests.push(request); return { text: 'done', finishReason: 'stop', costUsd: 0 }; } });
  const harness = new PrismHarness({ runtime, drivers: { owned: () => store }, stores: { ephemeral: 'owned', durable: 'owned' } });
  const parent = harness.forAgent('support').for({ type: 'User', id: 7 }).session('parent'), context = RunContext.root('root', new RunBudget(10));
  const result = await new SubagentRunner(harness, runtime).run(new Subagent('worker', 'Worker', 'child', new RunBudget(2)), parent, context, 'parent-run', 'go');
  expect(result.outcome).toBe('completed'); expect(context.ledger.steps).toBe(1);
  expect(requests.map(row => [row.provider, row.model])).toEqual([['anthropic', 'support-model']]);
});

it('refuses named child identity and scope arguments before work while retaining valid and legacy calls', async () => {
  const { store } = await fixture(), tools = new ToolRegistry(), modes = new ModeRegistry();
  let requests = 0;
  modes.forAgent('support').register('child', { tools: [] }, 'support'); modes.register('child', { tools: [] });
  const runtime = new AgentRuntime({ tools, modes, client: async () => { requests++; return { text: 'done', finishReason: 'stop', costUsd: 0 }; } });
  const harness = new PrismHarness({ runtime, drivers: { owned: () => store }, stores: { ephemeral: 'owned', durable: 'owned' } });
  const parent = harness.forAgent('support').for({ type: 'User', id: 7 }).session('parent'), context = RunContext.root('root', new RunBudget(10));
  const runner = new SubagentRunner(harness, runtime), child = new Subagent('worker', 'Worker', 'child', new RunBudget(2));
  for (const field of ['actor', 'agent', 'mode', 'scope']) await expect(runner.tool(child, parent, context, 'parent-run').handle({ task: 'go', [field]: 'foreign' })).rejects.toMatchObject({ code: 'run_not_permitted' });
  expect(requests).toBe(0); expect(context.ledger.steps).toBe(0);
  expect(await runner.tool(child, parent, context, 'parent-run').handle({ task: 'go' })).toBe('done');
  const legacy = harness.for({ type: 'User', id: 7 }).session('legacy-parent');
  expect(await runner.tool(child, legacy, context, 'legacy-run').handle({ task: 'go', actor: 'ignored-legacy-extra' })).toBe('done');
  expect(requests).toBe(2); expect(context.ledger.steps).toBe(2);
});
it.each([true, false])('inspection preserves logical state with native expiry housekeeping (expired=%s)', async expired => {
  class MeasuredStore extends FileSessionStore {
    purges = 0;
    override async forget(key: string): Promise<void> { this.purges++; return super.forget(key); }
  }
  const directory = await mkdtemp(join(tmpdir(), 'prism-inspection-expiry-')), store = new MeasuredStore(directory);
  const harness = new PrismHarness({ drivers: { owned: () => store }, stores: { ephemeral: 'owned', durable: 'owned' } });
  const session = harness.forAgent('support').for({ type: 'User', id: 7 }).session('ttl-inspection');
  const key = session.key() + ':ephemeral', value = { marker: 'kept' };
  await store.put(key, value, expired ? -60 : 60);
  const [name] = await readdir(directory), path = join(directory, name!), before = await readFile(path);
  const logicalBefore = await store.get(key);
  expect(logicalBefore).toEqual(expired ? null : value);
  // Restore the identical expired fixture after ordinary get() proves logical absence.
  if (expired) await writeFile(path, before);
  store.purges = 0;
  let requests = 0, handlers = 0, observed: unknown;
  const tools = new ToolRegistry(), modes = new ModeRegistry();
  tools.forAgent('support').registerProvider(async view => {
    observed = await view.state();
    return [{ name: 'lookup', handle: () => { handlers++; } }];
  }, 'support', 'state-reader');
  modes.forAgent('support').register('reply', { tools: ['lookup'] }, 'support').setDefault('reply');
  const runtime = new AgentRuntime({ tools, modes, client: async () => { requests++; throw Error('Inspection must not request a model'); } });
  expect(await runtime.effectiveTools(session, 'reply')).toEqual([{ name: 'lookup', owner: 'support', source: 'provider' }]);
  expect(observed).toEqual(expired ? {} : value);
  expect(store.purges).toBe(expired ? 1 : 0);
  expect(await store.get(key)).toEqual(logicalBefore);
  expect(requests).toBe(0); expect(handlers).toBe(0);
  if (expired) expect(await readdir(directory)).toEqual([]);
  else { expect(await readFile(path)).toEqual(before); expect(await readdir(directory)).toEqual([name]); }
  console.info(JSON.stringify({ inspection_expiry_housekeeping: true, expired, physical_purges: store.purges, logical_change: false, handlers, requests, new_entries: 0 }));
});
