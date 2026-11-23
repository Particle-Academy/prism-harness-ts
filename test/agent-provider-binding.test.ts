import { describe, expect, it } from 'vitest';
import { AgentRuntime, HarnessError, ModeRegistry, ToolRegistry } from '../src/index.js';
import { bindingWorld } from './support/binding-world.js';

/**
 * G-321 E — several agents in one app, each on its own provider and model.
 *
 * Every test enters where an application enters (a harness session, `send()`)
 * and asserts what the model client was ACTUALLY asked for, never a resolved
 * value read back from a helper.
 */

const pick = (seen: { provider: string; model: string }) => ({ provider: seen.provider, model: seen.model });

describe('provider and model precedence: session > mode > agent > global', () => {
  it('a mode model overrides the agent model', async () => {
    const world = await bindingWorld({
      global: { provider: 'anthropic', model: 'global-model' },
      agents: { planner: { model: 'agent-model', modes: { plan: { model: 'mode-model' } } } },
    }, { agent: 'planner', mode: 'plan' });

    await world.send();

    expect(world.requests.map(pick)).toEqual([{ provider: 'anthropic', model: 'mode-model' }]);
  });

  it('a session override beats the mode', async () => {
    const world = await bindingWorld({
      global: { provider: 'anthropic', model: 'global-model' },
      agents: { planner: { model: 'agent-model', modes: { plan: { provider: 'custom-cheap', model: 'mode-model' } } } },
    }, { agent: 'planner', mode: 'plan', session_model: 'session-model' });

    await world.send();

    expect(world.requests.map(pick)).toEqual([{ provider: 'custom-cheap', model: 'session-model' }]);
  });
});

describe('subagents bound to their own provider and model', () => {
  it('runs a subagent on a cheaper model under a parent on another', async () => {
    const world = await bindingWorld({
      global: { provider: 'anthropic', model: 'global-model' },
      agents: { planner: { model: 'agent-model', modes: {
        plan: { model: 'opus-like', subagents: { research: { mode: 'research', provider: 'custom-cheap', model: 'cheap-model' } } },
        research: {},
      } } },
    }, { agent: 'planner', mode: 'plan', delegate: 'research' });

    await world.send();

    expect(world.requests.map(pick)).toEqual([
      { provider: 'anthropic', model: 'opus-like' },
      { provider: 'custom-cheap', model: 'cheap-model' },
      { provider: 'anthropic', model: 'opus-like' },
    ]);
  });

  it('does not hand the parent session override to the child by default', async () => {
    const world = await bindingWorld({
      global: { provider: 'anthropic', model: 'global-model' },
      agents: { planner: { model: 'agent-model', modes: {
        plan: { subagents: { research: { mode: 'research' } } }, research: {},
      } } },
    }, { agent: 'planner', mode: 'plan', session_provider: 'custom-cheap', session_model: 'session-model', delegate: 'research' });

    await world.send();

    expect(world.requests.map(pick).slice(0, 2)).toEqual([
      { provider: 'custom-cheap', model: 'session-model' },
      { provider: 'anthropic', model: 'agent-model' },
    ]);
  });

  it('hands the parent session override to a child that asks for it', async () => {
    const world = await bindingWorld({
      global: { provider: 'anthropic', model: 'global-model' },
      agents: { planner: { model: 'agent-model', modes: {
        plan: { subagents: { research: { mode: 'research', model: 'cheap-model', inherit_session_model: true } } }, research: {},
      } } },
    }, { agent: 'planner', mode: 'plan', session_model: 'session-model', delegate: 'research' });

    await world.send();

    expect(world.requests.map(pick)[1]).toEqual({ provider: 'anthropic', model: 'session-model' });
  });

  it('never writes the parent override onto the child session', async () => {
    const world = await bindingWorld({
      global: { provider: 'anthropic', model: 'global-model' },
      modes: { chat: { subagents: { research: { mode: 'research' } } }, research: {} },
    }, { agent: null, mode: 'chat', session_model: 'session-model', delegate: 'research' });

    await world.send();

    expect(world.requests.map(pick)[1]).toEqual({ provider: 'anthropic', model: 'global-model' });
  });

  it('runs a subagent on a declared agent with that agent binding', async () => {
    const world = await bindingWorld({
      global: { provider: 'anthropic', model: 'global-model' },
      agents: {
        planner: { model: 'agent-model', modes: {
          plan: { model: 'opus-like', agents: ['researcher'], subagents: { research: { mode: 'research', agent: 'researcher' } } },
        } },
        researcher: { provider: 'custom-cheap', model: 'researcher-model', modes: { research: {} } },
      },
    }, { agent: 'planner', mode: 'plan', delegate: 'research' });

    await world.send();

    expect(world.requests.map(pick)[1]).toEqual({ provider: 'custom-cheap', model: 'researcher-model' });
  });
});

describe('delegation authority is declared, never inherited', () => {
  const refusedWith = async (subagent: Record<string, unknown>, agents: unknown[] | undefined, message: string) => {
    const world = await bindingWorld({
      global: { provider: 'anthropic', model: 'global-model' },
      agents: {
        planner: { model: 'agent-model', modes: {
          plan: { ...(agents === undefined ? {} : { agents: agents as string[] }), subagents: { research: { mode: 'research', ...subagent } } },
        } },
        researcher: { model: 'researcher-model', modes: { research: {} } },
      },
    }, { agent: 'planner', mode: 'plan', delegate: 'research' });

    const failure = await world.send().then(() => null, (error: unknown) => error);

    expect(failure).toBeInstanceOf(HarnessError);
    expect((failure as HarnessError).code).toBe('mode_malformed');
    expect((failure as HarnessError).message).toBe(message);
    expect(world.requests).toEqual([]);
  };

  it('refuses a subagent naming an agent the mode does not list, before any request', async () => {
    await refusedWith({ agent: 'researcher' }, undefined,
      'Harness mode [plan] declares subagent [research] for agent [researcher], which the mode does not list in its agents.');
  });

  it('refuses a padded agent id rather than trimming it', async () => {
    await refusedWith({ agent: ' researcher' }, ['researcher'],
      'Harness mode [plan] declares subagent [research] with a malformed agent id: agent ids are lowercase identifiers of 1-64 ASCII bytes.');
  });

  it('refuses a mixed-case agent id rather than folding it', async () => {
    await refusedWith({ agent: 'Researcher' }, ['researcher'],
      'Harness mode [plan] declares subagent [research] with a malformed agent id: agent ids are lowercase identifiers of 1-64 ASCII bytes.');
  });

  it('refuses a mixed-case or padded entry in the mode agents list', async () => {
    for (const entry of ['Researcher', 'researcher ', '\u0000researcher']) {
      await refusedWith({ agent: 'researcher' }, [entry],
        'Harness mode [plan] declares a malformed agents list: each entry must be a lowercase agent id of 1-64 ASCII bytes.');
    }
  });

  it('refuses malformed provider, model and inherit_session_model declarations', () => {
    const resolve = (mode: Record<string, unknown>) => () => new ModeRegistry({ modes: { m: mode as never, c: {} } }).resolve('m');

    expect(resolve({ model: '' })).toThrow('Harness mode [m] declares a malformed provider/model: expected a nonempty string or null.');
    expect(resolve({ provider: 7 })).toThrow('Harness mode [m] declares a malformed provider/model: expected a nonempty string or null.');
    expect(resolve({ subagents: { s: { mode: 'c', model: '' } } })).toThrow('Harness mode [m] declares subagent [s] with a malformed provider/model: expected a nonempty string or null.');
    expect(resolve({ subagents: { s: { mode: 'c', inherit_session_model: 'yes' } } })).toThrow('Harness mode [m] declares subagent [s] with a malformed inherit_session_model: expected a boolean.');
  });

  it('refuses a malformed spend_ceiling when the runtime is built', () => {
    for (const ceiling of [0, -1, Number.POSITIVE_INFINITY, Number.NaN, '1']) {
      expect(() => new AgentRuntime({
        modes: new ModeRegistry(), tools: new ToolRegistry(), client: async () => ({ text: '', finishReason: 'stop' }),
        agents: { planner: { spend_ceiling: ceiling as number } },
      })).toThrow('Agent spend_ceiling must be a positive finite number of USD or null.');
    }
  });
});

describe('per-agent spend ceilings on the shared root ledger', () => {
  it('stops only the agent over its ceiling; the parent agent completes', async () => {
    const world = await bindingWorld({
      global: { provider: 'anthropic', model: 'global-model' },
      agents: {
        planner: { model: 'agent-model', modes: {
          plan: { agents: ['researcher'], subagents: { research: { mode: 'research', agent: 'researcher' } } },
        } },
        researcher: { model: 'researcher-model', spend_ceiling: 0.05, modes: { research: {} } },
      },
    }, { agent: 'planner', mode: 'plan', delegate: 'research', costs: [0.5, 0.5, 0.5] });
    // The researcher has already used its whole ceiling earlier in this tree.
    world.context.ledger.recordCost(0.05, 'researcher');

    const response = await world.send();

    // Parent step, NO child request, parent step: the child was refused before it called.
    expect(world.requests.map(pick)).toEqual([
      { provider: 'anthropic', model: 'agent-model' },
      { provider: 'anthropic', model: 'agent-model' },
    ]);
    expect(response.text).toBe('answer 1');
    expect(world.context.ledger.agentCostUsd('planner')).toBe(1);
    expect(world.context.ledger.costUsd).toBeCloseTo(1.05);
    const failed = world.events.find((event) => event.type === 'run.failed');
    expect(failed).toMatchObject({ harnessAgent: 'researcher', failure: expect.stringContaining('spend ceiling') });
  });

  it('lets an agent under its ceiling spend, then refuses its next call', async () => {
    const world = await bindingWorld({
      global: { provider: 'anthropic', model: 'global-model' },
      agents: { planner: { model: 'agent-model', spend_ceiling: 0.02, modes: { plan: {} } } },
    }, { agent: 'planner', mode: 'plan', costs: [0.03] });

    // Under the ceiling when it starts: the call lands and is charged.
    await world.send();
    expect(world.context.ledger.agentCostUsd('planner')).toBeCloseTo(0.03);

    const failure = await world.send().then(() => null, (error: unknown) => error);
    expect((failure as HarnessError).code).toBe('agent_spend_ceiling_exceeded');
    expect(world.requests).toHaveLength(1);
  });

  it('refuses an unpriced provider under a ceiling (fail closed)', async () => {
    const world = await bindingWorld({
      global: { provider: 'anthropic', model: 'global-model' },
      agents: { planner: { provider: 'custom-cheap', model: 'unpriced', spend_ceiling: 1, modes: { plan: {} } } },
    }, { agent: 'planner', mode: 'plan', costs: [null] });

    await world.send();
    const failure = await world.send().then(() => null, (error: unknown) => error);

    expect((failure as HarnessError).code).toBe('budget_unpriced');
    expect(world.requests).toHaveLength(1);
  });
});

describe('harnessAgent on events', () => {
  it('tags every run event of a named agent and leaves legacy events unchanged', async () => {
    const named = await bindingWorld({
      global: { provider: 'anthropic', model: 'global-model' },
      agents: { planner: { modes: { plan: {} } } },
    }, { agent: 'planner', mode: 'plan' });
    await named.send();

    expect(named.events.map((event) => event.type)).toEqual(['run.started', 'run.finished']);
    expect(named.events.every((event) => 'harnessAgent' in event && event.harnessAgent === 'planner')).toBe(true);

    const legacy = await bindingWorld({
      global: { provider: 'anthropic', model: 'global-model' },
      modes: { chat: {} },
    }, { agent: null, mode: 'chat' });
    await legacy.send();

    expect(legacy.events).toHaveLength(2);
    expect(legacy.events.some((event) => 'harnessAgent' in event)).toBe(false);
  });

  it('tags a delegated child with its own agent', async () => {
    const world = await bindingWorld({
      global: { provider: 'anthropic', model: 'global-model' },
      agents: {
        planner: { modes: { plan: { agents: ['researcher'], subagents: { research: { mode: 'research', agent: 'researcher' } } } } },
        researcher: { modes: { research: {} } },
      },
    }, { agent: 'planner', mode: 'plan', delegate: 'research' });
    await world.send();

    expect(world.events.filter((event) => event.type === 'run.started').map((event) => 'harnessAgent' in event ? event.harnessAgent : null))
      .toEqual(['planner', 'researcher']);
  });
});
