import { describe, expect, it } from 'vitest';
import {
  AgentMode,
  MAX_DEPTH,
  ModeRegistry,
  RunBudget,
  RunContext,
  RunLedger,
  Subagent,
  subagentFromConfig,
} from '../src/index.js';

const config = {
  default: 'chat',
  modes: {
    chat: { system_prompt: 'Be brief.', tools: ['search'], max_steps: 4 },
    plan: {
      system_prompt: 'Plan first.',
      tools: ['search', 'write'],
      max_steps: 8,
      requires_approval: ['write'],
      subagents: { researcher: { mode: 'chat', max_steps: 3 } },
    },
  },
};

describe('AgentMode', () => {
  it('gates approval per mode, not per tool', () => {
    // The same tool is not equally consequential everywhere: `execute_op`
    // against a scratch project is routine and against production is not, and
    // the tool cannot tell which it is in.
    const mode = new ModeRegistry(config).resolve('plan');

    expect(mode.needsApproval('write')).toBe(true);
    expect(mode.needsApproval('search')).toBe(false);
    expect(new ModeRegistry(config).resolve('chat').needsApproval('write')).toBe(false);
  });

  it('treats * as gating every tool the mode offers', () => {
    const mode = new AgentMode('m', '', ['a', 'b'], [], 4, {}, ['*']);

    expect(mode.needsApproval('a')).toBe(true);
    expect(mode.needsApproval('anything')).toBe(true);
  });
});

describe('ModeRegistry', () => {
  it('falls back to the configured default', () => {
    expect(new ModeRegistry(config).resolve(null).name).toBe('chat');
    expect(new ModeRegistry({}).default()).toBe('chat');
  });

  it('names a mode that is not configured', () => {
    expect(() => new ModeRegistry(config).resolve('ghost')).toThrowError(/ghost/);
  });

  it('refuses a malformed max_steps', () => {
    const broken = { modes: { bad: { max_steps: 0 } } };

    expect(() => new ModeRegistry(broken).resolve('bad')).toThrowError(/malformed/);
  });

  it('resolves EVERY mode, which is what finds a break before someone enters it', () => {
    // A mode nobody has entered yet keeps its misconfiguration until the day
    // somebody switches to it, and the first person to find out is a user.
    expect(Object.keys(new ModeRegistry(config).all())).toEqual(['chat', 'plan']);
  });

  it('refuses a subagent whose mode is not configured, when the PARENT loads', () => {
    // A typo surfaces when the parent mode is loaded, rather than halfway
    // through a run that has already spent budget.
    const broken = {
      modes: { plan: { subagents: { helper: { mode: 'nope' } } } },
    };

    expect(() => new ModeRegistry(broken).resolve('plan')).toThrowError(/nope.*not configured/);
  });

  it('gives a mode only the subagents it declares', () => {
    // A subagent is authority, and authority a run inherits by being nested is
    // authority nobody granted.
    const registry = new ModeRegistry(config);

    expect(Object.keys(registry.resolve('plan').subagents)).toEqual(['researcher']);
    expect(Object.keys(registry.resolve('chat').subagents)).toEqual([]);
  });
});

describe('Subagent', () => {
  it('gives the child a DIFFERENT scope, which is what avoids the deadlock', () => {
    // A session's lock is taken on its address. A nested run asking for the
    // parent's address inside the parent's own lock would wait on a lock it
    // already holds; a separate address removes the contention rather than
    // making the lock reentrant.
    const subagent = subagentFromConfig('researcher', { mode: 'chat' });

    expect(subagent.scopeUnder('support')).toBe('support::sub::researcher');
    expect(subagent.scopeUnder('support')).not.toBe('support');
  });

  it('honours an explicit scope suffix, so a cold worker lands on the same child', () => {
    expect(subagentFromConfig('r', { mode: 'chat', scope: 'fixed' }).scopeUnder('s')).toBe(
      's::sub::fixed',
    );
  });

  it('defaults its mode to its own name and writes a usable description', () => {
    const subagent = subagentFromConfig('researcher', {});

    expect(subagent.mode).toBe('researcher');
    expect(subagent.description).toContain('researcher');
  });
});

describe('RunBudget', () => {
  // The budget a child gets is a TREE-ABSOLUTE ceiling, so the headroom it
  // actually has is `maxSteps - ledger.steps`. Expressing it as "remaining"
  // and comparing it against a cumulative ledger is the reference's bug — see
  // `nestedWithin`.
  const headroom = (budget: RunBudget, ledger: RunLedger): number =>
    budget.maxSteps - ledger.steps;

  it('NESTS rather than resets', () => {
    // A parent limited to 8 steps that may spawn subagents each entitled to a
    // fresh 8 has no bound at all — it has a bound per node in a tree whose
    // width it also controls, which is unbounded spend wearing a limit's
    // clothing.
    const parent = new RunBudget(8, 1.0);
    const ledger = RunLedger.start('root');
    ledger.recordSteps(6);
    ledger.recordCost(0.75);

    const child = new RunBudget(8, 1.0).nestedWithin(parent, ledger);

    expect(headroom(child, ledger)).toBe(2);
    expect(child.maxSteps).toBeLessThanOrEqual(parent.maxSteps);
    expect(child.maxCostUsd).toBeCloseTo(1.0);
  });

  it('gives an UNSPENT parent a child exactly what it declared', () => {
    // The case the reference gets right, kept so the fix cannot regress it.
    const ledger = RunLedger.start('root');
    const child = new RunBudget(2).nestedWithin(new RunBudget(8), ledger);

    expect(headroom(child, ledger)).toBe(2);
  });

  it('gives a NEARLY SPENT parent a child exactly what is left', () => {
    // The case the reference gets wrong: it would hand this child zero steps
    // while the tree still had one.
    const ledger = RunLedger.start('root');
    ledger.recordSteps(7);

    const child = new RunBudget(2).nestedWithin(new RunBudget(8), ledger);

    expect(headroom(child, ledger)).toBe(1);
    expect(ledger.exhaustion(child)).toBeNull();
  });

  it('never lets a child ask for more than remains', () => {
    const ledger = RunLedger.start('root');
    ledger.recordSteps(8);

    const child = new RunBudget(99).nestedWithin(new RunBudget(8), ledger);

    expect(headroom(child, ledger)).toBe(0);
    expect(ledger.exhaustion(child)).toMatch(/step budget exhausted/);
  });

  it('carries a parent cap down to a child that declared none', () => {
    const child = new RunBudget(4).nestedWithin(new RunBudget(8, 2.0), RunLedger.start('r'));

    expect(child.maxCostUsd).toBeCloseTo(2.0);
  });

  it('keeps a child cost cap tighter than the parent it sits under', () => {
    const ledger = RunLedger.start('root');
    const child = new RunBudget(4, 0.25).nestedWithin(new RunBudget(8, 2.0), ledger);

    expect(child.maxCostUsd).toBeCloseTo(0.25);
  });
});

describe('RunLedger', () => {
  it('reports WHY the tree may not spend again, not just that it may not', () => {
    // The states are genuinely different, and a caller that cannot tell them
    // apart writes one message for four causes.
    const budget = new RunBudget(2, 1.0, 60);
    const ledger = RunLedger.start('root');

    expect(ledger.exhaustion(budget)).toBeNull();

    ledger.recordSteps(2);
    expect(ledger.exhaustion(budget)).toMatch(/step budget exhausted/);
  });

  it('reports a cancellation ahead of any budget, with its reason', () => {
    const ledger = RunLedger.start('root');
    ledger.recordSteps(99);
    ledger.cancel('the user closed the tab');

    expect(ledger.exhaustion(new RunBudget(1))).toBe('the user closed the tab');
  });

  it('FAILS CLOSED when a cost cap cannot be enforced', () => {
    // A provider that reports no cost would otherwise fold into `+= 0`, leaving
    // a cap that can never trip — enforced in the documentation, absent at
    // runtime, and indistinguishable from a tree that spent nothing.
    const ledger = RunLedger.start('root');
    ledger.recordCost(null);

    expect(ledger.exhaustion(new RunBudget(10, 5.0))).toMatch(/cannot be enforced/);
    expect(ledger.unmeteredRuns).toBe(1);
  });

  it('does not complain about unmetered runs when there is no cost cap', () => {
    const ledger = RunLedger.start('root');
    ledger.recordCost(null);

    expect(ledger.exhaustion(new RunBudget(10))).toBeNull();
  });

  it('reports an exhausted cost budget', () => {
    const ledger = RunLedger.start('root');
    ledger.recordCost(1.5);

    expect(ledger.exhaustion(new RunBudget(10, 1.0))).toMatch(/cost budget exhausted/);
  });
});

describe('RunContext', () => {
  it('shares ONE ledger down the whole tree', () => {
    // A child with its own ledger would let every node report itself inside
    // budget while the tree spent without limit.
    const root = RunContext.root('run-1', new RunBudget(8));
    const child = root.forChild(new Subagent('r', '', 'chat', new RunBudget(4)), 'run-1');

    child.ledger.recordSteps(3);

    expect(root.ledger.steps).toBe(3);
    expect(child.rootRunId).toBe('run-1');
    expect(child.isChild()).toBe(true);
    expect(root.isChild()).toBe(false);
  });

  it('stops at the depth ceiling', () => {
    let context = RunContext.root('run-1', new RunBudget(64));
    const subagent = new Subagent('r', '', 'chat', new RunBudget(64));

    for (let depth = 0; depth < MAX_DEPTH; depth += 1) {
      expect(context.tooDeep()).toBe(false);
      context = context.forChild(subagent, 'run-1');
    }

    expect(context.tooDeep()).toBe(true);
  });
});
