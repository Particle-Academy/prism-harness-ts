import { mkdtemp } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { describe, expect, it } from 'vitest';
import { FileSessionStore, MemorySessionStore, PrismHarness, Session } from '../src/index.js';

async function harness(): Promise<PrismHarness> {
  const directory = await mkdtemp(join(tmpdir(), 'prism-harness-session-'));

  return new PrismHarness({
    drivers: { memory: () => new MemorySessionStore(), files: () => new FileSessionStore(directory) },
    stores: { ephemeral: 'memory', durable: 'files' },
  });
}

async function session(scope = 'support'): Promise<Session> {
  return (await harness()).for({ type: 'App\\Models\\User', id: 7 }).session(scope);
}

describe('addressing', () => {
  it('produces the SAME key the PHP reference produces', async () => {
    // sha1 of the participant type, truncated to 12 — byte for byte what
    // `Prism\Harness\Sessions\Session::key()` builds. Matching exactly is what
    // lets a PHP app and a TypeScript agent share one store and resolve the
    // same session; a different digest would give two conversations that look
    // identical and never meet.
    const { createHash } = await import('node:crypto');
    const digest = createHash('sha1').update('App\\Models\\User').digest('hex').slice(0, 12);

    expect((await session()).key()).toBe(`session:${digest}:7:support`);
  });

  it('keeps two scopes for one participant apart', async () => {
    const harnessed = await harness();
    const participant = { type: 'User', id: 1 };

    const support = harnessed.for(participant).session('support');
    const billing = harnessed.for(participant).session('billing');

    await support.usingMode('plan');

    expect(await support.mode()).toBe('plan');
    expect(await billing.mode()).toBeNull();
  });

  it('keeps the same id under two participant TYPES apart', async () => {
    // `7` means a different participant in each table, which is why the type is
    // part of the address rather than just the id.
    const harnessed = await harness();
    const user = harnessed.for({ type: 'User', id: 7 }).session('s');
    const team = harnessed.for({ type: 'Team', id: 7 }).session('s');

    await user.usingModel('claude-sonnet-4-5');

    expect(await team.model()).toBeNull();
    expect(user.key()).not.toBe(team.key());
  });
});

describe('the ephemeral half', () => {
  it('round-trips mode, model and provider', async () => {
    const live = await session();
    await live.usingMode('plan');
    await live.usingModel('claude-sonnet-4-5');
    await live.usingProvider('anthropic');

    expect(await live.mode()).toBe('plan');
    expect(await live.model()).toBe('claude-sonnet-4-5');
    expect(await live.provider()).toBe('anthropic');
  });

  it('is RESOLVED, not held — a second instance sees what the first wrote', async () => {
    // The whole design. Nothing survives in memory between turns, so a fresh
    // worker resolving the same address must see the same state.
    const harnessed = await harness();
    const participant = { type: 'User', id: 7 };

    await harnessed.for(participant).session('support').usingMode('plan');

    expect(await harnessed.for(participant).session('support').mode()).toBe('plan');
  });

  it('forget() drops the ephemeral half and LEAVES THE CONVERSATION', async () => {
    const live = await session();
    await live.usingMode('plan');
    await live.thread().record([{ type: 'user', content: 'hello' }]);

    await live.forget();

    expect(await live.mode()).toBeNull();
    expect(await live.thread().count()).toBe(1);
  });
});

describe('the durable half', () => {
  it('stores and forgets a capability', async () => {
    const live = await session();
    await live.usingCapability('search', { index: 'docs', k: 5 });

    expect(await live.capability('search')).toEqual({ index: 'docs', k: 5 });

    await live.forgetCapability('search');
    expect(await live.capability('search')).toBeNull();
  });

  it('survives the ephemeral half being dropped', async () => {
    const live = await session();
    await live.usingCapability('search', { index: 'docs' });
    await live.forget();

    expect(await live.capability('search')).toEqual({ index: 'docs' });
  });
});

describe('runs', () => {
  it('records a run through its lifecycle', async () => {
    const live = await session();
    await live.beginRun('run-1', 'plan', 'anthropic', 'claude-sonnet-4-5');

    expect(await live.run()).toMatchObject({ id: 'run-1', status: 'running', mode: 'plan' });

    await live.completeRun('run-1', 'stop', ['search', 'write']);

    expect(await live.run()).toMatchObject({
      id: 'run-1',
      status: 'completed',
      finish_reason: 'stop',
      tool_calls: ['search', 'write'],
    });
  });

  it('records a failure', async () => {
    const live = await session();
    await live.beginRun('run-1', 'plan', 'anthropic', 'm');
    await live.failRun('run-1', 'provider timed out');

    expect(await live.run()).toMatchObject({ status: 'failed', failure: 'provider timed out' });
  });

  it('records TOOL NAMES only, never arguments', async () => {
    // A tool name is not PII and is what an operator needs to audit a guardrail.
    // Arguments are, and prism-opentelemetry already carries them behind an
    // opt-in capture gate — recording them again here, ungated, would quietly
    // undo that decision for anyone who installed both.
    const live = await session();
    await live.beginRun('run-1', 'plan', 'anthropic', 'm');
    await live.completeRun('run-1', 'stop', ['search']);

    expect(JSON.stringify(await live.run())).not.toContain('argument');
    expect((await live.run())?.tool_calls).toEqual(['search']);
  });

  it('does NOT let a superseded run overwrite the one in flight', async () => {
    // A late worker reporting on a run that has already been replaced would
    // otherwise mark the live one finished.
    const live = await session();
    await live.beginRun('run-1', 'plan', 'anthropic', 'm');
    await live.beginRun('run-2', 'plan', 'anthropic', 'm');

    await live.completeRun('run-1', 'stop');

    expect(await live.run()).toMatchObject({ id: 'run-2', status: 'running' });
  });
});

describe('lock', () => {
  it('runs the callback and returns its value', async () => {
    expect(await (await session()).lock(() => 'done')).toBe('done');
  });

  it('RE-READS state inside the lock', async () => {
    // State written by whoever held the lock before us is otherwise invisible
    // to this instance, and acting on a stale read is what the lock exists to
    // prevent.
    const harnessed = await harness();
    const participant = { type: 'User', id: 7 };
    const one = harnessed.for(participant).session('support');
    const two = harnessed.for(participant).session('support');

    // Prime `one`'s cache, then change the state behind its back.
    expect(await one.mode()).toBeNull();
    await two.usingMode('plan');

    expect(await one.lock(async (live) => live.mode())).toBe('plan');
  });
});

describe('threads', () => {
  it('assigns positions from 1, in order', async () => {
    const thread = (await session()).thread();
    const recorded = await thread.record([
      { type: 'user', content: 'one' },
      { type: 'assistant', content: 'two' },
    ]);

    expect(recorded.map((entry) => entry.position)).toEqual([1, 2]);
    expect((await thread.messages()).map((entry) => entry.position)).toEqual([1, 2]);
  });

  it('continues numbering across separate calls', async () => {
    const thread = (await session()).thread();
    await thread.record([{ type: 'user', content: 'one' }]);
    const second = await thread.record([{ type: 'user', content: 'two' }]);

    expect(second[0]?.position).toBe(2);
  });

  it('does not lose a message when two turns land CONCURRENTLY', async () => {
    // Read-and-write inside one lock. Both callers would otherwise read length
    // 0, both write position 1, and the conversation would silently lose a
    // message — the race the reference tracks as prism-harness#2.
    const thread = (await session()).thread();

    await Promise.all([
      thread.record([{ type: 'user', content: 'a' }]),
      thread.record([{ type: 'user', content: 'b' }]),
    ]);

    const positions = (await thread.messages()).map((entry) => entry.position);
    expect(positions).toEqual([1, 2]);
  });

  it('carries the run id that produced a message', async () => {
    const thread = (await session()).thread();
    await thread.record([{ type: 'user', content: 'x' }], 'run-1');

    expect((await thread.messages())[0]?.runId).toBe('run-1');
  });

  it('records nothing, and returns nothing, for an empty list', async () => {
    const thread = (await session()).thread();

    expect(await thread.record([])).toEqual([]);
    expect(await thread.count()).toBe(0);
  });

  it('clear() empties the conversation without touching session state', async () => {
    const live = await session();
    await live.usingMode('plan');
    await live.thread().record([{ type: 'user', content: 'x' }]);

    await live.thread().clear();

    expect(await live.thread().count()).toBe(0);
    expect(await live.mode()).toBe('plan');
  });
});

describe('the default harness', () => {
  it('opens, and REFUSES durable state, because in-memory is volatile', async () => {
    // Not an oversight to smooth over. A package that silently accepted an
    // in-memory durable store would pass every test on one process and lose a
    // half-executed action the first time it was deployed on two.
    const harnessed = new PrismHarness();

    expect(() => harnessed.ephemeralStore()).not.toThrow();
    expect(() => harnessed.for({ type: 'User', id: 1 }).session('s')).toThrowError(
      /VOLATILE/,
    );
  });
});
