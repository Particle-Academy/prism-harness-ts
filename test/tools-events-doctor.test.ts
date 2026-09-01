import { mkdtemp } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { describe, expect, it } from 'vitest';
import {
  FileSessionStore,
  HarnessEvents,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  SessionStoreManager,
  ToolAuthorizer,
  ToolRegistry,
  diagnose,
  type HarnessEvent,
  type HarnessTool,
  type Session,
} from '../src/index.js';

function tool(name: string, handle: (args: Record<string, unknown>) => unknown = () => 'ok'): HarnessTool {
  return { name, description: `the ${name} tool`, handle: handle as HarnessTool['handle'] };
}

async function aSession(): Promise<Session> {
  const directory = await mkdtemp(join(tmpdir(), 'prism-harness-tools-'));
  const harness = new PrismHarness({
    drivers: { memory: () => new MemorySessionStore(), files: () => new FileSessionStore(directory) },
    stores: { ephemeral: 'memory', durable: 'files' },
  });

  return harness.for({ type: 'User', id: 1 }).session('support');
}

describe('ToolRegistry', () => {
  it('resolves named tools', async () => {
    const registry = new ToolRegistry().register(tool('search')).register(tool('write'));

    expect([...(await registry.resolve(['search'])).keys()]).toEqual(['search']);
  });

  it('resolves * to everything it can produce', async () => {
    const registry = new ToolRegistry().registerMany([tool('a'), tool('b')]);

    expect([...(await registry.resolve(['*'])).keys()].sort()).toEqual(['a', 'b']);
  });

  it('REFUSES a name it cannot produce rather than dropping it', async () => {
    // A mode that lists a tool it cannot get is a misconfiguration, and
    // dropping it quietly would leave a run wondering why the model never
    // called it.
    const registry = new ToolRegistry().register(tool('search'));

    await expect(registry.resolve(['ghost'])).rejects.toMatchObject({ code: 'tool_not_available' });
  });

  it('builds a factory tool with the session', async () => {
    const registry = new ToolRegistry().registerFactory('scoped', (session) =>
      tool(`scoped:${session.scope}`),
    );

    const resolved = await registry.resolve(['scoped'], await aSession());

    expect(resolved.get('scoped')?.name).toBe('scoped:support');
  });

  it('takes tools from a provider, for a catalogue discovered at resolve time', async () => {
    const registry = new ToolRegistry().registerProvider(() => [tool('mcp:read')]);

    expect([...(await registry.resolve(['*'], await aSession())).keys()]).toEqual(['mcp:read']);
  });
});

describe('ToolAuthorizer', () => {
  it('is OFF by default and offers everything', async () => {
    const authorizer = new ToolAuthorizer();
    const tools = await new ToolRegistry().registerMany([tool('a'), tool('b')]).resolve(['*']);

    expect(authorizer.enabled).toBe(false);
    expect((await authorizer.allowed(await aSession(), tools)).length).toBe(2);
  });

  it('REFUSES a policy that would never be consulted', () => {
    // Both at once is the one configuration not to leave in place: a defined
    // policy that is never consulted looks like a control to every reader and
    // is not one.
    expect(() => new ToolAuthorizer({ enabled: false, offer: () => true })).toThrowError(
      /never consulted/,
    );
  });

  it('filters what a run is OFFERED', async () => {
    const authorizer = new ToolAuthorizer({
      enabled: true,
      offer: (_session, candidate) => candidate.name !== 'danger',
    });
    const tools = await new ToolRegistry().registerMany([tool('safe'), tool('danger')]).resolve(['*']);

    expect((await authorizer.allowed(await aSession(), tools)).map((t) => t.name)).toEqual(['safe']);
  });

  it('asks AGAIN at call time, with the arguments', async () => {
    // At the moment the toolset is assembled the arguments do not exist yet, so
    // an offer policy can say "may use delete_file" and never "only under /tmp".
    const authorizer = new ToolAuthorizer({
      enabled: true,
      call: (_session, _tool, args) => String(args.path ?? '').startsWith('/tmp/'),
    });
    const tools = await new ToolRegistry().register(tool('delete_file')).resolve(['*']);
    const [wrapped] = await authorizer.allowed(await aSession(), tools);

    expect(await wrapped?.handle({ path: '/tmp/scratch' })).toBe('ok');
    await expect(wrapped?.handle({ path: '/etc/passwd' })).rejects.toMatchObject({
      code: 'call_not_authorized',
    });
  });

  it('THROWS a refusal rather than returning it as a result', async () => {
    // A refusal handed back as a tool result reads to the model as a failure it
    // might retry differently, and a denied action being retried is the
    // opposite of what a guard is for.
    const authorizer = new ToolAuthorizer({ enabled: true, call: () => false });
    const tools = await new ToolRegistry().register(tool('x')).resolve(['*']);
    const [wrapped] = await authorizer.allowed(await aSession(), tools);

    await expect(wrapped?.handle({})).rejects.toThrow();
  });
});

describe('HarnessEvents', () => {
  it('delivers to every listener and can unsubscribe', () => {
    const events = new HarnessEvents();
    const seen: HarnessEvent[] = [];
    const stop = events.listen((event) => seen.push(event));

    const started: HarnessEvent = {
      type: 'run.started',
      runId: 'r1',
      sessionKey: 'k',
      mode: 'chat',
      provider: 'anthropic',
      model: 'm',
      rootRunId: 'r1',
      depth: 0,
      at: new Date().toISOString(),
    };

    events.emit(started);
    stop();
    events.emit(started);

    expect(seen).toHaveLength(1);
  });

  it('does NOT let a throwing listener break the run', () => {
    // Telemetry that takes down the thing it observes is worse than no
    // telemetry, and a listener is by definition somebody else's code.
    const events = new HarnessEvents();
    const seen: string[] = [];

    events.listen(() => {
      throw new Error('listener exploded');
    });
    events.listen((event) => seen.push(event.type));

    expect(() =>
      events.emit({
        type: 'run.failed',
        runId: 'r1',
        sessionKey: 'k',
        failure: 'nope',
        steps: 1,
        at: '',
      }),
    ).not.toThrow();

    expect(seen).toEqual(['run.failed']);
  });
});

describe('the doctor', () => {
  it('reports a consistent configuration', async () => {
    const directory = await mkdtemp(join(tmpdir(), 'prism-harness-doctor-'));
    const report = diagnose({
      stores: new SessionStoreManager({
        drivers: { memory: () => new MemorySessionStore(), files: () => new FileSessionStore(directory) },
        stores: { ephemeral: 'memory', durable: 'files' },
      }),
      modes: new ModeRegistry({ modes: { chat: { tools: ['search'] } }, default: 'chat' }),
      tools: new ToolRegistry().register(tool('search')),
      authorizer: new ToolAuthorizer(),
    });

    expect(report.ok).toBe(true);
    expect(report.summary()).toContain('consistent');
  });

  it('catches a volatile durable store', () => {
    const report = diagnose({
      stores: new SessionStoreManager({
        drivers: { memory: () => new MemorySessionStore() },
        stores: { ephemeral: 'memory', durable: 'memory' },
      }),
    });

    expect(report.ok).toBe(false);
    expect(report.summary()).toContain('VOLATILE');
  });

  it('catches a broken mode NOBODY HAS ENTERED YET', () => {
    // The whole reason this exists. That mode keeps its broken subagent
    // reference until someone switches to it, and the first person to find out
    // is a user mid-conversation.
    const report = diagnose({
      modes: new ModeRegistry({
        default: 'chat',
        modes: { chat: {}, broken: { subagents: { helper: { mode: 'ghost' } } } },
      }),
    });

    expect(report.ok).toBe(false);
    expect(report.findings.find((f) => f.check === 'mode:broken')?.ok).toBe(false);
    // The default still resolves, so only the unentered mode is reported.
    expect(report.findings.find((f) => f.check === 'mode:chat')?.ok).toBe(true);
  });

  it('catches a mode naming a tool the registry cannot produce', () => {
    const report = diagnose({
      modes: new ModeRegistry({ default: 'chat', modes: { chat: { tools: ['ghost'] } } }),
      tools: new ToolRegistry().register(tool('search')),
    });

    expect(report.ok).toBe(false);
    expect(report.summary()).toContain('ghost');
  });

  it('says plainly when the authorizer is off', () => {
    const report = diagnose({ authorizer: new ToolAuthorizer() });

    expect(report.summary()).toContain('DISABLED');
    // Not a failure — off is the default and a legitimate choice.
    expect(report.ok).toBe(true);
  });

  it('reports having no modes at all as a problem', () => {
    expect(diagnose({ modes: new ModeRegistry({}) }).ok).toBe(false);
  });
});
