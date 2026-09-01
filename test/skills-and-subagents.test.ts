import { mkdir, mkdtemp, symlink, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { describe, expect, it } from 'vitest';
import {
  AgentRuntime,
  FileSessionStore,
  MemorySessionStore,
  ModeRegistry,
  PrismHarness,
  RunBudget,
  RunContext,
  SkillRegistry,
  SubagentRunner,
  ToolRegistry,
  type LlmResponse,
  type Session,
} from '../src/index.js';

async function skillRoot(): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), 'prism-harness-skills-'));
  await mkdir(join(root, 'research', 'notes'), { recursive: true });
  await writeFile(join(root, 'research', 'SKILL.md'), 'Search carefully.', 'utf8');
  await writeFile(join(root, 'research', 'notes', 'deep.md'), 'Deep note.', 'utf8');
  await writeFile(join(root, 'secret.txt'), 'not a skill', 'utf8');

  return root;
}

describe('SkillRegistry', () => {
  it('appends each named skill as a tagged section', async () => {
    const skills = new SkillRegistry(await skillRoot());
    const prompt = skills.augmentPrompt('Be brief.', ['research']);

    expect(prompt).toContain('Be brief.');
    expect(prompt).toContain('<skill name="research">');
    expect(prompt).toContain('Search carefully.');
  });

  it('leaves the prompt UNCHANGED when no skills are named', async () => {
    // Appending an empty preamble would tell the model skills are available
    // when none are.
    const skills = new SkillRegistry(await skillRoot());

    expect(skills.augmentPrompt('Be brief.', [])).toBe('Be brief.');
  });

  it('reads a nested file inside a skill', async () => {
    const skills = new SkillRegistry(await skillRoot());

    expect(skills.read('research', 'notes/deep.md')).toBe('Deep note.');
  });

  it('refuses a traversing PATH', async () => {
    const skills = new SkillRegistry(await skillRoot());

    expect(() => skills.read('research', '../secret.txt')).toThrowError(/stay inside/);
    expect(() => skills.read('research', 'notes/../../secret.txt')).toThrowError(/stay inside/);
  });

  it('refuses an absolute path', async () => {
    const skills = new SkillRegistry(await skillRoot());

    expect(() => skills.read('research', '/etc/passwd')).toThrowError(/stay inside/);
  });

  it('refuses a traversing NAME, before it is joined to anything', async () => {
    const skills = new SkillRegistry(await skillRoot());

    expect(() => skills.read('../', 'SKILL.md')).toThrowError(/not a valid name/);
    expect(() => skills.read('Research', 'SKILL.md')).toThrowError(/not a valid name/);
  });

  it('refuses a SYMLINK that points out of the skill', async () => {
    // The check the lexical ones cannot make: `notes/link.md` is lexically
    // innocent and resolves elsewhere.
    const root = await skillRoot();
    await symlink(join(root, 'secret.txt'), join(root, 'research', 'notes', 'link.md'));

    expect(() => new SkillRegistry(root).read('research', 'notes/link.md')).toThrowError(
      /outside the skill/,
    );
  });
});

// -- subagents ---------------------------------------------------------------

const modes = new ModeRegistry({
  default: 'parent',
  modes: {
    parent: {
      system_prompt: 'You delegate.',
      tools: [],
      max_steps: 8,
      subagents: { researcher: { mode: 'child', max_steps: 2 } },
    },
    child: { system_prompt: 'You research.', tools: [], max_steps: 2 },
  },
});

async function harnessed(): Promise<{ harness: PrismHarness; parent: Session }> {
  const directory = await mkdtemp(join(tmpdir(), 'prism-harness-sub-'));
  const harness = new PrismHarness({
    drivers: { memory: () => new MemorySessionStore(), files: () => new FileSessionStore(directory) },
    stores: { ephemeral: 'memory', durable: 'files' },
  });
  const parent = harness.for({ type: 'User', id: 1 }).session('support');
  await parent.usingMode('parent');
  await parent.usingProvider('anthropic');
  await parent.usingModel('m');

  return { harness, parent };
}

function runtimeReturning(text: string): AgentRuntime {
  return new AgentRuntime({
    client: async (): Promise<LlmResponse> => ({ text, finishReason: 'stop' }),
    modes,
    tools: new ToolRegistry(),
  });
}

describe('SubagentRunner', () => {
  it('runs the child and hands its text back as the tool result', async () => {
    const { harness, parent } = await harnessed();
    const runner = new SubagentRunner(harness, runtimeReturning('The child answered.'));
    const subagent = modes.resolve('parent').subagents.researcher!;
    const context = RunContext.root('root', new RunBudget(8));

    const tool = runner.tool(subagent, parent, context, 'root');

    expect(await tool.handle({ task: 'Find something' })).toBe('The child answered.');
  });

  it('gives the child its OWN session, under a different scope', async () => {
    // The parent is mid-run holding its own lock. A child asking for the
    // parent's address would wait on a lock its own caller holds.
    const { harness, parent } = await harnessed();
    const runner = new SubagentRunner(harness, runtimeReturning('done'));
    const subagent = modes.resolve('parent').subagents.researcher!;

    await runner.run(subagent, parent, RunContext.root('root', new RunBudget(8)), 'root', 'go');

    const child = harness.for(parent.participant).session(subagent.scopeUnder(parent.scope));

    expect(child.key()).not.toBe(parent.key());
    // The child ran: its own thread holds the turn.
    expect(await child.thread().count()).toBeGreaterThan(0);
    // The child's authority is its own mode, not the parent's.
    expect(await child.mode()).toBe('child');
  });

  it('REFUSES before spawning when the tree is already exhausted', async () => {
    // An exhausted tree must not pay for a session, a thread write and a
    // provider call to discover it is exhausted.
    const { harness, parent } = await harnessed();
    let called = 0;
    const runtime = new AgentRuntime({
      client: async (): Promise<LlmResponse> => {
        called += 1;

        return { text: 'x', finishReason: 'stop' };
      },
      modes,
      tools: new ToolRegistry(),
    });
    const context = RunContext.root('root', new RunBudget(2));
    context.ledger.recordSteps(2);

    const result = await new SubagentRunner(harness, runtime).run(
      modes.resolve('parent').subagents.researcher!,
      parent,
      context,
      'root',
      'go',
    );

    expect(called).toBe(0);
    expect(result.outcome).toBe('exhausted');
    expect(result.toToolResult()).toMatch(/did not run/);
  });

  it('reports a cancelled tree as cancelled, not merely exhausted', async () => {
    const { harness, parent } = await harnessed();
    const context = RunContext.root('root', new RunBudget(8));
    context.ledger.cancel('stopped by the user');

    const result = await new SubagentRunner(harness, runtimeReturning('x')).run(
      modes.resolve('parent').subagents.researcher!,
      parent,
      context,
      'root',
      'go',
    );

    expect(result.outcome).toBe('cancelled');
  });

  it('FRAMES a child failure rather than tearing down the parent run', async () => {
    // The parent is a legitimate audience for "that did not work", and
    // throwing would discard work the parent has already done.
    const { harness, parent } = await harnessed();
    const runtime = new AgentRuntime({
      client: async (): Promise<LlmResponse> => {
        throw new Error('the child provider is down');
      },
      modes,
      tools: new ToolRegistry(),
    });

    const result = await new SubagentRunner(harness, runtime).run(
      modes.resolve('parent').subagents.researcher!,
      parent,
      RunContext.root('root', new RunBudget(8)),
      'root',
      'go',
    );

    expect(result.outcome).toBe('failed');
    expect(result.toToolResult()).toContain('provider is down');
  });

  it('draws the child budget from what the TREE has left', async () => {
    const { harness, parent } = await harnessed();
    let steps = 0;
    const runtime = new AgentRuntime({
      client: async (): Promise<LlmResponse> => {
        steps += 1;

        return { text: 'again', finishReason: 'tool_calls', toolCalls: [] };
      },
      modes,
      tools: new ToolRegistry(),
    });
    const context = RunContext.root('root', new RunBudget(8));
    context.ledger.recordSteps(7);

    await new SubagentRunner(harness, runtime).run(
      modes.resolve('parent').subagents.researcher!,
      parent,
      context,
      'root',
      'go',
    );

    // One step left in the tree, so the child takes one — not the two its own
    // declaration asks for.
    expect(steps).toBe(1);
    expect(context.ledger.steps).toBe(8);
  });
});
