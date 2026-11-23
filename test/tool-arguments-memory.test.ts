import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {describe, expect, it} from 'vitest';

const DIST = new URL('../dist/index.js', import.meta.url).href;

// A hard limit on the heap of a separate process: the engine either fits or
// exits, so the result does not depend on when a collection happens to run.
const HEAP_LIMIT_MB = 72;

const SCRIPT = `
  import {mkdtempSync} from 'node:fs';
  import {tmpdir} from 'node:os';
  import {join} from 'node:path';
  globalThis.fetch = () => { throw new Error('A test tried to use the network.'); };
  const {AgentRuntime, FileSessionStore, MemorySessionStore, ModeRegistry, PrismHarness, ToolRegistry} = await import(${JSON.stringify(DIST)});
  const directory = mkdtempSync(join(tmpdir(), 'prism-arguments-'));
  const harness = new PrismHarness({drivers: {memory: () => new MemorySessionStore(), files: () => new FileSessionStore(directory)}, stores: {ephemeral: 'memory', durable: 'files'}});
  const session = harness.for({type: 'User', id: 1}).session('long');
  await session.usingMode('chat'); await session.usingProvider('synthetic'); await session.usingModel('offline');
  const replies = [
    {text: '', finishReason: 'tool_calls', toolCalls: [{id: 'call-1', name: 'lookup', arguments: '{"a":[' + '0,'.repeat(999999) + '0]}'}]},
    {text: 'done', finishReason: 'stop'},
  ];
  let modelCalls = 0;
  let toolRuns = 0;
  const tools = new ToolRegistry();
  tools.register({name: 'lookup', handle: () => { toolRuns += 1; return 'ran'; }});
  const runtime = new AgentRuntime({
    client: async () => { modelCalls += 1; return replies.shift(); },
    tools,
    modes: new ModeRegistry({default: 'chat', modes: {chat: {tools: ['lookup'], max_steps: 4}}}),
  });
  const response = await runtime.send(session, 'Look it up.');
  console.log(JSON.stringify({finish: response.finishReason, modelCalls, toolRuns}));
`;

describe('a long argument list', () => {
  it('completes a turn inside a bounded heap', () => {
    const run = spawnSync(
      process.execPath,
      [`--max-old-space-size=${HEAP_LIMIT_MB}`, '--input-type=module', '-e', SCRIPT],
      {cwd: fileURLToPath(new URL('..', import.meta.url)), encoding: 'utf8', timeout: 60_000},
    );

    expect(run.status).toBe(0);
    expect(JSON.parse(run.stdout.trim())).toEqual({finish: 'stop', modelCalls: 2, toolRuns: 1});
  }, 70_000);
});
