import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { HarnessError } from '../src/index.js';
import { bindingWorld, type BindingSpec, type RunSpec } from './support/binding-world.js';

/**
 * The cross-language agent-binding corpus (G-321 E).
 *
 * Which provider and model a run, and a subagent's child run, ACTUALLY
 * request, and which delegations are refused before any request. Every row's
 * `expected` was written by hand before any language recorded; this runner
 * drives the real runtime and compares what the client was asked for.
 */
interface BindingCase {
  id: string;
  title: string;
  global: BindingSpec['global'];
  modes: BindingSpec['modes'];
  agents: BindingSpec['agents'];
  run: RunSpec;
  expected: unknown;
}

const corpus = JSON.parse(
  readFileSync(new URL('./fixtures/harness-agent-binding.json', import.meta.url), 'utf8'),
) as { cases: BindingCase[] };

async function record(row: BindingCase): Promise<unknown> {
  const world = await bindingWorld({ global: row.global, modes: row.modes, agents: row.agents }, row.run);
  try {
    await world.send();
  } catch (error) {
    if (world.requests.length === 0 && error instanceof HarnessError && error.code === 'mode_malformed') {
      return { refused: 'config', parent: null, child: null };
    }
    throw error;
  }
  const at = (index: number) => {
    const seen = world.requests[index];
    return seen === undefined ? null : { provider: seen.provider, model: seen.model };
  };

  return { refused: null, parent: at(0), child: row.run.delegate ? at(1) : null };
}

describe('harness-agent-binding corpus', () => {
  it('has rows', () => {
    expect(corpus.cases.length).toBeGreaterThan(0);
  });

  for (const row of corpus.cases) {
    it(`${row.id}: ${row.title}`, async () => {
      expect(await record(row)).toEqual(row.expected);
    });
  }
});
