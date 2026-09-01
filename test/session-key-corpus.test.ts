import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { Session, type SessionStore } from '../src/index.js';

/**
 * The cross-language session-key corpus from `prism-parity`.
 *
 * This is the most consequential string in the ecosystem and the one whose
 * drift is hardest to notice. A session key is an ADDRESS: it is what lets a
 * PHP application and this agent resolve the same conversation, and it is what
 * the store is keyed by.
 *
 * If it drifted, nothing would error. Each language would read and write its
 * own key perfectly happily, and the two would simply stop seeing each other's
 * turns — a conversation that appears empty, with no exception, no log line and
 * no failing test anywhere in either codebase. That is precisely the shape a
 * per-language suite cannot see, because each one asserts against the key its
 * own code produced.
 */
interface KeyCase {
  id: string;
  title: string;
  participant: { type: string; id: string };
  scope: string;
  key: { php: string; ts: string; py: string };
  agrees: boolean;
  notes: string;
}

const corpus = JSON.parse(
  readFileSync(new URL('./fixtures/harness-session-key.json', import.meta.url), 'utf8'),
) as { cases: KeyCase[] };

/** Never read from. `key()` derives from the participant and scope alone. */
const unused: SessionStore = {
  get: async () => null,
  put: async () => {},
  forget: async () => {},
  withLock: async (_key, callback) => callback(),
  durability: () => 'volatile',
};

const keyOf = (entry: KeyCase): string =>
  new Session({
    participant: entry.participant,
    scope: entry.scope,
    ephemeral: unused,
    durable: unused,
  }).key();

describe('the cross-language session-key corpus', () => {
  it('is the whole suite, not a subset someone trimmed to green', () => {
    expect(corpus.cases).toHaveLength(9);
  });

  it.each(corpus.cases)('$id resolves to the reference address ($title)', (entry) => {
    expect(keyOf(entry)).toBe(entry.key.php);
  });

  it('agrees with the reference on EVERY row', () => {
    expect(corpus.cases.filter((entry) => !entry.agrees)).toEqual([]);
  });

  it('gives two participant TYPES with the same id different addresses', () => {
    // User 7 and Team 7 must not share a conversation, and the hashed type
    // segment is the only thing keeping them apart. Asserted directly rather
    // than inferred from two rows happening to differ.
    const user = corpus.cases.find((entry) => entry.id === 'key-0001');
    const team = corpus.cases.find((entry) => entry.id === 'key-0004');

    expect(user!.participant.id).toBe(team!.participant.id);
    expect(keyOf(user!)).not.toBe(keyOf(team!));
  });

  it('hashes the type as BYTES, so non-ASCII agrees across languages', () => {
    // sha1 is over bytes. A language hashing UTF-16 code units, or one that
    // latin-1'd the string first, produces a different digest from source that
    // looks identical in an editor.
    const entry = corpus.cases.find((row) => row.id === 'key-0008')!;

    expect(keyOf(entry)).toBe(entry.key.php);
  });

  it('does not treat a zero id as absent', () => {
    // Falsy in all three languages. A truthiness check anywhere on the id path
    // silently produces a different address, or an empty segment.
    const entry = corpus.cases.find((row) => row.id === 'key-0006')!;

    expect(keyOf(entry)).toContain(':0:');
  });
});
