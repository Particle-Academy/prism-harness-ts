import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import {
  HarnessError,
  StoreTaskSource,
  type AgentTaskRecord,
  type Durability,
  type JsonObject,
  type SessionStore,
  type StoredAgentTask,
  type TaskOutcome,
  type TaskState,
} from '../src/index.js';

/**
 * The cross-language agent-task-claim corpus from `prism-parity`.
 *
 * A task list is DURABLE state that outlives the process which wrote it, and
 * nothing says the process reading it back is in the same language. So the list
 * itself is the shared surface: a one-tick disagreement about when a lease has
 * lapsed hands one task to two workers, and a disagreement about how
 * `claimed_by` is compared lets one worker close another's work. Neither
 * errors, and both look like ordinary operation — which is exactly the shape a
 * per-language suite cannot see, because each one asserts against the value its
 * own code produced.
 *
 * Two rows DIVERGE from the PHP reference and are recorded as divergences
 * rather than skipped, because a skip would hide the finding. See the block
 * comment above `perform()` and the manifest's `ts` gap.
 */

/** The recorded shape of one row's outcome. Codes, never prose — decision 0004. */
interface RowOutcome {
  outcome: 'ok' | 'refused';
  code: string | null;
  record: AgentTaskRecord | null;
  pending: number | null;
}

/** 0002: a language that cannot express a row SKIPS it, with a mandatory reason. */
interface RowSkipped {
  skipped: string;
}

type RowResult = RowOutcome | RowSkipped;

interface SeedTask {
  id: string;
  instruction: string;
  claimed_by?: string;
  claimed_until?: number;
  state?: TaskState;
}

interface RowWhen {
  op: 'claim' | 'release' | 'pending' | 'find' | 'claim_then_find';
  worker?: string;
  lease_seconds?: number;
  task_id?: string;
  outcome?: string;
}

interface Row {
  id: string;
  title: string;
  notes: string;
  given: { tasks: SeedTask[]; now: number };
  when: RowWhen;
  result: { php: RowResult | null; ts: RowResult | null; py: RowResult | null };
}

interface Corpus {
  cases: Row[];
}

/**
 * Recording mode: `PRISM_PARITY_CASES=<abs path to suites/agent-task-claim/cases.json>`.
 *
 * The recorder and the assertions share ONE implementation of `run()`
 * deliberately. A separate recording script is a second copy of the thing under
 * test, and this repository's whole argument is that unchecked duplicates
 * drift — a recorder that fills the corpus with values the suite never asserts
 * is the "generator that measures nothing" defect wearing different clothes.
 */
const recordingInto = process.env.PRISM_PARITY_CASES;

const fixturePath = fileURLToPath(new URL('./fixtures/agent-task-claim.json', import.meta.url));

const corpus = JSON.parse(readFileSync(recordingInto ?? fixturePath, 'utf8')) as Corpus;

/**
 * The smallest store that satisfies the contract, mirroring the generator's.
 *
 * Deliberately NOT `MemorySessionStore`: that one reports itself volatile, and
 * `StoreTaskSource` refuses a volatile store at construction. The lock is a
 * no-op because this process is single-threaded and the locking PRIMITIVE has
 * its own failure modes, explicitly out of this suite's scope.
 */
class CorpusStore implements SessionStore {
  readonly rows = new Map<string, JsonObject>();

  async get(key: string): Promise<JsonObject | null> {
    return this.rows.get(key) ?? null;
  }

  async put(key: string, payload: JsonObject): Promise<void> {
    this.rows.set(key, payload);
  }

  async forget(key: string): Promise<void> {
    this.rows.delete(key);
  }

  async withLock<T>(_key: string, callback: () => T | Promise<T>): Promise<T> {
    return callback();
  }

  durability(): Durability {
    return 'durable';
  }
}

const CORPUS_KEY = 'corpus';

/**
 * The row's `given.tasks`, as the five canonical keys.
 *
 * `state` is derived the same way the reference derives it — a seeded row that
 * names a holder and no state is `claimed`. Seeded DIRECTLY rather than through
 * `add()`, so a row may describe a state the public API would refuse to build.
 * A corpus that can only express states the implementation is willing to create
 * cannot test what it does when it meets one it did not.
 */
function seed(task: SeedTask): AgentTaskRecord {
  const claimedBy = task.claimed_by ?? null;

  return {
    claimed_by: claimedBy,
    claimed_until: task.claimed_until ?? null,
    id: task.id,
    instruction: task.instruction,
    state: task.state ?? (claimedBy !== null ? 'claimed' : 'todo'),
  };
}

function stringField(when: RowWhen, field: 'worker' | 'task_id' | 'outcome'): string {
  const value = when[field];

  if (typeof value !== 'string') {
    throw new Error(`the corpus row's when.${field} is ${String(value)} rather than a string`);
  }

  return value;
}

/**
 * Drive one row's `when` against a seeded source.
 *
 * ## The outcome is handed over UNVALIDATED, on purpose
 *
 * The reference generator writes `TaskOutcome::from($when['outcome'])`, because
 * PHP has no other way to produce a `TaskOutcome` — the conversion is forced by
 * the type system at the call site, and an outcome that is not one of the two
 * is rejected there, before the package is reached.
 *
 * TypeScript's `TaskOutcome` is a union of string literals and has NO runtime
 * existence, so the same call site is a cast that costs nothing at run time.
 * That is not the runner being sloppy: it is the honest translation of what a
 * caller reading an outcome out of JSON, a queue payload or an HTTP body
 * actually does in this language. Supplying the guard HERE would measure the
 * runner rather than the port, and record agreement the package has not earned
 * — the exact defect this repository exists to catch, one level up.
 *
 * What the port does with it is recorded in `result.ts` for atc-0011 and
 * atc-0012, and asserted below.
 */
async function perform(source: StoreTaskSource, when: RowWhen): Promise<StoredAgentTask | null> {
  switch (when.op) {
    case 'claim':
      return source.claim(stringField(when, 'worker'), when.lease_seconds);

    case 'find':
      return source.find(stringField(when, 'task_id'));

    case 'pending':
      return null;

    case 'release': {
      const task = await source.find(stringField(when, 'task_id'));

      if (task === null) return null;

      await source.release(task, stringField(when, 'worker'), stringField(when, 'outcome') as TaskOutcome);

      return source.find(stringField(when, 'task_id'));
    }

    case 'claim_then_find': {
      await source.claim(stringField(when, 'worker'), when.lease_seconds);

      return source.find(stringField(when, 'task_id'));
    }

    default:
      throw new Error(`Unknown op ${String(when.op)}`);
  }
}

/**
 * Seed the row's list, run its `when`, and record the outcome.
 *
 * Time is FROZEN at `given.now` through the source's injectable clock. Without
 * that every expiry row would be racing the wall clock and the boundary rows —
 * atc-0005 and atc-0006, one tick apart — would mean nothing.
 *
 * Returns the store as well, so a test can look at what the list actually holds
 * afterwards rather than only at what the call returned.
 */
async function execute(row: Row): Promise<{ result: RowOutcome; store: CorpusStore }> {
  const store = new CorpusStore();

  // The key is handed to the constructor in this port, where the reference
  // derives it from a list name — so the mistake the generator made (seeding a
  // DIFFERENT key from the one the source reads) cannot be made here in that
  // exact form. The equivalent one can: `#read()` returns an empty list for any
  // payload whose `tasks` is not an array, so a seed under the wrong wrapper
  // key produces "outcome ok, pending 0, no record" for every row — entirely
  // plausible, and an answer to nothing.
  const source = new StoreTaskSource(store, CORPUS_KEY, { now: () => row.given.now });

  await store.put(CORPUS_KEY, { tasks: row.given.tasks.map(seed) as unknown as JsonObject[] });

  const first = row.given.tasks[0];

  if (first !== undefined && (await source.find(first.id)) === null) {
    throw new Error(
      `SEEDING FAILED for ${row.id}: the source cannot see the tasks it was given. ` +
        'Every row below would record "nothing happened" as though it were an answer.',
    );
  }

  try {
    const record = await perform(source, row.when);

    return {
      result: {
        outcome: 'ok',
        code: null,
        record: record?.toRecord() ?? null,
        pending: await source.pending(),
      },
      store,
    };
  } catch (error) {
    // The code is the contract; the sentence is explicitly not (0004).
    if (!(error instanceof HarnessError)) throw error;

    return { result: { outcome: 'refused', code: error.code, record: null, pending: null }, store };
  }
}

async function run(row: Row): Promise<RowOutcome> {
  return (await execute(row)).result;
}

/**
 * What the list ACTUALLY holds afterwards, not what the call returned.
 *
 * Refuses rather than defaults when the key holds no task array: a helper that
 * returned `[]` there would let every assertion below pass against a store
 * nothing was ever written to.
 */
function storedTasks(store: CorpusStore): AgentTaskRecord[] {
  const tasks = store.rows.get(CORPUS_KEY)?.tasks;

  if (!Array.isArray(tasks)) {
    throw new Error('the corpus store holds no task list at the key the source was given');
  }

  return tasks as unknown as AgentTaskRecord[];
}

/** Keys sorted, no whitespace — the suite's declared comparison mode. */
function canonical(value: unknown): string {
  if (value === null || value === undefined) return 'null';
  if (typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;

  const entries = Object.entries(value as Record<string, unknown>).sort(([a], [b]) =>
    a < b ? -1 : a > b ? 1 : 0,
  );

  return `{${entries.map(([key, held]) => `${JSON.stringify(key)}:${canonical(held)}`).join(',')}}`;
}

const isSkipped = (result: RowResult | null): result is RowSkipped =>
  result !== null && typeof result === 'object' && 'skipped' in result;

const comparable = corpus.cases.filter((row) => !isSkipped(row.result.php));
const referenceSkipped = corpus.cases.filter((row) => isSkipped(row.result.php));

describe.skipIf(Boolean(recordingInto))('the cross-language agent-task-claim corpus', () => {
  it('is the whole suite, not a subset someone trimmed to green', () => {
    expect(corpus.cases).toHaveLength(21);
  });

  it('has a recorded half for this language on EVERY row', () => {
    // A null `ts` is an unrun row, and an unrun row asserted against nothing is
    // indistinguishable from a passing one.
    expect(corpus.cases.filter((row) => row.result.ts === null).map((row) => row.id)).toEqual([]);
  });

  it.each(corpus.cases)('$id produces this language’s recorded result ($title)', async (row) => {
    expect(canonical(await run(row))).toBe(canonical(row.result.ts));
  });

  it('fails LOUDLY when the seed does not land where the source reads', async () => {
    // The guard is the reason the recorded numbers mean anything, so it is
    // checked rather than asserted. Written as the mistake actually available
    // in this port: the right key, the wrong wrapper.
    const row = corpus.cases.find((entry) => entry.id === 'atc-0001')!;
    const store = new CorpusStore();
    const source = new StoreTaskSource(store, CORPUS_KEY, { now: () => row.given.now });

    await store.put(CORPUS_KEY, { list: row.given.tasks.map(seed) as unknown as JsonObject[] });

    expect(await source.find(row.given.tasks[0]!.id)).toBeNull();
    expect(await source.pending()).toBe(0);
  });
});

describe.skipIf(Boolean(recordingInto))('agreement with the PHP reference', () => {
  const agreeing = comparable.filter((row) => canonical(row.result.ts) === canonical(row.result.php));
  const diverging = comparable.filter((row) => canonical(row.result.ts) !== canonical(row.result.php));

  it.each(agreeing)('$id agrees with the reference ($title)', async (row) => {
    expect(canonical(await run(row))).toBe(canonical(row.result.php));
  });

  it('agrees on 18 of the 20 rows the reference can express', () => {
    expect(agreeing).toHaveLength(18);
    expect(comparable).toHaveLength(20);
  });

  it('diverges on exactly the rows the manifest names', () => {
    expect(diverging.map((row) => row.id)).toEqual(['atc-0011', 'atc-0012']);
  });

  it.each(diverging)('$id refuses like the reference but under a DIFFERENT code ($title)', async (row) => {
    // Two assertions, and both matter. Recording a row as a "divergence" would
    // otherwise let a real hole — an outcome accepted outright — hide behind
    // the word. The refusal is asserted first, the disagreement second.
    const produced = await run(row);

    expect(produced.outcome).toBe('refused');
    expect(produced.code).not.toBe((row.result.php as RowOutcome).code);
  });

  it('refuses the invalid outcome only at the READER, after the write has landed', async () => {
    // The whole content of the divergence, stated as a fact about the stored
    // list rather than as a code comparison.
    //
    // `release()` validates the worker id and NOT the outcome: `TaskOutcome` is
    // a compile-time union, so a string from a queue payload or a JSON config
    // reaches `makeRecord()` unchallenged and is written to the durable list as
    // a state that is not one of the four. Nothing refuses until something
    // tries to READ the list back — at which point `taskRecordFrom()` refuses
    // it as unmappable, for this port AND for the other two, because the bytes
    // on the shared surface are now junk.
    for (const id of ['atc-0011', 'atc-0012']) {
      const row = corpus.cases.find((entry) => entry.id === id)!;
      const { result, store } = await execute(row);

      expect(result.code).toBe('unmappable_content');

      const written = storedTasks(store)[0]!;

      expect(written.state).toBe(row.when.outcome);
      expect(written.claimed_by).toBeNull();
    }
  });

  it('EXPRESSES the row the reference cannot, rather than inheriting its skip', async () => {
    // atc-0017 is skipped for PHP: `claim()` declares `?int`, so a fractional
    // lease is rejected by the type system before any guard in the package
    // runs. TypeScript has no such constraint — `90.4` is an ordinary `number`
    // and reaches `assertLease`, which refuses it as this port's own decision
    // rather than the language's. Recorded as a real result; a skip inherited
    // from another language's type system would assert nothing here.
    expect(referenceSkipped.map((row) => row.id)).toEqual(['atc-0017']);

    const row = referenceSkipped[0]!;

    expect(isSkipped(row.result.ts)).toBe(false);
    expect(await run(row)).toEqual({
      outcome: 'refused',
      code: 'task_lease_invalid',
      record: null,
      pending: null,
    });
  });
});

describe.skipIf(Boolean(recordingInto))('where the THREE languages stand', () => {
  // Asserted against the VENDORED corpus, so it is a statement about what the
  // fixture records rather than a second implementation of Python. It goes red
  // when someone re-vendors after the picture changed, which is the only moment
  // it needs to be read again.
  const disagreeing = corpus.cases.filter(
    (row) =>
      canonical(row.result.php) !== canonical(row.result.ts) ||
      canonical(row.result.ts) !== canonical(row.result.py),
  );

  it('has all three halves recorded — an unrun half is not an agreeing one', () => {
    expect(corpus.cases.filter((row) => row.result.py === null).map((row) => row.id)).toEqual([]);
  });

  it('has THREE rows the languages answer differently, and they are not the same three', () => {
    expect(disagreeing.map((row) => row.id)).toEqual(['atc-0011', 'atc-0012', 'atc-0017']);
  });

  it('leaves this port ALONE on the outcome rows: PHP and Python refuse at the door', () => {
    // The isolation matters more than the disagreement. Python's `TaskOutcome`
    // is a runtime enum like PHP's, so both refuse `complete` and ` done`
    // before anything is written. This port is the only one of the three that
    // writes first — see G-39.
    for (const id of ['atc-0011', 'atc-0012']) {
      const row = corpus.cases.find((entry) => entry.id === id)!;

      expect(canonical(row.result.py)).toBe(canonical(row.result.php));
      expect((row.result.ts as RowOutcome).code).toBe('unmappable_content');
    }
  });

  it('gives atc-0017 THREE different answers, and this port has the strict one', () => {
    // PHP skips it (its type system rejects `90.4` before any guard), this port
    // REFUSES it, and the corpus records Python ACCEPTING it — `claimed_until`
    // lands on 1735689690, which is `now + 90`. A fractional lease truncated to
    // a whole one is a configuration silently becoming a different
    // configuration, which is what the row's title says must not happen. Not
    // this port's finding to fix; recorded here so a re-vendor cannot bury it.
    const row = corpus.cases.find((entry) => entry.id === 'atc-0017')!;

    expect(isSkipped(row.result.php)).toBe(true);
    expect((row.result.ts as RowOutcome).code).toBe('task_lease_invalid');
    expect(row.result.py).toMatchObject({
      outcome: 'ok',
      record: { claimed_until: 1735689690 },
    });
  });
});

describe.skipIf(Boolean(recordingInto))('the properties the rows exist to pin', () => {
  it('holds a lease to be over AT its expiry, not one tick after', async () => {
    // atc-0005 and atc-0006 are one tick apart and both reclaim. A port using
    // `<` rather than `<=` passes atc-0005 and fails only atc-0006, which is
    // the whole reason the boundary row is separate.
    for (const id of ['atc-0005', 'atc-0006']) {
      const row = corpus.cases.find((entry) => entry.id === id)!;
      const produced = await run(row);

      expect(produced.record?.claimed_by).toBe('w-2');
    }
  });

  it('returns a lapsed lease to todo and NEVER to failed', async () => {
    const row = corpus.cases.find((entry) => entry.id === 'atc-0018')!;
    const { store } = await execute(row);

    // `pending()` does not write, so the seeded bytes must be untouched — the
    // expired lease is counted as claimable WITHOUT the row being rewritten.
    expect(storedTasks(store)[1]!.state).toBe('claimed');
    expect(await run(row)).toMatchObject({ pending: 2 });
  });

  it('writes claimed_until as an INTEGER, not a float', async () => {
    // JavaScript has no int/float distinction, so `1735689900.0` passes every
    // equality assertion in this file and still writes different bytes from the
    // other two ports. Only this check can see it.
    const row = corpus.cases.find((entry) => entry.id === 'atc-0021')!;
    const produced = await run(row);

    expect(Number.isInteger(produced.record?.claimed_until)).toBe(true);
    expect(canonical(produced.record)).toContain('"claimed_until":1735689900');
  });

  it('keeps claimed_by and claimed_until PRESENT-AND-NULL on an unclaimed task', async () => {
    // An optional property holding `undefined` is dropped by JSON.stringify,
    // and a dropped key is different bytes. Asserted on the serialised form,
    // because that is the only place the difference shows.
    const row = corpus.cases.find((entry) => entry.id === 'atc-0020')!;
    const produced = await run(row);

    expect(canonical(produced.record)).toBe(
      '{"claimed_by":null,"claimed_until":null,"id":"t-1","instruction":"first","state":"todo"}',
    );
  });

  it('compares a worker id EXACTLY, with no trimming', async () => {
    // `w-1 ` is a different worker from `w-1`. PHP's `trim`, JavaScript's
    // `String.prototype.trim` and Python's `str.strip` each strip a different
    // set of codepoints — the G-36 shape, asked about a different field.
    const row = corpus.cases.find((entry) => entry.id === 'atc-0013')!;

    expect(await run(row)).toMatchObject({ outcome: 'refused', code: 'task_lease_not_held' });
  });
});

/**
 * Recording mode. Off unless `PRISM_PARITY_CASES` names the suite's cases.json:
 *
 * ```sh
 * PRISM_PARITY_CASES=../prism-parity/suites/agent-task-claim/cases.json npx vitest run agent-task-claim
 * node -e "…"   # then re-vendor: the fixture is a byte copy of the suite file
 * ```
 *
 * The vendored copy is rewritten in the same pass, because a fixture that lags
 * the suite it was copied from is a corpus that silently tests an older
 * contract.
 */
describe.runIf(Boolean(recordingInto))('recording this language’s half', () => {
  it('fills result.ts for every row and re-vendors the fixture', async () => {
    const text = readFileSync(recordingInto!, 'utf8');
    const document = JSON.parse(text) as Corpus;

    for (const row of document.cases) row.result.ts = await run(row);

    // The existing indentation is preserved rather than imposed: this file is
    // also written by a PHP generator, and reformatting it would bury the one
    // changed key under a whole-file diff.
    const indent = /\n(\s+)"/.exec(text)?.[1]?.length ?? 2;
    const rendered = `${JSON.stringify(document, null, indent)}\n`;

    writeFileSync(recordingInto!, rendered);
    writeFileSync(fixturePath, rendered);

    expect(document.cases.filter((row) => row.result.ts === null)).toEqual([]);
  });
});
