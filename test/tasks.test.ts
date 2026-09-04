import { mkdtemp } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { describe, expect, it } from 'vitest';
import {
  DEFAULT_LEASE_SECONDS,
  FileSessionStore,
  HarnessError,
  MemorySessionStore,
  RunBudget,
  RunLedger,
  Session,
  StoreTaskSource,
  TASK_STATES,
  ToolAuthorizer,
  agentCompletionTool,
  asAgentTask,
  attributesToRecord,
  canonicalTaskJson,
  isTaskState,
  isTerminalState,
  recordToAttributes,
  taskRecordFrom,
  withAgentTask,
  type AgentTaskAttributeSource,
  type AgentTaskAttributes,
  type AgentTask,
  type AgentTaskRecord,
  type AgentTaskSource,
  type Durability,
  type JsonObject,
  type SessionStore,
  type TaskOutcome,
  type TaskState,
} from '../src/index.js';

/**
 * A store that persists for the life of a test and SAYS SO.
 *
 * `MemorySessionStore` reports itself volatile, which the task source refuses.
 * Declaring durability is an assertion about infrastructure, and inside one
 * test process this map genuinely does outlive everything that reads it — which
 * is the only claim `durable` makes. The refusal itself is tested against the
 * unmodified volatile store, below.
 */
class DurableMemoryStore extends MemorySessionStore {
  override durability(): Durability {
    return 'durable';
  }
}

async function fileStore(): Promise<FileSessionStore> {
  return new FileSessionStore(await mkdtemp(join(tmpdir(), 'prism-harness-tasks-')));
}

/**
 * The CODE, never the sentence.
 *
 * Prose is explicitly outside the contract (prism-parity decision 0004), so a
 * test that matched on wording would hold every improvement to a message
 * hostage. Returning a string rather than throwing also makes a passing call
 * report `no error` instead of a bare `undefined`.
 */
async function codeOf(work: () => Promise<unknown>): Promise<string> {
  try {
    await work();

    return 'no error';
  } catch (error) {
    return error instanceof HarnessError ? error.code : `not a HarnessError: ${String(error)}`;
  }
}

function codeOfSync(work: () => unknown): string {
  try {
    work();

    return 'no error';
  } catch (error) {
    return error instanceof HarnessError ? error.code : `not a HarnessError: ${String(error)}`;
  }
}

/**
 * Padding codepoints JavaScript's OWN `trim()` really strips.
 *
 * A padded fixture only proves "nothing is trimmed here" if the host language
 * would have trimmed it. Otherwise the case passes against a trimming
 * implementation and the test is decoration — which is not hypothetical: PHP's
 * version of these rows used U+00A0, and PHP's `trim()` leaves U+00A0 alone, so
 * it would have gone green against exactly the code it was written to catch.
 *
 * And the answer differs per language, so this list is NOT portable:
 *
 * | | PHP `trim` | Python `strip` | JS `trim` |
 * |---|---|---|---|
 * | U+0020 space | strips | strips | strips |
 * | U+0009 tab | strips | strips | strips |
 * | U+00A0 no-break space | **leaves** | strips | strips |
 * | U+3000 ideographic space | **leaves** | strips | strips |
 * | U+FEFF byte-order mark | **leaves** | **leaves** | strips |
 * | U+200B zero-width space | **leaves** | **leaves** | **leaves** |
 *
 * Written as code points rather than escapes because this file has had `\`
 * sequences mangled by tooling before, and a padding character that silently
 * became something else would make every row below assert the opposite of what
 * it claims.
 */
const PADDING: readonly string[] = [
  String.fromCharCode(0x20),
  String.fromCharCode(0x09),
  String.fromCharCode(0xa0),
  String.fromCharCode(0x3000),
  String.fromCharCode(0xfeff),
];

/** U+200B. Deliberately NOT in `PADDING`: no language strips it, so it proves nothing. */
const TOOTHLESS_PADDING = String.fromCharCode(0x200b);

describe('the padding fixtures themselves', () => {
  // A META-TEST, so the list above cannot quietly stop being adversarial. Every
  // row that uses `PADDING` is asserting "this is NOT trimmed"; if an entry
  // were one JavaScript ignores anyway, that row would pass against a trimming
  // implementation and nothing would say so.
  it('are all codepoints JavaScript would have stripped', () => {
    expect(PADDING.length).toBeGreaterThan(0);

    for (const pad of PADDING) {
      expect(`${pad}x${pad}`.trim()).toBe('x');
      expect(pad.trim()).toBe('');
    }
  });

  it('exclude U+200B, which would prove nothing', () => {
    // The negative control for the meta-test above: this is what a toothless
    // entry looks like, and it is why the list is checked rather than trusted.
    expect(TOOTHLESS_PADDING.trim()).toBe(TOOTHLESS_PADDING);
    expect(PADDING).not.toContain(TOOTHLESS_PADDING);
  });
});

/**
 * An `AgentTaskSource` that GUARDS NOTHING. Find it, set the state.
 *
 * Exactly what a third party writes from the signature `release(task, worker,
 * outcome)` — the name says "record what happened", and nothing about the
 * interface forces the implementation to check who is holding the task. That is
 * the whole point of this fixture: the completion tool's own pre-check is
 * invisible against `StoreTaskSource`, which already refuses a non-holder, so
 * deleting the check passes every other test in this file. Against this source
 * it does not.
 *
 * Deliberately NOT a subclass or a stub of the real one. A fixture that
 * inherited any of the guards could not prove their absence.
 */
class UnguardedTaskSource implements AgentTaskSource {
  #records: AgentTaskRecord[];

  readonly #initial: readonly AgentTaskRecord[];

  constructor(records: readonly AgentTaskRecord[]) {
    this.#initial = records.map((record) => ({ ...record }));
    this.#records = records.map((record) => ({ ...record }));
  }

  reset(): void {
    this.#records = this.#initial.map((record) => ({ ...record }));
  }

  record(id: string): AgentTaskRecord | undefined {
    return this.#records.find((candidate) => candidate.id === id);
  }

  async claim(worker: string): Promise<AgentTask | null> {
    const next = this.#records.find((candidate) => candidate.state === 'todo');

    if (next === undefined) return null;

    next.state = 'claimed';
    next.claimed_by = worker;

    return asAgentTask(recordToAttributes(next));
  }

  /** No holder check, no terminal check, no anything. */
  async release(task: AgentTask, _worker: string, outcome: TaskOutcome): Promise<void> {
    const found = this.#records.find((candidate) => candidate.id === task.id());

    if (found === undefined) return;

    found.state = outcome;
  }

  async pending(): Promise<number> {
    return this.#records.filter((candidate) => candidate.state === 'todo').length;
  }

  async find(id: string): Promise<AgentTask | null> {
    const found = this.record(id);

    return found === undefined ? null : asAgentTask(recordToAttributes(found));
  }
}

/**
 * A source whose tasks expose NO HOLDER. Three methods, exactly as the contract
 * allows.
 *
 * `AgentTask` is `id`, `instruction` and `state`; the lease lives on
 * `LeasedAgentTask`, which a consumer adapting an existing table need not
 * implement. So this is a CONFORMING source, not a broken one, and the
 * completion tool has to decide what an unknowable holder means. Treating it as
 * "no mismatch, therefore allowed" is reading silence as permission.
 */
class HolderlessTaskSource implements AgentTaskSource {
  constructor(private readonly state: TaskState = 'claimed') {}

  async claim(): Promise<AgentTask | null> {
    return this.bare();
  }

  async release(): Promise<void> {
    // Unguarded on purpose, like the fixture below.
  }

  async pending(): Promise<number> {
    return 1;
  }

  async find(id: string): Promise<AgentTask | null> {
    return id === 'h-1' ? this.bare() : null;
  }

  /** Deliberately NOT `asAgentTask`, which would supply `claimedBy`. */
  private bare(): AgentTask {
    const state = this.state;

    return {
      id: () => 'h-1',
      instruction: () => 'no holder is readable here',
      state: () => state,
    };
  }
}

function ticking(start: number): { now: () => number; set: (value: number) => void } {
  let value = start;

  return { now: () => value, set: (next) => (value = next) };
}

// -- the contracts and the state machine -------------------------------------

describe('the four states', () => {
  it('has exactly four, and they are these four', () => {
    // A fifth state is not an extension, it is a value another language cannot
    // map. Pinned as a list rather than described in prose.
    expect(TASK_STATES).toEqual(['todo', 'claimed', 'done', 'failed']);
  });

  it('recognises the four and refuses anything else', () => {
    for (const state of TASK_STATES) expect(isTaskState(state)).toBe(true);

    // The negative control. Every one of these is a state some queue has.
    for (const other of ['cancelled', 'retrying', 'skipped', 'pending', '', 'TODO', null, 7]) {
      expect(isTaskState(other)).toBe(false);
    }
  });

  it('treats done and failed as terminal, and the other two as not', () => {
    expect(isTerminalState('done')).toBe(true);
    expect(isTerminalState('failed')).toBe(true);
    expect(isTerminalState('todo')).toBe(false);
    expect(isTerminalState('claimed')).toBe(false);
  });
});

// -- canonical JSON ----------------------------------------------------------

describe('canonical JSON', () => {
  const unclaimed: AgentTaskRecord = {
    claimed_by: null,
    claimed_until: null,
    id: 't-1',
    instruction: '…',
    state: 'todo',
  };

  it('emits the bytes the spec pins for an unclaimed task', () => {
    // Byte-for-byte, from specs/agent-task-lists.md. Compared as a STRING, so
    // key order is asserted and not merely the set of keys — `toEqual` on two
    // objects would pass with the keys in any order at all.
    expect(canonicalTaskJson(unclaimed)).toBe(
      '{"claimed_by":null,"claimed_until":null,"id":"t-1","instruction":"…","state":"todo"}',
    );
  });

  it('sorts the keys however the record was built', () => {
    const scrambled = {
      state: 'todo',
      instruction: '…',
      id: 't-1',
      claimed_until: null,
      claimed_by: null,
    } as AgentTaskRecord;

    // The negative control: insertion order is NOT what comes out.
    expect(JSON.stringify(scrambled)).not.toBe(canonicalTaskJson(unclaimed));
    expect(canonicalTaskJson(scrambled)).toBe(canonicalTaskJson(unclaimed));
  });

  it('writes claimed_until as an integer, not a formatted date', () => {
    expect(
      canonicalTaskJson({
        claimed_by: 'worker-1',
        claimed_until: 1_700_000_300,
        id: 't-1',
        instruction: 'ship it',
        state: 'claimed',
      }),
    ).toBe(
      '{"claimed_by":"worker-1","claimed_until":1700000300,"id":"t-1","instruction":"ship it","state":"claimed"}',
    );
  });

  it('escapes neither slashes nor non-ASCII', () => {
    // JavaScript gets this right by default where PHP and Python do not; the
    // row exists so a future encoder change cannot quietly diverge from them.
    expect(
      canonicalTaskJson({ ...unclaimed, instruction: 'read https://plabs.gen/lab — 日本語 ü' }),
    ).toContain('"instruction":"read https://plabs.gen/lab — 日本語 ü"');
  });

  describe('undefined versus null — the TypeScript trap', () => {
    // A record whose claim fields are `undefined` rather than `null`. The types
    // forbid this, so it is forced here: the point is what happens when it
    // arrives anyway, from a store or from untyped consumer code.
    const leaky = {
      claimed_by: undefined,
      claimed_until: undefined,
      id: 't-1',
      instruction: '…',
      state: 'todo',
    } as unknown as AgentTaskRecord;

    it('DROPS the keys under a naive stringify — the failure being guarded against', () => {
      // The negative control, and the whole reason the guard exists. PHP has
      // one absent value; JavaScript has two, and one of them deletes the key.
      expect(JSON.stringify(leaky)).toBe('{"id":"t-1","instruction":"…","state":"todo"}');
    });

    it('keeps them present-and-null through the canonical encoder', () => {
      expect(canonicalTaskJson(leaky)).toBe(canonicalTaskJson(unclaimed));
    });

    it('keeps them present-and-null through taskRecordFrom', () => {
      const rebuilt = taskRecordFrom(leaky as unknown as JsonObject);

      expect(rebuilt.claimed_by).toBeNull();
      expect(rebuilt.claimed_until).toBeNull();
      expect('claimed_by' in rebuilt).toBe(true);
      expect('claimed_until' in rebuilt).toBe(true);
      expect(canonicalTaskJson(rebuilt)).toBe(canonicalTaskJson(unclaimed));
    });

    it('keeps them present-and-null through attributesToRecord', () => {
      const attributes = {
        id: 't-1',
        instruction: '…',
        state: 'todo',
        claimedBy: undefined,
        claimedUntil: undefined,
      } as unknown as AgentTaskAttributes;

      expect(canonicalTaskJson(attributesToRecord(attributes))).toBe(canonicalTaskJson(unclaimed));
    });
  });

  it('round-trips through the attribute shape without changing the bytes', () => {
    const claimed: AgentTaskRecord = {
      claimed_by: 'w-1',
      claimed_until: 1_700_000_300,
      id: 't-9',
      instruction: 'do the thing',
      state: 'claimed',
    };

    expect(canonicalTaskJson(attributesToRecord(recordToAttributes(claimed)))).toBe(
      canonicalTaskJson(claimed),
    );
  });

  it('refuses a corrupt stored row rather than defaulting it', () => {
    // Defaulting a missing state to `todo` would hand a half-finished task back
    // out as though it had never been attempted.
    expect(codeOfSync(() => taskRecordFrom({ instruction: 'x', state: 'todo' }))).toBe(
      'unmappable_content',
    );
    expect(codeOfSync(() => taskRecordFrom({ id: 't-1', state: 'todo' }))).toBe('unmappable_content');
    expect(codeOfSync(() => taskRecordFrom({ id: 't-1', instruction: 'x' }))).toBe('unmappable_content');
    expect(codeOfSync(() => taskRecordFrom({ id: 't-1', instruction: 'x', state: 'cancelled' }))).toBe(
      'unmappable_content',
    );

    // The positive control: a well-formed row goes through.
    expect(codeOfSync(() => taskRecordFrom({ id: 't-1', instruction: 'x', state: 'todo' }))).toBe(
      'no error',
    );
  });
});

// -- durability --------------------------------------------------------------

describe('a volatile store is refused', () => {
  it('refuses to construct against a store that reports itself volatile', () => {
    expect(codeOfSync(() => new StoreTaskSource(new MemorySessionStore(), 'k'))).toBe(
      'unsafe_state_configuration',
    );
  });

  it('constructs against a durable one — the positive control', () => {
    expect(codeOfSync(() => new StoreTaskSource(new DurableMemoryStore(), 'k'))).toBe('no error');
  });

  it('refuses at CONSTRUCTION, before anything can be written', async () => {
    const store = new MemorySessionStore();

    try {
      // eslint-disable-next-line no-new
      new StoreTaskSource(store, 'k');
    } catch {
      // expected
    }

    // Nothing was written on the way to the refusal.
    expect(await store.get('k')).toBeNull();
  });

  it('refuses through a session whose durable slot is volatile', () => {
    const volatileStore = new MemorySessionStore();
    const session = new Session({
      participant: { type: 'User', id: 1 },
      scope: 'tasks',
      ephemeral: new MemorySessionStore(),
      durable: volatileStore,
    });

    expect(codeOfSync(() => session.tasks())).toBe('unsafe_state_configuration');
  });

  it('works through a session whose durable slot is durable', async () => {
    const session = new Session({
      participant: { type: 'User', id: 1 },
      scope: 'tasks',
      ephemeral: new MemorySessionStore(),
      durable: await fileStore(),
    });

    const source = session.tasks();
    await source.add('write the thing');

    expect(await source.pending()).toBe(1);
  });
});

// -- the store-backed source, on both durable drivers ------------------------

describe.each<[string, () => Promise<SessionStore>]>([
  ['durable memory', async () => new DurableMemoryStore()],
  ['file', fileStore],
])('StoreTaskSource on the %s store', (_name, make) => {
  async function source(now?: () => number): Promise<StoreTaskSource> {
    return new StoreTaskSource(await make(), 'tasks', now === undefined ? {} : { now });
  }

  it('starts empty', async () => {
    expect(await (await source()).pending()).toBe(0);
  });

  it('hands tasks out in INSERTION ORDER', async () => {
    const tasks = await source();
    await tasks.addMany(['first', 'second', 'third']);

    const order = [
      (await tasks.claim('w'))?.instruction(),
      (await tasks.claim('w'))?.instruction(),
      (await tasks.claim('w'))?.instruction(),
    ];

    expect(order).toEqual(['first', 'second', 'third']);
    // The negative control: the order out is not the reverse, nor sorted.
    expect(order).not.toEqual(['third', 'second', 'first']);
  });

  it('returns null when nothing is claimable', async () => {
    const tasks = await source();
    await tasks.add('only one');

    expect((await tasks.claim('w-1'))?.id()).toBe('t-1');
    expect(await tasks.claim('w-2')).toBeNull();
  });

  it('writes the claim BEFORE it returns', async () => {
    // "Started and died" has to be distinguishable from "never started", and
    // that is only true if the store already says `claimed` by the time the
    // worker begins. Read through a SECOND source over the same store, which is
    // what a different worker would be holding.
    const store = await make();
    const worker = new StoreTaskSource(store, 'tasks');
    const observer = new StoreTaskSource(store, 'tasks');

    await worker.add('the work');

    expect((await observer.find('t-1'))?.state()).toBe('todo');

    const claimed = await worker.claim('w-1');

    expect(claimed?.state()).toBe('claimed');
    expect((await observer.find('t-1'))?.state()).toBe('claimed');
    expect((await observer.find('t-1'))?.claimedBy()).toBe('w-1');
    expect((await observer.find('t-1'))?.claimedUntil()).toBeTypeOf('number');
  });

  it('never hands the same task to two concurrent claims', async () => {
    const tasks = await source();
    await tasks.addMany(['a', 'b', 'c']);

    const claimed = await Promise.all([
      tasks.claim('w-1'),
      tasks.claim('w-2'),
      tasks.claim('w-3'),
      tasks.claim('w-4'),
      tasks.claim('w-5'),
    ]);

    const ids = claimed.filter((task) => task !== null).map((task) => task.id());

    expect(ids).toHaveLength(3);
    expect(new Set(ids).size).toBe(3);
    expect(claimed.filter((task) => task === null)).toHaveLength(2);
  });

  it('counts an expired lease as claimable and a live one as not', async () => {
    const clock = ticking(1_000);
    const tasks = new StoreTaskSource(await make(), 'tasks', { now: clock.now });
    await tasks.addMany(['a', 'b']);

    await tasks.claim('w-1', 60);

    expect(await tasks.pending()).toBe(1);

    clock.set(1_060);

    expect(await tasks.pending()).toBe(2);
  });

  it('returns an expired lease to todo, and NEVER to failed', async () => {
    const clock = ticking(1_000);
    const tasks = new StoreTaskSource(await make(), 'tasks', { now: clock.now });
    await tasks.add('the work');

    const claimed = await tasks.claim('w-1', 300);

    expect(claimed?.claimedUntil()).toBe(1_300);

    // One second before the lease ends it is still held. The positive control
    // for the assertion below: without it, a broken expiry check that always
    // returned true would pass.
    clock.set(1_299);
    expect((await tasks.find('t-1'))?.state()).toBe('claimed');
    expect(await tasks.claim('w-2')).toBeNull();

    // The instant the lease ends, it is over.
    clock.set(1_300);
    expect((await tasks.find('t-1'))?.state()).toBe('todo');
    expect((await tasks.find('t-1'))?.state()).not.toBe('failed');
    expect((await tasks.find('t-1'))?.claimedBy()).toBeNull();
    expect((await tasks.find('t-1'))?.claimedUntil()).toBeNull();

    const reclaimed = await tasks.claim('w-2');

    expect(reclaimed?.id()).toBe('t-1');
    expect(reclaimed?.claimedBy()).toBe('w-2');
  });

  it('persists the expiry even when the claim that noticed came up empty', async () => {
    const clock = ticking(1_000);
    const store = await make();
    const tasks = new StoreTaskSource(store, 'tasks', { now: clock.now });
    await tasks.add('the work');
    await tasks.claim('w-1', 60);

    clock.set(1_060);
    await tasks.claim('w-2');
    // w-2 took it. A third claim finds nothing, and must not undo anything.
    expect(await tasks.claim('w-3')).toBeNull();

    const raw = (await store.get('tasks')) as unknown as { tasks: AgentTaskRecord[] };

    expect(raw.tasks[0]?.claimed_by).toBe('w-2');
  });

  it('defaults the lease to five minutes', async () => {
    const clock = ticking(1_000);
    const tasks = new StoreTaskSource(await make(), 'tasks', { now: clock.now });
    await tasks.add('the work');

    expect((await tasks.claim('w-1'))?.claimedUntil()).toBe(1_000 + DEFAULT_LEASE_SECONDS);
    expect(DEFAULT_LEASE_SECONDS).toBe(300);
  });

  it('records done and failed, and refuses to release either again', async () => {
    const tasks = await source();
    await tasks.addMany(['a', 'b']);

    const first = await tasks.claim('w-1');
    const second = await tasks.claim('w-2');

    await tasks.release(first!, 'w-1', 'done');
    await tasks.release(second!, 'w-2', 'failed');

    expect((await tasks.find('t-1'))?.state()).toBe('done');
    expect((await tasks.find('t-2'))?.state()).toBe('failed');

    // Terminal means terminal, in both directions and to both outcomes.
    expect(await codeOf(() => tasks.release(first!, 'w-1', 'done'))).toBe('task_already_terminal');
    expect(await codeOf(() => tasks.release(first!, 'w-1', 'failed'))).toBe('task_already_terminal');
    expect(await codeOf(() => tasks.release(second!, 'w-2', 'done'))).toBe('task_already_terminal');

    // …and the refused call changed nothing.
    expect((await tasks.find('t-1'))?.state()).toBe('done');
  });

  it('refuses an outcome that is not one of the two, and WRITES NOTHING', async () => {
    // `TaskOutcome` is a compile-time union and does not exist at run time, so
    // the annotation on `release()` stops nothing: an outcome off a queue, an
    // HTTP body or a JSON config arrives as an ordinary string. The cast here
    // is what such a caller does, not a test convenience.
    //
    // Found by the cross-language corpus (G-39). Without the guard the string
    // was written into the list as a `state` that is not one of the four, and
    // nothing refused until a READER mapped the row back -- in any language.
    const store = await make();
    const tasks = new StoreTaskSource(store, 'tasks', { now: () => 1_000 });
    await tasks.add('the work');

    const claimed = await tasks.claim('w-1', 300);

    for (const outcome of ['complete', ' done', 'DONE', 'Done', '', 'todo', 'claimed']) {
      expect(await codeOf(() => tasks.release(claimed!, 'w-1', outcome as TaskOutcome))).toBe(
        'task_outcome_invalid',
      );
    }

    // Absent and null reach it too, from an untyped caller.
    expect(
      await codeOf(() => tasks.release(claimed!, 'w-1', undefined as unknown as TaskOutcome)),
    ).toBe('task_outcome_invalid');
    expect(await codeOf(() => tasks.release(claimed!, 'w-1', null as unknown as TaskOutcome))).toBe(
      'task_outcome_invalid',
    );

    // THE ASSERTION THAT MATTERS. A refusal that still wrote the row would be a
    // code comparison passing over a poisoned list: the state is untouched, the
    // holder is untouched, and the task is still claimable by its owner.
    const raw = (await store.get('tasks')) as unknown as { tasks: AgentTaskRecord[] };

    expect(raw.tasks[0]?.state).toBe('claimed');
    expect(raw.tasks[0]?.claimed_by).toBe('w-1');
    expect((await tasks.find('t-1'))?.state()).toBe('claimed');

    // The positive control: both real outcomes still go through.
    expect(await codeOf(() => tasks.release(claimed!, 'w-1', 'done'))).toBe('no error');
  });

  it('refuses the OUTCOME first when the call is wrong in two ways at once', async () => {
    // Observable ordering, chosen to match the reference: PHP converts the
    // string through the TaskOutcome enum in the argument expression, so the
    // outcome is judged before `release()` is entered. A blank worker AND an
    // invalid outcome therefore report the outcome in all three languages.
    const tasks = await source();
    const task = await tasks.add('the work');

    expect(await codeOf(() => tasks.release(task, '', 'complete' as TaskOutcome))).toBe(
      'task_outcome_invalid',
    );

    // …and each guard still fires on its own.
    expect(await codeOf(() => tasks.release(task, '', 'done'))).toBe('task_identifier_blank');
  });

  it('does not return a failed task to todo on its own', async () => {
    // Automatic retry is a policy; policy needs backoff and attempt counts, and
    // that is the scheduler this must not become.
    const tasks = await source();
    await tasks.add('a');
    await tasks.release((await tasks.claim('w-1'))!, 'w-1', 'failed');

    expect(await tasks.pending()).toBe(0);
    expect(await tasks.claim('w-2')).toBeNull();
    expect((await tasks.find('t-1'))?.state()).toBe('failed');
  });

  it('refuses to release a task nobody holds', async () => {
    const tasks = await source();
    const task = await tasks.add('a');

    expect(await codeOf(() => tasks.release(task, 'w-1', 'done'))).toBe('task_lease_not_held');

    // The positive control: claimed first, the same call succeeds.
    await tasks.claim('w-1');
    expect(await codeOf(() => tasks.release(task, 'w-1', 'done'))).toBe('no error');
  });

  it('stores claimed_until as an INTEGER, not a float', async () => {
    // JavaScript has no int/float distinction, so an equality assertion cannot
    // catch a fractional timestamp: `1000.5 === 1000.5` passes happily and the
    // bytes on the wire become `1000.5` where the other two ports write `1000`.
    // `Number.isInteger` on the STORED value is the only check that sees it.
    //
    // The clock is FRACTIONAL on purpose, and that is what makes this a check
    // rather than a claim: with the flooring removed, `1000.75 + 90` really is
    // written to the store as `1090.75` — PHP's equivalent mutation never
    // actually stored a float, so its test could not have failed. The assertion
    // below reads the STORED payload back, not the value handed to the writer.
    const store = await make();
    const tasks = new StoreTaskSource(store, 'tasks', { now: () => 1_000.75 });
    await tasks.add('the work');
    await tasks.claim('w-1', 90);

    const raw = (await store.get('tasks')) as unknown as { tasks: AgentTaskRecord[] };
    const until = raw.tasks[0]!.claimed_until!;

    expect(Number.isInteger(until)).toBe(true);
    expect(until).toBe(1_090);
    expect(JSON.stringify(until)).not.toContain('.');
    expect(canonicalTaskJson(raw.tasks[0]!)).toContain(`"claimed_until":${until},`);

    // And prove the store CAN carry a float, so the assertion above is testing
    // this code and not a limitation of the store.
    await store.put('float-probe', { value: 1_090.75 });
    expect(Number.isInteger(((await store.get('float-probe')) as { value: number }).value)).toBe(false);
  });

  it('refuses a lease of zero or less rather than clamping it', async () => {
    // Clamping to one second fails closed, which is why it was the first
    // answer. It is still a value quietly becoming a different value, and this
    // repository has already shipped a config that silently became a different
    // config and stayed green throughout.
    const tasks = await source();
    const task = await tasks.add('the work');

    // NaN and the infinities go in the same list, and they are not padding for
    // the sake of it. Every JavaScript number is a double, so they reach this
    // code through the same door as `0` — and `NaN <= 0` is FALSE, so a bare
    // positivity check lets NaN straight through to become a `claimed_until` of
    // NaN. `Number.isFinite` is what closes that, and this row is what says so.
    for (const lease of [0, -1, -300, 0.4, Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
      expect(await codeOf(() => tasks.claim('w-1', lease))).toBe('task_lease_invalid');
    }

    // The trap spelled out, so the row above cannot be trimmed back by someone
    // who reads `<= 0` and thinks NaN is covered by it.
    expect(Number.NaN <= 0).toBe(false);
    expect(Number.POSITIVE_INFINITY <= 0).toBe(false);

    // FRACTIONS are refused too, not truncated. "Truncation lands in the safe
    // direction" is the clamping argument restated, and it could never have
    // been honoured anyway: `claimed_until` is an integer timestamp in all
    // three languages, so a fractional lease was always going to become a
    // different lease.
    for (const fraction of [90.4, 0.5, 299.999, 1.000_000_1]) {
      expect(await codeOf(() => tasks.claim('w-1', fraction))).toBe('task_lease_invalid');
    }

    // THE UNTYPED DOOR, which is the one that actually mattered in the
    // reference. A lease read out of a JSON config is a STRING, the type
    // annotation never sees it, and `Math.trunc('90.4')` is 90 — so the guard
    // would have been defeated from inside the file that declares the setting.
    for (const untyped of ['90.4', '90', '300', true, null, [], {}]) {
      expect(
        await codeOf(() => tasks.claim('w-1', untyped as unknown as number)),
      ).toBe('task_lease_invalid');
    }

    // Nothing above claimed anything — not one of those refusals leaked past
    // the guard on its way to the store.
    expect((await tasks.find('t-1'))?.state()).toBe('todo');
    expect(await tasks.pending()).toBe(1);

    // The positive control, and the same rule on the extension path.
    expect(await codeOf(() => tasks.claim('w-1', 1))).toBe('no error');
    expect(
      await codeOf(() =>
        tasks.extendLease(task, 'w-1', 0, new RunLedger('r'), new RunBudget(8, null, 600)),
      ),
    ).toBe('task_lease_invalid');
    expect(
      await codeOf(() =>
        tasks.extendLease(task, 'w-1', 30.5, new RunLedger('r'), new RunBudget(8, null, 600)),
      ),
    ).toBe('task_lease_invalid');
  });

  it('accepts the lease default it SHIPS', async () => {
    // The shipped configuration, not only a direct call. A default that could
    // not pass the guard in front of it would be a package refusing its own
    // out-of-the-box behaviour — and `claim()` with no second argument is the
    // call every consumer makes first.
    const tasks = await source();
    await tasks.add('the work');

    expect(Number.isInteger(DEFAULT_LEASE_SECONDS)).toBe(true);
    expect(DEFAULT_LEASE_SECONDS).toBeGreaterThan(0);
    expect(DEFAULT_LEASE_SECONDS).toBe(300);

    expect(await codeOf(() => tasks.claim('w-1'))).toBe('no error');
    expect(Number.isInteger((await tasks.find('t-1'))!.claimedUntil()!)).toBe(true);
  });

  it('finds a task by id, which is all an external caller has', async () => {
    // On the CONTRACT, not just this class: `release()` takes a task and a tool
    // call carries only `{"id": "t-1"}`.
    const tasks: AgentTaskSource = await source();
    await tasks.claim('w-1');

    expect(await tasks.find('nope')).toBeNull();

    await (tasks as StoreTaskSource).add('the work');
    const found = await tasks.find('t-1');

    expect(found?.id()).toBe('t-1');
    expect(found?.instruction()).toBe('the work');

    // Driven entirely through the contract, holding only the id.
    const claimed = await tasks.claim('w-1');
    await tasks.release((await tasks.find(claimed!.id()))!, 'w-1', 'done');

    expect((await tasks.find('t-1'))?.state()).toBe('done');
  });

  it('refuses a blank worker id, and accepts a whitespace one', async () => {
    const tasks = await source();
    const task = await tasks.add('the work');

    expect(await codeOf(() => tasks.claim(''))).toBe('task_identifier_blank');
    expect(await codeOf(() => tasks.release(task, '', 'done'))).toBe('task_identifier_blank');
    expect(
      await codeOf(() =>
        tasks.extendLease(task, '', 30, new RunLedger('r'), new RunBudget(8, null, 600)),
      ),
    ).toBe('task_identifier_blank');

    // NO TRIMMING, deliberately, and the CODEPOINTS MATTER. Every entry in
    // `PADDING` is one JavaScript's own `trim()` would strip — the meta-test
    // above enforces that — so a trimming implementation turns each of these
    // into the empty string and gets refused, failing this row. A fixture the
    // host language ignores anyway would pass against exactly the code it was
    // written to catch, which is what happened to PHP's version of this test.
    //
    // This is the shape of prism-human-plus G-36, where each language's own
    // trim closed one hole and opened a different one.
    // COUNTED for the same reason as the padded outcomes: `vitest list` shows
    // this `it` and cannot show whether the loop inside it ran over five
    // fixtures or zero.
    let accepted = 0;

    for (const pad of PADDING) {
      await tasks.add(`work for ${JSON.stringify(pad)}`);

      // Claimed, not refused — and the id comes from the claim rather than
      // from the add, because `claim()` takes the OLDEST claimable task and
      // asserting against the one just added would be asserting the wrong row.
      const claimed = await tasks.claim(pad);

      expect(claimed).not.toBeNull();
      expect(claimed?.claimedBy()).toBe(pad);
      expect((await tasks.find(claimed!.id()))?.claimedBy()).toBe(pad);
      accepted += 1;
    }

    expect(accepted).toBe(PADDING.length);
    expect(accepted).toBeGreaterThanOrEqual(5);
  });

  it('refuses a blank task id, and a duplicate one', async () => {
    const tasks = await source();
    await tasks.add('the work', 'ticket-42');

    expect(await codeOf(() => tasks.add('x', ''))).toBe('task_identifier_blank');
    expect(await codeOf(() => tasks.add('x', 'ticket-42'))).toBe('duplicate_task_id');

    // The positive control, plus the no-trim rule again: a padded id is a
    // DIFFERENT id, not a duplicate.
    expect(await codeOf(() => tasks.add('x', 'ticket-43'))).toBe('no error');
    expect(await codeOf(() => tasks.add('x', 'ticket-42 '))).toBe('no error');
  });

  it('does not let a generated id land on a supplied one', async () => {
    const tasks = await source();
    await tasks.add('supplied', 't-1');
    const generated = await tasks.add('generated');

    expect(generated.id()).not.toBe('t-1');
    expect(await tasks.pending()).toBe(2);
  });

  it('refuses a release by anyone but the holder, without naming the holder', async () => {
    // A completion tool handed a source could otherwise close a task another
    // worker is midway through — and the refusal reaches a model as text, so it
    // must not hand over the identity of a peer worker either.
    const tasks = await source();
    const task = await tasks.add('the work');
    await tasks.claim('secret-holder-9');

    let message = '';

    try {
      await tasks.release(task, 'intruder', 'done');
    } catch (error) {
      message = error instanceof Error ? error.message : String(error);
    }

    expect(await codeOf(() => tasks.release(task, 'intruder', 'done'))).toBe('task_lease_not_held');
    expect(message).not.toContain('secret-holder-9');
    expect(message).toContain('intruder');

    // The capability question: the task is untouched, and the real holder can
    // still close it.
    expect((await tasks.find('t-1'))?.state()).toBe('claimed');
    expect(await codeOf(() => tasks.release(task, 'secret-holder-9', 'done'))).toBe('no error');
  });

  it('refuses a corrupt stored list rather than quietly shrinking it', async () => {
    // A dropped message degrades a conversation; a dropped TASK is work nobody
    // will ever do, with nothing saying so.
    const store = await make();
    const tasks = new StoreTaskSource(store, 'tasks');
    await tasks.addMany(['a', 'b']);

    const stored = (await store.get('tasks')) as unknown as { tasks: unknown[]; next_id: number };
    await store.put('tasks', {
      tasks: [stored.tasks[0], 'not an object'] as unknown as JsonObject[keyof JsonObject],
      next_id: stored.next_id,
    } as unknown as JsonObject);

    expect(await codeOf(() => tasks.pending())).toBe('unmappable_content');
    expect(await codeOf(() => tasks.claim('w-1'))).toBe('unmappable_content');

    // The positive control: the same list, with the entry put back, reads fine.
    await store.put('tasks', {
      tasks: [stored.tasks[0], stored.tasks[1]] as unknown as JsonObject[keyof JsonObject],
      next_id: stored.next_id,
    } as unknown as JsonObject);

    expect(await tasks.pending()).toBe(2);
  });

  it('refuses to release a task it does not hold', async () => {
    const tasks = await source();
    const other = asAgentTask({
      id: 'not-mine',
      instruction: 'x',
      state: 'claimed',
      claimedBy: 'w-1',
      claimedUntil: 1,
    });

    expect(await codeOf(() => tasks.release(other, 'w-1', 'done'))).toBe('task_not_found');
  });
});

// -- the non-atomic shape this design exists to prevent ----------------------

describe('claim() atomicity', () => {
  /**
   * Read the next task, then mark it taken. TWO calls, and the window between
   * them is the whole problem.
   *
   * This is the NEGATIVE CONTROL for the concurrency test above: without it,
   * a passing "no two workers got the same task" proves only that the test did
   * not manage to produce a race, not that the implementation prevents one.
   */
  async function racyClaim(store: SessionStore, worker: string): Promise<string | null> {
    const stored = (await store.get('tasks')) ?? {};
    const tasks = (Array.isArray(stored.tasks) ? stored.tasks : []) as Array<Record<string, unknown>>;
    const next = tasks.find((task) => task.state === 'todo');

    if (next === undefined) return null;

    next.state = 'claimed';
    next.claimed_by = worker;
    await store.put('tasks', { ...stored, tasks } as unknown as JsonObject);

    return next.id as string;
  }

  it('read-then-mark hands the SAME task to three workers', async () => {
    const store = new DurableMemoryStore();
    const tasks = new StoreTaskSource(store, 'tasks');
    await tasks.addMany(['a', 'b', 'c']);

    const claimed = await Promise.all([
      racyClaim(store, 'w-1'),
      racyClaim(store, 'w-2'),
      racyClaim(store, 'w-3'),
    ]);

    expect(claimed).toEqual(['t-1', 't-1', 't-1']);
    expect(new Set(claimed).size).toBe(1);
  });

  it('one atomic call does not, on the same store, with the same shape of test', async () => {
    const store = new DurableMemoryStore();
    const tasks = new StoreTaskSource(store, 'tasks');
    await tasks.addMany(['a', 'b', 'c']);

    const claimed = await Promise.all([tasks.claim('w-1'), tasks.claim('w-2'), tasks.claim('w-3')]);
    const ids = claimed.map((task) => task?.id());

    expect(new Set(ids).size).toBe(3);
    expect(ids).toEqual(['t-1', 't-2', 't-3']);
  });
});

// -- lease extension ---------------------------------------------------------

describe('lease self-extension', () => {
  async function held(lease: number): Promise<{ tasks: StoreTaskSource; id: string }> {
    const tasks = new StoreTaskSource(new DurableMemoryStore(), 'tasks');
    await tasks.add('the work');
    const claimed = await tasks.claim('w-1', lease);

    return { tasks, id: claimed!.id() };
  }

  function ledgerStartedSecondsAgo(seconds: number): RunLedger {
    return new RunLedger('run-1', Date.now() - seconds * 1000);
  }

  it('is REFUSED once the run has no wall-clock budget left', async () => {
    const { tasks, id } = await held(600);
    const budget = new RunBudget(8, null, 60);
    const spent = ledgerStartedSecondsAgo(61);

    expect(spent.remainingSeconds(budget)).toBe(0);
    expect(
      await codeOf(async () => tasks.extendLease((await tasks.find(id))!, 'w-1', 300, spent, budget)),
    ).toBe('run_not_permitted');
  });

  it('is REFUSED by a CANCELLED run, even with time on the clock', async () => {
    // Reading the spec's "remaining wall-clock budget" literally left exactly
    // this hole: a run someone had stopped could keep pushing its lease out and
    // go on holding a task it may no longer do anything with.
    const { tasks, id } = await held(600);
    const budget = new RunBudget(8, null, 600);
    const ledger = ledgerStartedSecondsAgo(1);

    // The positive control FIRST, on the same ledger: it is the cancel that
    // refuses this, not the clock.
    expect(
      await codeOf(async () => tasks.extendLease((await tasks.find(id))!, 'w-1', 30, ledger, budget)),
    ).toBe('no error');
    expect(ledger.remainingSeconds(budget)).toBeGreaterThan(0);

    ledger.cancel('operator stopped the run');

    expect(
      await codeOf(async () => tasks.extendLease((await tasks.find(id))!, 'w-1', 30, ledger, budget)),
    ).toBe('run_not_permitted');
  });

  it('is REFUSED once the run has spent its STEPS', async () => {
    const { tasks, id } = await held(600);
    const budget = new RunBudget(2, null, 600);
    const ledger = ledgerStartedSecondsAgo(1);

    expect(
      await codeOf(async () => tasks.extendLease((await tasks.find(id))!, 'w-1', 30, ledger, budget)),
    ).toBe('no error');

    ledger.recordSteps(2);

    expect(
      await codeOf(async () => tasks.extendLease((await tasks.find(id))!, 'w-1', 30, ledger, budget)),
    ).toBe('run_not_permitted');
  });

  it('is GRANTED while the run still has time — the positive control', async () => {
    const { tasks, id } = await held(600);
    const budget = new RunBudget(8, null, 60);
    const fresh = ledgerStartedSecondsAgo(1);

    expect(
      await codeOf(async () => tasks.extendLease((await tasks.find(id))!, 'w-1', 30, fresh, budget)),
    ).toBe('no error');
  });

  it('clamps the extension to what the run has left', async () => {
    // Claimed with a SHORT lease, so the clamp is visible: extending a long
    // lease by a smaller amount would correctly leave the long one in place and
    // prove nothing about the bound.
    const { tasks, id } = await held(5);
    const budget = new RunBudget(8, null, 60);
    const ledger = ledgerStartedSecondsAgo(40);
    const remaining = ledger.remainingSeconds(budget)!;
    const before = Math.floor(Date.now() / 1000);

    const until = await tasks.extendLease((await tasks.find(id))!, 'w-1', 3_000, ledger, budget);

    expect(remaining).toBeGreaterThan(0);
    expect(remaining).toBeLessThanOrEqual(20);
    // The bound, not the request. Asking for 3000 seconds got at most 20.
    expect(until - before).toBeLessThanOrEqual(remaining);
    expect(until - before).toBeGreaterThanOrEqual(remaining - 2);
    expect(until - before).toBeLessThan(3_000);
  });

  it('grants what was asked when the budget has no wall-clock cap', async () => {
    const { tasks, id } = await held(5);
    const budget = new RunBudget(8, null, null);
    const ledger = ledgerStartedSecondsAgo(0);
    const before = Math.floor(Date.now() / 1000);

    const until = await tasks.extendLease((await tasks.find(id))!, 'w-1', 120, ledger, budget);

    expect(budget.maxSeconds).toBeNull();
    expect(ledger.remainingSeconds(budget)).toBeNull();
    expect(until - before).toBeGreaterThanOrEqual(119);
    expect(until - before).toBeLessThanOrEqual(120);
  });

  it('sets now + granted EVEN WHEN that shortens the lease', async () => {
    // A first draft here took the later of the two, reasoning that a call named
    // "extend" should never take time away. Wrong: the granted figure is what
    // the RUN can still afford, so keeping the longer lease would hold a task
    // past the budget meant to bound it — and would make the answer depend on
    // the order calls happened to arrive in.
    const { tasks, id } = await held(600);
    const budget = new RunBudget(8, null, null);
    const claimedUntil = (await tasks.find(id))!.claimedUntil()!;
    const before = Math.floor(Date.now() / 1000);

    const until = await tasks.extendLease(
      (await tasks.find(id))!,
      'w-1',
      10,
      ledgerStartedSecondsAgo(0),
      budget,
    );

    expect(until).toBeLessThan(claimedUntil);
    expect(until - before).toBeLessThanOrEqual(10);
    expect(until - before).toBeGreaterThanOrEqual(9);
    // …and the store agrees, rather than only the return value.
    expect((await tasks.find(id))!.claimedUntil()).toBe(until);
  });

  it('is refused for a worker that is not the holder', async () => {
    const { tasks, id } = await held(600);

    expect(
      await codeOf(async () =>
        tasks.extendLease(
          (await tasks.find(id))!,
          'w-2',
          30,
          ledgerStartedSecondsAgo(0),
          new RunBudget(8, null, 600),
        ),
      ),
    ).toBe('task_lease_not_held');
  });

  it('is refused once the holder’s own lease has expired', async () => {
    // The SAME code as "someone else holds it", because the fix is the same:
    // claim it again. The sentence still says which case it was.
    const clock = ticking(1_000);
    const tasks = new StoreTaskSource(new DurableMemoryStore(), 'tasks', { now: clock.now });
    await tasks.add('the work');
    const claimed = await tasks.claim('w-1', 60);

    clock.set(1_060);

    expect(
      await codeOf(() =>
        tasks.extendLease(claimed!, 'w-1', 30, ledgerStartedSecondsAgo(0), new RunBudget(8, null, 600)),
      ),
    ).toBe('task_lease_not_held');
  });

  it('is refused for a task nobody holds, and for a terminal one', async () => {
    const tasks = new StoreTaskSource(new DurableMemoryStore(), 'tasks');
    await tasks.addMany(['a', 'b']);
    const untouched = await tasks.find('t-1');

    expect(
      await codeOf(() =>
        tasks.extendLease(untouched!, 'w-1', 30, ledgerStartedSecondsAgo(0), new RunBudget(8, null, 600)),
      ),
    ).toBe('task_lease_not_held');

    const claimed = await tasks.claim('w-1');
    await tasks.release(claimed!, 'w-1', 'done');

    expect(
      await codeOf(() =>
        tasks.extendLease(claimed!, 'w-1', 30, ledgerStartedSecondsAgo(0), new RunBudget(8, null, 600)),
      ),
    ).toBe('task_already_terminal');
  });

  it('is refused for a task the source does not hold', async () => {
    const tasks = new StoreTaskSource(new DurableMemoryStore(), 'tasks');
    const stranger = asAgentTask({
      id: 'elsewhere',
      instruction: 'x',
      state: 'claimed',
      claimedBy: 'w-1',
      claimedUntil: 9_999_999_999,
    });

    expect(
      await codeOf(() =>
        tasks.extendLease(stranger, 'w-1', 30, ledgerStartedSecondsAgo(0), new RunBudget(8, null, 600)),
      ),
    ).toBe('task_not_found');
  });
});

// -- completion authority ----------------------------------------------------

describe('an agent cannot complete its own task by default', () => {
  async function heldTask(): Promise<{ tasks: StoreTaskSource; session: Session; id: string }> {
    const durable = new DurableMemoryStore();
    const tasks = new StoreTaskSource(durable, 'tasks');
    await tasks.add('the work');
    const claimed = await tasks.claim('w-1');
    const session = new Session({
      participant: { type: 'User', id: 1 },
      scope: 'tasks',
      ephemeral: new MemorySessionStore(),
      durable,
    });

    return { tasks, session, id: claimed!.id() };
  }

  it('refuses to build the tool against a disabled authorizer', async () => {
    const { tasks, id } = await heldTask();
    const task = (await tasks.find(id))!;

    // `ToolAuthorizer` is off by default, and off means every call is allowed.
    // Wiring completion authority to it would grant the agent the authority
    // while a reader sees a control.
    expect(
      codeOfSync(() =>
        agentCompletionTool({ source: tasks, task, worker: 'w-1', authorizer: new ToolAuthorizer() }),
      ),
    ).toBe('unsafe_authorization_configuration');
  });

  it('leaves the task CLAIMED when the call policy refuses', async () => {
    const { tasks, session, id } = await heldTask();
    const task = (await tasks.find(id))!;
    const factory = agentCompletionTool({
      source: tasks,
      task,
      worker: 'w-1',
      authorizer: new ToolAuthorizer({ enabled: true, call: () => false }),
    });

    // A WELL-FORMED call, deliberately. Passing junk arguments would let this
    // pass for the wrong reason the day argument validation ran first.
    expect(await codeOf(() => Promise.resolve(factory(session).handle({ outcome: 'done' })))).toBe(
      'call_not_authorized',
    );

    // The assertion that matters is not which error came back — it is that the
    // capability is still unreachable. What can the agent still invoke?
    expect((await tasks.find(id))?.state()).toBe('claimed');
  });

  it('completes the task when the policy allows it — the positive control', async () => {
    const { tasks, session, id } = await heldTask();
    const task = (await tasks.find(id))!;
    const factory = agentCompletionTool({
      source: tasks,
      task,
      worker: 'w-1',
      authorizer: new ToolAuthorizer({ enabled: true, call: () => true }),
    });

    await factory(session).handle({ outcome: 'done' });

    expect((await tasks.find(id))?.state()).toBe('done');
  });

  it('refuses an outcome it does not recognise, rather than guessing at done', async () => {
    // The one value the MODEL supplies on this tool, deciding a terminal state.
    // Coercing anything unrecognised to `done` would mean a malformed argument
    // produces the more privileged result — an agent declaring victory by typo.
    const { tasks, session, id } = await heldTask();
    const tool = agentCompletionTool({
      source: tasks,
      task: (await tasks.find(id))!,
      worker: 'w-1',
      authorizer: new ToolAuthorizer({ enabled: true, call: () => true }),
    })(session);

    // PADDED values are the adversarial half, and they are BUILT from the
    // fixture list rather than typed out, so the meta-test guarantees every one
    // of them is a codepoint JavaScript's own `trim()` would remove. An
    // implementation that trimmed the outcome before comparing would turn every
    // one of these into a valid `done` and close the task; nothing here trims,
    // so all of them are refused. Same rule as the identifiers, same reason.
    const padded = PADDING.flatMap((pad) => [`${pad}done`, `done${pad}`, `${pad}done${pad}`]);
    const rejected = [
      'complete',
      'DONE',
      true,
      // PRESENT-and-null is present. The model said something; it is just not
      // something this tool acts on.
      null,
      // ABSENT, which the reference settled as a refusal too.
      undefined,
      ...padded,
    ];

    expect(padded).toHaveLength(PADDING.length * 3);

    // COUNTED, because `vitest list` shows the `it` and not the iterations
    // inside it. Python had a padded-outcome fixture that was defined and never
    // wired into the loop that was supposed to use it — the NBSP and U+3000
    // cases never executed, and a green run looked exactly the same. A count
    // asserted against the fixture length is what makes that visible here.
    let refused = 0;

    for (const outcome of rejected) {
      expect(
        await codeOf(() => Promise.resolve(tool.handle({ outcome } as unknown as JsonObject))),
      ).toBe('task_outcome_invalid');
      refused += 1;
    }

    expect(refused).toBe(rejected.length);
    expect(refused).toBe(5 + PADDING.length * 3);

    // The capability question, not the message: after every malformed call the
    // task is still open.
    expect((await tasks.find(id))?.state()).toBe('claimed');

    // The positive control: the exact value works.
    expect(await codeOf(() => Promise.resolve(tool.handle({ outcome: 'done' })))).toBe('no error');
    expect((await tasks.find(id))?.state()).toBe('done');
  });

  it('REFUSES an absent outcome rather than assuming done', async () => {
    // The argument for the other answer is real — calling a tool named
    // `complete_task` looks like the declaration by itself — and the reference
    // settled against it. Same code as a malformed value, so a model that
    // omitted the argument is told to say which outcome it means.
    const { tasks, session, id } = await heldTask();
    const tool = agentCompletionTool({
      source: tasks,
      task: (await tasks.find(id))!,
      worker: 'w-1',
      authorizer: new ToolAuthorizer({ enabled: true, call: () => true }),
    })(session);

    expect(await codeOf(() => Promise.resolve(tool.handle({})))).toBe('task_outcome_invalid');
    expect((await tasks.find(id))?.state()).toBe('claimed');

    // The positive control: naming it works.
    expect(await tool.handle({ outcome: 'done' })).toEqual({ id, state: 'done' });
  });

  it('lets the agent report a failure as well as a success', async () => {
    const { tasks, session, id } = await heldTask();
    const factory = agentCompletionTool({
      source: tasks,
      task: (await tasks.find(id))!,
      worker: 'w-1',
      authorizer: new ToolAuthorizer({ enabled: true, call: () => true }),
    });

    await factory(session).handle({ outcome: 'failed' });

    expect((await tasks.find(id))?.state()).toBe('failed');
  });

  it('cannot close a task its worker is not holding', async () => {
    // The tool is bound to one task AND one worker. Bound to the source alone
    // it could close anything in the list, including work another agent is
    // midway through, which is the failure completion authority is gated for.
    const { tasks, session, id } = await heldTask();
    const factory = agentCompletionTool({
      source: tasks,
      task: (await tasks.find(id))!,
      worker: 'someone-else',
      authorizer: new ToolAuthorizer({ enabled: true, call: () => true }),
    });

    expect(
      await codeOf(() => Promise.resolve(factory(session).handle({ outcome: 'done' }))),
    ).toBe('task_lease_not_held');
    expect((await tasks.find(id))?.state()).toBe('claimed');
  });

  it('CONTROL: the unguarded fixture really is unguarded', async () => {
    // A control needs its own control. Every test below measures the tool
    // against `UnguardedTaskSource`; if that fixture ever quietly grew a guard,
    // all of them would still pass and every one of them would be measuring the
    // fixture instead of the tool. Run first, and deliberately separate.
    const unguarded = new UnguardedTaskSource([
      { claimed_by: 'someone-else', claimed_until: 9_999_999_999, id: 'c-1', instruction: 'theirs', state: 'claimed' },
      { claimed_by: null, claimed_until: null, id: 'c-2', instruction: 'nobody has it', state: 'todo' },
      { claimed_by: null, claimed_until: null, id: 'c-3', instruction: 'over', state: 'done' },
    ]);
    const taskFor = (id: string) => asAgentTask(recordToAttributes(unguarded.record(id)!));

    // It closes a task held by someone else, one nobody holds, and one that is
    // already terminal — without a murmur. That is what "unguarded" has to mean
    // for the tool tests below to prove anything.
    await unguarded.release(taskFor('c-1'), 'w-1', 'done');
    await unguarded.release(taskFor('c-2'), 'w-1', 'done');
    await unguarded.release(taskFor('c-3'), 'w-1', 'failed');

    expect(unguarded.record('c-1')!.state).toBe('done');
    expect(unguarded.record('c-2')!.state).toBe('done');
    expect(unguarded.record('c-3')!.state).toBe('failed');
  });

  it('refuses a task whose holder CANNOT BE ESTABLISHED', async () => {
    // `AgentTask` is three methods, so this source is conforming rather than
    // broken. Reading its silence as permission is the same mistake as
    // inferring `done` from an absent outcome, one level down.
    const { session } = await heldTask();
    const holderless = new HolderlessTaskSource('claimed');
    const tool = agentCompletionTool({
      source: holderless,
      task: (await holderless.find('h-1'))!,
      worker: 'w-1',
      authorizer: new ToolAuthorizer({ enabled: true, call: () => true }),
    })(session);

    // The task says `claimed`, so every check except the holder one passes.
    expect((await holderless.find('h-1'))?.state()).toBe('claimed');
    expect(await codeOf(() => Promise.resolve(tool.handle({ outcome: 'done' })))).toBe(
      'task_lease_not_held',
    );
  });

  it('refuses on its OWN, against a source that guards nothing', async () => {
    // The tool's pre-check is invisible against `StoreTaskSource`, which
    // already refuses a non-holder — delete it and every other test here still
    // passes, so the mutation SURVIVES. That is the argument for it, not
    // against it: an interface cannot make an implementation check anything,
    // and `release(task, worker, outcome)` reads to a third party as "find it
    // and set the state", which is what they will write. The tool is the thing
    // a MODEL can call, so it does not get to assume the source behind it is
    // the careful one.
    //
    // Hence this fixture, which guards NOTHING. It is the only thing in the
    // suite that can tell whether the tool checks or merely delegates.
    const { session } = await heldTask();
    const unguarded = new UnguardedTaskSource([
      { claimed_by: 'someone-else', claimed_until: 9_999_999_999, id: 'u-1', instruction: 'theirs', state: 'claimed' },
      { claimed_by: null, claimed_until: null, id: 'u-2', instruction: 'nobody has it', state: 'todo' },
      { claimed_by: null, claimed_until: null, id: 'u-3', instruction: 'over', state: 'done' },
      { claimed_by: 'w-1', claimed_until: 9_999_999_999, id: 'u-4', instruction: 'mine', state: 'claimed' },
      // INCONSISTENT: the holder matches, the state does not. Only a source
      // with no guards can produce this, which is why it is here — it is the
      // one row that makes the tool's `state === 'claimed'` check observable
      // rather than shadowed by the holder check.
      { claimed_by: 'w-1', claimed_until: null, id: 'u-5', instruction: 'never started', state: 'todo' },
    ]);

    const toolFor = (id: string) =>
      agentCompletionTool({
        source: unguarded,
        task: asAgentTask(recordToAttributes(unguarded.record(id)!)),
        worker: 'w-1',
        authorizer: new ToolAuthorizer({ enabled: true, call: () => true }),
      })(session);

    expect(await codeOf(() => Promise.resolve(toolFor('u-1').handle({ outcome: 'done' })))).toBe(
      'task_lease_not_held',
    );
    expect(await codeOf(() => Promise.resolve(toolFor('u-2').handle({ outcome: 'done' })))).toBe(
      'task_lease_not_held',
    );
    expect(await codeOf(() => Promise.resolve(toolFor('u-3').handle({ outcome: 'done' })))).toBe(
      'task_already_terminal',
    );
    // Holder matches, state does not. The `claimed` check is the only thing
    // standing here, so this row is what makes it a check rather than a claim.
    expect(await codeOf(() => Promise.resolve(toolFor('u-5').handle({ outcome: 'done' })))).toBe(
      'task_lease_not_held',
    );

    // The refusal must not name the holder either. This one reaches a MODEL as
    // readable text, so it is the last place to leak a peer worker's identity.
    let message = '';
    try {
      await toolFor('u-1').handle({ outcome: 'done' });
    } catch (error) {
      message = error instanceof Error ? error.message : String(error);
    }
    expect(message).not.toContain('someone-else');
    expect(message).toContain('w-1');

    // None of the refusals touched anything.
    expect(unguarded.record('u-1')!.state).toBe('claimed');
    expect(unguarded.record('u-2')!.state).toBe('todo');
    expect(unguarded.record('u-3')!.state).toBe('done');
    expect(unguarded.record('u-5')!.state).toBe('todo');

    // The positive control: the task this worker really holds still closes.
    expect(await codeOf(() => Promise.resolve(toolFor('u-4').handle({ outcome: 'done' })))).toBe(
      'no error',
    );
    expect(unguarded.record('u-4')!.state).toBe('done');
  });

  it('refuses a task the source cannot find', async () => {
    const { session } = await heldTask();
    const unguarded = new UnguardedTaskSource([]);
    const tool = agentCompletionTool({
      source: unguarded,
      task: asAgentTask({
        id: 'gone',
        instruction: 'x',
        state: 'claimed',
        claimedBy: 'w-1',
        claimedUntil: 9_999_999_999,
      }),
      worker: 'w-1',
      authorizer: new ToolAuthorizer({ enabled: true, call: () => true }),
    })(session);

    expect(await codeOf(() => Promise.resolve(tool.handle({ outcome: 'done' })))).toBe(
      'task_not_found',
    );
  });

  it('offers no completion route at all until a consumer builds one', async () => {
    // Nothing in the package registers this tool, and a source on its own has
    // no method an agent could reach. `release()` is the application's call.
    const { tasks } = await heldTask();

    expect(Object.keys(tasks)).not.toContain('complete');
    expect(typeof (tasks as unknown as Record<string, unknown>).completeSelf).toBe('undefined');
  });
});

// -- the consumer's own model ------------------------------------------------

describe('a consumer’s own record as an AgentTask', () => {
  it('conforms structurally, and reads through to the live record', () => {
    const row: AgentTaskAttributes = {
      id: 'row-7',
      instruction: 'migrate the thing',
      state: 'todo',
      claimedBy: null,
      claimedUntil: null,
    };

    const task = asAgentTask(row);

    expect(task.id()).toBe('row-7');
    expect(task.instruction()).toBe('migrate the thing');
    expect(task.state()).toBe('todo');

    // Another worker claims the row underneath. A snapshot taken at
    // construction would still say `todo`.
    row.state = 'claimed';
    row.claimedBy = 'w-1';
    row.claimedUntil = 1_700_000_300;

    expect(task.state()).toBe('claimed');
    expect(task.claimedBy()).toBe('w-1');
    expect(canonicalTaskJson(task.toRecord())).toBe(
      '{"claimed_by":"w-1","claimed_until":1700000300,"id":"row-7","instruction":"migrate the thing","state":"claimed"}',
    );
  });

  it('works as a mixin on the consumer’s class', () => {
    // The equivalent of the reference's trait on an Eloquent model. The base
    // maps its own column names; nothing here assumes them.
    class Row implements AgentTaskAttributeSource {
      constructor(
        readonly key: string,
        readonly body: string,
        public status: TaskState,
      ) {}

      taskAttributes(): AgentTaskAttributes {
        return {
          id: this.key,
          instruction: this.body,
          state: this.status,
          claimedBy: null,
          claimedUntil: null,
        };
      }
    }

    const Base = withAgentTask(Row);

    class Todo extends Base {}

    const todo = new Todo('row-1', 'do it', 'todo');

    expect(todo.id()).toBe('row-1');
    expect(todo.instruction()).toBe('do it');
    expect(todo.state()).toBe('todo');
    expect(todo.claimedBy()).toBeNull();
    expect(canonicalTaskJson(todo.toRecord())).toBe(
      '{"claimed_by":null,"claimed_until":null,"id":"row-1","instruction":"do it","state":"todo"}',
    );

    todo.status = 'done';

    expect(todo.state()).toBe('done');
  });

  it('is accepted by a store-backed source wherever an id matches', async () => {
    // The point of one contract: the source takes any `AgentTask`, not only the
    // one it made.
    const tasks = new StoreTaskSource(new DurableMemoryStore(), 'tasks');
    await tasks.add('the work');
    await tasks.claim('w-1');

    const foreign = asAgentTask({
      id: 't-1',
      instruction: 'the work',
      state: 'claimed',
      claimedBy: 'w-1',
      claimedUntil: 1,
    });

    await tasks.release(foreign, 'w-1', 'done');

    expect((await tasks.find('t-1'))?.state()).toBe('done');
  });
});
