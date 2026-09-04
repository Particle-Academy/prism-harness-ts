import type { JsonObject, JsonValue } from './json.js';
import { isJsonObject } from './json.js';
import { HarnessError } from './errors.js';
import type { Durability, SessionStore } from './stores/session-store.js';
import { isDurable } from './stores/session-store.js';
import type { RunBudget, RunLedger } from './subagents.js';
import {
  DEFAULT_LEASE_SECONDS,
  isTerminalState,
  taskRecordFrom,
} from './tasks.js';
import type {
  AgentTask,
  AgentTaskRecord,
  AgentTaskSource,
  LeasedAgentTask,
  TaskOutcome,
  TaskState,
} from './tasks.js';

/** Integer Unix seconds. Injectable so a lease can be expired in a test without waiting for one. */
export type Clock = () => number;

export interface StoreTaskSourceOptions {
  /** Defaults to the system clock, truncated to whole seconds. */
  now?: Clock;
}

/**
 * A task whose state was read at a point in time.
 *
 * A SNAPSHOT, and it has to be: `AgentTask.state()` is synchronous, and the
 * only honest way to make it live would be to hit the store, which cannot be
 * done without returning a promise. Ask `StoreTaskSource.find()` for a fresh
 * read; do not treat a task object held across an await as current.
 */
export class StoredAgentTask implements LeasedAgentTask {
  readonly #record: AgentTaskRecord;

  constructor(record: AgentTaskRecord) {
    this.#record = { ...record };
  }

  id(): string {
    return this.#record.id;
  }

  instruction(): string {
    return this.#record.instruction;
  }

  state(): TaskState {
    return this.#record.state;
  }

  claimedBy(): string | null {
    return this.#record.claimed_by;
  }

  claimedUntil(): number | null {
    return this.#record.claimed_until;
  }

  toRecord(): AgentTaskRecord {
    return { ...this.#record };
  }
}

/**
 * The default task source: the list lives in the harness's own durable store.
 *
 * No schema, no migration, works on install. A consumer with an existing table
 * adapts it instead (`asAgentTask` / `withAgentTask`); both satisfy one
 * contract, and neither is the afterthought.
 *
 * ## Durable or nothing
 *
 * REFUSED AT CONSTRUCTION against a store that reports itself volatile. The
 * package already draws this line for the durable slot and this inherits the
 * rule rather than restating it: a half-finished task list that vanishes on a
 * deploy is indistinguishable from a finished one, so losing it is a
 * correctness failure and not a degradation to a default.
 *
 * ## The claim is one operation
 *
 * `claim()` does the whole read-expire-select-write inside ONE store lock.
 * Reading the next task and then marking it taken is two operations with a
 * window between them, and two workers arriving in that window both get the
 * same task. That race is the reason this class exists, so the lock wraps
 * everything rather than just the write.
 *
 * **The guarantee is the STORE'S, not this class's.** "Never the same task
 * twice" holds exactly as far as `SessionStore.withLock` does:
 * `MemorySessionStore`'s lock is process-local and `FileSessionStore`'s is one
 * machine. Two workers on different machines over a network filesystem are not
 * excluded by either, and no file lock can promise that -- point the durable
 * slot at a database or Redis there. Saying so is the point: a claim that looks
 * atomic and is not would hand the same expensive task to two agents.
 *
 * ## Size
 *
 * The whole list is one stored value, read and rewritten on every claim. That
 * is right for the tens-of-tasks a bounded agent run produces and wrong for
 * thousands: a consumer with a real backlog wants their own table and the
 * application-model adapter, which is why that adapter is a first-class shape
 * rather than an afterthought.
 *
 * ## Ordering
 *
 * Tasks are held in an array and `claim()` takes the first claimable one, so
 * the order out is the order in. Nothing here sorts, prioritises or shuffles.
 * Ordering is the divergence nothing reports: the agent just does the work in a
 * different sequence and produces a different result.
 */
export class StoreTaskSource implements AgentTaskSource {
  readonly #store: SessionStore;

  readonly #key: string;

  readonly #now: Clock;

  constructor(store: SessionStore, key: string, options: StoreTaskSourceOptions = {}) {
    // Checked here, before anything can be written, because "refuse to start"
    // is the requirement -- not "refuse once the first task is lost".
    if (!isDurable(store.durability())) {
      throw HarnessError.volatileTaskSource();
    }

    this.#store = store;
    this.#key = key;
    this.#now = options.now ?? (() => Math.floor(Date.now() / 1000));
  }

  /** What the underlying store says about itself. Always `durable` -- see the constructor. */
  durability(): Durability {
    return this.#store.durability();
  }

  /**
   * The current time, FLOORED TO A WHOLE SECOND.
   *
   * Every timestamp this class writes is derived from here, so flooring once is
   * what keeps `claimed_until` an integer. JavaScript has no int/float
   * distinction, so a fractional clock produced `1090.75` and passed every
   * equality assertion that had been written about it -- the bytes were
   * `"claimed_until":1090.75` where the other two ports write `1090`, and only
   * a `Number.isInteger` check on the stored value could see it. That test
   * exists, and it found this.
   */
  #seconds(): number {
    return Math.floor(this.#now());
  }

  /**
   * Append a task, in `todo`.
   *
   * The id may be supplied -- a consumer seeding a list from their own domain
   * keys wants their keys -- and is generated as `t-<n>` when it is not. A
   * supplied id that is already in the list is REFUSED rather than merged: two
   * units of work sharing an id would share a claim.
   *
   * An id of `""` is refused too, compared against the empty string EXACTLY.
   * Nothing is trimmed first, in any of the three languages: `trim`, `strip`
   * and `String.prototype.trim` each strip a different set of codepoints, so a
   * trim here would give three different answers to the same input. A single
   * space is a legal id, and that is the correct behaviour.
   */
  async add(instruction: string, id?: string): Promise<StoredAgentTask> {
    if (id !== undefined && id === '') {
      throw HarnessError.taskIdentifierBlank('task id');
    }

    return this.#store.withLock(this.#key, async () => {
      const stored = await this.#read();

      if (id !== undefined && stored.records.some((record) => record.id === id)) {
        throw HarnessError.duplicateTaskId(id);
      }

      const taken = new Set(stored.records.map((record) => record.id));
      let nextId = stored.nextId;

      // A generated id must not land on one a caller supplied earlier.
      while (id === undefined && taken.has(`t-${nextId}`)) nextId += 1;

      const record = makeRecord(id ?? `t-${nextId}`, instruction, 'todo', null, null);

      await this.#write([...stored.records, record], id === undefined ? nextId + 1 : stored.nextId);

      return new StoredAgentTask(record);
    });
  }

  /** Append several tasks at once, in the order given, with generated ids. */
  async addMany(instructions: readonly string[]): Promise<StoredAgentTask[]> {
    const added: StoredAgentTask[] = [];

    for (const instruction of instructions) added.push(await this.add(instruction));

    return added;
  }

  /**
   * Take the next claimable task, atomically, or null when there is none.
   *
   * The claim is WRITTEN BEFORE this returns, which is the point: the caller
   * cannot begin work until the store says the task is theirs, so a worker that
   * dies mid-task leaves a `claimed` row with an expiry rather than a `todo`
   * row that looks like it was never attempted.
   */
  async claim(worker: string, leaseSeconds: number = DEFAULT_LEASE_SECONDS): Promise<StoredAgentTask | null> {
    assertIdentifier(worker, 'worker id');

    // A lease of zero is a claim that has already expired, which would hand the
    // same task to a second worker in the same second -- the exact race this
    // class prevents. Clamped rather than accepted, which fails closed.
    const lease = Math.max(1, Math.trunc(leaseSeconds));

    return this.#store.withLock(this.#key, async () => {
      const now = this.#seconds();
      const stored = await this.#read();
      const { records, changed } = expireLeases(stored.records, now);
      const index = records.findIndex((record) => record.state === 'todo');

      if (index === -1) {
        // Nothing to claim, but an expiry we noticed is a fact about the list
        // and is persisted anyway. A reader must not see `claimed` for a lease
        // that ran out just because the claim that noticed came up empty.
        if (changed) await this.#write(records, stored.nextId);

        return null;
      }

      const previous = records[index] as AgentTaskRecord;
      const claimed = makeRecord(previous.id, previous.instruction, 'claimed', worker, now + lease);

      records[index] = claimed;
      await this.#write(records, stored.nextId);

      return new StoredAgentTask(claimed);
    });
  }

  /**
   * Record what happened, as the worker HOLDING the task. Both outcomes are TERMINAL.
   *
   * Refuses a task that is already terminal rather than ignoring it: two
   * callers believing they own one outcome is worth reporting, and the second
   * write silently winning would leave nothing recording the disagreement.
   *
   * Refuses a task this worker is not holding -- nobody holding it, someone
   * else holding it, or a lease that has since expired all reach the same
   * refusal, because they have the same answer. The state machine has no edge
   * from `todo` to `done`, and a lease that expired mid-flight has already put
   * the task back where anyone can take it, so the worker that finally finished
   * may no longer be the one whose result counts.
   */
  async release(task: AgentTask, worker: string, outcome: TaskOutcome): Promise<void> {
    assertIdentifier(worker, 'worker id');

    await this.#store.withLock(this.#key, async () => {
      const now = this.#seconds();
      const stored = await this.#read();
      const index = stored.records.findIndex((record) => record.id === task.id());

      if (index === -1) {
        throw HarnessError.taskNotFound(task.id());
      }

      const raw = stored.records[index] as AgentTaskRecord;
      const { records, changed } = expireLeases(stored.records, now);
      const refusal = refuseUnheld(raw, worker, now);

      if (refusal !== null) {
        if (changed) await this.#write(records, stored.nextId);

        throw refusal;
      }

      // The owner and the expiry go to null. A terminal task is held by nobody,
      // and leaving a stale holder on it would make `claimed_by` mean two
      // different things depending on the state it sits beside.
      records[index] = makeRecord(raw.id, raw.instruction, outcome, null, null);

      await this.#write(records, stored.nextId);
    });
  }

  /**
   * How many tasks a `claim()` could still hand out.
   *
   * Counts `todo` AND any `claimed` task whose lease has run out, because an
   * expired lease is claimable by anyone -- a loop that stopped while such a
   * task sat there would stop with work left.
   *
   * Reads without taking the lock. A store `get` is a single atomic read of one
   * key, so the count is consistent with some moment; taking the lock would
   * make it consistent with a moment that has already passed by the time the
   * caller reads it.
   */
  async pending(): Promise<number> {
    const now = this.#seconds();
    const { records } = expireLeases((await this.#read()).records, now);

    return records.filter((record) => record.state === 'todo').length;
  }

  /** A fresh read of one task, with any expired lease already applied, or null. */
  async find(id: string): Promise<StoredAgentTask | null> {
    const now = this.#seconds();
    const { records } = expireLeases((await this.#read()).records, now);
    const record = records.find((candidate) => candidate.id === id);

    return record === undefined ? null : new StoredAgentTask(record);
  }

  /**
   * Push this worker's lease out, and ONLY while it still holds it.
   *
   * Bounded by the run's REMAINING WALL-CLOCK BUDGET, which is the whole
   * argument. Unbounded self-extension is how a wedged worker holds a task
   * forever, and a second timeout invented here would be one limit spelled two
   * ways across an ecosystem -- which is how a limit ends up set in the place
   * that is not enforced. `RunLedger.remainingSeconds()` against the
   * `RunBudget` is already the stop condition, so extension stops when the
   * run's own allowance does. Nothing new to enforce and nothing new to forget.
   *
   * Granted seconds are `min(requested, remaining)`. A budget with no
   * wall-clock cap (`maxSeconds` null) grants what was asked, matching
   * `RunLedger.exhaustion()`, which also has nothing to enforce in that case.
   *
   * The new expiry is `now + granted` EVEN WHEN THAT SHORTENS the lease. A
   * first draft here took the later of the two, on the reasoning that a call
   * named "extend" should never take time away. That is the wrong reasoning:
   * the granted figure is what the RUN can still afford, so keeping a longer
   * lease would hold a task past the budget that is supposed to bound it, and
   * it would also make the result depend on the order calls happened to arrive
   * in. One rule, one answer.
   *
   * @returns the new expiry, an integer Unix timestamp.
   */
  async extendLease(
    task: AgentTask,
    worker: string,
    seconds: number,
    ledger: RunLedger,
    budget: RunBudget,
  ): Promise<number> {
    assertIdentifier(worker, 'worker id');

    const requested = Math.max(1, Math.trunc(seconds));

    return this.#store.withLock(this.#key, async () => {
      const now = this.#seconds();
      const stored = await this.#read();
      const index = stored.records.findIndex((record) => record.id === task.id());

      if (index === -1) {
        throw HarnessError.taskNotFound(task.id());
      }

      // Judged against the row AS STORED, not the expired-normalised copy, so
      // the message can still say which of the three cases it was even though
      // all three share one code.
      const raw = stored.records[index] as AgentTaskRecord;
      const { records, changed } = expireLeases(stored.records, now);
      const refusal = refuseUnheld(raw, worker, now);

      if (refusal !== null) {
        if (changed) await this.#write(records, stored.nextId);

        throw refusal;
      }

      const remaining = ledger.remainingSeconds(budget);

      if (remaining !== null && remaining <= 0) {
        if (changed) await this.#write(records, stored.nextId);

        throw HarnessError.runNotPermitted(
          `The lease held by [${worker}] on the task [${raw.id}] cannot be extended: the run has no ` +
            'wall-clock budget left. The run is the bound on how long a worker may hold a task, and ' +
            'extending past it would be a second timeout wearing the first one\'s name.',
        );
      }

      const granted = remaining === null ? requested : Math.min(requested, remaining);
      const until = now + granted;

      records[index] = makeRecord(raw.id, raw.instruction, 'claimed', worker, until);
      await this.#write(records, stored.nextId);

      return until;
    });
  }

  /**
   * Read the list back, refusing anything it cannot map.
   *
   * A junk entry is an ERROR, not something to filter out. The thread does
   * filter, and can afford to: a dropped message degrades a conversation. A
   * dropped task is work that will never be done and nothing anywhere says so
   * -- and a list that quietly shrinks is the failure mode this whole
   * capability exists to prevent, arriving through the back door.
   */
  async #read(): Promise<{ records: AgentTaskRecord[]; nextId: number }> {
    const stored = (await this.#store.get(this.#key)) ?? {};
    const tasks = stored.tasks;
    const records = Array.isArray(tasks)
      ? tasks.map((entry) => {
          if (!isJsonObject(entry)) {
            throw HarnessError.unmappableContent(
              `a stored task list entry is [${typeof entry}] rather than an object`,
            );
          }

          return taskRecordFrom(entry);
        })
      : [];
    const nextId = typeof stored.next_id === 'number' ? Math.trunc(stored.next_id) : records.length + 1;

    return { records, nextId };
  }

  async #write(records: readonly AgentTaskRecord[], nextId: number): Promise<void> {
    const payload: JsonObject = {
      tasks: records.map((record) => ({ ...record })) as unknown as JsonValue[],
      next_id: nextId,
    };

    await this.#store.put(this.#key, payload);
  }
}

/**
 * The five canonical keys, always all of them, always in sorted order.
 *
 * Built through one function so no call site can forget a claim field and leave
 * an `undefined` behind. `undefined` would survive every type check that only
 * looks at the values it was given and then vanish at serialisation, taking the
 * key with it.
 */
function makeRecord(
  id: string,
  instruction: string,
  state: TaskState,
  claimedBy: string | null,
  claimedUntil: number | null,
): AgentTaskRecord {
  return {
    claimed_by: claimedBy,
    claimed_until: claimedUntil,
    id,
    instruction,
    state,
  };
}

/**
 * Return every task whose lease has run out to `todo`.
 *
 * TO `todo`, NEVER TO `failed`. A worker dying is not the task failing, and
 * marking it failed burns a retry that never ran -- the application would then
 * have to re-queue work that was never attempted, if it noticed at all.
 *
 * The expiry test is `claimed_until <= now`: the timestamp is the instant the
 * lease ENDS, so at that instant it is over.
 */
function expireLeases(
  records: readonly AgentTaskRecord[],
  now: number,
): { records: AgentTaskRecord[]; changed: boolean } {
  let changed = false;

  const next = records.map((record) => {
    if (!leaseExpired(record, now)) return { ...record };

    changed = true;

    return makeRecord(record.id, record.instruction, 'todo', null, null);
  });

  return { records: next, changed };
}

function leaseExpired(record: AgentTaskRecord, now: number): boolean {
  return record.state === 'claimed' && record.claimed_until !== null && record.claimed_until <= now;
}

/**
 * Why this worker may not act on this task -- or null when it may.
 *
 * TWO codes, not four. "Already finished" is its own, because the fix is to
 * stop; the other three -- nobody holds it, someone else does, your lease ran
 * out -- share `task_lease_not_held`, because the fix for all three is to claim
 * it again. The sentence still distinguishes them for a human reading a log.
 *
 * The detail NEVER NAMES THE HOLDER. This refusal can surface inside a tool
 * call, where a failure reaches the model as readable text, and an agent has no
 * business learning the identity of a peer worker from an error message.
 */
function refuseUnheld(record: AgentTaskRecord, worker: string, now: number): HarnessError | null {
  if (isTerminalState(record.state)) {
    return HarnessError.taskAlreadyTerminal(record.id, record.state);
  }

  if (record.state !== 'claimed') {
    return HarnessError.taskLeaseNotHeld(record.id, worker, 'nothing is holding it');
  }

  // Checked before expiry, deliberately. A lease that was never this worker's
  // is not this worker's whether or not it has run out.
  if (record.claimed_by !== worker) {
    return HarnessError.taskLeaseNotHeld(record.id, worker, 'another worker is holding it');
  }

  if (leaseExpired(record, now)) {
    return HarnessError.taskLeaseNotHeld(record.id, worker, 'its lease has expired');
  }

  return null;
}

/**
 * Refuse the empty string, and NOTHING ELSE.
 *
 * Compared against `""` exactly, with no trimming. PHP's `trim`, JavaScript's
 * `String.prototype.trim` and Python's `str.strip` each strip a different set
 * of codepoints, so trimming here would mean three ports disagreeing about
 * whether a given id is blank -- the shape of `prism-human-plus` G-36, where a
 * single trailing space defeated a guard in all three languages at once. A
 * worker id of one space is accepted on purpose.
 */
function assertIdentifier(value: string, what: string): void {
  if (value === '') {
    throw HarnessError.taskIdentifierBlank(what);
  }
}
