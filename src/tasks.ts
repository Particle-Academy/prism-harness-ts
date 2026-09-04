import type { JsonObject } from './json.js';
import { HarnessError } from './errors.js';

/**
 * What remains to be done, and how far along it is.
 *
 * FOUR STATES, and no others. This is identity, not spelling: a port that adds
 * a fifth (`cancelled`, `retrying`, `skipped`) has changed what a stored record
 * can say, and a consumer reading a list written by another language would meet
 * a value it cannot map.
 *
 * The transitions are equally pinned:
 *
 * ```text
 *         claim()                    release(done)
 *   todo ---------> claimed -------------------------> done
 *    ^                 |            release(failed)
 *    |                 +-------------------------------> failed
 *    |                 |
 *    +-----------------+
 *       lease expires
 * ```
 *
 * Note what is NOT there. An expired lease goes back to `todo`, never to
 * `failed` -- a worker dying is not the task failing, and conflating the two
 * burns a retry that never ran. And nothing leaves `done` or `failed`: a
 * `failed` task is re-queued by the APPLICATION, because automatic retry needs
 * backoff and attempt counts, and that is the scheduler this must not become.
 */
export type TaskState = 'todo' | 'claimed' | 'done' | 'failed';

/** Every state, in the order the machine visits them. */
export const TASK_STATES: readonly TaskState[] = ['todo', 'claimed', 'done', 'failed'];

/**
 * What `release()` records. Both are TERMINAL.
 *
 * Deliberately not a boolean. "Did it work" reads the same for a task that
 * failed and a task that was never attempted, and those two are what this
 * design exists to keep apart.
 */
export type TaskOutcome = Extract<TaskState, 'done' | 'failed'>;

/**
 * How long a claim holds by default: FIVE MINUTES.
 *
 * Long enough for a model call plus tool work, short enough that a crashed
 * worker does not wedge the list for an hour. The number matters less than it
 * being the same number in all three languages, which is why it is a named
 * constant rather than a literal at the call site.
 */
export const DEFAULT_LEASE_SECONDS = 300;

/**
 * One unit of work.
 *
 * METHODS, not properties, and that is not decoration. An adapter over a
 * consumer's own record has to read through to the live row -- a value
 * snapshotted at construction would still report `claimed` after another
 * worker's lease expired underneath it.
 */
export interface AgentTask {
  /** Stable and unique WITHIN ITS SOURCE. Not globally unique. */
  id(): string;

  /** What the model is asked to do. */
  instruction(): string;

  state(): TaskState;
}

/**
 * Where tasks come from.
 *
 * Four methods, and the first is the reason the interface exists.
 *
 * `claim()` is ONE call. "Read the next task" followed by "mark it mine" is two
 * round trips with a window between them, and two workers arriving in that
 * window both get the same task. An implementation must take the next available
 * task and record the claim as a single atomic operation -- not for
 * convenience, but because a source that cannot is not an implementation of
 * this contract.
 */
export interface AgentTaskSource {
  /**
   * Atomically take the next available task, or null when there is none.
   *
   * Tasks come back in INSERTION ORDER unless the source defines otherwise. A
   * source may expose an explicit position; it must not reorder implicitly.
   * Nothing errors when ordering changes -- the agent simply does the work in a
   * different sequence and produces a different result, which is the hardest
   * kind of divergence to notice.
   */
  claim(worker: string, leaseSeconds?: number): Promise<AgentTask | null>;

  /**
   * Record what happened, as the worker that is HOLDING the task.
   *
   * Called by the APPLICATION, from evidence -- not by the agent. See
   * `agentCompletionTool` for the opt-in that changes that, and why it is off
   * until a consumer turns it on.
   *
   * The `worker` argument is not in the spec's three-method sketch and is here
   * on purpose. Without it, a completion tool handed a source can close ANY
   * task in the list, including one another worker is midway through -- and the
   * whole reason completion authority is gated is that an agent closing work it
   * did not do is the failure being designed against. A release by anyone but
   * the holder is refused.
   */
  release(task: AgentTask, worker: string, outcome: TaskOutcome): Promise<void>;

  /**
   * How many tasks remain claimable.
   *
   * A COUNT, not a listing. It exists to terminate the loop and a count is
   * enough for that; a listing invites the source to materialise every task on
   * every pass, and a consumer that wants one already has its own query.
   */
  pending(): Promise<number>;

  /**
   * The task with this id, as it stands now, or null.
   *
   * On the contract because `release()` takes a TASK and every external caller
   * holds only an ID. A tool call carries `{"id": "t-1"}`; an HTTP route has
   * `/tasks/t-1`; a worker resuming after a restart has whatever it wrote down.
   * Without this the contract could be driven only by code that still had the
   * object `claim()` returned, which is the one caller that does not need it.
   *
   * A FRESH read, with any expired lease already applied -- so a task whose
   * lease ran out reports `todo` here even if the store has not been rewritten
   * yet.
   */
  find(id: string): Promise<AgentTask | null>;
}

/**
 * The stored shape of a task. THE KEYS ARE THE CONTRACT.
 *
 * `claimed_by` and `claimed_until` are `| null` and NOT optional, and that
 * distinction is why this type is spelled out rather than inferred. TypeScript
 * has both `null` and `undefined` where PHP has one absent value; an optional
 * property holding `undefined` is DROPPED by `JSON.stringify`; and a dropped
 * key is different bytes -- an observable decision under prism-parity decision
 * 0002, not a formatting detail.
 *
 * `claimed_until` is an integer Unix timestamp rather than a formatted date,
 * because date formatting is exactly where three languages produce three
 * strings from one instant.
 */
export interface AgentTaskRecord {
  claimed_by: string | null;
  claimed_until: number | null;
  id: string;
  instruction: string;
  state: TaskState;
}

/**
 * A task that also exposes its lease.
 *
 * Kept apart from `AgentTask` because the three-method contract is what an
 * adapter over an existing table must satisfy, and such a table may have
 * nowhere to put an owner and an expiry. Everything this package produces
 * satisfies both.
 */
export interface LeasedAgentTask extends AgentTask {
  /** Present-and-null when unclaimed. */
  claimedBy(): string | null;

  /** Integer Unix timestamp; present-and-null when unclaimed. */
  claimedUntil(): number | null;

  toRecord(): AgentTaskRecord;
}

/**
 * The five attributes a consumer's own record has to expose.
 *
 * Camel-cased because this is the in-language shape; `AgentTaskRecord` is the
 * stored one, and the two are converted rather than conflated. Only the stored
 * one is pinned across languages.
 */
export interface AgentTaskAttributes {
  id: string;
  instruction: string;
  state: TaskState;
  /** PRESENT-AND-NULL when unclaimed. Never `undefined`; see `AgentTaskRecord`. */
  claimedBy: string | null;
  claimedUntil: number | null;
}

export function isTaskState(value: unknown): value is TaskState {
  return typeof value === 'string' && (TASK_STATES as readonly string[]).includes(value);
}

/** `done` and `failed`. Nothing leaves either. */
export function isTerminalState(state: TaskState): boolean {
  return state === 'done' || state === 'failed';
}

/**
 * Canonical JSON for one task record.
 *
 * ```json
 * {"claimed_by":null,"claimed_until":null,"id":"t-1","instruction":"...","state":"todo"}
 * ```
 *
 * Keys SORTED -- which for these five is also the order above, so one task has
 * one byte sequence rather than two.
 *
 * `undefined` is coerced to `null` instead of being handed to `JSON.stringify`,
 * which would silently drop the key. That coercion is a backstop for values
 * arriving from a store or an untyped consumer; the types above already make it
 * unreachable from typed code, and a test asserts both halves rather than
 * trusting the type.
 *
 * No whitespace, no escaped slashes, no escaped non-ASCII: JavaScript's
 * `JSON.stringify` already produces all three where PHP's and Python's defaults
 * do not. See prism-parity decision 0005.
 */
export function canonicalTaskJson(record: AgentTaskRecord): string {
  const loose = record as unknown as Record<string, unknown>;
  const ordered: Record<string, unknown> = {};

  for (const key of Object.keys(loose).sort()) {
    const value = loose[key];

    ordered[key] = value === undefined ? null : value;
  }

  return JSON.stringify(ordered);
}

/**
 * Rebuild a record from whatever a store handed back.
 *
 * A missing or `undefined` claim field becomes `null`, so there is exactly one
 * representation of "unclaimed" once a row has been through here. A missing id,
 * instruction or state is a CORRUPT row rather than a defaultable one, and is
 * refused -- defaulting a missing state to `todo` would hand a half-finished
 * task back out as if it had never been attempted.
 */
export function taskRecordFrom(value: JsonObject | Record<string, unknown>): AgentTaskRecord {
  const loose = value as Record<string, unknown>;
  const id = loose.id;
  const instruction = loose.instruction;
  const state = loose.state;

  if (typeof id !== 'string' || id === '') {
    throw HarnessError.unmappableContent('a stored task record has no id');
  }

  if (typeof instruction !== 'string') {
    throw HarnessError.unmappableContent(`the stored task [${id}] has no instruction`);
  }

  if (!isTaskState(state)) {
    throw HarnessError.unmappableContent(
      `the stored task [${id}] has the state [${String(state)}], which is not one of ${TASK_STATES.join(', ')}`,
    );
  }

  const claimedBy = loose.claimed_by;
  const claimedUntil = loose.claimed_until;

  return {
    claimed_by: typeof claimedBy === 'string' ? claimedBy : null,
    claimed_until: typeof claimedUntil === 'number' ? Math.trunc(claimedUntil) : null,
    id,
    instruction,
    state,
  };
}

export function recordToAttributes(record: AgentTaskRecord): AgentTaskAttributes {
  return {
    id: record.id,
    instruction: record.instruction,
    state: record.state,
    claimedBy: record.claimed_by,
    claimedUntil: record.claimed_until,
  };
}

/**
 * The stored record for a set of attributes.
 *
 * `?? null` on both claim fields is the load-bearing line in this file. An
 * untyped consumer handing over `undefined` would otherwise produce a record
 * whose keys vanish at serialisation time.
 */
export function attributesToRecord(attributes: AgentTaskAttributes): AgentTaskRecord {
  const until = attributes.claimedUntil;

  return {
    claimed_by: attributes.claimedBy ?? null,
    claimed_until: until === null || until === undefined ? null : Math.trunc(until),
    id: attributes.id,
    instruction: attributes.instruction,
    state: attributes.state,
  };
}

/**
 * A consumer's own record, as an `AgentTask`.
 *
 * This is the idiomatic equivalent of the reference's trait on an Eloquent
 * model, and the reason it looks nothing like a trait is decision 0002. PHP
 * needs a trait because a trait is how PHP shares behaviour into someone else's
 * class. TypeScript's conformance is STRUCTURAL: a consumer whose record
 * already carries the five attributes needs no inheritance at all, so the
 * natural spelling is a function that wraps one. Contorting the language to
 * imitate a trait would change the spelling and nothing observable.
 *
 * The attributes object is read at CALL TIME rather than copied, so a live
 * record another worker's claim changed underneath reports the new state.
 */
export function asAgentTask(attributes: AgentTaskAttributes): LeasedAgentTask {
  return {
    id: () => attributes.id,
    instruction: () => attributes.instruction,
    state: () => attributes.state,
    claimedBy: () => attributes.claimedBy ?? null,
    claimedUntil: () => attributes.claimedUntil ?? null,
    toRecord: () => attributesToRecord(attributes),
  };
}

/**
 * What the mixin needs from the class it is mixed into.
 *
 * ONE accessor rather than five fields, because a mixin cannot define a method
 * called `id()` on a class that already has an `id` property -- the instance
 * property shadows the prototype method and the method is never reached. That
 * is a real constraint of the language, not a preference, and it is the reason
 * the mixin asks for a mapping function instead of the attributes directly. A
 * consumer maps their own column names here, exactly as a PHP trait would need
 * them mapped.
 */
export interface AgentTaskAttributeSource {
  taskAttributes(): AgentTaskAttributes;
}

// A mixin's base constructor is unconstrained by construction; `any[]` here is
// the standard spelling and is confined to this one alias.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Constructor<T> = new (...args: any[]) => T;

/**
 * The mixin: `class Todo extends withAgentTask(Row) {}`.
 *
 * Same behaviour as `asAgentTask`, reached the way a consumer who wants the
 * methods ON their model reaches it. Both exist because TypeScript has two
 * natural spellings of "share this behaviour into someone else's class" and
 * neither is more correct than the other.
 */
export function withAgentTask<TBase extends Constructor<AgentTaskAttributeSource>>(
  Base: TBase,
): TBase & Constructor<LeasedAgentTask> {
  class AgentTaskRow extends Base implements LeasedAgentTask {
    id(): string {
      return this.taskAttributes().id;
    }

    instruction(): string {
      return this.taskAttributes().instruction;
    }

    state(): TaskState {
      return this.taskAttributes().state;
    }

    claimedBy(): string | null {
      return this.taskAttributes().claimedBy ?? null;
    }

    claimedUntil(): number | null {
      return this.taskAttributes().claimedUntil ?? null;
    }

    toRecord(): AgentTaskRecord {
      return attributesToRecord(this.taskAttributes());
    }
  }

  return AgentTaskRow;
}
