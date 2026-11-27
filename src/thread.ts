import { AsyncLocalStorage } from 'node:async_hooks';
import type { JsonObject, JsonValue } from './json.js';
import { isJsonObject } from './json.js';
import { ANSWERS_STEP_KEY, CALL_BINDING_KEY, ENDED_BY_START_KEY, NOT_RUN_KEY, REPLACED_STEP_KEY, STARTED_KEY, SUPERSEDED_KEY, markedOwner, pausedStepIndex, supersededRow, supersedes, withBoundRequests } from './thread-rows.js';
import { HarnessError } from './errors.js';

/**
 * How the RUNTIME records a row (G-404, BRIEF-R14). @internal Never a host's:
 * `supersede` and `build` are honoured only on the runtime's own in-run
 * record call (the private single-use token, G-406); `guard` only ever
 * refuses a write.
 */
export interface RecordOptions {
  /**
   * The runtime's new message: the batch supersedes the paused step whatever
   * run records it, and the replaced denial is written in the SAME write as
   * the superseding row (never a write of its own before it).
   */
  supersede?: boolean;
  /**
   * Checked inside the thread lock against every stored row (the reader view,
   * readerRows), before anything is written; it throws to refuse the whole
   * batch. How a decision's pending check, or a started row's "the call is
   * still open", is one step with its write, whoever else writes the thread.
   */
  guard?: (rows: readonly JsonObject[]) => void;
  /**
   * The batch, built inside the thread lock from every stored row (the
   * reader view): how a resolution records an outcome only for a call that
   * still has none (round 15), whoever wrote since it was read.
   */
  build?: (rows: readonly JsonObject[]) => JsonObject[];
}
import type { AppendLogStore, SessionStore } from './stores/session-store.js';
import { supportsAppendLog } from './stores/session-store.js';

export interface ThreadMessage {
  /** Position in the conversation, from 1. Assigned by the thread, never by the caller. */
  position: number;
  /** The serialized message — whatever `prism-ts`'s `messageFromObject` can rebuild. */
  message: JsonObject;
  /** The run that produced it, when there was one. */
  runId: string | null;
  recordedAt: string;
}

/**
 * The marks only the harness writes (G-387 round 15): honoured only on a row
 * the harness itself recorded under its private in-run token. On any other
 * row they are read exactly as if absent.
 */
const ROW_MARKS: readonly string[] = [ANSWERS_STEP_KEY, STARTED_KEY, ENDED_BY_START_KEY, SUPERSEDED_KEY, REPLACED_STEP_KEY];

/**
 * The rows as every harness reader takes them (G-387 round 15): a row the
 * harness wrote carries `harness: true`; any other row is read without the
 * harness's marks (ROW_MARKS on the row, `not_run` on its results) and
 * without a `harness` key of its own, so a host row can never present one.
 * The whole thread in order: the row at index i is at position i + 1.
 *
 * @internal
 */
export function readerRows(entries: readonly { message: JsonObject; harness?: boolean }[]): JsonObject[] {
  return entries.map((entry) => readerRow(entry.message, entry.harness === true));
}

function readerRow(message: JsonObject, harness: boolean): JsonObject {
  if (harness) return { ...message, harness: true };
  const marked = Object.hasOwn(message, 'harness') || ROW_MARKS.some((key) => Object.hasOwn(message, key))
    || (Array.isArray(message.tool_results) && message.tool_results.some((entry) => isJsonObject(entry) && Object.hasOwn(entry, NOT_RUN_KEY)))
    || (Array.isArray(message.tool_approval_requests) && message.tool_approval_requests.some((entry) => isJsonObject(entry) && Object.hasOwn(entry, CALL_BINDING_KEY)));
  if (!marked) return message;
  // Every mark is read as ABSENT (round 16, ONE TRUST SIGNAL), the request
  // binding included: a request on a row the runtime did not write binds to
  // its own step's call by position, the documented rule for a request with
  // none (the binding Thread.record() wrote on it names that same call).
  const without = (key: string) => (entry: JsonValue): JsonValue => {
    if (!isJsonObject(entry) || !Object.hasOwn(entry, key)) return entry;
    const { [key]: _mark, ...rest } = entry;
    return rest;
  };
  const row: JsonObject = {};
  for (const [key, value] of Object.entries(message)) {
    if (key === 'harness' || ROW_MARKS.includes(key)) continue;
    row[key] = key === 'tool_results' && Array.isArray(value) ? value.map(without(NOT_RUN_KEY))
      : key === 'tool_approval_requests' && Array.isArray(value) ? value.map(without(CALL_BINDING_KEY))
        : value as JsonValue;
  }
  return row;
}

/**
 * The stored conversation a session is bound to.
 *
 * DURABLE by construction: it lives in the durable slot, so it is the one thing
 * here a flushed cache cannot take away. The PHP reference keeps it as Eloquent
 * rows for the same reason.
 *
 * ## Position is assigned inside a lock
 *
 * `record()` takes the lock before reading the current length. Two turns
 * landing concurrently would otherwise both read position 4 and both write
 * position 5, and the conversation would silently lose a message — the same
 * race the reference tracks as prism-harness#2. Doing the read and the write
 * inside one lock is the fix, not a retry afterwards.
 */
export class Thread {
  constructor(
    private readonly store: SessionStore,
    private readonly key: string,
  ) {}

  /** Existence probe only: never starts a log, migrates data or acquires a lock. */
  async exists(): Promise<boolean> {
    if (supportsAppendLog(this.store) && await this.store.logLength(this.key) !== null) return true;
    return await this.store.get(this.key) !== null;
  }

  async messages(): Promise<ThreadMessage[]> {
    return (await this.#read()).map((entry) => this.#toMessage(entry));
  }

  /**
   * Every stored entry, read ONCE per turn's snapshot (BRIEF-R16 item 7):
   * inside a turn (withThreadSnapshots), the readers before its first write
   * share one read of the thread; any write through a Thread, and taking the
   * session lock, ends it. Outside a turn, a fresh read every time. The
   * thread's own locked reads (record()'s guard and build) never use it.
   */
  async #read(): Promise<JsonObject[]> {
    const snapshot = snapshots.getStore();
    const kept = snapshot?.get(this.key);
    if (kept !== undefined && kept.store === this.store) return kept.entries;
    let entries: JsonObject[] | null = null;
    if (supportsAppendLog(this.store)) entries = await this.store.readLog(this.key);
    entries ??= await this.#stored();
    snapshot?.set(this.key, { store: this.store, entries });
    return entries;
  }

  /** The whole-payload form: every store before the log, and any store without one. */
  async #stored(): Promise<JsonObject[]> {
    const stored = await this.store.get(this.key);

    if (stored === null) return [];

    const messages = stored.messages;

    return Array.isArray(messages) ? messages.filter(isJsonObject) : [];
  }

  /** How many messages the thread holds. */
  async count(): Promise<number> {
    if (supportsAppendLog(this.store)) {
      const length = await this.store.logLength(this.key);
      if (length !== null) return length;
    }

    return (await this.#stored()).length;
  }

  /**
   * Append messages, in order, and return them with their assigned positions.
   *
   * Read-and-write inside ONE lock. See the class docblock.
   */
  async record(messages: readonly JsonObject[], runId: string | null = null, options: RecordOptions = {}): Promise<ThreadMessage[]> {
    if (messages.length === 0) return [];
    try {
      return await this.#record(messages, runId, options);
    } finally {
      // A write ends the turn's snapshot of this thread (item 7), written or not.
      forgetThreadSnapshot(this.key);
    }
  }

  async #record(messages: readonly JsonObject[], runId: string | null, options: RecordOptions): Promise<ThreadMessage[]> {
    const store = this.store;
    // Inside the run that owns a step is only what the RUNTIME records for
    // that run (recordingInRun): a run id a caller passes here proves nothing.
    // The marker is SINGLE-USE (G-406): taken here, synchronously, by the one
    // record call the runtime made inside it, so a write a store hook or a
    // listener makes during that call's window is outside the run. A row
    // written under it is the harness's own (its entry says so), and only
    // such a row's marks are ever honoured (readerRows).
    const token = recording.getStore();
    const inRun = token !== undefined && !token.used && token.runId === runId ? runId : null;
    if (token !== undefined) token.used = true;
    const forced = inRun !== null && options.supersede === true;
    const build = inRun !== null ? options.build : undefined;
    const guard = options.guard;

    if (supportsAppendLog(store)) {
      return store.withLock(this.key, async () => {
        const length = (await store.logLength(this.key)) ?? (await this.#migrate(store));
        // A supersede needs the stored rows only while a paused step has an
        // open call. Whether one has is kept beside the log (its position,
        // never a copy of a row; written only when it changes), so an append
        // with nothing open never reads the whole log. A guarded or built
        // write reads it all: its check is about every stored row.
        const payload = (await store.get(this.key)) ?? {};
        const kept = keptOpen(payload[OPEN_STEP_KEY]);
        const full = kept === false && guard === undefined && build === undefined ? [] : (await store.readLog(this.key)) ?? [];
        const rows = this.#view(full);
        guard?.(rows);
        const batch = build === undefined ? [...messages] : build(rows);
        const appended = this.#entries(planned(rows, runs(full), batch, inRun, forced), length, runId, inRun !== null);
        if (appended.length > 0) await store.appendLog(this.key, appended);
        // With nothing open before, an open step can only be one just appended.
        const covered = kept === false ? appended : [...full, ...appended];
        const next = openStep(this.#view(covered), covered);
        if (kept !== next) {
          await store.put(this.key, { ...((await store.get(this.key)) ?? {}), [OPEN_STEP_KEY]: { v: 1, open: next } });
        }

        return appended.map((entry) => this.#toMessage(entry));
      });
    }

    return store.withLock(this.key, async () => {
      const stored = (await store.get(this.key)) ?? {};
      const existing = Array.isArray(stored.messages) ? stored.messages.filter(isJsonObject) : [];
      const rows = this.#view(existing);
      guard?.(rows);
      const batch = build === undefined ? [...messages] : build(rows);

      const appended = this.#entries(planned(rows, runs(existing), batch, inRun, forced), existing.length, runId, inRun !== null);
      if (appended.length === 0) return [];

      await store.put(this.key, {
        ...stored,
        messages: [...existing, ...appended] as unknown as JsonValue[],
      });

      return appended.map((entry) => this.#toMessage(entry));
    });
  }

  /** Stored entries as the readers take them (readerRows). */
  #view(entries: readonly JsonObject[]): JsonObject[] {
    return readerRows(entries.map((entry) => ({ message: this.#toMessage(entry).message, harness: entry.harness === true })));
  }

  /**
   * Every stored row as the harness's readers take them (readerRows): what
   * the runtime reads to decide. @internal
   */
  async readerRows(): Promise<JsonObject[]> {
    return this.#view(await this.#read());
  }

  /**
   * Whether the paused step has a call with no result (a new message would
   * supersede it): read from the open-step fact kept beside a log, else the
   * rows. @internal
   */
  async hasOpenStep(): Promise<boolean> {
    if (supportsAppendLog(this.store)) {
      const kept = keptOpen(((await this.store.get(this.key)) ?? {})[OPEN_STEP_KEY]);
      if (kept === false) return false;
    }

    return supersededRow(await this.readerRows()) !== null;
  }

  /**
   * Forget the conversation.
   *
   * Deliberately separate from `Session.forget()`, which drops only the
   * ephemeral half. Losing a thread is not a cache miss and must never be a
   * side effect of clearing session state.
   */
  async clear(): Promise<void> {
    const store = this.store;
    forgetThreadSnapshot(this.key);

    try {
      await store.withLock(this.key, async () => {
        if (supportsAppendLog(store)) await store.clearLog(this.key);
        const stored = await store.get(this.key);
        if (stored !== null) {
          const { [OPEN_STEP_KEY]: _open, ...rest } = stored;
          await store.put(this.key, { ...rest, messages: [] });
        }
      });
    } finally {
      forgetThreadSnapshot(this.key);
    }
  }

  /**
   * Move a whole-payload thread into the log, once, the first time it grows.
   *
   * The log is written FIRST and the payload emptied after it. A crash between
   * the two leaves both, and the log wins on every read, so nothing is lost
   * and nothing is counted twice. Returns how many entries the log now holds.
   */
  async #migrate(store: SessionStore & AppendLogStore): Promise<number> {
    const existing = await this.#stored();
    await store.appendLog(this.key, existing);

    if (existing.length > 0) {
      const stored = (await store.get(this.key)) ?? {};
      await store.put(this.key, { ...stored, messages: [] });
    }

    return existing.length;
  }

  /**
   * The entries to append. `harness`: the batch is the runtime's own in-run
   * write; a replaced denial the thread adds (planned) is always the
   * harness's own. Only such an entry carries `harness: true` (TS 0.11.x's
   * entry shape), and only its marks are read (readerRows).
   */
  #entries(messages: readonly Planned[], length: number, runId: string | null, harness: boolean): JsonObject[] {
    const recordedAt = new Date().toISOString();

    // Every stored step's approval requests are bound to their own calls
    // (round 15): a host's as well as the runtime's.
    return messages.map(({ message, own }, index) => ({
      position: length + index + 1,
      message: withBoundRequests(message),
      run_id: runId,
      recorded_at: recordedAt,
      ...(harness || own ? { harness: true } : {}),
    }));
  }

  #toMessage(entry: JsonObject): ThreadMessage {
    const message = entry.message;

    if (!isJsonObject(message)) {
      throw HarnessError.unmappableContent(
        `a stored thread entry has no message object (position ${String(entry.position)})`,
      );
    }

    return {
      position: typeof entry.position === 'number' ? entry.position : 0,
      message,
      runId: typeof entry.run_id === 'string' ? entry.run_id : null,
      recordedAt: typeof entry.recorded_at === 'string' ? entry.recorded_at : '',
    };
  }
}

/** The turn's snapshot of each thread it read (Thread.#read), by thread key. */
const snapshots = new AsyncLocalStorage<Map<string, { store: SessionStore; entries: JsonObject[] }>>();

/**
 * Run `body` as one turn (BRIEF-R16 item 7): the thread reads it makes before
 * its first write share one read of each thread. @internal Used by the runtime.
 */
export function withThreadSnapshots<T>(body: () => T): T {
  // A nested turn (a tool's own send) shares its outer turn's snapshot, so a
  // write by either ends it for both.
  return snapshots.run(snapshots.getStore() ?? new Map(), body);
}

/** End the current turn's snapshot of the thread under `key` (a write, a session lock). @internal */
export function forgetThreadSnapshot(key: string): void {
  snapshots.getStore()?.delete(key);
}

/**
 * The run the runtime is recording rows for, set only around its own record()
 * calls (DECISION B): a token the FIRST record() inside it takes (G-406).
 */
const recording = new AsyncLocalStorage<{ runId: string; used: boolean }>();

/**
 * Record rows as the harness's own, inside `runId` (DECISION B): the only way
 * a row counts as written inside the run that owns a paused step. The token
 * is private and single-use: the one record() `record` makes takes it, so
 * anything recorded during that write's window (a store hook, a listener, a
 * callback scheduled from inside it) is outside the run.
 *
 * @internal Called by the runtime around its own record() calls.
 */
export function recordingInRun<T>(runId: string, record: () => T): T {
  return recording.run({ runId, used: false }, record);
}

/** A row to append, and whether the thread itself wrote it (a replaced denial). */
type Planned = { message: JsonObject; own: boolean };

/**
 * The rows to append: `messages`, with the paused step's denials before the
 * first row that supersedes it (DECISION B), in the SAME write (G-404: never
 * a denial written alone, before its superseding row). Superseded by any
 * later assistant row with tool calls; or a later assistant text row or user
 * row recorded outside the runs that own the step (the run that recorded it
 * and every run that recorded one of its results): its run id null, another
 * run, or a run id the caller merely passed (only the runtime's
 * recordingInRun() makes a row the run's own). `forced`: the runtime's new
 * message, outside whatever run records it.
 *
 * NEVER REFUSED (round 15): a call the resolver has started (its started
 * row, and no result) is answered in progress, so the denial leaves it out,
 * and nothing about a running call can refuse a host's or a tool's write.
 * `rows` are every stored row as the readers take them; `runIds` the run
 * that recorded each.
 */
function planned(rows: readonly JsonObject[], runIds: readonly (string | null)[], messages: readonly JsonObject[], inRun: string | null, forced: boolean): Planned[] {
  const plain = messages.map((message) => ({ message, own: false }));
  const at = supersedingAt(rows, runIds, messages, inRun, forced);
  if (at === -1) return plain;
  const before = messages.slice(0, at).filter((message) => message.type === 'tool_result').map((message) => (inRun === null ? message : { ...message, harness: true }));
  const denial = supersededRow(rows, before);
  if (denial === null) return plain;

  return [...plain.slice(0, at), { message: denial as unknown as JsonObject, own: true }, ...plain.slice(at)];
}

/** The run that recorded each stored entry. */
function runs(entries: readonly JsonObject[]): (string | null)[] {
  return entries.map((entry) => (typeof entry.run_id === 'string' ? entry.run_id : null));
}

/**
 * The index in `messages` of the first row that supersedes the paused step
 * of `rows`, or -1 (DECISION B, the rule planned() writes the denial by).
 * The paused step, by the one selector (G-397), and the runs that own it:
 * the run that recorded it and every run that recorded one of its results.
 */
function supersedingAt(rows: readonly JsonObject[], runIds: readonly (string | null)[], messages: readonly JsonObject[], inRun: string | null, forced: boolean): number {
  const step = pausedStepIndex(rows);
  const owners = new Set<string>();
  if (step !== null) {
    // ONE TRUST SIGNAL (round 16): only rows the runtime wrote make a run an
    // owner; a run id a host row carries is never evidence of anything.
    const own = rows[step]?.harness === true ? runIds[step] ?? null : null;
    if (own !== null) owners.add(own);
    for (const [index, row] of rows.entries()) {
      if (index <= step || row.harness !== true) continue;
      const runId = runIds[index] ?? null;
      // A result row a harness mark gives to an EARLIER step (round 15) is
      // that step's: its run does not own this one.
      const owner = markedOwner(rows, index);
      if (owner !== undefined && owner !== step) continue;
      if (row.type === 'tool_result' && runId !== null && Array.isArray(row.tool_results) && row.tool_results.length > 0) owners.add(runId);
    }
  }
  const outside = forced || inRun === null || !owners.has(inRun);

  return messages.findIndex((message) => supersedes(message, outside));
}

/** Where an append-log thread keeps whether its paused step has an open call (DECISION B): its position, or false. */
const OPEN_STEP_KEY = 'harness_open_step';

/**
 * The kept open-step fact: the position of the paused step with an open
 * call, `false` for none, or undefined when it is not kept (a thread written
 * before it): then the log is read.
 */
function keptOpen(value: JsonValue | undefined): number | false | undefined {
  if (!isJsonObject(value) || value.v !== 1) return undefined;
  return value.open === false ? false : typeof value.open === 'number' && Number.isSafeInteger(value.open) ? value.open : undefined;
}

/**
 * The position of the paused step while it has a call with no answer (a
 * supersede would deny it), else false. `rows`: entries as the readers take
 * them, the row at index i at position i + 1 of what they cover.
 */
function openStep(rows: readonly JsonObject[], entries: readonly JsonObject[]): number | false {
  const step = pausedStepIndex(rows);
  if (step === null || supersededRow(rows) === null) return false;
  const position = entries[step]?.position;

  return typeof position === 'number' ? position : false;
}
