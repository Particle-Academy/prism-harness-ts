import { AsyncLocalStorage } from 'node:async_hooks';
import type { JsonObject, JsonValue } from './json.js';
import { isJsonObject } from './json.js';
import { pausedStepIndex, supersededRow, supersedes } from './thread-rows.js';
import { HarnessError } from './errors.js';

/**
 * How the RUNTIME records a row (G-404, BRIEF-R14). @internal Never a host's:
 * every field is the runtime's own, passed only by its own record calls.
 */
export interface RecordOptions {
  /**
   * The runtime's new message: the batch supersedes the paused step whatever
   * run records it, and the replaced denial is written in the SAME write as
   * the superseding row (never a write of its own before it).
   */
  supersede?: boolean;
  /**
   * Checked inside the thread lock against every stored row, before anything
   * is written; it throws to refuse the whole batch. How a decision's
   * pending check is one step with its write, whoever else writes the thread.
   */
  guard?: (rows: readonly JsonObject[]) => void;
  /** The resolver's in-flight claim this write settles: cleared right after it. */
  settles?: string;
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
    if (supportsAppendLog(this.store)) {
      const log = await this.store.readLog(this.key);
      if (log !== null) return log.map((entry) => this.#toMessage(entry));
    }

    return (await this.#stored()).map((entry) => this.#toMessage(entry));
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

    const store = this.store;
    // Inside the run that owns a step is only what the RUNTIME records for
    // that run (recordingInRun): a run id a caller passes here proves nothing.
    // The marker is SINGLE-USE (G-406): taken here, synchronously, by the one
    // record call the runtime made inside it, so a write a store hook or a
    // listener makes during that call's window is outside the run.
    const token = recording.getStore();
    const inRun = token !== undefined && !token.used && token.runId === runId ? runId : null;
    if (token !== undefined) token.used = true;
    const opts: RecordOptions = options;
    const forced = opts.supersede === true;

    if (supportsAppendLog(store)) {
      return store.withLock(this.key, async () => {
        const length = (await store.logLength(this.key)) ?? (await this.#migrate(store));
        // A supersede needs the stored rows only while a paused step has an
        // open call. Whether one has is kept beside the log (its position,
        // never a copy of a row; written only when it changes), so an append
        // with nothing open never reads the whole log. A guarded write (a
        // decision) reads it all: its check is about every stored row.
        const payload = (await store.get(this.key)) ?? {};
        const kept = keptOpen(payload[OPEN_STEP_KEY]);
        const full = kept === false && opts.guard === undefined ? [] : (await store.readLog(this.key)) ?? [];
        opts.guard?.(full.map((entry) => this.#toMessage(entry).message));
        const tail = openTail(full).map((entry) => this.#toMessage(entry));
        const appended = this.#entries(planned(tail, messages, inRun, forced, await this.#claimFor(tail, messages, inRun, forced)), length, runId);
        if (appended.length > 0) await store.appendLog(this.key, appended);
        // With nothing open before, an open step can only be one just appended.
        const next = openTail(kept === false ? appended : [...full, ...appended]);
        const open = next.length === 0 ? false : (next[0]!.position as number);
        if (kept !== open) {
          await store.put(this.key, { ...((await store.get(this.key)) ?? {}), [OPEN_STEP_KEY]: { v: 1, open } });
        }
        await this.#settle(opts.settles);

        return appended.map((entry) => this.#toMessage(entry));
      });
    }

    return store.withLock(this.key, async () => {
      const stored = (await store.get(this.key)) ?? {};
      const existing = Array.isArray(stored.messages) ? stored.messages.filter(isJsonObject) : [];
      const rows = existing.map((entry) => this.#toMessage(entry));
      opts.guard?.(rows.map((entry) => entry.message));

      const appended = this.#entries(planned(rows, messages, inRun, forced, await this.#claimFor(rows, messages, inRun, forced)), existing.length, runId);
      if (appended.length === 0) return [];

      await store.put(this.key, {
        ...stored,
        messages: [...existing, ...appended] as unknown as JsonValue[],
      });
      await this.#settle(opts.settles);

      return appended.map((entry) => this.#toMessage(entry));
    });
  }

  /** Where this thread's in-flight claim is kept: a key of its own, so claiming never rewrites the rows. */
  #claimKey(): string {
    return `${this.key}:in-flight`;
  }

  /** The live claim, read only when the batch would supersede the paused step (the only write it can refuse). */
  async #claimFor(existing: readonly ThreadMessage[], messages: readonly JsonObject[], inRun: string | null, forced: boolean): Promise<Claim | null> {
    if (supersedingAt(existing, messages, inRun, forced) === -1) return null;

    return liveClaim(await this.store.get(this.#claimKey()), this.key);
  }

  /** Clear the claim `token` names, right after the write that records its results. */
  async #settle(token: string | undefined): Promise<void> {
    if (token === undefined) return;
    const claim = await this.store.get(this.#claimKey());
    if (claim !== null && claim.token === token) await this.store.forget(this.#claimKey());
  }

  /**
   * Claim the paused step at `position` for a resolution that is about to run
   * its approved calls (G-404): while the claim lives, a write that would
   * supersede that step is refused `session_locked` and writes nothing, so
   * no call that may be running is ever given a "not run" denial beside its
   * real result. Checked and written inside the thread lock: false when the
   * step is no longer the paused one or has nothing open (it was superseded
   * or answered since it was read), and then nothing may run. The claim
   * expires after `ttlSeconds` (the session lock's own ttl), so a resolver
   * that died leaves nothing refusing for longer than its lock would have.
   * Kept under a key of its own (never rewriting the rows). Cleared right
   * after the write that records the results (`settles`), or by
   * releaseInFlight().
   *
   * @internal Called by the runtime's resolver, holding the session lock.
   */
  async claimInFlight(position: number, token: string, ttlSeconds: number): Promise<boolean> {
    const store = this.store;

    return store.withLock(this.key, async () => {
      const log = supportsAppendLog(store) ? await store.readLog(this.key) : null;
      const payload = log === null ? (await store.get(this.key)) ?? {} : {};
      const entries = (log ?? (Array.isArray(payload.messages) ? payload.messages.filter(isJsonObject) : [])).map((entry) => this.#toMessage(entry));
      const rows = entries.map((entry) => entry.message);
      const step = pausedStepIndex(rows);
      if (step === null || entries[step]!.position !== position || supersededRow(rows) === null) return false;
      await store.put(this.#claimKey(), { v: 1, position, token, until: Date.now() + ttlSeconds * 1000 }, Math.max(1, Math.ceil(ttlSeconds)));

      return true;
    });
  }

  /** Drop the claim `token` names, if it is still the one held. @internal */
  async releaseInFlight(token: string): Promise<void> {
    await this.store.withLock(this.key, async () => { await this.#settle(token); });
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

    return supersededRow((await this.messages()).map((entry) => entry.message)) !== null;
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

    await store.withLock(this.key, async () => {
      await store.forget(this.#claimKey());
      if (supportsAppendLog(store)) await store.clearLog(this.key);
      const stored = await store.get(this.key);
      if (stored !== null) {
        const { [OPEN_STEP_KEY]: _open, ...rest } = stored;
        await store.put(this.key, { ...rest, messages: [] });
      }
    });
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

  #entries(messages: readonly JsonObject[], length: number, runId: string | null): JsonObject[] {
    const recordedAt = new Date().toISOString();

    return messages.map((message, index) => ({
      position: length + index + 1,
      message,
      run_id: runId,
      recorded_at: recordedAt,
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

/**
 * The rows to append: `messages`, with the paused step's denials before the
 * first row that supersedes it (DECISION B), in the SAME write (G-404: never
 * a denial written alone, before its superseding row). Superseded by any
 * later assistant row with tool calls; or a later assistant text row or user
 * row recorded outside the runs that own the step (the run that recorded it
 * and every run that recorded one of its results): its run id null, another
 * run, or a run id the caller merely passed (only the runtime's
 * recordingInRun() makes a row the run's own). `forced`: the runtime's new
 * message, outside whatever run records it. Refused `session_locked`, with
 * nothing written, while `claim` (a resolution running the step's approved
 * calls) lives.
 */
function planned(existing: readonly ThreadMessage[], messages: readonly JsonObject[], inRun: string | null, forced: boolean, claim: Claim | null): JsonObject[] {
  const rows = existing.map((entry) => entry.message);
  const step = pausedStepIndex(rows);
  const at = supersedingAt(existing, messages, inRun, forced);
  if (at === -1) return [...messages];
  // A resolution is running this step's approved calls (G-404): a write that
  // would supersede it is refused, and nothing of the batch is written.
  if (claim !== null && step !== null && existing[step]!.position === claim.position) throw HarnessError.sessionLocked(claim.key, 0);
  const before = messages.slice(0, at).filter((message) => message.type === 'tool_result');
  const denial = supersededRow(rows, before);
  if (denial === null) return [...messages];

  return [...messages.slice(0, at), denial as unknown as JsonObject, ...messages.slice(at)];
}

/**
 * The index in `messages` of the first row that supersedes the paused step
 * of `existing`, or -1 (DECISION B, the rule planned() writes the denial by).
 * The paused step, by the one selector (G-397), and the runs that own it:
 * the run that recorded it and every run that recorded one of its results.
 */
function supersedingAt(existing: readonly ThreadMessage[], messages: readonly JsonObject[], inRun: string | null, forced: boolean): number {
  const step = pausedStepIndex(existing.map((entry) => entry.message));
  const owners = new Set<string>();
  if (step !== null) {
    const own = existing[step]!.runId;
    if (own !== null) owners.add(own);
    for (const entry of existing.slice(step + 1)) {
      if (entry.message.type === 'tool_result' && entry.runId !== null && Array.isArray(entry.message.tool_results) && entry.message.tool_results.length > 0) owners.add(entry.runId);
    }
  }
  const outside = forced || inRun === null || !owners.has(inRun);

  return messages.findIndex((message) => supersedes(message, outside));
}

/**
 * A resolution's in-flight claim (G-404), kept under the thread's own
 * `:in-flight` key: `{v: 1, position, token, until}`, the paused step a
 * resolver is running approved calls of, until when.
 */
type Claim = { key: string; position: number };

/** The live claim, or null (absent, malformed or expired). */
function liveClaim(claim: JsonObject | null, key: string): Claim | null {
  if (claim === null || claim.v !== 1 || typeof claim.position !== 'number' || typeof claim.until !== 'number') return null;

  return claim.until > Date.now() ? { key, position: claim.position } : null;
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
 * The stored entries from the paused step on (pausedStepIndex), while that
 * step has a call with no result; otherwise none. All a supersede reads.
 */
function openTail(entries: readonly JsonObject[]): JsonObject[] {
  const rows = entries.map((entry) => (isJsonObject(entry.message) ? entry.message : {}));
  const step = pausedStepIndex(rows);
  if (step === null) return [];
  const tail = rows.slice(step);

  return supersededRow(tail) === null ? [] : entries.slice(step);
}
