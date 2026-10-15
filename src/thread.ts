import type { JsonObject, JsonValue } from './json.js';
import { isJsonObject } from './json.js';
import { HarnessError } from './errors.js';
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
  async record(messages: readonly JsonObject[], runId: string | null = null): Promise<ThreadMessage[]> {
    if (messages.length === 0) return [];

    const store = this.store;

    if (supportsAppendLog(store)) {
      return store.withLock(this.key, async () => {
        const length = (await store.logLength(this.key)) ?? (await this.#migrate(store));
        const appended = this.#entries(messages, length, runId);
        await store.appendLog(this.key, appended);

        return appended.map((entry) => this.#toMessage(entry));
      });
    }

    return store.withLock(this.key, async () => {
      const stored = (await store.get(this.key)) ?? {};
      const existing = Array.isArray(stored.messages) ? stored.messages.filter(isJsonObject) : [];

      const appended = this.#entries(messages, existing.length, runId);

      await store.put(this.key, {
        ...stored,
        messages: [...existing, ...appended] as unknown as JsonValue[],
      });

      return appended.map((entry) => this.#toMessage(entry));
    });
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
      if (supportsAppendLog(store)) await store.clearLog(this.key);
      const stored = await store.get(this.key);
      if (stored !== null) await store.put(this.key, { ...stored, messages: [] });
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
