import type { JsonObject } from './json.js';

/**
 * What happened during a run, for anything that wants to watch.
 *
 * A plain discriminated union and a synchronous listener list rather than an
 * event bus: this package has no framework to hang one off, and a consumer that
 * wants queues or broadcasting already has somewhere better to put them. The
 * point is that the harness EMITS at the right moments, not that it owns the
 * delivery mechanism.
 *
 * NO PROMPTS, NO TOOL ARGUMENTS, and that boundary is the same one
 * `Session.completeRun()` holds: tool NAMES are what an operator needs to audit
 * a guardrail and are not PII, while arguments are — `prism-opentelemetry`
 * already carries those behind an opt-in capture gate with a length cap, and
 * emitting them here ungated would quietly undo that decision for anyone who
 * installed both.
 */
export type HarnessEvent = RunStarted | RunFinished | RunFailed;

export interface RunStarted {
  type: 'run.started';
  runId: string;
  sessionKey: string;
  mode: string;
  provider: string;
  model: string;
  /** The root of the tree this run belongs to; equal to `runId` for a root run. */
  rootRunId: string;
  depth: number;
  at: string;
}

export interface RunFinished {
  type: 'run.finished';
  runId: string;
  sessionKey: string;
  finishReason: string;
  /** NAMES only, in call order. */
  toolCalls: readonly string[];
  steps: number;
  costUsd: number | null;
  at: string;
}

export interface RunFailed {
  type: 'run.failed';
  runId: string;
  sessionKey: string;
  /** Why it stopped: a budget reason, a cancellation, or a provider failure. */
  failure: string;
  steps: number;
  at: string;
}

export type HarnessListener = (event: HarnessEvent) => void;

export class HarnessEvents {
  readonly #listeners: HarnessListener[] = [];

  /** Returns an unsubscribe function, so a caller can stop listening. */
  listen(listener: HarnessListener): () => void {
    this.#listeners.push(listener);

    return () => {
      const index = this.#listeners.indexOf(listener);
      if (index >= 0) this.#listeners.splice(index, 1);
    };
  }

  /**
   * Deliver an event to every listener.
   *
   * A THROWING LISTENER MUST NOT BREAK THE RUN. Telemetry that takes down the
   * thing it observes is worse than no telemetry, and a listener is by
   * definition somebody else's code. Failures are collected and reported after
   * every listener has had its turn.
   */
  emit(event: HarnessEvent): void {
    const failures: unknown[] = [];

    for (const listener of [...this.#listeners]) {
      try {
        listener(event);
      } catch (error) {
        failures.push(error);
      }
    }

    if (failures.length > 0 && typeof process !== 'undefined') {
      for (const failure of failures) {
        process.emitWarning(
          `A prism-harness event listener threw while handling [${event.type}]: ${String(failure)}`,
        );
      }
    }
  }

  /** For a consumer that wants to persist or forward the event as data. */
  static toObject(event: HarnessEvent): JsonObject {
    return { ...event } as unknown as JsonObject;
  }
}
