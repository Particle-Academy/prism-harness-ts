import type { JsonObject } from './json.js';
import type { RunStep } from './steps.js';

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
 *
 * The one exception is `run.stepped`, which carries a run step whole and is
 * therefore OFF unless the runtime is built with `stepEvents: true`.
 */
export type HarnessEvent = RunStarted | RunResumed | RunFinished | RunFailed | RunStepped;

/**
 * The named agent a run belongs to (G-321 E), on every event of a named
 * agent's run and ABSENT for the legacy namespace, so a legacy event is the
 * same object it always was.
 *
 * Its own name on purpose: it is the harness agent id, unrelated to the
 * presence `agentId` agent-integrations uses, and the harness never maps one
 * onto the other.
 */
export interface HarnessAgentTag {
  harnessAgent?: string;
}

export interface RunStarted extends HarnessAgentTag {
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

/**
 * A run parked for approval continued under its own id (G-284).
 *
 * Emitted INSTEAD of a second `run.started`: the run started once, and an
 * application counting starts would otherwise see two runs where there is one.
 * The run still ends with exactly one `run.finished` or `run.failed`. The same
 * fields as `run.started`.
 */
export interface RunResumed extends Omit<RunStarted, 'type'> {
  type: 'run.resumed';
}

/**
 * A run produced a step (G-282 A).
 *
 * OPT-IN: emitted only when the runtime was built with `stepEvents: true`. A
 * step carries tool arguments, tool results and model text — exactly what the
 * rest of these events leave out — so an app that forwards every event must
 * ask for it. `streamSteps()` and `streamDecision()` yield steps either way.
 * Observation only: a listener that throws is isolated, as for every event.
 */
export interface RunStepped extends HarnessAgentTag {
  type: 'run.stepped';
  runId: string;
  sessionKey: string;
  parentRunId: string | null;
  rootRunId: string;
  step: RunStep;
}

export interface RunFinished extends HarnessAgentTag {
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

export interface RunFailed extends HarnessAgentTag {
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
