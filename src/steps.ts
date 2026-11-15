import type { HarnessEvents } from './events.js';
import type { JsonObject } from './json.js';

/**
 * One observed moment of a run, tagged with the run it belongs to (G-282 A).
 *
 * The shape is an AGREED CONTRACT with Fancy's agent chat, versioned as
 * `contractVersion` so a consumer refuses a version it does not know rather
 * than mis-rendering it. Renaming a field, or adding one, is a contract change
 * and bumps the version; it is not a refactor.
 *
 * A step IS its wire shape: a plain object whose keys are in contract order —
 * the envelope first, then the type's fields — so `JSON.stringify(step)` is
 * byte-identical in field names, order and value types to the reference's
 * `RunStep::toArray()`. Steps are frozen; a listener cannot alter what the
 * stream's own reader receives.
 */
export const RUN_STEP_CONTRACT_VERSION = 'g282.1';

export type RunStepType =
  | 'thinking'
  | 'text'
  | 'tool_call'
  | 'tool_result'
  | 'approval_request'
  | 'error'
  | 'step_start'
  | 'step_finish'
  | 'decision_recorded';

export interface RunStepEnvelope {
  readonly contractVersion: typeof RUN_STEP_CONTRACT_VERSION;
  readonly runId: string;
  readonly parentRunId: string | null;
  readonly rootRunId: string;
  readonly turnId: string | null;
  /** From 1 per run, no gaps; the order to render in. */
  readonly seq: number;
  /** ISO 8601 UTC with microseconds, when Harness observed it. */
  readonly at: string;
  readonly type: RunStepType;
}

export interface ThinkingStep extends RunStepEnvelope { readonly type: 'thinking'; readonly delta: string }
export interface TextStep extends RunStepEnvelope { readonly type: 'text'; readonly delta: string }
export interface ToolCallStep extends RunStepEnvelope {
  readonly type: 'tool_call';
  readonly toolCallId: string;
  readonly name: string;
  readonly args: Readonly<JsonObject>;
  readonly startedAt: string;
}

/** Stable codes, for an app that decides what to show an end user. */
export type ToolResultErrorCode = 'tool_failed' | 'tool_call_refused' | 'approval_denied' | 'tool_not_available';

interface ToolResultTiming {
  /** True for an expected refusal; a tool that threw is `ok: false, refused: false`. */
  readonly refused: boolean;
  /** Measured where the tool ran, with a monotonic clock. Whole milliseconds. */
  readonly durationMs: number;
  /** False when `durationMs` is an upper bound: siblings in the same step, or a call run on resume. */
  readonly durationExact: boolean;
}

export interface ToolResultOkStep extends RunStepEnvelope, ToolResultTiming {
  readonly type: 'tool_result';
  readonly toolCallId: string;
  readonly ok: true;
  readonly result: string;
}

export interface ToolResultErrorStep extends RunStepEnvelope, ToolResultTiming {
  readonly type: 'tool_result';
  readonly toolCallId: string;
  readonly ok: false;
  /** `message` is what the model was given for this result, and no more. */
  readonly error: { readonly code: ToolResultErrorCode; readonly message: string };
}

export interface ApprovalRequestStep extends RunStepEnvelope {
  readonly type: 'approval_request';
  readonly approvalId: string;
  readonly toolCallId: string;
}

export interface ErrorStep extends RunStepEnvelope {
  readonly type: 'error';
  /** The message is a FIXED generic text; the provider's own message is never passed on. */
  readonly error: { readonly code: string; readonly message: string };
}

export interface StepStartStep extends RunStepEnvelope { readonly type: 'step_start' }
export interface StepFinishStep extends RunStepEnvelope { readonly type: 'step_finish'; readonly text: string }

export interface DecisionRecordedStep extends RunStepEnvelope {
  readonly type: 'decision_recorded';
  readonly approvalId: string;
  readonly verdict: 'approved' | 'denied';
  readonly reason: string | null;
  /** From the application's RunContext, never from the model or the request. */
  readonly actor: string | null;
}

export type RunStep =
  | ThinkingStep
  | TextStep
  | ToolCallStep
  | ToolResultOkStep
  | ToolResultErrorStep
  | ApprovalRequestStep
  | ErrorStep
  | StepStartStep
  | StepFinishStep
  | DecisionRecordedStep;

/** The fixed text an `error` step carries. The same rule as a failed run's events. */
export const RUN_STEP_ERROR_MESSAGE = 'The model stream reported an error.';

/** How a tool call ended, as the runtime knows it at the call site. */
export type ToolCallOutcome =
  | { ok: true }
  | { ok: false; code: ToolResultErrorCode; refused: boolean };

/**
 * Turns what a run observes into tagged steps.
 *
 * One per run: it owns the run's `seq`, so a step is numbered by the run it
 * belongs to and never by whoever happens to be listening.
 *
 * @internal Built by AgentRuntime; the shape it produces is the contract, not this class.
 */
export class RunStepTagger {
  #runId: string | null = null;
  #parentRunId: string | null = null;
  #rootRunId: string | null = null;
  #sessionKey = '';
  #seq: number;
  #lastObservedAt = performance.now();
  /** performance.now() at which each tool call was observed. */
  readonly #calls = new Map<string, number>();
  /** Tool calls observed in the current model step. */
  #stepCalls = 0;

  constructor(
    private readonly turnId: string | null,
    private readonly sink: ((step: RunStep) => void) | null,
    private readonly events: HarnessEvents | null,
    seq = 0,
  ) {
    this.#seq = seq;
  }

  get seq(): number {
    return this.#seq;
  }

  bind(sessionKey: string, runId: string, parentRunId: string | null, rootRunId: string | null): void {
    this.#sessionKey = sessionKey;
    this.#runId = runId;
    this.#parentRunId = parentRunId;
    this.#rootRunId = rootRunId ?? runId;
    this.#lastObservedAt = performance.now();
  }

  decision(approvalId: string, approved: boolean, reason: string | null, actor: string | null): void {
    this.#emit('decision_recorded', {
      approvalId,
      verdict: approved ? 'approved' : 'denied',
      reason,
      actor,
    });
  }

  stepStart(): void {
    this.#stepCalls = 0;
    this.#emit('step_start', {});
  }

  thinking(delta: string): void {
    this.#emit('thinking', { delta });
  }

  text(delta: string): void {
    this.#emit('text', { delta });
  }

  toolCall(toolCallId: string, name: string, args: JsonObject): void {
    this.#calls.set(toolCallId, performance.now());
    this.#stepCalls += 1;
    this.#emit('tool_call', {
      toolCallId,
      name,
      // A copy: the step is frozen and handed to listeners, and the original
      // is the object the tool is about to be called with.
      args: structuredClone(args),
      startedAt: timestamp(),
    });
  }

  toolResult(toolCallId: string, result: string, outcome: ToolCallOutcome): void {
    const now = performance.now();
    // Measured from the moment Harness observed the call. A result whose call
    // was observed in an EARLIER run (an approved call executed on resume) is
    // measured from the last thing this run observed, the closest bound this
    // process has to when it started.
    const started = this.#calls.get(toolCallId);
    const observed = started !== undefined;
    this.#calls.delete(toolCallId);
    const durationMs = Math.max(0, Math.floor(now - (started ?? this.#lastObservedAt)));
    // EXACT only when the call was observed here and ran alone in its step.
    // With siblings, a call's clock starts when it was observed, before the
    // siblings ahead of it ran; on resume the call was observed elsewhere.
    const durationExact = observed && this.#stepCalls <= 1;

    this.#emit('tool_result', {
      toolCallId,
      ok: outcome.ok,
      ...(outcome.ok ? { result } : { error: { code: outcome.code, message: result } }),
      refused: outcome.ok ? false : outcome.refused,
      durationMs,
      durationExact,
    }, now);
  }

  approvalRequest(approvalId: string, toolCallId: string): void {
    this.#emit('approval_request', { approvalId, toolCallId });
  }

  /**
   * The CODE travels, the provider's message does not: it can carry a request
   * URL or a key embedded in one, and a step can be broadcast to listeners that
   * put it on a screen.
   */
  error(code: string): void {
    this.#emit('error', { error: { code, message: RUN_STEP_ERROR_MESSAGE } });
  }

  stepFinish(text: string): void {
    this.#emit('step_finish', { text });
  }

  #emit(type: RunStepType, data: Record<string, unknown>, observedAt = performance.now()): void {
    const runId = this.#runId;

    if (runId === null) throw new Error('A run step was emitted before its run was bound.');

    this.#seq += 1;
    const step = deepFreeze({
      contractVersion: RUN_STEP_CONTRACT_VERSION,
      runId,
      parentRunId: this.#parentRunId,
      rootRunId: this.#rootRunId ?? runId,
      turnId: this.turnId,
      seq: this.#seq,
      at: timestamp(),
      type,
      ...data,
    }) as unknown as RunStep;
    this.#lastObservedAt = observedAt;

    this.sink?.(step);

    // Opt-in, decided by the runtime: null unless `stepEvents` is true. A step
    // carries tool arguments, tool results and model text, and an app that
    // records or broadcasts every event would otherwise capture them the
    // moment it upgraded. HarnessEvents isolates a listener that throws.
    this.events?.emit({
      type: 'run.stepped',
      runId,
      sessionKey: this.#sessionKey,
      parentRunId: step.parentRunId,
      rootRunId: step.rootRunId,
      step,
    });
  }
}

/**
 * Steps from a run in flight to the reader of its stream.
 *
 * PUSH, not pull: the run never waits for the reader. A pull-based stream
 * would leave a run — and the session lock it holds while it resolves
 * approvals — suspended for as long as a disconnected reader takes to be
 * noticed; this way a run always reaches its end and closes itself.
 *
 * @internal
 */
export class StepChannel {
  readonly #queue: RunStep[] = [];
  #wake: (() => void) | null = null;

  push(step: RunStep): void {
    this.#queue.push(step);
    this.#signal();
  }

  /**
   * Start the work, yield every step it pushes in order, then return its
   * result or rethrow its failure — after the last step, never instead of it.
   */
  async *drain<T>(start: () => Promise<T>): AsyncGenerator<RunStep, T, undefined> {
    const state: { settled: { ok: true; value: T } | { ok: false; error: unknown } | null } = { settled: null };

    // Settled into state rather than awaited: an abandoned stream must not
    // leave an unhandled rejection behind it.
    start().then(
      (value) => { state.settled = { ok: true, value }; this.#signal(); },
      (error: unknown) => { state.settled = { ok: false, error }; this.#signal(); },
    );

    for (;;) {
      while (this.#queue.length > 0) yield this.#queue.shift()!;

      const settled = state.settled;

      if (settled !== null) {
        if (!settled.ok) throw settled.error;

        return settled.value;
      }

      await new Promise<void>((resolve) => { this.#wake = resolve; });
    }
  }

  #signal(): void {
    const wake = this.#wake;
    this.#wake = null;
    wake?.();
  }
}

/**
 * ISO 8601 UTC with microseconds, the reference's `Y-m-d\TH:i:s.u\Z`.
 *
 * JavaScript's wall clock stops at milliseconds, so the last three digits come
 * from the monotonic clock's sub-millisecond part. The value is within a
 * millisecond of the truth and in the agreed format; order by `seq`, never `at`.
 */
export function timestamp(): string {
  const now = Date.now();
  const micros = Math.floor((performance.now() * 1000) % 1000);
  const iso = new Date(now).toISOString();

  return `${iso.slice(0, 23)}${String(micros).padStart(3, '0')}Z`;
}

function deepFreeze<T>(value: T): T {
  if (value !== null && typeof value === 'object' && !Object.isFrozen(value)) {
    Object.freeze(value);
    for (const child of Object.values(value as Record<string, unknown>)) deepFreeze(child);
  }

  return value;
}
