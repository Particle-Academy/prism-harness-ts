import { createHash, randomUUID } from 'node:crypto';
import { validateAgent } from './address.js';
import type { JsonObject, JsonValue } from './json.js';
import {
  assistantRow,
  approvalToolScope, approvalRunSteps,
  threadView,
  toolResultEntry,
  toolResultRow,
  type ApprovalDecisionEntry,
  type ApprovalRequestEntry,
  type AssistantRow,
  type ToolResultEntry,
  type ToolResultRow,
  unusableStoredCalls,
  ambiguousCalls,
  repeatsCallIds,
  REPEATED_CALL_ID_RESULT,
  replacedStepCalls,
  supersededAt,
  resultOwners,
  approvalSuperseded,
  REPLACED_STEP_KEY,
  REPLACED_STEP_RESULT,
  ARGUMENTS_AS_SENT,
} from './thread-rows.js';
import { usableArguments } from './tool-arguments.js';
import { admitAttachments } from './attachments.js';
import { HarnessError } from './errors.js';
import type { HarnessEvents } from './events.js';
import type { AgentMode, ModeRegistry } from './modes.js';
import type { Session } from './session.js';
import { schemaName, schemaProblems } from './structured.js';
import { RunBudget, RunContext } from './subagents.js';
import type { HarnessTool, ToolAuthorizer, ToolRegistry } from './tools.js';
import { settleWithRefusalScope } from './refusals.js';
import { RunStepTagger, StepChannel, type RunStep, type ToolCallOutcome } from './steps.js';

/**
 * What the runtime needs from a model, and NOTHING MORE.
 *
 * An INTERFACE rather than a dependency on `prism-ts`. The loop below — steps,
 * budgets, approvals, thread recording, events — is the part worth porting, and
 * none of it needs to know how a request reaches a provider. Keeping the seam
 * here also means this package stays at zero dependencies and a consumer can
 * drive it with `prism-ts`, with their own client, or with a fake in a test.
 *
 * The reference couples these because Prism is already a dependency there.
 */
export interface LlmRequest {
  /** Optional ceiling the model client must pass to its provider request. */
  maxOutputTokens?: number;
  systemPrompt: string;
  /** The conversation so far, serialized — oldest first. */
  messages: readonly JsonObject[];
  tools: readonly HarnessTool[];
  provider: string;
  model: string;
  /**
   * The mode's `provider_options`, unchanged. A client passes them to its
   * provider call (for prism-ts, `withProviderOptions()`); the harness does not
   * interpret them.
   */
  providerOptions: Readonly<JsonObject>;
  /**
   * The JSON Schema a structured turn asks the answer to satisfy.
   *
   * Absent on an ordinary turn. A client passes it to its provider's structured
   * mode (for prism-ts, `Prism.structured().withSchema()`) and returns what it
   * parsed as `structured`; the harness checks that against this same schema
   * before the caller sees it.
   */
  schema?: Readonly<JsonObject>;
}

export interface LlmToolCall {
  id: string;
  name: string;
  arguments: JsonObject;
  /**
   * The ids a provider keys the call's result and reasoning by, when they differ
   * from `id`. OpenAI's Responses API answers a `function_call` by its `call_id`,
   * and replays the reasoning item it came from by id. Recorded on the call, as
   * prism's ToolCall stores them, so the next request can send them back.
   */
  resultId?: string | null;
  reasoningId?: string | null;
  reasoningSummary?: readonly JsonValue[] | null;
}

export interface LlmResponse {
  text: string;
  toolCalls?: readonly LlmToolCall[];
  /** `stop`, `tool_calls`, `length`, … — the provider's own reason, passed through. */
  finishReason: string;
  /** Null when the provider does not report one. NOT zero — see `RunLedger.recordCost`. */
  costUsd?: number | null;
  /**
   * What the provider needs sent back with this turn on the next request, such
   * as Anthropic's `thinking` and `thinking_signature`. Recorded with the
   * assistant turn as `additional_content`, the key prism's AssistantMessage
   * uses, so a client built on prism-ts can pass `response.additionalContent`
   * straight through and read it back from `messages`.
   */
  additionalContent?: Readonly<JsonObject>;
  /**
   * The document a structured turn parsed out of `text`.
   *
   * Null when the text held none — an apology in prose, a truncated answer, a
   * fence that never closed. The harness tells that case apart from a document
   * with the wrong shape, because the caller's next move differs.
   */
  structured?: JsonValue | null;
}

export type LlmClient = (request: LlmRequest) => Promise<LlmResponse>;

export interface AgentResponse {
  runId: string;
  text: string;
  steps: number;
  /** NAMES only, in call order. */
  toolCalls: readonly string[];
  finishReason: string;
  /** Set when the run stopped because a tool needs a human. */
  pendingApprovals: readonly PendingApproval[];
  /** Set when the run stopped because the tree ran out of budget, or was cancelled. */
  stoppedBecause: string | null;
}

/**
 * One structured run's result: the document, and the text it was read from.
 *
 * BOTH, not one. `structured` is what the caller asked for, and `text` is what
 * the model actually sent — which is also exactly what the thread stored, so a
 * transcript and a parse can be compared rather than trusted.
 */
export interface StructuredAgentResponse extends AgentResponse {
  /** Checked against the schema before it got here; a run that could not produce one threw. */
  structured: JsonValue;
}

export interface PendingApproval {
  /** The APPROVAL id: what `recordApproval()` answers. Not the tool call id. */
  id: string;
  toolCallId: string;
  tool: string;
  arguments: JsonObject;
}

export interface AgentRuntimeOptions {
  /** Explicit true retains strict behavior. Explicit false now requires migration to the widening opt-out. */
  narrowRequestedTools?: boolean;
  /** Permit caller names to replace a LEGACY session's mode list. Named agents remain strict; authorizers still apply. */
  allowLegacyToolListWidening?: boolean;
  config?: RuntimeConfig;
  agents?: Record<string, RuntimeConfig>;
  client: LlmClient;
  modes: ModeRegistry;
  tools: ToolRegistry;
  authorizer?: ToolAuthorizer;
  events?: HarnessEvents;
  /**
   * Emit every run step as a `run.stepped` event. OFF BY DEFAULT: a step
   * carries tool arguments, tool results and model text, and an app that
   * records or forwards every event would otherwise capture them the moment it
   * upgraded. `streamSteps()` and `streamDecision()` yield steps either way.
   */
  stepEvents?: boolean;
}

/** One human decision on a pending approval. `approvalId` is `PendingApproval.id`. */
export interface ApprovalDecision {
  approvalId: string;
  approved: boolean;
  reason?: string | null;
}

/** What the step paths thread through a turn. */
interface Stepping {
  tagger: RunStepTagger;
  /** The id to run under, when the step path has already claimed one. */
  runId?: string;
  /** True when that id is a parked run's own, continued. */
  resuming?: boolean;
  /** True when the caller already holds the session lock for the whole turn. */
  locked?: boolean;
}

/** A tool call's result as recorded, and how it ended. */
interface Invoked {
  entry: ToolResultEntry;
  outcome: ToolCallOutcome;
}

const REFUSED_CALL: ToolCallOutcome = { ok: false, code: 'tool_call_refused', refused: true };

/** sha256 of the RFC 8785 canonical JSON of no arguments, `{}`. */
const NO_ARGUMENTS_DIGEST = `sha256:${createHash('sha256').update('{}', 'utf8').digest('hex')}`;

function approvalExpired(tool: string): string {
  return `Not run: the approval for ${tool} was recorded by an earlier version and has expired. `
    + 'Call the tool again if it is still needed, and the person will be asked to approve it again.';
}

export interface RuntimeConfig {
  provider?: string | null;
  model?: string | null;
  max_steps_ceiling?: number | null;
  max_output_tokens_per_call?: number | null;
  lock_ttl?: number;
  lock_wait?: number;
}

export interface EffectiveTool {
  name: string;
  owner: string;
  source: string;
}

/**
 * The loop: prompt in, turns out, everything recorded.
 *
 * Three properties matter more than the mechanics.
 *
 * **Every step is checked against the budget BEFORE it is taken**, not after.
 * Checking afterwards means the step that broke the limit has already been
 * paid for, which makes a budget a report rather than a control.
 *
 * **An approval stops the run and is written to the THREAD**, not held in
 * memory. That is what makes it survive: the approval a person grants this
 * morning is a durable row, so the worker that resumes tonight — a different
 * process, possibly after a deploy — reads the same answer.
 *
 * **A tool that needs approval and has none DOES NOT RUN.** Failing closed is
 * the only safe direction: an unanswered approval that executed anyway is
 * exactly the outcome the whole mechanism exists to prevent.
 */
/** How long resolving approvals may hold the session lock: long enough for a tool to run. */
const RESOLUTION_LOCK_SECONDS = 300;

/** How long a second worker waits for that lock before giving up without running anything. */
const RESOLUTION_WAIT_SECONDS = 30;

export class AgentRuntime {
  readonly #client: LlmClient;

  readonly #modes: ModeRegistry;

  readonly #tools: ToolRegistry;

  readonly #authorizer?: ToolAuthorizer;

  readonly #events?: HarnessEvents;
  readonly #config: RuntimeConfig;
  readonly #agents: Record<string, RuntimeConfig>;
  readonly #narrowRequestedTools: boolean;
  readonly #stepEvents: boolean;

  constructor(options: AgentRuntimeOptions) {
    if (options.stepEvents !== undefined && typeof options.stepEvents !== 'boolean') throw new TypeError('stepEvents must be a boolean');
    this.#stepEvents = options.stepEvents === true;
    if (options.narrowRequestedTools !== undefined && typeof options.narrowRequestedTools !== 'boolean') throw new TypeError('narrowRequestedTools must be a boolean');
    if (options.allowLegacyToolListWidening !== undefined && typeof options.allowLegacyToolListWidening !== 'boolean') throw new TypeError('allowLegacyToolListWidening must be a boolean');
    if (options.narrowRequestedTools === false) throw new TypeError('narrowRequestedTools: false is no longer supported. Remove it; use allowLegacyToolListWidening: true only to explicitly permit legacy tool-list widening.');
    if (options.narrowRequestedTools === true && options.allowLegacyToolListWidening === true) throw new TypeError('narrowRequestedTools: true conflicts with allowLegacyToolListWidening: true.');
    this.#narrowRequestedTools = options.allowLegacyToolListWidening !== true;
    this.#client = options.client;
    this.#modes = options.modes;
    this.#tools = options.tools;
    this.#authorizer = options.authorizer;
    this.#events = options.events;
    this.#config = { ...options.config };
    this.#agents = Object.fromEntries(Object.entries(options.agents ?? {}).map(([agent, config]) => {
      validateAgent(agent); validateRuntimeConfig(config); return [agent, { ...config }];
    }));
    validateRuntimeConfig(this.#config);
  }

  #setting<K extends keyof RuntimeConfig>(session: Session, key: K): RuntimeConfig[K] {
    const options = session.agent === null ? undefined : this.#agents[session.agent];
    return options && Object.hasOwn(options, key) ? options[key] : this.#config[key];
  }

  #boundedRun(session: Session, run: RunContext): RunContext {
    const steps = this.#setting(session, 'max_steps_ceiling'), output = this.#setting(session, 'max_output_tokens_per_call'), budget = run.budget;
    if (steps == null && output == null) return run;
    return run.withBudget(new RunBudget(
      steps == null ? budget.maxSteps : Math.min(budget.maxSteps, run.ledger.steps + steps),
      budget.maxCostUsd, budget.maxSeconds,
      output == null ? budget.maxOutputTokensPerCall : Math.min(budget.maxOutputTokensPerCall ?? output, output),
    ));
  }

  /** Inspect the offer without a run/handler or logical mutation; reads may purge expired entries. */
  async effectiveTools(session: Session, modeName?: string): Promise<EffectiveTool[]> {
    const mode = this.#modes.forAgent(session.agent).resolve(modeName ?? await session.mode());
    return (await this.#offer(session, mode)).descriptors;
  }

  async #offer(session: Session, mode: AgentMode, requested?: readonly string[]): Promise<{ tools: HarnessTool[]; descriptors: EffectiveTool[] }> {
    return session.resolveReadOnly(mode.name, async view => {
      const names = session.agent !== null || this.#narrowRequestedTools ? narrowedTools(mode.tools, requested) : (requested ?? mode.tools);
      const detailed = await this.#tools.resolveDetailed(names, view);
      const tools = new Map([...detailed].map(([name, entry]) => [name, entry.tool]));
      if (view.agent !== null && tools.has('complete_task')) {
        const authorizer = this.#authorizer?.forAgent(view.agent);
        if (!authorizer?.enabled || !authorizer.hasCallPolicy()) throw new HarnessError('unsafe_authorization_configuration', 'Named task completion requires enabled authorization and an explicit call policy.');
      }
      const offered = this.#authorizer ? await this.#authorizer.allowed(view, tools) : [...tools.values()];
      const ownership = new Map([...detailed.values()].map(entry => [entry.tool.name, entry]));
      const descriptors = offered.map(tool => ({ name: tool.name, owner: ownership.get(tool.name)!.owner, source: ownership.get(tool.name)!.source }));
      return { tools: offered, descriptors };
    });
  }

  /**
   * Run a turn.
   *
   * An EMPTY prompt is meaningful and not an error: it is how a run resumes
   * after an approval, because the conversation already contains the request,
   * the decision, and everything before them. A new prompt there would be a
   * second instruction competing with the one the tool call came from.
   */
  async send(
    session: Session,
    prompt: string,
    toolNames?: readonly string[],
    context?: RunContext,
    additionalContent: readonly unknown[] = [],
  ): Promise<AgentResponse> {
    return await this.#turn(session, prompt, toolNames, context, additionalContent, undefined, this.#observerStepping(context));
  }

  /**
   * The same run as `send()`, delivered as tagged run steps (G-282 A).
   *
   * Same arguments, same run: the same mode, tools, budget, approvals, thread
   * rows and run events. What the caller receives is the run's steps as they
   * happen — `step_start`, `thinking`, `text`, `tool_call`, `tool_result`,
   * `approval_request`, `error`, `step_finish` — each tagged with the run it
   * belongs to and numbered from 1 with no gaps. The generator's RETURN value
   * is the run's `AgentResponse`, for a caller that wants `pendingApprovals`.
   *
   * Lazy: nothing runs until the first step is read. Once started, the run
   * goes to its end whether or not anyone keeps reading — abandoning the loop
   * does not cancel it (that is G-285), and it still closes the run and
   * releases its lock. A run that throws rethrows here after every step it
   * produced has been yielded.
   *
   * HOLDS THE SESSION LOCK FOR THE WHOLE TURN, taken before anything else —
   * before a run exists and before any provider call — and released when the
   * turn ends on every path. A stream started while another holder has the
   * lock waits `lock_wait` seconds and then fails with `session_locked`,
   * having called nothing and recorded nothing.
   */
  async *streamSteps(
    session: Session,
    prompt: string,
    toolNames?: readonly string[],
    context?: RunContext,
    additionalContent: readonly unknown[] = [],
  ): AsyncGenerator<RunStep, AgentResponse, undefined> {
    const channel = new StepChannel();
    const tagger = this.#tagger(context, channel, 0);

    return yield* channel.drain(() => this.#holdingTurnLock(session, (live) =>
      this.#turn(live, prompt, toolNames, context, additionalContent, undefined, { tagger, locked: true })));
  }

  /**
   * Answer pending approvals and continue the run AS IT HAPPENS (G-284).
   *
   * The list is checked when this is called, so a malformed one is refused
   * before anything is recorded. Everything else happens when the stream
   * starts, UNDER THE SESSION LOCK held for the whole turn: the decisions are
   * written to the thread by the same code as `recordApproval()`, before any
   * tool runs or any provider is called; one `decision_recorded` step is
   * yielded per decision; then the run continues as `streamSteps()` does. A
   * stream dropped after it started has still recorded every decision, and
   * resuming again executes nothing twice. A stream that cannot take the lock
   * fails with `session_locked` and records nothing.
   *
   * When the session still knows the run that parked, the continuation runs
   * under that run's id, its `seq` continues where that run's stream ended,
   * and it emits `run.resumed` rather than a second `run.started`. The `actor`
   * on a decision step is the RunContext's — the application's own statement
   * of who decided — never anything from the model or the request.
   *
   * Authorize before calling: who may answer an approval is the application's
   * decision, as for `recordApproval()`.
   */
  streamDecision(
    session: Session,
    decisions: readonly ApprovalDecision[],
    context?: RunContext,
  ): AsyncGenerator<RunStep, AgentResponse, undefined> {
    return this.#continueSteps(session, assertDecisions(decisions), context);
  }

  async *#continueSteps(
    session: Session,
    decisions: readonly ApprovalDecision[],
    context: RunContext | undefined,
  ): AsyncGenerator<RunStep, AgentResponse, undefined> {
    const channel = new StepChannel();

    return yield* channel.drain(() => this.#holdingTurnLock(session, async (live) => {
      // Read inside the lock, so the run it names cannot change underneath.
      const parked = await this.#parkedRun(live);

      // DURABLE FIRST, through the one writer every decision path uses, and
      // before any tool or provider call.
      await writeDecisions(live, decisions);

      const tagger = this.#tagger(context, channel, parked?.lastSeq ?? 0);
      const execution = executionContext(context);
      const runId = parked?.id ?? randomUUID();
      tagger.bind(live.key(), runId, execution?.parentRunId ?? null, execution?.rootRunId ?? runId);

      for (const decision of decisions) {
        // From the APP's run context, never from the model or the request.
        tagger.decision(decision.approvalId, decision.approved, decision.reason ?? null, context?.actor ?? null);
      }

      // An EMPTY prompt: the thread already holds the request, the decisions
      // and everything before them, and the turn resolves approvals from it
      // before any model call.
      return await this.#turn(live, '', undefined, context, [], undefined, { tagger, runId, resuming: parked !== null, locked: true });
    }));
  }

  /**
   * One streamed turn under the session lock, released in `finally` by the
   * store on every path — including a turn that throws, and a stream whose
   * reader stopped early (the turn still runs to its end, then releases).
   */
  async #holdingTurnLock<T>(session: Session, turn: (live: Session) => Promise<T>): Promise<T> {
    return await session.lock(
      turn,
      this.#setting(session, 'lock_ttl') ?? RESOLUTION_LOCK_SECONDS,
      this.#setting(session, 'lock_wait') ?? RESOLUTION_WAIT_SECONDS,
    );
  }

  /**
   * The run that parked on the pending approvals, when the session still knows
   * it: the current run, if it recorded the last tool-calling turn and that
   * turn is waiting on approvals. A parked run in this port stays `running`
   * (send() returns `awaiting_approval` without closing it), so status alone
   * cannot say; the thread row's own run id can.
   */
  async #parkedRun(session: Session): Promise<{ id: string; lastSeq: number } | null> {
    const run = await session.run();

    if (run === null || run.status === 'failed') return null;

    const entries = await session.thread().messages();

    for (let index = entries.length - 1; index >= 0; index -= 1) {
      const entry = entries[index]!;
      const message = entry.message;

      if (message.type !== 'assistant' || !Array.isArray(message.tool_calls) || message.tool_calls.length === 0) continue;

      const requests = Array.isArray(message.tool_approval_requests) ? message.tool_approval_requests : [];

      if (requests.length === 0 || entry.runId !== run.id) return null;

      const seq = run.last_seq;

      return { id: run.id, lastSeq: typeof seq === 'number' && Number.isSafeInteger(seq) && seq >= 0 ? seq : 0 };
    }

    return null;
  }

  #tagger(context: RunContext | undefined, channel: StepChannel | null, seq: number): RunStepTagger {
    return new RunStepTagger(
      context?.turnId ?? null,
      channel === null ? null : (step) => channel.push(step),
      this.#stepEvents ? (this.#events ?? null) : null,
      seq,
    );
  }

  /** send()'s steps exist only to be emitted, so only when that was opted into. */
  #observerStepping(context: RunContext | undefined): Stepping | undefined {
    return this.#stepEvents && this.#events !== undefined ? { tagger: this.#tagger(context, null, 0) } : undefined;
  }

  /**
   * A turn whose answer is a document.
   *
   * The same run as `send()` — same mode, same tools, same budget, same events,
   * same approvals — with a schema the answer has to satisfy. The schema
   * travels on the request for the client to hand its provider, and what comes
   * back is checked against it here before the caller sees it.
   *
   * WHAT THE THREAD KEEPS IS THE TEXT, with the parsed document beside it as
   * `structured` in the assistant row's `additional_content`. Never instead of
   * it: a later turn replays this conversation as messages, and a transcript
   * that reads differently because of the SHAPE of the request that produced it
   * is a difference nothing reports.
   *
   * A FAILED DOCUMENT IS STILL RECORDED, and then thrown. The exchange
   * happened, and a thread that omits the answer it did not like cannot explain
   * the retry sitting next to it. The run is marked failed and `run.failed` is
   * emitted, as for any other failure.
   *
   * @throws HarnessError `structured_unreadable` or `structured_schema_violation`
   */
  async sendStructured(
    session: Session,
    prompt: string,
    schema: Readonly<JsonObject>,
    toolNames?: readonly string[],
    context?: RunContext,
    additionalContent: readonly unknown[] = [],
  ): Promise<StructuredAgentResponse> {
    const response = await this.#turn(session, prompt, toolNames, context, additionalContent, schema, this.#observerStepping(context));

    // Narrowed by #turn: with a schema it always returns the structured shape.
    return response as StructuredAgentResponse;
  }

  async #turn(
    session: Session,
    prompt: string,
    toolNames?: readonly string[],
    context?: RunContext,
    additionalContent: readonly unknown[] = [],
    schema?: Readonly<JsonObject>,
    stepping?: Stepping,
  ): Promise<AgentResponse> {
    // Refused before a run exists: a bad attachment is a mistake in the call,
    // and it should not cost a run, events or budget.
    const attachments = admitAttachments(prompt, additionalContent);
    const mode = this.#modes.forAgent(session.agent).resolve(await session.mode());
    const provider = (await session.provider()) ?? this.#setting(session, 'provider') ?? 'unknown';
    const model = (await session.model()) ?? this.#setting(session, 'model') ?? 'unknown';
    const runId = stepping?.runId ?? randomUUID();
    // A turnId/actor-only context is METADATA: the run takes exactly the path
    // and budget it would take with no context at all.
    const execution = executionContext(context);
    const run = this.#boundedRun(session, execution ?? RunContext.root(runId, new RunBudget(mode.maxSteps)));
    const thread = session.thread();
    const steps = stepping?.tagger;

    if (run.tooDeep()) {
      throw HarnessError.runNotPermitted(
        `This run is nested ${run.depth} deep, at or past the ceiling. Nobody debugs a tree that deep, ` +
          'and a configuration that produced one is a mistake worth reporting rather than executing.',
      );
    }

    steps?.bind(session.key(), runId, run.parentRunId, run.rootRunId);
    await session.beginRun(runId, mode.name, provider, model);
    this.#events?.emit({
      // A parked run continued under its own id has already STARTED once; a
      // second run.started for it would read as a new run.
      type: stepping?.resuming === true ? 'run.resumed' : 'run.started',
      runId,
      sessionKey: session.key(),
      mode: mode.name,
      provider,
      model,
      rootRunId: run.rootRunId,
      depth: run.depth,
      at: new Date().toISOString(),
    });

    try {
      let offered: HarnessTool[];
      if (session.agent === null && !this.#narrowRequestedTools) {
        const resolved = await this.#tools.resolve(toolNames ?? mode.tools, session);
        offered = this.#authorizer ? await this.#authorizer.allowed(session, resolved) : [...resolved.values()];
      } else offered = (await this.#offer(session, mode, toolNames)).tools;
      const stored = (await thread.messages()).map(entry => entry.message);
      const scope = approvalToolScope(stored);
      // A resume continues the paused run's step count; it never opens a new
      // allowance. Only for a root run: a child's steps are already on the
      // tree ledger its caller hands back in.
      if (prompt === '' && execution == null) run.ledger.recordSteps(approvalRunSteps(stored) ?? 0);
      if (scope !== null) {
        if (scope.legacy) console.warn('Pending approval has no valid stored tool scope; offering approved requests only.');
        offered = offered.filter(tool => scope.names.includes(tool.name)
          && (mode.tools.includes('*') || mode.tools.includes(tool.name)));
      }
      const called: string[] = [];

      // Decisions recorded since the run stopped are acted on FIRST, before a
      // new prompt is recorded, so the results land after the calls they answer
      // rather than after the new turn.
      offered = await this.#resolveApprovals(session, mode, offered, runId, called, scope?.legacy === true, steps, stepping?.locked === true);

      if (prompt !== '') {
        // With attachments, the shape prism-ts's UserMessage.toObject() writes:
        // the media parts, then the turn's own text as a trailing text part,
        // which fromObject() strips back off. Without them, unchanged.
        const turn: JsonObject =
          attachments.length === 0
            ? { type: 'user', content: prompt }
            : { type: 'user', content: prompt, additional_content: [...attachments, { text: prompt }], additional_attributes: {} };

        await thread.record([turn], runId);
      }

      return await this.#loop(session, mode, run, runId, provider, model, offered, called, schema, steps);
    } catch (error) {
      // THE MODEL'S OWN WORDS DO NOT BELONG IN A RUN ROW OR AN EVENT. A schema
      // violation names the values that missed, so its message carries pieces
      // of the document — and an event carrying those would put model output in
      // every listener's telemetry, which is the same reason tool arguments are
      // names-only here. The document is already in the thread, in full, where
      // it is read deliberately rather than shipped by default. An error that
      // holds one records its CODE.
      const failure =
        error instanceof HarnessError && error.document !== undefined
          ? error.code
          : error instanceof Error
            ? error.message
            : String(error);

      await session.failRun(runId, failure);
      this.#events?.emit({
        type: 'run.failed',
        runId,
        sessionKey: session.key(),
        failure,
        steps: run.ledger.steps,
        at: new Date().toISOString(),
      });

      throw error;
    } finally {
      if (steps !== undefined) await rememberSeq(session, runId, steps);
    }
  }

  async #loop(
    session: Session,
    mode: AgentMode,
    run: RunContext,
    runId: string,
    provider: string,
    model: string,
    offered: readonly HarnessTool[],
    called: string[],
    schema?: Readonly<JsonObject>,
    steps?: RunStepTagger,
  ): Promise<AgentResponse> {
    const thread = session.thread();
    let text = '';
    let finishReason = 'stop';

    for (;;) {
      // BEFORE the step, never after. Checking afterwards means the step that
      // broke the limit has already been paid for.
      const exhausted = run.ledger.exhaustion(run.budget);

      if (exhausted !== null) {
        return await this.#finish(session, runId, called, 'budget_exhausted', text, run, exhausted);
      }

      steps?.stepStart();
      const response = await this.#client({
        systemPrompt: mode.systemPrompt,
        messages: threadView((await thread.messages()).map((entry) => entry.message)),
        tools: offered,
        provider,
        model,
        providerOptions: mode.providerOptions,
        ...(run.budget.maxOutputTokensPerCall === null ? {} : { maxOutputTokens: run.budget.maxOutputTokensPerCall }),
        ...(schema === undefined ? {} : { schema }),
      });

      run.ledger.recordSteps(1);
      run.ledger.recordCost(response.costUsd ?? null);
      text = response.text;
      finishReason = response.finishReason;

      // A client can hand back arguments it could not parse. Such a call is
      // recorded with none, asks nobody and runs nothing: the model is told.
      const unusable = new Set<LlmToolCall>();
      const toolCalls = (response.toolCalls ?? []).map((call) => {
        const usable = usableArguments(call.arguments);
        if (usable !== null) return usable === call.arguments ? call : { ...call, arguments: usable };
        const recorded: LlmToolCall = { ...call, arguments: {} };
        unusable.add(recorded);
        return recorded;
      });
      if (steps !== undefined) {
        // Provider state names reasoning `thinking` (Anthropic, as prism-ts
        // records it); the client returns a whole response, so each is one delta.
        const thinking = response.additionalContent?.thinking;
        if (typeof thinking === 'string' && thinking !== '') steps.thinking(thinking);
        if (response.text !== '') steps.text(response.text);
      }
      const failedFinish = finishReason === 'error' || finishReason === 'unknown' ? finishReason : null;
      const gated = failedFinish === null ? toolCalls.filter((call) => !unusable.has(call) && mode.needsApproval(call.name)) : [];
      const requests: ApprovalRequestEntry[] = gated.map((call) => ({
        approval_id: `apr_${randomUUID().replaceAll('-', '')}`,
        tool_call_id: call.id,
        arguments_format: ARGUMENTS_AS_SENT,
      }));

      // The next step's request is built from this row, so it keeps what a
      // provider needs sent back: each call's arguments and provider ids, the
      // turn's provider state, and the approvals it is waiting on (G-58).
      const answering = schema !== undefined && toolCalls.length === 0;
      const metadata: JsonObject | undefined = answering
        ? { ...(response.additionalContent ?? {}), structured: response.structured ?? null }
        : (response.additionalContent as JsonObject | undefined);

      const row: JsonObject = assistantRow(response.text, toolCalls, metadata, requests);
      if (requests.length > 0) row.harness_tool_names = offered.map(tool => tool.name);
      await thread.record([row], runId);

      if (failedFinish !== null) {
        // The code travels; nothing the provider said does.
        steps?.error('model_step_failed');
        // Complete the transcript with refusals, never approval requests, so a
        // later resume cannot dispatch calls from this failed response.
        if (toolCalls.length > 0) {
          await thread.record([toolResultRow(toolCalls.map(call =>
            toolResultEntry(call, `Not run: model step failed (${failedFinish}).`)))], runId);
        }
        throw HarnessError.modelStepFailed(failedFinish);
      }

      if (toolCalls.length === 0) {
        steps?.stepFinish(response.text);
        if (schema !== undefined) {
          // Recorded first, then checked: the exchange happened either way, and
          // a failure here fails the run through the caller's catch.
          this.#assertDocument(schema, response);
        }

        const finished = await this.#finish(session, runId, called, finishReason, text, run, null);

        if (schema === undefined) return finished;

        const structured: StructuredAgentResponse = { ...finished, structured: response.structured as JsonValue };

        return structured;
      }

      // The calls that need nobody run now, as in the reference, and their
      // results are recorded even when the step then stops for a person.
      const results: ToolResultEntry[] = [];

      for (const call of toolCalls) steps?.toolCall(call.id, call.name, call.arguments);

      for (const call of toolCalls.filter((candidate) => !gated.includes(candidate))) {
        if (unusable.has(call)) {
          const entry = toolResultEntry(call, `Not run: the arguments for ${call.name} are not a JSON object.`);
          steps?.toolResult(call.id, entry.result, { ok: false, code: 'tool_failed', refused: false });
          results.push(entry);
          continue;
        }
        called.push(call.name);
        const invoked = await this.#invoke(offered, call);
        steps?.toolResult(call.id, invoked.entry.result, invoked.outcome);
        results.push(invoked.entry);
      }

      if (results.length > 0) {
        await thread.record([toolResultRow(results)], runId);
      }

      // After the results, as core yields them: the calls that ran, then the
      // ones waiting on a person.
      for (const request of requests) steps?.approvalRequest(request.approval_id, request.tool_call_id);
      steps?.stepFinish(response.text);

      if (requests.length > 0) {
        // FAILS CLOSED. The gated calls have not run, and the requests are in
        // the thread, so a different process can resume after a person answers.
        return {
          runId,
          text,
          steps: run.ledger.steps,
          toolCalls: called,
          finishReason: 'awaiting_approval',
          pendingApprovals: gated.map((call, index) => ({
            id: requests[index]!.approval_id,
            toolCallId: call.id,
            tool: call.name,
            arguments: call.arguments,
          })),
          stoppedBecause: null,
        };
      }
    }
  }

  /**
   * Act on the decisions recorded for the last turn that stopped for a person.
   *
   * The model is NOT asked again. Asked again, a provider issues the call
   * afresh under a new id, and a decision recorded against the old one never
   * matches. The calls that stopped the run are answered where they are:
   *
   * - approved: run, once;
   * - denied: the reason, as the result the model sees;
   * - no decision: refused, "No approval response provided". Record every
   *   decision before resuming.
   *
   * A call that already has a result is done and never runs again, whatever
   * its decision says. The results are recorded as one tool result row holding
   * every result and decision for the turn, as the reference writes it.
   */
  async #resolveApprovals(
    session: Session,
    mode: AgentMode,
    offered: HarnessTool[],
    runId: string,
    called: string[],
    legacyDiagnosticEmitted: boolean,
    steps?: RunStepTagger,
    alreadyLocked = false,
  ): Promise<HarnessTool[]> {
    // A read without the lock first: nearly every send() has nothing to resolve
    // and should not wait on the session lock to find that out.
    if ((await this.#unresolved(session, mode)) === null) return offered;

    // Then under the session lock, reading again inside it. Two workers resuming
    // the same session at once would otherwise both find the approved call with
    // no result, and both run it. The second now waits, finds the result the
    // first recorded, and runs nothing. If it cannot get the lock it throws
    // SessionLocked, and nothing runs. A streamed turn already holds that lock
    // for the whole turn, and the lock is not reentrant, so it resolves in place.
    const resolve = async (live: Session): Promise<HarnessTool[]> => {
      const work = await this.#unresolved(live, mode);

      if (work === null) return offered;

      // Work can change while waiting. Its calls, decisions AND scope must
      // come from the same locked snapshot, and bound the continuation too.
      if (work.scope?.legacy && !legacyDiagnosticEmitted) console.warn('Pending approval has no valid stored tool scope; offering approved requests only.');
      const bounded = offered.filter(tool => work.scope?.names.includes(tool.name) === true);

      const resolved: ToolResultEntry[] = [];

      for (const call of work.calls) {
        const input = { id: call.id, name: call.name, arguments: call.arguments, resultId: call.result_id };
        const approvalId = work.approvalIds.get(call.id);
        const decision = approvalId === undefined ? undefined : work.decisions.get(approvalId);

        if (work.repeated) {
          // The step repeats a call id, so no decision can be told apart from
          // another (G-392): none of its gated calls runs, whatever was decided.
          const entry = toolResultEntry(input, REPEATED_CALL_ID_RESULT);
          steps?.toolResult(call.id, entry.result, { ok: false, code: 'approval_denied', refused: true });
          resolved.push(entry);
        } else if (decision?.approved === true && work.unusable.has(call.id)) {
          const entry = toolResultEntry(input, `Not run: the arguments for ${call.name} are not a JSON object.`);
          steps?.toolResult(call.id, entry.result, { ok: false, code: 'tool_failed', refused: false });
          resolved.push(entry);
        } else if (decision?.approved === true && work.ambiguous.has(call.id)) {
          // Stored by 0.11.1 or earlier, which wrote no marker; 0.11.0 stored
          // a bare number or boolean as no arguments. Not run, and recorded
          // without arguments; the model is told to call again, and a new
          // call is a new approval for the person.
          const entry = toolResultEntry({ ...input, arguments: {} }, approvalExpired(call.name));
          this.#events?.emit({
            type: 'tool_approval_expired',
            runId,
            sessionKey: live.key(),
            tool: call.name,
            argsDigest: NO_ARGUMENTS_DIGEST,
            decidedBy: 'policy',
          });
          steps?.toolResult(call.id, entry.result, { ok: false, code: 'approval_denied', refused: true });
          resolved.push(entry);
        } else if (decision?.approved === true) {
          // Reported as called only when it is offered to run: a call refused
          // as not available did not run (G-399).
          if (bounded.some((tool) => tool.name === call.name)) called.push(call.name);
          const invoked = await this.#runApproved(bounded, input);
          steps?.toolResult(call.id, invoked.entry.result, invoked.outcome);
          resolved.push(invoked.entry);
        } else {
          // Denied, or never answered: either way not approved, and refused.
          const entry = toolResultEntry(input, decision === undefined ? 'No approval response provided' : (decision.reason ?? 'User denied tool execution'));
          steps?.toolResult(call.id, entry.result, { ok: false, code: 'approval_denied', refused: true });
          resolved.push(entry);
        }
      }

      // A replaced step's calls are answered with their own rows, kept out of
      // this step's results (G-397).
      const replaced = work.replaced.map(({ position, calls }) => ({
        ...toolResultRow(calls.map((call) => toolResultEntry(
          {
            id: typeof call.id === 'string' ? call.id : '',
            name: typeof call.name === 'string' ? call.name : '',
            arguments: typeof call.arguments === 'object' && call.arguments !== null && !Array.isArray(call.arguments) ? call.arguments : {},
            resultId: typeof call.result_id === 'string' ? call.result_id : null,
          },
          REPLACED_STEP_RESULT,
        ))),
        [REPLACED_STEP_KEY]: { position },
      }) as JsonObject);
      await live.thread().record([
        ...replaced,
        ...(work.calls.length === 0 ? [] : [toolResultRow([...work.results.values(), ...resolved], [...work.decisions.values()])]),
      ], runId);
      // Only replaced steps were answered: this turn's tools are not narrowed.
      return work.calls.length === 0 ? offered : bounded;
    };

    if (alreadyLocked) return await resolve(session);

    return await session.lock(
      resolve,
      this.#setting(session, 'lock_ttl') ?? RESOLUTION_LOCK_SECONDS,
      this.#setting(session, 'lock_wait') ?? RESOLUTION_WAIT_SECONDS,
    );
  }

  /**
   * The gated calls of the last tool-calling turn that have no result yet, with
   * what is needed to answer them, or null when there are none.
   */
  async #unresolved(session: Session, mode: AgentMode): Promise<{
    calls: AssistantRow['tool_calls'];
    results: Map<string, ToolResultEntry>;
    decisions: Map<string, ApprovalDecisionEntry>;
    approvalIds: Map<string, string>;
    /** The step repeats a tool call id: none of its gated calls may run (G-392). */
    repeated: boolean;
    /** Earlier parked steps a later tool-call step replaced, with their unanswered gated calls (G-397). */
    replaced: { position: number; calls: JsonObject[] }[];
    scope: ReturnType<typeof approvalToolScope>;
    unusable: Set<string>;
    ambiguous: Set<string>;
  } | null> {
    const messages = await session.thread().messages();
    const rows = messages.map((entry) => entry.message);
    const view = threadView(rows);
    const replaced = replacedStepCalls(messages, (name) => mode.needsApproval(name));
    let index = view.length - 1;

    while (index >= 0 && !(view[index]!.type === 'assistant' && (view[index] as AssistantRow).tool_calls.length > 0)) {
      index -= 1;
    }

    if (index === -1) return null;

    const assistant = view[index] as AssistantRow;
    // EVERY result and decision recorded after the step, not only the first
    // run of result rows: a row of another kind between them (a note, a
    // system row) must not hide a result and let its call run again
    // (G-397, G-399). A later entry for an id replaces an earlier one.
    const answered = view.slice(index + 1).filter((row) => row.type === 'tool_result') as ToolResultRow[];
    const results = new Map(answered.flatMap((row) => row.tool_results).map((entry) => [entry.tool_call_id, entry]));
    const decisions = new Map(answered.flatMap((row) => row.tool_approval_responses).map((entry) => [entry.approval_id, entry]));
    const approvalIds = new Map(assistant.tool_approval_requests.map((request) => [request.tool_call_id, request.approval_id]));

    // Read before any decision is used: a map keyed by a repeated call id
    // would hand one approval to every call carrying it (G-392).
    const repeated = repeatsCallIds(assistant.tool_calls as unknown as JsonObject[], assistant.tool_approval_requests as unknown as JsonObject[]);

    // A call that has a result is DONE, whatever its decision says. A call that
    // needs nobody is not this method's to run. In a step that repeats a call
    // id a result by id cannot say whose it is, so its gated calls stay open
    // until there are as many results for the id as calls carrying it.
    // Counted on the rows as stored: the replay view folds results by id.
    let stored = rows.length - 1;
    while (stored >= 0 && !(rows[stored]!.type === 'assistant' && Array.isArray(rows[stored]!.tool_calls) && (rows[stored]!.tool_calls as JsonValue[]).length > 0)) stored -= 1;
    // Which of its calls already has an outcome: each result belongs to one
    // call, matched by id and the arguments it records, in the segment it sits
    // in or an earlier step's, never by id alone (G-392, G-397).
    const done = stored >= 0 ? resultOwners(rows).get(stored) ?? new Set<number>() : new Set<number>();
    // A superseded step is never resolved here: it is answered as replaced
    // (G-397), and nothing of it runs.
    const calls = stored >= 0 && supersededAt(messages, stored) !== -1
      ? []
      : repeated
        ? assistant.tool_calls.filter((call, at) => mode.needsApproval(call.name) && !done.has(at))
        : assistant.tool_calls.filter(
          (call, at) => !done.has(at) && (approvalIds.has(call.id) || mode.needsApproval(call.name)),
        );

    return calls.length === 0 && replaced.length === 0 ? null : { calls, results, decisions, approvalIds, repeated, replaced, scope: approvalToolScope(rows), unusable: unusableStoredCalls(rows), ambiguous: ambiguousCalls(assistant) };
  }

  /**
   * Run a call a person approved, or record why it could not run.
   *
   * Recorded rather than thrown. Resolution runs at the start of every send(),
   * so a call that throws here, because its tool is no longer offered to this
   * run or the authorizer now refuses it, would throw again on every later
   * send(), and the session could never move on. The approval stands, but the
   * call does not run.
   */
  async #runApproved(offered: readonly HarnessTool[], call: LlmToolCall): Promise<Invoked> {
    if (!offered.some((tool) => tool.name === call.name)) {
      return {
        entry: toolResultEntry(call, `Not run: ${call.name} is not available to this run.`),
        outcome: { ok: false, code: 'tool_not_available', refused: true },
      };
    }

    try {
      return await this.#invoke(offered, call);
    } catch (error) {
      if (error instanceof HarnessError && error.code === 'call_not_authorized') {
        return { entry: toolResultEntry(call, `Not run: this call to ${call.name} is not authorized.`), outcome: REFUSED_CALL };
      }

      throw error;
    }
  }

  async #invoke(offered: readonly HarnessTool[], call: LlmToolCall): Promise<Invoked> {
    const tool = offered.find((candidate) => candidate.name === call.name);

    if (tool === undefined) {
      throw HarnessError.toolNotAvailable(
        call.name,
        offered.map((candidate) => candidate.name),
      );
    }

    // Each call gets its own refusal cell: a gate inside the tool (a subagent
    // the tree would not run, a refused task completion) says so there, and
    // nothing another call, run or session does can reach it.
    const settled = await settleWithRefusalScope(async () => await tool.handle(call.arguments));

    if (settled.ok) {
      const result = settled.value;

      return {
        entry: toolResultEntry(call, typeof result === 'string' ? result : JSON.stringify(result ?? null)),
        outcome: settled.refused ? REFUSED_CALL : { ok: true },
      };
    }

    const error = settled.error;

    // A failed tool is a RESULT, not a crashed run: the model can often
    // recover, and losing the whole turn to one bad call is worse. A refused
    // call is different and is left to propagate — see `authorizedTool`.
    if (error instanceof HarnessError && error.code === 'call_not_authorized') throw error;

    return {
      entry: toolResultEntry(call, `The tool failed: ${error instanceof Error ? error.message : String(error)}`),
      outcome: settled.refused ? REFUSED_CALL : { ok: false, code: 'tool_failed', refused: false },
    };
  }

  /**
   * Refuse a document the caller cannot use, naming which of the two it is.
   *
   * Text holding no document at all and a document with the wrong shape are
   * separate codes because the next move differs: the first is a prompting or
   * budget problem, the second a schema one.
   */
  #assertDocument(schema: Readonly<JsonObject>, response: LlmResponse): void {
    if (response.structured === undefined || response.structured === null) {
      throw HarnessError.structuredUnreadable(response.text);
    }

    const problems = schemaProblems(schema as JsonObject, response.structured, schemaName(schema));

    if (problems.length > 0) {
      throw HarnessError.structuredSchemaViolation(response.text, problems);
    }
  }

  async #finish(
    session: Session,
    runId: string,
    called: readonly string[],
    finishReason: string,
    text: string,
    run: RunContext,
    stoppedBecause: string | null,
  ): Promise<AgentResponse> {
    await session.completeRun(runId, finishReason, called);
    this.#events?.emit({
      type: 'run.finished',
      runId,
      sessionKey: session.key(),
      finishReason,
      toolCalls: called,
      steps: run.ledger.steps,
      costUsd: run.ledger.unmeteredRuns > 0 ? null : run.ledger.costUsd,
      at: new Date().toISOString(),
    });

    return {
      runId,
      text,
      steps: run.ledger.steps,
      toolCalls: called,
      finishReason,
      pendingApprovals: [],
      stoppedBecause,
    };
  }
}

/**
 * Answer a pending approval, durably.
 *
 * The decision is RECORDED IN THE THREAD, not held anywhere else. Who may
 * approve is the APPLICATION's decision, not this package's: the session is
 * already scoped to a participant, so nobody can answer another participant's
 * approval through it, but "this user may approve THIS action" is a question
 * only the host can answer. Authorize before calling.
 *
 * Nothing runs until the next `send()`, which acts on every decision recorded
 * by then and refuses any pending call still without one. With several pending,
 * record them all first.
 *
 * `approvalId` is `PendingApproval.id`, not the tool call id.
 */
function validateRuntimeConfig(config: RuntimeConfig): void {
  for (const key of ['provider', 'model'] as const) {
    const value = config[key];
    if (value !== undefined && value !== null && (typeof value !== 'string' || value === '')) throw new TypeError('Agent provider/model defaults must be nonempty strings or null.');
  }
  for (const key of ['max_steps_ceiling', 'max_output_tokens_per_call'] as const) {
    const value = config[key];
    if (value !== undefined && value !== null && (!Number.isSafeInteger(value) || value < 1)) throw new TypeError('Agent step/output ceilings must be positive integers or null.');
  }
  for (const key of ['lock_ttl', 'lock_wait'] as const) {
    const value = config[key];
    if (value !== undefined && (!Number.isSafeInteger(value) || value < (key === 'lock_ttl' ? 1 : 0))) throw new TypeError('Agent lock settings must be whole seconds with positive TTL and nonnegative wait.');
  }
}

function narrowedTools(ceiling: readonly string[], requested?: readonly string[]): readonly string[] {
  if (requested === undefined || requested.includes('*')) return ceiling;
  if (ceiling.includes('*')) return requested;
  const refused = requested.find(name => !ceiling.includes(name));
  if (refused !== undefined) throw HarnessError.toolNotAvailable(refused, ceiling);
  return requested;
}

export async function recordApproval(
  session: Session,
  approvalId: string,
  approved: boolean,
  reason: string | null = null,
): Promise<void> {
  await writeDecisions(session, [{ approvalId, approved, reason }]);
}

/**
 * Answer SEVERAL pending approvals durably, as one row, before resuming once.
 *
 * The same writer as `recordApproval()` and `streamDecision()`. A run that
 * stopped on more than one call needs every answer before it resumes, because
 * a call with no decision is refused when it does.
 */
export async function recordDecisions(session: Session, decisions: readonly ApprovalDecision[]): Promise<void> {
  await writeDecisions(session, assertDecisions(decisions));
}

/** Where decisions are made durable, for every path that takes them. */
async function writeDecisions(session: Session, decisions: readonly ApprovalDecision[]): Promise<void> {
  // A decision for a step that was superseded is refused before anything is
  // written: it could never run anything (G-397).
  const messages = await session.thread().messages();
  for (const decision of decisions) {
    if (approvalSuperseded(messages, decision.approvalId)) throw HarnessError.approvalNotPending(decision.approvalId);
  }
  const run = await session.run();

  await session.thread().record([toolResultRow([], decisions.map((decision) => ({
    approval_id: decision.approvalId,
    approved: decision.approved,
    reason: decision.reason ?? null,
  })))], run?.id ?? null);
}

/** Runtime guards: a decision arrives from an app's request handling, not from a typed caller. */
function assertDecisions(decisions: unknown): ApprovalDecision[] {
  if (!Array.isArray(decisions) || decisions.length === 0) {
    throw new TypeError('A decision list needs at least one decision.');
  }

  return decisions.map((decision: unknown) => {
    const candidate = decision as Partial<ApprovalDecision> | null;
    if (candidate === null || typeof candidate !== 'object'
      || typeof candidate.approvalId !== 'string' || candidate.approvalId === ''
      || typeof candidate.approved !== 'boolean'
      || (candidate.reason !== undefined && candidate.reason !== null && typeof candidate.reason !== 'string')) {
      throw new TypeError('A decision is { approvalId: nonempty string, approved: boolean, reason?: string | null }.');
    }

    return { approvalId: candidate.approvalId, approved: candidate.approved, reason: candidate.reason ?? null };
  });
}

/** The context a run EXECUTES under: a metadata-only context states no run tree. */
function executionContext(context: RunContext | undefined): RunContext | undefined {
  return context?.metadataOnly === true ? undefined : context;
}

/**
 * Where this run's step sequence ended, so a resumed run continues it.
 *
 * Last, and isolated: the run is already closed out, and failing to note a
 * sequence number must not turn a finished run into a failed one.
 */
async function rememberSeq(session: Session, runId: string, steps: RunStepTagger): Promise<void> {
  if (steps.seq === 0) return;

  try {
    await session.rememberStepSeq(runId, steps.seq);
  } catch (error) {
    process.emitWarning(`prism-harness could not record the step sequence of run [${runId}]: ${String(error)}`);
  }
}
