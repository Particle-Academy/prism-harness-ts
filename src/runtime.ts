import { randomUUID } from 'node:crypto';
import type { JsonObject, JsonValue } from './json.js';
import {
  assistantRow,
  threadView,
  toolResultEntry,
  toolResultRow,
  type ApprovalDecisionEntry,
  type ApprovalRequestEntry,
  type AssistantRow,
  type ToolResultEntry,
  type ToolResultRow,
} from './thread-rows.js';
import { admitAttachments } from './attachments.js';
import { HarnessError } from './errors.js';
import type { HarnessEvents } from './events.js';
import type { AgentMode, ModeRegistry } from './modes.js';
import type { Session } from './session.js';
import { schemaName, schemaProblems } from './structured.js';
import { RunBudget, RunContext } from './subagents.js';
import type { HarnessTool, ToolAuthorizer, ToolRegistry } from './tools.js';

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
  client: LlmClient;
  modes: ModeRegistry;
  tools: ToolRegistry;
  authorizer?: ToolAuthorizer;
  events?: HarnessEvents;
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

  constructor(options: AgentRuntimeOptions) {
    this.#client = options.client;
    this.#modes = options.modes;
    this.#tools = options.tools;
    this.#authorizer = options.authorizer;
    this.#events = options.events;
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
    return await this.#turn(session, prompt, toolNames, context, additionalContent);
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
    const response = await this.#turn(session, prompt, toolNames, context, additionalContent, schema);

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
  ): Promise<AgentResponse> {
    // Refused before a run exists: a bad attachment is a mistake in the call,
    // and it should not cost a run, events or budget.
    const attachments = admitAttachments(prompt, additionalContent);
    const mode = this.#modes.resolve(await session.mode());
    const provider = (await session.provider()) ?? 'unknown';
    const model = (await session.model()) ?? 'unknown';
    const runId = randomUUID();
    const run = context ?? RunContext.root(runId, new RunBudget(mode.maxSteps));
    const thread = session.thread();

    if (run.tooDeep()) {
      throw HarnessError.runNotPermitted(
        `This run is nested ${run.depth} deep, at or past the ceiling. Nobody debugs a tree that deep, ` +
          'and a configuration that produced one is a mistake worth reporting rather than executing.',
      );
    }

    await session.beginRun(runId, mode.name, provider, model);
    this.#events?.emit({
      type: 'run.started',
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
      const resolved = await this.#tools.resolve(toolNames ?? mode.tools, session);
      const offered = this.#authorizer ? await this.#authorizer.allowed(session, resolved) : [...resolved.values()];
      const called: string[] = [];

      // Decisions recorded since the run stopped are acted on FIRST, before a
      // new prompt is recorded, so the results land after the calls they answer
      // rather than after the new turn.
      await this.#resolveApprovals(session, mode, offered, runId, called);

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

      return await this.#loop(session, mode, run, runId, provider, model, offered, called, schema);
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

      const response = await this.#client({
        systemPrompt: mode.systemPrompt,
        messages: threadView((await thread.messages()).map((entry) => entry.message)),
        tools: offered,
        provider,
        model,
        providerOptions: mode.providerOptions,
        ...(schema === undefined ? {} : { schema }),
      });

      run.ledger.recordSteps(1);
      run.ledger.recordCost(response.costUsd ?? null);
      text = response.text;
      finishReason = response.finishReason;

      const toolCalls = response.toolCalls ?? [];
      const gated = toolCalls.filter((call) => mode.needsApproval(call.name));
      const requests: ApprovalRequestEntry[] = gated.map((call) => ({
        approval_id: `apr_${randomUUID().replaceAll('-', '')}`,
        tool_call_id: call.id,
      }));

      // The next step's request is built from this row, so it keeps what a
      // provider needs sent back: each call's arguments and provider ids, the
      // turn's provider state, and the approvals it is waiting on (G-58).
      const answering = schema !== undefined && toolCalls.length === 0;
      const metadata: JsonObject | undefined = answering
        ? { ...(response.additionalContent ?? {}), structured: response.structured ?? null }
        : (response.additionalContent as JsonObject | undefined);

      await thread.record([assistantRow(response.text, toolCalls, metadata, requests)], runId);

      if (toolCalls.length === 0) {
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

      for (const call of toolCalls.filter((candidate) => !gated.includes(candidate))) {
        called.push(call.name);
        results.push(await this.#invoke(offered, call));
      }

      if (results.length > 0) {
        await thread.record([toolResultRow(results)], runId);
      }

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
    offered: readonly HarnessTool[],
    runId: string,
    called: string[],
  ): Promise<void> {
    // A read without the lock first: nearly every send() has nothing to resolve
    // and should not wait on the session lock to find that out.
    if ((await this.#unresolved(session, mode)) === null) return;

    // Then under the session lock, reading again inside it. Two workers resuming
    // the same session at once would otherwise both find the approved call with
    // no result, and both run it. The second now waits, finds the result the
    // first recorded, and runs nothing. If it cannot get the lock it throws
    // SessionLocked, and nothing runs.
    await session.lock(
      async (live) => {
        const work = await this.#unresolved(live, mode);

        if (work === null) return;

        const resolved: ToolResultEntry[] = [];

        for (const call of work.calls) {
          const input = { id: call.id, name: call.name, arguments: call.arguments, resultId: call.result_id };
          const approvalId = work.approvalIds.get(call.id);
          const decision = approvalId === undefined ? undefined : work.decisions.get(approvalId);

          if (decision?.approved === true) {
            called.push(call.name);
            resolved.push(await this.#runApproved(offered, input));
          } else {
            resolved.push(
              toolResultEntry(input, decision === undefined ? 'No approval response provided' : (decision.reason ?? 'User denied tool execution')),
            );
          }
        }

        await live.thread().record([toolResultRow([...work.results.values(), ...resolved], [...work.decisions.values()])], runId);
      },
      RESOLUTION_LOCK_SECONDS,
      RESOLUTION_WAIT_SECONDS,
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
  } | null> {
    const view = threadView((await session.thread().messages()).map((entry) => entry.message));
    let index = view.length - 1;

    while (index >= 0 && !(view[index]!.type === 'assistant' && (view[index] as AssistantRow).tool_calls.length > 0)) {
      index -= 1;
    }

    if (index === -1) return null;

    const assistant = view[index] as AssistantRow;
    const answered = view.slice(index + 1).find((row) => row.type === 'tool_result') as ToolResultRow | undefined;
    const results = new Map((answered?.tool_results ?? []).map((entry) => [entry.tool_call_id, entry]));
    const decisions = new Map((answered?.tool_approval_responses ?? []).map((entry) => [entry.approval_id, entry]));
    const approvalIds = new Map(assistant.tool_approval_requests.map((request) => [request.tool_call_id, request.approval_id]));

    // A call that has a result is DONE, whatever its decision says. A call that
    // needs nobody is not this method's to run.
    const calls = assistant.tool_calls.filter(
      (call) => !results.has(call.id) && (approvalIds.has(call.id) || mode.needsApproval(call.name)),
    );

    return calls.length === 0 ? null : { calls, results, decisions, approvalIds };
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
  async #runApproved(offered: readonly HarnessTool[], call: LlmToolCall): Promise<ToolResultEntry> {
    if (!offered.some((tool) => tool.name === call.name)) {
      return toolResultEntry(call, `Not run: ${call.name} is not available to this run.`);
    }

    try {
      return await this.#invoke(offered, call);
    } catch (error) {
      if (error instanceof HarnessError && error.code === 'call_not_authorized') {
        return toolResultEntry(call, `Not run: this call to ${call.name} is not authorized.`);
      }

      throw error;
    }
  }

  async #invoke(offered: readonly HarnessTool[], call: LlmToolCall): Promise<ToolResultEntry> {
    const tool = offered.find((candidate) => candidate.name === call.name);

    if (tool === undefined) {
      throw HarnessError.toolNotAvailable(
        call.name,
        offered.map((candidate) => candidate.name),
      );
    }

    try {
      const result = await tool.handle(call.arguments);

      return toolResultEntry(call, typeof result === 'string' ? result : JSON.stringify(result ?? null));
    } catch (error) {
      // A failed tool is a RESULT, not a crashed run: the model can often
      // recover, and losing the whole turn to one bad call is worse. A refused
      // call is different and is left to propagate — see `authorizedTool`.
      if (error instanceof HarnessError && error.code === 'call_not_authorized') throw error;

      return toolResultEntry(call, `The tool failed: ${error instanceof Error ? error.message : String(error)}`);
    }
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
export async function recordApproval(
  session: Session,
  approvalId: string,
  approved: boolean,
  reason: string | null = null,
): Promise<void> {
  const run = await session.run();

  await session.thread().record([toolResultRow([], [{ approval_id: approvalId, approved, reason }])], run?.id ?? null);
}
