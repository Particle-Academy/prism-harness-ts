import { randomUUID } from 'node:crypto';
import type { JsonObject } from './json.js';
import { HarnessError } from './errors.js';
import type { HarnessEvents } from './events.js';
import type { AgentMode, ModeRegistry } from './modes.js';
import type { Session } from './session.js';
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
}

export interface LlmToolCall {
  id: string;
  name: string;
  arguments: JsonObject;
}

export interface LlmResponse {
  text: string;
  toolCalls?: readonly LlmToolCall[];
  /** `stop`, `tool_calls`, `length`, … — the provider's own reason, passed through. */
  finishReason: string;
  /** Null when the provider does not report one. NOT zero — see `RunLedger.recordCost`. */
  costUsd?: number | null;
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

export interface PendingApproval {
  id: string;
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
  ): Promise<AgentResponse> {
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

    if (prompt !== '') {
      await thread.record([{ type: 'user', content: prompt }], runId);
    }

    try {
      return await this.#loop(session, mode, run, runId, provider, model, toolNames);
    } catch (error) {
      const failure = error instanceof Error ? error.message : String(error);
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
    toolNames?: readonly string[],
  ): Promise<AgentResponse> {
    const thread = session.thread();
    const resolved = await this.#tools.resolve(toolNames ?? mode.tools, session);
    const offered = this.#authorizer
      ? await this.#authorizer.allowed(session, resolved)
      : [...resolved.values()];

    const called: string[] = [];
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
        messages: (await thread.messages()).map((entry) => entry.message),
        tools: offered,
        provider,
        model,
      });

      run.ledger.recordSteps(1);
      run.ledger.recordCost(response.costUsd ?? null);
      text = response.text;
      finishReason = response.finishReason;

      const toolCalls = response.toolCalls ?? [];

      await thread.record(
        [
          {
            type: 'assistant',
            content: response.text,
            tool_calls: toolCalls.map((call) => ({ id: call.id, name: call.name })),
          },
        ],
        runId,
      );

      if (toolCalls.length === 0) {
        return await this.#finish(session, runId, called, finishReason, text, run, null);
      }

      const pending = await this.#pendingApprovals(session, mode, toolCalls);

      if (pending.length > 0) {
        // FAILS CLOSED. The run stops here and the request is already in the
        // thread, so a different process can pick it up after a human answers.
        await thread.record(
          [{ type: 'tool_approval_request', approvals: pending as unknown as JsonObject[] }],
          runId,
        );

        return {
          runId,
          text,
          steps: run.ledger.steps,
          toolCalls: called,
          finishReason: 'awaiting_approval',
          pendingApprovals: pending,
          stoppedBecause: null,
        };
      }

      for (const call of toolCalls) {
        called.push(call.name);
        await thread.record([await this.#invoke(offered, call)], runId);
      }
    }
  }

  /**
   * Which of these calls needs a human, and has not had one.
   *
   * An approval already answered in the thread is NOT asked again — that is the
   * whole point of recording it durably. An answered-and-denied approval is
   * also not asked again; it is simply not executed.
   */
  async #pendingApprovals(
    session: Session,
    mode: AgentMode,
    toolCalls: readonly LlmToolCall[],
  ): Promise<PendingApproval[]> {
    const gated = toolCalls.filter((call) => mode.needsApproval(call.name));

    if (gated.length === 0) return [];

    const answered = await this.#answeredApprovals(session);

    return gated
      .filter((call) => !answered.has(call.id))
      .map((call) => ({ id: call.id, tool: call.name, arguments: call.arguments }));
  }

  async #answeredApprovals(session: Session): Promise<Map<string, boolean>> {
    const answered = new Map<string, boolean>();

    for (const entry of await session.thread().messages()) {
      if (entry.message.type !== 'tool_approval_response') continue;

      const id = entry.message.approval_id;
      const approved = entry.message.approved;

      if (typeof id === 'string') answered.set(id, approved === true);
    }

    return answered;
  }

  async #invoke(offered: readonly HarnessTool[], call: LlmToolCall): Promise<JsonObject> {
    const tool = offered.find((candidate) => candidate.name === call.name);

    if (tool === undefined) {
      throw HarnessError.toolNotAvailable(
        call.name,
        offered.map((candidate) => candidate.name),
      );
    }

    try {
      const result = await tool.handle(call.arguments);

      return {
        type: 'tool_result',
        tool_call_id: call.id,
        name: call.name,
        result: typeof result === 'string' ? result : JSON.stringify(result ?? null),
      };
    } catch (error) {
      // A failed tool is a RESULT, not a crashed run: the model can often
      // recover, and losing the whole turn to one bad call is worse. A refused
      // call is different and is left to propagate — see `authorizedTool`.
      if (error instanceof HarnessError && error.code === 'call_not_authorized') throw error;

      return {
        type: 'tool_result',
        tool_call_id: call.id,
        name: call.name,
        result: `The tool failed: ${error instanceof Error ? error.message : String(error)}`,
        failed: true,
      };
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
 */
export async function recordApproval(
  session: Session,
  approvalId: string,
  approved: boolean,
  reason: string | null = null,
): Promise<void> {
  const run = await session.run();

  await session.thread().record(
    [
      {
        type: 'tool_approval_response',
        approval_id: approvalId,
        approved,
        reason,
      },
    ],
    run?.id ?? null,
  );
}
