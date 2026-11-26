import { AsyncLocalStorage } from 'node:async_hooks';
import { randomUUID } from 'node:crypto';
import { validateAgent } from './address.js';
import type { JsonObject, JsonValue } from './json.js';
import {
  assistantRow,
  approvalToolScope, approvalRunSteps, pausedRunState, turnDecisions,
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
  pendingApprovalIds,
  ARGUMENTS_AS_SENT,
} from './thread-rows.js';
import { usableArguments } from './tool-arguments.js';
import { admitAttachments } from './attachments.js';
import { HarnessError } from './errors.js';
import type { HarnessEvents, ToolDecisionEvent } from './events.js';
import { ToolPolicy, argsDigest } from './tool-policy.js';
import type { AgentMode, ModeRegistry } from './modes.js';
import type { ChildRefs, Session } from './session.js';
import { schemaName, schemaProblems } from './structured.js';
import { RunBudget, RunContext, RunLedger, bindingRoute, type ModelBinding } from './subagents.js';
import { RunTrees, type RunTreeRef } from './run-trees.js';
import { RunBudgets, type AllowanceFacts } from './run-budgets.js';
import type { RunContextKind } from './plans/budget.js';
import type { HarnessTool, ToolAuthorizer, ToolRegistry } from './tools.js';
import { settleWithRefusalScope } from './refusals.js';
import { RunStepTagger, StepChannel, type RunStep, type ToolCallOutcome } from './steps.js';
import { planApprovalGate } from './plans/tools.js';

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
  /**
   * Which approval-gated tools run without asking a person (G-326). Owned by
   * the HOST: passed per launch and replaced with `setToolPolicy()`. Never
   * sent to the model and never reachable from a tool. Defaults to
   * `ToolPolicy.none()`: everything that needs approval asks.
   */
  toolPolicy?: ToolPolicy;
}

/** One human decision on a pending approval. `approvalId` is `PendingApproval.id`. */
export interface ApprovalDecision {
  approvalId: string;
  approved: boolean;
  reason?: string | null;
}

/** What the step paths thread through a turn. */
interface Stepping {
  /** Absent for a send() that only takes the session lock (a resumed root). */
  tagger?: RunStepTagger;
  /** The id to run under, when the step path has already claimed one. */
  runId?: string;
  /** True when that id is a parked run's own, continued. */
  resuming?: boolean;
  /** True when the caller already holds the session lock for the whole turn. */
  locked?: boolean;
  /** True when this turn should mix `plan_updated` steps in itself (G-360); a decision stream watches before its decision steps. */
  watchPlan?: boolean;
}

/** A tool call's result as recorded, and how it ended. */
interface Invoked {
  entry: ToolResultEntry;
  outcome: ToolCallOutcome;
}

const REFUSED_CALL: ToolCallOutcome = { ok: false, code: 'tool_call_refused', refused: true };

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
  /**
   * The most a named agent may spend, in USD, within one run TREE (G-321 E).
   *
   * Evaluated against the tree's own ledger, so it narrows the tree's spend and
   * never opens a second account. Per tree only: a per-day window is not
   * implemented. A provider that reports no cost cannot be held to it, so an
   * agent with a ceiling fails closed (`budget_unpriced`) once one does.
   */
  spend_ceiling?: number | null;
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
  /** A subagent child run's tree and plan budget references, by run id, while it runs. */
  readonly #childRefs = new Map<string, { tree: RunTreeRef | null; planLedger: { session: string; plan: string } | null }>();
  readonly #client: LlmClient;

  readonly #modes: ModeRegistry;

  readonly #tools: ToolRegistry;

  readonly #authorizer?: ToolAuthorizer;

  readonly #events?: HarnessEvents;
  readonly #config: RuntimeConfig;
  readonly #agents: Record<string, RuntimeConfig>;
  readonly #narrowRequestedTools: boolean;
  readonly #stepEvents: boolean;
  #toolPolicy: ToolPolicy;

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
    this.#toolPolicy = this.#admitPolicy(options.toolPolicy ?? ToolPolicy.none());
  }

  /** The host tool policy in force. */
  get toolPolicy(): ToolPolicy {
    return this.#toolPolicy;
  }

  /**
   * Replace the host tool policy (G-326).
   *
   * Applies from the next tool call. A call already waiting on a person keeps
   * waiting: a policy that would now allow it does not run it. A call the old
   * policy allowed that has not started yet is refused, not run, when the new
   * policy no longer allows it, so a revoke takes effect at once.
   *
   * Validated against the registered tools' declarations: a policy naming a
   * tool that declares `destructive` is refused and the old policy stays.
   *
   * @throws HarnessError `tool_policy_refused`
   */
  setToolPolicy(policy: ToolPolicy): void {
    this.#toolPolicy = this.#admitPolicy(policy);
  }

  #admitPolicy(policy: ToolPolicy): ToolPolicy {
    if (!(policy instanceof ToolPolicy)) throw new TypeError('toolPolicy must be a ToolPolicy.');

    return ToolPolicy.fromJson(policy.toJSON(), this.#tools.declarations());
  }

  #toolDecision(
    type: 'tool_auto_allowed' | 'tool_approval_asked' | 'tool_approved' | 'tool_denied' | 'tool_approval_expired',
    session: Session,
    runId: string,
    call: { id: string; name: string; arguments: unknown },
    tool: HarnessTool | undefined,
    decidedBy: ToolDecisionEvent['decidedBy'],
  ): void {
    this.#events?.emit({
      type,
      runId,
      sessionKey: session.key(),
      policyRevision: this.#toolPolicy.revision,
      server: tool?.server ?? null,
      tool: call.name,
      argsDigest: argsDigest(call.arguments),
      decidedBy,
      harnessAgent: session.agent,
    });
  }

  #setting<K extends keyof RuntimeConfig>(session: Session, key: K): RuntimeConfig[K] {
    const options = session.agent === null ? undefined : this.#agents[session.agent];
    return options && Object.hasOwn(options, key) ? options[key] : this.#config[key];
  }

  /**
   * The run under the agent's output ceiling. Its STEP ceiling,
   * max_steps_ceiling, is never applied here: it is one input of the run's
   * step allowance (RunBudgets.allowance, G-387), absolute on the run's
   * whole count under every context.
   */
  #boundedRun(session: Session, run: RunContext): RunContext {
    const output = this.#setting(session, 'max_output_tokens_per_call'), budget = run.budget;
    if (output == null) return run;
    return run.withBudget(new RunBudget(
      budget.maxSteps, budget.maxCostUsd, budget.maxSeconds,
      Math.min(budget.maxOutputTokensPerCall ?? output, output),
    ));
  }

  /**
   * What a run's step allowance is computed from (RunBudgets.allowance,
   * G-387), as it begins. `resumed` is the state of the parked run this turn
   * resumes, null for a turn that resumes nothing.
   *
   * - The ceiling saved with a subagent child applies on every turn the
   *   runner did not start.
   * - What a runner or host context has left is that context's own budget.
   * - The steps used are what the parked run wrote on the row it parked on
   *   (`harness_steps_used`, its children's steps included), never fewer
   *   than the model steps its own rows show: a row cannot be forged below
   *   the transcript it sits in, and a row written before the field existed
   *   is counted from its rows. A field that is present and malformed fails
   *   closed: the run has used its whole ceiling.
   */
  async #allowanceFacts(session: Session, mode: AgentMode, kind: RunContextKind, execution: RunContext | undefined, resumed: ReturnType<typeof pausedRunState>, rows: readonly JsonObject[]): Promise<AllowanceFacts> {
    const facts: AllowanceFacts = {
      kind,
      modeSteps: mode.maxSteps,
      savedCeiling: kind === 'runner' ? null : await session.childCeiling(),
      contextLeft: execution == null ? null : Math.max(0, execution.budget.maxSteps - execution.ledger.steps),
      stepsCeiling: this.#setting(session, 'max_steps_ceiling') ?? null,
      used: 0,
      treeLeft: null,
      planLeft: null,
      took: 0,
    };
    if (kind === 'runner' || resumed === null) return facts;
    if (resumed.stepsUsed === 'invalid') {
      const ceiling = RunBudgets.allowance(facts).ceiling;

      return { ...facts, savedCeiling: ceiling, used: ceiling };
    }

    return { ...facts, used: Math.max(resumed.stepsUsed ?? 0, approvalRunSteps(rows) ?? 0) };
  }

  /** The mode a session's next run resolves to, as the runtime would resolve it. */
  async modeOf(session: Session): Promise<AgentMode> {
    return this.#modes.forAgent(session.agent).resolve(await session.mode());
  }

  /**
   * The provider and model a run ACTUALLY requests, field by field (G-321 E).
   *
   * Root run: session override > mode > agent default > global.
   * Subagent child (a context carrying a binding): the parent's session
   * override only when the subagent asked to inherit it > the subagent
   * declaration > the child's mode > the parent's mode (same agent only) >
   * the child agent's default > global. The child session's own override
   * slot is not read for a child: the runner owns that address.
   */
  async #choose(session: Session, mode: AgentMode, binding: ModelBinding | null, key: 'provider' | 'model'): Promise<string> {
    const configured = this.#setting(session, key) ?? null;
    const chosen = binding === null
      ? (await session[key]()) ?? mode[key] ?? configured
      : binding.session[key] ?? binding.declared[key] ?? mode[key] ?? binding.parentMode[key] ?? configured;

    return chosen ?? 'unknown';
  }

  /**
   * How long a run tree hold for this session's turns lasts before it is
   * charged in full: the agent's `lock_ttl`, as the turn itself holds it.
   *
   * @internal Read by the decision writer.
   */
  holdTtl(session: Session): number {
    return this.#setting(session, 'lock_ttl') ?? RESOLUTION_LOCK_SECONDS;
  }

  /** Inspect the offer without a run/handler or logical mutation; reads may purge expired entries. */
  async effectiveTools(session: Session, modeName?: string): Promise<EffectiveTool[]> {
    const mode = this.#modes.forAgent(session.agent).resolve(modeName ?? await session.mode());
    return (await this.#offer(session, mode)).descriptors;
  }

  async #offer(session: Session, mode: AgentMode, requested?: readonly string[], planTools: readonly HarnessTool[] = []): Promise<{ tools: HarnessTool[]; descriptors: EffectiveTool[] }> {
    return session.resolveReadOnly(mode.name, async view => {
      const names = session.agent !== null || this.#narrowRequestedTools ? narrowedTools(mode.tools, requested) : (requested ?? mode.tools);
      const detailed = await this.#tools.resolveDetailed(names, view);
      const tools = new Map([...detailed].map(([name, entry]) => [name, entry.tool]));
      const owners = new Map<string, { owner: string; source: string }>([...detailed.values()].map(entry => [entry.tool.name, { owner: entry.owner, source: entry.source }]));
      // G-360: the plan tools, built for the live session; never during inspection.
      for (const tool of planTools) {
        tools.set(tool.name, tool);
        owners.set(tool.name, { owner: 'harness.plans', source: 'builtin' });
      }
      if (view.agent !== null && tools.has('complete_task')) {
        const authorizer = this.#authorizer?.forAgent(view.agent);
        if (!authorizer?.enabled || !authorizer.hasCallPolicy()) throw new HarnessError('unsafe_authorization_configuration', 'Named task completion requires enabled authorization and an explicit call policy.');
      }
      const offered = this.#authorizer ? await this.#authorizer.allowed(view, tools) : [...tools.values()];
      const descriptors = offered.map(tool => ({ name: tool.name, owner: owners.get(tool.name)!.owner, source: owners.get(tool.name)!.source }));
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
    return await this.#resumeLocked(session, prompt, context, (live, stepping) =>
      this.#turn(live, prompt, toolNames, context, additionalContent, undefined, stepping));
  }

  /**
   * A turn that resumes a parked root (an empty prompt, no run tree handed
   * in, and a run tree the parked run began in) runs under the session lock,
   * as a decision stream does: two resumes of one parked root never overlap,
   * so the second is held to the tree record the first opened (G-381). Any
   * other turn runs as before.
   */
  async #resumeLocked<T>(session: Session, prompt: string, context: RunContext | undefined, turn: (live: Session, stepping: Stepping | undefined) => Promise<T>): Promise<T> {
    const observer = this.#observerStepping(context);
    // The same test the turn uses for a bare context: none (null included) or metadata only.
    const kind = RunTrees.contextKind(context);
    if (prompt !== '' || (kind !== 'none' && kind !== 'metadata') || (await session.resumedRunTree()) === null) return turn(session, observer);

    return this.#holdingTurnLock(session, (live) => turn(live, { ...observer, locked: true }));
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
      this.#turn(live, prompt, toolNames, context, additionalContent, undefined, { tagger, locked: true, watchPlan: true })));
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
   * streaming the same decision again is refused with `approval_not_pending`,
   * since that approval is no longer pending. A decision for an id the run
   * never issued is refused the same way, inside the lock, before anything is
   * recorded or sent (G-354). A stream that cannot take the lock fails with
   * `session_locked` and records nothing.
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

  /**
   * Record a person's decision, as `recordApproval()` does, under the session
   * lock with this session's agent's `lock_ttl` / `lock_wait`.
   *
   * `recordApproval()` and `recordDecisions()` take the same lock with fixed
   * values (300 s / 30 s). The lock is not re-entrant: a caller already holding
   * it is refused with `session_locked` once the wait runs out, and nothing is
   * recorded. Nothing runs until the next `send()`. Authorize before calling.
   */
  async decide(session: Session, approvalId: string, approved: boolean, reason: string | null = null): Promise<void> {
    await writeDecisionsLocked(
      session,
      assertDecisions([{ approvalId, approved, reason }]),
      this.#setting(session, 'lock_ttl') ?? RESOLUTION_LOCK_SECONDS,
      this.#setting(session, 'lock_wait') ?? RESOLUTION_WAIT_SECONDS,
    );
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
      // before any tool or provider call. A plan the decisions wrote is noted
      // (G-360): this stream opens with it even when the decision ended it, or
      // the reader would never learn that a draft was rejected.
      let decidedPlan = false;
      const capture = live.planService()?.watch(live.key(), () => { decidedPlan = true; }) ?? (() => {});
      try {
        await writeDecisions(live, decisions, context);
      } finally {
        capture();
      }

      const tagger = this.#tagger(context, channel, parked?.lastSeq ?? 0);
      const execution = executionContext(context);
      const runId = parked?.id ?? randomUUID();
      tagger.bind(live.key(), runId, execution?.parentRunId ?? null, execution?.rootRunId ?? runId);
      const unwatch = await this.#watchPlan(live, tagger, decidedPlan);

      try {
        for (const decision of decisions) {
          // From the APP's run context, never from the model or the request.
          tagger.decision(decision.approvalId, decision.approved, decision.reason ?? null, context?.actor ?? null);
        }

        // An EMPTY prompt: the thread already holds the request, the decisions
        // and everything before them, and the turn resolves approvals from it
        // before any model call.
        return await this.#turn(live, '', undefined, context, [], undefined, { tagger, runId, resuming: parked !== null, locked: true });
      } finally {
        unwatch();
      }
    }));
  }

  /**
   * Mix `plan_updated` steps into this stream when the host turned them on
   * (G-360, contract g282.3): the plans config's `events.run_steps`, and plans
   * on for this session. Off by default: the stream then stays g282.1.
   */
  async #watchPlan(session: Session, tagger: RunStepTagger, written = false): Promise<() => void> {
    const plans = session.planService();
    if (plans === null) return () => {};
    try {
      if (!(await plans.enabledFor(session)) || !plans.limits().runSteps) return () => {};
    } catch {
      return () => {};
    }
    const initial = await plans.load(session);
    tagger.plans(() => initial, written);

    return plans.watch(session.key(), (plan) => tagger.planUpdated(plan));
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
    const response = await this.#resumeLocked(session, prompt, context, (live, stepping) =>
      this.#turn(live, prompt, toolNames, context, additionalContent, schema, stepping));

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
    // A turnId/actor-only context is METADATA: the run takes exactly the path
    // and budget it would take with no context at all.
    const execution = executionContext(context);
    // A resume with no run tree handed in continues the run that paused: the
    // agent's spend so far and, for a subagent's child run, the binding it ran
    // under, both stored on the row that asked (G-321 E). Without them a child
    // resumed outside the runner would resolve its model like a root run, and
    // any pause would restart a spend ceiling at nothing.
    //
    // Any turn on a subagent child the runner did not start (that resume, a
    // new message, or a run context the host built) continues the run tree
    // and the shared plan budget the child was started under (G-366): kept on
    // the session, or on the paused row for a child paused before that.
    const contextKind = RunTrees.contextKind(context);
    const tail = contextKind === 'runner'
      ? null
      : pausedRunState((await session.thread().messages()).map(entry => entry.message));
    const paused = prompt === '' && execution == null ? tail : null;
    const savedRefs: ChildRefs = (await session.childRefs()) ?? { tree: tail?.tree ?? null, planLedger: tail?.planLedger ?? null };
    // The binding: from the runner's own context when the runner starts the child,
    // even under a run tree the host built (RunTrees.bindingContextKind);
    // otherwise the one SAVED with the child session, which a context
    // the host supplies never replaces. A child paused before the binding was
    // saved there carries it on its paused row. A session the runner started
    // as a child (it has saved run tree references) with no binding anywhere
    // is refused rather than run on the root rules, which would send the
    // child's conversation to a provider nobody chose for it.
    // A host context on a session that was never a child keeps the binding it
    // states for this turn; it is never saved. The rule is bindingRoute(),
    // the one the harness-plans corpus pins in every language.
    const legacyBinding = tail?.binding == null ? null
      : { session: tail.binding.session, declared: tail.binding.declared, parentMode: tail.binding.parent_mode };
    const savedBinding = (await session.childBinding()) ?? legacyBinding;
    const route = bindingRoute(RunTrees.bindingContextKind(execution), execution?.binding != null, savedBinding !== null, (await session.childRefs()) !== null);
    if (route.binding === 'refused') throw HarnessError.childBindingMissing();
    const runnerBinding = route.binding === 'runner' ? execution?.binding ?? null : null;
    const binding = route.binding === 'saved' ? savedBinding
      : (route.binding === 'runner' || route.binding === 'host' ? execution?.binding ?? null : null);
    const provider = await this.#choose(session, mode, binding, 'provider');
    const model = await this.#choose(session, mode, binding, 'model');
    const runId = stepping?.runId ?? randomUUID();
    const tag = session.agent === null ? {} : { harnessAgent: session.agent };
    // G-381: which run tree this run belongs to, its budget route and its
    // depth, decided ONCE here and recorded; every consumer below reads that
    // record and nothing else. A child the runner starts takes its parent
    // run's tree, one level deeper; a turn the runner did not start continues
    // the tree its session saved, at least as deep as the runner started it; a
    // host context is in the host's tree whatever run ids it states.
    // A root turn resuming its parked run stays in the tree that run began in.
    const resumedTree = prompt === '' && (contextKind === 'none' || contextKind === 'metadata') ? await session.resumedRunTree() : null;
    const membership = RunTrees.membershipOf(session, runId, context, savedRefs, await session.childDepth(), resumedTree);
    let run = this.#boundedRun(session, execution === null || execution === undefined
      ? new RunContext(RunLedger.start(runId), new RunBudget(mode.maxSteps), null, 0, {}, binding)
      : (execution.binding === binding ? execution : execution.withBinding(binding)));
    if (run.depth !== membership.depth) run = run.atDepth(membership.depth);
    if (session.agent !== null && paused?.agentSpend != null) {
      run.ledger.carryAgentSpend(session.agent, paused.agentSpend.cost_usd, paused.agentSpend.unmetered_runs);
    }
    const thread = session.thread();
    const steps = stepping?.tagger;

    if (run.tooDeep()) {
      throw HarnessError.runNotPermitted(
        `This run is nested ${run.depth} deep, at or past the ceiling. Nobody debugs a tree that deep, ` +
          'and a configuration that produced one is a mistake worth reporting rather than executing.',
      );
    }

    steps?.bind(session.key(), runId, run.parentRunId, run.rootRunId, session.agent);
    const unwatchPlan = steps !== undefined && stepping?.watchPlan === true ? await this.#watchPlan(session, steps) : () => {};
    await session.beginRun(runId, mode.name, provider, model, membership.tree, membership.depth);
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
      ...tag,
    });

    try {
      // Recorded before any tool can run, so a child this run starts finds it.
      RunTrees.enter(session, runId, membership, run);
      // G-360: the plan, in the system prompt of this turn and never in the
      // thread, read after the run begins and before any model call. A turn
      // with a new user message ends every pending approval, so the plan is
      // reconciled as if nothing were pending before it is rendered.
      const plans = session.planService();
      const planContext = plans === null ? '' : await plans.context(session, prompt);
      const systemPrompt = planContext.trim() === '' ? mode.systemPrompt
        : (mode.systemPrompt === '' ? planContext : `${mode.systemPrompt}\n\n${planContext}`);
      // The plan tools, when the host and this mode turned plans on. They gate
      // themselves, so the mode's approval list and the host policy never
      // wrap them; the authorizer still filters them like any other tool.
      const planTools = plans === null ? [] : await plans.tools(session, mode.name);
      let offered: HarnessTool[];
      if (session.agent === null && !this.#narrowRequestedTools) {
        const resolved = await this.#tools.resolve(toolNames ?? mode.tools, session);
        for (const tool of planTools) resolved.set(tool.name, tool);
        offered = this.#authorizer ? await this.#authorizer.allowed(session, resolved) : [...resolved.values()];
      } else offered = (await this.#offer(session, mode, toolNames, planTools)).tools;
      const stored = (await thread.messages()).map(entry => entry.message);
      const scope = approvalToolScope(stored);
      // G-387: how many model steps this run may take is decided ONCE, here,
      // beside its membership, and recorded with it (RunTrees.allowed); the
      // run is narrowed to it once, and every consumer reads that record or
      // that budget. A turn resuming a parked run continues the count the
      // parked run wrote (its children's steps included), never below the
      // steps its own rows show; a runner start is a new run.
      const facts = await this.#allowanceFacts(session, mode, contextKind, execution, prompt === '' ? tail : null, stored);
      // A turn with no run context continues the count on its own ledger.
      if (execution == null) run.ledger.recordSteps(facts.used);
      const startSteps = run.ledger.steps;
      const own = RunBudgets.allowance(facts);
      const holdTtl = this.#setting(session, 'lock_ttl') ?? RESOLUTION_LOCK_SECONDS;
      // A root run opens a tree; any turn on a child session the runner did
      // not start continues the run tree it was started in, holding what it
      // may spend of it (G-366).
      const held = await RunTrees.holdTree(session, runId, run, membership, holdTtl, own.own_left);
      run = held.run;
      // What a subagent child carries into every later turn the runner does
      // not start: the run tree it belongs to and the shared plan budget that
      // tree spends from, kept on the session and on the row it pauses on.
      if (run.binding !== null) {
        // Only the runner's own context writes them: a context the host
        // supplies never replaces the tree or the binding the child saved.
        // Under a run tree the host built there is no durable tree to name:
        // the references the child saved before are kept, never replaced
        // with nothing, so a later turn the runner does not start still
        // continues the tree it last had rather than a fresh root budget.
        // The plan ledger is the one of the run the child spends from
        // (membership.spends), not of the root its ledger names: a grandchild
        // of a child resumed under a host context keeps the child's.
        const runnerStart = runnerBinding !== null && execution != null;
        if (runnerStart) await RunTrees.childStarted(session, membership);
        const refs: ChildRefs = runnerStart
          ? RunTrees.keptRefs(membership, membership.spends === null ? null : plans?.ledgerOf(membership.spends) ?? null,
            savedRefs.tree, savedRefs.planLedger, false)
          : { tree: typeof membership.tree === 'object' ? membership.tree : null, planLedger: savedRefs.planLedger };
        this.#childRefs.set(runId, refs);
        if (runnerStart) {
          // With the ceiling the runner gave it (G-387), re-applied on every
          // turn on the child the runner does not start.
          if (JSON.stringify(await session.childBinding()) !== JSON.stringify(runnerBinding)
            || await session.childDepth() !== membership.depth
            || await session.childCeiling() !== own.ceiling) await session.rememberChildBinding(runnerBinding, membership.depth, own.ceiling);
          if (JSON.stringify(await session.childRefs()) !== JSON.stringify(refs)) await session.rememberChildRefs(refs);
        }
      }
      // G-366: a turn serving a plan step with a shared budget holds what the
      // plan has left of the steps its own and its tree's allowance leave it.
      const planned = plans === null ? { run, planLeft: null }
        : await plans.planTurn(session, runId, run, context, prompt, Math.min(own.own_left, held.treeLeft ?? Number.POSITIVE_INFINITY));
      run = planned.run;
      const allowance = RunBudgets.allowance({ ...facts, treeLeft: held.treeLeft, planLeft: planned.planLeft });
      // Narrowed ONCE: the loop's step gate, a subagent's pre-check and every
      // hold it takes read this budget. A host or runner context can only
      // lower what it offered, never raise it.
      run = run.withBudget(new RunBudget(startSteps + allowance.allowance, run.budget.maxCostUsd, run.budget.maxSeconds, run.budget.maxOutputTokensPerCall));
      RunTrees.allowed(runId, { ...facts, treeLeft: held.treeLeft, planLeft: planned.planLeft }, startSteps, run);
      if (scope !== null) {
        if (scope.legacy) console.warn('Pending approval has no valid stored tool scope; offering approved requests only.');
        offered = offered.filter(tool => scope.names.includes(tool.name)
          && (mode.tools.includes('*') || mode.tools.includes(tool.name) || planApprovalGate(tool) !== null));
      }
      const called: string[] = [];

      // The tools this turn calls run inside it: a subagent tool reads the
      // turn's own run context from here (see activeTurn()). That includes a
      // call a person approved, run below before the model is called again:
      // outside this scope an approved subagent would run in no tree, at the
      // depth and on the budget of the context the host wired its tool with.
      const turn = run;
      return await activeTurns.run({ sessionKey: session.key(), runId, run: turn }, async () => {
        // Decisions recorded since the run stopped are acted on FIRST, before a
        // new prompt is recorded, so the results land after the calls they answer
        // rather than after the new turn.
        offered = await this.#resolveApprovals(session, mode, offered, runId, called, scope?.legacy === true, steps, stepping?.locked === true);

        if (prompt !== '') {
          // With attachments, the shape prism-ts's UserMessage.toObject() writes:
          // the media parts, then the turn's own text as a trailing text part,
          // which fromObject() strips back off. Without them, unchanged.
          const message: JsonObject =
            attachments.length === 0
              ? { type: 'user', content: prompt }
              : { type: 'user', content: prompt, additional_content: [...attachments, { text: prompt }], additional_attributes: {} };

          await thread.record([message], runId);
        }

        return this.#loop(session, mode, turn, runId, provider, model, offered, called, systemPrompt, schema, steps);
      });
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
        ...tag,
      });

      throw error;
    } finally {
      unwatchPlan();
      this.#childRefs.delete(runId);
      // A root run that holds part of a run tree's budget settles it.
      await RunTrees.finished(session, runId);
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
    systemPrompt: string,
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

      // A named agent's ceiling, on the same root ledger: refused before the
      // call, and only this agent is refused (G-321 E).
      const ceiling = session.agent === null ? null : (this.#agents[session.agent]?.spend_ceiling ?? null);
      if (session.agent !== null && ceiling !== null) run.ledger.assertAgentCeiling(session.agent, ceiling);

      steps?.stepStart();
      const response = await this.#client({
        systemPrompt,
        messages: threadView((await thread.messages()).map((entry) => entry.message)),
        tools: offered,
        provider,
        model,
        providerOptions: mode.providerOptions,
        ...(run.budget.maxOutputTokensPerCall === null ? {} : { maxOutputTokens: run.budget.maxOutputTokensPerCall }),
        ...(schema === undefined ? {} : { schema }),
      });

      run.ledger.recordSteps(1);
      run.ledger.recordCost(response.costUsd ?? null, session.agent);
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
      // The host policy is read per call, never cached for the run, so a
      // replacement applies from the next call (G-326).
      const autoAllowed = new Set<LlmToolCall>();
      const gated: LlmToolCall[] = [];
      for (const call of failedFinish === null ? toolCalls : []) {
        if (unusable.has(call)) continue;
        const tool = offered.find((candidate) => candidate.name === call.name);
        // A plan tool gates itself (G-360), in place of the mode's list and
        // the host policy: neither can ask about it or let it through.
        const own = planApprovalGate(tool);
        if (own !== null) {
          // The host's call policy first: a call it forbids records nothing,
          // asks no one, and is refused by the authorizer when it runs.
          const authorized = this.#authorizer === undefined || tool === undefined
            ? true : await this.#authorizer.allowsCall(session, tool, call.arguments);
          if (await own(call.arguments, authorized)) gated.push(call);
          continue;
        }
        if (!mode.needsApproval(call.name)) continue;
        if (tool !== undefined && this.#toolPolicy.allows(tool)) {
          autoAllowed.add(call);
          continue;
        }
        gated.push(call);
      }
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
      if (requests.length > 0) {
        row.harness_tool_names = offered.map(tool => tool.name);
        // The steps this run has used, its children's included, which its
        // resume continues from; and what it has left of its allowance, plan
        // narrowing included, which nothing reads to decide a budget (G-387).
        const allowance = RunTrees.allowanceOf(runId);
        if (allowance !== null) {
          row.harness_steps_used = allowance.used_at_park;
          row.harness_steps_left = allowance.left_at_park;
        }
        // What the resume continues from (pausedRunState): the agent's spend
        // so far, and a subagent child's model binding.
        if (session.agent !== null) {
          row.harness_agent_spend = {
            cost_usd: run.ledger.agentCostUsd(session.agent),
            unmetered_runs: run.ledger.agentUnmeteredRuns(session.agent),
          };
        }
        if (run.binding !== null) {
          row.harness_binding = {
            session: { ...run.binding.session },
            declared: { ...run.binding.declared },
            parent_mode: { ...run.binding.parentMode },
          };
          const refs = this.#childRefs.get(runId);
          // Only for a run in a tree it did not open (a child the runner
          // started, or a turn continuing a saved tree): read back, a tree a
          // root opened would be continued as a saved tree that has no record.
          const member = RunTrees.memberOf(runId);
          const tree = member?.tree;
          if (member !== null && member.route !== 'open' && tree !== undefined && typeof tree === 'object') {
            row.harness_tree = { session: tree.session, run: tree.run };
          }
          if (refs?.planLedger != null) row.harness_plan_ledger = { ...refs.planLedger };
        }
      }
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

      for (const call of gated) this.#toolDecision('tool_approval_asked', session, runId, call, offered.find((tool) => tool.name === call.name), 'person');

      for (const call of toolCalls.filter((candidate) => !gated.includes(candidate))) {
        if (unusable.has(call)) {
          const entry = toolResultEntry(call, `Not run: the arguments for ${call.name} are not a JSON object.`);
          steps?.toolResult(call.id, entry.result, { ok: false, code: 'tool_failed', refused: false });
          results.push(entry);
          continue;
        }
        if (autoAllowed.has(call)) {
          // Checked again at the moment it would run: a revoke since the step
          // began refuses it rather than letting the old answer stand.
          const tool = offered.find((candidate) => candidate.name === call.name);
          if (tool === undefined || !this.#toolPolicy.allows(tool)) {
            this.#toolDecision('tool_denied', session, runId, call, tool, 'policy');
            const entry = toolResultEntry(call, 'Not run: this call needs approval, and the tool policy no longer allows it.');
            steps?.toolResult(call.id, entry.result, REFUSED_CALL);
            results.push(entry);
            continue;
          }
          this.#toolDecision('tool_auto_allowed', session, runId, call, tool, 'policy');
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
        // G-360: the run stops here for a person; a plan binds the approvals
        // this turn staged, as it does when a run finishes.
        await session.planService()?.afterTurn(session);

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

        const declared = bounded.find((tool) => tool.name === call.name);
        if (decision?.approved === true && work.unusable.has(call.id)) {
          const entry = toolResultEntry(input, `Not run: the arguments for ${call.name} are not a JSON object.`);
          steps?.toolResult(call.id, entry.result, { ok: false, code: 'tool_failed', refused: false });
          resolved.push(entry);
        } else if (decision?.approved === true && work.ambiguous.has(call.id)) {
          // Stored by 0.11.0 or earlier, which wrote no marker and stored a
          // bare number or boolean as no arguments. Not run, and recorded
          // without arguments; the model is told to call again, and a new
          // call is a new approval for the person.
          const entry = toolResultEntry({ ...input, arguments: {} }, approvalExpired(call.name));
          this.#toolDecision('tool_approval_expired', live, runId, { ...input, arguments: {} }, declared, 'policy');
          steps?.toolResult(call.id, entry.result, { ok: false, code: 'approval_denied', refused: true });
          resolved.push(entry);
        } else if (decision?.approved === true) {
          this.#toolDecision('tool_approved', live, runId, input, declared, 'person');
          called.push(call.name);
          const invoked = await this.#runApproved(bounded, input);
          steps?.toolResult(call.id, invoked.entry.result, invoked.outcome);
          resolved.push(invoked.entry);
        } else {
          // Denied, or never answered: either way not approved, and refused.
          this.#toolDecision('tool_denied', live, runId, input, declared, 'person');
          const entry = toolResultEntry(input, decision === undefined ? 'No approval response provided' : (decision.reason ?? 'User denied tool execution'));
          steps?.toolResult(call.id, entry.result, { ok: false, code: 'approval_denied', refused: true });
          resolved.push(entry);
        }
      }

      await live.thread().record([toolResultRow([...work.results.values(), ...resolved], [...work.decisions.values()])], runId);
      return bounded;
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
    scope: ReturnType<typeof approvalToolScope>;
    unusable: Set<string>;
    ambiguous: Set<string>;
  } | null> {
    const rows = (await session.thread().messages()).map((entry) => entry.message);
    const view = threadView(rows);
    let index = view.length - 1;

    while (index >= 0 && !(view[index]!.type === 'assistant' && (view[index] as AssistantRow).tool_calls.length > 0)) {
      index -= 1;
    }

    if (index === -1) return null;

    const assistant = view[index] as AssistantRow;
    const answered = view.slice(index + 1).find((row) => row.type === 'tool_result') as ToolResultRow | undefined;
    const results = new Map((answered?.tool_results ?? []).map((entry) => [entry.tool_call_id, entry]));
    // From the stored rows, not the replay view: the FIRST answer to an
    // approval is the answer, as the plan reads it too.
    const decisions = turnDecisions(rows);
    const approvalIds = new Map(assistant.tool_approval_requests.map((request) => [request.tool_call_id, request.approval_id]));

    // A call that has a result is DONE, whatever its decision says. A call that
    // needs nobody is not this method's to run.
    const calls = assistant.tool_calls.filter(
      (call) => !results.has(call.id) && (approvalIds.has(call.id) || mode.needsApproval(call.name)),
    );

    return calls.length === 0 ? null : { calls, results, decisions, approvalIds, scope: approvalToolScope(rows), unusable: unusableStoredCalls(rows), ambiguous: ambiguousCalls(assistant) };
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
      ...(session.agent === null ? {} : { harnessAgent: session.agent }),
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
  const ceiling: unknown = config.spend_ceiling;
  if (ceiling !== undefined && ceiling !== null && (typeof ceiling !== 'number' || !Number.isFinite(ceiling) || ceiling <= 0)) {
    throw new TypeError('Agent spend_ceiling must be a positive finite number of USD or null.');
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
  await writeDecisionsLocked(session, [{ approvalId, approved, reason }]);
}

/**
 * Answer SEVERAL pending approvals durably, as one row, before resuming once.
 *
 * The same writer as `recordApproval()` and `streamDecision()`. A run that
 * stopped on more than one call needs every answer before it resumes, because
 * a call with no decision is refused when it does.
 */
export async function recordDecisions(session: Session, decisions: readonly ApprovalDecision[]): Promise<void> {
  await writeDecisionsLocked(session, assertDecisions(decisions));
}

/**
 * The pending check and the write under the session lock, so two decisions
 * for one id cannot both pass the check: the second waits, finds the id
 * answered, and is refused with `approval_not_pending`. A caller that cannot
 * get the lock gets `session_locked` and nothing is recorded.
 * `streamDecision()` already holds this lock for its whole turn.
 */
async function writeDecisionsLocked(
  session: Session,
  decisions: readonly ApprovalDecision[],
  ttlSeconds: number = RESOLUTION_LOCK_SECONDS,
  waitSeconds: number = RESOLUTION_WAIT_SECONDS,
): Promise<void> {
  await session.lock((live) => writeDecisions(live, decisions), ttlSeconds, waitSeconds);
}

/**
 * Where decisions are made durable, for every path that takes them.
 *
 * Every decision must name an approval the session is waiting on (G-354). An
 * id the run never issued, an id already answered, or the same id twice in one
 * batch refuses the WHOLE batch with `approval_not_pending`, before anything
 * is written and so before any tool or provider call.
 *
 * Call it holding the session lock: the check and the write are one step.
 */
async function writeDecisions(session: Session, decisions: readonly ApprovalDecision[], context?: RunContext): Promise<void> {
  const rows = (await session.thread().messages()).map((entry) => entry.message);
  const pending = new Set(pendingApprovalIds(rows));
  for (const decision of decisions) {
    if (!pending.delete(decision.approvalId)) throw HarnessError.approvalNotPending(decision.approvalId);
  }
  // A decision is refused only when its resume could not act on it at all:
  // a shared plan budget, or the run tree a subagent child was saved with,
  // whose COST is spent. A spent STEP budget never refuses one (G-387):
  // acting on recorded decisions costs no model step, and the resume runs
  // the approved call and takes no further step.
  await session.planService()?.assertDecisionsFunded(session, decisions.map((decision) => decision.approvalId), context);
  const kept = (await session.childRefs()) ?? { tree: pausedRunState(rows)?.tree ?? null, planLedger: null };
  await RunTrees.assertDecisionsFunded(session, kept, context, session.planRuntime()?.holdTtl(session) ?? RESOLUTION_LOCK_SECONDS,
    await session.resumedRunTree());

  const run = await session.run();

  await session.thread().record([toolResultRow([], decisions.map((decision) => ({
    approval_id: decision.approvalId,
    approved: decision.approved,
    reason: decision.reason ?? null,
  })))], run?.id ?? null);

  // A plan held on one of these approvals reads the decision from the thread,
  // before the run resumes (G-360).
  await session.planService()?.decided(session);
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

/** The turn whose tools are running, for the code those tools call. */
const activeTurns = new AsyncLocalStorage<{ sessionKey: string; runId: string; run: RunContext }>();

/**
 * The run id and run context of the turn running on this session whose tool
 * is executing now, or null outside one. A subagent the turn delegates to
 * spends from this context's ledger, so from whatever the turn spends from:
 * the run tree and, for a plan step, the plan's shared budget.
 *
 * @internal Read by SubagentRunner.
 */
export function activeTurn(sessionKey: string): { runId: string; run: RunContext } | null {
  const turn = activeTurns.getStore();

  return turn === undefined || turn.sessionKey !== sessionKey ? null : { runId: turn.runId, run: turn.run };
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
