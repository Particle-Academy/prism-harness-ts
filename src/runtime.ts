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
  unresolvedApprovalIds,
  endedByStartRow,
  pairStepApprovals,
  repeatedStepSettled,
  stepResultsByPosition,
  pausedStepRepeatsCallIds,
  pausedStepIndex,
  REPEATED_CALL_ID,
  NOT_RUN_KEY,
  REPEATED_CALL_ID_RESULT,
  ARGUMENTS_AS_SENT,
} from './thread-rows.js';
import { usableArguments } from './tool-arguments.js';
import { admitAttachments } from './attachments.js';
import { HarnessError } from './errors.js';
import { errorCode, failureName, runFailure } from './failures.js';
import type { HarnessEvents, ToolDecisionEvent } from './events.js';
import { ToolPolicy, argsDigest } from './tool-policy.js';
import type { AgentMode, ModeRegistry } from './modes.js';
import { inOwnTurnScope, type ChildRefs, type Session } from './session.js';
import { schemaName, schemaProblems } from './structured.js';
import { MAX_DEPTH, RunBudget, RunContext, RunLedger, bindingRoute, type ModelBinding, type RunStop } from './subagents.js';
import { RunTrees, type RunTreeRef, type TreeMembership } from './run-trees.js';
import { RunStarts, StartRefused } from './run-starts.js';
import { RunBudgets, type AllowanceFacts } from './run-budgets.js';
import { PLAN_BUDGET_EXHAUSTED, type RunContextKind } from './plans/budget.js';
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
  /**
   * Checked against the schema before it got here; a run that could not
   * produce one threw. `null` when the run stopped before any document: it
   * parked on approval (`finishReason` `awaiting_approval`, with
   * `pendingApprovals`) or was stopped by a budget. Resume it with
   * `sendStructured(session, '', schema)` once the approvals are answered.
   */
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
  /** Each running run's hold release (#turnIn), so #finish releases BEFORE run.finished (round 13). */
  readonly #releases = new Map<string, () => Promise<void>>();
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
   * resumes, null for a turn that resumes nothing; `linked` says the session
   * is a subagent child (saved run tree references or a binding, on the
   * session or on its parked row). Returns the facts, and whether the turn
   * holds part of its delegation's step budget (settled when it ends).
   *
   * - A subagent child's ceiling is the runner's grant, saved when the
   *   runner started it: the delegation's budget for its whole life, until
   *   the runner starts it again. Every turn the runner did not start claims
   *   from it in one locked write (Session.claimChildSteps), so two turns on
   *   one child never both spend the same remainder. A linked session with
   *   no usable saved ceiling, or no usable count, fails closed.
   * - What a runner or host context has left is that context's own budget;
   *   a budget that is not a finite number has nothing left.
   * - The steps used are the most of: the delegation's count (live holds
   *   included), what the parked run wrote on the row it parked on
   *   (`harness_steps_used`, its children's steps included) and the model
   *   steps its own rows show. A malformed row count fails closed, as does
   *   a missing one on a child.
   */
  async #allowanceFacts(session: Session, runId: string, mode: AgentMode, kind: RunContextKind, execution: RunContext | undefined, resumed: ReturnType<typeof pausedRunState>, rows: readonly JsonObject[], linked: boolean, holdTtl: number): Promise<{ facts: AllowanceFacts; claimed: boolean; readSteps: number | null }> {
    const max: unknown = execution?.budget.maxSteps;
    // The context's steps as `contextLeft` is read from them: the run is
    // narrowed from THIS count, so a step spent on the context after the
    // read is counted against it (G-387 round 8).
    const readSteps = execution == null ? null : execution.ledger.steps;
    const saved = kind === 'runner' ? null : await session.childCeiling();
    const facts: AllowanceFacts = {
      kind,
      modeSteps: mode.maxSteps,
      savedCeiling: saved === 'invalid' || (saved === null && linked) ? 0 : saved,
      contextLeft: execution == null || readSteps === null ? null : (typeof max === 'number' && Number.isFinite(max) ? Math.max(0, max - readSteps) : 0),
      stepsCeiling: this.#setting(session, 'max_steps_ceiling') ?? null,
      used: 0,
      treeLeft: null,
      planLeft: null,
      took: 0,
    };
    if (kind === 'runner') return { facts, claimed: false, readSteps };
    const closed = (base: AllowanceFacts): AllowanceFacts => {
      const ceiling = RunBudgets.allowance({ ...base, used: 0 }).ceiling;

      return { ...base, savedCeiling: ceiling, used: ceiling };
    };
    // A session that saved a child's ceiling was written by a version that
    // also writes the row count: an absent one there is not a legacy row.
    const rowClosed = resumed !== null && (resumed.stepsUsed === 'invalid' || (resumed.stepsUsed === null && facts.savedCeiling !== null));
    const rowUsed = resumed === null || rowClosed ? 0 : Math.max(resumed.stepsUsed === 'invalid' ? 0 : resumed.stepsUsed ?? 0, approvalRunSteps(rows) ?? 0);
    if (facts.savedCeiling === null) return { facts: rowClosed ? closed(facts) : { ...facts, used: rowUsed }, claimed: false, readSteps };

    let claimedFacts: AllowanceFacts = closed(facts);
    await session.claimChildSteps(runId, Math.floor(Date.now() / 1000), holdTtl, (count) => {
      claimedFacts = count === null || rowClosed ? closed(facts) : { ...facts, used: Math.max(count, rowUsed) };

      return RunBudgets.allowance(claimedFacts).own_left;
    });

    return { facts: claimedFacts, claimed: true, readSteps };
  }

  /**
   * A runner start (G-387 round 7): READ every fact the start is decided
   * by, ask RunStarts.gate() once, and only on start commit, in order:
   * (a) the tree hold (and a plan hold, when one applies) in a conditional
   * locked write, refused if a race since the read shrank it; (b) the
   * child's records (binding, depth, ceiling with its count started again,
   * run tree and plan refs) in ONE locked write that re-checks a live turn
   * on the child, refused if a resume claimed it since; a refusal there
   * releases (a). Nothing before the gate writes: a refusal leaves every
   * record as it was. A read that fails throws, so the start is framed as
   * failed with nothing written. The run itself (c) begins in #turn.
   *
   * `host` is the context the subagent tool was wired with, when the child
   * runs in a parent turn's context instead: its cancel, time, steps and
   * cost stop the start too.
   */
  async #startRunner(session: Session, runId: string, prompt: string, mode: AgentMode, run: RunContext, membership: TreeMembership, context: RunContext, start: { mode: string | null; host: RunContext | null }, runnerBinding: ModelBinding | null, savedRefs: ChildRefs, holdTtl: number): Promise<RunnerStart> {
    const host = start.host;
    const plans = session.planService();
    const now = Math.floor(Date.now() / 1000);
    // -- read: nothing is written before the gate ------------------------------
    const { facts, readSteps } = await this.#allowanceFacts(session, runId, mode, 'runner', run, null, [], false, holdTtl);
    const own = RunBudgets.allowance(facts);
    const contexts = host === null || host === run || host === context ? [run] : [run, host];
    // A limit that is not a finite number counts as spent (G-387 round 8).
    const timeSpent = (c: RunContext): boolean => c.budget.maxSeconds !== null
      && !(Number.isFinite(c.budget.maxSeconds) && c.ledger.elapsedSeconds() < c.budget.maxSeconds);
    let stepsLeft = own.own_left;
    // NaN (a host budget that is not a number) reaches the gate as no steps.
    if (host !== null && contexts.length > 1) stepsLeft = Math.min(stepsLeft, Math.max(0, host.budget.maxSteps - host.ledger.steps));
    const ceiling = session.agent === null ? null : (this.#agents[session.agent]?.spend_ceiling ?? null);
    const agentSpent = session.agent !== null && ceiling !== null && run.ledger.agentCostUsd(session.agent) >= ceiling;
    const agentUnmetered = session.agent !== null && ceiling !== null && run.ledger.agentUnmeteredRuns(session.agent) > 0;
    const costSpent = agentSpent || contexts.some((c) => c.budget.maxCostUsd !== null && !(Number.isFinite(c.budget.maxCostUsd) && c.ledger.costUsd < c.budget.maxCostUsd));
    const unenforceable = agentUnmetered || contexts.some((c) => c.budget.maxCostUsd !== null && c.ledger.unmeteredRuns > 0);
    const busy = await session.childTurnLive(now, holdTtl);
    const rows = (await session.thread().messages()).map((entry) => entry.message);
    const pending = await pendingState(session, rows);
    const tree = await RunTrees.startGrant(session, membership);
    const wanted = Math.min(own.own_left, tree?.steps ?? Number.POSITIVE_INFINITY);
    const planLeft = plans === null ? null : await plans.startGrant(session, runId, run, context, prompt, wanted);
    const gate = RunStarts.gate({
      task: prompt,
      tooDeep: run.tooDeep() || context.tooDeep(),
      cancelled: contexts.some((c) => c.ledger.cancelled),
      timeSpent: contexts.some(timeSpent),
      stepsLeft,
      cost: costSpent ? 'spent' : unenforceable ? 'unenforceable' : 'ok',
      childBusy: busy,
      childPending: pending,
      treeLeft: tree?.steps ?? null,
      planLeft,
    });
    if ('refused' in gate) throw new StartRefused(gate.refused, gate.outcome);

    // -- commit (a): the tree hold, then a plan hold ---------------------------
    let treeHold: { opened: boolean } | null = null;
    if (tree !== null) {
      const held = await RunTrees.holdStart(session, tree.request, tree.steps, runId);
      if (held === 'refused') throw new StartRefused('tree', 'exhausted');
      treeHold = held;
    }
    let released = false;
    const release = async (): Promise<void> => {
      if (released) return;
      released = true;
      if (tree !== null && treeHold !== null) await RunTrees.releaseStart(session, tree.request, treeHold, runId);
      await plans?.release(session, runId);
    };
    try {
      let planned: { run: RunContext; planLeft: number | null } = { run, planLeft: null };
      if (plans !== null && planLeft !== null) {
        planned = await plans.planTurn(session, runId, run, context, prompt, wanted);
        if ((planned.planLeft ?? 0) < planLeft) throw new StartRefused('plan', 'exhausted');
      }
      // -- commit (b): the child's records and its mode, in one write ---------
      // It re-checks, under the child's lock, a turn holding its count and an
      // approval a person could answer that parked since the read (round 8).
      const refs = RunTrees.keptRefs(membership, membership.spends === null ? null : plans?.ledgerOf(membership.spends) ?? null,
        savedRefs.tree, savedRefs.planLedger, false);
      const committed = await session.rememberChildStart(runnerBinding, membership.depth, own.ceiling,
        { runId, steps: own.own_left, at: now }, holdTtl, refs, start.mode,
        async () => (await pendingState(session, (await session.thread().messages()).map((entry) => entry.message))) === 'answerable');
      if (committed === 'busy') throw new StartRefused('child_busy', 'denied');
      if (committed === 'waiting') throw new StartRefused('child_waiting', 'denied');
      const steps = run.ledger.steps, cost = run.ledger.costUsd, unmetered = run.ledger.unmeteredRuns;
      let undone = false;
      // A start that fails before its task row is recorded is undone: the
      // child's records and mode go back (compare-and-restore), its count
      // hold with them, and the holds of (a) are released while nothing has
      // been spent under them (else they settle what was, as any run's do).
      const undo = async (): Promise<void> => {
        if (undone) return;
        undone = true;
        try {
          await session.undoChildStart(committed, Math.floor(Date.now() / 1000), holdTtl);
        } finally {
          if (run.ledger.steps === steps && run.ledger.costUsd === cost && run.ledger.unmeteredRuns === unmetered) await release();
        }
      };

      return { run: planned.run, facts, readSteps: readSteps ?? run.ledger.steps, treeLeft: tree?.steps ?? null, planLeft: planned.planLeft, refs, recorded: runnerBinding !== null, undo };
    } catch (error) {
      await release();
      throw error;
    }
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
        // Each decision AS GIVEN, as the PHP reference streams it; what the
        // thread recorded (a repeated-id step's as denials) is in its rows.
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
    // The paused step (G-397), by the one selector.
    const step = pausedStepIndex(entries.map((entry) => entry.message));
    if (step === null) return null;
    const entry = entries[step]!;
    const message = entry.message;
    const requests = Array.isArray(message.tool_approval_requests) ? message.tool_approval_requests : [];
    if (!Array.isArray(message.tool_calls) || message.tool_calls.length === 0 || requests.length === 0 || entry.runId !== run.id) return null;
    const seq = run.last_seq;

    return { id: run.id, lastSeq: typeof seq === 'number' && Number.isSafeInteger(seq) && seq >= 0 ? seq : 0 };
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

    // A run that ended with a document carries it (#loop, checked against the
    // schema). One that stopped before any document (parked on approval, or
    // stopped by a budget) returns the same shape with `structured: null`,
    // its pending approvals beside it: never a missing field.
    if (Object.hasOwn(response, 'structured')) return response as StructuredAgentResponse;
    const shaped: StructuredAgentResponse = { ...response, structured: null };
    const stop = STOPS.get(response);
    if (stop !== undefined) STOPS.set(shaped, stop);

    return shaped;
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
    // Every turn has a scope of its own (round 11): a runner start's view of
    // its own records reaches no other turn, nested or not.
    return inOwnTurnScope(() => this.#turnIn(session, prompt, toolNames, context, additionalContent, schema, stepping));
  }

  async #turnIn(
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
    // A runner start runs in the mode the runner names; it is written on the
    // child only once the start has committed (G-387 round 7).
    const start = context != null && RunTrees.contextKind(context) === 'runner' ? RunStarts.startOf(context) : null;
    const mode = this.#modes.forAgent(session.agent).resolve(start?.mode ?? await session.mode());
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
    // A child whose records a runner start left provisional (no task row,
    // G-387 round 9) has NO binding: not its saved one, and not one its
    // paused row carries, so it is refused below.
    const legacyBinding = tail?.binding == null || await session.childStartUnconfirmed() ? null
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

    const holdTtl = this.#setting(session, 'lock_ttl') ?? RESOLUTION_LOCK_SECONDS;
    // G-387 round 7: a runner start reads every stop condition, asks ONE
    // gate, and commits its tree hold (a) and the child's records (b) before
    // its run begins (c). A refusal, from the gate or a commit race, has
    // written nothing. Once committed, a later stop only stops model steps.
    const started = start === null ? null
      : await this.#startRunner(session, runId, prompt, mode, run, membership, context!, start, runnerBinding, savedRefs, holdTtl);
    if (started === null && run.tooDeep()) {
      throw HarnessError.runNotPermitted(
        `This run is nested ${run.depth} deep, at or past the ceiling. Nobody debugs a tree that deep, ` +
          'and a configuration that produced one is a mistake worth reporting rather than executing.',
      );
    }
    if (started !== null) run = started.run;

    steps?.bind(session.key(), runId, run.parentRunId, run.rootRunId, session.agent);
    let unwatchPlan: () => void = () => {};
    try {
      // Inside this try (round 13): a start that committed (b) and fails
      // here, before its run began, ends its own view and is undone too.
      if (steps !== undefined && stepping?.watchPlan === true) unwatchPlan = await this.#watchPlan(session, steps);
      // A runner start wrote the child's mode with its records (b).
      // A runner start's run is provisional until its task row (round 9).
      await session.beginRun(runId, mode.name, provider, model, membership.tree, membership.depth, started !== null);
    } catch (error) {
      // The run never began: a runner start is undone (its child records,
      // mode and holds), and it fails as before.
      unwatchPlan();
      session.endChildStartView(runId);
      await started?.undo();
      throw error;
    }
    // A turn that holds part of its delegation's step budget (G-387): the
    // hold is replaced by what the run took when it ends, parks or fails.
    let delegation: { session: Session; startSteps: number } | null = null;
    // A runner start is undone by any failure before its task row is recorded.
    let taskRecorded = false;
    // The run's holds (its tree's and its delegation's), released ONCE: on
    // the failure path BEFORE run.failed is emitted (G-387 round 12), so a
    // listener never observes, or races, a hold about to be released.
    let holdsReleased = false;
    const releaseHolds = async (): Promise<void> => {
      if (holdsReleased) return;
      holdsReleased = true;
      // A root run that holds part of a run tree's budget settles it.
      await RunTrees.finished(session, runId);
      if (delegation !== null) await settleDelegation(delegation.session, runId, run.ledger.steps - delegation.startSteps, this.#setting(session, 'lock_ttl') ?? RESOLUTION_LOCK_SECONDS);
    };
    // On success too (round 13): #finish releases before run.finished, so
    // no listener observes a hold of a run that has already ended.
    this.#releases.set(runId, releaseHolds);
    try {
      // Inside the run's try (G-387 round 12): whatever this emit throws (a
      // host's own HarnessEvents may throw anything) fails the run, releases
      // its holds and emits the terminal event, like any other failure.
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
      // A NEW prompt on a parked session under a context that was cancelled
      // or whose time is spent is refused: it would act on none of the parked
      // turn's decisions, and its user row would end that turn, so they would
      // never be acted on. (On a session with nothing parked it stops as
      // before.) Not for a runner start: its one gate refused a cancelled or
      // timed-out start before anything was written, and once it committed a
      // later stop only stops its model steps (G-387 round 7).
      if (started === null && prompt !== '' && scope !== null && (run.ledger.cancelled || (run.budget.maxSeconds !== null && run.ledger.elapsedSeconds() >= run.budget.maxSeconds))) {
        throw HarnessError.runNotPermitted('This run may not start: its run context was cancelled or has no time left.');
      }
      // G-387: how many model steps this run may take is decided ONCE, here,
      // beside its membership, and recorded with it (RunTrees.allowed); the
      // run is narrowed to it once, and every consumer reads that record or
      // that budget. A turn resuming a parked run continues the count the
      // parked run wrote (its children's steps included), never below the
      // steps its own rows show; a runner start is a new run.
      // A subagent child: saved references or a saved binding, or a parked
      // row that names a run tree, or names a binding on a child's address (a
      // root under a host-supplied binding writes one on its row too).
      // Any address the runner builds (`::sub::`) is a child's, whatever else
      // is saved: a child a released version parked saved no child fields at
      // all, and with no saved ceiling it fails closed.
      const linked = session.scope.includes('::sub::') || (await session.childBinding()) !== null
        || (await session.childRefs()) !== null || tail?.tree != null;
      const { facts, claimed, readSteps } = started !== null ? { facts: started.facts, claimed: false, readSteps: started.readSteps }
        : await this.#allowanceFacts(session, runId, mode, contextKind, execution, prompt === '' ? tail : null, stored, linked, holdTtl);
      // A turn with no run context continues the count on its own ledger.
      if (execution == null) run.ledger.recordSteps(facts.used);
      const startSteps = run.ledger.steps;
      // The count its context's remainder was read at (G-387 round 8): a
      // step spent on a shared context since is counted, never re-added.
      const narrowBase = execution == null || readSteps === null ? startSteps : Math.min(readSteps, startSteps);
      if (claimed) delegation = { session, startSteps };
      const own = RunBudgets.allowance(facts);
      // A root run opens a tree; any turn on a child session the runner did
      // not start continues the run tree it was started in, holding what it
      // may spend of it (G-366).
      const held = started !== null ? { run, treeLeft: started.treeLeft }
        : await RunTrees.holdTree(session, runId, run, membership, holdTtl, own.own_left);
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
        // A runner start wrote them when it committed (#startRunner).
        const refs: ChildRefs = started !== null ? started.refs
          : { tree: typeof membership.tree === 'object' ? membership.tree : null, planLedger: savedRefs.planLedger };
        this.#childRefs.set(runId, refs);
        if (started?.recorded === true) delegation = { session, startSteps };
      }
      // G-366: a turn serving a plan step with a shared budget holds what the
      // plan has left of the steps its own and its tree's allowance leave it.
      const planned = started !== null ? { run, planLeft: started.planLeft } : plans === null ? { run, planLeft: null }
        : await plans.planTurn(session, runId, run, context, prompt, Math.min(own.own_left, held.treeLeft ?? Number.POSITIVE_INFINITY));
      run = planned.run;
      const allowance = RunBudgets.allowance({ ...facts, treeLeft: held.treeLeft, planLeft: planned.planLeft });
      // Narrowed ONCE: the loop's step gate, a subagent's pre-check and every
      // hold it takes read this budget. A host or runner context can only
      // lower what it offered, never raise it.
      run = run.withBudget(new RunBudget(narrowBase + allowance.allowance, run.budget.maxCostUsd, run.budget.maxSeconds, run.budget.maxOutputTokensPerCall));
      RunTrees.allowed(runId, { ...facts, treeLeft: held.treeLeft, planLeft: planned.planLeft }, startSteps, run);
      // A NEW prompt resumes no recorded decisions: with no step to take it
      // is refused before anything is recorded. Only a resume (an empty
      // prompt) left no step acts on its decisions without a model step.
      // A runner start that committed is never refused here: its loop's
      // step gate stops it, with its task recorded.
      if (started === null && prompt !== '' && allowance.acts_without_step) {
        const planIsZero = allowance.plan_left === 0 && allowance.own_left > 0 && allowance.tree_left !== 0;
        throw planIsZero
          ? new HarnessError(PLAN_BUDGET_EXHAUSTED, 'This plan step may not proceed (plan_budget_exhausted): the shared plan budget has nothing left for it.')
          : HarnessError.runNotPermitted('This run may not start: its step budget has nothing left (its own ceiling, its run tree or a cost budget is spent).');
      }
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
        // rather than after the new turn. Never in a run tree that was
        // cancelled, nor under a context whose time is spent: an approved
        // call does not run after its tree was stopped or its time ran out
        // (the loop then stops the run before any model step), and the
        // decisions are left recorded for a later resume that can act.
        //
        // A RUNNER START acts on none of them (G-387 round 9): its gate only
        // lets it start onto approvals nobody could answer, and a decision
        // recorded on one since (or while an earlier binding was alive) is
        // not this delegation's to act on. Its task row ends them: each
        // pending call is recorded as not run, in the same write as the task.
        const outOfTime = turn.budget.maxSeconds !== null && turn.ledger.elapsedSeconds() >= turn.budget.maxSeconds;
        const ended = started !== null ? await this.#endApprovals(session, mode, runId, steps) : null;
        if (started === null && !turn.ledger.cancelled && !outOfTime) offered = await this.#resolveApprovals(session, mode, offered, runId, called, scope?.legacy === true, steps, stepping?.locked === true);

        if (prompt !== '') {
          // With attachments, the shape prism-ts's UserMessage.toObject() writes:
          // the media parts, then the turn's own text as a trailing text part,
          // which fromObject() strips back off. Without them, unchanged.
          const message: JsonObject =
            attachments.length === 0
              ? { type: 'user', content: prompt }
              : { type: 'user', content: prompt, additional_content: [...attachments, { text: prompt }], additional_attributes: {} };

          await thread.record([...(ended?.rows ?? []), message], runId);
          // Recorded the moment its row is (round 11), before any event: a
          // listener that throws or re-enters never sees a recorded task as
          // unrecorded, and a start whose task row exists is never undone.
          taskRecorded = true;
          // Only once those rows are stored (round 10): a write that failed
          // reports no denial that was never recorded.
          ended?.announce();
        }
        taskRecorded = true;
        // The task row CONFIRMS a runner start's records (round 9); clearing
        // their provisional marks is housekeeping only, and its failure
        // changes nothing.
        if (started !== null) {
          try {
            await session.confirmChildStart(runId);
          } catch {
            // The task row is the authority; the marks read as confirmed.
          }
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
      //
      // NOR ANY EXCEPTION'S OWN WORDS (G-387 round 8): this harness has no
      // content-capture setting, so the run row, run state and event carry
      // a fixed description with the class name and harness code only; the
      // exception itself still reaches the caller.
      //
      // Cleanup FIRST, and nothing read from the error before it (round 9).
      // A runner start that never recorded its task row is provisional: its
      // records already read as absent, so the undo is housekeeping only and
      // its failure changes nothing; its holds expire.
      // The start's own view ends first (round 11): from here its Session
      // reads its records by the provisional rule, as every other reader.
      session.endChildStartView(runId);
      if (started !== null && !taskRecorded) {
        try {
          await started.undo();
        } catch {
          // Housekeeping only; the failure below is the run's.
        }
      }
      const failure = documentCode(error) ?? runFailure(error);

      try {
        await session.failRun(runId, failure);
      } catch {
        // The event is still emitted and the error still reaches the caller.
      }
      // Released BEFORE the event (round 12): settling never throws (it
      // reports by class name), and the finally does not settle again.
      try {
        await releaseHolds();
      } catch {
        // Each settle reports its own failure; the event is still emitted.
      }
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
      session.endChildStartView(runId);
      unwatchPlan();
      this.#childRefs.delete(runId);
      this.#releases.delete(runId);
      await releaseHolds();
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
  /**
   * The rows a runner start's task row ends its child's pending calls with
   * (G-387 round 9): each one NOT RUN, whatever decision is recorded on it.
   * Nothing is invoked; the rows are written with the task row, in one write.
   */
  async #endApprovals(session: Session, mode: AgentMode, runId: string, steps?: RunStepTagger): Promise<{ rows: JsonObject[]; announce: () => void }> {
    const work = await this.#unresolved(session, mode);
    if (work === null) return { rows: [], announce: () => {} };
    const inputs = work.calls.map((call) => ({ id: call.id, name: call.name, arguments: work.ambiguous.has(call.id) ? {} : call.arguments, resultId: call.result_id }));
    // ONE row, in the reference's shape (round 12): the ended calls, each
    // with the one fixed result, and the row marked with the approvals they
    // were asked under, each call's OWN approval, paired by position (G-392).
    // A tool controls only its `result` string, and the replay view and
    // every builder rebuild rows from their known keys, so no tool result,
    // argument or model output can produce the mark.
    const row = endedByStartRow(inputs, inputs.map((_input, index) => work.approvalIds[index]));

    return {
      rows: [row as unknown as JsonObject],
      // The events and run steps for them, emitted by the caller only once
      // the rows are recorded with the task row (G-387 round 10).
      announce: () => {
        inputs.forEach((input, index) => {
          this.#toolDecision('tool_denied', session, runId, input, undefined, 'policy');
          steps?.toolResult(input.id, row.tool_results[index]!.result, { ok: false, code: 'approval_denied', refused: true });
        });
      },
    };
  }

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

      for (const [index, call] of work.calls.entries()) {
        const input = { id: call.id, name: call.name, arguments: call.arguments, resultId: call.result_id };
        // The call's OWN approval, paired by position (G-392).
        const approvalId = work.approvalIds[index];
        const decision = approvalId === undefined ? undefined : work.decisions.get(approvalId);

        const declared = bounded.find((tool) => tool.name === call.name);
        if (work.repeated) {
          // The step repeats a call id, so no decision can be told apart from
          // another (G-392): none of its gated calls runs, whatever was
          // decided. Marked, so the step reads settled and no later decision
          // or resume revives it, and the plan never reads it as approved.
          this.#toolDecision('tool_denied', live, runId, input, declared, 'policy');
          const entry = { ...toolResultEntry(input, REPEATED_CALL_ID_RESULT), [NOT_RUN_KEY]: REPEATED_CALL_ID, ...(approvalId === undefined ? {} : { approval_id: approvalId }) };
          steps?.toolResult(call.id, entry.result, { ok: false, code: 'approval_denied', refused: true });
          resolved.push(entry);
        } else if (decision?.approved === true && work.unusable.has(call.id)) {
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
    /** Each call's own approval id, by position in `calls` (G-392). */
    approvalIds: (string | undefined)[];
    /** The step repeats a tool call id: none of its gated calls may run (G-392). */
    repeated: boolean;
    scope: ReturnType<typeof approvalToolScope>;
    unusable: Set<string>;
    ambiguous: Set<string>;
  } | null> {
    const rows = (await session.thread().messages()).map((entry) => entry.message);
    // THE paused step (G-397): the one selector every decision reader uses.
    const stored = pausedStepIndex(rows);
    if (stored === null || list(rows[stored]!.tool_calls).length === 0) return null;
    const view = threadView(rows);
    // The same row in the replay view (which folds a 0.3.0 request row into it).
    const ordinal = rows.slice(0, stored + 1).filter((row) => row.type === 'assistant').length;
    const index = view.findIndex((row, at) => row.type === 'assistant' && view.slice(0, at + 1).filter((candidate) => candidate.type === 'assistant').length === ordinal);
    if (index === -1) return null;

    const assistant = view[index] as AssistantRow;
    // Every result row of the step's own turn (G-399), not only the first run
    // of them: what is carried into the resolution row.
    const next = view.findIndex((row, at) => at > index && row.type === 'assistant');
    const results = new Map(view.slice(index + 1, next === -1 ? undefined : next)
      .filter((row): row is ToolResultRow => row.type === 'tool_result')
      .flatMap((row) => row.tool_results).map((entry) => [entry.tool_call_id, entry]));
    // From the stored rows, not the replay view: the FIRST answer to an
    // approval is the answer, as the plan reads it too.
    const decisions = turnDecisions(rows);
    // Each call paired with its OWN request by position, never by a map keyed
    // by call id, where the last request for a repeated id would decide every
    // call carrying it (G-392).
    // Results pair with calls BY POSITION (round 13), never by id: in a step
    // whose calls share an id, an ungated call's result (or the immediate
    // result of a call with unusable arguments) must not make the gated call
    // look finished, nor take its approval.
    const answeredAt = stepResultsByPosition(rows, stored);
    // A request pairs only with a call that has no result of its own: the
    // calls the step ran at once (ungated, or with unusable arguments) never
    // take an approval. (Not the mode's list: a plan tool gates itself.)
    const pairing = pairStepApprovals(assistant.tool_calls, assistant.tool_approval_requests, (_call, position) => !answeredAt[position]);
    // A step that repeats a call id: its gated calls are open until the
    // harness settled the step.
    const settled = pairing.repeated && repeatedStepSettled(rows, stored);

    // A call that has a result is DONE, whatever its decision says. A call that
    // needs nobody is not this method's to run.
    const open = assistant.tool_calls
      .map((call, index) => ({ call, approvalId: pairing.approvalIds[index], index }))
      .filter(({ call, approvalId, index }) => (approvalId !== undefined || mode.needsApproval(call.name))
        && !answeredAt[index] && !settled);
    const calls = open.map(({ call }) => call);
    const approvalIds = open.map(({ approvalId }) => approvalId);

    return calls.length === 0 ? null : { calls, results, decisions, approvalIds, repeated: pairing.repeated, scope: approvalToolScope(rows), unusable: unusableStoredCalls(rows), ambiguous: ambiguousCalls(assistant) };
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
      entry: toolResultEntry(call, `The tool failed: ${toolFailureText(error)}`),
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
    // Released BEFORE the event, as on the failure path (round 13): settling
    // never throws (each settle reports its own failure by class name).
    try {
      await this.#releases.get(runId)?.();
    } catch {
      // The hold stays and only over-charges until it expires.
    }
    const stop = stoppedBecause === null ? null : run.ledger.stopCode(run.budget);
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

    const response: AgentResponse = {
      runId,
      text,
      steps: run.ledger.steps,
      toolCalls: called,
      finishReason,
      pendingApprovals: [],
      stoppedBecause,
    };
    if (stop !== null) STOPS.set(response, stop);

    return response;
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

/** A runner start that committed its holds and the child's records (#startRunner). */
interface RunnerStart {
  run: RunContext;
  facts: AllowanceFacts;
  treeLeft: number | null;
  planLeft: number | null;
  refs: ChildRefs;
  /** The context's steps when the start's facts were read: the run is narrowed from it. */
  readSteps: number;
  /** Whether the child's records (and its count's hold) were written. */
  recorded: boolean;
  /** Undo a start that failed before its task row: its child records, mode and holds. */
  undo: () => Promise<void>;
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
  await session.lock(async (live) => { await writeDecisions(live, decisions); }, ttlSeconds, waitSeconds);
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
async function writeDecisions(session: Session, decisions: readonly ApprovalDecision[], context?: RunContext): Promise<ApprovalDecision[]> {
  // A decision is never inside a runner start's own view (round 11).
  return inOwnTurnScope(() => writeDecisionsIn(session, decisions, context));
}

/** Writes the decisions and returns them as RECORDED (a repeated-id step's as denials). */
async function writeDecisionsIn(session: Session, given: readonly ApprovalDecision[], context?: RunContext): Promise<ApprovalDecision[]> {
  const rows = (await session.thread().messages()).map((entry) => entry.message);
  const pending = new Set(pendingApprovalIds(rows));
  for (const decision of given) {
    if (!pending.delete(decision.approvalId)) throw HarnessError.approvalNotPending(decision.approvalId);
  }
  // A step whose call ids repeat runs none of its gated calls (G-392), so
  // every decision on it is RECORDED as a denial with that reason (round
  // 13, as the PHP reference records it): what the thread says was decided
  // is what was acted on, and no reader can take it for approved work.
  const decisions: ApprovalDecision[] = pausedStepRepeatsCallIds(rows)
    ? given.map((decision) => ({ approvalId: decision.approvalId, approved: false, reason: REPEATED_CALL_ID_RESULT }))
    : [...given];
  // No budget, steps or cost, refuses a decision (G-387): acting on an
  // approved call makes no provider call, so the resume runs it and takes no
  // model step. A decision the resuming turn could not act on at all is
  // refused here, with nothing written; so is a plan step handed a run tree
  // of its own.
  await assertActionable(session, rows, context, decisions.every((decision) => !decision.approved));
  await session.planService()?.assertDecisionsFunded(session, decisions.map((decision) => decision.approvalId), context);

  const run = await session.run();

  await session.thread().record([toolResultRow([], decisions.map((decision) => ({
    approval_id: decision.approvalId,
    approved: decision.approved,
    reason: decision.reason ?? null,
  })))], run?.id ?? null);

  // A plan held on one of these approvals reads the decision from the thread,
  // before the run resumes (G-360).
  await session.planService()?.decided(session);

  return decisions;
}

/**
 * Refuse a decision the turn that resumes it could not act on, with the
 * checks that turn runs, before anything is recorded (G-387): a subagent
 * child whose binding is missing, a run nested too deep, and a supplied run
 * context whose ledger was cancelled or (for a batch with an approval in it)
 * whose time is spent. A recorded
 * decision no turn can act on would be an approval accepted and never run.
 * (This port has no per-run tool-call limit to check.)
 *
 * @throws HarnessError `child_binding_missing` or `run_not_permitted`
 */
/**
 * True when a person could still answer the approvals `session` is parked on:
 * a decision on them, with no run context, would be accepted (G-387). The
 * subagent runner reads it before it starts a parked child again.
 *
 * @internal Read by SubagentRunner.
 */
export async function decisionsAnswerable(session: Session): Promise<boolean> {
  // ONLY the known reasons nobody could answer make an approval unanswerable
  // (G-387 round 8). Any read that fails throws, so the start that asked is
  // refused (failed) rather than ending an approval a person could answer.
  return await actionability(session, (await session.thread().messages()).map((entry) => entry.message), undefined) === null;
}

/**
 * A parked child's approvals, as the start gate and (b)'s locked re-check
 * read them. A decision a person recorded that no resume has acted on yet
 * counts as an approval a person can answer (G-387 round 10): a runner start
 * would end its call as not run, so a model re-delegating could void a
 * person's decision. While a resume could still act on it, the start is
 * refused `child_waiting`, exactly as for an undecided one.
 */
async function pendingState(session: Session, rows: readonly JsonObject[]): Promise<'none' | 'answerable' | 'unanswerable'> {
  if (pendingApprovalIds(rows).length === 0 && unresolvedApprovalIds(rows).length === 0) return 'none';

  return await decisionsAnswerable(session) ? 'answerable' : 'unanswerable';
}

async function assertActionable(session: Session, rows: readonly JsonObject[], context: RunContext | undefined, onlyDenials = false): Promise<void> {
  const refusal = await actionability(session, rows, context, onlyDenials);
  if (refusal === null) return;
  if (refusal.reason === 'binding') throw HarnessError.childBindingMissing();
  if (refusal.reason === 'depth') throw HarnessError.runNotPermitted(`This decision may not be recorded: its run would be nested ${refusal.depth} deep, at or past the ceiling.`);
  if (refusal.reason === 'cancelled') throw HarnessError.runNotPermitted('This decision may not be recorded: the run tree it would resume in was cancelled.');
  throw HarnessError.runNotPermitted('This decision may not be recorded: the run it would resume in has no time left.');
}

/**
 * Why the turn that resumes `rows`' decisions could not act on them, or null
 * when it could: one of the KNOWN reasons only (no binding, too deep, a
 * cancelled tree, no time left). A read that fails is not a reason: it throws.
 */
async function actionability(session: Session, rows: readonly JsonObject[], context: RunContext | undefined, onlyDenials = false): Promise<{ reason: 'binding' | 'depth' | 'cancelled' | 'time'; depth: number } | null> {
  const execution = executionContext(context);
  const tail = RunTrees.contextKind(context) === 'runner' ? null : pausedRunState(rows);
  // A provisional child record (G-387 round 9) has no binding, its paused
  // row's included: a decision on it is refused.
  const savedBinding = (await session.childBinding()) ?? (await session.childStartUnconfirmed() ? null : tail?.binding ?? null);
  const saved = await session.childRefs();
  const route = bindingRoute(RunTrees.bindingContextKind(execution), execution?.binding != null, savedBinding !== null, saved !== null);
  if (route.binding === 'refused') return { reason: 'binding', depth: 0 };
  const membership = RunTrees.membershipOf(session, 'decision', context, saved ?? { tree: tail?.tree ?? null, planLedger: null }, await session.childDepth());
  if (membership.depth >= MAX_DEPTH) return { reason: 'depth', depth: membership.depth };
  if (execution === undefined) return null;
  if (execution.ledger.cancelled) return { reason: 'cancelled', depth: membership.depth };
  // Acting on a denial runs nothing, so a batch of denials only is not
  // refused for a spent time limit; a cancelled context still refuses it,
  // and a batch with an approval in it keeps the check.
  const seconds = execution.budget.maxSeconds;
  if (!onlyDenials && seconds !== null && execution.ledger.elapsedSeconds() >= seconds) return { reason: 'time', depth: membership.depth };

  return null;
}

/** What a response stopped on, by code, for the subagent runner (never the host's cancel words). */
const STOPS = new WeakMap<AgentResponse, RunStop>();

/**
 * The fixed code of the stop a run's response reports (RunLedger.stopCode),
 * or null when it did not stop on one.
 *
 * @internal Read by SubagentRunner.
 */
export function stopCodeOf(response: AgentResponse): RunStop | null {
  return STOPS.get(response) ?? null;
}

/**
 * What the model is told a tool's failure was: its message, as before (a
 * tool result goes to the model by design, so it can recover), by a
 * conversion that cannot throw (G-387 round 12). A value String() cannot
 * convert, or a throwing message getter, reads as its class by the
 * named-class rule; it never turns a tool's failure into a broken run.
 */
function toolFailureText(error: unknown): string {
  try {
    const text: unknown = error instanceof Error ? error.message : String(error);
    return typeof text === 'string' ? text : failureName(error);
  } catch {
    return failureName(error);
  }
}

/** A structured run's failure records its CODE (it holds the model's document), guarded; else null. */
function documentCode(error: unknown): string | null {
  try {
    return error instanceof HarnessError && error.document !== undefined ? errorCode(error) ?? 'error' : null;
  } catch {
    return null;
  }
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
 * Replace a turn's hold on its delegation's step budget with what it took.
 * Isolated: a failed write leaves the hold, which is charged in full once
 * the lock TTL has passed, so a crash can only over-charge the delegation.
 */
async function settleDelegation(session: Session, runId: string, took: number, ttl: number): Promise<void> {
  // Tried twice (round 13): one failed read at the end of a completed run
  // must not leave its delegation over-charged, which would refuse the
  // child's next turn. A settle is idempotent (a hold settled once is gone,
  // and a hold already charged in full is corrected to what was taken), so
  // a second try can never charge twice.
  for (let attempt = 1; ; attempt += 1) {
    try {
      await session.settleChildSteps(runId, took, Math.floor(Date.now() / 1000), ttl);
      return;
    } catch (error) {
      if (attempt >= 2) {
        process.emitWarning(`A delegation step budget update failed: ${failureName(error)}.`);
        return;
      }
    }
  }
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
    // The class by the named-class rule, never the error's own words (round 11).
    process.emitWarning(`prism-harness could not record the step sequence of run [${runId}]: ${failureName(error)}.`);
  }
}

function list(value: JsonValue | undefined): JsonObject[] {
  return Array.isArray(value) ? value.filter((entry): entry is JsonObject => typeof entry === 'object' && entry !== null && !Array.isArray(entry)) : [];
}
