/**
 * Failures carry a stable CODE.
 *
 * The PHP reference identifies a failure by its exception class and an English
 * sentence. A class name does not survive a port and a sentence is not a
 * contract, so the code is what a consumer branches on here: treat
 * `HarnessError.code` as stable and `HarnessError.message` as free to change in
 * any release. Same decision, and the same reasoning, as `prism-ts`.
 */
export type HarnessErrorCode =
  /** A lock on a session key could not be acquired before the wait expired. */
  | 'session_locked'
  /** A store that reports itself volatile was configured for durable state. */
  | 'unsafe_state_configuration'
  /** A store slot names a driver nothing is registered under. */
  | 'unknown_store_driver'
  /** A tool was asked for that this session cannot reach. */
  | 'tool_not_available'
  /** A run was refused — by budget, by depth, or by a cancelled ledger. */
  | 'run_not_permitted'
  /** A stored payload could not be mapped back into a value object. */
  | 'unmappable_content'
  /** A session was asked to do something that needs a runtime it does not have. */
  | 'no_agent_runtime'
  /** A mode was asked for that the application has not configured. */
  | 'mode_not_configured'
  /** A configured mode is malformed, or names something that does not exist. */
  | 'mode_malformed'
  /** A tool policy is defined but the authorizer that would consult it is off. */
  | 'unsafe_authorization_configuration'
  /** A tool call was refused by the call-time policy. */
  | 'call_not_authorized'
  /** A skill file was asked for that would resolve outside its own skill. */
  | 'skill_path_refused'
  /** A task was added under an id the source already holds. */
  | 'duplicate_task_id'
  /** A worker id or a task id was the empty string. */
  | 'task_identifier_blank'
  /** A lease or extension that was not a whole positive number of seconds. */
  | 'task_lease_invalid'
  /** A task id was named that its source does not hold. */
  | 'task_not_found'
  /** A task that is already `done` or `failed` was released again. */
  | 'task_already_terminal'
  /**
   * A worker acted on a task it is not holding.
   *
   * ONE code for every shape of "not yours": nobody holds it, someone else
   * does, or your own lease has expired. Decision 0020 would rather see those
   * split, and the first draft here did split them -- but decision 0004 pins
   * codes ACROSS LANGUAGES, and the Python port settled on one. A consumer
   * branching on a code that exists in one port and not another is worse than a
   * coarse code that means the same thing everywhere. The sentence still says
   * which case it was.
   */
  | 'task_lease_not_held'
  /**
   * A completion tool was called with an outcome that is not `done` or `failed`.
   *
   * NOT in the set the Python port settled, and raised rather than dropped --
   * see `taskOutcomeInvalid` for why coercing instead is a security defect.
   */
  | 'task_outcome_invalid';

export interface HarnessErrorOptions {
  cause?: unknown;
}

export class HarnessError extends Error {
  readonly code: HarnessErrorCode;

  constructor(code: HarnessErrorCode, message: string, options: HarnessErrorOptions = {}) {
    super(message, options);
    this.name = 'HarnessError';
    this.code = code;
  }

  static sessionLocked(key: string, waitSeconds: number): HarnessError {
    return new HarnessError(
      'session_locked',
      `Could not acquire the lock on session [${key}] within ${waitSeconds}s. ` +
        'Another worker is holding it; the callback was NOT run.',
    );
  }

  /**
   * The guard this package exists for.
   *
   * A cache is disposable by definition, and the durable slot holds pending
   * tool approvals — a half-executed action waiting on a human. Losing one is a
   * correctness failure, not a cache miss, so a store that reports itself
   * volatile is refused here rather than accepted and discovered later.
   */
  static volatileDurableStore(slot: string, driver: string): HarnessError {
    return new HarnessError(
      'unsafe_state_configuration',
      `The [${slot}] slot is configured with the [${driver}] driver, which reports itself VOLATILE. ` +
        'Durable state (threads, pending tool approvals) must survive a deploy. Either point this ' +
        'slot at a durable driver, or — if this store really does persist — declare it durable when ' +
        'you register it. That declaration is an assertion about your infrastructure, not a preference.',
    );
  }

  static unknownStoreDriver(slot: string, driver: string): HarnessError {
    return new HarnessError(
      'unknown_store_driver',
      `The [${slot}] slot names the driver [${driver}], which is not registered.`,
    );
  }

  static toolNotAvailable(name: string, available: readonly string[]): HarnessError {
    return new HarnessError(
      'tool_not_available',
      `The tool [${name}] is not available to this session (has: ${available.join(', ') || 'none'}).`,
    );
  }

  static runNotPermitted(reason: string): HarnessError {
    return new HarnessError('run_not_permitted', reason);
  }

  static unmappableContent(description: string, options: HarnessErrorOptions = {}): HarnessError {
    return new HarnessError(
      'unmappable_content',
      `Could not rebuild stored content: ${description}.`,
      options,
    );
  }

  static modeNotConfigured(name: string): HarnessError {
    return new HarnessError('mode_not_configured', `Harness mode [${name}] is not configured.`);
  }

  static modeMalformed(name: string, detail: string): HarnessError {
    return new HarnessError('mode_malformed', `Harness mode [${name}] is malformed: ${detail}.`);
  }

  /**
   * Both at once is the one configuration not to leave in place.
   *
   * A defined policy that is never consulted looks like a control to every
   * reader and is not one -- every registered tool is offered to every run
   * while the code says otherwise.
   */
  static policyDefinedButDisabled(): HarnessError {
    return new HarnessError(
      'unsafe_authorization_configuration',
      'A tool authorization policy was supplied, but the authorizer is disabled, so that policy is ' +
        'never consulted and every registered tool is offered to every run. Either enable the ' +
        'authorizer, or remove the policy so nothing suggests tool access is being restricted.',
    );
  }

  /**
   * Thrown rather than returned as a tool result.
   *
   * A refusal handed back as a result reads to the model as a failure it might
   * retry differently, and a denied action being retried is the opposite of
   * what a guard is for.
   */
  static callNotAuthorized(tool: string): HarnessError {
    return new HarnessError(
      'call_not_authorized',
      `This call to [${tool}] was refused by the tool authorization policy.`,
    );
  }

  /**
   * A skill name or path that would leave the skill directory.
   *
   * Refused rather than sanitised. Silently rewriting a traversal to something
   * safe teaches a caller — or a model — that the request was fine.
   */
  static skillPathRefused(detail: string): HarnessError {
    return new HarnessError('skill_path_refused', `Refused to read a skill file: ${detail}.`);
  }

  /**
   * A task list pointed at a store whose contents can vanish.
   *
   * The SAME code as `volatileDurableStore`, deliberately. It is the same
   * misconfiguration reaching the same conclusion, and a consumer branching on
   * "durable state was pointed somewhere it cannot live" should not have to
   * know which feature noticed. Prose is free per prism-parity decision 0004;
   * the code is not.
   *
   * A half-finished task list that vanishes on a deploy is indistinguishable
   * from a finished one, which is why this refuses at construction rather than
   * degrading to a default.
   */
  static volatileTaskSource(): HarnessError {
    return new HarnessError(
      'unsafe_state_configuration',
      'An agent task list was pointed at a store that reports itself VOLATILE. The list is durable ' +
        'state: losing it is a correctness failure, because a half-finished list that vanishes on a ' +
        'deploy looks exactly like a finished one. Point it at a durable store, or -- if this store ' +
        'really does persist -- declare it durable when you register it. That declaration is an ' +
        'assertion about your infrastructure, not a preference.',
    );
  }

  /**
   * A lease that is not a whole positive number of seconds. REFUSED, not adjusted.
   *
   * Both halves were learned the hard way. Zero or less was CLAMPED to one
   * second at first, because a clamp fails closed; a positive fraction was then
   * still TRUNCATED. Both are a value quietly becoming a different value, and
   * this repository has already shipped a configuration that did exactly that
   * and stayed green throughout. A caller asking for a zero-second or
   * fractional lease has a bug, and the useful thing to hand back is the bug.
   *
   * Takes `unknown` rather than `number` deliberately: the value that reaches
   * this is whatever actually arrived, including a string out of a JSON config
   * that the type annotation never saw.
   */
  static taskLeaseInvalid(seconds: unknown): HarnessError {
    return new HarnessError(
      'task_lease_invalid',
      `A lease of [${String(seconds)}] seconds was asked for. It must be a whole number of seconds, ` +
        'at least one. A fraction cannot be honoured -- the expiry is an integer timestamp -- and a ' +
        'lease that has already expired when it is granted hands the same task to a second worker.',
    );
  }

  static taskNotFound(id: string): HarnessError {
    return new HarnessError('task_not_found', `No task with the id [${id}] is in this source.`);
  }

  static duplicateTaskId(id: string): HarnessError {
    return new HarnessError(
      'duplicate_task_id',
      `A task with the id [${id}] is already in this source. Ids are unique within a source, and ` +
        'reusing one would let two units of work share a claim.',
    );
  }

  /**
   * An empty identifier, compared against `""` EXACTLY.
   *
   * DELIBERATELY NOT TRIMMED, in any of the three languages. PHP's `trim`,
   * JavaScript's `String.prototype.trim` and Python's `str.strip` each strip a
   * different set of codepoints, so trimming here would produce three different
   * answers for the same input -- which is precisely how `prism-human-plus`
   * G-36 happened. A worker id of one space is accepted, and that is the
   * correct behaviour rather than an oversight.
   */
  static taskIdentifierBlank(what: string): HarnessError {
    return new HarnessError(
      'task_identifier_blank',
      `The ${what} is the empty string. An identifier that is not an identifier would make one ` +
        'claim indistinguishable from another.',
    );
  }

  /**
   * An error, never a silent no-op.
   *
   * A second `release()` means two things believe they own the outcome of one
   * task, and swallowing it leaves whichever wrote last as the answer with
   * nothing recording that they disagreed.
   */
  static taskAlreadyTerminal(id: string, state: string): HarnessError {
    return new HarnessError(
      'task_already_terminal',
      `The task [${id}] is already [${state}], which is terminal. Re-releasing it is refused rather ` +
        'than ignored: two callers believing they own one outcome is worth reporting.',
    );
  }

  /**
   * "You are not holding this" -- and NOT WHO IS.
   *
   * The holder is deliberately left out of the sentence. This error can be
   * raised inside a tool call, and a tool's failure comes back to the model as
   * readable text: naming the other worker would hand an agent the identity of
   * a peer it has no business knowing, from a refusal. The caller that needs to
   * know who holds a task can ask the source.
   *
   * Covers three situations -- nobody holds it, someone else does, and your own
   * lease expired -- because all three have the same answer: claim it again.
   */
  static taskLeaseNotHeld(id: string, worker: string, detail: string): HarnessError {
    return new HarnessError(
      'task_lease_not_held',
      `The worker [${worker}] is not holding the task [${id}]: ${detail}. Claim it again rather ` +
        'than acting on it.',
    );
  }

  /**
   * An outcome that is not `done` or `failed` was named.
   *
   * REFUSED, not coerced. The outcome decides a TERMINAL state; mapping
   * `"complete"`, `"DONE"`, `null` or a MISSING argument onto `done` would mean
   * a malformed call produces the MORE privileged result -- an agent declaring
   * victory by typo. Failing closed here costs a retry with a valid value.
   *
   * Absent is covered by the same code deliberately. The argument for treating
   * it as `done` is real -- calling a tool named `complete_task` looks like the
   * declaration by itself -- and the reference settled against it.
   *
   * `where` names the door that refused rather than always a tool, because
   * there are two: the completion tool a model can call, and
   * `StoreTaskSource.release()`, which is reachable from any untyped caller.
   */
  static taskOutcomeInvalid(where: string, given: unknown): HarnessError {
    return new HarnessError(
      'task_outcome_invalid',
      `[${where}] was given the outcome [${String(given)}]. It must be exactly [done] ` +
        'or [failed]; anything else is refused rather than guessed at, because guessing would let a ' +
        'malformed argument close a task.',
    );
  }

  /**
   * A self-completion tool built against an authorizer that is switched off.
   *
   * The same code, and the same argument, as `policyDefinedButDisabled`: a
   * control that is never consulted reads as a control to every reader and is
   * not one. An agent closing its own tasks is exactly the authority that must
   * not be granted by accident.
   */
  static completionToolWithoutAuthorizer(name: string): HarnessError {
    return new HarnessError(
      'unsafe_authorization_configuration',
      `The tool [${name}] would let the agent mark its own task complete, but the ToolAuthorizer it ` +
        'was given is disabled, so nothing would ever be asked whether that call may proceed. An ' +
        'agent that can close its own tasks turns "run until the goal is met" into "run until it ' +
        'decides it is met". Enable the authorizer and give it a call policy, or do not register ' +
        'this tool.',
    );
  }

  static noAgentRuntime(action: string): HarnessError {
    return new HarnessError(
      'no_agent_runtime',
      `This session cannot ${action}: it was built without an agent runtime.`,
    );
  }
}
