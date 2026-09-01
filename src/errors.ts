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
  | 'skill_path_refused';

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

  static noAgentRuntime(action: string): HarnessError {
    return new HarnessError(
      'no_agent_runtime',
      `This session cannot ${action}: it was built without an agent runtime.`,
    );
  }
}
