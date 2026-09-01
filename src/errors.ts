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
  | 'no_agent_runtime';

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

  static noAgentRuntime(action: string): HarnessError {
    return new HarnessError(
      'no_agent_runtime',
      `This session cannot ${action}: it was built without an agent runtime.`,
    );
  }
}
