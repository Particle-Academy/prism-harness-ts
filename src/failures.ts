import { HarnessError } from './errors.js';

/**
 * What the harness says of a failure it did not write (G-387 round 8): the
 * error's class name and, for a HarnessError, its code (a fixed identifier
 * the harness defines). NEVER its message: a store's, a provider's or a
 * host's own words reach neither a model nor a run row nor telemetry.
 *
 * Every read here is guarded and made once: an error whose `code` or class
 * getter throws, or answers differently each time it is read, gives a fixed
 * name, never a rejection and never text the error chose.
 *
 * @internal Used by the runtime and the subagent runner.
 */
export function errorCode(error: unknown): string | null {
  try {
    if (!(error instanceof HarnessError)) return null;
    const code: unknown = error.code;

    return typeof code === 'string' && ERROR_CODE.test(code) ? code : null;
  } catch {
    return null;
  }
}

/**
 * The class name and harness code of `error`, as `HarnessError <code>`, or
 * the class name alone. `code` is the code as already read (errorCode()),
 * so a caller that branched on it reports the same one.
 *
 * @internal
 */
export function failureName(error: unknown, code: string | null = errorCode(error)): string {
  try {
    if (error instanceof HarnessError) return `HarnessError ${code ?? 'error'}`;
    if (!(error instanceof Error)) return 'Error';
    // The class as its prototype says, never an own `constructor` the thrown
    // object carries; a name that is not an identifier is not reported.
    const name: unknown = (Object.getPrototypeOf(error) as { constructor?: { name?: unknown } } | null)?.constructor?.name;

    return typeof name === 'string' && CLASS_NAME.test(name) ? name : 'Error';
  } catch {
    return 'Error';
  }
}

/**
 * What a failed run records (its run row, run state and `run.failed`
 * event): a fixed description with the class name and harness code. This
 * harness has no content-capture setting, so exception text is never
 * written there; the exception itself still reaches the caller.
 *
 * @internal
 */
export function runFailure(error: unknown): string {
  return `The run failed (${failureName(error)}).`;
}

/** What a reported class name may look like: an identifier, at most 64 characters. */
const CLASS_NAME = /^[A-Za-z_$][A-Za-z0-9_$]{0,63}$/;

/** What a reported harness error code may look like (`session_locked`, `plan.budget_context`). */
const ERROR_CODE = /^[a-z][a-z0-9_.]{0,63}$/;
