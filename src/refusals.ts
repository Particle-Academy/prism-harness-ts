import { AsyncLocalStorage } from 'node:async_hooks';

/**
 * Which tool results were REFUSALS Harness produced, said at the point of refusal.
 *
 * A refusal Harness hands the model — a subagent the tree would not let run, a
 * task completion gate — travels to the model as an ordinary result, on
 * purpose: it is information the model should act on. Recovering "refused"
 * from the TEXT would be a guess any tool could imitate, so the site that
 * refuses says so here, and the run step stream reads it back.
 *
 * SCOPED TO ONE TOOL CALL, by construction rather than by cleanup. The reference
 * keeps a static map keyed by run id and forgets each run in a `finally`; this
 * port has AsyncLocalStorage, so each call the runtime invokes gets its own
 * cell, owned by that invocation's frame. A refusal can only mark the call it
 * happens inside: there is no registry another run, session or later call could
 * read, and nothing outlives the call. A nested run (a subagent) gets its own
 * cells for its own calls, and the parent's cell is untouched by them.
 *
 * NOT EXPORTED from the package. A third-party tool cannot mark its own result
 * as a refusal, which is what keeps `refused` a statement Harness makes.
 *
 * @internal
 */
interface RefusalCell {
  refused: boolean;
}

const calls = new AsyncLocalStorage<RefusalCell>();

/**
 * Run one tool call with its own refusal cell, and report how it settled and
 * whether it was refused. Reported on the throw path too, because a gate that
 * refuses by throwing is still a refusal.
 */
export async function settleWithRefusalScope<T>(
  invoke: () => Promise<T>,
): Promise<{ ok: true; value: T; refused: boolean } | { ok: false; error: unknown; refused: boolean }> {
  const cell: RefusalCell = { refused: false };

  try {
    const value = await calls.run(cell, invoke);

    return { ok: true, value, refused: cell.refused };
  } catch (error) {
    return { ok: false, error, refused: cell.refused };
  }
}

/** Mark the CURRENT tool call as refused. Outside a call there is nothing to mark. */
export function markRefused(): void {
  const cell = calls.getStore();

  if (cell !== undefined) cell.refused = true;
}
