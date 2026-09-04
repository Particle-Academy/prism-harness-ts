import type { JsonObject } from './json.js';
import { HarnessError } from './errors.js';
import { authorizedTool } from './tools.js';
import type { HarnessTool, ToolAuthorizer, ToolFactory } from './tools.js';
import type { AgentTask, AgentTaskSource, TaskOutcome } from './tasks.js';

export interface CompletionToolOptions {
  /** Where the task lives. */
  source: AgentTaskSource;

  /** The task this tool may close. ONE task, the one the worker is holding. */
  task: AgentTask;

  /**
   * The worker holding the task. REQUIRED, and load-bearing.
   *
   * A completion tool that names only a source can close any task in the list,
   * including one another worker is midway through -- and a tool the model can
   * call is the last place to leave that reachable. Passing the worker through
   * means the source refuses anything this worker is not actually holding.
   */
  worker: string;

  /** Must be ENABLED. See below for why that is checked rather than assumed. */
  authorizer: ToolAuthorizer;

  /** Defaults to `complete_task`. */
  name?: string;
}

/**
 * A tool that lets the AGENT close its own task. OFF UNLESS A CONSUMER BUILDS IT.
 *
 * ## Why this is not the default
 *
 * If the model can set its own task to `done`, then "run until the goal is met"
 * silently becomes "run until it decides it is met", and a run that has stalled
 * ends by declaring victory. That is the same failure `prism-human-plus`
 * addresses by reserving confirmation for the human, and the same reason the
 * default here is off in all three languages. A port that ships it on has
 * changed an observable decision, not a default.
 *
 * By default `release()` is called by the APPLICATION, from evidence. Nothing
 * in this package registers this tool, and a `StoreTaskSource` on its own gives
 * an agent no route to a terminal state.
 *
 * ## Why the authorizer is checked
 *
 * Through the EXISTING `ToolAuthorizer` -- no new permission mechanism, because
 * a second mechanism is a second place to grant something by accident.
 *
 * A disabled authorizer is refused rather than tolerated. `ToolAuthorizer`
 * defaults to off, and off means every registered tool is offered to every run
 * and every call is allowed; a consumer who wires this tool up against one has
 * granted the agent completion authority while believing a control is in place.
 * Same argument, and the same error code, as a policy that is defined and never
 * consulted.
 *
 * The returned factory produces a tool wrapped by `authorizedTool`, so the call
 * policy is asked again AT CALL TIME with the arguments -- an offer-time check
 * alone can say "may complete tasks" and never "only this task, only once the
 * tests pass".
 */
export function agentCompletionTool(options: CompletionToolOptions): ToolFactory {
  const name = options.name ?? 'complete_task';

  if (!options.authorizer.enabled) {
    throw HarnessError.completionToolWithoutAuthorizer(name);
  }

  const { source, task, worker, authorizer } = options;

  const tool: HarnessTool = {
    name,
    description:
      `Record the outcome of the task [${task.id()}] as either "done" or "failed". Only call this ` +
      'when the work is actually finished; "failed" is a legitimate answer and is not a worse one.',
    async handle(args: JsonObject): Promise<unknown> {
      const given = args.outcome;

      // VALIDATED, not coerced, and this is the security-relevant line in the
      // file. `outcome` is the one value the MODEL supplies, and it decides a
      // terminal state. Mapping anything unrecognised onto `done` -- including
      // a missing argument -- would mean a malformed call produces the more
      // privileged result, which is an agent declaring victory by typo.
      if (given !== 'done' && given !== 'failed') {
        throw HarnessError.taskOutcomeInvalid(name, given);
      }

      const outcome: TaskOutcome = given;

      await source.release(task, worker, outcome);

      return { id: task.id(), state: outcome };
    },
  };

  return (session) => authorizedTool(tool, session, authorizer);
}
