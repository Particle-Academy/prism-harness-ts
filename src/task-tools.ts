import type { JsonObject } from './json.js';
import { HarnessError } from './errors.js';
import { authorizedTool } from './tools.js';
import type { HarnessTool, ToolAuthorizer, ToolFactory } from './tools.js';
import { isTerminalState } from './tasks.js';
import type { AgentTask, AgentTaskSource, LeasedAgentTask, TaskOutcome } from './tasks.js';

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

      // REQUIRED and VALIDATED. `outcome` is the one value the MODEL supplies
      // and it decides a terminal state, so mapping anything else onto `done`
      // -- `"complete"`, `"DONE"`, `null`, or nothing at all -- would let a
      // malformed call produce the more privileged result: an agent declaring
      // victory by typo.
      //
      // ABSENT is refused too, and that is a ruling rather than an obvious
      // reading. The argument for allowing it is real: calling a tool named
      // `complete_task` looks like the declaration by itself. The reference
      // settled it the other way, and the same code covers both, so a model
      // that omitted the argument is told to say which outcome it means rather
      // than having one chosen for it.
      if (given !== 'done' && given !== 'failed') {
        throw HarnessError.taskOutcomeInvalid(name, given);
      }

      const outcome: TaskOutcome = given;

      // CHECKED HERE TOO, and this is not belt-and-braces.
      //
      // `StoreTaskSource.release()` refuses a worker that is not the holder, so
      // against that source this block changes nothing -- deleting it is
      // invisible, and the mutation SURVIVES. That is precisely the argument
      // for it: an interface cannot make an implementation check anything, and
      // `release(task, worker, outcome)` reads to a third party like "find it
      // and set the state", which is what they will write. This tool is the
      // thing a MODEL can call, so it does not get to assume the source behind
      // it is the careful one. A fixture source with no guard at all is in the
      // suite for exactly this, because nothing else would notice.
      const current = await source.find(task.id());

      if (current === null) {
        throw HarnessError.taskNotFound(task.id());
      }

      if (isTerminalState(current.state())) {
        throw HarnessError.taskAlreadyTerminal(task.id(), current.state());
      }

      if (current.state() !== 'claimed') {
        throw HarnessError.taskLeaseNotHeld(task.id(), worker, 'nothing is holding it');
      }

      // ONE comparison, and it must be `!==` against the worker rather than a
      // hunt for a mismatch. The three ways this is not the worker's task --
      // somebody else holds it, nobody holds it, or the holder CANNOT BE
      // ESTABLISHED -- all have to land on refusal, and the third is the one
      // that is easy to get wrong.
      //
      // `AgentTask` is `id`, `instruction`, `state` and nothing else, so a
      // conforming source may legally return a task from which no holder can be
      // read. The first version here treated that as "no mismatch, therefore
      // allowed", which is READING SILENCE AS PERMISSION -- the same mistake as
      // inferring `done` from an absent outcome, asked about a different field
      // and one level down. An unknowable holder resolves to a sentinel that
      // matches no worker, so it refuses.
      if (holderOf(current) !== worker) {
        throw HarnessError.taskLeaseNotHeld(task.id(), worker, 'it is not held by this worker');
      }

      await source.release(task, worker, outcome);

      return { id: task.id(), state: outcome };
    },
  };

  return (session) => authorizedTool(tool, session, authorizer);
}

/**
 * A holder that MATCHES NO WORKER, for a task whose holder cannot be read.
 *
 * A symbol rather than a string, so no worker id can ever equal it -- not the
 * empty string, not `'unknown'`, not whatever a consumer happens to name a
 * worker. "Impossible to collide with" is the entire job.
 */
const UNKNOWABLE_HOLDER = Symbol('the holder could not be established');

/**
 * Who holds this task -- or the sentinel, when that cannot be established.
 *
 * `AgentTaskSource.find()` returns an `AgentTask`: three methods, no lease.
 * That is deliberate, because a consumer adapting an existing table may have
 * nowhere to put an owner, and it means a perfectly conforming source can hand
 * back a task from which no holder is readable. `LeasedAgentTask` adds the
 * lease, and everything this package produces is one.
 *
 * The unreadable case resolves to a sentinel rather than to `undefined`,
 * because `undefined` invites the caller to skip the comparison -- which is
 * what the first version of this did, and skipping the comparison is granting
 * the permission.
 */
function holderOf(task: AgentTask): string | null | typeof UNKNOWABLE_HOLDER {
  const candidate = task as Partial<LeasedAgentTask>;

  if (typeof candidate.claimedBy !== 'function') {
    return UNKNOWABLE_HOLDER;
  }

  return candidate.claimedBy();
}
