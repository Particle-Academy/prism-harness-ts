import { HarnessError } from './errors.js';
import type { Subagent } from './subagents.js';
import { subagentFromConfig } from './subagents.js';

export interface ModeConfig {
  system_prompt?: string;
  tools?: readonly string[];
  skills?: readonly string[];
  max_steps?: number;
  subagents?: Record<string, unknown>;
  requires_approval?: readonly string[];
}

export interface ModeRegistryConfig {
  default?: string;
  modes?: Record<string, ModeConfig>;
}

/**
 * What a mode lets a run do.
 *
 * A mode is the unit of authority here: the tools it may reach, the skills it
 * loads, how many steps it gets, which nested agents it may call, and which of
 * its tools need a human first.
 */
export class AgentMode {
  constructor(
    readonly name: string,
    readonly systemPrompt: string,
    readonly tools: readonly string[],
    readonly skills: readonly string[],
    readonly maxSteps: number,
    /** Nested agents this mode may call, by name. */
    readonly subagents: Readonly<Record<string, Subagent>> = {},
    /** Tools that must not run until a human says so. */
    readonly requiresApproval: readonly string[] = [],
  ) {}

  /**
   * Whether a named tool needs a human before it runs IN THIS MODE.
   *
   * Declared per mode rather than on the tool, because the same tool is not
   * equally consequential everywhere: `execute_op` against a scratch project is
   * routine and against production is not, and the tool cannot tell which it is
   * in. `'*'` gates every tool the mode offers.
   */
  needsApproval(tool: string): boolean {
    return this.requiresApproval.includes('*') || this.requiresApproval.includes(tool);
  }
}

export class ModeRegistry {
  constructor(private readonly config: ModeRegistryConfig = {}) {}

  default(): string {
    const configured = this.config.default;

    return typeof configured === 'string' && configured !== '' ? configured : 'chat';
  }

  /** Every mode name this application has configured. */
  names(): string[] {
    return Object.keys(this.config.modes ?? {});
  }

  /**
   * Every mode, RESOLVED.
   *
   * Resolving them all is the point: `resolve()` validates as it goes, so a
   * mode nobody has entered yet keeps its misconfiguration until the day
   * somebody switches to it. This is what the doctor uses to find that on a
   * Tuesday rather than in front of a user.
   */
  all(): Record<string, AgentMode> {
    return Object.fromEntries(this.names().map((name) => [name, this.resolve(name)]));
  }

  resolve(name?: string | null): AgentMode {
    const wanted = name ?? this.default();
    const mode = this.config.modes?.[wanted];

    if (mode === undefined) {
      throw HarnessError.modeNotConfigured(wanted);
    }

    const prompt = mode.system_prompt ?? '';
    const maxSteps = mode.max_steps ?? 8;

    if (typeof prompt !== 'string' || !Number.isInteger(maxSteps) || maxSteps < 1) {
      throw HarnessError.modeMalformed(wanted, 'system_prompt must be a string and max_steps a positive integer');
    }

    return new AgentMode(
      wanted,
      prompt,
      strings(mode.tools),
      strings(mode.skills),
      maxSteps,
      this.#subagentsFor(wanted, mode),
      strings(mode.requires_approval),
    );
  }

  /**
   * The nested agents a mode may call.
   *
   * DECLARED PER MODE, which is the point: a subagent is authority, and
   * authority a run inherits by being nested is authority nobody granted. A
   * mode that names no subagents cannot spawn one.
   *
   * A subagent's own mode must EXIST, and it is checked here rather than at
   * call time so a typo surfaces when the parent mode is loaded instead of
   * halfway through a run that has already spent budget.
   */
  #subagentsFor(name: string, mode: ModeConfig): Record<string, Subagent> {
    const declared = mode.subagents ?? {};
    const subagents: Record<string, Subagent> = {};

    for (const [key, config] of Object.entries(declared)) {
      if (typeof config !== 'object' || config === null || Array.isArray(config)) {
        throw HarnessError.modeMalformed(name, `subagent [${key}] is not an object`);
      }

      const subagent = subagentFromConfig(key, config as Record<string, unknown>);

      if (this.config.modes?.[subagent.mode] === undefined) {
        throw HarnessError.modeMalformed(
          name,
          `it declares subagent [${key}], whose mode [${subagent.mode}] is not configured`,
        );
      }

      subagents[key] = subagent;
    }

    return subagents;
  }
}

function strings(value: readonly unknown[] | undefined): string[] {
  return (value ?? []).filter((entry): entry is string => typeof entry === 'string');
}
