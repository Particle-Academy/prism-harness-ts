import { HarnessError } from './errors.js';
import type { JsonObject } from './json.js';
import { isJsonObject } from './json.js';
import type { Subagent } from './subagents.js';
import { isAgentId, subagentFromConfig } from './subagents.js';
import { validateAgent } from './address.js';
import { RegistrationCollision, registrationOwner } from './registration.js';

export interface ModeConfig {
  system_prompt?: string;
  tools?: readonly string[];
  skills?: readonly string[];
  max_steps?: number;
  subagents?: Record<string, unknown>;
  requires_approval?: readonly string[];
  /** Passed unchanged to the model client on every run in this mode. Must be a map. */
  provider_options?: Record<string, unknown>;
  /** The provider every run in this mode uses, above the agent default (G-321 E). */
  provider?: string | null;
  /** The model every run in this mode uses, above the agent default. */
  model?: string | null;
  /** Named agents this mode's subagents may run as. Declared, never inherited. */
  agents?: readonly string[];
}

export interface ModeRegistryConfig {
  refuseDuplicates?: boolean;
  owner?: string;
  agents?: Record<string, ModeRegistryConfig>;
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
    /** Options handed to the provider on every run in this mode, unchanged. */
    readonly providerOptions: Readonly<JsonObject> = {},
    /** The provider runs in this mode use, or null to fall through to the agent. */
    readonly provider: string | null = null,
    /** The model runs in this mode use, or null to fall through to the agent. */
    readonly model: string | null = null,
    /** Named agents this mode's subagents may run as. */
    readonly agents: readonly string[] = [],
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
  readonly #agents = new Map<string, ModeRegistry>();
  readonly #owners = new Map<string, string>();

  constructor(private readonly config: ModeRegistryConfig = {}, private readonly agent: string | null = null, private readonly root?: ModeRegistry) {
    if (config.refuseDuplicates !== undefined && typeof config.refuseDuplicates !== 'boolean') throw new TypeError('refuseDuplicates must be a boolean');
    validateAgent(agent);
    for (const name of Object.keys(config.modes ?? {})) this.#owners.set(name, registrationOwner(config.owner, agent));
    for (const key of ['config', 'agent', 'root']) Object.defineProperty(this, key, { writable: false, configurable: false });
  }

  forAgent(agent: string | null): ModeRegistry {
    validateAgent(agent);
    const root = this.root ?? this;
    if (agent === null) return root;
    let view = root.#agents.get(agent);
    if (!view) {
      const configured = root.config.agents?.[agent] ?? {};
      view = new ModeRegistry({ ...structuredClone(configured), refuseDuplicates:root.config.refuseDuplicates, default: configured.default ?? root.default() }, agent, root);
      root.#agents.set(agent, view);
    }
    return view;
  }

  register(name: string, mode: ModeConfig, owner?: string): this {
    const label = registrationOwner(owner, this.agent), previous = this.#owners.get(name);
    if ((this.agent !== null || this.config.refuseDuplicates === true) && previous !== undefined) throw new RegistrationCollision(this.agent, name, previous, label);
    this.#set(name, mode, label);
    return this;
  }

  replace(name: string, mode: ModeConfig, predecessorOwner: string, owner?: string): this {
    const label = registrationOwner(owner, this.agent);
    if (this.#owners.get(name) !== predecessorOwner) throw new Error('Replacement must name the registered predecessor owner.');
    this.#set(name, mode, label);
    return this;
  }

  #set(name: string, mode: ModeConfig, owner: string): void {
    this.config.modes ??= {};
    this.config.modes[name] = structuredClone(mode);
    this.#owners.set(name, owner);
  }

  setDefault(name: string): this {
    this.resolve(name);
    this.config.default = name;
    return this;
  }

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

    for (const key of ['provider', 'model'] as const) {
      const value: unknown = mode[key];
      if (value !== undefined && value !== null && (typeof value !== 'string' || value === '')) {
        throw new HarnessError('mode_malformed', `Harness mode [${wanted}] declares a malformed provider/model: expected a nonempty string or null.`);
      }
    }
    const agents = agentsFor(wanted, mode.agents);

    return new AgentMode(
      wanted,
      prompt,
      strings(mode.tools),
      strings(mode.skills),
      maxSteps,
      this.#subagentsFor(wanted, mode, agents),
      strings(mode.requires_approval),
      providerOptionsFor(wanted, mode.provider_options),
      mode.provider ?? null,
      mode.model ?? null,
      agents,
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
  #subagentsFor(name: string, mode: ModeConfig, agents: readonly string[]): Record<string, Subagent> {
    const declared = mode.subagents ?? {};
    const subagents: Record<string, Subagent> = {};

    for (const [key, config] of Object.entries(declared)) {
      if (typeof config !== 'object' || config === null || Array.isArray(config)) {
        throw HarnessError.modeMalformed(name, `subagent [${key}] is not an object`);
      }

      const subagent = subagentFromConfig(key, config as Record<string, unknown>, name);

      // DELEGATION IS AUTHORITY, so it is declared by the parent mode and never
      // inherited: a subagent may run as another agent only when this mode
      // lists that agent. Running as the mode's own agent delegates nothing.
      if (subagent.agent !== null && subagent.agent !== this.agent && !agents.includes(subagent.agent)) {
        throw new HarnessError(
          'mode_malformed',
          `Harness mode [${name}] declares subagent [${key}] for agent [${subagent.agent}], which the mode does not list in its agents.`,
        );
      }

      // The child's mode is looked up where the child RUNS: in the delegated
      // agent's registry when it names one. Names only; resolving it here
      // would recurse through any cycle of modes.
      const target = subagent.agent === null || subagent.agent === this.agent ? this : (this.root ?? this).forAgent(subagent.agent);
      if (target.config.modes?.[subagent.mode] === undefined) {
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

/**
 * A mode's `agents` list, held to the agent-id rule exactly: never trimmed and
 * never case-folded, because a folded lookalike would select another agent.
 */
function agentsFor(name: string, declared: unknown): string[] {
  if (declared === undefined || declared === null) return [];
  if (!Array.isArray(declared) || !declared.every(isAgentId)) {
    throw new HarnessError('mode_malformed', `Harness mode [${name}] declares a malformed agents list: each entry must be a lowercase agent id of 1-64 ASCII bytes.`);
  }

  return [...declared];
}

function strings(value: readonly unknown[] | undefined): string[] {
  return (value ?? []).filter((entry): entry is string => typeof entry === 'string');
}

/**
 * A mode's provider options, refused rather than ignored when they are not a map.
 *
 * A list or a scalar would reach the provider as nothing, and the mode would run
 * without the option its author believes is on.
 */
function providerOptionsFor(name: string, declared: unknown): Readonly<JsonObject> {
  if (declared === undefined) {
    return {};
  }

  if (!isJsonObject(declared)) {
    throw HarnessError.modeMalformed(name, 'provider_options must be a map of option names to values');
  }

  return declared;
}
