import type { JsonObject } from './json.js';
import { HarnessError } from './errors.js';
import type { Session } from './session.js';

/**
 * The shape this package needs from a tool.
 *
 * STRUCTURAL, not an import. `prism-ts`'s `Tool` satisfies it, and so does
 * anything else with a name and a handler — which keeps this package at zero
 * dependencies and lets a consumer bring their own tool type. The reference
 * imports `Prism\Prism\Tool` directly because it is already a dependency there.
 */
export interface HarnessTool {
  readonly name: string;
  readonly description?: string;
  handle(args: JsonObject): unknown | Promise<unknown>;
}

export type ToolFactory = (session: Session) => HarnessTool;
export type ToolProvider = (session: Session) => Iterable<HarnessTool> | Promise<Iterable<HarnessTool>>;

/**
 * Which tools exist, and which a given run may be offered.
 *
 * Three ways in, because the three answer different questions: a tool that is
 * always the same (`register`), one that needs the session to build
 * (`registerFactory`), and a set discovered at resolve time — an MCP server's
 * catalogue, say (`registerProvider`).
 */
export class ToolRegistry {
  readonly #tools = new Map<string, HarnessTool>();

  readonly #factories = new Map<string, ToolFactory>();

  readonly #providers: ToolProvider[] = [];

  register(tool: HarnessTool): this {
    this.#tools.set(tool.name, tool);

    return this;
  }

  registerMany(tools: Iterable<HarnessTool>): this {
    for (const tool of tools) this.register(tool);

    return this;
  }

  registerFactory(name: string, factory: ToolFactory): this {
    this.#factories.set(name, factory);

    return this;
  }

  registerProvider(provider: ToolProvider): this {
    this.#providers.push(provider);

    return this;
  }

  /** Every name this registry can produce without a session. */
  names(): string[] {
    return [...new Set([...this.#tools.keys(), ...this.#factories.keys()])].sort();
  }

  /**
   * Resolve the named tools for a session.
   *
   * `'*'` means every tool this registry can produce. A name that resolves to
   * nothing is an ERROR rather than a silent omission: a mode that lists a tool
   * it cannot get is a misconfiguration, and dropping it quietly would leave a
   * run wondering why the model never called it.
   */
  async resolve(names: readonly string[], session?: Session): Promise<Map<string, HarnessTool>> {
    const provided = new Map<string, HarnessTool>();

    if (session !== undefined) {
      for (const provider of this.#providers) {
        for (const tool of await provider(session)) provided.set(tool.name, tool);
      }
    }

    const available = [
      ...new Set([...this.#tools.keys(), ...this.#factories.keys(), ...provided.keys()]),
    ];
    const selected = names.includes('*') ? available : names;

    const resolved = new Map<string, HarnessTool>();

    for (const name of selected) {
      const tool = this.#tools.get(name) ?? provided.get(name);

      if (tool !== undefined) {
        resolved.set(name, tool);
        continue;
      }

      const factory = this.#factories.get(name);

      if (factory === undefined) {
        throw HarnessError.toolNotAvailable(name, available);
      }

      if (session === undefined) {
        throw HarnessError.toolNotAvailable(`${name} (needs a session to build)`, available);
      }

      resolved.set(name, factory(session));
    }

    return resolved;
  }
}

/** Whether a tool may be OFFERED to a run. */
export type OfferPolicy = (session: Session, tool: HarnessTool) => boolean | Promise<boolean>;

/** Whether THIS call, with THESE arguments, may proceed. */
export type CallPolicy = (
  session: Session,
  tool: HarnessTool,
  args: JsonObject,
) => boolean | Promise<boolean>;

export interface ToolAuthorizerOptions {
  enabled?: boolean;
  offer?: OfferPolicy;
  call?: CallPolicy;
}

/**
 * The two authorization questions, kept apart.
 *
 * Whether a tool may be OFFERED to a run, and whether THIS invocation of it —
 * with these arguments, this many calls in — may proceed. At the moment the
 * toolset is assembled the arguments do not exist yet, so an offer policy can
 * say "may use delete_file" and never "only under /tmp".
 *
 * OFF BY DEFAULT, matching the reference. But a policy that is defined and
 * never consulted is REFUSED at construction rather than tolerated: it looks
 * like a control to every reader and is not one. That is the one configuration
 * not to leave in place — both at once.
 */
export class ToolAuthorizer {
  readonly enabled: boolean;

  readonly #offer?: OfferPolicy;

  readonly #call?: CallPolicy;

  constructor(options: ToolAuthorizerOptions = {}) {
    this.enabled = options.enabled ?? false;
    this.#offer = options.offer;
    this.#call = options.call;

    if (!this.enabled && (options.offer !== undefined || options.call !== undefined)) {
      throw HarnessError.policyDefinedButDisabled();
    }
  }

  /** The tools a run may be offered, each wrapped so the call policy is asked again. */
  async allowed(session: Session, tools: Map<string, HarnessTool>): Promise<HarnessTool[]> {
    if (!this.enabled) {
      return [...tools.values()];
    }

    const allowed: HarnessTool[] = [];

    for (const tool of tools.values()) {
      if (this.#offer === undefined || (await this.#offer(session, tool))) {
        // Wrapped so the SAME policy is asked when the tool is actually called,
        // with arguments. Offer-time filtering alone cannot bound how a tool is
        // used, only whether it is present.
        allowed.push(authorizedTool(tool, session, this));
      }
    }

    return allowed;
  }

  /**
   * Whether this call may proceed.
   *
   * True when the authorizer is disabled, matching `allowed()` — the
   * constructor has already refused the configuration where that silence would
   * be mistaken for enforcement.
   */
  async allowsCall(session: Session, tool: HarnessTool, args: JsonObject): Promise<boolean> {
    if (!this.enabled || this.#call === undefined) {
      return true;
    }

    return this.#call(session, tool, args);
  }
}

/**
 * A tool that asks the policy again, at call time, with the arguments.
 *
 * Refuses by THROWING rather than returning an error string: a refusal handed
 * back as a tool result reads to the model as a failure it might retry
 * differently, and a denied action being retried is the opposite of what a
 * guard is for.
 */
export function authorizedTool(
  tool: HarnessTool,
  session: Session,
  authorizer: ToolAuthorizer,
): HarnessTool {
  return {
    name: tool.name,
    description: tool.description,
    async handle(args: JsonObject): Promise<unknown> {
      if (!(await authorizer.allowsCall(session, tool, args))) {
        throw HarnessError.callNotAuthorized(tool.name);
      }

      return tool.handle(args);
    },
  };
}
