import type { JsonObject } from './json.js';
import { HarnessError } from './errors.js';
import type { Session } from './session.js';
import { validateAgent } from './address.js';
import { RegistrationCollision, registrationOwner } from './registration.js';

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
export interface ResolvedTool {
  tool: HarnessTool;
  owner: string;
  source: 'static' | 'factory' | 'provider';
}

export class ToolRegistry {
  readonly #tools = new Map<string, HarnessTool>();
  readonly #factories = new Map<string, ToolFactory>();
  readonly #metadata = new Map<string, Omit<ResolvedTool, 'tool'>>();
  readonly #providers = new Map<string, { provider: ToolProvider; owner: string }>();
  readonly #agents = new Map<string, ToolRegistry>();

  constructor(private readonly namespace: string | null = null, private readonly root?: ToolRegistry) {
    validateAgent(namespace);
    for (const key of ['namespace', 'root']) Object.defineProperty(this, key, { writable: false, configurable: false });
  }

  /** Immutable namespace views share registrations, never a current-agent flag. */
  forAgent(agent: string | null): ToolRegistry {
    validateAgent(agent);
    const root = this.root ?? this;
    if (agent === null) return root;
    let view = root.#agents.get(agent);
    if (!view) { view = new ToolRegistry(agent, root); root.#agents.set(agent, view); }
    return view;
  }

  register(tool: HarnessTool, owner?: string): this {
    this.#claim(tool.name, owner, 'static');
    this.#tools.set(tool.name, tool);
    return this;
  }

  registerMany(tools: Iterable<HarnessTool>, owner?: string): this {
    for (const tool of tools) this.register(tool, owner);
    return this;
  }

  registerFactory(name: string, factory: ToolFactory, owner?: string): this {
    this.#claim(name, owner, 'factory');
    this.#factories.set(name, factory);
    return this;
  }

  registerProvider(provider: ToolProvider, owner?: string, name?: string): this {
    const label = registrationOwner(owner, this.namespace);
    const id = name ?? (this.namespace === null ? `provider:${this.#providers.size}` : undefined);
    if (!id) throw new TypeError('A named provider requires a registration name.');
    const previous = this.#providers.get(id);
    if (previous) throw new RegistrationCollision(this.namespace, id, previous.owner, label);
    this.#providers.set(id, { provider, owner: label });
    return this;
  }

  replace(tool: HarnessTool, predecessorOwner: string, owner?: string): this {
    this.#claim(tool.name, owner, 'static', predecessorOwner);
    this.#tools.set(tool.name, tool);
    return this;
  }

  replaceFactory(name: string, factory: ToolFactory, predecessorOwner: string, owner?: string): this {
    this.#claim(name, owner, 'factory', predecessorOwner);
    this.#factories.set(name, factory);
    return this;
  }

  replaceProvider(name: string, provider: ToolProvider, predecessorOwner: string, owner?: string): this {
    const label = registrationOwner(owner, this.namespace), previous = this.#providers.get(name);
    if (!previous || previous.owner !== predecessorOwner) throw new Error('Replacement must name the registered predecessor owner.');
    this.#providers.set(name, { provider, owner: label });
    return this;
  }

  #claim(name: string, owner: string | undefined, source: 'static' | 'factory', predecessor?: string): void {
    const label = registrationOwner(owner, this.namespace), previous = this.#metadata.get(name);
    if (predecessor !== undefined) {
      if (!previous || previous.owner !== predecessor) throw new Error('Replacement must name the registered predecessor owner.');
    } else if (previous) throw new RegistrationCollision(this.namespace, name, previous.owner, label);
    this.#tools.delete(name); this.#factories.delete(name);
    this.#metadata.set(name, { owner: label, source });
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
    const detailed = await this.resolveDetailed(names, session);
    return new Map([...detailed].map(([name, entry]) => [name, entry.tool]));
  }

  /** The offer path and inspection path consume the same ownership result. */
  async resolveDetailed(names: readonly string[], session?: Session): Promise<Map<string, ResolvedTool>> {
    if (session && session.agent !== this.namespace) {
      if (this.namespace !== null) throw new Error('Harness registry/session agent mismatch.');
      return this.forAgent(session.agent).resolveDetailed(names, session);
    }
    const provided = new Map<string, HarnessTool>(), metadata = new Map(this.#metadata);

    if (session !== undefined) {
      for (const { provider, owner } of this.#providers.values()) {
        for (const tool of await provider(session)) {
          const previous = metadata.get(tool.name);
          if (previous) throw new RegistrationCollision(this.namespace, tool.name, previous.owner, owner);
          provided.set(tool.name, tool); metadata.set(tool.name, { owner, source: 'provider' });
        }
      }
    }

    const available = [
      ...new Set([...this.#tools.keys(), ...this.#factories.keys(), ...provided.keys()]),
    ];
    const selected = names.includes('*') ? available : names;

    const resolved = new Map<string, ResolvedTool>();

    for (const name of selected) {
      const tool = this.#tools.get(name) ?? provided.get(name);

      if (tool !== undefined) {
        resolved.set(name, { tool, ...metadata.get(name)! });
        continue;
      }

      const factory = this.#factories.get(name);

      if (factory === undefined) {
        throw HarnessError.toolNotAvailable(name, available);
      }

      if (session === undefined) {
        throw HarnessError.toolNotAvailable(`${name} (needs a session to build)`, available);
      }

      const built = factory(session);
      if (this.namespace !== null && built.name !== name) throw new Error(`Session-bound tool factory [${name}] returned [${built.name}].`);
      if (built.name !== name) {
        const previous = metadata.get(built.name);
        if (previous) throw new RegistrationCollision(this.namespace, built.name, previous.owner, metadata.get(name)!.owner);
        metadata.set(built.name, metadata.get(name)!);
      }
      resolved.set(name, { tool: built, ...metadata.get(name)! });
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
  agents?: Record<string, ToolAuthorizerOptions>;
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
  readonly #agents: Record<string, ToolAuthorizerOptions>;
  readonly #scoped = new Map<string, ToolAuthorizer>();

  constructor(options: ToolAuthorizerOptions = {}, private readonly namespace: string | null = null, private readonly root?: ToolAuthorizer) {
    validateAgent(namespace);
    this.enabled = options.enabled ?? false;
    this.#offer = options.offer;
    this.#call = options.call;
    this.#agents = Object.fromEntries(Object.entries(options.agents ?? {}).map(([agent, config]) => {
      validateAgent(agent);
      return [agent, { ...config }];
    }));

    if (!this.enabled && (options.offer !== undefined || options.call !== undefined)) {
      throw HarnessError.policyDefinedButDisabled();
    }
    for (const key of ['enabled', 'namespace', 'root']) Object.defineProperty(this, key, { writable: false, configurable: false });
  }

  forAgent(agent: string | null): ToolAuthorizer {
    validateAgent(agent);
    const root = this.root ?? this;
    if (agent === null) return root;
    let scoped = root.#scoped.get(agent);
    if (!scoped) {
      const overrides = root.#agents[agent] ?? {};
      const options = { enabled: root.enabled, offer: root.#offer, call: root.#call, ...overrides, agents: {} };
      if (typeof options.enabled !== 'boolean' || (options.offer !== undefined && typeof options.offer !== 'function') || (options.call !== undefined && typeof options.call !== 'function')) throw new TypeError('Malformed agent tool authorization options.');
      scoped = new ToolAuthorizer(options, agent, root);
      root.#scoped.set(agent, scoped);
    }
    return scoped;
  }

  hasCallPolicy(): boolean { return this.#call !== undefined; }

  /** The tools a run may be offered, each wrapped so the call policy is asked again. */
  async allowed(session: Session, tools: Map<string, HarnessTool>): Promise<HarnessTool[]> {
    if (session.agent !== this.namespace) {
      if (this.namespace !== null) throw new Error('Harness authorizer/session agent mismatch.');
      return this.forAgent(session.agent).allowed(session, tools);
    }
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
    if (session.agent !== this.namespace) {
      if (this.namespace !== null) throw new Error('Harness authorizer/session agent mismatch.');
      return this.forAgent(session.agent).allowsCall(session, tool, args);
    }
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
