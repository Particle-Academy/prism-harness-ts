import { createHash } from 'node:crypto';
import { HarnessError } from './errors.js';

/**
 * A tool's own declaration of how much it can change (advisory).
 *
 * Hosts may use it to build their list; the list decides. `destructive` can
 * never be auto-allowed: a policy naming one is refused, and a call to one
 * always asks.
 */
export type ToolTier = 'read_only' | 'write' | 'destructive';

export const TOOL_TIERS: readonly ToolTier[] = ['read_only', 'write', 'destructive'];

/** One allowed name. No `server` means a native harness tool, never an MCP tool. */
export interface ToolPolicyEntry {
  server?: string;
  tool: string;
}

/** The JSON shape, identical in every language, so a host builds it once. */
export interface ToolPolicyJson {
  revision: string;
  allow: ToolPolicyEntry[];
  honor: boolean;
}

/** What the policy needs to know about a tool: its name and what it declares. */
export interface ToolDeclaration {
  readonly name: string;
  readonly server?: string;
  readonly tier?: ToolTier;
}

/**
 * Characters that are invisible at either end of a tool name.
 *
 * The same codepoint set human-plus's TrustPolicy uses for its confirmation
 * reservation, spelled out identically in all three harness languages rather
 * than left to each language's own `trim()`, which disagree about the Unicode
 * spaces and U+FEFF.
 */
const INVISIBLE_CLASS =
  '\u0000\u0009-\u000D\u0020\u0085\u00A0\u1680\u2000-\u200D\u2028\u2029\u202F\u205F\u3000\uFEFF';

const INVISIBLE = new RegExp(`^[${INVISIBLE_CLASS}]+|[${INVISIBLE_CLASS}]+$`, 'gu');

/** Verbs whose tools belong to the person: confirmation and approval. */
const RESERVED_VERBS: ReadonlySet<string> = new Set(['confirm', 'reject', 'accept', 'approve', 'deny']);

/** The agent plan tools (G-360): a plan is a person's agreement, never auto-allowed. */
export const RESERVED_TOOLS: readonly string[] = Object.freeze(['plan_draft', 'plan_propose_change', 'plan_pause_for_person', 'plan_complete_step']);

/** A name with the invisible edges removed. */
export function normaliseToolName(name: string): string {
  return name.replace(INVISIBLE, '');
}

/**
 * ASCII letters, digits, `_ . : -`; a letter, digit or underscore first; at
 * most 128 characters. ASCII-only is what makes a homoglyph unmatchable.
 */
export function isWellFormedToolName(name: string): boolean {
  return /^[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}$/.test(name);
}

/**
 * Is this name reserved for the person (a confirmation or approval tool)?
 *
 * Normalised, then split on every separator and camelCase boundary; reserved
 * when ANY token is a reserved verb. The same rule as human-plus.
 *
 * The plan tools are reserved by name as well (G-360), because the verb
 * tokeniser does not catch them: a plan_draft or a plan change that a policy
 * let run without asking would be a plan no person approved. Compared after
 * normalising, in ASCII lower case, as the reference does.
 */
export function isPersonOnlyTool(name: string): boolean {
  const normalised = normaliseToolName(name);
  if (RESERVED_TOOLS.includes(normalised.replace(/[A-Z]/g, (letter) => letter.toLowerCase()))) return true;

  return normalised
    .split(/[_.:-]+/)
    .some((segment) => segment.split(/(?<=[a-z0-9])(?=[A-Z])/).some((token) => RESERVED_VERBS.has(token.toLowerCase())));
}

/** True only for a name already in normal form and well formed. */
function isExactName(name: unknown): name is string {
  return typeof name === 'string' && normaliseToolName(name) === name && isWellFormedToolName(name);
}

function refuse(reason: string): never {
  throw new HarnessError('tool_policy_refused', `This tool policy is refused: ${reason}`);
}

/**
 * Which tools run without asking a person. Owned by the HOST.
 *
 * Built from the shared JSON shape, `{ revision, allow: [{ server?, tool }],
 * honor }`. Matching is exact: a name must already be in normal form and well
 * formed, so a padded or look-alike name never matches. A name without
 * `server` matches only a native tool, never one an MCP server provides. There
 * are no wildcards, patterns or prefixes.
 *
 * REFUSED AT CONSTRUCTION, never silently dropped, so the host learns of it:
 * a malformed entry, a confirmation or approval tool, and any entry naming a
 * tool that declares `tier: destructive` among the declarations passed in.
 *
 * `honor: false` turns the list off: everything asks.
 *
 * The policy has no tool surface and is never sent to the model.
 */
export class ToolPolicy {
  readonly revision: string;
  readonly honor: boolean;
  readonly #allow: readonly Readonly<ToolPolicyEntry>[];

  private constructor(revision: string, allow: readonly Readonly<ToolPolicyEntry>[], honor: boolean) {
    this.revision = revision;
    this.honor = honor;
    this.#allow = allow;
    Object.freeze(this);
  }

  /** Nothing runs without asking. */
  static none(): ToolPolicy {
    return new ToolPolicy('', [], true);
  }

  /**
   * @param tools declarations the host knows about; an entry naming one that
   *   declares `destructive` is refused.
   * @throws HarnessError `tool_policy_refused`
   */
  static fromJson(json: unknown, tools: Iterable<ToolDeclaration> = []): ToolPolicy {
    if (typeof json !== 'object' || json === null || Array.isArray(json)) refuse('it must be an object.');
    const value = json as Record<string, unknown>;
    for (const key of Object.keys(value)) {
      if (!['revision', 'allow', 'honor'].includes(key)) refuse(`unknown key "${key}".`);
    }
    if (typeof value.revision !== 'string') refuse('revision must be a string.');
    if (typeof value.honor !== 'boolean') refuse('honor must be a boolean.');
    if (!Array.isArray(value.allow)) refuse('allow must be a list.');

    const declared = [...tools];
    const allow: Readonly<ToolPolicyEntry>[] = [];
    for (const raw of value.allow as unknown[]) {
      if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) refuse('every allow entry must be an object.');
      const entry = raw as Record<string, unknown>;
      for (const key of Object.keys(entry)) {
        if (key !== 'server' && key !== 'tool') refuse(`unknown entry key "${key}".`);
      }
      if (!isExactName(entry.tool)) refuse('an entry names a tool that is not an exact, well-formed name.');
      if (entry.server !== undefined && !isExactName(entry.server)) refuse('an entry names a server that is not an exact, well-formed name.');
      const tool = entry.tool;
      const server = entry.server as string | undefined;
      if (isPersonOnlyTool(tool)) refuse(`"${tool}" is a confirmation or approval tool, which only a person may run.`);
      if (declared.some((candidate) => candidate.tier === 'destructive' && sameIdentity(candidate, server, tool))) {
        refuse(`"${tool}" declares tier destructive, which is never auto-allowed.`);
      }
      allow.push(Object.freeze(server === undefined ? { tool } : { server, tool }));
    }

    return new ToolPolicy(value.revision, Object.freeze(allow), value.honor);
  }

  /** The entries, as given. */
  get allow(): readonly Readonly<ToolPolicyEntry>[] {
    return this.#allow;
  }

  /**
   * Does this call run without asking?
   *
   * False when the policy is not honoured, when the tool is malformed or not
   * in normal form, reserved for the person, or declares `destructive`, and
   * when no entry names exactly this server and tool.
   */
  allows(tool: ToolDeclaration): boolean {
    if (!this.honor) return false;
    if (!isExactName(tool.name) || isPersonOnlyTool(tool.name) || tool.tier === 'destructive') return false;
    if (tool.server !== undefined && !isExactName(tool.server)) return false;

    return this.#allow.some((entry) => sameIdentity(tool, entry.server, entry.tool));
  }

  toJSON(): ToolPolicyJson {
    return { revision: this.revision, allow: this.#allow.map((entry) => ({ ...entry })), honor: this.honor };
  }
}

function sameIdentity(tool: ToolDeclaration, server: string | undefined, name: string): boolean {
  return tool.name === name && (tool.server ?? null) === (server ?? null);
}

/**
 * RFC 8785 (JCS) canonical JSON: keys sorted by UTF-16 code units, no
 * whitespace, ECMAScript number formatting. The same bytes in every language,
 * so the same arguments give the same digest.
 */
export function canonicalJson(value: unknown): string {
  if (value === null) return 'null';
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new TypeError('Canonical JSON has no representation for a non-finite number.');
    return JSON.stringify(value);
  }
  if (typeof value === 'string') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map((item) => canonicalJson(item)).join(',')}]`;
  if (typeof value === 'object') {
    const record = value as Record<string, unknown>;
    const keys = Object.keys(record).filter((key) => record[key] !== undefined).sort();
    return `{${keys.map((key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`).join(',')}}`;
  }
  throw new TypeError(`Canonical JSON has no representation for a ${typeof value}.`);
}

/** `sha256:` and the hex sha256 of the arguments' canonical JSON. Never the arguments. */
export function argsDigest(args: unknown): string {
  return `sha256:${createHash('sha256').update(canonicalJson(args ?? {}), 'utf8').digest('hex')}`;
}
