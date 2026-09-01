import { readFileSync, realpathSync, statSync } from 'node:fs';
import { join, resolve, sep } from 'node:path';
import { HarnessError } from './errors.js';

const SKILL_NAME = /^[a-z0-9][a-z0-9_-]*$/;
const TRAVERSAL = /(^|\/)\.\.(\/|$)/;

/**
 * Skills a mode can load into its system prompt, read from disk.
 *
 * ## The read is GUARDED, and that is the whole of it
 *
 * A skill name and a path both arrive from configuration or, worse, from a
 * model that was told it may fetch a referenced file. Reading either without
 * checking is a path traversal with extra steps, and the file it would reach is
 * on the machine running the agent.
 *
 * Three checks, in this order, each closing something the others do not:
 *
 *  1. the NAME matches a strict pattern — this is what stops `../` in the
 *     name itself, before it is ever joined to anything;
 *  2. the relative PATH is rejected lexically if it is absolute or contains a
 *     `..` segment;
 *  3. the resolved REAL path must still sit inside the skill's own real root,
 *     which is what catches a symlink pointing out of it — the one thing the
 *     first two checks cannot see.
 *
 * The third is not redundant. A lexically innocent `notes/link.md` that is a
 * symlink to `/etc/passwd` passes both earlier checks.
 */
export class SkillRegistry {
  readonly #root: string;

  constructor(root: string) {
    this.#root = resolve(root);
  }

  /**
   * The system prompt with each named skill appended as a tagged section.
   *
   * Returns the prompt UNCHANGED when no skills are named, rather than
   * appending an empty preamble that would tell the model skills are available
   * when none are.
   */
  augmentPrompt(systemPrompt: string, names: readonly string[]): string {
    const sections = names.map(
      (name) => `<skill name="${name}">\n${this.read(name, 'SKILL.md')}\n</skill>`,
    );

    if (sections.length === 0) {
      return systemPrompt;
    }

    return `${systemPrompt}\n\nThe following Harness-owned skills are available. Follow their routing instructions and use skill_read for referenced files. Do not copy skill files into the project workspace.\n\n${sections.join('\n\n')}`.trim();
  }

  /** Read one file from inside a skill. See the class docblock for the guard. */
  read(name: string, path: string): string {
    if (!SKILL_NAME.test(name)) {
      throw HarnessError.skillPathRefused(`the skill name [${name}] is not a valid name`);
    }

    const relative = path.replaceAll('\\', '/').trim();

    if (relative === '' || relative.startsWith('/') || TRAVERSAL.test(relative)) {
      throw HarnessError.skillPathRefused(`[${path}] must stay inside the skill`);
    }

    let skillRoot: string;
    let file: string;

    try {
      skillRoot = realpathSync.native(join(this.#root, name));
      file = realpathSync.native(join(this.#root, name, ...relative.split('/')));
    } catch {
      throw HarnessError.skillPathRefused(`[${name}/${relative}] was not found`);
    }

    // The check the lexical ones cannot make: a symlink inside the skill that
    // points out of it is lexically innocent and resolves elsewhere.
    if (!file.startsWith(skillRoot + sep) || !statSync(file).isFile()) {
      throw HarnessError.skillPathRefused(`[${name}/${relative}] resolves outside the skill`);
    }

    return readFileSync(file, 'utf8');
  }
}
