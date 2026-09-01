import type { ModeRegistry } from './modes.js';
import type { SessionStoreManager } from './stores/store-manager.js';
import type { ToolRegistry } from './tools.js';
import type { ToolAuthorizer } from './tools.js';

export interface DoctorFinding {
  check: string;
  ok: boolean;
  detail: string;
}

export interface DoctorReport {
  findings: readonly DoctorFinding[];
  problems: number;
  ok: boolean;
  summary(): string;
}

export interface DoctorSubjects {
  modes?: ModeRegistry;
  tools?: ToolRegistry;
  stores?: SessionStoreManager;
  authorizer?: ToolAuthorizer;
}

/**
 * Check the harness configuration BEFORE a run does it for you.
 *
 * Every check here corresponds to a failure this package already refuses at
 * runtime. Those refusals are correct and they are also LATE: a mode nobody has
 * entered yet keeps its broken subagent reference until someone switches to it,
 * and the first person to find out is a user mid-conversation.
 *
 * So this resolves EVERY mode rather than the default one, and reports what a
 * run would have thrown. A CLI is left to the consumer — this returns the
 * report as data, which is also what makes it testable and what lets a health
 * endpoint serve it.
 */
export function diagnose(subjects: DoctorSubjects): DoctorReport {
  const findings: DoctorFinding[] = [];

  if (subjects.stores !== undefined) {
    findings.push(checkStores(subjects.stores));
  }

  if (subjects.authorizer !== undefined) {
    findings.push({
      check: 'authorizer',
      ok: true,
      detail: subjects.authorizer.enabled
        ? 'enabled — tools are filtered per run and per call'
        : 'DISABLED — every registered tool is offered to every run',
    });
  }

  if (subjects.modes !== undefined) {
    findings.push(...checkModes(subjects.modes, subjects.tools));
  }

  const problems = findings.filter((finding) => !finding.ok).length;

  return {
    findings,
    problems,
    ok: problems === 0,
    summary(): string {
      const lines = findings.map(
        (finding) => `  ${finding.ok ? 'ok  ' : 'FAIL'} ${finding.check}: ${finding.detail}`,
      );

      lines.unshift(
        problems === 0
          ? 'Harness configuration is consistent.'
          : `${problems} problem(s) found in the harness configuration.`,
      );

      return lines.join('\n');
    },
  };
}

function checkStores(stores: SessionStoreManager): DoctorFinding {
  try {
    const durable = stores.durable();
    stores.ephemeral();

    return { check: 'stores', ok: true, detail: `durable slot is ${durable.durability()}` };
  } catch (error) {
    return { check: 'stores', ok: false, detail: message(error) };
  }
}

/**
 * Resolve EVERY mode, and check the tools each one names.
 *
 * A mode listing a tool the registry cannot produce fails only when a run
 * reaches for it — which is to say, in front of whoever is talking to the
 * agent. Checked here against the registry's static names; a tool that only a
 * provider can supply needs a session, so its absence is reported as unknown
 * rather than missing.
 */
function checkModes(modes: ModeRegistry, tools?: ToolRegistry): DoctorFinding[] {
  const findings: DoctorFinding[] = [];
  const names = modes.names();

  if (names.length === 0) {
    return [{ check: 'modes', ok: false, detail: 'no modes are configured' }];
  }

  const known = new Set(tools?.names() ?? []);

  for (const name of names) {
    try {
      const mode = modes.resolve(name);
      const detail: string[] = [
        `${mode.tools.length} tool(s)`,
        `${mode.maxSteps} step(s)`,
        `${Object.keys(mode.subagents).length} subagent(s)`,
      ];

      if (mode.requiresApproval.length > 0) {
        detail.push(`approval: ${mode.requiresApproval.join(', ')}`);
      }

      const missing =
        tools === undefined || mode.tools.includes('*')
          ? []
          : mode.tools.filter((tool) => !known.has(tool));

      findings.push(
        missing.length > 0
          ? {
              check: `mode:${name}`,
              ok: false,
              detail: `names ${missing.length} tool(s) the registry cannot produce: ${missing.join(', ')}`,
            }
          : { check: `mode:${name}`, ok: true, detail: detail.join(', ') },
      );
    } catch (error) {
      findings.push({ check: `mode:${name}`, ok: false, detail: message(error) });
    }
  }

  // The default has to resolve, or every session that does not name a mode
  // fails — and that is the ordinary path, not an edge.
  try {
    modes.resolve(null);
    findings.push({ check: 'default mode', ok: true, detail: modes.default() });
  } catch (error) {
    findings.push({ check: 'default mode', ok: false, detail: message(error) });
  }

  return findings;
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
