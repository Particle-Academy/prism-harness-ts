import { describe, expect, it } from 'vitest';
import { RESERVED_TOOLS, ToolPolicy, isPersonOnlyTool } from '../src/index.js';

// G-360: a plan is a person's agreement. No host policy may let a plan tool
// run without asking, however the name is spelled.

describe('plan tools in a host tool policy (G-360)', () => {
  it.each([
    ['plan_draft', 'plan_draft'],
    ['plan_propose_change', 'plan_propose_change'],
    ['plan_pause_for_person', 'plan_pause_for_person'],
    ['plan_complete_step', 'plan_complete_step'],
    ['upper case', 'PLAN_DRAFT'],
    ['trailing space', 'plan_draft '],
    ['trailing NBSP', 'plan_draft '],
    ['leading ZWSP', '​plan_draft'],
    ['Cyrillic a', 'plаn_draft'],
  ])('refuses a policy that names a plan tool, in any spelling: %s', (_name, tool) => {
    expect(() => ToolPolicy.fromJson({ revision: 'r1', allow: [{ tool }], honor: true }))
      .toThrow(expect.objectContaining({ code: 'tool_policy_refused' }));
  });

  it('never lets a plan tool run under a policy, even one built without the tool list', () => {
    const policy = ToolPolicy.fromJson({ revision: 'r1', allow: [{ tool: 'search' }], honor: true });

    for (const name of RESERVED_TOOLS) {
      expect(policy.allows({ name })).toBe(false);
      expect(isPersonOnlyTool(name)).toBe(true);
      expect(isPersonOnlyTool(name.toUpperCase())).toBe(true);
    }
  });
});
