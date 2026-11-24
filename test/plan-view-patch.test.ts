import { describe, expect, it } from 'vitest';
import { PlanIds, PlanLimits, PlanMachine, PlanView, type Plan } from '../src/index.js';

// G-360: a patch carries changed steps by id and cannot say where a new step
// goes. A pause inserts the person's step BEFORE the current one, so a reader
// applying a patch put it last. Any change to the list of ids is a snapshot.

function plan(): { plan: Plan; ids: PlanIds; limits: PlanLimits } {
  const limits = new PlanLimits();
  let n = 0;
  const ids = new PlanIds((prefix) => `${prefix}_${String(++n).padStart(26, '0')}`);
  const steps = [{ title: 'One', instruction: 'Do one.', owner: 'agent' }, { title: 'Two', instruction: 'Do two.', owner: 'agent' }];
  const drafted = PlanMachine.propose(PlanMachine.checkDraft({ goal: 'g', steps }, limits), 'd', ids, 0, null, limits);

  return { plan: PlanMachine.activate(drafted, 'd', 0, null), ids, limits };
}

describe('plan_updated patches', () => {
  it('patches only the changed steps while the list of steps is the same', () => {
    const { plan: first } = plan();
    const before = PlanView.steps(first);
    const claimed = PlanMachine.claim(first, first.steps[0]!.id, 'w', 100, 0);
    const update = PlanView.update(claimed, before);
    expect(update.snapshot).toBe(false);
    expect(update.steps.map((step) => step.id)).toEqual([claimed.steps[0]!.id]);
  });

  it('sends a snapshot, in order, when a step is inserted before the current one', () => {
    const { plan: first, ids, limits } = plan();
    const claimed = PlanMachine.claim(first, first.steps[0]!.id, 'w', 100, 0);
    const before = PlanView.steps(claimed);
    const [paused] = PlanMachine.pause(claimed, { kind: 'input', question: 'Which?' }, null, ids, limits, 0, null);
    const update = PlanView.update(paused, before);
    expect(update.snapshot).toBe(true);
    expect(update.steps.map((step) => step.owner)).toEqual(['human', 'agent', 'agent']);
  });
});
