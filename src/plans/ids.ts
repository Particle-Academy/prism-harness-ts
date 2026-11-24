import { randomBytes } from 'node:crypto';

const CROCKFORD = '0123456789ABCDEFGHJKMNPQRSTVWXYZ';

const STEP_PATTERN = /^ps_[0-9A-HJKMNP-TV-Z]{26}$/;

/**
 * Host-minted plan and step ids: `pl_` / `ps_` and a ULID (G-360).
 *
 * Never chosen by the model. The generator is replaceable so the parity corpus
 * can mint the same ids in all three languages.
 */
export class PlanIds {
  /** @param mint given a prefix (`pl` or `ps`), returns a full id */
  constructor(private readonly mint: ((prefix: 'pl' | 'ps') => string) | null = null) {}

  plan(): string {
    return this.mint !== null ? this.mint('pl') : `pl_${PlanIds.ulid()}`;
  }

  step(): string {
    return this.mint !== null ? this.mint('ps') : `ps_${PlanIds.ulid()}`;
  }

  static isStepId(id: unknown): id is string {
    return typeof id === 'string' && STEP_PATTERN.test(id);
  }

  static ulid(): string {
    let time = Date.now();
    let out = '';
    for (let i = 0; i < 10; i += 1) {
      out = CROCKFORD[time % 32]! + out;
      time = Math.floor(time / 32);
    }
    const bits = [...randomBytes(10)].map((byte) => byte.toString(2).padStart(8, '0')).join('');
    for (let i = 0; i < bits.length; i += 5) out += CROCKFORD[Number.parseInt(bits.slice(i, i + 5), 2)]!;

    return out;
  }
}
