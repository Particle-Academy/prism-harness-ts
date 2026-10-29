export class RegistrationCollision extends Error {
  readonly code = 'registration_collision';

  constructor(agent: string | null, name: string, previous: string, incoming: string) {
    super(`Harness registry collision: agent [${agent ?? 'legacy'}], name [${name}], owners [${previous}] and [${incoming}]. Use replace naming predecessor owner [${previous}].`);
  }
}

/** Diagnostic labels are host configuration, never permission grants. */
export function registrationOwner(owner: string | undefined, agent: string | null): string {
  const label = owner ?? (agent === null ? 'legacy' : undefined);
  if (typeof label !== 'string' || label.length === 0 || Buffer.byteLength(label) > 128
    || Buffer.from(label, 'utf8').toString('utf8') !== label || /[\u0000-\u001f\u007f]/.test(label)) {
    throw new TypeError('A named registration requires a non-empty UTF-8 owner label of at most 128 bytes without control characters.');
  }
  return label;
}
