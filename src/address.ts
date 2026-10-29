import { createHash } from 'node:crypto';
import type { Participant } from './session.js';

export function validateAgent(agent: unknown): asserts agent is string | null {
  if (agent !== null && (typeof agent !== 'string' || /^[a-z][a-z0-9_-]{0,63}$/.exec(agent)?.[0] !== agent)) {
    throw new TypeError('Harness namespace must be a configured lowercase identifier of 1–64 ASCII bytes.');
  }
}

function utf8(value: unknown): value is string {
  return typeof value === 'string' && Buffer.from(value, 'utf8').toString('utf8') === value;
}

/** An explicit host-defined identity; its actor comes from a registered adapter. */
export class ParticipantRef implements Participant {
  readonly type: string;
  readonly id: string;

  constructor(type: string, id: string | number) {
    if (!utf8(type) || type.length === 0 || Buffer.byteLength(type) > 255
      || (typeof id !== 'string' && (typeof id !== 'number' || !Number.isSafeInteger(id)))
      || !utf8(String(id)) || String(id).length === 0 || Buffer.byteLength(String(id)) > 255) {
      throw new TypeError('Participant references require non-empty UTF-8 type and id, at most 255 bytes each.');
    }
    this.type = type;
    this.id = String(id);
    Object.freeze(this);
  }
}

export class ParticipantAdapters {
  readonly #actors = new Map<string, (participant: ParticipantRef) => object>();

  register(type: string, actor: (participant: ParticipantRef) => object): void {
    new ParticipantRef(type, 'validation');
    if (this.#actors.has(type)) throw new Error('A participant adapter is already registered for this type.');
    this.#actors.set(type, actor);
  }

  actor(participant: ParticipantRef, inheritedActor?: object): object {
    const resolve = this.#actors.get(participant.type);
    if (!resolve) throw new Error('No participant actor adapter is registered for this type.');
    const actor = inheritedActor ?? resolve(participant);
    if (actor === null || typeof actor !== 'object') throw new TypeError('Participant actor adapter must return an actor object.');
    return actor;
  }
}

/** Immutable address. Persona/display names are deliberately absent. */
export class SessionAddress {
  readonly participantType: string | null;
  readonly participantId: string | null;

  constructor(participant: Participant | null, readonly scope: string, readonly agent: string | null = null) {
    validateAgent(agent);
    if ((agent !== null || participant instanceof ParticipantRef) && (!utf8(scope) || (participant !== null && (!utf8(participant.type)
      || (typeof participant.id !== 'string' && (typeof participant.id !== 'number' || !Number.isSafeInteger(participant.id)))
      || !utf8(String(participant.id)))))) throw new TypeError('Harness addresses require valid UTF-8 and integer or string ids.');
    this.participantType = participant?.type ?? null;
    this.participantId = participant === null ? null : String(participant.id);
    Object.freeze(this);
  }

  sessionKey(): string {
    if (this.participantType === null || this.participantId === null) throw new Error('A session requires a participant.');
    if (this.agent === null) return `session:${digest('sha1', this.participantType).slice(0, 12)}:${this.participantId}:${this.scope}`;
    return `session:${this.#digest('prism-harness.session.v1')}`;
  }

  threadKey(): string {
    if (this.agent !== null) return this.#digest('prism-harness.thread.v1');
    // PHP's legacy json_encode escapes Unicode, slashes and DEL. Keep its bytes.
    const json = JSON.stringify([this.participantType, this.participantId, this.scope])
      .replace(/\//g, '\\/').replace(/[\u007f-\uffff]/g, c => `\\u${c.charCodeAt(0).toString(16).padStart(4, '0')}`);
    return digest('sha256', json);
  }

  indexKey(): string { return this.#digest('prism-harness.address.v1'); }

  #digest(domain: string): string {
    return digest('sha256', JSON.stringify([domain, this.agent, this.participantType, this.participantId, this.scope]));
  }
}

function digest(algorithm: string, value: string): string {
  return createHash(algorithm).update(value, 'utf8').digest('hex');
}
