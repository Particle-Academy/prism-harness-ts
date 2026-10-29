import { createHash } from 'node:crypto';
import { SessionAddress } from './address.js';
import type { Participant } from './session.js';
import type { JsonObject } from './json.js';
import { isJsonObject } from './json.js';
import type { SessionStore } from './stores/session-store.js';
import { supportsAppendLog } from './stores/session-store.js';
import { HarnessError } from './errors.js';

export class AddressLimitExceeded extends Error {
  readonly code = 'address_limit_exceeded';
  constructor() { super('The configured max_addresses_per_participant limit was reached; no resource was written.'); this.name = 'AddressLimitExceeded'; }
}

export class IndexLockScopeUnsupported extends Error {
  readonly code = 'index_lock_scope_unsupported';
  constructor() { super('Indexed namespaces require declared sharing and lock scopes, with the lock covering the sharing scope.'); this.name = 'IndexLockScopeUnsupported'; }
}

export function storeLockRank(store: SessionStore): number {
  const ranks = { process: 0, host: 1, distributed: 2 };
  const sharing = store.sharingScope?.(), locking = store.lockScope?.();
  if (!sharing || !locking || sharing === 'unsupported' || locking === 'unsupported'
    || !Object.hasOwn(ranks, sharing) || !Object.hasOwn(ranks, locking) || ranks[locking] < ranks[sharing]) throw new IndexLockScopeUnsupported();
  return ranks[locking];
}

export class IndexRecoveryRequired extends Error {
  readonly code = 'index_recovery_required';
  constructor() { super('Recover the interrupted address append before reading a complete index or planning erasure.'); this.name = 'IndexRecoveryRequired'; }
}

export function addressLimit(value: number | null): number | null {
  if (value !== null && (!Number.isSafeInteger(value) || value < 1)) throw new TypeError('max_addresses_per_participant must be a positive integer or null.');
  return value;
}

export interface AddressIndexPage { entries: JsonObject[]; next_cursor: number | null; }

/** Fixed pages and a small commit head; no participant-wide value or ceiling. */
export class StoreAddressIndex {
  static readonly PAGE_SIZE = 100;

  constructor(private readonly store: SessionStore, private readonly maxAddressesPerParticipant: number | null = null) {
    if (store.durability() !== 'durable') throw new HarnessError('unsafe_state_configuration', 'The address index requires a durable store.');
    storeLockRank(store);
    addressLimit(maxAddressesPerParticipant);
  }

  #root(type: string | null, id: string | null): string {
    return `harness-address-index:v1:${createHash('sha256').update(JSON.stringify([type, id])).digest('hex')}`;
  }

  #head(value: JsonObject | null): JsonObject & { count: number; pages: number } {
    if (value === null) return { version: 1, count: 0, pages: 0, pending: null };
    if (value.version !== 1 || typeof value.count !== 'number' || !Number.isSafeInteger(value.count) || value.count < 0
      || value.pages !== Math.ceil(value.count / StoreAddressIndex.PAGE_SIZE)
      || (value.pending !== null && !isJsonObject(value.pending))) throw new Error('Invalid harness address index head.');
    return value as JsonObject & { count: number; pages: number };
  }

  async #pageIds(root: string, number: number): Promise<string[]> {
    const page = await this.store.get(`${root}:page:${number}`);
    if (page === null) return [];
    if (page.version !== 1 || !Array.isArray(page.ids) || page.ids.length > StoreAddressIndex.PAGE_SIZE
      || page.ids.some(id => typeof id !== 'string' || /^[a-f0-9]{64}$/.exec(id)?.[0] !== id)) throw new Error('Invalid harness address index page.');
    return page.ids as string[];
  }

  async #recover(root: string, head: JsonObject & { count: number; pages: number }): Promise<JsonObject & { count: number; pages: number }> {
    if (head.pending === null) return head;
    const pending = head.pending;
    if (!isJsonObject(pending) || typeof pending.hash !== 'string' || /^[a-f0-9]{64}$/.exec(pending.hash)?.[0] !== pending.hash
      || pending.position !== head.count || !isJsonObject(pending.entry) || pending.entry.position !== head.count) throw new Error('Invalid harness address append journal.');
    const number = Math.floor(head.count / StoreAddressIndex.PAGE_SIZE), offset = head.count % StoreAddressIndex.PAGE_SIZE;
    const ids = await this.#pageIds(root, number);
    if (ids.length === offset) { ids.push(pending.hash); await this.store.put(`${root}:page:${number}`, { version: 1, ids }); }
    else if (ids.length !== offset + 1 || ids[offset] !== pending.hash) throw new Error('Conflicting harness address append journal.');
    await this.store.put(`${root}:entry:${pending.hash}`, pending.entry);
    const committed = { version: 1, count: head.count + 1, pages: number + 1, pending: null };
    // This is the visibility/commit point. Resource writes happen only after it.
    await this.store.put(`${root}:head`, committed);
    return committed;
  }

  async record(address: SessionAddress, slot: 'ephemeral' | 'durable', kind: 'state' | 'lock' | 'tasks' | 'thread', key: string): Promise<void> {
    const root = this.#root(address.participantType, address.participantId);
    await this.store.withLock(root, async () => {
      const head = await this.#recover(root, this.#head(await this.store.get(`${root}:head`)));
      const hash = address.indexKey();
      const identity = { agent: address.agent, participant_type: address.participantType, participant_id: address.participantId, scope: address.scope };
      const existing = await this.store.get(`${root}:entry:${hash}`);
      if (existing !== null && (!isJsonObject(existing.address) || Object.entries(identity).some(([key, value]) => (existing.address as JsonObject)[key] !== value)
        || typeof existing.position !== 'number' || !Number.isSafeInteger(existing.position) || existing.position < 0 || existing.position >= head.count
        || !Array.isArray(existing.resources))) throw new Error('Invalid harness address index entry.');
      if (existing !== null) {
        const position = existing.position as number, ids = await this.#pageIds(root, Math.floor(position / StoreAddressIndex.PAGE_SIZE));
        if (ids[position % StoreAddressIndex.PAGE_SIZE] !== hash) throw new Error('Unlinked harness address index entry.');
      }
      const resources = isJsonObject(existing) && Array.isArray(existing.resources) ? [...existing.resources] : [];
      const resource = { slot, kind, key };
      if (resources.some(item => isJsonObject(item) && item.slot === slot && item.kind === kind && item.key === key)) return;
      resources.push(resource);
      const entry = { address: identity, resources, position: existing?.position ?? head.count };
      if (existing !== null) { await this.store.put(`${root}:entry:${hash}`, entry); return; }
      if (this.maxAddressesPerParticipant !== null && head.count >= this.maxAddressesPerParticipant) {
        throw new AddressLimitExceeded();
      }
      const prepared = { ...head, pending: { hash, position: head.count, entry } };
      await this.store.put(`${root}:head`, prepared);
      await this.#recover(root, prepared);
    });
  }

  async recover(participant: Participant | null): Promise<void> {
    const root = this.#root(participant?.type ?? null, participant === null ? null : String(participant.id));
    await this.store.withLock(root, async () => { await this.#recover(root, this.#head(await this.store.get(`${root}:head`))); });
  }

  async page(participant: Participant | null, cursor = 0, limit = 100): Promise<AddressIndexPage> {
    if (!Number.isSafeInteger(cursor) || cursor < 0 || !Number.isInteger(limit) || limit < 1 || limit > 100) throw new TypeError('Invalid address index page cursor or limit.');
    const root = this.#root(participant?.type ?? null, participant === null ? null : String(participant.id));
    const head = this.#head(await this.store.get(`${root}:head`)), end = Math.min(cursor + limit, head.count), entries: JsonObject[] = [];
    if (head.pending !== null) throw new IndexRecoveryRequired();
    let pageNumber = -1, ids: string[] = [];
    for (let position = cursor; position < end; position++) {
      const number = Math.floor(position / StoreAddressIndex.PAGE_SIZE);
      if (number !== pageNumber) { ids = await this.#pageIds(root, number); pageNumber = number; }
      const hash = ids[position % StoreAddressIndex.PAGE_SIZE];
      if (hash === undefined) throw new Error('Missing committed harness address page member.');
      const entry = await this.store.get(`${root}:entry:${hash}`);
      if (entry === null || entry.position !== position) throw new Error('Missing committed harness address index entry.');
      entries.push({ address_key: hash, ...entry });
    }
    return { entries, next_cursor: end < head.count ? end : null };
  }
}

/** Preserve append-log capabilities and existing transcript bytes while indexing writes. */
export function indexedStore(inner: SessionStore, index: StoreAddressIndex, address: SessionAddress, slot: 'ephemeral' | 'durable', kind: 'state' | 'tasks' | 'thread' = 'state'): SessionStore {
  const wrapper: SessionStore = {
    lockScope: () => inner.lockScope?.() ?? 'unsupported',
    sharingScope: () => inner.sharingScope?.() ?? 'unsupported',
    get: key => inner.get(key),
    put: async (key, payload, ttl) => { await index.record(address, slot, kind, key); await inner.put(key, payload, ttl); },
    forget: key => inner.forget(key),
    durability: () => inner.durability(),
    withLock: async (key, callback, ttl, wait) => { await index.record(address, slot, 'lock', key); return inner.withLock(key, callback, ttl, wait); },
  };
  if (supportsAppendLog(inner)) {
    return Object.assign(wrapper, {
      readLog: (key: string) => inner.readLog(key),
      logLength: (key: string) => inner.logLength(key),
      clearLog: (key: string) => inner.clearLog(key),
      appendLog: async (key: string, entries: readonly JsonObject[]) => { await index.record(address, slot, kind, key); await inner.appendLog(key, entries); },
    });
  }
  return wrapper;
}
