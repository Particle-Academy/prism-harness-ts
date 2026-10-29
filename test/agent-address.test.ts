import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { mkdtemp, readdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { expect, it } from 'vitest';
import { AddressLimitExceeded, FileSessionStore, IndexLockScopeUnsupported, IndexRecoveryRequired, MemorySessionStore, ParticipantAdapters, ParticipantRef, PrismHarness, Session, SessionAddress, StoreAddressIndex, type JsonObject } from '../src/index.js';

async function setup() {
  const directory = await mkdtemp(join(tmpdir(), 'prism-agent-address-'));
  const store = new FileSessionStore(directory), adapters = new ParticipantAdapters();
  const harness = new PrismHarness({ drivers: { owned: () => store }, stores: { ephemeral: 'owned', durable: 'owned' }, participantAdapters: adapters });
  return { directory, store, adapters, harness };
}

it('inherits a typed participants bound actor without resolving its adapter again', async () => {
  const { harness, adapters } = await setup(), participant = new ParticipantRef('automation', 'child-owner');
  const actor = { principal: 'bound' }, later = { principal: 'later' };
  let resolutions = 0;
  expect(() => harness.pending(participant, 'support', actor).session('missing')).toThrow('actor adapter');
  adapters.register('automation', () => ++resolutions === 1 ? actor : later);
  const parent = harness.forAgent('support').for(participant).session('parent');
  const child = harness.pending(parent.participant, parent.agent, parent.actor).session('parent::sub::worker');
  expect(child.actor).toBe(actor); expect(child.participant).toBe(participant); expect(resolutions).toBe(1);
});

it('isolates named state, transcripts, capabilities and task lists while preserving legacy keys', async () => {
  const { harness } = await setup(), participant = { type: 'User', id: 7 };
  const legacy = harness.for(participant).session('same:/雪');
  const support = harness.forAgent('support').for(participant).session('same:/雪');
  const charlotte = harness.forAgent('charlotte').for(participant).session('same:/雪');
  expect(legacy.key()).toBe(`session:${createHash('sha1').update('User').digest('hex').slice(0, 12)}:7:same:/雪`);
  await support.usingMode('reply'); await support.usingCapability('private', { value: true });
  await support.tasks().add('support only', 'one');
  await support.thread().record([{ type: 'user', text: 'support only' }]);
  expect(await charlotte.mode()).toBeNull(); expect(await charlotte.capability('private')).toBeNull();
  expect(await charlotte.tasks().claim('worker')).toBeNull(); expect(await charlotte.thread().messages()).toEqual([]);
  expect(await legacy.thread().messages()).toEqual([]);
  expect(await harness.forAgent('support').for({ ...participant, id: '7' }).session('same:/雪').mode()).toBe('reply');
});

it('does not create files, index records or locks on noncreating reads', async () => {
  const { harness, directory } = await setup();
  const session = harness.forAgent('support').for({ type: 'User', id: 7 }).session('absent');
  const before = await readdir(directory);
  expect(await session.findThread()).toBeNull(); expect(await harness.forAgent('support').findShared('absent')).toBeNull();
  expect(await readdir(directory)).toEqual(before);
  await session.thread().record([{ type: 'user', text: 'one' }]);
  expect(await (await session.findThread())?.count()).toBe(1);
  expect(await harness.forAgent('charlotte').for(session.participant).session('absent').findThread()).toBeNull();
});

it('requires a registered actor for explicit references and keeps the actor separate from ownership', async () => {
  const { harness, adapters } = await setup(), participant = new ParticipantRef('automation', 'bot/雪'), actor = { service: true };
  expect(() => harness.forAgent('support').for(participant).session('s')).toThrow('actor adapter');
  adapters.register('automation', () => actor);
  const session = harness.forAgent('support').for(participant).session('s');
  expect(session.actor).toBe(actor); expect(session.participant).toEqual(participant);
  expect(() => adapters.register('automation', () => actor)).toThrow('already registered');
  expect(() => harness.for(participant).asActor(actor)).toThrow('registered actor adapter');
  const plain = { type: 'User', id: 1 };
  expect(harness.for(plain).asActor(actor).session('s').key()).toBe(harness.for(plain).session('s').key());
});

it('uses immutable configured ids and rejects hostile namespaces without normalization', async () => {
  const { harness } = await setup(), participant = { type: 'User', id: 7 };
  const session = harness.forAgent('support').for(participant).session('s'), key = session.key();
  participant.id = 9; participant.type = 'Team';
  expect(session.key()).toBe(key);
  expect(() => { (session as unknown as { address: SessionAddress }).address = new SessionAddress(participant, 's', 'other'); }).toThrow();
  for (const agent of ['', 'Support', ' support', 'support ', 'support\n', '../support', 'a/b', 'sυpport', 'a'.repeat(65)]) {
    expect(() => harness.forAgent(agent)).toThrow();
  }
});

it('records bounded ownership metadata and preserves native append logs', async () => {
  const { harness, store } = await setup(), participant = { type: 'User', id: 7 };
  const session = harness.forAgent('support').for(participant).session('s'), index = new StoreAddressIndex(store);
  expect((await index.page(participant)).entries).toEqual([]);
  await session.usingMode('reply'); await session.thread().record([{ type: 'user', text: 'one' }]);
  const rows = await index.page(participant);
  expect(rows.entries).toHaveLength(1); expect(rows.entries[0]?.address_key).toBe(session.address.indexKey());
  expect(await store.logLength(`${session.key()}:thread`)).toBe(1);
  expect((await index.page(participant, 1)).entries).toEqual([]);
  await expect(index.page(participant, 0, 101)).rejects.toThrow();
  await harness.forAgent('support').tasks('same').add('owned', 'one');
  expect(await harness.forAgent('charlotte').tasks('same').claim('worker')).toBeNull();
  expect(await harness.tasks('same').claim('worker')).toBeNull();
});

it('executes every shared vector against an independently authored expectation', () => {
  const document = JSON.parse(readFileSync(new URL('./fixtures/harness-agent-addresses.json', import.meta.url), 'utf8')) as {
    cases: { id: string; input: { participant: { type: string; id: number | string } | null; scope: string; agent: string | null }; expected: Record<string, string | null> }[];
  };
  expect(document.cases).toHaveLength(46);
  for (const row of document.cases) {
    let result;
    try {
      const address = new SessionAddress(row.input.participant, row.input.scope, row.input.agent);
      result = { session_key: row.input.participant === null ? null : address.sessionKey(), thread_key: address.threadKey(), index_key: address.indexKey() };
    } catch (error) {
      if (!(error instanceof TypeError)) throw error;
      result = { error: 'invalid_address' };
    }
    expect(result, row.id).toEqual(row.expected);
  }
});

it('refuses resource data if its ownership index cannot be written', async () => {
  const { store } = await setup();
  class RefusingIndex extends StoreAddressIndex {
    override async record(): Promise<void> { throw new Error('Synthetic index failure'); }
  }
  const session = new Session({ participant: { type: 'User', id: 7 }, scope: 's', agent: 'support', ephemeral: store, durable: store, addressIndex: new RefusingIndex(store) });
  await expect(session.usingMode('reply')).rejects.toThrow('Synthetic index failure');
  expect(await store.get(`${session.key()}:ephemeral`)).toBeNull();
});

it('crosses a fixed page boundary with no default quota and applies an explicit quota before state', async () => {
  const { harness, store } = await setup(), participant = { type: 'User', id: 7 };
  for (let i = 0; i < 101; i++) await harness.forAgent('support').for(participant).session(`scope:${i}`).usingMode('reply');
  const index = new StoreAddressIndex(store), first = await index.page(participant), second = await index.page(participant, first.next_cursor!);
  expect(first.entries).toHaveLength(100); expect(first.next_cursor).toBe(100);
  expect(second.entries).toHaveLength(1); expect(second.next_cursor).toBeNull();
  const limited = new PrismHarness({ drivers: { owned: () => store }, stores: { ephemeral: 'owned', durable: 'owned' }, maxAddressesPerParticipant: 101 });
  const extra = limited.forAgent('support').for(participant).session('scope:extra');
  await expect(extra.usingMode('refused')).rejects.toBeInstanceOf(AddressLimitExceeded);
  await expect(extra.usingMode('refused')).rejects.toMatchObject({ code: 'address_limit_exceeded' });
  expect(await extra.mode()).toBeNull();
  await limited.forAgent('support').for(participant).session('scope:0').usingMode('still usable');
});

it('recovers every interrupted page publication before any resource data write', async () => {
  for (const phase of ['before', 'after']) {
  for (const failAt of [1, 2, 3, 4]) {
    const directory = await mkdtemp(join(tmpdir(), 'prism-index-crash-'));
    class InterruptedStore extends FileSessionStore {
      armed = true; writes = 0;
      override async put(key: string, value: JsonObject, ttl: number | null = null): Promise<void> {
        const interrupt = this.armed && key.startsWith('harness-address-index:') && ++this.writes === failAt;
        if (interrupt && phase === 'before') throw new Error('Synthetic publication interruption');
        await super.put(key, value, ttl);
        if (interrupt && phase === 'after') throw new Error('Synthetic publication interruption');
      }
    }
    const store = new InterruptedStore(directory), participant = { type: 'User', id: 7 };
    const harness = new PrismHarness({ drivers: { owned: () => store }, stores: { ephemeral: 'owned', durable: 'owned' } });
    const session = harness.forAgent('support').for(participant).session('s'), index = new StoreAddressIndex(store);
    await expect(session.usingMode('reply')).rejects.toThrow('Synthetic publication interruption');
    expect(await store.get(`${session.key()}:ephemeral`)).toBeNull();
    if (phase === 'before' && failAt === 1) expect((await index.page(participant)).entries).toEqual([]);
    else if (phase === 'after' && failAt === 4) expect((await index.page(participant)).entries).toHaveLength(1);
    else await expect(index.page(participant)).rejects.toBeInstanceOf(IndexRecoveryRequired);
    store.armed = false;
    await index.recover(participant); await index.recover(participant);
    await session.usingMode('reply');
    await session.usingMode('recovery is idempotent');
    expect((await index.page(participant)).entries).toHaveLength(1);
    expect(await harness.forAgent('support').for(participant).session('s').mode()).toBe('recovery is idempotent');
  }
  }
});

it('refuses indexed namespaces on process-only or undeclared locks with a typed error', async () => {
  expect(() => new StoreAddressIndex(new MemorySessionStore())).toThrow('durable store');
  const directory = await mkdtemp(join(tmpdir(), 'prism-index-scope-'));
  class ProcessOnly extends FileSessionStore { override lockScope(): 'process' { return 'process'; } }
  const store = new ProcessOnly(directory);
  const harness = new PrismHarness({ drivers: { owned: () => store }, stores: { ephemeral: 'owned', durable: 'owned' } });
  expect(() => harness.forAgent('support').for({ type: 'User', id: 7 }).session('s')).toThrow(IndexLockScopeUnsupported);
  expect(() => new Session({ participant: { type: 'User', id: 7 }, scope: 's', agent: 'support', ephemeral: store, durable: store })).toThrow(IndexLockScopeUnsupported);
  await harness.for({ type: 'User', id: 7 }).session('s').usingMode('legacy works');
});

it('supports process-local memory state with a durable shared file slot', async () => {
  const { store } = await setup(), memory = new MemorySessionStore();
  const harness = new PrismHarness({ drivers: { memory: () => memory, files: () => store }, stores: { ephemeral: 'memory', durable: 'files' } });
  const session = harness.forAgent('support').for({ type: 'User', id: 7 }).session('s');
  await session.usingMode('memory works');
  await session.thread().record([{ type: 'user', text: 'durable' }]);
  expect(await session.mode()).toBe('memory works'); expect(await session.thread().count()).toBe(1);
});

it('refuses undeclared custom sharing and lock scopes before named state is written', async () => {
  const { store, directory } = await setup();
  const undeclared = {
    get: store.get.bind(store), put: store.put.bind(store), forget: store.forget.bind(store),
    withLock: store.withLock.bind(store), durability: store.durability.bind(store),
  };
  const harness = new PrismHarness({ drivers: { owned: () => undeclared }, stores: { ephemeral: 'owned', durable: 'owned' } });
  expect(() => harness.forAgent('support').for({ type: 'User', id: 7 }).session('s')).toThrow(IndexLockScopeUnsupported);
  expect(() => new Session({ participant: { type: 'User', id: 7 }, scope: 's', agent: 'support', ephemeral: undeclared, durable: undeclared })).toThrow(IndexLockScopeUnsupported);
  expect(await readdir(directory)).toEqual([]);
  await harness.for({ type: 'User', id: 7 }).session('s').usingMode('legacy works');
});
