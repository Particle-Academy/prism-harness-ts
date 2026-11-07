import {mkdtemp} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {afterEach, describe, expect, it, vi} from 'vitest';
import {AgentRuntime, FileSessionStore, HarnessError, MemorySessionStore, ModeRegistry, PrismHarness, ToolRegistry, recordApproval,
  type JsonObject, type JsonValue, type LlmRequest, type LlmResponse, type Session} from '../src/index.js';
import {approvalToolScope, assistantRow} from '../src/thread-rows.js';

afterEach(() => vi.restoreAllMocks());

async function exercise(options: {
  requested?: readonly string[]; decision?: boolean; structured?: boolean; ceiling?: string[]; ceilingAfter?: string[];
  seed?: {scope?: JsonValue}; lateAfter?: boolean; legacyWidening?: boolean; refreshBeforeLock?: boolean;
}) {
  const directory = await mkdtemp(join(tmpdir(), 'prism-approval-scope-'));
  const open = async () => {
    const harness = new PrismHarness({drivers: {memory: () => new MemorySessionStore(), files: () => new FileSessionStore(directory)}, stores: {ephemeral: 'memory', durable: 'files'}});
    const session = harness.for({type: 'User', id: 1}).session('scope');
    await session.usingMode('chat'); await session.usingProvider('synthetic'); await session.usingModel('offline');
    return session;
  };
  const counts = {scope_safe: 0, scope_ceiling: 0, scope_late: 0};
  const offers: string[][] = [];
  let ceiling = options.ceiling ?? ['scope_safe', 'scope_ceiling'];
  let late = false;
  const responses: LlmResponse[] = [
    ...(options.decision === undefined || options.seed ? [] : [{text: '', finishReason: 'tool_calls', toolCalls: [{id: 'safe-call', name: 'scope_safe', arguments: {}}]}]),
    {text: '', finishReason: 'tool_calls', toolCalls: [{id: 'ceiling-call', name: 'scope_ceiling', arguments: {}}]},
    {text: '{"ok":true}', structured: {ok: true}, finishReason: 'stop'},
  ];
  const client = async (request: LlmRequest): Promise<LlmResponse> => {
    offers.push(request.tools.map(tool => tool.name));
    expect(JSON.stringify(request.messages)).not.toContain('harness_tool_names');
    const response = responses.shift();
    if (!response) throw new Error('Unexpected extra native model call');
    return response;
  };
  const runtime = () => {
    const tools = new ToolRegistry();
    for (const name of ['scope_safe', 'scope_ceiling', ...(late ? ['scope_late'] : [])] as (keyof typeof counts)[]) {
      tools.register({name, handle: () => {counts[name]++; return 'done';}});
    }
    return new AgentRuntime({client, tools, allowLegacyToolListWidening: options.legacyWidening,
      modes: new ModeRegistry({default: 'chat', modes: {chat: {tools: ceiling, max_steps: 4, requires_approval: options.refreshBeforeLock ? ['scope_safe', 'scope_ceiling'] : ['scope_safe']}}})});
  };
  const session = await open();
  let pendingId: string | undefined;
  let storedScope: JsonValue | undefined;
  let refusal: string | null = null;
  try {
    if (options.seed) {
      const row: JsonObject = assistantRow('', [{id: 'safe-call', name: 'scope_safe', arguments: {}}], {}, [{approval_id: 'approval-1', tool_call_id: 'safe-call'}]);
      if ('scope' in options.seed) row.harness_tool_names = options.seed.scope!;
      await session.thread().record([row]);
      pendingId = 'approval-1';
    } else {
      const first = await runtime().send(session, 'A test turn.', options.requested);
      pendingId = first.pendingApprovals[0]?.id;
      if (options.decision !== undefined) expect(first.pendingApprovals).toHaveLength(1);
    }
    if (options.decision !== undefined) {
      expect(counts).toEqual({scope_safe: 0, scope_ceiling: 0, scope_late: 0});
      const rows = await session.thread().messages();
      storedScope = rows.find(entry => entry.message.type === 'assistant')?.message.harness_tool_names;
      const fresh = await open();
      expect(fresh).not.toBe(session); expect(fresh.key()).toBe(session.key());
      ceiling = options.ceilingAfter ?? ceiling;
      late = options.lateAfter ?? false;
      await recordApproval(fresh, pendingId!, options.decision);
      if (options.refreshBeforeLock) {
        const nativeLock: typeof fresh.lock = fresh.lock.bind(fresh);
        let scheduled = false;
        vi.spyOn(fresh, 'lock').mockImplementation(async <T>(callback: (session: Session) => T | Promise<T>, ttl = 10, wait = 5): Promise<T> => {
          if (!scheduled) {
            scheduled = true;
            // Schedule a concurrent durable writer before the real lock. Its
            // native pending row belongs to a narrower turn, even if a person
            // affirms the model's request for an unoffered gated tool.
            await fresh.thread().record([
              {type: 'user', content: 'A concurrent narrowed turn.'},
              {...assistantRow('', [{id: 'new-call', name: 'scope_ceiling', arguments: {}}], {}, [{approval_id: 'new-approval', tool_call_id: 'new-call'}]), harness_tool_names: ['scope_safe']},
              {type: 'tool_result', tool_results: [], tool_approval_responses: [{approval_id: 'new-approval', approved: true, reason: null}]},
            ]);
          }
          return nativeLock(callback, ttl, wait);
        });
      }
      if (options.structured) await runtime().sendStructured(fresh, '', {type: 'object', properties: {ok: {type: 'boolean'}}, required: ['ok']});
      else await runtime().send(fresh, '');
    }
  } catch (failure) {
    if (!(failure instanceof HarnessError) || failure.code !== 'tool_not_available') throw failure;
    refusal = failure.code;
  }
  expect(offers.length).toBeGreaterThan(0);
  return {counts, offers, storedScope, refusal};
}

describe('durable approval tool scope', () => {
  it('keeps a fresh narrowed offer closed', async () => {
    const row = await exercise({requested: ['scope_safe']});
    expect(row.counts.scope_ceiling).toBe(0); expect(row.offers[0]).toEqual(['scope_safe']); expect(row.refusal).toBe('tool_not_available');
  });
  it('keeps a positive full-ceiling control reachable', async () => {
    const row = await exercise({});
    expect(row.counts.scope_ceiling).toBe(1); expect(row.refusal).toBeNull();
  });
  it.each([true, false].flatMap(decision => [true, false].map(structured => ({decision, structured}))))('resumes without widening ($decision/$structured)', async options => {
    const row = await exercise({...options, requested: ['scope_safe']});
    expect(row.counts.scope_safe).toBe(options.decision ? 1 : 0); expect(row.counts.scope_ceiling).toBe(0);
    expect(row.storedScope).toEqual(['scope_safe']); expect(row.offers.every(offer => !offer.includes('scope_ceiling'))).toBe(true);
  });
  it('uses the narrower current ceiling', async () => {
    const row = await exercise({requested: ['scope_safe'], decision: true, ceilingAfter: []});
    expect(row.counts.scope_safe).toBe(0); expect(row.offers.at(-1)).toEqual([]);
  });
  it.each([true, false])('fails closed for a legacy decision %s with a constant diagnostic', async decision => {
    const warning = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const row = await exercise({seed: {}, decision});
    expect(row.counts.scope_safe).toBe(decision ? 1 : 0); expect(row.counts.scope_ceiling).toBe(0);
    expect(row.offers[0]).toEqual(decision ? ['scope_safe'] : []);
    expect(warning).toHaveBeenCalledExactlyOnceWith('Pending approval has no valid stored tool scope; offering approved requests only.');
  });
  it('preserves an empty stored list', async () => {
    const row = await exercise({seed: {scope: []}, decision: true});
    expect(row.counts.scope_safe).toBe(0); expect(row.offers[0]).toEqual([]);
  });
  it('does not interpret a malformed stored wildcard as authority', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const row = await exercise({seed: {scope: ['*']}, decision: false});
    expect(row.counts.scope_ceiling).toBe(0); expect(row.offers[0]).toEqual([]);
  });
  it('freezes actual names when the mode is a wildcard', async () => {
    const row = await exercise({ceiling: ['*'], decision: true, lateAfter: true});
    expect(row.storedScope).toEqual(['scope_safe', 'scope_ceiling']);
    expect(row.offers.at(-1)).not.toContain('scope_late'); expect(row.counts.scope_ceiling).toBe(1);
  });
  it('still intersects the current ceiling in explicit legacy widening mode', async () => {
    const row = await exercise({requested: ['scope_safe'], decision: true, ceilingAfter: [], legacyWidening: true});
    expect(row.counts.scope_safe).toBe(0); expect(row.offers.at(-1)).toEqual([]);
  });
  it('does not reuse a scope after a completed assistant turn', () => {
    expect(approvalToolScope([assistantRow('', [], {}, []), {type: 'user', content: 'new'}])).toBeNull();
  });
  it('keeps a refreshed approval within its own stored list after waiting for the real lock', async () => {
    const row = await exercise({decision: true, refreshBeforeLock: true});
    expect(row.counts.scope_ceiling).toBe(0);
    expect(row.offers.at(-1)).toEqual(['scope_safe']);
  });
});
