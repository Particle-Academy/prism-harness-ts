import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { HarnessError, admitAttachments, type JsonObject } from '../src/index.js';

/**
 * The cross-language turn-attachments corpus from `prism-parity`.
 *
 * Which attachments a turn will carry. If this port admitted what the reference
 * refuses, an application moving an agent from PHP to TypeScript would lose the
 * guard with no error anywhere, and a file read from a request-supplied path
 * would simply go to the model.
 */
interface Spec {
  $: string;
  [key: string]: unknown;
}

interface AttachmentCase {
  id: string;
  title: string;
  prompt: string;
  attachments: Spec[];
  verdict: { php: string; ts: string; py: string };
  agrees: boolean;
  notes: string;
}

const corpus = JSON.parse(
  readFileSync(new URL('./fixtures/harness-turn-attachments.json', import.meta.url), 'utf8'),
) as { cases: AttachmentCase[] };

/** The same spec, built the way a TypeScript caller would hold it. Mirrors prism-parity's recorder. */
function attachmentFor(spec: Spec): unknown {
  if (spec.$ === 'Text') return { text: spec.text };
  if (spec.$ === 'String') return spec.value;

  const kind = spec.$ === 'Document' ? 'document' : 'image';
  const base: JsonObject = { kind, url: null, base64: null, mime_type: (spec.mimeType as string | undefined) ?? null, file_id: null, filename: null };
  const titled = (object: JsonObject): JsonObject =>
    kind === 'document' ? { ...object, document_title: (spec.title as string | undefined) ?? null, chunks: object.chunks ?? null } : object;

  switch (spec.from) {
    case 'base64':
      return titled({ ...base, base64: spec.base64 as string });
    case 'url':
      return titled({ ...base, url: spec.url as string });
    case 'urlWithBytes':
      return titled({ ...base, url: spec.url as string, base64: spec.base64 as string });
    case 'localPath':
      return {
        toObject: () => titled({ ...base, base64: Buffer.from(spec.bytes as string).toString('base64') }),
        isUrl: () => false,
        isFile: () => true,
      };
    case 'fileId':
      return titled({ ...base, file_id: spec.fileId as string });
    case 'chunks':
      return titled({ ...base, chunks: spec.chunks as string[] });
    case 'text':
      return titled({ ...base, base64: Buffer.from(spec.text as string).toString('base64'), mime_type: 'text/plain' });
    case 'nothing':
      return titled(base);
    default:
      throw new Error(`Unknown media source ${String(spec.from)}`);
  }
}

function verdictOf(entry: AttachmentCase): string {
  try {
    admitAttachments(entry.prompt, entry.attachments.map(attachmentFor));

    return 'admitted';
  } catch (error) {
    if (error instanceof HarnessError) return error.code;
    throw error;
  }
}

describe('the cross-language turn-attachments corpus', () => {
  it('is the whole suite, not a subset someone trimmed to green', () => {
    expect(corpus.cases).toHaveLength(18);
  });

  it.each(corpus.cases)('$id gets the reference verdict ($title)', (entry) => {
    expect(verdictOf(entry)).toBe(entry.verdict.php);
  });

  it('agrees with the reference on EVERY row', () => {
    for (const entry of corpus.cases) {
      expect([entry.verdict.ts, entry.verdict.py], entry.id).toEqual([entry.verdict.php, entry.verdict.php]);
      expect(entry.agrees, entry.id).toBe(true);
    }
  });
});
