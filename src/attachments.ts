import { HarnessError } from './errors.js';
import type { JsonObject } from './json.js';
import { isJsonObject } from './json.js';

/**
 * Something a caller can attach to a turn.
 *
 * Either a `prism-ts` media object (anything with `toObject()`), or media
 * already in its serialized form. This package has no dependency on `prism-ts`,
 * so it asks the object the questions it needs rather than checking its class.
 */
export type Attachment = JsonObject | MediaLike;

export interface MediaLike {
  toObject(): JsonObject;
  isUrl?(): boolean;
  isFile?(): boolean;
}

const MEDIA_KINDS = new Set(['image', 'audio', 'video', 'document']);

function isMediaLike(value: unknown): value is MediaLike {
  return typeof value === 'object' && value !== null && typeof (value as { toObject?: unknown }).toObject === 'function';
}

function nonEmptyString(value: unknown): boolean {
  return typeof value === 'string' && value !== '';
}

/**
 * Bytes, decoded the way PHP's base64_decode() and Python's lenient decoder read
 * them. A string that decodes to nothing (whitespace, or no base64 characters at
 * all) carries nothing, however long it is.
 */
function carriesBytes(value: unknown): boolean {
  return typeof value === 'string' && Buffer.from(value, 'base64').length > 0;
}

/**
 * Which media a turn will carry, decided before a run starts.
 *
 * The same rules and the same four codes as the PHP reference and the Python
 * port, pinned across all three by prism-parity's `harness-turn-attachments`
 * corpus. Returns the attachments serialized, in the form `prism-ts`'s
 * `UserMessage.fromObject` rebuilds.
 */
export function admitAttachments(prompt: string, attachments: readonly unknown[]): JsonObject[] {
  if (attachments.length === 0) {
    return [];
  }

  if (prompt === '') {
    throw HarnessError.attachmentWithoutPrompt();
  }

  return attachments.map((attachment, index) => {
    let serialized: JsonObject;

    if (isMediaLike(attachment)) {
      // Asked of the OBJECT first. A prism-ts media built from a local path
      // serializes as bytes with no path, so the serialized form alone could not
      // tell that it came from a file.
      if (attachment.isUrl?.() === true) {
        throw HarnessError.attachmentByReference(index, 'a URL');
      }

      if (attachment.isFile?.() === true) {
        throw HarnessError.attachmentByReference(index, 'a file path');
      }

      serialized = attachment.toObject();
    } else if (isJsonObject(attachment)) {
      serialized = attachment;
    } else {
      throw HarnessError.attachmentNotMedia(index, attachment === null ? 'null' : `a ${typeof attachment}`);
    }

    if (typeof serialized.kind !== 'string' || !MEDIA_KINDS.has(serialized.kind)) {
      throw HarnessError.attachmentNotMedia(index, 'an object with no image, document, audio or video kind');
    }

    // ANY string, including the empty one. The reference's isUrl() asks whether a
    // URL was set at all, and `fromUrl('')` is still media built from a URL.
    if (typeof serialized.url === 'string') {
      throw HarnessError.attachmentByReference(index, 'a URL');
    }

    if (typeof serialized.local_path === 'string' || typeof serialized.storage_path === 'string') {
      throw HarnessError.attachmentByReference(index, 'a file path');
    }

    const chunks = serialized.chunks;
    const carries =
      carriesBytes(serialized.base64) ||
      nonEmptyString(serialized.file_id) ||
      (serialized.kind === 'document' && Array.isArray(chunks) && chunks.length > 0);

    if (!carries) {
      throw HarnessError.attachmentEmpty(index);
    }

    return serialized;
  });
}
