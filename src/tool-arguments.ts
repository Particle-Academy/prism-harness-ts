import type { JsonObject } from './json.js';
import { canonicalJson } from './tool-policy.js';

/**
 * The deepest nesting accepted. A provider wraps the arguments in its own
 * request envelope, and the PHP reference encodes that request with a depth
 * limit of 512, so the limit leaves room for the deepest envelope.
 */
const MAX_ARGUMENT_DEPTH = 500;
const LONE_SURROGATE = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/;
const CONTROL_ESCAPES: Readonly<Record<string, string>> = {
  '\b': '\\b',
  '\t': '\\t',
  '\n': '\\n',
  '\f': '\\f',
  '\r': '\\r',
};

/**
 * A tool call's arguments as a map of named values, or null when they are
 * unusable. This package has no dependency on prism core, so this is core's
 * classification, spelled the same way in each language.
 *
 * An empty string, the text `0`, `null`, an empty list and an empty object all
 * mean no arguments, and read as `{}`. Anything else that is not a map of named
 * values, or that a follow-up request could not carry, is refused.
 */
export function usableArguments(value: unknown): JsonObject | null {
  if (value === undefined || value === null || value === '' || value === '0') return {};

  let decoded: unknown = value;

  if (typeof value === 'string') {
    try {
      decoded = decode(value);
    } catch {
      return null;
    }
  }

  if (decoded === null || (Array.isArray(decoded) && decoded.length === 0)) return {};
  if (typeof decoded !== 'object' || Array.isArray(decoded)) return null;

  const map = decoded as JsonObject;

  return Object.keys(map).some(isIntegerKey) || !canBeSentBack(map) || !digestible(map) ? null : map;
}

/**
 * Every decision about a call carries the digest of its arguments, so
 * arguments the digest cannot be computed for are unusable: otherwise the
 * call's first decision event would throw out of the run instead.
 */
function digestible(map: JsonObject): boolean {
  try {
    canonicalJson(map);
    return true;
  } catch {
    return false;
  }
}

function decode(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return JSON.parse(escapeControlCharactersInStrings(text));
  }
}

/** Walked a level at a time, so deep input cannot overflow the call stack, holding only nested values. */
function canBeSentBack(root: JsonObject): boolean {
  let level: object[] = [root];

  for (let depth = 1; level.length > 0; depth++) {
    const next: object[] = [];

    for (const value of level) {
      if (Array.isArray(value)) {
        for (const item of value) {
          if (!admitsValue(item, depth, next, true)) return false;
        }

        continue;
      }

      const map = value as Record<string, unknown>;

      for (const key of Object.keys(map)) {
        if (key.startsWith('\u0000') || LONE_SURROGATE.test(key)) return false;
        if (!admitsValue(map[key], depth, next, false)) return false;
      }
    }

    level = next;
  }

  try {
    return JSON.stringify(root) !== undefined;
  } catch {
    return false;
  }
}

/** Checks a scalar in place and holds only a nested value for the next level. */
function admitsValue(item: unknown, depth: number, next: object[], inList: boolean): boolean {
  if (typeof item === 'number') return Number.isFinite(item);
  if (typeof item === 'string') return !LONE_SURROGATE.test(item);
  if (item === null || typeof item === 'boolean') return true;
  // An undefined property is an absent one, in JSON and in the digest alike. In
  // a list it would be sent back as null and has no digest. A function, symbol
  // or bigint has no JSON form at all.
  if (item === undefined) return !inList;
  if (typeof item !== 'object') return false;
  // Only a list or a plain object: anything else (a Date, a Map, a class
  // instance) is sent back as one value and digested as another.
  if (!Array.isArray(item)) {
    const prototype: unknown = Object.getPrototypeOf(item);
    if (prototype !== Object.prototype && prototype !== null) return false;
  }
  if (depth >= MAX_ARGUMENT_DEPTH) return false;

  next.push(item);

  return true;
}

/** A key PHP would store as an integer: canonical decimal, within 64 bits. */
function isIntegerKey(key: string): boolean {
  // Twenty characters is the longest 64-bit integer, with its sign.
  if (key.length > 20 || !/^(0|-?[1-9][0-9]*)$/.test(key)) return false;

  const value = BigInt(key);

  return value >= -9223372036854775808n && value <= 9223372036854775807n;
}

/**
 * Escape raw control characters inside JSON string literals. Anything else is
 * left as it is, so that the decode refuses it: a control character outside a
 * string, or straight after a backslash, has no value it can be repaired to.
 */
function escapeControlCharactersInStrings(json: string): string {
  let result = '';
  let inString = false;
  let escaped = false;

  for (const char of json) {
    const code = char.codePointAt(0) ?? 0;

    if (code <= 0x1f) {
      result += inString && !escaped
        ? CONTROL_ESCAPES[char] ?? `\\u${code.toString(16).padStart(4, '0')}`
        : char;
      escaped = false;
      continue;
    }

    result += char;

    if (escaped) {
      escaped = false;
    } else if (inString && char === '\\') {
      escaped = true;
    } else if (char === '"') {
      inString = !inString;
    }
  }

  return result;
}
