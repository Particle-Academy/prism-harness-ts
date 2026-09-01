export type JsonValue = string | number | boolean | null | JsonValue[] | JsonObject;

export interface JsonObject {
  [key: string]: JsonValue | undefined;
}

export function isJsonObject(value: unknown): value is JsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * A deep copy, by round trip.
 *
 * Stores hand payloads back to callers, and a caller that mutates what it was
 * given would otherwise reach into the store's own state — which the in-memory
 * driver makes trivially possible and every other driver makes impossible. A
 * bug that only appears on one driver is the worst kind here, because the
 * in-memory one is what tests run against.
 */
export function clone<T extends JsonValue | JsonObject>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}
