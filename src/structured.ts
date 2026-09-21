import type { JsonObject, JsonValue } from './json.js';

/**
 * Does a document the model returned actually satisfy the schema it was given?
 *
 * A provider that supports strict structured output answers yes by
 * construction. Most do not, and the ones that do can still be pointed at a
 * model or an endpoint where the mode is unavailable — so the question has to
 * be asked here, once, on the way back.
 *
 * WHY THIS IS NOT COERCION. The tempting alternative is to keep the fields that
 * fit and drop the rest, which turns a wrong answer into a plausible one: a
 * planning agent that returns a document missing its `nodes` array becomes a
 * plan with no steps, and an empty plan settles a batch as done.
 *
 * WHAT IT CHECKS is what a JSON schema says without needing a resolver: the
 * declared type, the required keys, the members of an enum, the items of an
 * array, and, where a schema closes itself to additions, keys nobody declared.
 *
 * WHAT IT DOES NOT CHECK: `$ref`, `allOf`, `oneOf`, `not`, and the numeric and
 * string facets (`minimum`, `pattern`, `minLength`). Each would be a partial
 * implementation of a specification this package has no business owning, and a
 * partial validator that reports nothing for a constraint it cannot read is
 * worse than one whose limits are written down. What it cannot read, it passes.
 *
 * The same rules as `Prism\Harness\Structured\SchemaCheck` in the PHP
 * reference, message for message, so a document refused in one language is
 * refused in the others for the same stated reason.
 */
export function schemaProblems(schema: JsonObject, document: JsonValue, name = 'document'): string[] {
  return check(schema, document, name);
}

function check(schema: JsonObject, value: JsonValue, path: string): string[] {
  if (Array.isArray(schema.anyOf)) {
    return checkAnyOf(schema.anyOf, value, path);
  }

  if (Array.isArray(schema.enum)) {
    return schema.enum.some((member) => member === value)
      ? []
      : [`${path} is ${describe(value)}, which is not one of ${members(schema.enum)}.`];
  }

  const types = declaredTypes(schema);

  if (types.length === 0) {
    // Nothing declared to check against. A schema that says nothing about a
    // value cannot be violated by it.
    return [];
  }

  if (!types.some((type) => matches(type, value))) {
    return [`${path} is ${describe(value)}, and the schema asks for ${members(types)}.`];
  }

  if (types.includes('object') && isObject(value)) {
    return checkObject(schema, value, path);
  }

  if (types.includes('array') && Array.isArray(value)) {
    return checkArray(schema, value, path);
  }

  return [];
}

function checkAnyOf(branches: readonly JsonValue[], value: JsonValue, path: string): string[] {
  for (const branch of branches) {
    if (isObject(branch) && check(branch, value, path).length === 0) return [];
  }

  return [`${path} is ${describe(value)}, which satisfies none of the alternatives the schema allows.`];
}

function checkObject(schema: JsonObject, value: JsonObject, path: string): string[] {
  const problems: string[] = [];
  const properties = isObject(schema.properties) ? schema.properties : {};
  const required = Array.isArray(schema.required) ? schema.required : [];

  for (const name of required) {
    if (typeof name === 'string' && !(name in value)) {
      problems.push(`${path}.${name} is required and missing.`);
    }
  }

  for (const [name, property] of Object.entries(properties)) {
    if (!isObject(property) || !(name in value)) continue;

    problems.push(...check(property, value[name] as JsonValue, `${path}.${name}`));
  }

  // Only where the schema closed itself. An open object invites the extra key,
  // so reporting it would be this package's opinion rather than the schema's.
  if (schema.additionalProperties === false) {
    for (const name of Object.keys(value)) {
      if (!(name in properties)) {
        problems.push(`${path}.${name} was returned, and the schema declares no such property.`);
      }
    }
  }

  return problems;
}

function checkArray(schema: JsonObject, value: readonly JsonValue[], path: string): string[] {
  if (!isObject(schema.items)) return [];

  const items = schema.items;

  return value.flatMap((item, index) => check(items, item, `${path}[${index}]`));
}

/** The declared types, as a list — a nullable schema declares two. */
function declaredTypes(schema: JsonObject): string[] {
  if (typeof schema.type === 'string') return [schema.type];
  if (Array.isArray(schema.type)) return schema.type.filter((type): type is string => typeof type === 'string');

  return [];
}

function matches(type: string, value: JsonValue): boolean {
  switch (type) {
    case 'string':
      return typeof value === 'string';
    // An integer satisfies `number`, as JSON Schema says it does. The other
    // direction does not, and JavaScript is where that is easiest to get wrong:
    // 2.0 parses as the same number as 2, so `Number.isInteger` is the test
    // rather than the absence of a fractional part in the source text.
    case 'number':
      return typeof value === 'number' && Number.isFinite(value);
    case 'integer':
      return typeof value === 'number' && Number.isInteger(value);
    case 'boolean':
      return typeof value === 'boolean';
    case 'null':
      return value === null;
    case 'array':
      return Array.isArray(value);
    case 'object':
      return isObject(value);
    default:
      return true;
  }
}

function describe(value: JsonValue): string {
  if (value === null) return 'null';
  if (Array.isArray(value)) return 'an array';
  if (typeof value === 'object') return 'an object';
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'string') {
    return `the string "${value.length > 40 ? `${value.slice(0, 40)}…` : value}"`;
  }

  return `the number ${value}`;
}

function members(values: readonly JsonValue[]): string {
  return values
    .map((value) => (typeof value === 'string' ? `'${value}'` : JSON.stringify(value)))
    .join(', ');
}

function isObject(value: unknown): value is JsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * What to call the document in a problem message.
 *
 * A schema that names itself — Prism's `ObjectSchema` writes `name` — gets its
 * own name in the path, so `plan.steps[1].do` reads the way the schema's author
 * wrote it. Anything else is just "document".
 */
export function schemaName(schema: Readonly<JsonObject>): string {
  return typeof schema.name === 'string' && schema.name !== '' ? schema.name : 'document';
}
