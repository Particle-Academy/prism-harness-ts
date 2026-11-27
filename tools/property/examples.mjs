// The examples count for the property-test nightly (G-392 QC). Fails, with
// exit code 1, unless the value is a whole number from 1 to MAX: no silent
// pass, and no fallback to a default when a value was given but is invalid.
// A scheduled run has no input and takes the default.
//
//   node tools/property/examples.mjs <event_name> <input>
// prints the count on success.

import { pathToFileURL } from 'node:url';

export const DEFAULT = 5000;
export const MAX = 1_000_000;

/** The count for this run, or an Error saying why the value is refused. */
export function examples(event, input) {
  if (event === 'schedule') return DEFAULT;
  if (typeof input !== 'string' || !/^[1-9][0-9]{0,6}$/.test(input)) {
    return new Error(`examples must be a whole number from 1 to ${MAX}, got ${JSON.stringify(input)}`);
  }
  const value = Number(input);
  return value <= MAX ? value : new Error(`examples must be a whole number from 1 to ${MAX}, got ${JSON.stringify(input)}`);
}

if (process.argv[1] !== undefined && pathToFileURL(process.argv[1]).href === import.meta.url) {
  const result = examples(process.argv[2], process.argv[3]);
  if (result instanceof Error) {
    console.error(`::error::${result.message}`);
    process.exit(1);
  }
  console.log(String(result));
}
