// The branch the property-test nightly checks out (G-387 round 16, item 11).
// Fails, with exit code 1, unless the value is the repository's default
// branch or a release line (`release/<line>`): no tags, no SHAs, no refs from
// forks, no other branches, and no fallback to the default branch when a
// value was given but is refused. A scheduled run has no input and takes the
// default branch.
//
//   node tools/property/ref.mjs <event_name> <input> <default_branch>
// prints the branch name on success (the workflow checks out refs/heads/<it>).

import { pathToFileURL } from 'node:url';

/** A release line's branch: `release/` and one path segment of plain characters (release/0.11.x). */
const RELEASE = /^release\/[0-9A-Za-z][0-9A-Za-z._-]{0,62}$/;

/** The branch for this run, or an Error saying why the value is refused. */
export function ref(event, input, defaultBranch) {
  if (typeof defaultBranch !== 'string' || !/^[0-9A-Za-z][0-9A-Za-z._/-]{0,99}$/.test(defaultBranch)) {
    return new Error(`the default branch is not a plain branch name, got ${JSON.stringify(defaultBranch)}`);
  }
  if (event === 'schedule') return defaultBranch;
  if (typeof input !== 'string') return new Error(`ref must be ${defaultBranch} or release/<line>, got ${JSON.stringify(input)}`);
  if (input === defaultBranch) return input;
  // `..` and a trailing `.lock` or `.` are never valid branch names (git check-ref-format).
  if (RELEASE.test(input) && !input.includes('..') && !input.endsWith('.lock') && !input.endsWith('.')) return input;
  return new Error(`ref must be ${defaultBranch} or release/<line>, got ${JSON.stringify(input)}`);
}

if (process.argv[1] !== undefined && pathToFileURL(process.argv[1]).href === import.meta.url) {
  const result = ref(process.argv[2], process.argv[3], process.argv[4]);
  if (result instanceof Error) {
    console.error(`::error::${result.message}`);
    process.exit(1);
  }
  console.log(result);
}
