# AGENTS.md — prism-harness-ts

The TypeScript port of `particle-academy/prism-harness`. Read the shared agent
guide in `prism-parity/docs/AGENTS.md` first: the boundary, the satellite map,
the rules that bind, and the review skills.

## Gates — run them on EXIT CODES

```sh
npm run typecheck   # tsc -p tsconfig.typecheck.json; COVERS src/ AND test/
npm run build
npx vitest run
```

`typecheck` has its own tsconfig because `tsconfig.json` builds, and a build
config cannot include `test/` without emitting it into `dist/`. It used to run
against the build config and silently checked `src/` only — the line above said
otherwise, which is a claim rather than a check. Verify it with
`npx tsc -p tsconfig.typecheck.json --listFiles | grep test/` if you change it.

Never pipe a gate into `head`/`tail`/`grep` and read `$?` — that is the
FILTER's exit code, not the gate's. Redirect to a file, echo `$?`, then look.

## Conformance corpora — vendored, and recorded from HERE

Two cross-language suites from `prism-parity` run in `test/`, each against a
fixture VENDORED into `test/fixtures/`. The fixture is a byte copy of the
suite's `cases.json`; a runner that reached for a sibling checkout would work in
one directory layout and silently no-op in CI.

`agent-task-claim` records this language's half itself, so the recorder and the
assertions cannot drift apart — they are one `run()`:

```sh
PRISM_PARITY_CASES=<abs path>/prism-parity/suites/agent-task-claim/cases.json \
  npx vitest run agent-task-claim      # fills result.ts AND re-vendors the fixture
node <parity>/tools/sync-corpus.mjs    # the loaders ship their own copies
```

Off unless that variable is set. Two rows DIVERGE from the reference and are
recorded as divergences rather than skipped — see the suite manifest's `ts` gap
and G-39. Do not make a row agree by validating in the runner: a runner that
supplies the guard measures the runner.

## What this package holds

Threads, session state, store drivers, and agent task lists. See the port gaps
register in the envelope for what the PHP reference has that this does not.

## Four things that are load-bearing

1. **`Session.key()` must stay byte-identical to PHP's.** sha1 of the
   participant type, truncated to 12. It is what lets all three languages share
   one store. Changing it silently splits a conversation in two.

2. **A volatile store is refused for the durable slot**, at resolve time. That
   check is the reason the package exists. Do not add a "just this once" escape.

3. **Thread positions are assigned inside the lock.** Read-then-write outside
   one loses a message when two turns land together, and nothing reports it.

4. **A task record's `claimed_by` and `claimed_until` are `| null`, never
   optional, and `claimed_until` is an INTEGER.** An optional property holding
   `undefined` is dropped by `JSON.stringify`, and a dropped key is different
   bytes from a key that is present and null. JavaScript also has no int/float
   distinction, so a fractional timestamp passes every equality assertion and
   still writes `1090.75` where the other ports write `1090` — assert
   `Number.isInteger` on the STORED value, not equality. Both traps are ones
   PHP cannot express; this is the port that can.
   `specs/agent-task-lists.md` in `prism-parity` pins the exact string.

## Traps already hit here

- **The in-memory lock claimed its slot AFTER an `await`**, so two callers in
  the same tick both found it empty and both ran. Claim synchronously. Same
  shape as a check-then-create on a file.
- **Windows does not reliably return `EEXIST`** when creating a lockfile that
  another handle is mid-delete on — it returns `EPERM`, `EACCES` or `EBUSY`.
  All four mean "held, retry". That window is the common case under contention,
  not an edge.
