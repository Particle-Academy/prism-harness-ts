import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { ref } from './ref.mjs';

const script = fileURLToPath(new URL('./ref.mjs', import.meta.url));

test('a scheduled run takes the default branch', () => {
  assert.equal(ref('schedule', '', 'main'), 'main');
  assert.equal(ref('schedule', undefined, 'main'), 'main');
});

test('a dispatch names the default branch or a release line', () => {
  for (const input of ['main', 'release/0.11.x', 'release/0.27.x', 'release/1.0']) assert.equal(ref('workflow_dispatch', input, 'main'), input);
});

test('every other dispatched ref is refused, never defaulted', () => {
  for (const input of [
    '', ' ', ' main', 'main ', 'Main', 'refs/heads/main', 'origin/main', 'dev', 'feature/x', 'release', 'release/', 'release/a/b',
    'release/..', 'release/0..1', 'release/x.lock', 'release/x.', 'release/-x', 'release/x y', 'release/x\n', 'v0.11.2', 'refs/tags/v0.11.2',
    '12b4cdbcfe2337a5f46fcf38abd7d104b7471e8b', 'refs/pull/1/head', 'pull/1/head', 'someone:release/0.11.x', 'release/' + 'x'.repeat(64), undefined,
  ]) {
    assert.ok(ref('workflow_dispatch', input, 'main') instanceof Error, `refused: ${JSON.stringify(input)}`);
  }
});

test('a default branch that is not a plain name refuses every run', () => {
  for (const defaultBranch of ['', ' main', 'main;rm', undefined]) assert.ok(ref('schedule', '', defaultBranch) instanceof Error, JSON.stringify(defaultBranch));
});

test('the script exits 1 on a refused value and prints the branch otherwise', () => {
  assert.equal(execFileSync(process.execPath, [script, 'workflow_dispatch', 'release/0.11.x', 'main']).toString().trim(), 'release/0.11.x');
  assert.equal(execFileSync(process.execPath, [script, 'schedule', '', 'main']).toString().trim(), 'main');
  for (const input of ['', 'dev', 'v0.11.2', '12b4cdb', 'release/a/b']) {
    assert.throws(() => execFileSync(process.execPath, [script, 'workflow_dispatch', input, 'main'], { stdio: 'pipe' }), (error) => error.status === 1);
  }
});
