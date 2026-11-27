import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { examples, DEFAULT, MAX } from './examples.mjs';

const script = fileURLToPath(new URL('./examples.mjs', import.meta.url));

test('a scheduled run takes the default', () => {
  assert.equal(examples('schedule', ''), DEFAULT);
  assert.equal(examples('schedule', undefined), DEFAULT);
});

test('a dispatched whole number from 1 to MAX is taken', () => {
  for (const [input, want] of [['1', 1], ['5000', 5000], [String(MAX), MAX]]) assert.equal(examples('workflow_dispatch', input), want);
});

test('every invalid dispatched value is refused, never defaulted', () => {
  for (const input of ['', ' ', '0', '00', '-1', '1.5', '1e3', '0x10', 'abc', '5000 ', ' 5000', '+5', String(MAX + 1), '99999999999999999999', undefined]) {
    assert.ok(examples('workflow_dispatch', input) instanceof Error, `refused: ${JSON.stringify(input)}`);
  }
});

test('the script exits 1 on a refused value and prints the count otherwise', () => {
  assert.equal(execFileSync(process.execPath, [script, 'workflow_dispatch', '25']).toString().trim(), '25');
  for (const input of ['', '0', '-3', '2.5', 'many']) {
    assert.throws(() => execFileSync(process.execPath, [script, 'workflow_dispatch', input], { stdio: 'pipe' }), (error) => error.status === 1);
  }
});
