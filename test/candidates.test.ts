import { test } from 'node:test';
import assert from 'node:assert/strict';
import { renderCandidate, byteLength, type Candidate } from '../src/candidates.js';

const candidate: Candidate = { id: 'secret-model-name', baseSha: 'a'.repeat(40), diff: '+fix', testOutput: '1 passed' };

test('renders anonymous evidence with a content digest', async () => {
  const a = await renderCandidate(candidate);
  const b = await renderCandidate({ ...candidate, id: 'other-model' });
  assert.equal(a.sha256, b.sha256);
  assert.equal(a.sha256.length, 64);
  assert.ok(!a.text.includes(candidate.id));
  assert.equal(a.truncated, false);
});

test('defaults to rejection; opt-in middle truncation respects UTF-8 byte bounds', async () => {
  const big = { ...candidate, diff: '🧪éabc'.repeat(2000) };
  await assert.rejects(() => renderCandidate(big, 600), /exceeds evidence budget/);
  const rendered = await renderCandidate(big, 600, true);
  assert.ok(rendered.truncated);
  assert.ok(byteLength(rendered.text) <= 600);
  assert.match(rendered.text, /EVIDENCE MIDDLE OMITTED/);
  assert.ok(!rendered.text.includes('\ufffd'));
  assert.ok(rendered.text.includes(rendered.sha256));
});
