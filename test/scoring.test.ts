import { test } from 'node:test';
import assert from 'node:assert/strict';
import { extractScores } from '../src/verifier/scoring.js';
import { completion } from './fixtures.js';

const near = (a: number, b: number) => assert.ok(Math.abs(a - b) < 1e-10, `${a} != ${b}`);

test('A is best, T worst; sampled token is not the expected reward', () => {
  const result = extractScores(completion('A', 'T', { alternativesA: [
    { token: ' A', logprob: Math.log(0.6) },
    { token: ' T', logprob: Math.log(0.2) },
    { token: 'other', logprob: Math.log(0.2) },
  ] }));
  near(result.score_A!.normalizedScore, 0.75);
  near(result.score_A!.capturedMass, 0.8);
  assert.equal(result.score_A!.coverage, 2);
  assert.equal(result.score_A!.selectedToken, 'A');
  assert.equal(result.score_B!.normalizedScore, 0);
});

test('finds tagged token with reasoning prefix, Unicode and fused >A spellings', () => {
  const result = extractScores(completion('A', 'T', { hidden: 'thinking about <score_A> T </score_A>...', fused: true }));
  assert.equal(result.score_A!.normalizedScore, 1);
  assert.equal(result.score_B!.normalizedScore, 0);
});

test('aligns split UTF-8 bytes including astral symbols before and between scores', () => {
  const visible = '判断🙂 <score_A> A </score_A> → <score_B> T </score_B>';
  const bytes = Buffer.from('hidden reasoning 🧪\n' + visible);
  const positions = [...bytes].map(byte => ({
    token: Buffer.from([byte]).toString('utf8'), bytes: [byte], logprob: 0,
    top_logprobs: [{ token: String.fromCharCode(byte), bytes: [byte], logprob: 0 }],
  }));
  const payload = { choices: [{ finish_reason: 'stop', message: { content: visible }, logprobs: { content: positions } }] };
  assert.ok(!positions.map(p => p.token).join('').includes(visible));
  const scores = extractScores(payload);
  assert.equal(scores.score_A!.normalizedScore, 1);
  assert.equal(scores.score_B!.normalizedScore, 0);
  const corrupt = structuredClone(payload);
  corrupt.choices[0]!.logprobs.content[0]!.bytes = [255];
  assert.throws(() => extractScores(corrupt));
  const mismatch = structuredClone(payload);
  mismatch.choices[0]!.message.content += 'extra';
  assert.throws(() => extractScores(mismatch), /align/);
});

test('missing verdicts remain errors even with valid byte alignment', () => {
  const payload = { choices: [{ finish_reason: 'stop', message: { content: 'No verdict.' },
    logprobs: { content: [{ token: 'No verdict.', bytes: [...Buffer.from('No verdict.')], logprob: 0, top_logprobs: [] }] } }] };
  assert.throws(() => extractScores(payload), /exactly one/);
});

test('merges different spellings by summed probability, not max', () => {
  const result = extractScores(completion('A', 'T', { alternativesA: [
    { token: 'A', logprob: Math.log(0.2) }, { token: ' A', logprob: Math.log(0.2) },
    { token: '>A', logprob: Math.log(0.2) }, { token: 'T', logprob: Math.log(0.4) },
  ] }));
  near(result.score_A!.normalizedScore, 0.6);
  near(result.score_A!.capturedMass, 1);
});

test('stable expectation for underflowed probabilities; exposes log mass', () => {
  const result = extractScores(completion('A', 'T', { alternativesA: [
    { token: 'A', logprob: -1000 }, { token: 'T', logprob: -1000 },
  ] }));
  near(result.score_A!.normalizedScore, 0.5);
  assert.equal(result.score_A!.capturedMass, 0);
  assert.ok(Number.isFinite(result.score_A!.logCapturedMass));
  assert.throws(() => extractScores(completion('A', 'T', { alternativesA: [{ token: 'A', logprob: -1000 }] }), 0.01), /Insufficient/);
});

test('requires both fixed verdict tags and a valid mass threshold', () => {
  const payload = completion();
  assert.deepEqual(Object.keys(extractScores(payload)), ['score_A', 'score_B']);
  for (const threshold of [-1, 2, NaN]) assert.throws(() => extractScores(payload, threshold), /minCapturedMass/);
  payload.choices[0]!.message.content = '<score_A> A </score_A>';
  assert.throws(() => extractScores(payload), /score_B/);
});

test('fails closed on absent logprobs, bad alignment, duplicate tags, and truncated responses', () => {
  assert.throws(() => extractScores({ choices: [{ message: { content: '<score_A>A</score_A>' }, finish_reason: 'stop' }] }));
  const misaligned = completion();
  misaligned.choices[0]!.message.content += 'not in token stream';
  assert.throws(() => extractScores(misaligned), /align/);
  const duplicate = completion();
  const extra = '\n<score_A> A </score_A>';
  duplicate.choices[0]!.message.content += extra;
  duplicate.choices[0]!.logprobs.content.push({ token: extra, logprob: 0, top_logprobs: [] });
  assert.throws(() => extractScores(duplicate), /exactly one/);
  const truncated = completion();
  truncated.choices[0]!.finish_reason = 'length';
  assert.throws(() => extractScores(truncated));
});

test('rejects unusable distributions without discrete or tie fallback', () => {
  for (const alternativesA of [
    [], [{ token: 'other', logprob: 0 }], [{ token: 'A', logprob: NaN }],
    [{ token: 'A', logprob: Infinity }], [{ token: 'A', logprob: 1 }],
    [{ token: 'A', logprob: 0 }, { token: 'T', logprob: 0 }],
    [{ token: 'A', logprob: -1 }, { token: 'A', logprob: -2 }],
  ]) assert.throws(() => extractScores(completion('A', 'T', { alternativesA })));
});
