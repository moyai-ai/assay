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
  assert.throws(() => extractScores(completion('A', 'T', { alternativesA: [{ token: 'A', logprob: -1000 }] }), ['score_A'], 0.01), /Insufficient/);
});

test('validates all score tags including ordering and whitespace', () => {
  const payload = completion();
  assert.deepEqual(Object.keys(extractScores(payload, ['score_B', 'score_A'])), ['score_B', 'score_A']);
  assert.throws(() => extractScores(payload, ['score_A', 'score_A']), /unique/);
  assert.throws(() => extractScores(payload, ['score_A.*']), /Invalid/);
  assert.throws(() => extractScores(payload, []), /nonempty/);
  assert.throws(() => extractScores(payload, ['score_C']), /exactly one/);
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
