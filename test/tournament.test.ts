import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { ringCycle, selectBest, bradleyTerry, type Pair } from '../src/verifier/tournament.js';

const ring: Pair[] = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]];

test('seeded ring has one appearance per slot and is reproducible', () => {
  const a = ringCycle(20, 42);
  assert.deepEqual(a, ringCycle(20, 42));
  assert.notDeepEqual(a, ringCycle(20, 43));
  assert.equal(new Set(a.map(p => p[0])).size, 20);
  assert.equal(new Set(a.map(p => p[1])).size, 20);
});

test('matches Python upstream on fixed ring and deterministic scores', async () => {
  const golden = JSON.parse(await readFile(new URL('fixtures/tournament-golden.json', import.meta.url), 'utf8'));
  const result = await selectBest(golden.qualities.length, async (a, b) => [golden.qualities[a], golden.qualities[b]], {
    pivots: golden.k, ring: golden.ring,
  });
  assert.equal(result.winner, golden.winner);
  assert.equal(result.comparisonCount, golden.comparisonCount);
  assert.deepEqual(result.pivots, golden.pivots);
  assert.deepEqual(result.counts, golden.counts);
  result.meanPreferences.forEach((value, i) => assert.ok(Math.abs(value - golden.meanPreferences[i]) < 1e-12));
});

test('counts repeated phase edges twice but calls directed scorer once', async () => {
  const calls: string[] = [];
  const result = await selectBest(5, async (a, b) => { calls.push(`${a},${b}`); return [0.5, 0.5]; }, { pivots: 2, ring });
  assert.equal(result.comparisonCount, 5 + 2 * 3 + 1);
  assert.equal(result.uniquePairCount, new Set(calls).size);
  assert.ok(result.uniquePairCount < result.comparisonCount);
  assert.ok(result.counts.every(c => c >= 2));
  assert.equal(result.counts.reduce((a, b) => a + b, 0), result.comparisonCount * 2);
  assert.deepEqual(result.ranking, [0, 1, 2, 3, 4]);
  assert.deepEqual(result.pivots, [0, 1]);
});

test('all N/k sizes follow reference logical comparison formula', async () => {
  for (let n = 2; n <= 12; n++) for (let k = 1; k <= n + 1; k++) {
    const result = await selectBest(n, async () => [0.5, 0.5], { pivots: k });
    const actualK = Math.min(n, k);
    assert.equal(result.comparisonCount, n + actualK * (n - actualK) + actualK * (actualK - 1) / 2);
  }
});

test('N=1 makes no verifier calls; N=2 keeps both directed ring edges', async () => {
  const one = await selectBest(1, async () => { throw new Error('Must not call'); });
  assert.equal(one.winner, 0);
  assert.equal(one.comparisonCount, 0);
  const two = await selectBest(2, async () => [0.5, 0.5], { pivots: 1 });
  assert.equal(two.comparisonCount, 3);
  assert.equal(two.uniquePairCount, 2);
});

test('rejects invalid configuration, disjoint cycles, scorer errors and invalid rewards', async () => {
  await assert.rejects(() => selectBest(0, async () => [0, 0]));
  await assert.rejects(() => selectBest(2, async () => [0, 0], { pivots: 0 }));
  await assert.rejects(() => selectBest(4, async () => [0, 0], { ring: [[0, 1], [1, 0], [2, 3], [3, 2]] }), /Hamiltonian/);
  await assert.rejects(() => selectBest(2, async () => [NaN, 0]), /Rewards/);
  await assert.rejects(() => selectBest(2, async () => { throw new Error('provider down'); }), /provider down/);
  assert.throws(() => bradleyTerry(2, 0), /Rewards/);
  assert.equal(bradleyTerry(0.5, 0.5), 0.5);
});
