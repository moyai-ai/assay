import { test } from 'node:test';
import assert from 'node:assert/strict';
import { setTimeout as delay } from 'node:timers/promises';
import OpenAI from 'openai';
import { mapConcurrent, RequestLimiter } from '../src/concurrency.js';
import { PairwiseVerifier } from '../src/verifier/pairwise.js';
import { selectBest } from '../src/verifier/tournament.js';
import { completion } from './fixtures.js';

test('bounded mapping keeps input order despite out-of-order completion', async () => {
  let active = 0; let maximum = 0;
  const values = await mapConcurrent([4, 3, 2, 1], 2, async value => {
    active++; maximum = Math.max(maximum, active);
    await delay(value); active--;
    return value * 2;
  });
  assert.deepEqual(values, [8, 6, 4, 2]);
  assert.equal(maximum, 2);
});

test('raise cancels peers and waits for their cleanup before rejection', async () => {
  let active = 0; let started = 0;
  await assert.rejects(() => mapConcurrent([0, 1, 2, 3], 2, async (value, _index, signal) => {
    started++; active++;
    try {
      if (value === 0) { await delay(2); throw new Error('first failure'); }
      await delay(10000, undefined, { signal });
      return value;
    } finally { active--; }
  }), /first failure/);
  assert.equal(active, 0);
  assert.equal(started, 2);
});

test('shared endpoint limiter bounds calls across concurrent comparisons', async () => {
  let active = 0; let maximum = 0;
  const client = new OpenAI({ apiKey: 'synthetic-key', baseURL: 'https://synthetic.example/v1', maxRetries: 0,
    fetch: async () => {
      active++; maximum = Math.max(maximum, active);
      await delay(3); active--;
      return new Response(JSON.stringify(completion()), { headers: { 'content-type': 'application/json' } });
    },
  });
  const limiter = new RequestLimiter(2);
  const config = { model: 'test', contextWindowTokens: 131072, concurrency: 4 };
  const a = new PairwiseVerifier(client, config, undefined, undefined, limiter);
  const b = new PairwiseVerifier(client, config, undefined, undefined, limiter);
  await Promise.all([a.compare('task', 'a', 'b'), b.compare('task', 'c', 'd')]);
  assert.equal(maximum, 2);
});

test('parse failure gets bounded resampling, archives attempts; probe is explicit', async () => {
  let calls = 0;
  const attempts: number[] = [];
  const client = new OpenAI({ apiKey: 'synthetic-key', baseURL: 'https://synthetic.example/v1', maxRetries: 0,
    fetch: async () => {
      calls++;
      return new Response(JSON.stringify(calls === 1 ? { choices: [] } : completion()), { headers: { 'content-type': 'application/json' } });
    },
  });
  const verifier = new PairwiseVerifier(client, { model: 'test', contextWindowTokens: 131072 }, event => { attempts.push(event.attempt); });
  const result = await verifier.probe();
  assert.equal(result.a.normalizedScore, 1);
  assert.equal(calls, 2);
  assert.deepEqual(attempts, [0, 1]);
});

test('explicit degraded mode records ties per candidate and never caches failures', async () => {
  let attempts = 0;
  const result = await selectBest(3, async (a, b) => {
    if (a === 2 && b === 0 && attempts++ === 0) throw new Error('transient failure');
    return [0.5, 0.5];
  }, { ring: [[0, 1], [1, 2], [2, 0]], pivots: 1, onError: 'tie', maxErrorFraction: 0.2 });
  assert.equal(attempts, 2); // Same directed pair succeeds in pivot phase, not cached as a tie.
  assert.equal(result.errors.length, 1);
  assert.equal(result.verificationComplete, false);
  assert.deepEqual(result.candidateErrorCounts, [1, 0, 1]);
  await assert.rejects(() => selectBest(3, async () => { throw new Error('unsupported endpoint'); }, {
    onError: 'tie', maxErrorFraction: 0.1,
  }), /error fraction/);
});
