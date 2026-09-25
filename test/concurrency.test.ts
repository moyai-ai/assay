import { test } from 'node:test';
import assert from 'node:assert/strict';
import { setTimeout as delay } from 'node:timers/promises';
import OpenAI from 'openai';
import { mapConcurrent } from '../src/concurrency.js';
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
  const verifier = new PairwiseVerifier(client, { model: 'test', contextWindowTokens: 131072, concurrency: 2 });
  await Promise.all([verifier.compare('task', 'a', 'b'), verifier.compare('task', 'c', 'd')]);
  assert.equal(maximum, 2);
});

test('probe archives malformed responses and fails without resampling', async () => {
  let calls = 0;
  const responses: unknown[] = [];
  const client = new OpenAI({ apiKey: 'synthetic-key', baseURL: 'https://synthetic.example/v1', maxRetries: 0,
    fetch: async () => {
      calls++;
      return new Response(JSON.stringify(calls === 1 ? { choices: [] } : completion()), { headers: { 'content-type': 'application/json' } });
    },
  });
  const verifier = new PairwiseVerifier(client, { model: 'test', contextWindowTokens: 131072 }, event => { responses.push(event.response); });
  await assert.rejects(() => verifier.probe());
  assert.equal(calls, 1);
  assert.deepEqual(responses, [{ choices: [] }]);
  assert.equal((await verifier.probe()).a.normalizedScore, 1);
  assert.equal(calls, 2);
});

test('tournament failure stops scheduling and never substitutes a tie', async () => {
  let calls = 0;
  await assert.rejects(() => selectBest(3, async () => {
    calls++;
    throw new Error('unsupported endpoint');
  }, { concurrency: 1 }), /unsupported endpoint/);
  assert.equal(calls, 1);
});
