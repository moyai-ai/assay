import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createServer } from 'node:http';
import { createModelClient } from '../src/models.js';
import { baselineConnection, createBenchmarkTransport, recordingFetch } from '../benchmark/transport.js';

test('baseline needs only OPENAI_API_KEY and defaults to OpenAI without sharing generator credentials', () => {
  assert.deepEqual(baselineConnection({ OPENAI_API_KEY: 'fake-openai' }), {
    baseURL: 'https://api.openai.com/v1', apiKey: 'fake-openai',
  });
  assert.deepEqual(baselineConnection({ OPENAI_API_KEY: 'fake-openai', BASELINE_BASE_URL: 'http://localhost:1234/v1' }), {
    baseURL: 'http://localhost:1234/v1', apiKey: 'fake-openai',
  });
  assert.equal(baselineConnection({ GENERATOR_API_KEY: 'fake-generator', VERIFIER_API_KEY: 'fake-verifier',
    BASELINE_API_KEY: 'legacy-not-used' }).apiKey, undefined);
});

test('benchmark transport archives truncated raw replies before SDK conversion without headers or keys', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'assay-transport-'));
  const secret = 'synthetic-secret';
  const payload = { choices: [{ finish_reason: 'length', message: { content: `partial ${secret}` } }],
    usage: { completion_tokens: 8192 } };
  try {
    const fetcher = recordingFetch(directory, [secret], async () => Response.json(payload, {
      headers: { 'x-request-id': 'request-123', 'set-cookie': 'never-persist-this' },
    }));
    const response = await fetcher('https://provider.invalid', {
      headers: { Authorization: `Bearer ${secret}` }, body: JSON.stringify({ stream: false, model: 'mock' }),
    });
    assert.deepEqual(await response.json(), payload, 'SDK must receive the unmodified response');
    const raw = await readFile(join(directory, 'model-0000-response.json'), 'utf8');
    assert.ok(!raw.includes(secret));
    assert.ok(!raw.includes('never-persist-this'));
    const archived = JSON.parse(raw);
    assert.equal(archived.body.choices[0].finish_reason, 'length');
    assert.equal(archived.body.usage.completion_tokens, 8192);
    assert.equal(archived.requestId, 'request-123');
    assert.deepEqual(JSON.parse(await readFile(join(directory, 'model-0000-request.json'), 'utf8')), {
      stream: false, model: 'mock',
    });
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('benchmark transport archives failed request diagnostics with secret redaction', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'assay-transport-'));
  try {
    const fetcher = recordingFetch(directory, ['synthetic-secret'], async () => {
      throw new Error('network failed synthetic-secret');
    });
    await assert.rejects(() => fetcher('https://provider.invalid', { body: JSON.stringify({ stream: false }) }), /network failed/);
    const raw = await readFile(join(directory, 'model-0000-error.json'), 'utf8');
    assert.ok(raw.includes('[REDACTED]'));
    assert.ok(!raw.includes('synthetic-secret'));
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('benchmark SDK uses the configured transport header timeout, not a hidden shorter ceiling', { timeout: 12000 }, async () => {
  const directory = await mkdtemp(join(tmpdir(), 'assay-headers-'));
  const server = createServer((_request, response) => {
    const timer = setTimeout(() => {
      response.setHeader('Content-Type', 'application/json');
      response.end(JSON.stringify({ choices: [{ message: { role: 'assistant', content: 'done' }, finish_reason: 'stop' }] }));
    }, 2000);
    response.on('close', () => clearTimeout(timer));
  });
  const short = createBenchmarkTransport(500);
  const long = createBenchmarkTransport(5000);
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  assert.ok(address && typeof address !== 'string');
  const base = createModelClient({ baseURL: `http://127.0.0.1:${address.port}/v1/`, apiKey: 'mock', timeoutMs: 5000 });
  const body = { model: 'mock', messages: [{ role: 'user' as const, content: 'go' }], stream: false as const };
  try {
    await assert.rejects(base.withOptions({ fetch: short.fetch, fetchOptions: short.fetchOptions }).chat.completions.create(body), /timed out/);
    const client = base.withOptions({
      fetch: recordingFetch(directory, [], long.fetch), fetchOptions: long.fetchOptions,
    });
    const result = await client.chat.completions.create(body);
    assert.equal(result.choices[0]?.message.content, 'done');
    const raw = JSON.parse(await readFile(join(directory, 'model-0000-response.json'), 'utf8'));
    assert.equal(raw.status, 200);
    assert.ok(raw.elapsedMs >= 2000);
  } finally {
    await Promise.all([short.close(), long.close()]);
    server.closeAllConnections();
    await new Promise<void>(resolve => server.close(() => resolve()));
    await rm(directory, { recursive: true, force: true });
  }
});

test('benchmark transport rejects streaming before sending a request', async () => {
  let calls = 0;
  const fetcher = recordingFetch('/not-used', [], async () => { calls++; return new Response(); });
  await assert.rejects(() => fetcher('https://provider.invalid', { body: JSON.stringify({ stream: true }) }), /non-streaming/);
  assert.equal(calls, 0);
});
