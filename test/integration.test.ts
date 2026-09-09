import { test } from 'node:test';
import assert from 'node:assert/strict';
import OpenAI from 'openai';
import { tool } from '@openai/agents';
import { z } from 'zod';
import { PairwiseVerifier } from '../src/verifier/pairwise.js';
import { selectCandidate } from '../src/select.js';
import { createModelClient } from '../src/models.js';
import { runCodingAgent } from '../src/agents/coder.js';
import { completion } from './fixtures.js';

function mockClient(responder: (body: any, call: number) => unknown = () => completion()) {
  const requests: Array<{ url: string; body: any }> = [];
  const client = new OpenAI({
    baseURL: 'https://custom.example/v1', apiKey: 'fake-test-key', maxRetries: 0,
    fetch: async (url, init) => {
      const body = JSON.parse(String(init?.body));
      requests.push({ url: String(url), body });
      return new Response(JSON.stringify(responder(body, requests.length)), { status: 200, headers: { 'content-type': 'application/json' } });
    },
  });
  return { client, requests };
}
const config = {
  model: 'custom-verifier', contextWindowTokens: 131072,
  criteria: [{ id: 'correctness', description: 'Assess correctness.' }],
};

test('uses configured endpoint, preserves logprobs, swaps slots, remaps and averages', async () => {
  const { client, requests } = mockClient();
  const events: unknown[] = [];
  const verifier = new PairwiseVerifier(client, { ...config, extraBody: { reasoning: { enabled: true } } }, event => { events.push(event); });
  const result = await verifier.compare('Fix issue', 'implementation one', 'implementation two');
  assert.deepEqual(result.rewards, [0.5, 0.5]); // Fixture always favors prompt slot A.
  assert.equal(requests.length, 2);
  assert.equal(events.length, 2);
  assert.equal(requests[0]!.url, 'https://custom.example/v1/chat/completions');
  assert.equal(requests[0]!.body.model, 'custom-verifier');
  assert.equal(requests[0]!.body.logprobs, true);
  assert.equal(requests[0]!.body.top_logprobs, 20);
  assert.deepEqual(requests[0]!.body.reasoning, { enabled: true });
  assert.equal(JSON.parse(requests[0]!.body.messages[1].content).candidate_A, 'implementation one');
  assert.equal(JSON.parse(requests[1]!.body.messages[1].content).candidate_A, 'implementation two');
  assert.equal(result.evaluations[1]!.a.normalizedScore, 0);
  assert.equal(result.evaluations[1]!.b.normalizedScore, 1);
});

test('averages criteria and repetitions equally', async () => {
  const { client } = mockClient((_body, call) => call <= 2 ? completion('A', 'A') : completion('T', 'T'));
  const verifier = new PairwiseVerifier(client, { ...config, criteria: [
    { id: 'one', description: 'first' }, { id: 'two', description: 'second' },
  ] });
  const result = await verifier.compare('Task', 'a', 'b');
  assert.deepEqual(result.rewards, [0.5, 0.5]);
  assert.equal(result.evaluations.length, 4);
});

test('end-to-end candidate selection with synthetic custom-endpoint responses', async () => {
  const { client, requests } = mockClient(body => {
    const prompt = JSON.parse(body.messages[1].content);
    const quality = (text: string) => text.includes('correct-code') ? 'A' : 'T';
    return completion(quality(prompt.candidate_A), quality(prompt.candidate_B));
  });
  const verifier = new PairwiseVerifier(client, config);
  const candidates = ['broken-code', 'correct-code', 'incomplete-code'].map((diff, i) => ({
    id: `candidate-${i}`, baseSha: 'a'.repeat(40), diff, testOutput: 'synthetic evidence',
  }));
  const result = await selectCandidate('Fix issue', candidates, verifier, { pivots: 1, seed: 9 });
  assert.equal(result.winnerId, 'candidate-1');
  assert.equal(requests.length, result.tournament.uniquePairCount * 2);
  assert.equal(result.evidence.length, 3);
  assert.ok(!requests.some(r => r.body.messages[1].content.includes('candidate-1')));
});

test('input validation and context limits fail before any model call', async () => {
  const { client, requests } = mockClient();
  const verifier = new PairwiseVerifier(client, config);
  const c = { id: 'one', baseSha: 'a'.repeat(40), diff: 'x', testOutput: '' };
  await assert.rejects(() => selectCandidate('task', [], verifier), /At least one/);
  await assert.rejects(() => selectCandidate('task', [c, c], verifier), /unique/);
  await assert.rejects(() => selectCandidate('task', [c, { ...c, id: 'two', baseSha: 'b'.repeat(40) }], verifier), /same base/);
  await assert.rejects(() => verifier.compare('task', 'a'.repeat(150000), 'b'), /context budget/);
  assert.throws(() => new PairwiseVerifier(client, { ...config, extraBody: { logprobs: false } }), /cannot override/);
  assert.throws(() => new PairwiseVerifier(client, { ...config, repetitions: 0 }));
  const controller = new AbortController(); controller.abort();
  await assert.rejects(() => selectCandidate('task', [c], verifier, { signal: controller.signal }));
  assert.equal(requests.length, 0);
});

test('archives malformed raw responses then fails, without a neutral fallback', async () => {
  const payload = completion();
  const malformed = { ...payload, choices: [{ ...payload.choices[0], logprobs: null }] };
  const { client, requests } = mockClient(() => malformed);
  let archived = 0;
  const verifier = new PairwiseVerifier(client, { ...config, concurrency: 1, parseRetries: 0 }, () => { archived++; });
  await assert.rejects(() => verifier.compare('task', 'a', 'b'));
  assert.equal(archived, 1);
  assert.equal(requests.length, 1);
});

test('model client rejects insecure or credential-bearing endpoint configuration', () => {
  assert.throws(() => createModelClient({ baseURL: 'http://remote.example/v1', apiKey: 'key' }), /HTTPS/);
  assert.throws(() => createModelClient({ baseURL: 'https://user:pass@example.com/v1', apiKey: 'key' }), /credentials/);
  assert.throws(() => createModelClient({ baseURL: 'https://example.com/v1', apiKey: '' }), /apiKey/);
  assert.throws(() => createModelClient({ baseURL: 'https://example.com/v1', apiKey: 'key', timeoutMs: 0 }), /timeoutMs/);
  assert.equal(createModelClient({ baseURL: 'http://localhost:8000/v1', apiKey: 'local' }).baseURL, 'http://localhost:8000/v1');
  assert.equal(createModelClient({ baseURL: 'http://10.0.0.2:8000/v1', apiKey: 'local', allowInsecureHttp: true }).baseURL, 'http://10.0.0.2:8000/v1');
  assert.throws(() => createModelClient({ baseURL: 'ftp://example.com', apiKey: 'local', allowInsecureHttp: true }), /HTTPS/);
  assert.throws(() => createModelClient({ baseURL: 'http://user:pass@example.com', apiKey: 'local', allowInsecureHttp: true }), /credentials/);
});

test('Agents SDK runs injected tool against custom Chat Completions endpoint', async () => {
  let executions = 0;
  const readTool = tool({
    name: 'read_evidence', description: 'Read evidence from assigned synthetic sandbox.',
    parameters: z.object({}), execute: async () => { executions++; return 'file contents'; },
  });
  const { client, requests } = mockClient((_body, call) => ({
    id: `chat-${call}`, object: 'chat.completion', created: 0, model: 'custom-coder',
    usage: { prompt_tokens: 10, completion_tokens: 10, total_tokens: 20 },
    choices: [{ index: 0, finish_reason: call === 1 ? 'tool_calls' : 'stop',
      message: call === 1 ? {
        role: 'assistant', content: null,
        tool_calls: [{ id: 'call-1', type: 'function', function: { name: 'read_evidence', arguments: '{}' } }],
      } : { role: 'assistant', content: 'Done with synthetic task.' },
    }],
  }));
  const result = await runCodingAgent(client, { model: 'custom-coder', tools: [readTool] }, 'Read the evidence', { maxTurns: 3 });
  assert.equal(result.finalOutput, 'Done with synthetic task.');
  assert.equal(executions, 1);
  assert.equal(requests.length, 2);
  assert.ok(requests.every(r => r.body.model === 'custom-coder'));
  assert.ok(requests.every(r => r.url.endsWith('/chat/completions')));
  assert.ok(requests[1]!.body.messages.some((m: any) => m.role === 'tool' && m.content === 'file contents'));
});
