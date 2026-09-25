import { test } from 'node:test';
import assert from 'node:assert/strict';
import OpenAI from 'openai';
import { tool } from '@openai/agents';
import { z } from 'zod';
import * as assay from '../src/index.js';
import { completion } from './fixtures.js';

const { PairwiseVerifier, selectCandidate, createModelClient, runCodingAgent } = assay;

test('public entrypoint exposes only the four supported operations', () => {
  assert.deepEqual(Object.keys(assay).sort(), ['PairwiseVerifier', 'createModelClient', 'runCodingAgent', 'selectCandidate']);
});

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

test('parallel pairs saturate 16/32 HTTP slots without multiplying the shared limit', async () => {
  for (const limit of [16, 32]) {
    let active = 0, peak = 0, calls = 0;
    const client = new OpenAI({ baseURL: 'https://synthetic.invalid/v1', apiKey: 'fake', maxRetries: 0,
      fetch: async () => {
        active++; calls++; peak = Math.max(peak, active);
        try {
          await new Promise(resolve => setTimeout(resolve, 20));
          return new Response(JSON.stringify(completion()), { headers: { 'content-type': 'application/json' } });
        } finally { active--; }
      },
    });
    const verifier = new PairwiseVerifier(client, { model: 'synthetic', contextWindowTokens: 131072,
      repetitions: 4, concurrency: limit });
    const candidates = Array.from({ length: 4 }, (_, i) => ({ id: `c${i}`, baseSha: 'a'.repeat(64), diff: `code ${i}`, testOutput: '' }));
    const result = await selectCandidate('Synthetic concurrency test', candidates, verifier, { pivots: 2, concurrency: 8 });
    assert.equal(peak, limit);
    assert.equal(active, 0);
    assert.equal(calls, result.tournament.uniquePairCount * 12);
  }
});

test('input validation and context limits fail before any model call', async () => {
  const { client, requests } = mockClient();
  const verifier = new PairwiseVerifier(client, config);
  const c = { id: 'one', baseSha: 'a'.repeat(40), diff: 'x', testOutput: '' };
  await assert.rejects(() => selectCandidate('task', [], verifier), /At least two/);
  await assert.rejects(() => selectCandidate('task', [c], verifier), /At least two/);
  await assert.rejects(() => selectCandidate('task', [c, c], verifier), /unique/);
  await assert.rejects(() => selectCandidate('task', [c, { ...c, id: 'two', baseSha: 'b'.repeat(40) }], verifier), /same base/);
  await assert.rejects(() => verifier.compare('task', 'a'.repeat(150000), 'b'), /context budget/);
  assert.throws(() => new PairwiseVerifier(client, { ...config, extraBody: { logprobs: false } }), /cannot override/);
  assert.throws(() => new PairwiseVerifier(client, { ...config, repetitions: 0 }));
  assert.throws(() => new PairwiseVerifier(client, { ...config, parseRetries: 1 } as any), /Unrecognized key/);
  const controller = new AbortController(); controller.abort();
  await assert.rejects(() => selectCandidate('task', [c], verifier, { signal: controller.signal }));
  assert.equal(requests.length, 0);
});

test('archives malformed raw responses then fails, without a neutral fallback', async () => {
  const payload = completion();
  const malformed = { ...payload, choices: [{ ...payload.choices[0], logprobs: null }] };
  const { client, requests } = mockClient(() => malformed);
  let archived = 0;
  const verifier = new PairwiseVerifier(client, { ...config, concurrency: 1 }, () => { archived++; });
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
  assert.equal(createModelClient({ baseURL: 'https://example.com/v1', apiKey: 'key' }).maxRetries, 0);
  assert.throws(() => createModelClient({ baseURL: 'http://10.0.0.2:8000/v1', apiKey: 'local' }), /HTTPS/);
  assert.throws(() => createModelClient({ baseURL: 'ftp://example.com', apiKey: 'local' }), /HTTPS/);
  assert.throws(() => createModelClient({ baseURL: 'http://user:pass@localhost', apiKey: 'local' }), /credentials/);
});

test('client helper sends one request on a retryable transport failure', async () => {
  let calls = 0;
  const client = createModelClient({ baseURL: 'https://example.invalid/v1', apiKey: 'fake' }).withOptions({
    fetch: async () => { calls++; return new Response('unavailable', { status: 503 }); },
  });
  await assert.rejects(() => new PairwiseVerifier(client, config).probe(), /503/);
  assert.equal(calls, 1);
});

test('coding agent rejects token-truncated text and tool calls before treating them as complete', async () => {
  for (const withTool of [false, true]) {
    let executions = 0;
    const { client, requests } = mockClient(() => ({
      id: 'truncated', object: 'chat.completion', created: 0, model: 'custom-coder',
      usage: { prompt_tokens: 10, completion_tokens: 8192, total_tokens: 8202 },
      choices: [{ index: 0, finish_reason: 'length', message: {
        role: 'assistant', content: 'Let me now write the implementation.',
        ...(withTool ? { tool_calls: [{ id: 'call-cut', type: 'function',
          function: { name: 'read_evidence', arguments: '{}' } }] } : {}),
      } }],
    }));
    const readTool = tool({ name: 'read_evidence', description: 'Synthetic tool.', parameters: z.object({}),
      execute: async () => { executions++; return 'must not execute'; },
    });
    await assert.rejects(() => runCodingAgent(client, { model: 'custom-coder', tools: [readTool] },
      'Implement a task', { maxTurns: 2 }), /output token limit.*generation is incomplete/);
    assert.equal(executions, 0);
    assert.equal(requests.length, 1);
  }
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
