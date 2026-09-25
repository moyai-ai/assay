import assert from 'node:assert/strict';
import test from 'node:test';
import { tool } from '@openai/agents';
import { z } from 'zod';
import { selectionRequestSchema } from '../benchmark/selection.js';
import { runCodingAgent } from '../src/agents/coder.js';
import { createModelClient } from '../src/models.js';

const candidate = { id: 'c001', baseSha: 'a'.repeat(64), diff: 'patch', testOutput: 'public self-test output' };
const input = {
  mode: 'select', task: 'Implement the task', candidates: [candidate, { ...candidate, id: 'c002' }],
  config: { model: 'mock', contextWindowTokens: 131072, maxOutputTokens: 4096, repetitions: 2,
    concurrency: 4, requestTimeoutSec: 900, minCapturedMass: 0, extraBody: {} },
  options: { pivots: 2, seed: 0, maxEvidenceBytes: 48000, allowTruncation: false },
};

test('blind selector accepts only evidence/settings, never added grading fields or paths', () => {
  assert.equal(selectionRequestSchema.parse(input).mode, 'select');
  for (const extra of [{ official_reward: 1 }, { graderPath: '/tests' }, { result: { reward: 0 } }]) {
    assert.throws(() => selectionRequestSchema.parse({ ...input, ...extra }));
    assert.throws(() => selectionRequestSchema.parse({ ...input, candidates: [{ ...candidate, ...extra }, input.candidates[1]] }));
  }
  assert.throws(() => selectionRequestSchema.parse({ ...input, candidates: [candidate] }));
});

test('Responses coding adapter rejects incomplete tool calls before execution', async () => {
  let executed = false;
  const shell = tool({ name: 'shell', description: 'Mock tool', parameters: z.object({ command: z.string() }),
    execute: async () => { executed = true; return 'must not execute'; } });
  const client = createModelClient({ baseURL: 'https://example.invalid/v1', apiKey: 'fake' })
    .withOptions({ fetch: async () => new Response(JSON.stringify({
      id: 'resp_mock', object: 'response', created_at: 0, model: 'mock', status: 'incomplete',
      incomplete_details: { reason: 'max_output_tokens' },
      output: [{ type: 'function_call', id: 'fc_mock', call_id: 'call_mock', name: 'shell', status: 'completed',
        arguments: JSON.stringify({ command: 'partial' }) }],
      usage: { input_tokens: 1, output_tokens: 10, total_tokens: 11 },
    }), { headers: { 'content-type': 'application/json' } }) });
  await assert.rejects(runCodingAgent(client, { model: 'mock', api: 'responses', tools: [shell] }, 'task', { maxTurns: 1 }), /incomplete/i);
  assert.equal(executed, false);
});
