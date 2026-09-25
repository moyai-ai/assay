// Host-only blind selector. stdin contains task + frozen evidence, never trial
// paths, official tests, rewards, or grader logs. There are no model tools.
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
import { z } from 'zod';
import { candidateSchema } from '../src/candidates.js';
import { createModelClient } from '../src/models.js';
import { selectCandidate } from '../src/select.js';
import { PairwiseVerifier } from '../src/verifier/pairwise.js';
import { createBenchmarkTransport, recordingFetch } from './transport.js';

const configSchema = z.object({
  model: z.string().min(1), contextWindowTokens: z.number().int().positive(),
  maxOutputTokens: z.number().int().positive().max(32768),
  repetitions: z.number().int().min(1).max(32),
  concurrency: z.number().int().min(1).max(32),
  requestTimeoutSec: z.number().int().min(15).max(900),
  minCapturedMass: z.number().min(0).max(1),
  extraBody: z.record(z.string(), z.unknown()),
}).strict();
export const selectionRequestSchema = z.discriminatedUnion('mode', [
  z.object({ mode: z.literal('probe'), config: configSchema }).strict(),
  z.object({
    mode: z.literal('select'), config: configSchema, task: z.string().min(1),
    candidates: z.array(candidateSchema.strict()).min(2).max(16),
    options: z.object({
      pivots: z.number().int().min(1).max(16), seed: z.number().int(),
      maxEvidenceBytes: z.number().int().min(512), allowTruncation: z.boolean(),
      concurrency: z.number().int().min(1).max(32).default(8),
    }).strict(),
  }).strict(),
]);

export async function selectFrozenCandidates(input: unknown, logDir: string) {
  const request = selectionRequestSchema.parse(input);
  const baseURL = process.env.VERIFIER_BASE_URL;
  const apiKey = process.env.VERIFIER_API_KEY;
  if (!baseURL || !apiKey) throw new Error('Explicit VERIFIER_BASE_URL and VERIFIER_API_KEY are required');
  const started = performance.now();
  const { requestTimeoutSec, ...config } = request.config;
  const transport = createBenchmarkTransport(requestTimeoutSec * 1000);
  await mkdir(logDir, { recursive: true, mode: 0o700 });
  let sequence = 0;
  try {
    const client = createModelClient({ baseURL, apiKey, timeoutMs: requestTimeoutSec * 1000 })
      .withOptions({ fetch: recordingFetch(join(logDir, 'model'), [apiKey], transport.fetch), fetchOptions: transport.fetchOptions });
    const verifier = new PairwiseVerifier(client, config, async event => {
      // Raw bodies already live in model/. Keep a separate request-to-criterion ledger.
      await writeFile(join(logDir, `evaluation-${sequence++}.json`), JSON.stringify({
        criterion: event.criterionId, repetition: event.repetition, swapped: event.swapped,
      }), { mode: 0o600 });
    });
    if (request.mode === 'probe') {
      const { response: _response, ...evaluation } = await verifier.probe();
      return { capabilityPassed: true, evaluation, elapsed_seconds: (performance.now() - started) / 1000 };
    }
    const result = await selectCandidate(request.task, request.candidates, verifier, request.options);
    return {
      ...result,
      verificationComplete: true, // Benchmark wire format: selection now always throws on failure.
      evaluations: result.evaluations.map(({ a, b, result: comparison }) => ({
        a, b, rewards: comparison.rewards,
        evaluations: comparison.evaluations.map(({ response: _response, ...evaluation }) => evaluation),
      })),
      elapsed_seconds: (performance.now() - started) / 1000,
      score_interpretation: 'A=20 through T=1; normalized expected score in [0,1]. Tournament preferences are relative, not correctness probabilities.',
    };
  } finally {
    await transport.close();
  }
}

async function main() {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of process.stdin) {
    size += chunk.length;
    if (size > 32 * 1024 * 1024) throw new Error('Selector input exceeds 32 MiB');
    chunks.push(Buffer.from(chunk));
  }
  if (!process.argv[2]) throw new Error('Missing selector log directory');
  const result = await selectFrozenCandidates(JSON.parse(Buffer.concat(chunks).toString('utf8')), process.argv[2]);
  process.stdout.write(JSON.stringify(result));
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(error => {
    let message = String(error?.message ?? error);
    for (const [name, value] of Object.entries(process.env)) {
      if (name.endsWith('API_KEY') && value) message = message.split(value).join('[REDACTED]');
    }
    process.stderr.write(message + '\n');
    process.exitCode = 1;
  });
}
