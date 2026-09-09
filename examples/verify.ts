import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { createModelClient, PairwiseVerifier, selectCandidate } from '../src/index.js';

// Explicit invocation only. This example makes paid requests and persists raw responses.
// npx tsx --env-file=.env examples/verify.ts candidates.json
const path = process.argv[2];
if (!path) throw new Error('Usage: npx tsx --env-file=.env examples/verify.ts candidates.json');
const required = (key: string) => {
  const value = process.env[key];
  if (!value) throw new Error(`${key} is required`);
  return value;
};
const input = JSON.parse(await readFile(path, 'utf8'));
const client = createModelClient({
  baseURL: required('VERIFIER_BASE_URL'), apiKey: required('VERIFIER_API_KEY'),
  allowInsecureHttp: process.env.VERIFIER_ALLOW_INSECURE_HTTP === 'true',
});
const directory = `artifacts/${crypto.randomUUID()}`;
await mkdir(directory, { recursive: true });
let responseIndex = 0;
const verifier = new PairwiseVerifier(client, {
  model: required('VERIFIER_MODEL'),
  contextWindowTokens: Number(required('VERIFIER_CONTEXT_TOKENS')),
  extraBody: JSON.parse(process.env.VERIFIER_EXTRA_BODY ?? '{}'),
}, async event => {
  await writeFile(`${directory}/response-${responseIndex++}.json`, JSON.stringify(event, null, 2));
});
const result = await selectCandidate(input.task, input.candidates, verifier, { pivots: 2, seed: 42 });
await writeFile(`${directory}/result.json`, JSON.stringify(result, null, 2));
console.log(JSON.stringify({ winnerId: result.winnerId, ranking: result.ranking, directory }, null, 2));
