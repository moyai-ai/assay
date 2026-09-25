# Assay

Generate independent coding candidates and select one using LLM pairwise scores
and a seeded pivot tournament. Assay is a TypeScript library, not a sandbox manager
or a deployed service. No general accuracy or cost advantage has been established.

## Development

Requires Node.js 22+.

```sh
npm ci
npm run check
npm test
npm run build
```

Tests use local mocks, not paid model calls. `src/index.ts` defines the public API;
compiled code goes to `dist/`.

## Select a candidate

Configure a trusted Chat Completions endpoint that supports score-token top
logprobs. Copy `.env.example` to `.env` if needed. This example makes **paid API
calls** with real credentials:

```js
import { readFile } from 'node:fs/promises';
import { createModelClient, PairwiseVerifier, selectCandidate } from './dist/index.js';

const { task, candidates } = JSON.parse(await readFile('candidates.json', 'utf8'));
const client = createModelClient({
  baseURL: process.env.VERIFIER_BASE_URL,
  apiKey: process.env.VERIFIER_API_KEY,
});
const verifier = new PairwiseVerifier(client, {
  model: process.env.VERIFIER_MODEL,
  contextWindowTokens: Number(process.env.VERIFIER_CONTEXT_TOKENS),
});
const result = await selectCandidate(task, candidates, verifier);
console.log(result.winnerId);
```

Run as an ES module: `node --env-file=.env compare.mjs`.
Supply **at least two** candidates, each with a unique `id`, the same immutable Git
`baseSha`, a `diff`, independently collected `testOutput`, and optional `trajectory`.
The caller must establish eligibility first. Never include hidden benchmark rewards.

## Public API

- `createModelClient(config)` creates an OpenAI client with a timeout and no retries.
  Endpoints require HTTPS, except localhost. For retries or private-network HTTP,
  supply your own OpenAI client; HTTP sends credentials unencrypted.
- `runCodingAgent(client, config, task, options?)` runs the coding agent with
  caller-supplied, sandbox-bound tools. It does not create or isolate environments.
- `new PairwiseVerifier(client, config, onResponse?)` compares evidence using
  `compare(task, a, b)`. The optional callback records raw responses before parsing;
  `probe()` makes one paid synthetic capability check, not a quality evaluation.
- `selectCandidate(task, candidates, verifier, options?)` returns the winner,
  ranking, tournament comparisons, evidence metadata, and evaluations.

Defaults: three criteria, two evaluations per criterion with swapped A/B positions,
20 top logprobs, two pivots, and seed zero. Each verifier instance shares one request
limit across concurrent comparisons. Tournament concurrency controls parallel pairs.

The verifier requires `<score_A> LETTER </score_A>` and `<score_B> LETTER </score_B>`.
It normalizes the returned A–T probability mass, computes expected scores (A=20,
T=1), maps them to [0,1], then compares using `sigmoid(rewardA - rewardB)`.
Inspect `capturedMass` and `coverage`: truncated top logprobs and relative preferences
are **not calibrated correctness probabilities**.

Errors throw. There is no tie fallback, parser retry, custom parser, or custom ring.
The client helper disables transport retries; a caller-supplied client controls its
own retry policy. Oversized evidence is rejected unless `allowTruncation` is explicit;
context overflow, malformed verdicts, and missing logprobs always fail. A selected
candidate still needs independent tests and human review.

## Benchmark and publication

The optional [local benchmark](benchmark/README.md) generates isolated candidate
pools, locks blind selections, then runs official Harbor grading. It is not shipped
in the package. `npm run benchmark:test` runs its local tests; `npm run benchmark:scale`
creates a plan, and `--execute` enables paid calls. Keep credentials on the host,
never in model-controlled tools or containers.

The package allowlist contains built code, README, package metadata, and `LICENSE`.
Credentials, dependencies, caches, builds, and raw run artifacts are Git-ignored.
`private: true` prevents accidental npm publication, not source-repository publication.

## License

Assay is licensed under the [MIT License](LICENSE). Upstream attribution and license
terms for the adapted tournament algorithm are preserved in
[`src/verifier/tournament.ts`](src/verifier/tournament.ts).
