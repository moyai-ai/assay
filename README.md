# Assay

Assay is a TypeScript library that picks the best of several candidate code changes.
An LLM verifier scores candidates in pairs, and a seeded pivot tournament ranks them.
It does not create sandboxes, run tests, or host a service. We have not measured a
general accuracy or cost advantage.

## Quickstart

Requires Node.js 22+ and an OpenAI-compatible Chat Completions endpoint that returns
the top 20 logprobs. Every comparison is a **paid API call**.

1. Install and build (Assay is not on npm):

   ```sh
   npm ci && npm run build
   ```

2. Copy `.env.example` to `.env` and set `VERIFIER_BASE_URL` (HTTPS unless localhost),
   `VERIFIER_API_KEY`, `VERIFIER_MODEL`, and `VERIFIER_CONTEXT_TOKENS` (the model's
   context window).

3. Create `compare.mjs` in the repository root:

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

4. Run it: `node --env-file=.env compare.mjs`

`candidates.json` holds a `task` and at least two candidates. Each candidate needs a
unique `id`, the same Git `baseSha`, a `diff`, independently collected `testOutput`,
and an optional `trajectory`. Filter out ineligible candidates first, and never include
hidden benchmark rewards. Review and test the winner before you merge it.

## API

- `createModelClient(config)`: OpenAI client with a timeout and no retries. Pass your
  own client for retries or private-network HTTP.
- `runCodingAgent(client, config, task, options?)`: runs a coding agent with your
  sandbox-bound tools. It does not isolate environments.
- `new PairwiseVerifier(client, config, onResponse?)`: `compare(task, a, b)` scores a
  pair; `onResponse` receives raw responses; `probe()` makes one paid format check.
- `selectCandidate(task, candidates, verifier, options?)`: returns the winner, ranking,
  comparisons, evidence metadata, and evaluations.

Defaults: 3 criteria, 2 evaluations per criterion (A/B swapped), 20 top logprobs,
2 pivots, seed 0.

## Scoring

The model grades each candidate A (best) to T (worst) as
`<score_A> LETTER </score_A>` and `<score_B> LETTER </score_B>`. Assay turns the letter
logprobs into expected rewards and compares them with `sigmoid(rewardA - rewardB)`.
Check `capturedMass` and `coverage`: these are relative preferences, **not calibrated
probabilities of correctness**.

Errors always throw. Malformed verdicts, missing logprobs, and context overflow fail;
oversized evidence fails unless you set `allowTruncation`.

## Development

```sh
npm ci
npm run check
npm test        # local mocks, no paid calls
npm run build
```

The [benchmark](benchmark/README.md) (not shipped) grades blind selections with Harbor.
`npm run benchmark:test` runs its tests; `npm run benchmark:scale` plans a run, and
`--execute` makes paid calls. Keep credentials on the host, never in containers.

## License

[MIT](LICENSE). Attribution for the adapted tournament algorithm is in
[`src/verifier/tournament.ts`](src/verifier/tournament.ts).
