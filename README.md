# Assay

A foundation for **best-of-N coding → verification → one winning pull request**.

For each issue or instruction, generate N independent implementations of the **whole task**, collect code and execution evidence, and select one with an LLM-as-a-Verifier probabilistic pivot tournament. This is different from splitting one implementation into subtasks.

**Status: runnable verifier core and Agents SDK adapter, not a deployed GitHub service.** No GitHub connection, worktree provisioning, sandbox execution adapter, webhook server, or automatic PR publication is implemented yet. The proposed deployment is in `ARCHITECTURE.md`.

## Included

- OpenAI-compatible custom endpoint client; generator and verifier can use different models/providers.
- OpenAI Agents SDK coding agent with injected sandbox-bound tools, turn limits, cancellation, and tracing disabled by default.
- Direct Chat Completions verifier with raw token logprobs, A–T score extraction, probability-mass diagnostics, configurable criteria, and repeated A/B slot swapping.
- Reference-style ring → pivots → probabilistic aggregation tournament, deterministic seeds, explicit rings, tie-breaking, bounded concurrency, and run-local directed-pair caching.
- Candidate evidence rendering, immutable-base validation, conservative context checks, and explicit opt-in truncation.
- Offline demo, mocked integration tests, and a Python-upstream tournament parity fixture.

There are **no accuracy or cost-uplift results for Assay yet**. The supplied chart motivates the experiment; it is not a measured result for this harness, prompt, or model configuration.

## Run locally

Node.js 22+ is required. Dependencies are locked in `package-lock.json`.

```sh
npm ci
npm run check
npm test
npm run build
npm run demo
```

The demo uses synthetic rewards and makes **no model calls**. Tests use synthetic provider-shaped JSON and an intercepted SDK transport, not real model endpoints.

## Verify existing implementations

`examples/verify.ts` makes real, potentially paid API requests. It does not generate code or publish a PR.

1. Copy `.env.example` to `.env` and configure a model endpoint that supports Chat Completions **and score-token top logprobs**. Ordinary chat compatibility alone is insufficient.
   For a self-hosted LAN/VPC endpoint without TLS, `allowInsecureHttp: true` (or `VERIFIER_ALLOW_INSECURE_HTTP=true` in the example) is an explicit development-only opt-in: API credentials and source code travel in cleartext. HTTPS is the default; URLs with embedded credentials are always rejected.
2. Create an input file:

```json
{
  "task": "Fix the empty-input crash without changing the nonempty behavior.",
  "candidates": [
    {
      "id": "candidate-0",
      "baseSha": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "diff": "actual complete patch collected by the runner",
      "testOutput": "actual independent test output, including exit status",
      "trajectory": "optional agent/tool transcript"
    },
    {
      "id": "candidate-1",
      "baseSha": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "diff": "alternative complete patch",
      "testOutput": "independently collected test output"
    }
  ]
}
```

Use the actual common commit SHA and real evidence, not these placeholders. Only already-eligible candidates belong in this input. The core does not validate whether their patches apply or tests pass.

3. Run explicitly:

```sh
npx tsx --env-file=.env examples/verify.ts candidates.json
```

The example records raw responses (including malformed ones and resamples) and the selection result under `artifacts/<run-id>/`. These artifacts can contain private source code and model output; protect and expire them. It uses bounded in-memory concurrency, has no durable resume, and retains comparisons in memory. It is not the production scheduler.

`await verifier.probe()` is an explicit paid one-pair/one-criterion capability check (plus configured retries). In the future generation workflow, run it before spending on N coding runs. A successful smoke probe is not validation of scoring quality or all provider tokenization cases.

## Scoring contract

The verifier analyzes both candidates for one criterion and emits:

```xml
<score_A> A </score_A>
<score_B> T </score_B>
```

At each letter's generated-token position, Assay extracts returned A–T alternatives, sums probability mass for equivalent spellings, and computes:

```text
value(A) = 20, …, value(T) = 1
q(letter) = returned_probability(letter) / captured_score_probability_mass
reward = (sum(q(letter) * value(letter)) - 1) / 19
pair rewards = mean over criteria and repeated evaluations
preference(a beats b) = sigmoid(reward_a - reward_b)
```

A–T alternatives in `top_logprobs` are usually a **truncated** distribution, not all model logits. Results expose captured mass, log mass, distinct-letter coverage, and conditional distributions. `minCapturedMass` can reject insufficient coverage by mass; there is no universally validated threshold. Scores are **not calibrated probabilities of a correct PR**. `topLogprobs` is configurable from 1 to 20; set it to the endpoint's supported limit and inspect `coverage` and `capturedMass` on the first live run rather than assuming a top-20 request guarantees A–T coverage.

The parser fails closed on absent logprobs, ambiguous verdicts, unsupported token boundaries, non-finite distributions, and truncated completions. It never silently substitutes a sampled-letter score. One bounded fresh-sample retry on extraction failure is enabled by default (`parseRetries: 1`); every attempt is separately charged and available to the response archiver. Transport retries are handled separately by the OpenAI client. Provider-specific tokenization still needs live validation: reasoning-prefixed token alignment and fused `>A` handling are **unvalidated against any real provider**; current provider-shaped fixtures are synthetic.

Defaults: three coding criteria, two repeated evaluations (A/B then B/A), top 20 logprobs, two pivots, four concurrent pairs, and at most four simultaneous HTTP requests per verifier instance. Share a `RequestLimiter` across instances targeting the same endpoint budget. Network completion order does not affect aggregation order, and failures abort/settle in-flight work before returning. Odd repeat counts are supported but not fully slot-balanced. `extraBody` passes native provider options at the request body's top level; critical scoring fields cannot be overridden. Endpoints that only accept `max_completion_tokens` or nonstandard logprob formats need a separate tested adapter.

## Tournament semantics

For N > 1 and k clamped to N:

```text
logical comparisons = N + k(N-k) + k(k-1)/2
model requests before retries ≤ logical comparisons × criteria × repetitions
```

A repeated **directed** pair across ring and pivot phases reuses its score but contributes to both phases. Reverse pairs are distinct. `uniquePairCount` counts distinct attempted pairs; parse/transport retries and recomputation of uncached failures can add requests. For N=1, no verification occurs: returning the sole candidate is not a correctness endorsement, and `verificationComplete` is false.

Default `onError: 'raise'` aborts selection after retries are exhausted. Optional `onError: 'tie'` is an explicitly **degraded experimental mode**: failed comparisons use 0.5/0.5, are never cached, and are recorded in `errors` and `candidateErrorCounts`. `maxErrorFraction` (default 0.1) bounds failed logical comparisons divided by all planned comparisons, checked after each phase. Any fallback makes `verificationComplete: false`; a publisher must refuse automatic publication of a degraded result. Ties dilute scores toward 0.5 and are not statistically neutral. No publisher is implemented here.

PPT is O(Nk) in comparison count, not a guarantee of fewer calls at small N. For example N=5, k=2 gives 12 logical comparisons versus 10 unordered round-robin pairs; compare equal orientation/repetition budgets. The ring seed uses Mulberry32, not Python's RNG. Supply `ring` explicitly for cross-language reproducibility.

## Layout

```text
src/agents/coder.ts       Agents SDK adapter; caller supplies isolated tools
src/models.ts            Custom OpenAI-compatible client
src/concurrency.ts       Ordered bounded mapping and shared endpoint limiter
src/candidates.ts        Evidence contracts and rendering
src/select.ts            Candidate-to-winner integration
src/verifier/scoring.ts   Strict logprob expectation
src/verifier/pairwise.ts  Criteria, repetitions, provider requests
src/verifier/tournament.ts Reference-style pivot tournament
examples/                Offline demo and explicit live-verification example
test/                    Synthetic transport, parser, and upstream parity tests
ARCHITECTURE.md           Cloudflare/GitHub/execution design and milestones
THIRD_PARTY_NOTICES.md    Upstream attribution and MIT notice
```

Next milestone: a sandbox-backed execution adapter that creates isolated candidates from one base SHA, runs N bounded coding agents, freezes patches, and independently tests them before invoking this verifier. GitHub publication follows that local vertical slice.
