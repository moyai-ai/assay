# Assay architecture

Status: proposed service architecture, with the verifier core and injected-tools coding adapter implemented. References inspected on September 9, 2026. Nothing has been deployed or connected to a GitHub account.

## 1. Product contract

Input: an authorized GitHub repository, an immutable base commit, and an issue or free-form task. Output: one selected implementation, its verification evidence, and a reviewable PR—not an automatic merge or a guarantee of correctness.

Each of N coding agents independently implements the **entire same task**. They do not collaborate, inspect siblings, share conversation history, or merge their patches. A shared task specification and test policy are frozen before generation. Optional diverse generation strategies must still solve that identical specification.

Use a deterministic application coordinator for fan-out, budgets, eligibility, comparison scheduling, and publication. A planning agent may propose a task specification, but should not control N, authorize credentials, mutate evaluation criteria after seeing candidates, or choose its own winner.

An existing-PR verification mode is a separate entry point: compare that PR's frozen head to authorized alternatives, not automatically regenerate every PR on every webhook event.

## 2. Cloudflare topology

```text
GitHub App webhook / authenticated instruction API
                       │
                       ▼
        Cloudflare Worker: authenticate, authorize,
        validate, deduplicate, enqueue → return job ID
                       │
                       ▼
        Durable orchestration (proposed: Workflows)
        immutable job spec + bounded fan-out + joins
             │                         │
             ▼                         ▼
   N isolated execution sandboxes    job metadata / leases
   each with own repository state   (proposed: D1)
             │                         │
             └──── frozen artifacts ───┘
                       │
                       ▼
              independent test workers
                       │
                       ▼
          eligible candidates → verifier tournament
          custom Chat Completions + top logprobs
                       │
                       ▼
           trusted publisher → one draft PR
                       │
                       ▼
               ordinary CI + human review

Large patches, transcripts, test logs, raw model responses → R2
```

**Workers are the control plane, not the git execution environment.** Do not depend on local host `git`/subprocess execution inside an ordinary Worker or use one open HTTP request as durable job storage. The Agents SDK documents Workers support with `nodejs_compat` and tracing limitations. Its current hosted clients include `CloudflareSandboxClient`, which talks over HTTP to a deployed **Sandbox bridge Worker**, not an arbitrary Worker URL. This makes an all-Cloudflare deployment plausible, but it still needs a real bridge/container deployment and an execution-adapter spike. No Cloudflare runtime compatibility test has been run for Assay.

Candidate backend to validate first: Cloudflare Sandbox through the bridge, with a toolchain image containing git and the repository's dependencies. Fallback: Docker-backed development runner or an external sandbox provider, keeping the same contracts. Do not promise arbitrary benchmark compatibility: tasks may require privileged Docker, unusual CPU/memory, long-lived services, or network behavior unavailable on a chosen sandbox backend.

Workflow steps should schedule bounded batches and persist artifacts, rather than retain the entire tournament and SDK run in a Worker heap. Confirm actual platform limits and request-duration behavior before choosing where the coding loop and long verifier calls run. Bounded in-memory library calls in this repo are a correctness baseline, not the durable production scheduler.

## 3. GitHub authorization and PR lifecycle

Use a **GitHub App**, not a user's broad personal access token. Request installation-scoped, repository-limited permissions: metadata read, contents read/write, PR read/write, issues read (write only for progress comments), and checks write if publishing check runs. Keep app private keys and installation tokens outside candidate environments.

Webhook handler requirements:

1. Verify `X-Hub-Signature-256` against the raw body before parsing; enforce payload limits.
2. Deduplicate delivery IDs and semantic requests. Authorize the triggering actor and the installation's access to the specific repo. A valid GitHub signature alone does not authorize a public commenter to spend money or access private code.
3. Only explicit approved triggers create jobs: authorized issue assignment/label, approved slash command, or authenticated API request. Fork events and bot-generated events must not bypass policy or trigger loops.
4. Snapshot the task, base SHA, installation/repo IDs, actor, model configurations, criteria/protocol, toolchain image digest, seed, N, k, repeat count, and budgets. Bind all steps and artifacts to the tenant/job.
5. Respond quickly with a job ID and durable status. Never embed a provider key or arbitrary backend URL in an untrusted job request; select server-side allowlisted configurations to avoid credential exfiltration and SSRF.

### When to open the PR

Prefer a check/status or issue progress comment first, and open the PR **after the winning branch has a real diff**. The request to create a PR before generation should be interpreted as establishing a tracked job, not inventing a meaningful diff on an identical head/base branch. Do not create placeholder code commits merely to manufacture an early PR. An early draft can instead be opened once a genuine implementation exists, then updated with explicit provenance.

The publisher receives a frozen winner patch/digest, never arbitrary agent-chosen branch commands. It uses a separate checkout and narrowly scoped write credentials, verifies the current target state, creates `assay/<job-id>` idempotently, pushes only the winner, and creates/updates a single draft PR. Branch names and git arguments come from validated server data. Never let the coding agent push, create PRs, alter branch protections, or merge.

If the base moved, mark the result stale and revalidate the exact new patch/base combination before publication. A rebase or conflict resolution produces a different candidate; the previous verdict must not be treated as evidence for it.

## 4. Execution isolation and worktrees

**Git worktrees isolate working directories, not security or information access.** Worktrees in one shared clone share Git object storage and often give access to sibling code, refs, processes, credentials, and network resources. A prompt telling an agent to stay in its worktree is not an isolation boundary.

Development-only trusted mode could use N worktrees in one clone, with branch naming and cleanup controlled by the host. Production should give each candidate an independent sandbox and independently writable repository metadata. Options:

- A separate clone/materialized repository per sandbox, with a worktree inside that sandbox if useful.
- A carefully constructed immutable seed snapshot with no writable shared Git directory or sibling-visible mount.

Prefer the first for the initial implementation. Cross-candidate writable caches, dependency directories, home directories, and histories are forbidden. Every candidate starts from the same base SHA and toolchain. Credentials for model inference stay in the trusted agent coordinator; tools proxy operations into the sandbox. Repository materialization should not leave reusable GitHub tokens, credential helpers, app keys, or model keys in the agent environment.

Before adding a backend, define enforceable CPU/memory/disk, wall-clock, command-output, network/egress, concurrent-job, and tool-call limits. Prevent metadata-service access. Use least-privilege artifact access. Do not mount the Docker socket or host home. Host-level allowlists and path normalization are not substitutes for a sandbox. Repository hooks, build scripts, dependencies, and test code are untrusted executable content.

### Candidate lifecycle

```text
pending → provisioning → generating → freezing → testing → eligible
                      ↘ failed / timed_out / cancelled
```

- Pin base commit before any agent runs.
- Bound N and concurrency separately; N=5 must not imply unbounded nested sub-agent creation.
- Record tool calls, command exit codes, and outputs with byte/time limits.
- When generation ends, stop all model-controlled commands and background processes.
- Collect the final artifact from the runner: all tracked, added, deleted, renamed and binary changes; do not rely on `git diff` alone to include untracked files. Record modes and symlinks, reject unsafe paths, enforce patch size, and exclude secrets/artifacts.
- Reconstruct the artifact in a clean test sandbox at the exact base. Verify patch application and digest there. Never run final tests against a still-mutating agent workspace.
- Run immutable, independently supplied smoke/build/test policy, and archive the commands, environment, exit status, and logs. Agent-edited tests may be additional evidence, not authority to override protected checks.
- Freeze evidence and eligibility before ranking. If every candidate fails hard gates, stop with `no_eligible_candidate`; don't choose the least broken and call it verified. Handle pre-existing baseline failures using explicit policy and comparison to baseline, not a blanket “all tests must pass” rule.
- Cleanup on success, failure, cancellation, and lease expiration; retain artifacts according to tenant policy.

Suggested execution adapter contract (not implemented):

```ts
interface CandidateExecutor {
  // server-validated immutable spec; caller receives no host filesystem authority
  generate(spec: CandidateSpec, signal: AbortSignal): Promise<FrozenArtifact>;
  test(artifact: FrozenArtifact, policy: TestPolicy, signal: AbortSignal): Promise<TestEvidence>;
  dispose(candidateId: string): Promise<void>;
}
```

`src/agents/coder.ts` is the model-loop building block. Its injected tools are not implemented sandbox tools. Plain SDK `Agent` plus function tools permits custom Chat Completions models without assuming support for every hosted Responses tool. A backend spike should also test `SandboxAgent`/`CloudflareSandboxClient` with the selected custom model; changing to native sandbox capabilities is an adapter decision, not a reason to change verifier math.

## 5. Verification and selection

The implemented protocol follows the upstream pairwise method's structure, not an exact reproduction of its prompts:

1. Render anonymous final diff, independently collected tests, and optional trajectory evidence.
2. For each criterion and repetition, compare both candidates in one prompt.
3. Extract A–T alternatives at `<score_A>` and `<score_B>` generated-token positions.
4. Normalize over captured score-token mass, compute A=20 … T=1 expectation, map to [0,1].
5. Alternate A/B presentation on odd repetitions and map rewards back to candidate order.
6. Average rewards over criteria/repetitions, then compute `sigmoid(Ra - Rb)` (no added temperature multiplier).
7. Aggregate a seeded Hamiltonian ring, select top-k pivots by mean preference, aggregate nonpivot/pivot and pivot/pivot comparisons, return maximal total mean preference. Break ties by original candidate index.

Every repeated directed pair is cached within a selection and still counted at each logical phase occurrence. Reverse directions are distinct. Persist the actual ring as well as seed; the TypeScript RNG differs from Python. A fixed-ring test fixture checks numerical parity with the pinned upstream Python algorithm.

Differences from upstream are deliberate: fail-closed selection by default after bounded parse resampling, summed rather than max probability for synonymous token spellings (matching local Harbor scoring), different prompts/criteria, stricter token alignment, and explicit evidence/context validation. An opt-in degraded `onError: 'tie'` mode records every substituted comparison, marks the result incomplete, and enforces a maximum failed-comparison fraction; such a result must not be automatically published. Failures never enter the score cache. This is not a reproduction claim for the paper's benchmark numbers.

The local Harbor project also contains a **two-stage assessment → plain A–T token** pointwise verifier. That is a distinct protocol and is not implemented here. It may be worth testing for reasoning models, but must have its own prompt/version, parser, cost accounting, and controlled comparison against pairwise XML. Do not silently swap it in and call it the same experiment.

### Endpoint capability gate

The core exposes an explicit paid `verifier.probe()` for a tiny pairwise scoring request. Wiring it into a mandatory pre-generation gate and validating real provider captures remain next-milestone work. Before generating N expensive candidates, probe the configured model for:

- tool calls for the coding role;
- valid pairwise visible verdicts and aligned generated-token logprobs for the verifier role;
- accepted `top_logprobs`, `max_tokens`/`max_completion_tokens`, reasoning options, finish reasons, context window, and provider response shape;
- score-token coverage/mass across varied examples, not just a single all-A smoke test;
- timeout, cancellation, retries and rate limits.

Top-20 response entries do not guarantee all 20 scoring letters. Partial mass is measured, not misrepresented as full-distribution access. Never derive a fake probability distribution from a text score. Providers without usable logprobs are unsupported for this verifier mode until an explicitly different discrete-judge baseline is selected.

### Evidence and costs

Default evidence overflow rejects the request before inference; opt-in truncation includes a digest and marker, but can hide decisive bugs and must be tracked as an experimental condition. The current byte-based context check is conservative for byte-based tokenizers, not a universal tokenizer guarantee. Production should use tested provider tokenization or a provider count endpoint.

The implementation schedules independent comparisons concurrently within a ring/pivot phase, preserving the phase barrier and deterministic aggregation order. A shared request limiter caps verifier calls; share it across instances using the same endpoint. Failure propagates cancellation and waits for in-flight work to settle. Production adds durable batch scheduling, recovery and distributed endpoint quotas. Snapshot/verify all configuration and artifacts before accessing a persistent cache. Cache keys must include tenant/repo, task/spec digest, base SHA, candidate artifact digests in directed order, criterion/prompt/protocol version, provider/model/options, and repetition. Never cache errors as successful ties. The optional degraded mode is for explicit experiments, not bypassing publication policy.

At N=5, k=1: 9 logical comparisons. With 3 criteria and 2 repetitions, that is at most 54 verifier requests before directed-pair reuse and retries, **plus five coding runs and test execution**. The default one parse resample can double calls that fail extraction, separately from SDK transport retries; include any explicit capability probes too. Count SDK retries, provider reasoning/output tokens, repeated trajectory input, cache hit pricing, sandbox startup/runtime, test compute, and failed jobs. A best-of-N chart does not imply this implementation is cheaper than a stronger single agent.

## 6. Durable state and safety

```text
queued → generating → validating → verifying → selected → publishing → completed
                    ↘ no_eligible_candidate
any active state → failed / cancelled / expired / stale
```

Use idempotent job/step/candidate IDs, leases, fencing tokens and optimistic state transitions. Replayed webhooks or workflow retries must not create extra candidates or multiple PRs. Retrying model inference is not guaranteed to return the same candidate and must count against budget. Treat publication timeout as an ambiguous side effect: reconcile the branch/PR before retrying.

Record artifact digests, frozen request metadata, comparison schedule, all successful/failed raw responses, parsed distributions/mass, token usage, latency, retries, total cost, eligibility decisions, and winner provenance. Raw responses can contain private data. Encrypt/authorize artifact access, redact secrets before inference, disable external traces by default, and require explicit tenant consent for external model processing. Prompt-injection instructions in code, comments, logs, or trajectories remain untrusted; a verifier prompt is defense in depth, not a guarantee.

Do not expose this library directly as an unauthenticated service. It has no tenant auth, durable scheduler, global token/cost cap, or hardened sandbox implementation.

## 7. Evaluation plan before an accuracy claim

Run all selectors on the **same frozen candidate pools** at N=1/3/5, with independently isolated generation:

- first candidate / random candidate;
- deterministic public tests only;
- discrete sampled-letter judge;
- pointwise expected-score selector (separate future implementation);
- pairwise PPT expected-score selector;
- oracle best-of-N, used only as an evaluation ceiling.

Keep hidden test outcomes, oracle rewards, and reference patches inaccessible to generator, verifier, prompt development, and production cache. Evaluate those only after selection. Split tuning and held-out tasks, freeze prompts/models/options, randomize seeds, and report paired task-level uncertainty and repeated-run variability.

Measure selected success rate (not just verifier agreement), oracle headroom captured, harmful selection over the first candidate, provider failure/abstention rate, captured score mass, total cost per task and per solved task, and p50/p95 latency. Include generation/test/verifier costs and failed tasks. Tune N, k, criteria and repetitions against an explicit budget; adding more verification can erase cheap-generator savings.

The service must allow no winner and human review. Even the best candidate can be wrong, and a relative tournament score must never bypass ordinary CI or branch protections.

## 8. Delivery milestones

1. **Done here:** deterministic verifier/scorer core, custom model client, injected-tools Agents SDK adapter, synthetic tests and parity fixture, explicit live-verification example.
2. **Next:** backend capability spike + local sandbox vertical slice: real isolated code generation, frozen patches, independent tests, artifact manifests, cleanup, budgets. No GitHub writes required.
3. **Then:** GitHub App, authorized webhooks, durable state/artifacts, idempotent winner publisher and check-run evidence.
4. **Then:** Cloudflare bridge deployment, workflow batching/resume, provider probes, persistent scoped cache, tenant quotas, tracing/redaction, cancellation and recovery tests.
5. **Before performance claims:** held-out benchmark study and adversarial isolation/prompt-injection tests.

Open choices: Cloudflare Sandbox versus another executor; repository languages/images; default N and total spend ceiling; custom provider/model and usable logprob limits; triggers/approval policy; retention/data-sharing requirements. These remain configuration/product decisions, not hidden assumptions in this prototype.

## Reference snapshots

- User-provided framework: `https://github.com/llm-as-a-verifier/llm-as-a-verifier`, inspected at `8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770`; specifically `llm_verifier/pivot_tournament.py`, `fine_grained_reward.py`, and `__init__.py`.
- Method description: `https://llm-as-a-verifier.com/`. The screenshot's headline numbers are not adopted as Assay results.
- Agents SDK source: `https://github.com/openai/openai-agents-js`, inspected at `064fcb20c40706feb4a4ffec4249490e5bc3e9b3`; specifically `docs/src/content/docs/guides/models.mdx`, `guides/troubleshooting.mdx`, `guides/sandbox-agents/clients.mdx`, and `examples/sandbox/extensions/cloudflare-runner.ts`. Installed packages: `@openai/agents` 0.17.2 and `openai` 7.12.1.
- GitHub PR API: `https://docs.github.com/en/rest/pulls/pulls#create-a-pull-request`.
- Local scoring reference (not a runtime dependency): `/Users/roberthommes/moyai/projects/harbor-benchmarks/verification/src/logprob_scoring.py`, `verification/src/inference.py`, `verification/scripts/pairwise_comparison.py`, and `verification/scripts/runtime/llm_as_a_verifier.py`.
