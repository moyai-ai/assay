import { candidateSchema, renderCandidate, type Candidate } from './candidates.js';
import { PairwiseVerifier, PROTOCOL_VERSION, type ComparisonResult } from './verifier/pairwise.js';
import { selectBest, type TournamentOptions } from './verifier/tournament.js';

export interface SelectionOptions extends TournamentOptions {
  maxEvidenceBytes?: number;
  allowTruncation?: boolean;
  signal?: AbortSignal;
}

/** Rank already generated, eligible candidates sharing exactly the same immutable base. */
export async function selectCandidate(
  task: string,
  candidates: readonly Candidate[],
  verifier: PairwiseVerifier,
  options: SelectionOptions = {},
) {
  options.signal?.throwIfAborted();
  if (!task.trim()) throw new Error('Task must be nonempty');
  const parsed = candidates.map(candidate => candidateSchema.parse(candidate));
  if (!parsed.length) throw new Error('At least one candidate is required');
  if (new Set(parsed.map(c => c.id)).size !== parsed.length) throw new Error('Candidate ids must be unique');
  if (new Set(parsed.map(c => c.baseSha)).size !== 1) throw new Error('All candidates must share the same base SHA');
  const evidence = await Promise.all(parsed.map(c => renderCandidate(c, options.maxEvidenceBytes, options.allowTruncation)));
  // Validate every possible directed pair before sending the first model request.
  for (let a = 0; a < parsed.length; a++) {
    for (let b = 0; b < parsed.length; b++) {
      if (a !== b) verifier.preflight(task, evidence[a]!.text, evidence[b]!.text);
    }
  }
  const evaluations: Array<{ a: string; b: string; result: ComparisonResult }> = [];
  const tournament = await selectBest(parsed.length, async (a, b, signal) => {
    signal.throwIfAborted();
    const result = await verifier.compare(task, evidence[a]!.text, evidence[b]!.text, signal);
    evaluations.push({ a: parsed[a]!.id, b: parsed[b]!.id, result });
    return result.rewards;
  }, options);
  return {
    protocolVersion: PROTOCOL_VERSION,
    verificationComplete: tournament.verificationComplete,
    winnerId: parsed[tournament.winner]!.id,
    ranking: tournament.ranking.map(i => parsed[i]!.id),
    tournament,
    evidence: evidence.map(({ text: _text, ...metadata }, i) => ({ candidateId: parsed[i]!.id, ...metadata })),
    evaluations,
  };
}
