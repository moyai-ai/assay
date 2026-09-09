// Algorithm adapted from llm-as-a-verifier (MIT); see THIRD_PARTY_NOTICES.md.
import { mapConcurrent } from '../concurrency.js';
export type Pair = readonly [number, number];
export type Rewards = readonly [number, number];
export type PairScorer = (a: number, b: number, signal: AbortSignal) => Promise<Rewards>;

export interface TournamentOptions {
  pivots?: number;
  seed?: number;
  /** Explicit ring for reproducibility across languages (PRNGs differ). */
  ring?: readonly Pair[];
  concurrency?: number;
  signal?: AbortSignal;
  /** Explicit degraded benchmark mode. Default is fail-closed. */
  onError?: 'raise' | 'tie';
  /** Maximum failed logical comparisons / total planned comparisons. */
  maxErrorFraction?: number;
}

export interface TournamentResult {
  winner: number;
  ranking: number[];
  pivots: number[];
  meanPreferences: number[];
  counts: number[];
  comparisons: Array<{ phase: 'ring' | 'pivot'; pair: Pair; rewards: Rewards; preference: number }>;
  /** Logical comparison count includes repeated directed pairs between phases. */
  comparisonCount: number;
  uniquePairCount: number;
  errors: Array<{ phase: 'ring' | 'pivot'; pair: Pair; message: string }>;
  candidateErrorCounts: number[];
  verificationComplete: boolean;
}

function integer(value: number, name: string, minimum: number): void {
  if (!Number.isSafeInteger(value) || value < minimum) throw new Error(`${name} must be an integer >= ${minimum}`);
}

/** Seeded Fisher–Yates with Mulberry32, not Python random.shuffle. */
export function ringCycle(n: number, seed = 0): Pair[] {
  integer(n, 'n', 1);
  if (!Number.isSafeInteger(seed)) throw new Error('seed must be an integer');
  if (n === 1) return [];
  let state = seed >>> 0;
  const random = () => {
    state = (state + 0x6d2b79f5) | 0;
    let x = Math.imul(state ^ (state >>> 15), 1 | state);
    x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
  const order = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [order[i], order[j]] = [order[j]!, order[i]!];
  }
  return order.map((a, i) => [a, order[(i + 1) % n]!] as const);
}

function validateRing(n: number, ring: readonly Pair[]): void {
  if (n === 1 && ring.length === 0) return;
  if (ring.length !== n) throw new Error('Ring must contain N edges');
  const next = new Map<number, number>();
  for (const [a, b] of ring) {
    integer(a, 'ring index', 0); integer(b, 'ring index', 0);
    if (a >= n || b >= n || a === b || next.has(a)) throw new Error('Invalid ring edge');
    next.set(a, b);
  }
  const visited = new Set<number>();
  let current = 0;
  for (let i = 0; i < n; i++) {
    if (visited.has(current) || !next.has(current)) throw new Error('Ring must be one Hamiltonian cycle');
    visited.add(current);
    current = next.get(current)!;
  }
  if (current !== 0) throw new Error('Ring must close');
}

export function pivotRoundPairs(n: number, pivots: readonly number[]): Pair[] {
  const selected = new Set(pivots);
  const pairs: Pair[] = [];
  for (let a = 0; a < n; a++) if (!selected.has(a)) for (const b of pivots) pairs.push([a, b]);
  const sorted = [...pivots].sort((a, b) => a - b);
  for (let i = 0; i < sorted.length; i++) {
    for (let j = i + 1; j < sorted.length; j++) pairs.push([sorted[i]!, sorted[j]!]);
  }
  return pairs;
}

export function bradleyTerry(a: number, b: number): number {
  if (![a, b].every(x => Number.isFinite(x) && x >= 0 && x <= 1)) throw new Error('Rewards must be in [0, 1]');
  return 1 / (1 + Math.exp(-(a - b)));
}

export async function selectBest(n: number, score: PairScorer, options: TournamentOptions = {}): Promise<TournamentResult> {
  integer(n, 'n', 1);
  const requestedPivots = options.pivots ?? 2;
  integer(requestedPivots, 'pivots', 1);
  const concurrency = options.concurrency ?? 4;
  integer(concurrency, 'concurrency', 1);
  const onError = options.onError ?? 'raise';
  if (!['raise', 'tie'].includes(onError)) throw new Error('onError must be raise or tie');
  const maxErrorFraction = options.maxErrorFraction ?? 0.1;
  if (!Number.isFinite(maxErrorFraction) || maxErrorFraction < 0 || maxErrorFraction > 1) throw new Error('maxErrorFraction must be in [0, 1]');
  options.signal?.throwIfAborted();
  const k = Math.min(requestedPivots, n);
  const ring = options.ring ?? ringCycle(n, options.seed ?? 0);
  validateRing(n, ring);
  if (n === 1) return {
    winner: 0, ranking: [0], pivots: [], meanPreferences: [1], counts: [0],
    comparisons: [], comparisonCount: 0, uniquePairCount: 0,
    errors: [], candidateErrorCounts: [0], verificationComplete: false,
  };
  const wins = Array<number>(n).fill(0);
  const counts = Array<number>(n).fill(0);
  const comparisons: TournamentResult['comparisons'] = [];
  // Run-local only: safe because candidate inputs and protocol are fixed for one selection.
  const cache = new Map<string, Promise<Rewards>>();
  const attempted = new Set<string>();
  const errors: TournamentResult['errors'] = [];
  const candidateErrorCounts = Array<number>(n).fill(0);
  const plannedComparisons = n + k * (n - k) + k * (k - 1) / 2;
  const means = () => wins.map((w, i) => counts[i] ? w / counts[i]! : 0);
  const rank = () => {
    const values = means();
    return Array.from({ length: n }, (_, i) => i).sort((a, b) => values[b]! - values[a]! || a - b);
  };
  const accumulate = async (pairs: readonly Pair[], phase: 'ring' | 'pivot') => {
    const scored = await mapConcurrent(pairs, concurrency, async ([a, b], _index, signal) => {
      const key = `${a},${b}`;
      let pending = cache.get(key);
      if (!pending) {
        attempted.add(key);
        pending = Promise.resolve().then(async (): Promise<Rewards> => {
          const rewards = await score(a, b, signal);
          bradleyTerry(...rewards);
          return [rewards[0], rewards[1]];
        });
        cache.set(key, pending);
      }
      try { return { rewards: await pending, error: undefined }; }
      catch (error) {
        cache.delete(key); // Never cache errors or their substitute ties.
        if (onError === 'raise' || signal.aborted) throw error;
        return { rewards: [0.5, 0.5] as Rewards, error: String(error) };
      }
    }, options.signal);
    // Aggregate in scheduled pair order, never network completion order.
    for (let i = 0; i < pairs.length; i++) {
      const [a, b] = pairs[i]!;
      const { rewards, error } = scored[i]!;
      if (error !== undefined) {
        errors.push({ phase, pair: [a, b], message: error });
        candidateErrorCounts[a] = candidateErrorCounts[a]! + 1;
        candidateErrorCounts[b] = candidateErrorCounts[b]! + 1;
      }
      const preference = bradleyTerry(...rewards);
      wins[a] = wins[a]! + preference; counts[a] = counts[a]! + 1;
      wins[b] = wins[b]! + 1 - preference; counts[b] = counts[b]! + 1;
      comparisons.push({ phase, pair: [a, b], rewards, preference });
    }
    if (errors.length / plannedComparisons > maxErrorFraction) throw new Error(`Verifier error fraction exceeds ${maxErrorFraction}`);
  };
  await accumulate(ring, 'ring');
  const pivots = rank().slice(0, k);
  await accumulate(pivotRoundPairs(n, pivots), 'pivot');
  const ranking = rank();
  return {
    winner: ranking[0]!, ranking, pivots, meanPreferences: means(), counts, comparisons,
    comparisonCount: comparisons.length, uniquePairCount: attempted.size,
    errors, candidateErrorCounts, verificationComplete: errors.length === 0,
  };
}
