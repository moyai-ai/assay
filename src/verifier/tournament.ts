/*
 * Algorithm adapted from llm_verifier/pivot_tournament.py in
 * llm-as-a-verifier/llm-as-a-verifier, commit
 * 8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770.
 *
 * MIT License
 *
 * Copyright (c) 2026 llm-as-a-verifier
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
import { mapConcurrent } from '../concurrency.js';
export type Pair = readonly [number, number];
export type Rewards = readonly [number, number];
export type PairScorer = (a: number, b: number, signal: AbortSignal) => Promise<Rewards>;

export interface TournamentOptions {
  pivots?: number;
  seed?: number;
  concurrency?: number;
  signal?: AbortSignal;
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
}

function integer(value: number, name: string, minimum: number): void {
  if (!Number.isSafeInteger(value) || value < minimum) throw new Error(`${name} must be an integer >= ${minimum}`);
}

/** Seeded Fisher–Yates with Mulberry32, not Python random.shuffle. */
export function ringCycle(n: number, seed = 0): Pair[] {
  integer(n, 'n', 2);
  if (!Number.isSafeInteger(seed)) throw new Error('seed must be an integer');
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
  integer(n, 'n', 2);
  const requestedPivots = options.pivots ?? 2;
  integer(requestedPivots, 'pivots', 1);
  const concurrency = options.concurrency ?? 4;
  integer(concurrency, 'concurrency', 1);
  options.signal?.throwIfAborted();
  const k = Math.min(requestedPivots, n);
  const ring = ringCycle(n, options.seed ?? 0);
  const wins = Array<number>(n).fill(0);
  const counts = Array<number>(n).fill(0);
  const comparisons: TournamentResult['comparisons'] = [];
  // Each phase has unique directed pairs; only completed pairs repeat across phases.
  // Cache is run-local because evidence and protocol are fixed for one selection.
  const cache = new Map<string, Rewards>();
  const means = () => wins.map((w, i) => counts[i] ? w / counts[i]! : 0);
  const rank = () => {
    const values = means();
    return Array.from({ length: n }, (_, i) => i).sort((a, b) => values[b]! - values[a]! || a - b);
  };
  const accumulate = async (pairs: readonly Pair[], phase: 'ring' | 'pivot') => {
    const scored = await mapConcurrent(pairs, concurrency, async ([a, b], _index, signal) => {
      const key = `${a},${b}`;
      const cached = cache.get(key);
      if (cached) return cached;
      const rewards = await score(a, b, signal);
      bradleyTerry(...rewards); // Reject invalid rewards before caching or aggregation.
      const snapshot: Rewards = [rewards[0], rewards[1]];
      cache.set(key, snapshot);
      return snapshot;
    }, options.signal);
    // Aggregate in scheduled pair order, never network completion order.
    for (let i = 0; i < pairs.length; i++) {
      const [a, b] = pairs[i]!;
      const rewards = scored[i]!;
      const preference = bradleyTerry(...rewards);
      wins[a] = wins[a]! + preference; counts[a] = counts[a]! + 1;
      wins[b] = wins[b]! + 1 - preference; counts[b] = counts[b]! + 1;
      comparisons.push({ phase, pair: [a, b], rewards, preference });
    }
  };
  await accumulate(ring, 'ring');
  const pivots = rank().slice(0, k);
  await accumulate(pivotRoundPairs(n, pivots), 'pivot');
  const ranking = rank();
  return {
    winner: ranking[0]!, ranking, pivots, meanPreferences: means(), counts, comparisons,
    comparisonCount: comparisons.length, uniquePairCount: cache.size,
  };
}
