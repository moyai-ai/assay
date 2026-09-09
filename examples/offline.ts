import { selectBest } from '../src/verifier/tournament.js';

// Deterministic synthetic rewards demonstrate mechanics, NOT benchmark accuracy.
const qualities = [0.2, 0.9, 0.4, 0.6, 0.3];
const result = await selectBest(qualities.length, async (a, b) => [qualities[a]!, qualities[b]!], {
  pivots: 2,
  seed: 42,
});
console.log(JSON.stringify({
  note: 'Offline synthetic demo; no model calls, git operations, or benchmark claims.',
  ...result,
}, null, 2));
