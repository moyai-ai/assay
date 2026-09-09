import { z } from 'zod';

export const SCORE_LETTERS = 'ABCDEFGHIJKLMNOPQRST';
const logprob = z.number().finite().max(0);
const alternativeSchema = z.object({ token: z.string(), logprob });
const positionSchema = alternativeSchema.extend({
  top_logprobs: z.array(alternativeSchema),
});
const responseSchema = z.object({
  choices: z.array(z.object({
    finish_reason: z.literal('stop'),
    message: z.object({ content: z.string().min(1) }),
    logprobs: z.object({ content: z.array(positionSchema).min(1) }),
  })).length(1),
});

export interface ScoreDistribution {
  selectedToken: string;
  normalizedScore: number;
  averageScore: number;
  /** Conditional on returned A–T alternatives, NOT a calibrated success probability. */
  distribution: Record<string, number>;
  capturedMass: number;
  logCapturedMass: number;
  coverage: number;
}

export interface ScoreExtractor {
  (response: unknown, tags: readonly string[]): Record<string, ScoreDistribution>;
}

function letterOf(token: string): string | undefined {
  const letter = token.trim().replace(/^>\s*/, '');
  return letter.length === 1 && SCORE_LETTERS.includes(letter) ? letter : undefined;
}

function logSumExp(values: number[]): number {
  const max = Math.max(...values);
  return max + Math.log(values.reduce((sum, x) => sum + Math.exp(x - max), 0));
}

/** Harbor-compatible A=20 … T=1 expectation; never silently uses the sampled letter. */
export function extractScores(
  response: unknown,
  tags: readonly string[] = ['score_A', 'score_B'],
  minCapturedMass = 0,
): Record<string, ScoreDistribution> {
  if (!Number.isFinite(minCapturedMass) || minCapturedMass < 0 || minCapturedMass > 1) {
    throw new Error('minCapturedMass must be in [0, 1]');
  }
  if (!tags.length || new Set(tags).size !== tags.length) throw new Error('Tags must be nonempty and unique');
  const choice = responseSchema.parse(response).choices[0]!;
  const content = choice.message.content;
  const positions = choice.logprobs.content;
  const generated = positions.map(p => p.token).join('');
  // Some providers prepend reasoning tokens to the visible-content logprob stream.
  const visibleOffset = generated.lastIndexOf(content);
  if (visibleOffset < 0) throw new Error('Visible content does not align with logprob tokens');
  const results: Record<string, ScoreDistribution> = Object.create(null);
  for (const tag of tags) {
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(tag)) throw new Error(`Invalid score tag: ${tag}`);
    const matches = [...content.matchAll(new RegExp(`<${tag}>\\s*([A-T])\\s*</${tag}>`, 'g'))];
    // Strict protocol: duplicate verdicts, even identical ones, are ambiguous.
    if (matches.length !== 1 || content.split(`<${tag}>`).length !== 2) {
      throw new Error(`Expected exactly one <${tag}>A–T</${tag}> verdict`);
    }
    const match = matches[0]!;
    const selected = match[1]!;
    const insideOffset = match[0].indexOf('>') + 1;
    const letterOffset = match[0].slice(insideOffset).search(/[A-T]/) + insideOffset;
    const target = visibleOffset + match.index! + letterOffset;
    let offset = 0;
    const position = positions.find(p => {
      const contains = offset <= target && target < offset + p.token.length;
      offset += p.token.length;
      return contains;
    });
    if (!position || letterOf(position.token) !== selected) {
      throw new Error(`Unsupported score token boundary for ${tag}`);
    }
    const byLetter = new Map<string, number[]>();
    const seen = new Set<string>();
    for (const alt of position.top_logprobs) {
      if (seen.has(alt.token)) throw new Error('Duplicate top-logprob token');
      seen.add(alt.token);
      const letter = letterOf(alt.token);
      if (letter) byLetter.set(letter, [...(byLetter.get(letter) ?? []), alt.logprob]);
    }
    if (!byLetter.size) throw new Error(`No A–T alternatives for ${tag}`);
    // Merge equivalent spellings ("A", " A", ">A") by probability mass.
    const merged = [...byLetter].map(([letter, values]) => [letter, logSumExp(values)] as const);
    const logCapturedMass = logSumExp(merged.map(([, value]) => value));
    if (logCapturedMass > Math.log(1 + 1e-6)) throw new Error('Score probability mass exceeds one');
    if (minCapturedMass > 0 && logCapturedMass < Math.log(minCapturedMass)) {
      throw new Error(`Insufficient score probability mass for ${tag}`);
    }
    const distribution = Object.fromEntries(merged.map(([letter, value]) => [letter, Math.exp(value - logCapturedMass)]));
    const averageScore = Object.entries(distribution).reduce(
      (sum, [letter, probability]) => sum + (20 - SCORE_LETTERS.indexOf(letter)) * probability, 0,
    );
    results[tag] = {
      selectedToken: selected,
      normalizedScore: Math.max(0, Math.min(1, (averageScore - 1) / 19)),
      averageScore,
      distribution,
      capturedMass: Math.min(1, Math.exp(logCapturedMass)),
      logCapturedMass,
      coverage: byLetter.size,
    };
  }
  return results;
}
