import { z } from 'zod';

export const SCORE_LETTERS = 'ABCDEFGHIJKLMNOPQRST';
const logprob = z.number().finite().max(0);
const alternativeSchema = z.object({
  token: z.string(), logprob,
  bytes: z.array(z.number().int().min(0).max(255)).nullable().optional(),
});
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
  minCapturedMass = 0,
): Record<'score_A' | 'score_B', ScoreDistribution> {
  if (!Number.isFinite(minCapturedMass) || minCapturedMass < 0 || minCapturedMass > 1) {
    throw new Error('minCapturedMass must be in [0, 1]');
  }
  const choice = responseSchema.parse(response).choices[0]!;
  const content = choice.message.content;
  const positions = choice.logprobs.content;
  // A UTF-8 code point can straddle tokens. Individually decoded token strings
  // then contain replacement characters; concatenating those strings is lossy.
  // Preserve byte offsets all the way to the selected score-token position.
  const tokenBytes = (p: z.infer<typeof alternativeSchema>) => p.bytes == null
    ? Buffer.from(p.token, 'utf8') : Buffer.from(p.bytes);
  const buffers = positions.map(tokenBytes);
  const generated = Buffer.concat(buffers);
  // Reject corrupt byte streams, not just missing verdicts. Reasoning prefixes
  // remain supported, but the complete visible text must align exactly.
  new TextDecoder('utf-8', { fatal: true }).decode(generated);
  const visibleOffset = generated.lastIndexOf(Buffer.from(content, 'utf8'));
  if (visibleOffset < 0) throw new Error('Visible content does not align with logprob tokens');
  const results = {} as Record<'score_A' | 'score_B', ScoreDistribution>;
  for (const tag of ['score_A', 'score_B'] as const) {
    const matches = [...content.matchAll(new RegExp(`<${tag}>\\s*([A-T])\\s*</${tag}>`, 'g'))];
    // Strict protocol: duplicate verdicts, even identical ones, are ambiguous.
    if (matches.length !== 1 || content.split(`<${tag}>`).length !== 2) {
      throw new Error(`Expected exactly one <${tag}>A–T</${tag}> verdict`);
    }
    const match = matches[0]!;
    const selected = match[1]!;
    const insideOffset = match[0].indexOf('>') + 1;
    const letterOffset = match[0].slice(insideOffset).search(/[A-T]/) + insideOffset;
    const target = visibleOffset + Buffer.byteLength(content.slice(0, match.index! + letterOffset), 'utf8');
    let offset = 0;
    const position = positions.find((_p, index) => {
      const length = buffers[index]!.length;
      const contains = offset <= target && target < offset + length;
      offset += length;
      return contains;
    });
    if (!position || letterOf(tokenBytes(position).toString('utf8')) !== selected) {
      throw new Error(`Unsupported score token boundary for ${tag}`);
    }
    const byLetter = new Map<string, number[]>();
    const seen = new Set<string>();
    for (const alt of position.top_logprobs) {
      if (seen.has(alt.token)) throw new Error('Duplicate top-logprob token');
      seen.add(alt.token);
      const letter = letterOf(tokenBytes(alt).toString('utf8'));
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
