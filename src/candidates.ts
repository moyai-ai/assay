import { createHash } from 'node:crypto';
import { z } from 'zod';

/** Evidence must be collected by the runner, not trusted from the agent's final answer. */
export const candidateSchema = z.object({
  id: z.string().min(1).max(200),
  baseSha: z.string().regex(/^[a-f0-9]{40}(?:[a-f0-9]{24})?$/),
  diff: z.string(),
  testOutput: z.string(),
  trajectory: z.string().optional(),
});
export type Candidate = z.infer<typeof candidateSchema>;
function takeBytes(text: string, budget: number, tail = false): string {
  const chars = Array.from(text);
  if (tail) chars.reverse();
  const chosen: string[] = [];
  let size = 0;
  for (const char of chars) {
    size += Buffer.byteLength(char);
    if (size > budget) break;
    chosen.push(char);
  }
  return (tail ? chosen.reverse() : chosen).join('');
}

/** Anonymous evidence; no model names, candidate ids, oracle rewards, or secrets. */
export function renderCandidate(candidate: Candidate, maxBytes = 100_000, allowTruncation = false) {
  const parsed = candidateSchema.parse(candidate);
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 512) throw new Error('maxBytes must be an integer >= 512');
  const original = JSON.stringify({ diff: parsed.diff, testOutput: parsed.testOutput, trajectory: parsed.trajectory });
  const sha256 = createHash('sha256').update(original).digest('hex');
  const originalBytes = Buffer.byteLength(original);
  const truncated = originalBytes > maxBytes;
  if (truncated && !allowTruncation) throw new Error(`Candidate ${parsed.id} exceeds evidence budget (${originalBytes} > ${maxBytes}); provide curated evidence or explicitly allow truncation`);
  let text = original;
  if (truncated) {
    const marker = `\n[EVIDENCE MIDDLE OMITTED; originalBytes=${originalBytes}; sha256=${sha256}]\n`;
    const available = maxBytes - Buffer.byteLength(marker);
    const half = Math.floor(available / 2);
    text = takeBytes(original, half) + marker + takeBytes(original, available - half, true);
  }
  return { text, originalBytes, renderedBytes: Buffer.byteLength(text), truncated, sha256 };
}
