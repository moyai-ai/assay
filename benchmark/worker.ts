import { createInterface } from 'node:readline';
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { MaxTurnsExceededError, tool } from '@openai/agents';
import { z } from 'zod';
import { runCodingAgent } from '../src/agents/coder.js';
import { createModelClient } from '../src/models.js';
import { baselineConnection, createBenchmarkTransport, recordingFetch } from './transport.js';

// stdout is exclusively JSONL protocol; diagnostics belong on stderr.
const initSchema = z.object({
  type: z.literal('init'), instruction: z.string().min(1), logsDir: z.string().min(1),
  model: z.string().min(1), maxTurns: z.number().int().min(1).max(200),
  maxOutputTokens: z.number().int().min(256).max(32768),
  requestTimeoutSec: z.number().int().min(15).max(900),
  reasoningEffort: z.enum(['provider', 'none', 'low', 'medium', 'high']),
  api: z.enum(['chat-completions', 'responses']).default('chat-completions'),
  credentialProfile: z.enum(['generator', 'baseline']).default('generator'),
});
const commandSchema = z.object({
  // Local validation only. Do not pass this Zod schema to tool(): the SDK
  // requires server strict mode for Zod, which is incompatible with this endpoint.
  command: z.string().min(1).max(32768),
  cwd: z.string().nullable().describe('Absolute container directory, or null for its default. Shell state does not persist.'),
  timeout_sec: z.number().int().min(1).max(120),
});
const send = (message: unknown) => process.stdout.write(`${JSON.stringify(message)}\n`);
const pending = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>();
let nextId = 0;
const abort = new AbortController();
process.once('SIGTERM', () => abort.abort(new Error('Harbor cancelled the trial')));
process.once('SIGINT', () => abort.abort(new Error('Interrupted')));

async function main() {
  const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
  const iterator = lines[Symbol.asyncIterator]();
  const first = await iterator.next();
  if (first.done) throw new Error('Missing bridge initialization');
  const init = initSchema.parse(JSON.parse(first.value));
  const receiver = (async () => {
    try {
      for await (const line of iterator) {
        const reply = JSON.parse(line);
        const request = pending.get(reply.id);
        if (reply.type !== 'result' || !request) throw new Error('Unexpected bridge response');
        pending.delete(reply.id);
        request.resolve(reply.result);
      }
      if (!abort.signal.aborted) abort.abort(new Error('Harbor bridge closed'));
    } catch (error) {
      abort.abort(error);
    } finally {
      for (const request of pending.values()) request.reject(new Error('Harbor bridge closed'));
      pending.clear();
    }
  })();
  const shell = tool({
    name: 'shell',
    description: 'Execute a noninteractive bash command in your isolated task container. Use commands to read/write files and run tests. Each call starts a new shell; background services must redirect output. Output is bounded to the last 16 KiB per stream. Do not seek benchmark tests, rewards, or reference solutions outside the provided task workspace.',
    strict: false,
    parameters: {
      type: 'object' as const,
      properties: {
        command: { type: 'string' },
        cwd: { type: ['string', 'null'], description: 'Absolute container directory, or null for its default. Shell state does not persist.' },
        timeout_sec: { type: 'integer', minimum: 1, maximum: 120 },
      },
      // SDK non-strict schemas require this; commandSchema strips extra fields locally.
      required: ['command', 'cwd', 'timeout_sec'], additionalProperties: true as const,
    },
    execute: async (input) => {
      const args = commandSchema.parse(input);
      abort.signal.throwIfAborted();
      const id = nextId++;
      const result = new Promise<unknown>((resolve, reject) => pending.set(id, { resolve, reject }));
      send({ type: 'exec', id, ...args });
      return JSON.stringify(await result);
    },
  });
  await mkdir(init.logsDir, { recursive: true, mode: 0o700 });
  const transport = createBenchmarkTransport(init.requestTimeoutSec * 1000);
  try {
    const { baseURL, apiKey } = init.credentialProfile === 'baseline' ? baselineConnection() : {
      baseURL: process.env.GENERATOR_BASE_URL || process.env.VERIFIER_BASE_URL,
      apiKey: process.env.GENERATOR_API_KEY || process.env.NEBIUS_API_KEY || process.env.VERIFIER_API_KEY,
    };
    if (!baseURL || !apiKey) throw new Error(init.credentialProfile === 'baseline'
      ? 'Set OPENAI_API_KEY for the baseline' : 'Missing generator endpoint or API key');
    const client = createModelClient({ baseURL, apiKey, timeoutMs: init.requestTimeoutSec * 1000 })
      .withOptions({
        fetch: recordingFetch(join(init.logsDir, 'model'), [apiKey], transport.fetch),
        fetchOptions: transport.fetchOptions,
      });
    let outcome;
    try {
      const result = await runCodingAgent(client, {
        model: init.model, api: init.api, tools: [shell],
        modelSettings: {
          maxTokens: init.maxOutputTokens, parallelToolCalls: false,
          // Stateless Responses turns must carry opaque reasoning context with
          // tool results; otherwise a reasoning-model baseline loses its state.
          ...(init.api === 'responses' ? {
            store: false, providerData: { include: ['reasoning.encrypted_content'] },
          } : {}),
          ...(init.reasoningEffort === 'provider' ? {} : { reasoning: { effort: init.reasoningEffort } }),
        },
      }, init.instruction, { maxTurns: init.maxTurns, signal: abort.signal });
      outcome = { stopReason: 'completed', finalOutput: result.finalOutput, usage: result.state.usage, history: result.history };
    } catch (error) {
      // A declared budget stop still leaves a gradeable implementation. Do not
      // turn provider/transport/parse errors into successful generations.
      if (!(error instanceof MaxTurnsExceededError) || !error.state) throw error;
      outcome = { stopReason: 'turn_limit', finalOutput: null, usage: error.state.usage, history: error.state.history };
    }
    const { history, ...generation } = outcome;
    await writeFile(join(init.logsDir, 'trajectory.json'), JSON.stringify(history, null, 2), { mode: 0o600 });
    await writeFile(join(init.logsDir, 'generation.json'), JSON.stringify(generation, null, 2), { mode: 0o600 });
    send({ type: 'done', usage: generation.usage, stop_reason: generation.stopReason });
  } finally {
    lines.close();
    process.stdin.destroy();
    await receiver;
    await transport.close();
  }
}

main().catch(error => {
  let message = String(error?.message ?? error);
  for (const name of ['GENERATOR_API_KEY', 'NEBIUS_API_KEY', 'VERIFIER_API_KEY', 'BASELINE_API_KEY', 'OPENAI_API_KEY']) {
    const secret = process.env[name];
    if (secret) message = message.split(secret).join('[REDACTED]');
  }
  send({ type: 'error', message });
  process.exitCode = 1;
});
