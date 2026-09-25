import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { Agent, fetch as undiciFetch } from 'undici';

/** Baseline credentials are independent of the DeepSeek generator/verifier. */
export function baselineConnection(env: NodeJS.ProcessEnv = process.env) {
  return {
    baseURL: env.BASELINE_BASE_URL || 'https://api.openai.com/v1',
    apiKey: env.OPENAI_API_KEY,
  };
}

/** Keep transport header/body deadlines aligned with the SDK request deadline.
 * Node's bundled fetch otherwise has an independent 300-second header timeout.
 * Use fetch and dispatcher from the same pinned Undici package, not a global override.
 */
export function createBenchmarkTransport(timeoutMs: number) {
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs <= 0) throw new Error('Invalid HTTP timeout');
  const dispatcher = new Agent({ headersTimeout: timeoutMs, bodyTimeout: timeoutMs });
  return {
    // The SDK uses URL/string inputs and JSON bodies; Undici's DOM types differ.
    fetch: undiciFetch as unknown as typeof globalThis.fetch,
    fetchOptions: { dispatcher },
    close: () => dispatcher.destroy(),
  };
}

/** Archive bodies before SDK conversion, including unsuccessful/truncated replies.
 * Never persist request headers (Authorization), URLs, or cookies.
 * Deliberately non-streaming: the benchmark uses Chat Completions stream=false.
 */
export function recordingFetch(directory: string, secrets: readonly string[], upstream: typeof fetch = fetch): typeof fetch {
  let sequence = 0;
  const save = async (name: string, value: unknown) => {
    let text = JSON.stringify(value, null, 2);
    for (const secret of secrets) if (secret) text = text.split(secret).join('[REDACTED]');
    await mkdir(directory, { recursive: true, mode: 0o700 });
    await writeFile(join(directory, name), text, { mode: 0o600 });
  };
  return async (input, init) => {
    const id = String(sequence++).padStart(4, '0');
    const body = typeof init?.body === 'string' ? JSON.parse(init.body) : undefined;
    if (!body || body.stream === true) throw new Error('Benchmark recorder requires a non-streaming JSON request');
    await save(`model-${id}-request.json`, body);
    const started = Date.now();
    try {
      const response = await upstream(input, init);
      const text = await response.clone().text();
      let payload: unknown;
      try { payload = JSON.parse(text); } catch { payload = { nonJsonBody: text }; }
      await save(`model-${id}-response.json`, {
        status: response.status, requestId: response.headers.get('x-request-id'),
        elapsedMs: Date.now() - started, body: payload,
      });
      return response;
    } catch (error) {
      const cause = error instanceof Error ? error.cause : undefined;
      const causeCode = cause && typeof cause === 'object' && 'code' in cause ? String(cause.code) : undefined;
      await save(`model-${id}-error.json`, { elapsedMs: Date.now() - started, message: String(error), causeCode });
      throw error;
    }
  };
}
