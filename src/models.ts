import OpenAI from 'openai';

export interface ModelEndpoint {
  baseURL: string;
  apiKey: string;
  timeoutMs?: number;
}

/** Credentials/URLs are trusted server config, never fields accepted from public jobs. */
export function createModelClient(config: ModelEndpoint): OpenAI {
  const url = new URL(config.baseURL);
  const local = ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname);
  if (url.protocol !== 'https:' && !(url.protocol === 'http:' && local)) throw new Error('Model endpoints require HTTPS (except localhost)');
  if (url.username || url.password || url.search || url.hash) throw new Error('Do not put credentials, query strings, or fragments in baseURL');
  if (!config.apiKey.trim()) throw new Error('apiKey is required');
  const timeout = config.timeoutMs ?? 120_000;
  if (!Number.isSafeInteger(timeout) || timeout <= 0) throw new Error('timeoutMs must be a positive integer');
  return new OpenAI({ baseURL: config.baseURL, apiKey: config.apiKey, timeout, maxRetries: 0 });
}
