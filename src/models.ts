import OpenAI from 'openai';

export interface ModelEndpoint {
  baseURL: string;
  apiKey: string;
  timeoutMs?: number;
  maxRetries?: number;
  /** Explicit opt-in for trusted private-network development endpoints. Credentials are sent in cleartext. */
  allowInsecureHttp?: boolean;
}

/** Credentials/URLs are trusted server config, never fields accepted from public jobs. */
export function createModelClient(config: ModelEndpoint): OpenAI {
  const url = new URL(config.baseURL);
  const local = ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname);
  if (url.protocol !== 'https:' && !(url.protocol === 'http:' && (local || config.allowInsecureHttp === true))) throw new Error('Model endpoints require HTTPS (except local development or explicit allowInsecureHttp)');
  if (url.username || url.password || url.search || url.hash) throw new Error('Do not put credentials, query strings, or fragments in baseURL');
  if (!config.apiKey.trim()) throw new Error('apiKey is required');
  const timeout = config.timeoutMs ?? 120_000;
  const maxRetries = config.maxRetries ?? 2;
  if (!Number.isSafeInteger(timeout) || timeout <= 0) throw new Error('timeoutMs must be a positive integer');
  if (!Number.isSafeInteger(maxRetries) || maxRetries < 0 || maxRetries > 5) throw new Error('maxRetries must be an integer in [0, 5]');
  return new OpenAI({ baseURL: config.baseURL, apiKey: config.apiKey, timeout, maxRetries });
}
