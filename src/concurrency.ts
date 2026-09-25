/** FIFO request limit shared by all comparisons on one verifier instance. */
export class RequestLimiter {
  private active = 0;
  private readonly waiting: Array<() => void> = [];
  constructor(readonly concurrency = 4) {
    if (!Number.isSafeInteger(concurrency) || concurrency < 1) throw new Error('concurrency must be a positive integer');
  }
  async run<T>(fn: () => Promise<T>, signal?: AbortSignal): Promise<T> {
    signal?.throwIfAborted();
    await new Promise<void>(resolve => {
      const enter = () => { this.active++; resolve(); };
      if (this.active < this.concurrency) enter();
      else this.waiting.push(enter);
    });
    try {
      signal?.throwIfAborted();
      return await fn();
    } finally {
      this.active--;
      this.waiting.shift()?.();
    }
  }
}

/** Deterministic result order; fail fast, abort peers, and settle before returning. */
export async function mapConcurrent<T, R>(
  items: readonly T[], concurrency: number,
  fn: (item: T, index: number, signal: AbortSignal) => Promise<R>,
  signal?: AbortSignal,
): Promise<R[]> {
  if (!Number.isSafeInteger(concurrency) || concurrency < 1) throw new Error('concurrency must be a positive integer');
  signal?.throwIfAborted();
  const controller = new AbortController();
  const combined = signal ? AbortSignal.any([signal, controller.signal]) : controller.signal;
  const results = new Array<R>(items.length);
  let next = 0;
  let failed = false;
  let error: unknown;
  const worker = async () => {
    while (!failed && next < items.length) {
      const index = next++;
      try {
        combined.throwIfAborted();
        results[index] = await fn(items[index]!, index, combined);
      } catch (cause) {
        if (!failed) { failed = true; error = cause; controller.abort(cause); }
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, worker));
  if (failed) throw error;
  return results;
}
