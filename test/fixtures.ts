// Synthetic provider-shaped fixtures. These are NOT captured provider responses.
export function completion(a = 'A', b = 'T', options: {
  hidden?: string;
  fused?: boolean;
  alternativesA?: Array<{ token: string; logprob: number }>;
  alternativesB?: Array<{ token: string; logprob: number }>;
} = {}) {
  const fused = options.fused ?? false;
  const prefix = 'Assessment: code evidence considered. 🧪\n<score_A' + (fused ? '' : '>');
  const aToken = fused ? `>${a}` : ` ${a}`;
  const middle = ' </score_A>\n<score_B' + (fused ? '' : '>');
  const bToken = fused ? `>${b}` : ` ${b}`;
  const end = ' </score_B>';
  const ordinary = (token: string) => ({ token, logprob: -0.1, top_logprobs: [{ token, logprob: -0.1 }] });
  const content = prefix + aToken + middle + bToken + end;
  return {
    id: 'synthetic-completion', object: 'chat.completion', created: 0, model: 'test-model',
    usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 },
    choices: [{
      index: 0, finish_reason: 'stop',
      message: { role: 'assistant', content },
      logprobs: { content: [
        ...(options.hidden ? [ordinary(options.hidden)] : []),
        ordinary(prefix),
        { token: aToken, logprob: -0.1, top_logprobs: options.alternativesA ?? [{ token: aToken, logprob: 0 }] },
        ordinary(middle),
        { token: bToken, logprob: -0.1, top_logprobs: options.alternativesB ?? [{ token: bToken, logprob: 0 }] },
        ordinary(end),
      ] },
    }],
  };
}
