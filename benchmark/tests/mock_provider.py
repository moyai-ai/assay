"""Synthetic provider fixture. No external requests or real credentials."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class MockProvider:
    def __init__(self, generation_barrier=0):
        self.requests = []
        self.generation_barrier = generation_barrier
        self.generators_arrived = 0
        self.generators_ready = threading.Event()
        self.conversations = 0
        self.lock = threading.Lock()
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if (fixture.generation_barrier and not body.get('logprobs') and not self.path.endswith('/responses')
                        and not any(m.get('role') == 'tool' for m in body.get('messages', []))):
                    with fixture.lock:
                        fixture.generators_arrived += 1
                        if fixture.generators_arrived >= fixture.generation_barrier:
                            fixture.generators_ready.set()
                    if not fixture.generators_ready.wait(120):
                        self.send_error(503, 'Synthetic generation barrier timed out')
                        return
                with fixture.lock:
                    fixture.requests.append({'body': body, 'authorization': self.headers.get('Authorization'), 'path': self.path})
                    is_responses = self.path.endswith('/responses')
                    if body.get('logprobs'):
                        data = json.loads(next(m['content'] for m in body['messages'] if m['role'] == 'user'))
                        good = lambda text: '+print(2)' in text or 'return a + b' in text
                        a = 'A' if good(data['candidate_A']) else 'T'
                        b = 'A' if good(data['candidate_B']) else 'T'
                        parts = ['<score_A>', ' ' + a, ' </score_A>\n<score_B>', ' ' + b, ' </score_B>']
                        token = lambda s: {'token': s, 'logprob': 0, 'top_logprobs': [{'token': s, 'logprob': 0}]}
                        payload = {'id': 'mock-judge', 'object': 'chat.completion', 'created': 0, 'model': body['model'],
                                   'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
                                   'choices': [{'index': 0, 'finish_reason': 'stop',
                                                'message': {'role': 'assistant', 'content': ''.join(parts)},
                                                'logprobs': {'content': [token(p) for p in parts]}}]}
                    else:
                        messages = body.get('input', []) if is_responses else body['messages']
                        finished = any(m.get('role') == 'tool' or m.get('type') == 'function_call_output' for m in messages)
                        if not finished and not is_responses:
                            fixture.conversations += 1
                        value = 2 if is_responses or fixture.conversations % 2 == 0 else 1
                        arguments = json.dumps({'command': 'test -z "${GENERATOR_API_KEY}${VERIFIER_API_KEY}${BASELINE_API_KEY}${NEBIUS_API_KEY}${OPENAI_API_KEY}" && '
                                                + f"test ! -e /tests/test.sh && printf 'print({value})\\n' > /app/answer.py",
                                                'cwd': '/app', 'timeout_sec': 5})
                        if is_responses:
                            if finished and not any(m.get('type') == 'reasoning' and m.get('encrypted_content') == 'opaque-fixture' for m in messages):
                                self.send_response(400)
                                self.send_header('Content-Type', 'application/json')
                                self.end_headers()
                                self.wfile.write(b'{"error":{"message":"Stateless reasoning context was not preserved"}}')
                                return
                            output = [{'type': 'message', 'id': 'msg_mock', 'role': 'assistant', 'status': 'completed',
                                       'content': [{'type': 'output_text', 'text': 'Done.', 'annotations': []}]}] if finished else [
                                {'type': 'function_call', 'id': 'fc_mock', 'call_id': 'call_mock',
                                 'status': 'completed', 'name': 'shell', 'arguments': arguments}]
                            if not finished:
                                reasoning = {'type': 'reasoning', 'id': 'rs_mock', 'summary': []}
                                if 'reasoning.encrypted_content' in body.get('include', []):
                                    reasoning['encrypted_content'] = 'opaque-fixture'
                                output.insert(0, reasoning)
                            payload = {'id': 'resp_mock', 'object': 'response', 'created_at': 0, 'model': body['model'],
                                       'status': 'completed', 'output': output,
                                       'usage': {'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15}}
                        else:
                            message = {'role': 'assistant', 'content': 'Done.'} if finished else {
                                'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'call_mock', 'type': 'function',
                                    'function': {'name': 'shell', 'arguments': arguments}}]}
                            payload = {'id': 'mock-generator', 'object': 'chat.completion', 'created': 0, 'model': body['model'],
                                       'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
                                       'choices': [{'index': 0, 'finish_reason': 'stop' if finished else 'tool_calls', 'message': message}]}
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode())

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def env(self):
        result = {}
        for role in ('GENERATOR', 'VERIFIER', 'BASELINE'):
            result[role + '_BASE_URL'] = f'http://127.0.0.1:{self.server.server_port}/v1'
            result['OPENAI_API_KEY' if role == 'BASELINE' else role + '_API_KEY'] = 'fake-' + role.lower() + '-secret'
            result[role + '_MODEL'] = role.lower() + '-fixture'
        result['VERIFIER_CONTEXT_TOKENS'] = '131072'
        result['VERIFIER_EXTRA_BODY'] = '{}'
        return result

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.generators_ready.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
