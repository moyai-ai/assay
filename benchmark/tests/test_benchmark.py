import argparse
import asyncio
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from harbor.environments.base import ExecResult
from harbor.models.agent.context import AgentContext

from benchmark.agent import AssayAgent, execute, validate_exec
from benchmark.cli import SUITE, build_config, positive_multiplier
from benchmark.report import summarize


class FakeEnvironment:
    def __init__(self):
        self.commands = []
        self.downloads = []

    async def exec(self, command, **kwargs):
        self.commands.append((command, kwargs))
        return ExecResult(stdout='sandbox-only-output', stderr='', return_code=0)

    async def download_file(self, source, target):
        self.downloads.append((source, target))
        Path(target).write_bytes(b'mock archive')


class BridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_exec_sandbox_only_and_bounded(self):
        environment = FakeEnvironment()
        result = await execute(environment, {'type': 'exec', 'id': 0, 'command': 'echo hi', 'cwd': '/app', 'timeout_sec': 3})
        self.assertEqual(result['exit_code'], 0)
        command, kwargs = environment.commands[0]
        self.assertIn('timeout --kill-after=5s 3s', command)
        self.assertIn('tail -c 16384', command)
        self.assertEqual(kwargs, {'cwd': '/app', 'timeout_sec': 18})
        self.assertNotIn('env', kwargs)

    async def test_invalid_arguments_are_recoverable_without_execution(self):
        environment = FakeEnvironment()
        result = await execute(environment, {'type': 'exec', 'id': 0, 'command': 'true', 'cwd': '.', 'timeout_sec': 1})
        self.assertTrue(result['validation_error'])
        self.assertIsNone(result['exit_code'])
        self.assertEqual(environment.commands, [])

    async def test_execution_error_is_not_a_success(self):
        class Broken(FakeEnvironment):
            async def exec(self, *args, **kwargs):
                raise RuntimeError('execution unavailable')
        result = await execute(Broken(), {'type': 'exec', 'id': 0, 'command': 'true', 'cwd': None, 'timeout_sec': 1})
        self.assertIsNone(result['exit_code'])
        self.assertTrue(result['execution_error'])

    async def test_snapshot_failure_is_explicit_and_not_a_fake_archive(self):
        class TooLarge(FakeEnvironment):
            async def exec(self, command, **kwargs):
                return ExecResult(stdout='', stderr='file size limit', return_code=153)

        with tempfile.TemporaryDirectory() as directory:
            agent = AssayAgent(logs_dir=Path(directory), model_name='mock')
            environment = TooLarge()
            await agent._snapshot(environment)
            self.assertIn('snapshot failed', (Path(directory) / 'snapshot-error.txt').read_text())
            self.assertFalse((Path(directory) / 'app-before-grading.tar.gz').exists())
            self.assertEqual(environment.downloads, [])

    async def test_actual_assay_worker_with_mock_provider_and_sandbox(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(body)
                calls_tool = len(requests) <= 2
                message = {'role': 'assistant', 'content': 'Mock task complete.'}
                if calls_tool:
                    message = {'role': 'assistant', 'content': None, 'tool_calls': [
                        {'id': f'call{len(requests)}', 'type': 'function', 'function': {'name': 'shell', 'arguments': json.dumps({
                            'command': 'printf mock', 'cwd': '.' if len(requests) == 1 else '/app', 'timeout_sec': 5,
                        })}}]}
                payload = {'id': f'chat-{len(requests)}', 'object': 'chat.completion', 'created': 0,
                           'model': 'mock-coder', 'usage': {'prompt_tokens': 11, 'completion_tokens': 7, 'total_tokens': 18},
                           'choices': [{'index': 0, 'finish_reason': 'tool_calls' if calls_tool else 'stop', 'message': message}]}
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode())

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                'GENERATOR_BASE_URL': f'http://127.0.0.1:{server.server_port}/v1',
                'GENERATOR_API_KEY': 'fake-test-secret',
            }):
                agent = AssayAgent(logs_dir=Path(directory), model_name='mock-coder', max_turns=3)
                environment = FakeEnvironment()
                context = AgentContext()
                await agent.setup(environment)
                await asyncio.wait_for(agent.run('Implement the mock task', environment, context), timeout=30)
                self.assertEqual(len(requests), 3)
                self.assertEqual(context.n_input_tokens, 33)
                self.assertEqual(context.n_output_tokens, 21)
                self.assertIsNone(context.cost_usd)
                self.assertEqual(requests[0]['model'], 'mock-coder')
                self.assertEqual(requests[0]['reasoning_effort'], 'none')
                self.assertTrue(all(request['max_tokens'] == 32768 for request in requests))
                function = requests[0]['tools'][0]['function']
                self.assertFalse(function['strict'])
                self.assertEqual(function['parameters']['properties']['command'], {'type': 'string'})
                self.assertTrue(any(m['role'] == 'tool' and 'validation_error' in m['content'] for m in requests[1]['messages']))
                self.assertTrue(any(m['role'] == 'tool' and 'sandbox-only-output' in m['content'] for m in requests[2]['messages']))
                self.assertTrue(any('printf mock' in command for command, _ in environment.commands))
                self.assertTrue(all('env' not in kwargs for _, kwargs in environment.commands))
                for path in Path(directory).rglob('*'):
                    if path.is_file():
                        self.assertNotIn(b'fake-test-secret', path.read_bytes())
                responses = sorted((Path(directory) / 'model').glob('*-response.json'))
                self.assertEqual(len(responses), 3)
                self.assertEqual(json.loads(responses[-1].read_text())['body']['choices'][0]['finish_reason'], 'stop')
                self.assertTrue((Path(directory) / 'trajectory.json').is_file())
                self.assertTrue((Path(directory) / 'commands.jsonl').is_file())
                self.assertEqual(len(environment.downloads), 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    async def test_cancelled_trial_terminates_worker(self):
        original = asyncio.create_subprocess_exec
        children = []

        async def spawn(*args, **kwargs):
            import sys
            child = await original(sys.executable, '-c', 'import time; time.sleep(60)', **kwargs)
            children.append(child)
            return child

        with tempfile.TemporaryDirectory() as directory, patch('benchmark.agent.asyncio.create_subprocess_exec', spawn):
            agent = AssayAgent(logs_dir=Path(directory), model_name='mock')
            run = asyncio.create_task(agent.run('task', FakeEnvironment(), AgentContext()))
            while not children:
                await asyncio.sleep(0.01)
            run.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await run
            self.assertIsNotNone(children[0].returncode)

    async def test_eof_without_done_fails_trial(self):
        original = asyncio.create_subprocess_exec

        async def spawn(*args, **kwargs):
            import sys
            return await original(sys.executable, '-c', 'import sys; sys.stdin.readline()', **kwargs)

        with tempfile.TemporaryDirectory() as directory, patch('benchmark.agent.asyncio.create_subprocess_exec', spawn):
            agent = AssayAgent(logs_dir=Path(directory), model_name='mock')
            with self.assertRaisesRegex(RuntimeError, 'without a completion'):
                await agent.run('task', FakeEnvironment(), AgentContext())


class ConfigTests(unittest.TestCase):
    def test_invalid_exec(self):
        valid = {'type': 'exec', 'id': 0, 'command': 'true', 'cwd': None, 'timeout_sec': 1}
        for updates in [{'timeout_sec': 0}, {'timeout_sec': 121}, {'timeout_sec': True},
                        {'cwd': '../host'}, {'command': ''}, {'command': 'x' * 32769},
                        {'id': '0'}, {'type': 'host_exec'}]:
            with self.assertRaises(ValueError):
                validate_exec({**valid, **updates})

    def test_job_is_ten_tasks_grading_enabled_no_secret_injection(self):
        self.assertEqual(len(SUITE['tasks']), 10)
        self.assertEqual(len(set(SUITE['tasks'])), 10)
        config = build_config(Path('/tmp/run'), SUITE['tasks'], 'deepseek-ai/DeepSeek-V4.1-Flash', 1, 1, 60, 8192)
        self.assertEqual(len(config['tasks']), 10)
        self.assertFalse(config['verifier']['disable'])
        self.assertEqual(config['retry']['max_retries'], 0)
        self.assertTrue(config['environment']['delete'])
        self.assertFalse(config['environment'].get('mounts'))
        self.assertFalse(config['environment'].get('env'))
        self.assertFalse(config['agents'][0].get('env'))
        self.assertEqual(config['agents'][0]['import_path'], 'benchmark.agent:AssayAgent')
        self.assertEqual(config['agents'][0]['kwargs']['reasoning_effort'], 'none')
        self.assertEqual(config['agents'][0]['kwargs']['request_timeout_sec'], 300)
        self.assertEqual(config['agent_timeout_multiplier'], 1.0)
        self.assertEqual(config['verifier_timeout_multiplier'], 1.0)

    def test_phase_multipliers_leave_request_timeout_and_other_phases_unchanged(self):
        config = build_config(Path('/tmp/run'), SUITE['tasks'], 'mock', 1, 1, 60, 8192,
                              agent_timeout_multiplier=2, verifier_timeout_multiplier=8)
        self.assertEqual(config['agent_timeout_multiplier'], 2)
        self.assertEqual(config['verifier_timeout_multiplier'], 8)
        self.assertEqual(config['timeout_multiplier'], 1)
        self.assertIsNone(config['agent_setup_timeout_multiplier'])
        self.assertIsNone(config['environment_build_timeout_multiplier'])
        self.assertEqual(config['agents'][0]['kwargs']['request_timeout_sec'], 300)

    def test_positive_finite_multipliers(self):
        self.assertEqual(positive_multiplier('0.5'), 0.5)
        self.assertEqual(positive_multiplier('8'), 8)
        for value in ['0', '-1', 'nan', 'inf', '-inf']:
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                positive_multiplier(value)


class ReportTests(unittest.TestCase):
    def test_pass_fail_error_missing_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'manifest.json').write_text(json.dumps({
                'tasks': ['a', 'b', 'c', 'd'], 'attempts': 1, 'model': 'mock', 'dataset_revision': 'test',
            }))
            for task, reward, exception in [('a', 1, None), ('b', 0, None), ('c', None, {'exception_type': 'Timeout'})]:
                trial = root / 'job' / f'{task}__test'
                trial.mkdir(parents=True)
                (trial / 'result.json').write_text(json.dumps({
                    'task_name': f'terminal-bench/{task}', 'trial_name': trial.name,
                    'finished_at': '2026-09-24T00:00:00Z',
                    'verifier_result': {'rewards': {'reward': reward}}, 'exception_info': exception,
                }))
            # The aggregate Harbor result must not be mistaken for a trial.
            (root / 'job/result.json').write_text('{}')
            report = summarize(root)
            self.assertEqual([report[k] for k in ('passed', 'failed', 'errors', 'pending')], [1, 1, 1, 1])
            self.assertEqual(report['success_fraction_of_planned'], 0.25)
            self.assertEqual(report['success_fraction_of_graded'], 0.5)
            self.assertFalse(report['complete'])
            self.assertEqual(report['agent_timeout_multiplier'], 1.0)
            self.assertEqual(report['verifier_timeout_multiplier'], 1.0)
            manifest = json.loads((root / 'manifest.json').read_text())
            manifest.update(agent_timeout_multiplier=2, verifier_timeout_multiplier=8)
            (root / 'manifest.json').write_text(json.dumps(manifest))
            report = summarize(root)
            self.assertEqual(report['agent_timeout_multiplier'], 2)
            self.assertEqual(report['verifier_timeout_multiplier'], 8)

    def test_missing_reward_and_duplicate_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'manifest.json').write_text(json.dumps({
                'tasks': ['a'], 'attempts': 1, 'model': 'mock', 'dataset_revision': 'test',
            }))
            trial = root / 'job/one'
            trial.mkdir(parents=True)
            result = {'task_name': 'a', 'trial_name': 'one', 'finished_at': 'now', 'verifier_result': {'rewards': None}}
            (trial / 'result.json').write_text(json.dumps(result))
            self.assertEqual(summarize(root)['errors'], 1)
            second = root / 'job/two'
            second.mkdir()
            (second / 'result.json').write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, 'Too many results'):
                summarize(root)


if __name__ == '__main__':
    unittest.main()
