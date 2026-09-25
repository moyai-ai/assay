import asyncio
import io
import json
import os
import tarfile
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from harbor.models.agent.context import AgentContext
from harbor.trial.hooks import TrialEvent

from benchmark.agent import AssayAgent
from benchmark.evidence import read_tree, candidate_evidence, digest, save_json
from benchmark.scaling import experiment_arms, make_manifest, parser, run_task_pool, run_experiment, validate_task_compatibility
from benchmark.scaling_report import grader_health, model_usage, paired_comparison, write_scaling_report
from benchmark.tests.mock_provider import MockProvider
from benchmark.tests.test_benchmark import FakeEnvironment


def archive(entries):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as tar:
        for name, data in entries:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    buffer.seek(0)
    return buffer


def manifest(env, baseline=False):
    with patch.dict(os.environ, env):
        args = parser().parse_args(['--task', 'regex-log', '--candidate-counts', '1,2',
                                    '--verifier-repetitions', '2', '--max-turns', '1', '--concurrency', '1',
                                    *(['--baseline-model', 'baseline-fixture'] if baseline else [])])
        value = make_manifest(args)
    value['tasks'] = ['toy']
    return value


def write_grade(path, reward, stdout='one test executed'):
    path.mkdir(parents=True, exist_ok=True)
    (path / 'test-stdout.txt').write_text(stdout)
    save_json(path / 'ctrf.json', {'results': {'summary': {'tests': 1, 'passed': reward, 'failed': 1 - reward}}})


class EvidenceTests(unittest.TestCase):
    def test_archive_read_is_safe_and_content_based(self):
        before = read_tree(archive([('./answer.py', b'print(1)\n'), ('./.git/secret', b'ignored')]))
        after = read_tree(archive([('./answer.py', b'print(2)\n'), ('./binary', b'\0\xff')]))
        self.assertNotIn('.git/secret', before['files'])
        self.assertIn('omitted', after['files']['binary'])
        evidence = candidate_evidence('c001', 'a' * 64, before, after, Path('/nonexistent-transcript'))
        self.assertIn('+print(2)', evidence['diff'])
        self.assertIn('content_not_reviewed', evidence['diff'])
        self.assertNotIn('reward', evidence)
        for name in ('../../escape', '/absolute'):
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                read_tree(archive([(name, b'x')]))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            read_tree(archive([('a', b'x'), ('a', b'y')]))
        with self.assertRaisesRegex(ValueError, 'byte limit'):
            read_tree(archive([('large', b'x' * 30_000)]), byte_limit=20_480)

    def test_grader_health_does_not_accept_false_browser_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_grade(path, 1, 'Failed to create driver or process file: unavailable')
            self.assertEqual(grader_health(path, 1)['status'], 'suspect')
            write_grade(path, 1)
            self.assertEqual(grader_health(path, 1)['status'], 'ok')
            (path / 'test-stdout.txt').unlink()
            (path / 'test-stdout.txt').symlink_to(path / 'ctrf.json')
            self.assertEqual(grader_health(path, 1)['status'], 'suspect')

    def test_fixed_grid_no_singleton_verifier_and_no_duplicate_effective_pivots(self):
        arms = experiment_arms([1, 2, 4], [2, 4], [2, 4], 'baseline')
        self.assertEqual(sum(a['kind'] == 'vanilla' for a in arms), 1)
        self.assertEqual(sum(a['n'] == 2 and a['kind'] == 'selection' for a in arms), 2)
        self.assertEqual(sum(a['n'] == 4 and a['kind'] == 'selection' for a in arms), 4)


class ComparisonTests(unittest.TestCase):
    def test_unsupported_task_lifecycles_are_rejected_before_model_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / 'tasks/toy'
            (task / 'environment').mkdir(parents=True)
            config = task / 'task.toml'
            config.write_text('')
            validate_task_compatibility(root, ['toy'])
            for content in ('[verifier]\nenvironment_mode="separate"', '[[steps]]\nname="first"'):
                config.write_text(content)
                with self.assertRaisesRegex(ValueError, 'single-step'):
                    validate_task_compatibility(root, ['toy'])
            config.write_text('')
            (task / 'environment/compose.yaml').write_text('services: {}')
            with self.assertRaisesRegex(ValueError, 'multi-service'):
                validate_task_compatibility(root, ['toy'])

    def test_paired_differences_do_not_claim_equivalence(self):
        def arm(name, rewards):
            return {'id': name, 'observations': [{'reward': r} for r in rewards],
                    'pass_rate': sum(rewards) / len(rewards) if None not in rewards else None}
        reference = arm('external', [0] * 6)
        result = paired_comparison(arm('scaled', [1] * 6), reference)
        self.assertEqual(result['pass_rate_difference_pp'], 100)
        self.assertEqual((result['wins'], result['losses'], result['ties']), (6, 0, 0))
        self.assertEqual(result['exact_paired_two_sided_p'], 0.03125)
        tied = paired_comparison(reference, reference)
        self.assertEqual(tied['pass_rate_difference_pp'], 0)
        self.assertFalse(tied['equivalence_test_performed'])
        incomplete = paired_comparison(arm('scaled', [1] * 5 + [None]), reference)
        self.assertEqual(incomplete['paired_graded_tasks'], 5)
        self.assertIsNone(incomplete['pass_rate_difference_pp'])
        self.assertIsNone(incomplete['exact_paired_two_sided_p'])

    def test_malformed_api_bodies_leave_usage_unknown_without_breaking_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            save_json(path / 'model/model-0000-request.json', {})
            for body in (None, [], 'bad', {'usage': 'bad'}, {'usage': {'prompt_tokens_details': [1]}}):
                save_json(path / 'model/model-0000-response.json', {'status': 200, 'body': body})
                usage = model_usage(path)
                self.assertFalse(usage['complete'])
                self.assertEqual((usage['requests'], usage['responses']), (1, 1))

    def test_explicit_verifier_reasoning_is_recorded_without_changing_generator(self):
        env = {'VERIFIER_MODEL': 'mock', 'VERIFIER_CONTEXT_TOKENS': '131072',
               'VERIFIER_EXTRA_BODY': '{"reasoning_effort":"high","temperature":0}'}
        with patch.dict(os.environ, env):
            default = make_manifest(parser().parse_args([]))
            explicit = make_manifest(parser().parse_args(['--verifier-reasoning-effort', 'none']))
            provider = make_manifest(parser().parse_args(['--verifier-reasoning-effort', 'provider']))
        self.assertEqual(default['verifier']['extraBody']['reasoning_effort'], 'high')
        self.assertEqual(explicit['verifier']['extraBody'], {'reasoning_effort': 'none', 'temperature': 0})
        self.assertNotIn('reasoning_effort', provider['verifier']['extraBody'])
        self.assertEqual(explicit['generator'], default['generator'])


class BudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_turn_limit_is_gradeable_and_responses_baseline_uses_separate_key(self):
        with MockProvider() as provider, patch.dict(os.environ, provider.env), tempfile.TemporaryDirectory() as directory:
            for profile, api, turns in [('generator', 'chat-completions', 1), ('baseline', 'responses', 1), ('baseline', 'responses', 2)]:
                logs = Path(directory) / f'{profile}-{turns}'
                context = AgentContext()
                agent = AssayAgent(logs_dir=logs, model_name=profile + '-fixture', max_turns=turns,
                                   credential_profile=profile, api=api)
                await asyncio.wait_for(agent.run('Write answer.py', FakeEnvironment(), context), 30)
                reason = 'turn_limit' if turns == 1 else 'completed'
                self.assertEqual(context.metadata['stop_reason'], reason)
                self.assertEqual(context.n_input_tokens, 10 * turns)
                data = json.loads((logs / 'generation.json').read_text())
                self.assertEqual(data['stopReason'], reason)
                self.assertEqual(data['finalOutput'], None if turns == 1 else 'Done.')
                self.assertTrue((logs / 'trajectory.json').exists())
                self.assertTrue((logs / 'app-before-grading.tar.gz').exists())
            self.assertEqual(provider.requests[0]['authorization'], 'Bearer fake-generator-secret')
            self.assertEqual(provider.requests[1]['authorization'], 'Bearer fake-baseline-secret')
            self.assertEqual(provider.requests[1]['path'], '/v1/responses')
            self.assertFalse(provider.requests[1]['body']['store'])
            self.assertIn('reasoning.encrypted_content', provider.requests[1]['body']['include'])
            self.assertTrue(any(item.get('encrypted_content') == 'opaque-fixture'
                                for item in provider.requests[-1]['body']['input']))
            for path in Path(directory).rglob('*.json'):
                self.assertNotIn('fake-baseline-secret', path.read_text())


class PoolTests(unittest.IsolatedAsyncioTestCase):
    async def harness(self, root, config, *, missing=None, hanging=False):
        events, workspaces = [], []
        selector_started = asyncio.Event()
        empty = read_tree(archive([]))
        task_path = root / 'source'
        task_path.mkdir()
        (task_path / 'instruction.md').write_text('Produce the correct answer.py independently.')

        class Workspace:
            image_id = 'sha256:fixture-image'
            paused = False
            captures = 0

            def __init__(self, candidate):
                self.candidate = candidate
                workspaces.append(self)

            async def pause(self):
                self.paused = True

            async def unpause(self):
                self.paused = False

            async def capture(self):
                self.captures += 1
                return empty if self.captures == 1 else read_tree(archive([('answer.py',
                    b'print(2)\n' if self.candidate == 'c002' else b'print(1)\n')]))

        class FakeTrial:
            def __init__(self, cfg):
                self.candidate = Path(cfg.agent.kwargs['host_logs_dir']).parent.name
                self.agent_environment = SimpleNamespace(candidate=self.candidate)
                trial_dir = cfg.trials_dir / cfg.trial_name
                trial_dir.mkdir(parents=True)
                self.paths = SimpleNamespace(result_path=trial_dir / 'result.json', verifier_dir=trial_dir / 'verifier')
                self.hooks = {}
                now = datetime.now(timezone.utc)
                self.result = SimpleNamespace(agent_result=AgentContext(metadata={'stop_reason': 'turn_limit'}),
                    agent_execution=SimpleNamespace(started_at=now, finished_at=now), exception_info=None, verifier_result=None)

            def add_hook(self, event, hook):
                self.hooks[event] = hook

            async def run(self):
                event = SimpleNamespace(result=self.result)
                await self.hooks[TrialEvent.AGENT_START](event)
                events.append(('generated', self.candidate))
                await self.hooks[TrialEvent.AGENT_END](event)
                await self.hooks[TrialEvent.VERIFICATION_START](event)
                events.append(('graded', self.candidate))
                reward = int(self.candidate == 'c002')
                self.result.verifier_result = SimpleNamespace(rewards={'reward': reward})
                write_grade(self.paths.verifier_dir, reward)
                self.paths.result_path.write_text('{}')
                return self.result

        async def factory(cfg):
            if Path(cfg.agent.kwargs['host_logs_dir']).parent.name == missing:
                raise RuntimeError('synthetic setup failure')
            return FakeTrial(cfg)

        async def workspace_factory(env, _):
            return Workspace(env.candidate)

        async def selector(payload, directory, timeout):
            self.assertEqual([event for event in events if event[0] == 'graded'], [])
            self.assertTrue(all(workspace.paused for workspace in workspaces))
            self.assertEqual([c['id'] for c in payload['candidates']], [f'c{i:03}' for i in range(1, len(payload['candidates']) + 1)])
            events.append(('input', payload))
            self.assertNotIn('official_reward', json.dumps(payload))
            self.assertNotIn('trial_result', json.dumps(payload))
            events.append(('selected', 'c002'))
            selector_started.set()
            if hanging:
                await asyncio.Event().wait()
            return {'winnerId': 'c002', 'verificationComplete': True, 'elapsed_seconds': 0.1}

        run = asyncio.create_task(run_task_pool(root / 'tasks' / 'toy', task_path, config,
                                 trial_factory=factory, workspace_factory=workspace_factory, selector=selector))
        return run, selector_started, events, workspaces

    async def test_barrier_no_deadlock_when_pool_exceeds_concurrency_and_correct_uplift(self):
        with MockProvider() as provider, tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = manifest(provider.env)
            save_json(root / 'manifest.json', config)
            run, _, events, workspaces = await self.harness(root, config)
            record = await asyncio.wait_for(run, 10)
            self.assertTrue(record['selection_locked'])
            self.assertTrue(all(not w.paused for w in workspaces))
            self.assertLess(events.index(('selected', 'c002')), events.index(('graded', 'c001')))
            report = write_scaling_report(root)
            vanilla, selected = report['arms']
            self.assertEqual(vanilla['pass_rate'], 0)
            self.assertEqual(selected['pass_rate'], 1)
            self.assertEqual(selected['uplift_vs_vanilla_pp'], 100)
            self.assertEqual(selected['observations'][0]['oracle_any_pass'], 1)
            self.assertEqual(selected['observations'][0]['random_selection_expected_reward'], 0.5)

    async def test_multiple_budgets_reuse_identical_pool_before_any_grading(self):
        with MockProvider() as provider, tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = manifest(provider.env)
            config.update(max_candidates=4, candidate_counts=[1, 2, 4],
                          arms=experiment_arms([1, 2, 4], [2, 4], [1, 2]))
            save_json(root / 'manifest.json', config)
            run, _, events, _ = await self.harness(root, config)
            await asyncio.wait_for(run, 10)
            self.assertEqual(sum(e[0] == 'generated' for e in events), 4)
            payloads = [e[1] for e in events if e[0] == 'input']
            self.assertEqual(len(payloads), 8)
            hashes = {}
            for payload in payloads:
                for candidate in payload['candidates']:
                    key = candidate['id']
                    self.assertEqual(hashes.setdefault(key, digest(candidate)), digest(candidate))
            self.assertLess(max(i for i, e in enumerate(events) if e[0] == 'selected'),
                            min(i for i, e in enumerate(events) if e[0] == 'graded'))
            report = write_scaling_report(root)
            self.assertTrue(all(a['pass_rate'] == 1 for a in report['arms'] if a['kind'] == 'selection'))

    async def test_missing_candidate_is_not_replaced_or_dropped(self):
        with MockProvider() as provider, tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = manifest(provider.env)
            save_json(root / 'manifest.json', config)
            run, _, events, _ = await self.harness(root, config, missing='c002')
            await asyncio.wait_for(run, 10)
            self.assertNotIn(('selected', 'c002'), events)
            report = write_scaling_report(root)
            self.assertEqual(report['arms'][1]['errors'], 1)
            self.assertIsNone(report['arms'][1]['pass_rate'])
            self.assertEqual(report['arms'][0]['failed'], 1)

    async def test_cancellation_releases_paused_environments_without_grading(self):
        with MockProvider() as provider, tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = manifest(provider.env)
            run, started, events, workspaces = await self.harness(root, config, hanging=True)
            await asyncio.wait_for(started.wait(), 10)
            run.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await run
            self.assertTrue(all(not w.paused for w in workspaces))
            self.assertFalse(any(e[0] == 'graded' for e in events))
            self.assertFalse((root / 'tasks/toy/decisions.json').exists())


@unittest.skipUnless(os.getenv('ASSAY_DOCKER_TEST') == '1', 'Set ASSAY_DOCKER_TEST=1 for real Docker orchestration')
class DockerPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_harbor_worker_tournament_grader_and_external_responses_baseline(self):
        with MockProvider() as provider, patch.dict(os.environ, provider.env), tempfile.TemporaryDirectory(prefix='assay-scaling-') as directory:
            root = Path(directory) / Path(directory).name
            root.mkdir()
            config = manifest(provider.env, baseline=True)
            save_json(root / 'manifest.json', config)
            task = Path(directory) / 'dataset/tasks/toy'
            (task / 'environment').mkdir(parents=True)
            (task / 'tests').mkdir()
            (task / 'instruction.md').write_text('Create /app/answer.py which prints the integer 2. Use the provided shell.')
            (task / 'task.toml').write_text('''schema_version = "1.1"
[task]
name = "assay-fixture/toy"
[agent]
timeout_sec = 60
[verifier]
timeout_sec = 30
[environment]
build_timeout_sec = 180
cpus = 1
memory_mb = 512
storage_mb = 1024
allow_internet = true
''')
            (task / 'environment/Dockerfile').write_text('FROM python:3.13-slim-bookworm\nWORKDIR /app\n')
            (task / 'tests/test.sh').write_text('''#!/bin/bash
mkdir -p /logs/verifier
python3 - <<'PY'
import json,subprocess
from pathlib import Path
print('HIDDEN_GRADER_CANARY')
p=subprocess.run(['python3','/app/answer.py'],capture_output=True,text=True)
r=int(p.returncode==0 and p.stdout.strip()=='2')
Path('/logs/verifier/reward.txt').write_text(str(r))
Path('/logs/verifier/ctrf.json').write_text(json.dumps({'results':{'summary':{'tests':1,'passed':r,'failed':1-r}}}))
PY
''')
            report = await asyncio.wait_for(run_experiment(root, config, task_checkout=Path(directory) / 'dataset'), 300)
            vanilla, selected, baseline = report['arms']
            self.assertEqual((vanilla['pass_rate'], selected['pass_rate'], baseline['pass_rate']), (0, 1, 1),
                             json.dumps(json.loads((root / 'tasks/toy/result.json').read_text()), indent=2))
            self.assertEqual(selected['observations'][0]['generation_stop'], 'turn_limit')
            self.assertTrue(report['actual_model_usage_all_experiment_cells']['complete'])
            self.assertEqual(selected['gap_vs_external_pp'], 0)
            self.assertEqual(vanilla['gap_vs_external_pp'], -100)
            self.assertFalse(selected['comparisons']['external']['equivalence_test_performed'])
            for request in provider.requests:
                text = json.dumps(request['body'])
                self.assertNotIn('HIDDEN_GRADER_CANARY', text)
                self.assertNotIn('official_reward', text)
            judge_inputs = list((root / 'tasks/toy/selection').glob('*/input.json'))
            self.assertEqual(len(judge_inputs), 1)
            text = judge_inputs[0].read_text()
            self.assertNotIn('baseline-fixture', text)
            self.assertNotIn('fake-generator-secret', text)
            self.assertNotIn('/verifier/', text)

    async def test_twenty_real_containers_with_sixteen_simultaneous_generators(self):
        with MockProvider(generation_barrier=16) as provider, patch.dict(os.environ, provider.env), tempfile.TemporaryDirectory(prefix='assay-parallel-') as directory:
            root = Path(directory) / 'parallel-study'
            root.mkdir()
            config = manifest(provider.env, baseline=True)
            config.update(tasks=[f'toy{i}' for i in range(4)], max_candidates=4, candidate_counts=[1, 4],
                          arms=experiment_arms([1, 4], [4], [2], 'baseline-fixture'),
                          concurrency=16, baseline_concurrency=4, task_concurrency=4,
                          max_live_containers=20, setup_concurrency=20,
                          grading_concurrency=4, snapshot_concurrency=2, arm_concurrency=4,
                          pair_concurrency=8, judge_concurrency=32, provider_concurrency=32)
            config['verifier']['concurrency'] = 16
            save_json(root / 'manifest.json', config)
            for name in config['tasks']:
                task = Path(directory) / 'dataset/tasks' / name
                (task / 'environment').mkdir(parents=True)
                (task / 'tests').mkdir()
                (task / 'instruction.md').write_text('Create /app/answer.py which prints 2. Use the provided shell.')
                (task / 'task.toml').write_text('''[agent]
timeout_sec=180
[verifier]
timeout_sec=60
[environment]
build_timeout_sec=180
cpus=1
memory_mb=512
storage_mb=1024
''')
                (task / 'environment/Dockerfile').write_text('FROM python:3.13-slim-bookworm\nWORKDIR /app\n')
                (task / 'tests/test.sh').write_text('''#!/bin/bash
mkdir -p /logs/verifier
python3 - <<'PY'
import json,subprocess
from pathlib import Path
print('HIDDEN_PARALLEL_CANARY')
p=subprocess.run(['python3','/app/answer.py'],capture_output=True,text=True)
r=int(p.returncode==0 and p.stdout.strip()=='2')
Path('/logs/verifier/reward.txt').write_text(str(r))
Path('/logs/verifier/ctrf.json').write_text(json.dumps({'results':{'summary':{'tests':1,'passed':r,'failed':1-r}}}))
PY
''')
            report = await asyncio.wait_for(run_experiment(root, config, task_checkout=Path(directory) / 'dataset'), 600)
            self.assertEqual(provider.generators_arrived, 16)
            self.assertTrue(all(arm['errors'] == arm['pending'] == 0 for arm in report['arms']))
            self.assertEqual(report['arms'][-1]['pass_rate'], 1)
            self.assertTrue(report['actual_model_usage_all_experiment_cells']['complete'])
            progress = json.loads((root / 'progress.json').read_text())
            self.assertEqual(progress['budgets']['generation']['peak'], 16)
            self.assertEqual(progress['budgets']['containers']['peak'], 20)
            self.assertTrue(all(b['active'] == 0 and b['peak'] <= b['capacity'] for b in progress['budgets'].values()))
            for task in config['tasks']:
                record = json.loads((root / 'tasks' / task / 'result.json').read_text())
                lock = json.loads((root / 'tasks' / task / 'decisions.json').read_text())
                self.assertTrue(record['selection_locked'])
                self.assertEqual(len(record['candidates']), 5)
                self.assertTrue(all(c['grading_started_at'] >= lock['locked_at'] for c in record['candidates'].values()))
            self.assertFalse(any('HIDDEN_PARALLEL_CANARY' in json.dumps(req['body']) for req in provider.requests))


if __name__ == '__main__':
    unittest.main()
