import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark.evidence import save_json
from benchmark.scaling import make_manifest, parser, run_experiment
from benchmark.scheduling import Budget, Resources


class BudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_weighted_fifo_and_cancelled_waiters_do_not_leak(self):
        budget = Budget(32)
        await budget.acquire(16)
        first = asyncio.create_task(budget.acquire(32))
        second = asyncio.create_task(budget.acquire(16))
        await asyncio.sleep(0)
        self.assertFalse(second.done())  # FIFO: no starvation of large requests
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        await second
        self.assertEqual(budget.active, 32)
        budget.release(32)
        self.assertEqual(budget.active, 0)
        with self.assertRaises(ValueError):
            await budget.acquire(33)

    async def test_cancellation_after_grant_returns_the_permit(self):
        budget = Budget(1)
        await budget.acquire()
        waiter = asyncio.create_task(budget.acquire())
        await asyncio.sleep(0)
        budget.release()  # grant, but don't let the awaiting task resume yet
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        self.assertEqual(budget.active, 0)
        async with budget.lease():
            self.assertEqual(budget.active, 1)
        self.assertEqual(budget.active, 0)


class ParallelStudyTests(unittest.IsolatedAsyncioTestCase):
    def config(self):
        env = {'VERIFIER_MODEL': 'synthetic', 'VERIFIER_CONTEXT_TOKENS': '131072', 'VERIFIER_EXTRA_BODY': '{}'}
        with patch.dict(os.environ, env):
            result = make_manifest(parser().parse_args([
                '--candidate-counts', '1,4', '--verifier-repetitions', '4', '--baseline-model', 'synthetic-baseline',
                '--task-concurrency', '4', '--concurrency', '16', '--baseline-concurrency', '4',
                '--verifier-concurrency', '16', '--judge-concurrency', '32', '--max-live-containers', '20',
                '--grading-concurrency', '4', '--arm-concurrency', '4']))
        result['tasks'] = [f'toy{i}' for i in range(4)]
        return result

    async def test_16_generations_and_32_judge_reservations_are_global_not_per_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config()
            save_json(root / 'manifest.json', config)
            for task in config['tasks']:
                path = root / 'dataset/tasks' / task
                path.mkdir(parents=True)
                (path / 'task.toml').write_text('')
            generated = asyncio.Event()
            judged = asyncio.Event()
            at_generation_peak = asyncio.Event()
            at_judge_peak = asyncio.Event()
            observed = {}

            async def selector(*_):
                return {'capabilityPassed': True}

            async def fake_pool(path, task_path, manifest, *, selector, resources):
                observed['resources'] = resources
                path.mkdir(parents=True)
                resources.phases[path.name] = 'generating'

                async def generate(budget):
                    async with budget.lease():
                        if resources.generation.active == 16 and resources.baseline.active == 4:
                            at_generation_peak.set()
                        await generated.wait()
                await asyncio.gather(*[generate(resources.generation) for _ in range(4)], generate(resources.baseline))
                resources.phases[path.name] = 'selecting'
                async with resources.arms.lease(), resources.judge.lease(16):
                    if resources.judge.active == 32:
                        at_judge_peak.set()
                    await judged.wait()
                resources.phases[path.name] = 'grading'
                async with resources.grading.lease():
                    await asyncio.sleep(0.01)
                save_json(path / 'result.json', {'decisions': {}, 'candidates': {}, 'selection_locked': True, 'wall_seconds': 1})
                resources.phases[path.name] = 'finished'

            job = asyncio.create_task(run_experiment(root, config, task_checkout=root / 'dataset',
                                                     selector=selector, pool_runner=fake_pool))
            try:
                await asyncio.wait_for(at_generation_peak.wait(), 5)
                resources = observed['resources']
                self.assertEqual(resources.containers.active, 20)
                self.assertEqual(resources.generation.active, 16)
                self.assertEqual(resources.baseline.active, 4)
                generated.set()
                await asyncio.wait_for(at_judge_peak.wait(), 5)
                self.assertEqual(resources.judge.active, 32)
                judged.set()
                await asyncio.wait_for(job, 5)
                self.assertTrue(all(b.active == 0 for b in resources.budgets.values()))
                self.assertEqual(resources.generation.peak, 16)
                self.assertEqual(resources.judge.peak, 32)
                self.assertTrue(all(b.peak <= b.capacity for b in resources.budgets.values()))
                progress = json.loads((root / 'progress.json').read_text())
                self.assertEqual(progress['budgets']['judge']['peak'], 32)
            finally:
                job.cancel()
                await asyncio.gather(job, return_exceptions=True)

    async def test_whole_pool_reservation_and_cancelled_admission_release_all_resources(self):
        budget = Budget(10)
        entered = []
        gate = asyncio.Event()

        async def pool(index):
            async with budget.lease(5):
                entered.append(index)
                await gate.wait()
        jobs = [asyncio.create_task(pool(i)) for i in range(4)]
        await asyncio.sleep(0)
        self.assertEqual(entered, [0, 1])
        for job in jobs:
            job.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
        self.assertEqual(budget.active, 0)
        self.assertEqual(budget.snapshot()['queued'], 0)

    def test_invalid_capacities_fail_before_calls(self):
        env = {'VERIFIER_MODEL': 'synthetic', 'VERIFIER_CONTEXT_TOKENS': '131072', 'VERIFIER_EXTRA_BODY': '{}'}
        with patch.dict(os.environ, env):
            for args in [
                ['--max-live-containers', '3', '--candidate-counts', '4'],
                ['--verifier-concurrency', '32', '--judge-concurrency', '16'],
            ]:
                with self.assertRaises(ValueError):
                    make_manifest(parser().parse_args(args))
            result = make_manifest(parser().parse_args(['--concurrency', '32', '--verifier-concurrency', '32']))
            self.assertEqual(result['concurrency'], 32)
            self.assertEqual(result['verifier']['concurrency'], 32)


if __name__ == '__main__':
    unittest.main()
