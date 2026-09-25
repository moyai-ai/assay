import asyncio
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from benchmark.dependencies import UV_ASSETS, bootstrap_grader, prepare_assets, profile


class DependencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_bootstrap_is_generic_pinned_and_does_not_upload_tests(self):
        commands, uploads = [], []

        class Environment:
            async def exec(self, command, **_):
                commands.append(command)
                return SimpleNamespace(return_code=0, stdout='x86_64' if command == 'uname -m' else 'pytest 8.4.1', stderr='')

            async def upload_file(self, source, target):
                uploads.append((source, target))

        with tempfile.TemporaryDirectory() as directory:
            receipt = await bootstrap_grader(Environment(), 'regex-log', Path(directory), Path(directory) / 'logs')
        self.assertEqual(receipt['uv_wheel_sha256'], UV_ASSETS['x86_64']['sha256'])
        self.assertEqual({source.name for source, _ in uploads}, {'uv', 'uvx'})
        self.assertIn('pytest --version', commands[-1])
        for forbidden in ('/tests/', 'test_outputs', 'reward', 'API_KEY', 'solution'):
            self.assertNotIn(forbidden, '\n'.join(commands))
        with self.assertRaises(ValueError):
            profile('undeclared-task')

    async def test_failed_dependency_preflight_is_not_treated_as_success(self):
        class Environment:
            async def exec(self, command, **_):
                return SimpleNamespace(return_code=0 if command == 'uname -m' else 1,
                                       stdout='x86_64', stderr='synthetic failure')

            async def upload_file(self, *_):
                pass
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'preflight failed'):
                await bootstrap_grader(Environment(), 'regex-log', Path(directory), Path(directory))
            self.assertTrue((Path(directory) / 'grading-dependencies.json').exists())

    def test_corrupt_cache_is_rejected_without_a_model_or_binary_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'x86_64.whl').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                prepare_assets(root)


@unittest.skipUnless(os.getenv('ASSAY_DOCKER_TEST') == '1', 'Opt-in real Docker dependency preflight')
class DockerToolsTests(unittest.IsolatedAsyncioTestCase):
    async def test_prepared_uv_grades_offline_even_when_installer_fails(self):
        name = 'assay-dependency-test-' + uuid4().hex[:12]
        cache = Path(__file__).resolve().parents[1] / '.cache/grading-tools-uv-0.9.5'
        await asyncio.to_thread(prepare_assets, cache)

        async def docker(*args):
            process = await asyncio.create_subprocess_exec('docker', *args, stdout=asyncio.subprocess.PIPE,
                                                           stderr=asyncio.subprocess.PIPE)
            try:
                out, err = await asyncio.wait_for(process.communicate(), 240)
                return SimpleNamespace(return_code=process.returncode, stdout=out.decode(), stderr=err.decode())
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()

        class Environment:
            async def exec(self, command, **_):
                return await docker('exec', name, 'bash', '-lc', command)

            async def upload_file(self, source, target):
                result = await docker('cp', str(source), name + ':' + target)
                if result.return_code:
                    raise RuntimeError(result.stderr)

        try:
            started = await docker('run', '-d', '--name', name, '--memory', '1g',
                                   'python:3.13-slim-bookworm', 'tail', '-f', '/dev/null')
            self.assertEqual(started.return_code, 0, started.stderr)
            with tempfile.TemporaryDirectory() as directory:
                await bootstrap_grader(Environment(), 'regex-log', cache, Path(directory))
                isolated = await docker('network', 'disconnect', 'bridge', name)
                self.assertEqual(isolated.return_code, 0, isolated.stderr)
                # A failed upstream installer must not remove the seeded uv/env
                # or require another package fetch. Synthetic assertion only.
                result = await Environment().exec('''false # simulated failed official installer
source "$HOME/.local/bin/env"
printf 'def test_synthetic():\\n    assert 2 + 2 == 4\\n' > /tmp/test_synthetic.py
uvx -p 3.13 -w pytest==8.4.1 -w pytest-json-ctrf==0.3.5 pytest --ctrf /tmp/ctrf.json /tmp/test_synthetic.py -q
''')
                self.assertEqual(result.return_code, 0, result.stdout + result.stderr)
                self.assertIn('1 passed', result.stdout)
        finally:
            await docker('rm', '-f', name)


if __name__ == '__main__':
    unittest.main()
