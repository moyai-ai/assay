"""Pinned, generic grading tools only. No grader tests/solutions enter preparation.

Each original sandbox gets its own tools/cache before generation. Wheels are
hash-checked data on the host; only selected binary members are copied, never
executed on the host. No shared writable cache is mounted into candidates.
"""
import hashlib
import os
import shlex
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from uuid import uuid4

from benchmark.evidence import digest, save_json

# PyPI uv 0.9.5 release metadata, retrieved 2026-09-25. The official suite's
# installer also requests 0.9.5. Keep upstream checksums in source, not a TOFU cache.
UV_ASSETS = {
    'x86_64': {
        'url': 'https://files.pythonhosted.org/packages/9b/83/a0bdf4abf86ede79b427778fe27e2b4a022c98a7a8ea1745dcd6c6561f17/uv-0.9.5-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl',
        'sha256': '6507bbbcd788553ec4ad5a96fa19364dc0f58b023e31d79868773559a83ec181',
    },
    'aarch64': {
        'url': 'https://files.pythonhosted.org/packages/48/8a/a990d9a39094d4d47bd11edff17573247f3791c33a19626e92c995498e68/uv-0.9.5-py3-none-manylinux_2_17_aarch64.manylinux2014_aarch64.musllinux_1_1_aarch64.whl',
        'sha256': '48a3542835d37882ff57d1ff91b757085525d98756712fa61cf9941d3dda8ebf',
    },
}
UV_TASKS = {'cancel-async-tasks', 'filter-js-from-html', 'log-summary-date-ranges',
            'multi-source-data-merger', 'polyglot-c-py', 'regex-log',
            'schemelike-metacircular-eval', 'write-compressor'}


def profile(task):
    if task in UV_TASKS:
        packages = ['pytest==8.4.1', 'pytest-json-ctrf==0.3.5']
        if task == 'filter-js-from-html':
            packages += ['selenium==4.38.0', 'bs4==0.0.2']
        if task == 'multi-source-data-merger':
            packages += ['pandas==2.3.3', 'pyarrow==22.0.0']
        return {'installer': 'uvx', 'python': '3.13', 'packages': packages}
    if task == 'build-cython-ext':
        return {'installer': 'pip', 'packages': ['pytest==8.4.1', 'pytest-json-ctrf==0.3.5']}
    if task == 'kv-store-grpc':
        return {'installer': 'pip', 'packages': ['pytest==8.4.2', 'requests==2.32.5', 'psutil==7.0.0', 'pytest-json-ctrf==0.3.5']}
    raise ValueError(f'No declared grading dependency profile for {task}')


def prepare_assets(cache: Path):
    """Fetch at most two public wheels; detect corrupt caches rather than trusting them."""
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    for arch, asset in UV_ASSETS.items():
        wheel = cache / (arch + '.whl')
        if not wheel.exists():
            request = urllib.request.Request(asset['url'], headers={'User-Agent': 'assay-dependency-preflight'})
            with urllib.request.urlopen(request, timeout=120) as response, tempfile.NamedTemporaryFile(dir=cache, delete=False) as tmp:
                path = Path(tmp.name)
                try:
                    size = 0
                    while chunk := response.read(1024 * 1024):
                        size += len(chunk)
                        if size > 32 * 1024 * 1024:
                            raise ValueError('UV wheel exceeds download limit')
                        tmp.write(chunk)
                    tmp.flush()
                    if hashlib.sha256(path.read_bytes()).hexdigest() != asset['sha256']:
                        raise ValueError('UV wheel checksum mismatch')
                    os.replace(path, wheel)
                finally:
                    path.unlink(missing_ok=True)
        if (wheel.is_symlink() or not wheel.is_file() or wheel.stat().st_size > 32 * 1024 * 1024
                or hashlib.sha256(wheel.read_bytes()).hexdigest() != asset['sha256']):
            raise ValueError('UV cached wheel checksum mismatch')
        target = cache / arch
        target.mkdir(exist_ok=True, mode=0o700)
        with zipfile.ZipFile(wheel) as archive:
            for executable in ('uv', 'uvx'):
                name = f'uv-0.9.5.data/scripts/{executable}'
                matches = [entry for entry in archive.infolist() if entry.filename == name]
                if len(matches) != 1 or not 0 < matches[0].file_size <= 80 * 1024 * 1024:
                    raise ValueError('Invalid UV binary member')
                # Read named members only; no extraction of arbitrary paths or links.
                content = archive.read(matches[0])
                destination = target / executable
                if destination.is_symlink():
                    raise ValueError('Refusing symlink in dependency cache')
                with tempfile.NamedTemporaryFile(dir=target, delete=False) as output:
                    staging = Path(output.name)
                    try:
                        output.write(content)
                        output.flush()
                        staging.chmod(0o600)  # not host-executable
                        os.replace(staging, destination)
                    finally:
                        staging.unlink(missing_ok=True)
    return cache


async def bootstrap_grader(environment, task, assets: Path, logs: Path):
    config = profile(task)
    result = await environment.exec('uname -m', timeout_sec=15)
    arch = (result.stdout or '').strip()
    if result.return_code or arch not in UV_ASSETS:
        raise RuntimeError('Unsupported grading-tool architecture')
    temporary = '/tmp/assay-tools-' + uuid4().hex
    commands = ['set -eu']
    if config['installer'] == 'uvx':
        for executable in ('uv', 'uvx'):
            await environment.upload_file(assets / arch / executable, temporary + '-' + executable)
        commands += [
            'mkdir -p "$HOME/.local/bin"',
            f'install -m 755 {temporary}-uv "$HOME/.local/bin/uv"',
            f'install -m 755 {temporary}-uvx "$HOME/.local/bin/uvx"',
            'printf \'export PATH="$HOME/.local/bin:$PATH"\\n\' > "$HOME/.local/bin/env"',
            '. "$HOME/.local/bin/env"',
            f'rm -f {temporary}-uv {temporary}-uvx',
            'uv --version | grep -E "^uv 0[.]9[.]5( |$)"',
            'export UV_HTTP_RETRIES=3 UV_HTTP_TIMEOUT=120',
        ]
        flags = ' '.join('--with ' + shlex.quote(p) for p in config['packages'])
        commands += [f'uvx -p 3.13 {flags} pytest --version']
    else:
        commands += ['python3 -m pip install --retries 3 --timeout 120 ' + shlex.join(config['packages']),
                     'python3 -m pytest --version']
    # No test collection/execution: only pinned generic packages and a version check.
    result = await environment.exec('\n'.join(commands), timeout_sec=900)
    receipt = {'profile': config, 'architecture': arch,
               'uv_wheel_sha256': UV_ASSETS[arch]['sha256'] if config['installer'] == 'uvx' else None,
               'return_code': result.return_code, 'stdout_tail': (result.stdout or '')[-16000:],
               'stderr_tail': (result.stderr or '')[-16000:]}
    receipt['profile_sha256'] = digest({key: receipt[key] for key in ('profile', 'architecture', 'uv_wheel_sha256')})
    save_json(logs / 'grading-dependencies.json', receipt, exclusive=True)
    if result.return_code:
        raise RuntimeError('Pinned grading dependency preflight failed; see grading-dependencies.json')
    return receipt
