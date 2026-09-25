"""Run from the repository root: uv run --project benchmark python -m benchmark.cli."""
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from harbor.models.job.config import JobConfig

from benchmark.report import write_report

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / 'benchmark'
SUITE = json.loads((BENCH / 'tasks.json').read_text())
CHECKOUT = BENCH / '.cache' / 'terminal-bench-2-1'


def git(*args: str) -> str:
    return subprocess.check_output(['git', '-C', str(CHECKOUT), *args], text=True).strip()


def check_dataset() -> None:
    if not CHECKOUT.is_dir():
        raise ValueError('Run the prepare command first')
    if git('rev-parse', 'HEAD') != SUITE['revision']:
        raise ValueError('Dataset revision differs from tasks.json; refusing to run')
    if git('status', '--porcelain', '--untracked-files=all'):
        raise ValueError('Dataset checkout has modifications; refusing to run')
    for task in SUITE['tasks']:
        if not (CHECKOUT / 'tasks' / task / 'instruction.md').is_file():
            raise ValueError(f'Missing pinned task: {task}')


def prepare() -> None:
    if not CHECKOUT.exists():
        CHECKOUT.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'init', str(CHECKOUT)], check=True)
        subprocess.run(['git', '-C', str(CHECKOUT), 'remote', 'add', 'origin', SUITE['repository']], check=True)
        subprocess.run(['git', '-C', str(CHECKOUT), 'fetch', '--depth', '1', 'origin', SUITE['revision']], check=True)
        subprocess.run(['git', '-C', str(CHECKOUT), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
    check_dataset()
    print(f"Prepared {len(SUITE['tasks'])} tasks at {SUITE['revision']}. No images pulled or model calls made.")


def positive_multiplier(value: str) -> float:
    multiplier = float(value)
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise argparse.ArgumentTypeError('Timeout multiplier must be finite and greater than zero')
    return multiplier


def build_config(run_dir: Path, tasks: list[str], model: str, attempts: int, concurrency: int,
                 max_turns: int, max_output_tokens: int, request_timeout_sec: int = 300,
                 reasoning_effort: str = 'none', agent_timeout_multiplier: float = 1.0,
                 verifier_timeout_multiplier: float = 1.0) -> dict:
    # Credentials must never be stored in JobConfig or passed as agent-env,
    # verifier-env, or environment.env. The worker inherits them on the host.
    config = JobConfig.model_validate({
        'job_name': 'job', 'jobs_dir': str(run_dir), 'n_attempts': attempts,
        'n_concurrent_trials': concurrency, 'retry': {'max_retries': 0},
        'agent_timeout_multiplier': agent_timeout_multiplier,
        'verifier_timeout_multiplier': verifier_timeout_multiplier,
        'environment': {'type': 'docker', 'delete': True, 'force_build': False},
        'agents': [{'import_path': 'benchmark.agent:AssayAgent', 'model_name': model,
                    'kwargs': {'max_turns': max_turns, 'max_output_tokens': max_output_tokens,
                               'request_timeout_sec': request_timeout_sec, 'reasoning_effort': reasoning_effort}}],
        'tasks': [{'path': str(CHECKOUT / 'tasks' / name), 'source': SUITE['dataset']} for name in tasks],
    })
    return config.model_dump(mode='json')


def source_fingerprint() -> dict:
    paths = sorted([*ROOT.glob('src/**/*.ts'), *BENCH.glob('*.ts'), *BENCH.glob('*.py'),
                    ROOT / 'package.json', ROOT / 'package-lock.json', ROOT / 'tsconfig.json',
                    ROOT / 'tsconfig.build.json', BENCH / 'pyproject.toml',
                    BENCH / 'uv.lock', BENCH / 'tasks.json'])
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def run(args) -> int:
    check_dataset()
    load_dotenv(ROOT / '.env', override=False)
    model = os.getenv('GENERATOR_MODEL') or os.getenv('VERIFIER_MODEL')
    if not model:
        raise ValueError('Set GENERATOR_MODEL or VERIFIER_MODEL in .env')
    tasks = args.task or SUITE['tasks']
    if len(set(tasks)) != len(tasks) or any(task not in SUITE['tasks'] for task in tasks):
        raise ValueError('--task must name distinct tasks from benchmark/tasks.json')
    if not 1 <= args.attempts <= 10 or not 1 <= args.concurrency <= 4:
        raise ValueError('attempts must be 1..10 and concurrency 1..4')
    if not 1 <= args.max_turns <= 200 or not 256 <= args.max_output_tokens <= 32768:
        raise ValueError('Invalid generation limits')
    if not 15 <= args.request_timeout_sec <= 900:
        raise ValueError('request-timeout-sec must be 15..900')
    name = args.name or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:6]}"
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,100}', name):
        raise ValueError('Invalid run name')
    if args.execute:
        key = os.getenv('GENERATOR_API_KEY') or os.getenv('NEBIUS_API_KEY') or os.getenv('VERIFIER_API_KEY')
        if not key or key == 'replace-me':
            raise ValueError('Missing generator API key')
        if not (os.getenv('GENERATOR_BASE_URL') or os.getenv('VERIFIER_BASE_URL')):
            raise ValueError('Missing generator base URL')
        if not shutil.which('node') or not (ROOT / 'node_modules/tsx').is_dir():
            raise ValueError('Node 22+ and npm ci are required')
        subprocess.run(['docker', 'info'], check=True, stdout=subprocess.DEVNULL)
        subprocess.run(['docker', 'compose', 'version'], check=True, stdout=subprocess.DEVNULL)
    run_dir = BENCH / 'runs' / name
    run_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    config = build_config(run_dir, tasks, model, args.attempts, args.concurrency,
                          args.max_turns, args.max_output_tokens, args.request_timeout_sec, args.reasoning_effort,
                          agent_timeout_multiplier=args.agent_timeout_multiplier,
                          verifier_timeout_multiplier=args.verifier_timeout_multiplier)
    config_path = run_dir / 'job-config.json'
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    manifest = {
        'dataset': SUITE['dataset'], 'dataset_revision': SUITE['revision'], 'harbor_version': '0.23.0',
        'model': model, 'tasks': tasks, 'attempts': args.attempts,
        'max_turns': args.max_turns, 'max_output_tokens': args.max_output_tokens,
        'request_timeout_sec': args.request_timeout_sec, 'reasoning_effort': args.reasoning_effort,
        'tool_strict': False,
        'http_transport': {'client': 'undici', 'headers_timeout_sec': args.request_timeout_sec,
                           'body_timeout_sec': args.request_timeout_sec},
        'agent_timeout_multiplier': args.agent_timeout_multiplier,
        'verifier_timeout_multiplier': args.verifier_timeout_multiplier,
        'concurrency': args.concurrency, 'source_sha256': source_fingerprint(),
        'scope': 'Generation only; no Assay tournament. Official Harbor grading.',
    }
    (run_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f"Run directory: {run_dir}\nPlan: {len(tasks)} tasks × {args.attempts} attempts, model={model}", flush=True)
    if not args.execute:
        print('Plan only. Add --execute to a new run to pull images and make paid model requests.')
        return 0
    # Invoke the Harbor executable from this exact uv environment.
    harbor = Path(sys.executable).parent / 'harbor'
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    returncode = 1
    try:
        returncode = subprocess.run([str(harbor), 'run', '-c', str(config_path)], cwd=ROOT, env=env).returncode
    finally:
        report = write_report(run_dir)
        print(f"Report: {run_dir / 'summary.md'}", flush=True)
    # Graded task failures are valid experiment outcomes. Harness errors and
    # incomplete jobs must not masquerade as successful smoke-test execution.
    return returncode or (2 if report['errors'] or report['pending'] or report['graded_with_exceptions'] else 0)


def main() -> int:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('prepare', help='Fetch and verify the pinned task repository; no paid calls')
    execute = commands.add_parser('run', help='Plan a run; --execute enables paid generation and Docker')
    execute.add_argument('--execute', action='store_true')
    execute.add_argument('--task', action='append', help='One of the fixed tasks; repeat for a subset')
    execute.add_argument('--name')
    execute.add_argument('--attempts', type=int, default=1)
    execute.add_argument('--concurrency', type=int, default=1)
    execute.add_argument('--max-turns', type=int, default=60)
    execute.add_argument('--max-output-tokens', type=int, default=32768)
    execute.add_argument('--request-timeout-sec', type=int, default=300)
    execute.add_argument('--agent-timeout-multiplier', type=positive_multiplier, default=1.0,
                         help='Scale upstream agent execution deadlines (default: 1); does not scale API request timeouts')
    execute.add_argument('--verifier-timeout-multiplier', type=positive_multiplier, default=1.0,
                         help='Scale upstream official grading deadlines (default: 1)')
    execute.add_argument('--reasoning-effort', choices=['provider', 'none', 'low', 'medium', 'high'], default='none')
    report = commands.add_parser('report', help='Summarize a run without any model calls')
    report.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            prepare()
            return 0
        if args.command == 'report':
            print(json.dumps(write_report(args.run_dir.resolve()), indent=2))
            return 0
        return run(args)
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
