"""Independent generation -> frozen blind selection -> official grading.

Harbor AGENT_END hooks are OUTSIDE its agent timeout. The barrier therefore does
not charge selection/waiting time against a candidate's generation budget. Pools
pipeline under global budgets; whole-pool container reservations prevent deadlock
when completed candidates remain frozen while other candidates are queued.
"""
import argparse
import asyncio
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from harbor.models.task.config import TaskConfig
from harbor.models.task.verifier_mode import VerifierEnvironmentMode, resolve_task_verifier_mode
from harbor.models.trial.config import TrialConfig
from harbor.trial.hooks import TrialEvent
from harbor.trial.trial import Trial

from benchmark.cli import ROOT, BENCH, SUITE, CHECKOUT, check_dataset, source_fingerprint, positive_multiplier
from benchmark.evidence import DockerWorkspace, browser_health, candidate_evidence, digest, save_json
from benchmark.scaling_report import grader_health, model_usage, write_scaling_report
from benchmark.scheduling import Resources
from benchmark.dependencies import prepare_assets, bootstrap_grader, profile as dependency_profile


def redact(message) -> str:
    message = str(message)
    for name, value in os.environ.items():
        if name.endswith('API_KEY') and value:
            message = message.replace(value, '[REDACTED]')
    return message


def integer_list(value):
    try:
        numbers = sorted(set(int(n) for n in value.split(',')))
        if not numbers or any(n < 1 or n > 16 for n in numbers):
            raise ValueError()
        return numbers
    except ValueError:
        raise argparse.ArgumentTypeError('Expected comma-separated integers in [1,16]')


def experiment_arms(counts, repetitions, pivots, baseline_model=None):
    arms = [{'id': 'vanilla', 'kind': 'vanilla', 'n': 1, 'repetitions': 0, 'pivots': 0}]
    for n in counts:
        if n == 1:
            continue
        for repetitions_count in repetitions:
            for k in sorted({min(n, p) for p in pivots}):
                arms.append({'id': f'n{n}-r{repetitions_count}-k{k}', 'kind': 'selection', 'n': n,
                             'repetitions': repetitions_count, 'pivots': k})
    if baseline_model:
        arms.append({'id': 'external-vanilla', 'kind': 'external', 'n': 1, 'repetitions': 0, 'pivots': 0})
    return arms


def verifier_config(manifest, repetitions):
    return {**manifest['verifier'], 'repetitions': repetitions}


async def invoke_selector(payload: dict, directory: Path, timeout: float) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    # This file is the complete input boundary; it deliberately has no trial
    # paths, model identities per candidate, official tests, or reward fields.
    save_json(directory / 'input.json', payload, exclusive=True)
    process = await asyncio.create_subprocess_exec('node', '--import', 'tsx', str(BENCH / 'selection.ts'),
                                                   str(directory), cwd=ROOT, stdin=asyncio.subprocess.PIPE,
                                                   stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(process.communicate(json.dumps(payload).encode()), timeout)
        if process.returncode:
            raise RuntimeError('Blind selector failed: ' + redact(err.decode(errors='replace')[-4000:]))
        result = json.loads(out)
        save_json(directory / 'result.json', result, exclusive=True)
        return result
    finally:
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 10)
            except TimeoutError:
                process.kill()
                await process.wait()


def trial_config(root, task_path, candidate_id, manifest):
    role = 'baseline' if candidate_id == 'baseline' else 'generator'
    profile = manifest[role]
    host_logs = root / 'candidates' / candidate_id / 'generation'
    return TrialConfig.model_validate({
        'task': {'path': str(task_path)}, 'trials_dir': str(root / 'trials'),
        'trial_name': f'{root.parent.parent.name[:35]}-{root.name[:25]}-{candidate_id}',
        'agent_timeout_multiplier': manifest['agent_timeout_multiplier'],
        'verifier_timeout_multiplier': manifest['verifier_timeout_multiplier'],
        'agent': {'import_path': 'benchmark.agent:AssayAgent', 'model_name': profile['model'],
                  'kwargs': {**manifest['generation_limits'], 'reasoning_effort': profile['reasoning_effort'],
                             'api': profile['api'], 'credential_profile': role, 'capture_snapshot': False,
                             'host_logs_dir': str(host_logs.resolve())}},
        'environment': {'type': 'docker', 'delete': True,
                        'force_build': manifest['force_build'] or root.name in manifest.get('build_tasks', []),
                        'env': {'SE_CHROMEDRIVER': manifest['chromedriver_path']}
                            if root.name == 'filter-js-from-html' and manifest.get('chromedriver_path') else {}},

    })


async def run_task_pool(root: Path, task_path: Path, manifest: dict, *,
                        trial_factory=Trial.create, workspace_factory=DockerWorkspace.resolve,
                        selector=invoke_selector, resources=None) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    ids = [f'c{i:03}' for i in range(1, manifest['max_candidates'] + 1)]
    if manifest.get('baseline'):
        ids.append('baseline')
    release = asyncio.Event()
    ready = {key: asyncio.get_running_loop().create_future() for key in ids}
    resources = resources or Resources(manifest)
    grading_slots = resources.grading
    resources.phases[root.name] = 'generating'
    states, decisions = {}, {}
    locked = False
    started = time.time()
    active_started = time.monotonic()

    async def run_one(candidate_id):
        state = states[candidate_id] = {'id': candidate_id, 'stop_reason': None, 'error': None}
        lease, provider_lease, setup_lease, grading_lease, workspace, trial = False, False, False, False, None, None
        generation_slots = resources.baseline if candidate_id == 'baseline' else resources.generation
        candidate_dir = root / 'candidates' / candidate_id
        before = None

        def release_generation():
            nonlocal lease, provider_lease
            if provider_lease:
                resources.provider.release()
                provider_lease = False
            if lease:
                generation_slots.release()
                lease = False

        def release_setup():
            nonlocal setup_lease
            if setup_lease:
                resources.setup.release()
                setup_lease = False

        async def before_agent(event):
            nonlocal workspace, before, lease, provider_lease
            if manifest.get('prepare_grader'):
                receipt = await bootstrap_grader(trial.agent_environment, root.name,
                    BENCH / '.cache' / 'grading-tools-uv-0.9.5', candidate_dir)
                state['dependency_profile_sha256'] = receipt['profile_sha256']
            if root.name == 'filter-js-from-html':
                await browser_health(trial.agent_environment)
            workspace = await workspace_factory(trial.agent_environment, manifest['snapshot_byte_limit'])
            async with resources.snapshots.lease():
                try:
                    await workspace.pause()
                    before = await workspace.capture()
                finally:
                    await workspace.unpause()
            state['image_id'] = workspace.image_id
            state['image_fingerprint'] = getattr(workspace, 'image_fingerprint', workspace.image_id)
            state['base_sha'] = digest({'dataset': manifest['dataset_revision'], 'task': root.name,
                                        'image': state['image_fingerprint'], 'app': before['sha256'],
                                        'dependencies': state.get('dependency_profile_sha256')})
            save_json(candidate_dir / 'before.json', before, exclusive=True)
            release_setup()
            # AGENT_START runs before Harbor starts the generation clock.
            await generation_slots.acquire()
            lease = True
            if candidate_id != 'baseline':
                await resources.provider.acquire()
                provider_lease = True

        async def after_agent(event):
            nonlocal grading_lease
            try:
                if asyncio.current_task().cancelling():
                    raise asyncio.CancelledError()
                metadata = (event.result.agent_result.metadata or {}) if event.result.agent_result else {}
                stop = metadata.get('stop_reason')
                # wait_for cancels only the child agent on a Harbor deadline;
                # external cancellation cancels this trial task instead.
                if stop == 'interrupted':
                    stop = 'time_limit'
                state['stop_reason'] = stop
                if stop not in ('completed', 'turn_limit', 'time_limit'):
                    raise RuntimeError('Generation did not produce a budget-stopped or completed candidate')
                await workspace.pause()
                evidence = None
                try:
                    async with resources.snapshots.lease():
                        after = await workspace.capture()
                    save_json(candidate_dir / 'after.json', after, exclusive=True)
                    evidence = candidate_evidence(candidate_id, state['base_sha'], before, after,
                                                  candidate_dir / 'generation' / 'commands.jsonl')
                    save_json(candidate_dir / 'evidence.json', evidence, exclusive=True)
                    state['evidence_sha256'] = digest(evidence)
                    state['artifact_sha256'] = after['sha256']
                except Exception as error:
                    # Missing review evidence invalidates a tournament prefix,
                    # not the original implementation's ability to be graded.
                    state['evidence_error'] = redact(error)
                state['generation_finished_at'] = datetime.now(timezone.utc).isoformat()
                ready[candidate_id].set_result(evidence)
                release_generation()
                await release.wait()
                if not locked:
                    raise RuntimeError('Experiment cancelled before selections were locked')
                # Bound grading concurrency separately; queued candidates stay
                # paused, including live background services, until their turn.
                await grading_slots.acquire()
                grading_lease = True
            except BaseException as error:
                state['error'] = redact(error) or type(error).__name__
                raise
            finally:
                if not ready[candidate_id].done():
                    ready[candidate_id].set_result(None)
                release_generation()
                if workspace:
                    await workspace.unpause()

        async def verification_start(event):
            if not locked or not (root / 'decisions.json').is_file():
                raise RuntimeError('Official grading attempted before selection lock')
            state['grading_started_at'] = datetime.now(timezone.utc).isoformat()

        try:
            await resources.setup.acquire()
            setup_lease = True
            config = trial_config(root, task_path, candidate_id, manifest)
            save_json(candidate_dir / 'trial-config.json', config.model_dump(mode='json'), exclusive=True)
            trial = await trial_factory(config)
            trial.add_hook(TrialEvent.AGENT_START, before_agent)
            trial.add_hook(TrialEvent.AGENT_END, after_agent)
            trial.add_hook(TrialEvent.VERIFICATION_START, verification_start)
            result = await trial.run()
            state['trial_result'] = str(trial.paths.result_path.relative_to(root))
            state['trial_exception'] = result.exception_info.model_dump(mode='json') if result.exception_info else None
            state['official_reward'] = (result.verifier_result.rewards or {}).get('reward') if result.verifier_result else None
            if type(state['official_reward']) in (int, float) and state['official_reward'] in (0, 1):
                state['grader_health'] = grader_health(trial.paths.verifier_dir, state['official_reward'])
            else:
                state['error'] = state['error'] or 'No valid official reward'
            if result.agent_execution and result.agent_execution.finished_at:
                state['generation_seconds'] = (result.agent_execution.finished_at - result.agent_execution.started_at).total_seconds()
        except BaseException as error:
            state['error'] = redact(error) or type(error).__name__
            if isinstance(error, asyncio.CancelledError):
                raise
        finally:
            release_generation()
            release_setup()
            if grading_lease:
                grading_slots.release()
            if not ready[candidate_id].done():
                ready[candidate_id].set_result(None)
            # Hook cleanup normally unpauses before Harbor tears down. Handle a
            # failed/cancelled setup as well, without touching other containers.
            if workspace and workspace.paused:
                try:
                    await workspace.unpause()
                except Exception as error:
                    state['cleanup_error'] = redact(error)
            state['usage'] = model_usage(candidate_dir / 'generation')
            save_json(candidate_dir / 'status.json', state)

    jobs = [asyncio.create_task(run_one(key), name=key) for key in ids]
    try:
        inputs = dict(zip(ids, await asyncio.gather(*ready.values())))
        resources.phases[root.name] = 'selecting'

        async def choose(arm):
            decision = {'winner_id': None}
            selection_dir = root / 'selection' / arm['id']
            try:
                if arm['kind'] in ('vanilla', 'external'):
                    winner = 'baseline' if arm['kind'] == 'external' else 'c001'
                    if states[winner].get('stop_reason') not in ('completed', 'turn_limit', 'time_limit') or not states[winner].get('generation_finished_at'):
                        raise RuntimeError('Baseline generation unavailable')
                    if winner == 'baseline' and states['c001'].get('base_sha') and states[winner]['base_sha'] != states['c001']['base_sha']:
                        raise RuntimeError('External baseline initial environment differs from generator')
                    decision['winner_id'] = winner
                else:
                    candidates = [inputs[f'c{i:03}'] for i in range(1, arm['n'] + 1)]
                    if any(candidate is None for candidate in candidates):
                        raise RuntimeError('Candidate prefix is incomplete; no replacement sampling or fallback selection')
                    payload = {'mode': 'select', 'task': (task_path / 'instruction.md').read_text(),
                               'candidates': candidates, 'config': verifier_config(manifest, arm['repetitions']),
                               'options': {'pivots': arm['pivots'], 'seed': manifest['seed'],
                                           'maxEvidenceBytes': manifest['max_evidence_bytes'],
                                           'allowTruncation': manifest['allow_evidence_truncation'],
                                           'concurrency': manifest.get('pair_concurrency', 1)}}
                    # Each process reserves its full HTTP ceiling. This is a
                    # conservative, process-wide bound, not a per-task multiplier.
                    async with (resources.arms.lease(),
                                resources.judge.lease(manifest['verifier']['concurrency']),
                                resources.provider.lease(manifest['verifier']['concurrency'])):
                        result = await selector(payload, selection_dir, manifest['selection_timeout_sec'])
                    if not result.get('verificationComplete') or result.get('winnerId') not in {c['id'] for c in candidates}:
                        raise RuntimeError('Invalid or incomplete tournament selection')
                    decision.update(winner_id=result['winnerId'], selection=result,
                                    elapsed_seconds=result.get('elapsed_seconds'))
            except Exception as error:
                decision['error'] = redact(error)
            decision['usage'] = model_usage(selection_dir)
            decisions[arm['id']] = decision

        selections = [asyncio.create_task(choose(arm)) for arm in manifest['arms']]
        try:
            await asyncio.gather(*selections)
        finally:
            for selection in selections:
                if not selection.done():
                    selection.cancel()
            await asyncio.gather(*selections, return_exceptions=True)
        # Immutable decision record BEFORE releasing ANY official grader in this
        # pool. All experimental cells share exactly the same frozen candidates.
        save_json(root / 'decisions.json', {'locked_at': datetime.now(timezone.utc).isoformat(),
                  'candidate_evidence_sha256': {key: value.get('evidence_sha256') for key, value in states.items()},
                  'decisions': decisions}, exclusive=True)
        locked = True
        resources.phases[root.name] = 'grading'
        release.set()
        await asyncio.gather(*jobs)
    finally:
        for job in jobs:
            if not job.done():
                job.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
        record = {'decisions': decisions, 'candidates': states, 'selection_locked': locked,
                  'started_at': datetime.fromtimestamp(started, timezone.utc).isoformat(),
                  'finished_at': datetime.now(timezone.utc).isoformat(),
                  'wall_seconds': time.time() - started,
                  'active_clock_seconds': time.monotonic() - active_started}
        save_json(root / 'result.json', record)
        resources.phases[root.name] = 'finished' if locked else 'interrupted-or-error'
    return record


def validate_task_compatibility(task_checkout: Path, tasks: list[str]):
    for name in tasks:
        path = task_checkout / 'tasks' / name
        config = TaskConfig.model_validate_toml((path / 'task.toml').read_text())
        if config.steps or resolve_task_verifier_mode(config) != VerifierEnvironmentMode.SHARED:
            raise ValueError(f'{name}: scaling requires a single-step task with grading in the original shared environment')
        if config.environment.os != 'linux':
            raise ValueError(f'{name}: scaling requires a Linux Docker environment')
        if any((path / 'environment' / file).exists() for file in
               ('docker-compose.yaml', 'docker-compose.yml', 'compose.yaml', 'compose.yml')):
            raise ValueError(f'{name}: custom Compose/multi-service tasks are not supported by main-container freezing')


async def run_experiment(root, manifest, *, task_checkout=CHECKOUT, selector=invoke_selector,
                         pool_runner=run_task_pool):
    validate_task_compatibility(task_checkout, manifest['tasks'])
    if manifest.get('prepare_grader'):
        for task in manifest['tasks']:
            dependency_profile(task)
        await asyncio.to_thread(prepare_assets, BENCH / '.cache' / 'grading-tools-uv-0.9.5')
    if any(arm['kind'] == 'selection' for arm in manifest['arms']):
        # Fail before expensive generation if score-token logprobs are unavailable.
        probe = await selector({'mode': 'probe', 'config': verifier_config(manifest, 1)},
                               root / 'probe', manifest['selection_timeout_sec'])
        if not probe.get('capabilityPassed'):
            raise RuntimeError('Verifier capability probe failed')
    resources = Resources(manifest)
    pool_size = manifest['max_candidates'] + int(bool(manifest.get('baseline')))

    def progress():
        save_json(root / 'progress.json', {'updated_at': datetime.now(timezone.utc).isoformat(),
                                         **resources.snapshot()})

    async def heartbeat():
        while True:
            progress()
            await asyncio.sleep(15)

    async def run_pool(task):
        # Atomic reservation of the ENTIRE pool prevents multiple partially
        # frozen pools from consuming all capacity and deadlocking each other.
        async with resources.tasks.lease(), resources.containers.lease(pool_size):
            print(f'Generating independent pool: {task}', flush=True)
            await pool_runner(root / 'tasks' / task, task_checkout / 'tasks' / task, manifest,
                              selector=selector, resources=resources)
            report = write_scaling_report(root)
            progress()
            print(' | '.join(f"{a['id']}: {a['passed']} pass, {a['failed']} fail, {a['errors']} error, {a['pending']} pending"
                             for a in report['arms']), flush=True)

    monitor = asyncio.create_task(heartbeat())
    try:
        async with asyncio.TaskGroup() as group:
            for task in manifest['tasks']:
                group.create_task(run_pool(task), name=task)
    finally:
        monitor.cancel()
        await asyncio.gather(monitor, return_exceptions=True)
        progress()
    return write_scaling_report(root)


def prices_file(path):
    if path is None:
        return {}
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict) or set(value) - {'generator', 'verifier', 'baseline'}:
        raise ValueError('Price file roles must be generator, verifier, baseline')
    for rates in value.values():
        if not isinstance(rates, dict) or set(rates) != {'input', 'cached_input', 'output'}:
            raise ValueError('Each price role needs input, cached_input, output USD per million tokens')
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in rates.values()):
            raise ValueError('Prices must be finite and nonnegative')
    return value


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true', help='Enable Docker and paid model calls; otherwise plan only')
    p.add_argument('--report', type=Path, help='Refresh a scaling report, no model calls')
    p.add_argument('--name')
    p.add_argument('--task', action='append')
    p.add_argument('--candidate-counts', type=integer_list, default=[1, 2, 4])
    p.add_argument('--verifier-repetitions', type=integer_list, default=[2, 4])
    p.add_argument('--pivot-counts', type=integer_list, default=[2])
    p.add_argument('--concurrency', type=int, choices=range(1, 33), default=2,
                   help='GLOBAL DeepSeek generation slots (1..32), not slots per task')
    p.add_argument('--task-concurrency', type=int, choices=range(1, 33), default=1)
    p.add_argument('--baseline-concurrency', type=int, choices=range(1, 33), default=2)
    p.add_argument('--max-live-containers', type=int, choices=range(1, 129), default=32,
                   help='Whole-pool reservation ceiling, including paused containers')
    p.add_argument('--grading-concurrency', type=int, choices=range(1, 33), default=4)
    p.add_argument('--setup-concurrency', type=int, choices=range(1, 33), default=4)
    p.add_argument('--snapshot-concurrency', type=int, choices=range(1, 9), default=2)
    p.add_argument('--arm-concurrency', type=int, choices=range(1, 33), default=4)
    p.add_argument('--pair-concurrency', type=int, choices=range(1, 33), default=8)
    p.add_argument('--provider-concurrency', type=int, choices=range(1, 65), default=32,
                   help='Combined DeepSeek generation/judging ceiling; generation reserves one slot per active agent')
    p.add_argument('--judge-concurrency', type=int, choices=range(1, 65), default=32,
                   help='GLOBAL judge HTTP ceiling, shared conservatively across selector processes')
    p.add_argument('--prepare-grader', action='store_true',
                   help='Seed checksum-pinned uv and prewarm declared generic grading dependencies before generation')
    p.add_argument('--max-turns', type=int, choices=range(1, 201), default=60)
    p.add_argument('--max-output-tokens', type=int, default=32768)
    p.add_argument('--request-timeout-sec', type=int, default=900)
    p.add_argument('--agent-timeout-multiplier', type=positive_multiplier, default=2.0)
    p.add_argument('--verifier-timeout-multiplier', type=positive_multiplier, default=8.0)
    p.add_argument('--reasoning-effort', choices=['provider', 'none', 'low', 'medium', 'high'], default='none')
    p.add_argument('--generator-api', choices=['chat-completions', 'responses'], default='chat-completions')
    p.add_argument('--baseline-model', help='Optional OpenAI single-candidate baseline; requires OPENAI_API_KEY')
    p.add_argument('--baseline-api', choices=['chat-completions', 'responses'], default='responses')
    p.add_argument('--baseline-reasoning-effort', choices=['provider', 'none', 'low', 'medium', 'high'], default='provider')
    p.add_argument('--verifier-output-tokens', type=int, default=32768)
    p.add_argument('--verifier-reasoning-effort', choices=['provider', 'none', 'low', 'medium', 'high'],
                   help='Explicit scorer reasoning budget; overrides reasoning_effort in VERIFIER_EXTRA_BODY')
    p.add_argument('--verifier-concurrency', type=int, choices=range(1, 33), default=16,
                   help='Per-selector HTTP ceiling, reserved from --judge-concurrency')
    p.add_argument('--min-captured-mass', type=float, default=0.0)
    p.add_argument('--selection-timeout-sec', type=int, default=3600)
    p.add_argument('--max-evidence-bytes', type=int, default=48000)
    p.add_argument('--allow-evidence-truncation', action='store_true')
    p.add_argument('--snapshot-byte-limit', type=int, default=256 * 1024 * 1024)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--force-build', action='store_true', help='Build all pinned task Dockerfiles natively instead of prebuilt images')
    p.add_argument('--build-task', action='append', default=[], help='Build this task natively (repeatable); useful for HTML/Chromium on ARM')
    p.add_argument('--chromedriver-path', help='HTML-only SE_CHROMEDRIVER override, e.g. /usr/bin/chromedriver; no test changes')
    p.add_argument('--prices', type=Path, help='Optional JSON of user-supplied flat token prices')
    return p


def make_manifest(args):
    tasks = args.task or SUITE['tasks']
    if len(set(tasks)) != len(tasks) or not tasks or any(t not in SUITE['tasks'] for t in tasks):
        raise ValueError('Tasks must be distinct names from tasks.json')
    if not 256 <= args.max_output_tokens <= 32768 or not 15 <= args.request_timeout_sec <= 900:
        raise ValueError('Invalid generator limits')
    if not 256 <= args.verifier_output_tokens <= 32768 or not 0 <= args.min_captured_mass <= 1:
        raise ValueError('Invalid verifier limits')
    if not 512 <= args.max_evidence_bytes <= 1_000_000 or not 1_048_576 <= args.snapshot_byte_limit <= 1_073_741_824:
        raise ValueError('Invalid evidence limits')
    if not 15 <= args.selection_timeout_sec <= 86400:
        raise ValueError('Invalid selection timeout')
    if any(task not in tasks for task in args.build_task):
        raise ValueError('--build-task must belong to this experiment task list')
    if args.chromedriver_path and (not args.chromedriver_path.startswith('/') or '\0' in args.chromedriver_path):
        raise ValueError('ChromeDriver path must be an absolute container path')
    model = os.getenv('GENERATOR_MODEL') or os.getenv('VERIFIER_MODEL')
    if not model:
        raise ValueError('Set GENERATOR_MODEL or VERIFIER_MODEL')
    counts = sorted(set([1, *args.candidate_counts]))
    arms = experiment_arms(counts, args.verifier_repetitions, args.pivot_counts, args.baseline_model)
    uses_selector = any(a['kind'] == 'selection' for a in arms)
    if args.max_live_containers < max(counts) + int(bool(args.baseline_model)):
        raise ValueError('max-live-containers must fit one complete candidate pool including the baseline')
    if args.verifier_concurrency > min(args.judge_concurrency, args.provider_concurrency):
        raise ValueError('verifier-concurrency cannot exceed global judge/provider concurrency')
    if len(arms) > 128:
        raise ValueError('At most 128 experiment arms are supported')
    if args.prepare_grader:
        for task in tasks:
            dependency_profile(task)
    context = int(os.getenv('VERIFIER_CONTEXT_TOKENS', '0'))
    verifier_model = os.getenv('VERIFIER_MODEL')
    extra_body = json.loads(os.getenv('VERIFIER_EXTRA_BODY', '{}'))
    if not isinstance(extra_body, dict):
        raise ValueError('VERIFIER_EXTRA_BODY must be an object')
    if args.verifier_reasoning_effort == 'provider':
        extra_body.pop('reasoning_effort', None)
    elif args.verifier_reasoning_effort is not None:
        extra_body['reasoning_effort'] = args.verifier_reasoning_effort
    if uses_selector and (not verifier_model or context <= args.verifier_output_tokens + 2048):
        raise ValueError('Set VERIFIER_MODEL and an adequate VERIFIER_CONTEXT_TOKENS')
    return {
        'kind': 'verification-scaling', 'created_at': datetime.now(timezone.utc).isoformat(),
        'dataset': SUITE['dataset'], 'dataset_revision': SUITE['revision'], 'harbor_version': '0.23.0',
        'tasks': tasks, 'arms': arms, 'candidate_counts': counts, 'max_candidates': max(counts),
        'generator': {'model': model, 'api': args.generator_api, 'reasoning_effort': args.reasoning_effort},
        'baseline': {'model': args.baseline_model, 'api': args.baseline_api, 'reasoning_effort': args.baseline_reasoning_effort}
                    if args.baseline_model else None,
        'generation_limits': {'max_turns': args.max_turns, 'max_output_tokens': args.max_output_tokens,
                              'request_timeout_sec': args.request_timeout_sec},
        'verifier': {'model': verifier_model, 'contextWindowTokens': context, 'maxOutputTokens': args.verifier_output_tokens,
                     'requestTimeoutSec': args.request_timeout_sec, 'concurrency': args.verifier_concurrency,
                     'minCapturedMass': args.min_captured_mass, 'extraBody': extra_body},
        **{key: getattr(args, key) for key in ('concurrency', 'agent_timeout_multiplier', 'verifier_timeout_multiplier',
            'selection_timeout_sec', 'max_evidence_bytes', 'allow_evidence_truncation', 'snapshot_byte_limit', 'seed', 'force_build')},
        **{key: getattr(args, key) for key in ('task_concurrency', 'baseline_concurrency', 'max_live_containers',
            'grading_concurrency', 'setup_concurrency', 'snapshot_concurrency', 'arm_concurrency',
            'pair_concurrency', 'judge_concurrency', 'provider_concurrency', 'prepare_grader')},
        'scoring_protocol': 'assay-pairwise-xml-at20-v2',
        'dependency_profiles': {task: dependency_profile(task) for task in tasks} if args.prepare_grader else {},
        'build_tasks': sorted(set(args.build_task)), 'chromedriver_path': args.chromedriver_path,
        'prices': prices_file(args.prices), 'source_sha256': source_fingerprint(),
        'sampling': 'Fixed candidate slots, fresh containers/conversations, no replacement or success-conditioned sampling. '
                    'Nested prefixes share artifacts; vanilla is slot c001, never the best graded candidate.',
        'grading': 'All available pool candidates graded in their original environments only after every selection is locked. '
                   'Non-winner grades are post-selection oracle/random diagnostics only.',
    }


def main():
    os.umask(0o077)
    args = parser().parse_args()
    if args.report:
        print(json.dumps(write_scaling_report(args.report.resolve()), indent=2))
        return 0
    root = None
    try:
        check_dataset()
        load_dotenv(ROOT / '.env', override=False)
        manifest = make_manifest(args)
        validate_task_compatibility(CHECKOUT, manifest['tasks'])
        name = args.name or f"scaling-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:6]}"
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,100}', name):
            raise ValueError('Invalid run name')
        if args.execute:
            if not shutil.which('node') or not (ROOT / 'node_modules/tsx').is_dir():
                raise ValueError('Node and npm ci are required')
            generator_key = os.getenv('GENERATOR_API_KEY') or os.getenv('NEBIUS_API_KEY') or os.getenv('VERIFIER_API_KEY')
            if not generator_key or generator_key == 'replace-me' or not (os.getenv('GENERATOR_BASE_URL') or os.getenv('VERIFIER_BASE_URL')):
                raise ValueError('Generator endpoint/key missing')
            roles = (['VERIFIER'] if any(a['kind'] == 'selection' for a in manifest['arms']) else [])
            if manifest['baseline'] and os.getenv('OPENAI_API_KEY', '') in ('', 'replace-me'):
                raise ValueError('Set OPENAI_API_KEY for the baseline (OpenAI URL is the default)')
            for role in roles:
                if not os.getenv(role + '_BASE_URL') or os.getenv(role + '_API_KEY', '') in ('', 'replace-me'):
                    raise ValueError(f'Explicit {role}_BASE_URL and {role}_API_KEY required')
            subprocess.run(['docker', 'info'], stdout=subprocess.DEVNULL, check=True)
            subprocess.run(['docker', 'compose', 'version'], stdout=subprocess.DEVNULL, check=True)
        root = BENCH / 'runs' / name
        root.mkdir(parents=True, exist_ok=False, mode=0o700)
        # Never persist secrets, including accidentally supplied extra-body values.
        serialized = json.dumps(manifest)
        if redact(serialized) != serialized:
            raise ValueError('Refusing to write a manifest containing a configured API key')
        save_json(root / 'manifest.json', manifest, exclusive=True)
        write_scaling_report(root)
        n = len(manifest['tasks']) * (manifest['max_candidates'] + int(manifest['baseline'] is not None))
        print(f'Experiment: {root}\nPlan: {n} independent generations, up to {n} official gradings; '
              f"{len(manifest['arms'])} arms across {len(manifest['tasks'])} tasks.", flush=True)
        print('Arms: ' + ', '.join(a['id'] for a in manifest['arms']), flush=True)
        if not args.execute:
            print('Plan only. Use --execute with a NEW run name for paid generation and verification.')
            return 0
        async def execute():
            task = asyncio.current_task()
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, task.cancel)
            try:
                return await run_experiment(root, manifest)
            finally:
                for sig in (signal.SIGTERM, signal.SIGINT):
                    loop.remove_signal_handler(sig)
        report = asyncio.run(execute())
        return 2 if any(a['errors'] or a['pending'] for a in report['arms']) else 0
    except (KeyboardInterrupt, asyncio.CancelledError):
        if root and (root / 'manifest.json').exists():
            save_json(root / 'interrupted.json', {'interrupted_at': datetime.now(timezone.utc).isoformat()})
        print('Interrupted; trial cleanup completed. No partial selection was used.', file=sys.stderr)
        return 130
    except Exception as error:
        print(redact(error), file=sys.stderr)
        if root and (root / 'manifest.json').exists():
            save_json(root / 'error.json', {'error': redact(error)})
        return 1
    finally:
        if root and (root / 'manifest.json').exists():
            write_scaling_report(root)


if __name__ == '__main__':
    raise SystemExit(main())
