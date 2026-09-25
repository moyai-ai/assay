"""Report official rewards without dropping failed or missing attempts."""
import json
from collections import Counter
from pathlib import Path


def summarize(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / 'manifest.json').read_text())
    tasks = manifest['tasks']
    attempts = manifest['attempts']
    seen = Counter()
    rows = []
    for path in sorted((run_dir / 'job').glob('*/result.json')):
        result = json.loads(path.read_text())
        task = result['task_name'].split('/')[-1]
        if task not in tasks:
            raise ValueError(f'Unexpected task result: {task}')
        seen[task] += 1
        if seen[task] > attempts:
            raise ValueError(f'Too many results for {task}; do not combine jobs or retries')
        reward = ((result.get('verifier_result') or {}).get('rewards') or {}).get('reward')
        exception = result.get('exception_info')
        valid_reward = type(reward) in (int, float) and reward in (0, 1)
        error = exception.get('exception_type', 'UnknownError') if exception else None
        if not result.get('finished_at'):
            status = 'pending'
        elif valid_reward:
            # Harbor can grade a completed solution even after an agent timeout.
            # Preserve the official reward AND the run warning; do not discard a pass.
            status = 'pass' if reward == 1 else 'fail'
        else:
            status, error = 'error', error or 'MissingOrInvalidReward'
        agent = result.get('agent_result') or {}
        rows.append({'task': task, 'trial': result['trial_name'], 'status': status,
                     'reward': reward, 'error': error,
                     'stop_reason': (agent.get('metadata') or {}).get('stop_reason'),
                     'input_tokens': agent.get('n_input_tokens'), 'output_tokens': agent.get('n_output_tokens'),
                     'result': str(path.relative_to(run_dir))})
    for task in tasks:
        for _ in range(attempts - seen[task]):
            rows.append({'task': task, 'trial': None, 'status': 'pending', 'reward': None,
                         'error': None, 'input_tokens': None, 'output_tokens': None, 'result': None})
    counts = Counter(row['status'] for row in rows)
    expected = len(tasks) * attempts
    graded = counts['pass'] + counts['fail']
    return {
        'scope': 'Assay coding-agent smoke test; no best-of-N selection',
        'model': manifest['model'], 'dataset_revision': manifest['dataset_revision'],
        'reasoning_effort': manifest.get('reasoning_effort', 'provider'),
        'tool_strict': manifest.get('tool_strict', True),
        'agent_timeout_multiplier': manifest.get('agent_timeout_multiplier', 1.0),
        'verifier_timeout_multiplier': manifest.get('verifier_timeout_multiplier', 1.0),
        'expected_attempts': expected, 'passed': counts['pass'], 'failed': counts['fail'],
        'errors': counts['error'], 'pending': counts['pending'],
        'graded_with_exceptions': sum(row['status'] in ('pass', 'fail') and row['error'] is not None for row in rows),
        'complete': counts['pending'] == 0,
        'budget_stops': sum(row.get('stop_reason') in ('turn_limit', 'time_limit') for row in rows),
        'success_fraction_of_planned': counts['pass'] / expected,
        'success_fraction_of_graded': counts['pass'] / graded if graded else None,
        'reported_input_tokens': sum(row['input_tokens'] or 0 for row in rows),
        'reported_output_tokens': sum(row['output_tokens'] or 0 for row in rows),
        'usage_note': 'Token totals may omit failed/interrupted runs. No USD cost estimated.',
        'trials': rows,
    }


def write_report(run_dir: Path) -> dict:
    if json.loads((run_dir / 'manifest.json').read_text()).get('kind') == 'verification-scaling':
        from benchmark.scaling_report import write_scaling_report
        return write_scaling_report(run_dir)
    report = summarize(run_dir)
    (run_dir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = [
        '# Assay Terminal-Bench 2.1 smoke test', '',
        f"Model: `{report['model']}`", '',
        f"Reasoning effort: `{report['reasoning_effort']}` | Server strict tool mode: `{report['tool_strict']}`", '',
        f"Timeout multipliers: agent `{report['agent_timeout_multiplier']}` | verifier `{report['verifier_timeout_multiplier']}`", '',
        f"Planned: {report['expected_attempts']} | Pass: {report['passed']} | Fail: {report['failed']} | "
        f"Errors: {report['errors']} | Pending: {report['pending']}", '',
        f"Success / planned attempts: {report['success_fraction_of_planned']:.1%}", '',
        f"Graded attempts with run warnings: {report['graded_with_exceptions']}. Official rewards remain authoritative; see the Error column.", '',
        f"Budget-stopped generations: {report['budget_stops']}; their official rewards are retained.", '',
        'A curated smoke subset, not a full-benchmark score or evidence of selection uplift.', '',
        '| Task | Trial | Status | Reward | Error |', '|---|---|---|---|---|',
    ]
    for row in report['trials']:
        lines.append(f"| {row['task']} | {row['trial'] or '—'} | {row['status']} | {row['reward']} | {row['error'] or '—'} |")
    lines.extend(['', report['usage_note'], ''])
    (run_dir / 'summary.md').write_text('\n'.join(lines))
    return report
