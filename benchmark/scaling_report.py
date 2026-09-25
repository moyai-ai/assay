"""Post-selection scoring only. Never imported by the LLM selector."""
import json
import math
import os
import stat
from collections import Counter
from pathlib import Path

from benchmark.evidence import save_json


def read_log(path: Path, limit=16 * 1024 * 1024) -> str:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError('Grader log is not a regular file')
        data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError('Grader log exceeds inspection limit')
        return data.decode('utf8', errors='replace')


def grader_health(directory: Path, reward) -> dict:
    issues = []
    try:
        stdout = read_log(directory / 'test-stdout.txt')
        summary = json.loads(read_log(directory / 'ctrf.json'))['results']['summary']
        total, passed, failed = (summary.get(k) for k in ('tests', 'passed', 'failed'))
        if any(type(n) is not int or n < 0 for n in (total, passed, failed)) or total <= 0 or passed + failed != total:
            issues.append('Missing, skipped, or unexecuted tests')
        if reward == 1 and (passed != total or failed != 0):
            issues.append('Reward contradicts test counts')
        if reward == 0 and failed == 0:
            issues.append('Zero reward without a recorded test failure')
        for marker in ('Failed to create driver or process file', 'Unable to obtain driver for chrome',
                       'NoSuchDriverException', 'no tests ran', 'ERROR collecting'):
            if marker in stdout:
                issues.append(marker)
    except (OSError, ValueError, KeyError, TypeError) as error:
        issues.append(f'Unverifiable grader execution: {type(error).__name__}')
    return {'status': 'suspect' if issues else 'ok', 'issues': issues,
            'note': 'Execution checks, not a proof that the official tests cover every requirement.'}


def model_usage(directory: Path) -> dict:
    result = {'requests': 0, 'responses': 0, 'input_tokens': 0, 'cached_input_tokens': 0,
              'output_tokens': 0, 'http_seconds_sum': 0.0, 'complete': True}
    requests = list(directory.glob('**/model/*-request.json'))
    result['requests'] = len(requests)
    for path in directory.glob('**/model/*-response.json'):
        try:
            response = json.loads(path.read_text())
            result['responses'] += 1
            if not isinstance(response, dict) or not isinstance(response.get('body'), dict):
                raise ValueError('Malformed model response record')
            body = response['body']
            usage = body.get('usage') or {}
            if not isinstance(usage, dict):
                raise ValueError('Malformed token usage')
            inp = usage.get('prompt_tokens', usage.get('input_tokens'))
            out = usage.get('completion_tokens', usage.get('output_tokens'))
            details = usage.get('prompt_tokens_details') or usage.get('input_tokens_details') or {}
            if not isinstance(details, dict):
                raise ValueError('Malformed token details')
            cached = details.get('cached_tokens', 0)
            result['http_seconds_sum'] += response.get('elapsedMs', 0) / 1000
            if response['status'] != 200 or any(type(n) is not int or n < 0 for n in (inp, out, cached)) or cached > inp:
                result['complete'] = False
                continue
            result['input_tokens'] += inp
            result['output_tokens'] += out
            result['cached_input_tokens'] += cached
        except (ValueError, TypeError, KeyError, OSError):
            result['complete'] = False
    result['complete'] &= result['responses'] == result['requests'] and not any(directory.glob('**/model/*-error.json'))
    return result


def combine_usage(usages) -> dict:
    usages = list(usages)
    keys = ('requests', 'responses', 'input_tokens', 'cached_input_tokens', 'output_tokens', 'http_seconds_sum')
    return {**{key: sum(u.get(key, 0) for u in usages) for key in keys},
            'complete': all(u.get('complete', False) for u in usages)}


def estimated_cost(usage: dict, rates: dict | None):
    if rates is None or not usage['complete']:
        return None
    return ((usage['input_tokens'] - usage['cached_input_tokens']) * rates['input']
            + usage['cached_input_tokens'] * rates['cached_input'] + usage['output_tokens'] * rates['output']) / 1_000_000


def paired_comparison(arm: dict, reference: dict) -> dict:
    pairs = [(a['reward'], b['reward']) for a, b in zip(arm['observations'], reference['observations'])
             if a['reward'] is not None and b['reward'] is not None]
    wins = sum(a > b for a, b in pairs)
    losses = sum(a < b for a, b in pairs)
    complete = arm['pass_rate'] is not None and reference['pass_rate'] is not None
    discordant = wins + losses
    # Exact two-sided paired sign/McNemar test. No test on a successful-only
    # subset, and failure to detect a difference is NOT evidence of equivalence.
    p_value = None if not complete else (1.0 if not discordant else
        min(1.0, 2 * sum(math.comb(discordant, i) for i in range(min(wins, losses) + 1)) / 2 ** discordant))
    return {'reference': reference['id'], 'paired_graded_tasks': len(pairs),
            'wins': wins, 'losses': losses, 'ties': len(pairs) - discordant,
            'pass_rate_difference_pp': (arm['pass_rate'] - reference['pass_rate']) * 100 if complete else None,
            'exact_paired_two_sided_p': p_value, 'equivalence_test_performed': False}


def write_scaling_report(root: Path) -> dict:
    manifest = json.loads((root / 'manifest.json').read_text())
    tasks = manifest['tasks']
    records = {}
    for task in tasks:
        path = root / 'tasks' / task / 'result.json'
        if path.exists():
            records[task] = json.loads(path.read_text())
    arms = []
    for arm in manifest['arms']:
        rows = []
        for task in tasks:
            record = records.get(task)
            row = {'task': task, 'status': 'pending', 'official_reward': None, 'reward': None,
                   'winner_id': None, 'error': None}
            if record is None:
                rows.append(row)
                continue
            decision = record['decisions'].get(arm['id'], {})
            winner = decision.get('winner_id')
            candidates = record['candidates']
            chosen = candidates.get(winner, {})
            reward = chosen.get('official_reward')
            valid_reward = type(reward) in (int, float) and reward in (0, 1)
            error = decision.get('error') or chosen.get('error')
            healthy = chosen.get('grader_health', {}).get('status') == 'ok'
            trusted = valid_reward and healthy and not decision.get('error')
            row.update(winner_id=winner, official_reward=reward, reward=reward if trusted else None,
                       status=('pass' if reward == 1 else 'fail') if trusted else 'error',
                       error=None if trusted else error or 'Missing grade or suspect grader',
                       generation_stop=chosen.get('stop_reason'), grader_health=chosen.get('grader_health'),
                       trial_exception=chosen.get('trial_exception'), selection=decision.get('selection'),
                       selection_seconds=decision.get('elapsed_seconds'))
            ids = ['baseline'] if arm['kind'] == 'external' else [f'c{i:03}' for i in range(1, arm['n'] + 1)]
            usage = combine_usage(candidates.get(i, {}).get('usage', {'complete': False}) for i in ids)
            verification = decision.get('usage', combine_usage([]))
            row['generation_usage'], row['verification_usage'] = usage, verification
            rates = manifest.get('prices', {})
            gen_cost = estimated_cost(usage, rates.get('baseline' if arm['kind'] == 'external' else 'generator'))
            ver_cost = 0 if verification['requests'] == 0 else estimated_cost(verification, rates.get('verifier'))
            row['estimated_cost_usd'] = None if gen_cost is None or ver_cost is None else gen_cost + ver_cost
            pool = [candidates.get(i, {}) for i in ids]
            if all(type(c.get('official_reward')) in (int, float) and c['official_reward'] in (0, 1)
                   and c.get('grader_health', {}).get('status') == 'ok' for c in pool):
                row['oracle_any_pass'] = int(any(c['official_reward'] == 1 for c in pool))
                row['random_selection_expected_reward'] = sum(c['official_reward'] for c in pool) / len(pool)
            else:
                row['oracle_any_pass'] = row['random_selection_expected_reward'] = None
            rows.append(row)
        counts = Counter(row['status'] for row in rows)
        valid = counts['pass'] + counts['fail']
        full = valid == len(tasks)
        # No conditional-on-graded headline: a complete quality score requires
        # one trustworthy selected reward for EVERY planned task.
        arms.append({**arm, 'passed': counts['pass'], 'failed': counts['fail'], 'errors': counts['error'],
                     'pending': counts['pending'], 'pass_rate': counts['pass'] / len(tasks) if full else None,
                     'confirmed_pass_fraction_of_planned': counts['pass'] / len(tasks),
                     'generation_usage': combine_usage(r.get('generation_usage', {'complete': False}) for r in rows),
                     'verification_usage': combine_usage(r.get('verification_usage', {'complete': False}) for r in rows),
                     'estimated_cost_usd': sum(r['estimated_cost_usd'] for r in rows)
                         if all(r.get('estimated_cost_usd') is not None for r in rows) else None,
                     'observations': rows})
    baseline = next(arm for arm in arms if arm['kind'] == 'vanilla')
    external = next((arm for arm in arms if arm['kind'] == 'external'), None)
    for arm in arms:
        comparison = paired_comparison(arm, baseline)
        arm['comparisons'] = {'vanilla': comparison}
        arm['uplift_vs_vanilla_pp'] = comparison['pass_rate_difference_pp']
        arm['paired_wins_vs_vanilla'] = comparison['wins']
        arm['paired_losses_vs_vanilla'] = comparison['losses']
        arm['paired_graded_tasks'] = comparison['paired_graded_tasks']
        if external is not None:
            arm['comparisons']['external'] = paired_comparison(arm, external)
        arm['gap_vs_external_pp'] = arm['comparisons'].get('external', {}).get('pass_rate_difference_pp')
    report = {'scope': 'Frozen independent candidate pools; blind pivot selection; official grading after selection lock',
              'tasks': len(tasks), 'finished_tasks': len(records), 'complete': len(records) == len(tasks),
              'models': {'generator': manifest.get('generator', {}).get('model'),
                         'verifier': manifest.get('verifier', {}).get('model'),
                         'external_baseline': (manifest.get('baseline') or {}).get('model')},
              'dataset_revision': manifest.get('dataset_revision'),
              'claims': {'generator_uplift': 'Compare selection arms with the same-generator vanilla baseline.',
                         'external_parity': 'Not evaluated without a configured external baseline. Observed score gaps '
                                            'and non-significant difference tests do not establish equivalence or general SOTA.'},
              'arms': arms, 'actual_model_usage_all_experiment_cells': model_usage(root),
              'cost_note': 'Optional user-supplied flat USD/million-token estimates, not provider bills. '
                           'Per-arm costs include its candidate prefix and its selection; shared pools are not paid anew. '
                           'Probe overhead, grading, infrastructure, and provider pricing tiers are not in per-arm costs.',
              'limitations': 'Small fixed subset; paired comparisons are exploratory and p-values are not adjusted for multiple arms. '
                              'Oracle/random diagnostics use post-lock labels and never drive selection.'}
    save_json(root / 'summary.json', report)
    lines = ['# Assay verification scaling', '', report['scope'], '',
             f"Generator: `{report['models']['generator']}`. Verifier: `{report['models']['verifier']}`. "
             f"External baseline: `{report['models']['external_baseline'] or 'not configured'}`.", '',
             '| Arm | Pass | Fail | Error | Pending | Pass rate | Uplift vs vanilla (pp) | Gap vs external (pp) |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in arms:
        rate = 'incomplete' if arm['pass_rate'] is None else f"{arm['pass_rate']:.1%}"
        delta = '—' if arm['uplift_vs_vanilla_pp'] is None else f"{arm['uplift_vs_vanilla_pp']:+.1f}"
        gap = '—' if arm['gap_vs_external_pp'] is None else f"{arm['gap_vs_external_pp']:+.1f}"
        lines.append(f"| {arm['id']} | {arm['passed']} | {arm['failed']} | {arm['errors']} | {arm['pending']} | {rate} | {delta} | {gap} |")
    lines += ['', '## Selected outcomes', '', '| Task | Arm | Winner | Official reward | Status | Stop / error |', '|---|---|---|---|---|---|']
    for arm in arms:
        for row in arm['observations']:
            note = str(row.get('error') or row.get('generation_stop') or '').replace('|', '/').replace('\n', ' ')[:180]
            lines.append(f"| {row['task']} | {arm['id']} | {row['winner_id'] or '—'} | {row['official_reward']} | {row['status']} | {note} |")
    lines += ['', report['cost_note'], '', report['limitations'], '', report['claims']['external_parity'], '',
              'Detailed rankings, expected score distributions, captured probability mass, evidence hashes, '
              'token usage and paired outcomes are in summary.json and each task selection directory.', '']
    (root / 'summary.md').write_text('\n'.join(lines))
    return report
