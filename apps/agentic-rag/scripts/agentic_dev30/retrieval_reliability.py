"""只读汇总新2.3轮次的私有检索证据；不修改旧自动成绩或原始证据。"""
from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path

from jsonschema import Draft202012Validator

try:
    from scripts.agentic_dev30.experiment import (
        ARMS, file_sha, lines, load, obj_sha, read_costs, save, validate_evaluation_description,
    )
except ModuleNotFoundError as error:
    if error.name not in {'scripts', 'scripts.agentic_dev30'}:
        raise
    from experiment import (
        ARMS, file_sha, lines, load, obj_sha, read_costs, save, validate_evaluation_description,
    )


ASSESSMENT_SCHEMA = Path(__file__).resolve().parents[2]/'contracts/agent_rag/retrieval-assessment.schema.json'
HUMAN_REVIEW = 'awaiting_human_review'
METRICS = ('ranking_complete_tasks', 'model_assessed_with_limits_tasks', 'blocked_tasks',
    'rerank_physical_requests', 'rerank_failures_by_code', 'duplicate_paid_dispatches_prevented',
    'latency_seconds', 'known_cost_estimate_cny', 'unknown_reserve_cny')
COVERAGE_FIELDS = {'schema_version', 'expected_units', 'scored_units', 'failed_units',
    'ranking_complete', 'topk_scope', 'attempts_exhausted', 'failure_counts', 'failed_unit_ids',
    'expected_set_sha256', 'scored_set_sha256', 'failed_set_sha256', 'rerank_protocol'}
POST_RANKING_ERRORS = {'CANDIDATE_EVIDENCE_EXCEEDS_CONTEXT', 'RETRIEVAL_AUDIT_WRITE_FAILED'}
LOCAL_RERANK_FAILURES = {'QUOTE_MISMATCH', 'SPAN_REFERENCE_MISMATCH', 'INVALID_JSON',
    'INVALID_COMPLETION', 'TRANSPORT_TIMEOUT', 'TRANSPORT_ERROR', 'HTTP_429', 'HTTP_5XX'}


def _integer(value):
    return type(value) is int and value >= 0


def _hash(value):
    return isinstance(value, str) and re.fullmatch('[a-f0-9]{64}', value) is not None


def _coverage(query, protocol):
    error_code = query.get('error_code')
    if error_code is not None and (not isinstance(error_code, str)
            or not re.fullmatch('[A-Z_0-9]{1,64}', error_code) or query['complete']):
        raise ValueError('RETRIEVAL_COVERAGE_INVALID')
    post_ranking_failure = (query['complete'] is False and query['status'] in {'partial', 'error'}
        and error_code in POST_RANKING_ERRORS)
    value = query['ranking_coverage']
    if value is None:
        if query['complete'] or query['status'] != 'error':
            raise ValueError('RETRIEVAL_COVERAGE_INVALID')
        return None
    invalid = not isinstance(value, dict) or set(value) != COVERAGE_FIELDS
    if not invalid:
        counts = value['failure_counts']
        ids = value['failed_unit_ids']
        invalid = (
            value['schema_version'] != 'finrisk-ranking-coverage/1.0'
            or value['rerank_protocol'] != protocol
            or any(not _integer(value[k]) or value[k] > 100
                for k in ('expected_units', 'scored_units', 'failed_units'))
            or type(value['ranking_complete']) is not bool
            or type(value['attempts_exhausted']) is not bool
            or not isinstance(counts, dict)
            or any(not re.fullmatch('[A-Z_0-9]{1,64}', code)
                or not _integer(n) or n == 0 for code, n in counts.items())
            or not isinstance(ids, list)
            or any(not isinstance(uid, str) or not uid or len(uid) > 200 for uid in ids)
            or any(not _hash(value[k]) for k in
                ('expected_set_sha256', 'scored_set_sha256', 'failed_set_sha256')))
        if not invalid:
            complete = value['failed_units'] == 0
            invalid = (
                value['expected_units'] != value['scored_units'] + value['failed_units']
                or value['ranking_complete'] != complete
                or query['complete'] != complete and not (complete and post_ranking_failure)
                or value['topk_scope'] != ('full_pool' if complete else 'scored_subset')
                or complete and value['attempts_exhausted']
                or sum(counts.values()) != value['failed_units']
                or len(ids) != min(3, value['failed_units']) or len(set(ids)) != len(ids)
                or not post_ranking_failure and query['status'] not in (('ok', 'empty') if complete else ('partial',)))
    if invalid:
        raise ValueError('RETRIEVAL_COVERAGE_INVALID')
    return value


def _completed_point_status(point, report):
    """Mirror the Node unknown-condition projection without treating it as a retrieval failure."""
    checks = point.get('tag_checks', [])
    unknown = [tag for tag in checks if isinstance(tag, dict)
        and tag.get('status') == 'unknown' and isinstance(tag.get('conditions'), list)
        and any(isinstance(c, dict) and c.get('status') == 'unknown'
            for c in tag['conditions'])] if isinstance(checks, list) else []
    if not unknown:
        return 'model_assessed'
    validation = report.get('validation') or {}
    pending = point.get('pending_tags')
    completed_cases = {'reliable_match','partial_match','no_reliable_match'}
    if (report.get('needs_manual_review') is True
            and report.get('overall_judgment') in {'suspected','review_required'}
            and validation.get('schema_valid') is True
            and validation.get('references_valid') is True
            and validation.get('errors') == []
            and (report.get('coverage') or {}).get('all_submitted_segments_reviewed') is True
            and point.get('judgment') in {'suspected','review_required'}
            and point.get('status') == 'review_required'
            and point.get('case_match_status') in completed_cases
            and isinstance(pending, list)
            and all(isinstance(tag.get('tag'), str) and tag['tag'] in pending
                and tag.get('case_match_status') in completed_cases for tag in unknown)):
        return 'needs_input'
    return None


def _assessment(directory, task, task_id, plan):
    report_path = directory/'report.json'
    assessment_path = directory/'retrieval-assessment.json'
    if not report_path.exists():
        if assessment_path.exists():
            raise ValueError('RETRIEVAL_REPORT_MISMATCH')
        return None, 'report_missing'
    report = load(report_path)
    expected_input = obj_sha({'text':task['input']['text'],
        'user_context':task['input']['user_context']})
    versions = report.get('versions', {})
    if (report.get('task_id') != task_id or report.get('simulated')
            or report.get('execution_mode') != 'live'
            or versions.get('input_sha256') != expected_input
            or versions.get('snapshot_id') != plan['snapshot_id']
            or versions.get('evaluation_arm') != task['arm']
            or versions.get('evaluation_protocol_sha256') != plan['evaluation_protocol_sha256']):
        raise ValueError('RETRIEVAL_REPORT_IDENTITY_MISMATCH')
    if not assessment_path.exists():
        return None, 'assessment_missing'
    assessment = load(assessment_path)
    if list(Draft202012Validator(load(ASSESSMENT_SCHEMA)).iter_errors(assessment)):
        raise ValueError('RETRIEVAL_ASSESSMENT_INVALID')
    if (assessment['task_id'] != task_id or assessment['input_sha256'] != expected_input
            or assessment['snapshot_id'] != plan['snapshot_id']
            or assessment['evaluation_protocol_sha256'] != plan['evaluation_protocol_sha256']
            or assessment['report_sha256'] != file_sha(report_path)):
        raise ValueError('RETRIEVAL_ASSESSMENT_BINDING_MISMATCH')
    queries, points = assessment['queries'], assessment['points']
    if (len({q['query_id'] for q in queries}) != len(queries)
            or len({p['risk_point_id'] for p in points}) != len(points)):
        raise ValueError('RETRIEVAL_ASSESSMENT_IDENTITY_MISMATCH')
    report_points = {p['risk_point_id']:p for p in report.get('risk_points', [])}
    if (len(report_points) != len(report.get('risk_points', []))
            or set(report_points) != {p['risk_point_id'] for p in points}
            or any(p['case_match_status'] != report_points[p['risk_point_id']].get('case_match_status')
                or p['needs_manual_review'] != report.get('needs_manual_review') for p in points)):
        raise ValueError('RETRIEVAL_POINT_BINDING_MISMATCH')
    allowed_statuses = ({'not_run'} if not queries else {'error'} if any(q['status'] == 'error' for q in queries)
        else {'completed'} if all(q['complete'] for q in queries) else {'partial', 'error'})
    # A partial query can still be a hard retrieval error (zero scores or a global stop).
    if assessment['retrieval_status'] not in allowed_statuses:
        raise ValueError('RETRIEVAL_STATUS_MISMATCH')
    coverages = [_coverage(q, plan['retrieval_binding']['rerank_protocol']) for q in queries]
    limited = any(p['assessment_status'] == 'model_assessed_with_retrieval_limits' for p in points)
    blocked = assessment['retrieval_status'] == 'error' or any(p['assessment_status'] == 'blocked' for p in points)
    pending = [(q, c) for q, c in zip(queries, coverages) if not q['complete']]
    if limited and (assessment['retrieval_status'] != 'partial' or not pending
            or not all(q.get('error_code') is None and q['status'] == 'partial'
                and c and c['failed_units'] and c['scored_units'] and c['attempts_exhausted']
                and set(c['failure_counts']) <= LOCAL_RERANK_FAILURES for q, c in pending)):
        raise ValueError('RETRIEVAL_LIMITED_STATUS_MISMATCH')
    if report.get('task_status') == 'completed' and (assessment['retrieval_status'] != 'completed'
            or any(p['assessment_status'] != _completed_point_status(
                report_points[p['risk_point_id']], report) for p in points)):
        raise ValueError('RETRIEVAL_COMPLETED_STATUS_MISMATCH')
    return {'ranking_complete_tasks':int(
        bool(coverages) and all(c and c['ranking_complete'] for c in coverages)),
        'model_assessed_with_limits_tasks':int(limited), 'blocked_tasks':int(blocked)}, 'verified'


def _attempt_metrics(directory):
    path = directory/'rerank-ledger.json.attempts.json'
    if not path.exists():
        return None, None
    audit = load(path)
    if (not isinstance(audit, dict) or audit.get('schema_version') != 'finrisk-rerank-attempts/1.0'
            or not isinstance(audit.get('attempts'), list)):
        raise ValueError('RETRIEVAL_ATTEMPT_AUDIT_INVALID')
    ordinals, dispatched, failures = set(), set(), Counter()
    unfinished = False
    for row in audit['attempts']:
        if not isinstance(row, dict):
            raise ValueError('RETRIEVAL_ATTEMPT_AUDIT_INVALID')
        ordinal, state, code = row.get('ordinal'), row.get('state'), row.get('error_code')
        started = row.get('transport_started_at')
        terminal = state in {'settled', 'finished_usage_unknown', 'aborted_unsent'}
        if (not _integer(ordinal) or ordinal == 0 or ordinal in ordinals
                or state not in {'transport_started', 'settled', 'finished_usage_unknown', 'aborted_unsent'}
                or not isinstance(row.get('pair_id'), str) or not row['pair_id']
                or type(row.get('retry_index')) is not int or row['retry_index'] not in (0, 1)
                or type(row.get('will_retry')) is not bool
                or terminal != isinstance(row.get('completed_at'), str)
                or (state == 'aborted_unsent' and started is not None)
                or (state != 'aborted_unsent' and not isinstance(started, str))
                or code is not None and (not isinstance(code, str) or not re.fullmatch('[A-Z_0-9]{1,64}', code))):
            raise ValueError('RETRIEVAL_ATTEMPT_AUDIT_INVALID')
        ordinals.add(ordinal)
        if started is not None:
            dispatched.add(ordinal)
            if code is not None:
                failures[code] += 1
            unfinished |= not terminal
    ledger_path = directory/'rerank-ledger.json'
    if ledger_path.exists():
        reservations = load(ledger_path).get('request_reservations')
        if not isinstance(reservations, list):
            return None, None
        expected = {i for i, row in enumerate(reservations, 1) if row.get('transport_started_at') is not None}
        if expected != dispatched:
            return None, None  # A missing audit ordinal cannot prove a zero-failure dispatch.
        for row in audit['attempts']:
            if row['ordinal'] > len(reservations) or reservations[row['ordinal']-1].get('pair_id') != row['pair_id']:
                raise ValueError('RETRIEVAL_ATTEMPT_BINDING_MISMATCH')
    return len(dispatched), None if unfinished else dict(sorted(failures.items()))


def _duplicate_prevention(directory, plan):
    path = directory/'events.jsonl'
    if not path.exists():
        return None
    counts, seen = [], set()
    for event in lines(path):
        data = event.get('data', {})
        if event.get('type') not in {'tool.result', 'tool.failure'} or data.get('name') != 'search_cases':
            continue
        if event.get('type') == 'tool.failure':
            return None
        meta = data.get('response', {}).get('meta', {})
        value = meta.get('duplicate_paid_dispatches_prevented')
        if not _integer(value):
            return None
        sequence = event.get('sequence')
        if not _integer(sequence) or sequence in seen or meta.get('snapshot_id') != plan['snapshot_id']:
            raise ValueError('RETRIEVAL_EVENT_BINDING_MISMATCH')
        seen.add(sequence)
        counts.append(value)
    return sum(counts) if counts else None


def _item(task, plan, execution):
    key = task['task_key']
    basename = key.replace(':', '_')
    if not basename or any(c in basename for c in '/\\') or basename in {'.', '..'}:
        raise ValueError('RETRIEVAL_TASK_IDENTITY_MISMATCH')
    item = {'task_key':key, 'arm':task['arm'], 'evidence_status':'missing',
        **dict.fromkeys(METRICS), 'human_semantic_review_status':HUMAN_REVIEW}
    records = [load(p) for p in (execution/(basename+'.json'), execution/(basename+'.attempt.json')) if p.exists()]
    current = execution/'current-task.json'
    if current.exists():
        value = load(current)
        if value.get('task_key') == key:
            records.append(value)
    if not records:
        return item
    if any('task_key' in r and r['task_key'] != key for r in records):
        raise ValueError('RETRIEVAL_TASK_IDENTITY_MISMATCH')
    task_ids = {r.get('task_id') for r in records}
    if len(task_ids) != 1:
        raise ValueError('RETRIEVAL_TASK_IDENTITY_MISMATCH')
    task_id = next(iter(task_ids))
    if not isinstance(task_id, str) or not re.fullmatch('[A-Za-z0-9_-]{1,128}', task_id):
        raise ValueError('RETRIEVAL_TASK_IDENTITY_MISMATCH')
    item['task_id'] = task_id
    directory = execution/'tasks'/task_id
    metrics, item['evidence_status'] = _assessment(directory, task, task_id, plan)
    if metrics:
        item.update(metrics)
    durations = [r['wall_seconds'] for r in records if 'wall_seconds' in r]
    if durations:
        if any(type(n) not in (int, float) or not math.isfinite(n) or n < 0 for n in durations) or len(set(durations)) != 1:
            raise ValueError('RETRIEVAL_LATENCY_INVALID')
        item['latency_seconds'] = durations[0]
    item['rerank_physical_requests'], item['rerank_failures_by_code'] = _attempt_metrics(directory)
    item['duplicate_paid_dispatches_prevented'] = _duplicate_prevention(directory, plan)
    try:
        known, unknown = read_costs(execution/'model-ledgers'/(basename+'.json'), directory/'rerank-ledger.json')
    except (OSError, ValueError):
        known, unknown = 0, 1_500_000  # Same conservative interpretation as execute.
    item.update(known_cost_estimate_cny=known/1_000_000, unknown_reserve_cny=unknown/1_000_000)
    return item


def _aggregate(items):
    result = {'planned_tasks':len(items), 'human_semantic_review_status':HUMAN_REVIEW}
    for metric in METRICS:
        values = [r[metric] for r in items]
        if any(v is None for v in values):
            result[metric] = None
        elif metric == 'rerank_failures_by_code':
            counts = Counter()
            for value in values:
                counts.update(value)
            result[metric] = dict(sorted(counts.items()))
        else:
            result[metric] = sum(values)
    return result


def build_comparison(plan_path, execution_dir=None):
    """Read all planned tasks. Missing evidence propagates null, never invented zero."""
    plan_path = Path(plan_path).resolve()
    plan = load(plan_path)
    description = plan.get('evaluation_runtime_description')
    validate_evaluation_description(description)
    binding = plan.get('retrieval_binding') or {}
    if (plan.get('schema_version') != 'finrisk-three-arm-experiment/1.0'
            or plan.get('evaluation_protocol_sha256') != description['evaluation_protocol_sha256']
            or binding.get('rerank_protocol') not in {'quote-copy-v1', 'span-ref-v2'}
            or not _hash(binding.get('rules_sha256'))):
        raise ValueError('EVALUATION_PROTOCOL_MISMATCH')
    execution = Path(execution_dir).resolve() if execution_dir is not None else plan_path.parent/'execution'
    tasks = plan['tasks']
    if len({t['task_key'] for t in tasks}) != len(tasks) or any(t['arm'] not in ARMS for t in tasks):
        raise ValueError('RETRIEVAL_TASK_IDENTITY_MISMATCH')
    items = [_item(task, plan, execution) for task in tasks]
    task_ids = [item['task_id'] for item in items if 'task_id' in item]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError('RETRIEVAL_TASK_IDENTITY_MISMATCH')
    return {'schema_version':'finrisk-retrieval-reliability-comparison/1.0',
        'plan_sha256':file_sha(plan_path), **_aggregate(items),
        'arms':{arm:_aggregate([r for r in items if r['arm'] == arm]) for arm in ARMS},
        'items':items, 'not_account_statement':True,
        'interpretation':'retrieval reliability evidence only; ledger cost estimates are not billing; '
            'missing evidence remains null and all planned tasks remain in the denominator; '
            'business semantics await human review'}


def write_comparison(plan_path, execution_dir=None):
    """Only write the independent statistics artifact after every binding validates."""
    result = build_comparison(plan_path, execution_dir)
    directory = Path(execution_dir).resolve() if execution_dir is not None else Path(plan_path).resolve().parent/'execution'
    target = directory/'retrieval-reliability-comparison.json'
    save(target, result)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--execution-dir', type=Path)
    args = parser.parse_args()
    print(write_comparison(args.plan, args.execution_dir))


if __name__ == '__main__':
    main()
