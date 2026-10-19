"""冻结三臂实验；默认prepare零网络，execute须绑定本轮新授权。"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
from statistics import median
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ARMS = ('R1', 'R2', 'A')
PILOT_IDS = ('S02', 'C06', 'C15', 'C01', 'C11', 'C08')
TASK_RESERVE_MICRO = 1_500_000  # 旧模型账本1元 + 重排控制0.5元；新整轮授权另行约束。
EXTERNAL_SCOPE = 'selected texts, user contexts and retrieved penalty excerpts to official DeepSeek API'
RESILIENCE_POLICY = {'version':'finrisk-resilience/1.0','task_transport_failure_limit':3,'round_infrastructure_failure_limit':2,'confirmed_content_filter_resets_streak':True,'permanent_http_statuses':[400,401,402,403]}
EVALUATION_VERSION = 'finrisk-evaluation/2.4'
RETRIEVAL_RELIABILITY = {'coverage_version':'finrisk-ranking-coverage/1.0',
    'schema_version':'finrisk-retrieval-assessment/1.0','pair_attempt_limit':2,
    'rerank_protocols':['quote-copy-v1','span-ref-v2'],'configured_in_v2':True}


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def obj_sha(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    tmp.replace(path)


def verify_files(records):
    for item in records:
        p = Path(item['path'])
        if not p.is_file() or file_sha(p) != item['sha256']:
            raise ValueError('FROZEN_FILE_CHANGED:' + str(p))


def schedule(queries, references, pilot=True, review_ids=None):
    for q in queries:
        if set(q) != {'sample_id', 'text', 'user_context'}:
            raise ValueError('INPUT_ALLOWLIST')
        if not isinstance(q['text'], str) or not isinstance(q['user_context'], dict):
            raise ValueError('INPUT_TYPE')
        if set(q['user_context']) - {'declared_scene','author_role','audience_role','purpose_hint','additional_notes'}:
            raise ValueError('CONTEXT_ALLOWLIST')
    query_map = {q['sample_id']: q for q in queries}
    ref_map = {r['sample_id']: r for r in references}
    if len(query_map) != len(queries) or len(ref_map) != len(references) or set(query_map) != set(ref_map):
        raise ValueError('INPUT_REFERENCE_MAPPING')
    if pilot or review_ids is not None:
        by_review = {r['review_id']: r for r in references}
        chosen = list(PILOT_IDS) if review_ids is None else review_ids
        if (not isinstance(chosen,list) or not chosen or len(set(chosen))!=len(chosen)
            or any(rid not in by_review for rid in chosen)):
            raise ValueError('INVALID_REVIEW_SELECTION')
        selected = [query_map[by_review[rid]['sample_id']] for rid in chosen]
    else:
        selected = sorted(queries, key=lambda q: q['sample_id'])
    orders = list(itertools.permutations(ARMS))
    return [{'task_key': q['sample_id'] + ':' + arm, 'arm': arm, 'input': q}
            for i, q in enumerate(selected) for arm in orders[i % 6]]


class RoundBudget:
    """单进程串行调度的整轮账本；创建排他且绝不自动恢复/重发。"""
    def __init__(self, path, plan_sha256, cap_micro):
        self.path = Path(path)
        if type(cap_micro) is not int or cap_micro < TASK_RESERVE_MICRO:
            raise ValueError('ROUND_BUDGET_TOO_SMALL')
        self.value = {'schema_version':'finrisk-evaluation-round-ledger/1.0',
                      'plan_sha256':plan_sha256, 'cap_micro_cny':cap_micro, 'tasks':[]}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.path.open('x', encoding='utf-8') as f:
                json.dump(self.value, f)
        except FileExistsError:
            raise ValueError('ALREADY_STARTED:no automatic restart') from None

    def remaining(self):
        return self.value['cap_micro_cny'] - sum(r['known_micro_cny'] + r['unknown_micro_cny'] for r in self.value['tasks'])

    def reserve(self, task_key, reserve_micro=TASK_RESERVE_MICRO):
        if type(reserve_micro) is not int or reserve_micro not in (TASK_RESERVE_MICRO, 500_000):
            raise ValueError('INVALID_ROUND_RESERVATION')
        if any(r['task_key'] == task_key for r in self.value['tasks']):
            raise ValueError('DUPLICATE_TASK')
        if self.remaining() < reserve_micro:
            raise ValueError('ROUND_BUDGET_EXHAUSTED')
        self.value['tasks'].append({'task_key':task_key, 'status':'started',
                                   'known_micro_cny':0, 'unknown_micro_cny':reserve_micro})
        save(self.path, self.value)

    def finish(self, task_key, known_micro, unknown_micro):
        if any(type(v) is not int or v < 0 for v in (known_micro, unknown_micro)):
            raise ValueError('INVALID_COST')
        row = next(r for r in self.value['tasks'] if r['task_key'] == task_key)
        if row['status'] != 'started':
            raise ValueError('TASK_ALREADY_SETTLED')
        row.update(status='finished', known_micro_cny=known_micro, unknown_micro_cny=unknown_micro)
        save(self.path, self.value)
        if self.remaining() < 0:
            raise ValueError('USAGE_EXCEEDED_RESERVATION_STOP')


class RoundCircuit:
    """Durable round health; business review outcomes never count as outages."""
    def __init__(self,path):
        self.path=Path(path)
        self.value=load(self.path) if self.path.exists() else {
            'schema_version':'finrisk-round-health/1.0','failure_limit':2,
            'failure_streak':0,'open':False,'reason':None,'last_failure':None,'observed_tasks':[]}
        v=self.value
        if (v.get('schema_version')!='finrisk-round-health/1.0'
            or type(v.get('failure_limit')) is not int or v['failure_limit']!=2
            or type(v.get('failure_streak')) is not int or v['failure_streak']<0
            or type(v.get('open')) is not bool
            or v.get('reason') not in (None,'consecutive_infrastructure_failures','permanent_provider_rejection')
            or not isinstance(v.get('observed_tasks'),list)
            or not all(isinstance(k,str) for k in v['observed_tasks'])
            or len(set(v['observed_tasks']))!=len(v['observed_tasks'])
            or v['open']!=(v['reason'] is not None)):
            raise ValueError('INVALID_ROUND_HEALTH')
        if not self.path.exists():save(self.path,self.value)

    def before_task(self):
        if self.value['open']:raise ValueError('ROUND_CIRCUIT_OPEN:'+self.value['reason'])

    @staticmethod
    def _safe_diagnostic(value):
        if not isinstance(value,dict):return None
        stage=value.get('stage');category=value.get('category');code=value.get('error_code');status=value.get('http_status')
        return {
            'stage':stage if isinstance(stage,str) and stage in {'fetch','response','stream'} else None,
            'category':category if isinstance(category,str) and category in {'dns','connect','tls','timeout','abort','http','validation','ok','unknown'} else 'unknown',
            'error_code':code if isinstance(code,str) and code in {'ENOTFOUND','EAI_AGAIN','EAI_NONAME','ECONNREFUSED','ECONNRESET','ECONNABORTED','ENETUNREACH','EHOSTUNREACH','ETIMEDOUT','EPIPE','EINTR','CERTIFICATE_VERIFY_FAILED','SSL_ERROR','UNEXPECTED_EOF_WHILE_READING','WRONG_VERSION_NUMBER','TLSV1_ALERT_PROTOCOL_VERSION','TLSV1_ALERT_INTERNAL_ERROR','UND_ERR_CONNECT_TIMEOUT','UND_ERR_HEADERS_TIMEOUT','UND_ERR_BODY_TIMEOUT','ABORT_ERR'} else None,
            'http_status':status if type(status) is int and 100<=status<=599 else None}

    def observe(self,task_key,report,events):
        if self.value['open'] or task_key in self.value['observed_tasks']:return self.value
        self.value['observed_tasks'].append(task_key)
        failures=[]
        resources=(report or {}).get('resources',{})
        retrieval=resources.get('retrieval_accounting',{})
        if retrieval.get('transport_circuit_open') is True:
            failures.append(('RETRIEVAL_TRANSPORT_CIRCUIT_OPEN',self._safe_diagnostic(retrieval.get('transport_diagnostic'))))
        confirmed_semantic_response=False
        allowed={'MODEL_TRANSPORT_FAILURE','MODEL_STREAM_FAILURE','MODEL_STREAM_NOT_FINALIZED','model_request_timeout'}
        for event in events:
            if event.get('type')!='model.failure':continue
            data=event.get('data',{});reason=data.get('reason')
            if not isinstance(reason,str):continue
            if reason=='MODEL_CONTENT_FILTER' and type(data.get('physical_requests')) is int and data['physical_requests']>0:
                confirmed_semantic_response=True
            status=int(reason[-3:]) if re.fullmatch(r'PROVIDER_HTTP_[0-9]{3}',reason) else None
            if reason in allowed or status in (400,401,402,403,429) or status is not None and status>=500:
                diag=self._safe_diagnostic(data.get('diagnostic'))
                if status is not None:
                    diag=diag or {'stage':'response','category':'http','error_code':None,'http_status':status}
                    diag['http_status']=status
                failures.append((reason,diag))
        if failures:
            permanent=next(((reason,diag) for reason,diag in failures if (diag or {}).get('http_status') in (400,401,402,403)),None)
            reason,diag=permanent or failures[-1]
            self.value['failure_streak']+=1
            self.value['last_failure']={'task_key':task_key,'reason':reason,'diagnostic':diag}
            if permanent or self.value['failure_streak']>=self.value['failure_limit']:
                self.value.update(open=True,reason='permanent_provider_rejection' if permanent else 'consecutive_infrastructure_failures')
        elif (report and report.get('task_status')=='completed') or confirmed_semantic_response:
            self.value.update(failure_streak=0,last_failure=None)
        save(self.path,self.value)
        return self.value


def require_authorization(approval, plan_sha, plan=None):
    if approval.get('approval_status') != 'approved':
        raise ValueError('APPROVAL_REQUIRED')
    if approval.get('plan_sha256') != plan_sha:
        raise ValueError('PLAN_MISMATCH')
    if not approval.get('human_statement') or not approval.get('approved_at'):
        raise ValueError('HUMAN_APPROVAL_RECORD_REQUIRED')
    if plan is not None and (approval.get('external_data_scope') != EXTERNAL_SCOPE
        or approval.get('max_judgment_requests') != plan['max_judgment_requests']
        or approval.get('max_rerank_requests') != plan['max_rerank_requests']):
        raise ValueError('APPROVAL_SCOPE_MISMATCH:prepare a smaller plan before approval')
    cap = approval.get('total_cap_cny')
    if isinstance(cap, bool) or not isinstance(cap, (float, int)) or not math.isfinite(cap) or cap < 1.5:
        raise ValueError('INVALID_APPROVED_CAP')
    return math.floor(cap * 1_000_000)


def score_results(tasks, references, reports, timings=None, attempts=None):
    reference_map = {r['sample_id']: r for r in references}
    result = {'arms':{}, 'strata':{}, 'items':[], 'semantic_evidence_status':'awaiting_human_review',
              'interpretation':'automatic judgment/label checks only; not full business correctness'}
    for arm in ARMS:
        result['arms'][arm] = dict.fromkeys(['planned','completed','missing','failed','judgment_exact','supported_tags_exact','label_true_positive','label_false_positive','label_false_negative'], 0)
    for task in tasks:
        stats = result['arms'][task['arm']]
        stats['planned'] += 1
        ref = reference_map[task['input']['sample_id']]
        report = reports.get(task['task_key'])
        attempt = (attempts or {}).get(task['task_key'])
        if report and report.get('simulated'):
            raise ValueError('SIMULATED_RESULTS_ARE_NOT_QUALITY_SCORES')
        complete = bool(report and report.get('task_status') == 'completed')
        stats['missing' if report is None and not attempt else 'completed' if complete else 'failed'] += 1
        expected = set(ref['reference']['supported_tags'])
        actual = {tag for point in (report or {}).get('risk_points', []) for tag in point.get('supported_tags', [])}
        exact_judgment = bool(complete and report['overall_judgment'] == ref['reference']['overall_judgment'])
        exact_tags = complete and actual == expected
        stats['judgment_exact'] += exact_judgment
        stats['supported_tags_exact'] += exact_tags
        stats['label_true_positive'] += len(actual & expected) if complete else 0
        stats['label_false_positive'] += len(actual - expected)
        stats['label_false_negative'] += len(expected - actual) if complete else len(expected)
        result['items'].append({'task_key':task['task_key'], 'sample_id':ref['sample_id'],
                                'difficulty':ref['difficulty'], 'arm':task['arm'],
                                'stratum':'complex' if ref['difficulty']=='complex' else 'simple_risk' if expected else 'simple_normal',
                                'status':report.get('task_status') if report else (attempt or {}).get('status','missing'),
                                'judgment_exact':exact_judgment, 'supported_tags_exact':exact_tags,
                                'end_to_end_seconds':(timings or {}).get(task['task_key']),
                                'resources':(report or {}).get('resources'),
                                'conditions_and_case_semantics':'awaiting_human_review'})
    for group in ('simple_risk','simple_normal','complex'):
        result['strata'][group]={}
        for arm in ARMS:
            items=[r for r in result['items'] if r['stratum']==group and r['arm']==arm]
            durations=[r['end_to_end_seconds'] for r in items if r['end_to_end_seconds'] is not None]
            result['strata'][group][arm]={'planned':len(items),
                'completed':sum(r['status']=='completed' for r in items),
                'judgment_exact':sum(r['judgment_exact'] for r in items),
                'supported_tags_exact':sum(r['supported_tags_exact'] for r in items),
                'timed_tasks':len(durations), 'median_seconds':median(durations) if durations else None}
    if not any(r.get('task_status')=='completed' for r in reports.values()):
        result['quality_status']='not_run' if not reports else 'no_completed_results'
        for stats in result['arms'].values():
            for key in ('judgment_exact','supported_tags_exact','label_true_positive','label_false_positive','label_false_negative'):
                stats[key]=None
        for groups in result['strata'].values():
            for stats in groups.values():
                stats['judgment_exact']=stats['supported_tags_exact']=None
        for item in result['items']:
            item['judgment_exact']=item['supported_tags_exact']=None
    else:
        result['quality_status']='exploratory_automatic_checks_not_full_semantic_pass'
    return result


def format_repair_limits(plan):
    limits = plan.get('format_repair_limits', {'R1': 0, 'R2': 0, 'A': 0})
    if (not isinstance(limits, dict) or set(limits) != set(ARMS)
            or any(type(v) is not int for v in limits.values())
            or limits['R1'] != 0 or limits['R2'] != 0 or limits['A'] not in (0, 1)):
        raise ValueError('INVALID_FORMAT_REPAIR_LIMITS')
    return dict(limits)


def bind_format_repair_config(bundle, arm, plan):
    limits = format_repair_limits(plan)
    if arm not in ARMS:
        raise ValueError('INVALID_EVALUATION_ARM')
    return {**bundle, 'config': {**bundle.get('config', {}),
            'evaluation_format_repair_limit': limits[arm]}}


def frozen_runtime(project):
    selected = set()
    for folder, suffixes in [('agent_harness/src', {'.ts'}), ('agent_harness/dist/src', {'.js'}),
                             ('configs/agent_rag', {'.json','.txt'}), ('contracts/agent_rag', {'.json','.csv'}),
                             ('src/agent_rag', {'.py'}), ('scripts/agentic_dev30', {'.py'})]:
        selected.update(p for p in (project / folder).rglob('*') if p.is_file() and p.suffix in suffixes)
    selected.update(project / p for p in ['agent_harness/package.json','agent_harness/package-lock.json',
                                         'requirements-agent-rag.txt','resources/agent_rag/deepseek-rerank-rules.md',
                                         'resources/agent_rag/deepseek-rerank-span-rules.md'])
    if not (project/'agent_harness/dist/src/evaluation_cli.js').exists():
        raise ValueError('EVALUATION_BUILD_REQUIRED')
    return [{'path':str(p.resolve()), 'sha256':file_sha(p)} for p in sorted(selected)]


def snapshot_rerank_binding(snapshot, project):
    """A missing manifest protocol retains the historical quote-copy protocol."""
    backend = snapshot.get('backend') or {}
    protocol = backend.get('rerank_protocol', 'quote-copy-v1')
    rule_names = {'quote-copy-v1':'deepseek-rerank-rules.md',
                  'span-ref-v2':'deepseek-rerank-span-rules.md'}
    if protocol not in rule_names or backend.get('mode') != 'live':
        raise ValueError('RERANK_PROTOCOL_MISMATCH')
    rules = backend.get('assets', {}).get('rules') or {}
    path = Path(rules.get('path', ''))
    expected = Path(project)/'resources/agent_rag'/rule_names[protocol]
    if (not path.is_file() or not expected.is_file()
            or rules.get('sha256') != file_sha(path)
            or rules['sha256'] != file_sha(expected)):
        raise ValueError('RERANK_RULES_MISMATCH')
    return {'rerank_protocol':protocol,'rules_sha256':rules['sha256']}


def validate_evaluation_description(description, project=None):
    if (not isinstance(description, dict)
            or description.get('schema_version') != EVALUATION_VERSION
            or canonical(description.get('retrieval_reliability')) != canonical(RETRIEVAL_RELIABILITY)):
        raise ValueError('EVALUATION_PROTOCOL_MISMATCH')
    digest = description.get('evaluation_protocol_sha256')
    if not isinstance(digest, str) or not re.fullmatch('[a-f0-9]{64}', digest):
        raise ValueError('EVALUATION_PROTOCOL_MISMATCH')
    if project is not None and description.get('system_prompt_sha256') != file_sha(
            Path(project)/'configs/agent_rag/evaluation-system-prompt.txt'):
        raise ValueError('EVALUATION_PROMPT_MISMATCH')


def paired_prefetch_module():
    try:
        from scripts.agentic_dev30 import paired_prefetch
    except ModuleNotFoundError as error:
        if error.name not in {'scripts', 'scripts.agentic_dev30'}:
            raise
        import paired_prefetch
    return paired_prefetch


def validate_dense_startup_timeout_seconds(value):
    if value is not None and (type(value) not in (int, float)
            or not math.isfinite(value) or not 0 < value <= 600):
        raise ValueError('INVALID_DENSE_STARTUP_TIMEOUT')


def prepare(project, reference_dir, snapshot_dir, output, node_path, pilot=True, review_ids=None,
            adaptive_format_repair_limit=0, snapshot_mode='full9639', paired_prefetch=False,
            dense_startup_timeout_seconds=None):
    validate_dense_startup_timeout_seconds(dense_startup_timeout_seconds)
    if type(adaptive_format_repair_limit) is not int or adaptive_format_repair_limit not in (0, 1):
        raise ValueError('INVALID_ADAPTIVE_FORMAT_REPAIR_LIMIT')
    if type(paired_prefetch) is not bool:
        raise ValueError('INVALID_PAIRED_PREFETCH_SETTING')
    project, reference_dir, snapshot_dir, output = map(lambda p: Path(p).resolve(), (project, reference_dir, snapshot_dir, output))
    if output.exists():
        raise ValueError('OUTPUT_EXISTS')
    manifest_path = reference_dir/'manifest.json'
    manifest = load(manifest_path)
    tracked_manifest=project/'resources/agent_rag/evaluation/dev30-human-reference-manifest.json'
    if manifest != load(tracked_manifest):
        raise ValueError('APPROVAL_MANIFEST_NOT_PROJECT_FROZEN_VERSION')
    for row in manifest['files']:
        path = reference_dir/row['path']
        if file_sha(path) != row['sha256']:
            raise ValueError('HUMAN_REFERENCE_CHANGED:' + row['path'])
    if manifest['human_approved_count'] != 30:
        raise ValueError('ALL_30_APPROVALS_REQUIRED')
    queries, refs = lines(reference_dir/'inputs/queries.jsonl'), lines(reference_dir/'evaluation/references.jsonl')
    if len(queries)!=30 or len(refs)!=30 or len({r['review_id'] for r in refs})!=30:
        raise ValueError('EXACT_30_UNIQUE_APPROVED_CASES_REQUIRED')
    for ref in refs:
        q = next(q for q in queries if q['sample_id'] == ref['sample_id'])
        if not ref['human_approved'] or obj_sha({'text':q['text'], 'user_context':q['user_context']}) != ref['input_sha256']:
            raise ValueError('APPROVED_INPUT_MISMATCH')
    tasks = schedule(queries, refs, pilot, review_ids)
    prefetch_module = paired_prefetch_module()
    snapshot = prefetch_module.validate_snapshot(snapshot_dir, snapshot_mode, project)
    retrieval_binding = snapshot_rerank_binding(snapshot, project)
    assets = [{'path':str(snapshot_dir/'manifest.json'), 'sha256':file_sha(snapshot_dir/'manifest.json')}]
    for group in (snapshot['assets'], snapshot['backend']['assets']):
        assets.extend({'path':v['path'], 'sha256':v['sha256']} for v in group.values())
    dense = load(next(v['path'] for k,v in snapshot['backend']['assets'].items() if k == 'dense_config'))
    for field in ('vector_meta_path', 'vector_path'):
        assets.append({'path':dense[field], 'sha256':file_sha(dense[field])})
    # 冻结模型文件，而非只冻结目录名；显式不hash密钥。
    for p in sorted(Path(dense['model_path']).rglob('*')):
        if p.is_file() and p.suffix in {'.json','.txt','.safetensors','.bin','.model'}:
            assets.append({'path':str(p), 'sha256':file_sha(p)})
    verify_files(assets)
    bindings = frozen_runtime(project)
    description=json.loads(subprocess.check_output([str(node_path),str(project/'agent_harness/dist/src/evaluation_cli.js'),'--describe'],timeout=20))
    validate_evaluation_description(description, project)
    output.mkdir(parents=True)
    (output/'evaluation').mkdir()
    (output/'inputs').mkdir()
    inputs = [t['input'] for i,t in enumerate(tasks) if i % 3 == 0]
    (output/'inputs/queries.jsonl').write_bytes(('\n'.join(canonical(q) for q in inputs)+'\n').encode('utf-8'))
    (output/'evaluation/references.jsonl').write_bytes((reference_dir/'evaluation/references.jsonl').read_bytes())
    plan = {
        'schema_version':'finrisk-three-arm-experiment/1.0',
        'created_at':datetime.now(timezone.utc).isoformat(), 'project_root':str(project),
        'scope':'smoke'+str(len(tasks)//3) if review_ids is not None else 'pilot6' if pilot else 'development30', 'tasks':tasks, 'task_count':len(tasks),
        'selected_review_ids':review_ids, 'resilience_policy':RESILIENCE_POLICY,
        'format_repair_limits': {'R1': 0, 'R2': 0, 'A': adaptive_format_repair_limit},
        'source_approval':{'path':str(manifest_path), 'sha256':file_sha(manifest_path)},
        'model_input_file':str(output/'inputs/queries.jsonl'),
        'reference_file':str(output/'evaluation/references.jsonl'),
        'snapshot_dir':str(snapshot_dir), 'snapshot_id':snapshot['snapshot_id'],
        'snapshot_mode':snapshot_mode,
        'evaluation_protocol_sha256':description['evaluation_protocol_sha256'],
        'evaluation_runtime_description':description,
        'retrieval_binding':retrieval_binding,
        'frozen_files':bindings + assets + [{'path':str(manifest_path), 'sha256':file_sha(manifest_path)},
                         {'path':str(tracked_manifest),'sha256':file_sha(tracked_manifest)},
                         {'path':str(output/'inputs/queries.jsonl'), 'sha256':file_sha(output/'inputs/queries.jsonl')},
                         {'path':str(output/'evaluation/references.jsonl'), 'sha256':file_sha(output/'evaluation/references.jsonl')}],
        'limits':{'task_model_requests':6, 'R1_model_requests':1, 'task_searches':2,
                  'task_tool_actions':8, 'task_deadline_seconds':270,
                  'task_rerank_requests':240, 'task_rerank_cap_cny':0.5,
                  'task_model_ledger_cap_cny':1, 'round_task_reserve_cny':1.5},
        'max_judgment_requests':sum(1 if t['arm']=='R1' else 6 for t in tasks),
        'max_rerank_requests':len(tasks)*240,
        'scheduling':'all six arm permutations, repeated; sequential, independent decision runs',
        'cache_policy':'one resident backend; only identical bound rerank pairs reused; actual cache usage logged; no cross-arm model judgments',
        'quality_status':'not_run', 'network_calls_during_preparation':0,
    }
    if dense_startup_timeout_seconds is not None:
        plan['dense_startup_timeout_seconds'] = dense_startup_timeout_seconds
    if paired_prefetch:
        config = prefetch_module.configuration(tasks, snapshot['snapshot_id'])
        plan.update(paired_prefetch=config, cache_policy=prefetch_module.CACHE_POLICY,
                    timing_policy=prefetch_module.TIMING_POLICY,
                    max_rerank_requests=(len(tasks) + config['task_count']) * 240)
        plan['limits'].update(prefetch_module.BOUND_LIMITS)
    save(output/'plan.json', plan)
    save(output/'authorization.template.json', {'approval_status':'pending', 'plan_sha256':file_sha(output/'plan.json'),
        'human_statement':None, 'approved_at':None, 'total_cap_cny':None,
        'external_data_scope':EXTERNAL_SCOPE,
        'max_judgment_requests':plan['max_judgment_requests'], 'max_rerank_requests':plan['max_rerank_requests']})
    save(output/'preflight.json', {'status':'offline_prepared', 'tasks':len(tasks), 'inputs':len(inputs),
        'paid_api_calls':0, 'quality_scores':None, 'approval_status':'pending',
        'recommended_initial_cap_cny':3 if review_ids is not None else 6 if pilot else None,
        'unconditional_all_tasks_reservation_cny':len(tasks)*1.5,
        'unconditional_prefetch_reservation_cny':len(inputs)*0.5 if paired_prefetch else 0,
        'budget_note':'Each task reserves 1.5 CNY from the NEW round budget before start; optional shared prefetch reserves 0.5 CNY once per material. Known ledger usage releases unused reserve; unknown usage stays booked. The approved round cap may stop execution early; it is not a promise of completion or billing estimate.'})
    return plan


def read_costs(model_path, rerank_path):
    known = unknown = 0
    for p, maximum in ((model_path, 1_000_000), (rerank_path, 500_000)):
        if not p.exists():
            unknown += maximum
            continue
        value = load(p)
        k, u = value.get('settled_cost_micro_cny'), value.get('reserved_cost_micro_cny')
        if type(k) is not int or type(u) is not int or min(k,u) < 0:
            unknown += maximum
        else:
            known += k
            unknown += u
    return known, unknown


def cleanup_resources(service, snapshot, server, thread, sock, old_env):
    errors=[]
    actions=[]
    if service:
        actions.extend(('cancel',control.cancel) for control in service.controls.values())
    if server:
        actions.append(('server',lambda:setattr(server,'should_exit',True)))
    if thread:
        actions.append(('thread',lambda:thread.join(timeout=10)))
    if sock:
        actions.append(('socket',sock.close))
    dense=getattr(getattr(snapshot,'backend',None),'dense',None)
    if dense:
        actions.append(('dense',dense.close))
    try:
        for name,action in actions:
            try:
                action()
            except BaseException as error:
                errors.append({'resource':name,'error':type(error).__name__})
    finally:
        os.environ.clear()
        os.environ.update(old_env)
    return errors


def local_tool_app(service):
    from fastapi import APIRouter, FastAPI
    from agent_rag import router
    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
    allowed_paths = {'/internal/agent-rag/search','/internal/agent-rag/source/read',
        '/internal/agent-rag/source/preview','/internal/agent-rag/token-count',
        '/internal/agent-rag/risk-definitions','/internal/agent-rag/verify',
        '/internal/agent-rag/retrieval/cancel'}
    internal=APIRouter()
    internal.routes.extend(route for route in router.router.routes if getattr(route, 'path', None) in allowed_paths)
    # include_router重建路由，保证依赖覆盖绑定本轮service。
    app.include_router(internal)
    app.dependency_overrides[router.get_service] = lambda: service
    return app


def bind_evaluation_contract(bundle, description, project):
    """Bind the frozen experimental prompt without changing ordinary service tasks."""
    project = Path(project)
    validate_evaluation_description(description, project)
    prompt_hash = file_sha(project/'configs/agent_rag/evaluation-system-prompt.txt')
    contract = bundle.get('execution_contract') or {}
    if (description.get('system_prompt_sha256') != prompt_hash
            or contract.get('prompt_sha256') != file_sha(project/'configs/agent_rag/system-prompt.txt')):
        raise ValueError('EVALUATION_PROMPT_MISMATCH')
    protocol_hash = description['evaluation_protocol_sha256']
    bound = {**bundle, 'execution_contract': {**contract, 'prompt_sha256':prompt_hash,
            'evaluation_protocol_sha256':protocol_hash}}
    if bundle.get('config', {}).get('retrieval_profile') == 'v2-hybrid-flash-v1':
        bound['config'] = {**bundle['config'], 'retrieval_reliability':True}
    return bound


def write_retrieval_comparison(plan_path):
    try:
        from scripts.agentic_dev30.retrieval_reliability import write_comparison
    except ModuleNotFoundError as error:
        if error.name not in {'scripts', 'scripts.agentic_dev30'}:
            raise
        from retrieval_reliability import write_comparison
    return write_comparison(plan_path)


def execute(plan_path, authorization_path, node_path, key_file, port=0, *,
            dense_startup_timeout_seconds=None):
    """授权后才导入本地服务/加载检索后端；每题创建独立原硬限额账本。"""
    plan_path = Path(plan_path).resolve()
    plan = load(plan_path)
    cap = require_authorization(load(authorization_path), file_sha(plan_path), plan)
    validate_dense_startup_timeout_seconds(dense_startup_timeout_seconds)
    frozen_startup_timeout = plan.get('dense_startup_timeout_seconds')
    if 'dense_startup_timeout_seconds' in plan and frozen_startup_timeout is None:
        raise ValueError('INVALID_DENSE_STARTUP_TIMEOUT')
    validate_dense_startup_timeout_seconds(frozen_startup_timeout)
    if (dense_startup_timeout_seconds is not None
            and dense_startup_timeout_seconds != frozen_startup_timeout):
        raise ValueError('DENSE_STARTUP_TIMEOUT_BINDING_MISMATCH')
    if plan.get('resilience_policy') != RESILIENCE_POLICY:
        raise ValueError('RESILIENCE_POLICY_MISMATCH')
    format_repair_limits(plan)
    description = plan.get('evaluation_runtime_description')
    validate_evaluation_description(description)
    if plan.get('evaluation_protocol_sha256') != description['evaluation_protocol_sha256']:
        raise ValueError('EVALUATION_PROTOCOL_MISMATCH')
    verify_files(plan['frozen_files'])
    project = Path(plan['project_root'])
    prefetch_module = paired_prefetch_module()
    paired = prefetch_module.validate_plan(plan, project)
    validate_evaluation_description(description, project)
    snapshot_manifest = load(Path(plan['snapshot_dir'])/'manifest.json')
    if (snapshot_manifest.get('snapshot_id') != plan['snapshot_id']
            or plan.get('retrieval_binding') != snapshot_rerank_binding(snapshot_manifest, project)):
        raise ValueError('RERANK_SNAPSHOT_BINDING_MISMATCH')
    if 'snapshot_mode' in plan:
        prefetch_module.validate_snapshot(Path(plan['snapshot_dir']), plan['snapshot_mode'], project)
    if not Path(node_path).is_file() or not Path(key_file).is_file():
        raise ValueError('LOCAL_RUNTIME_OR_KEY_FILE_MISSING')
    output = plan_path.parent/'execution'
    if output.exists():
        raise ValueError('ALREADY_STARTED:read existing evidence; do not rerun')
    output.mkdir()
    budget = RoundBudget(output/'round-ledger.json', file_sha(plan_path), cap)
    circuit = RoundCircuit(output/'round-health.json')
    save(output/'authorization.json', load(authorization_path))
    sys.path.insert(0, str(project/'src'))
    # 只有越过授权与冻结检查才能到此；prepare不导入HTTP/模型组件。
    import uvicorn
    from agent_rag import router
    from agent_rag.task_service import TaskService, atomic_json
    from agent_rag.v2_runtime import load_v2_snapshot
    service = None
    snapshot = None
    sock = None
    server = None
    thread = None
    reports = {}
    timings = {}
    attempts = {}
    old_env = dict(os.environ)
    try:
        token = secrets.token_urlsafe(32)
        os.environ.update(FINRISK_AGENT_SERVICE_TOKEN=token, FINRISK_AGENT_ALLOW_LIVE='1',
                          FINRISK_AGENT_ALLOW_EVALUATION='1', FINRISK_AGENT_ALLOW_RETRIEVAL_LIVE='1',
                          FINRISK_AGENT_RERANK_CAP_CNY='0.5', FINRISK_AGENT_RERANK_MAX_REQUESTS='240',
                          FINRISK_AGENT_DEEPSEEK_KEY_FILE=str(Path(key_file).resolve()))
        os.environ['DEEPSEEK_API_KEY_FILE']=str(Path(key_file).resolve())
        dense_kwargs = ({} if frozen_startup_timeout is None
            else {'startup_timeout_seconds':frozen_startup_timeout})
        snapshot = load_v2_snapshot(Path(plan['snapshot_dir']), **dense_kwargs)
        service = TaskService(output/'tasks', snapshot)
        shared = prefetch_module.SharedPrefetch(snapshot, output, budget, file_sha(plan_path)) if paired else None
        app = local_tool_app(service)
        sock = socket.socket()
        sock.bind(('127.0.0.1', port))
        sock.listen(128)
        actual_port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level='warning'))
        thread = threading.Thread(target=server.run, kwargs={'sockets':[sock]}, daemon=True)
        thread.start()
        deadline = time.monotonic()+20
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise ValueError('LOCAL_TOOL_SERVER_NOT_READY')
            time.sleep(0.05)
        for task in plan['tasks']:
            key = task['task_key']
            circuit.before_task()
            shared_row = shared.ensure(task['input']) if shared is not None else None
            budget.reserve(key)
            model_ledger = output/'model-ledgers'/(key.replace(':','_')+'.json')
            os.environ['FINRISK_AGENT_TRIAL_LEDGER'] = str(model_ledger)
            record, reused = service.admit({
                'text':task['input']['text'], 'user_context':task['input']['user_context'],
                'engine':'fixed_rag' if task['arm']=='R2' else 'controlled_agent',
                'experiment_profile_id':'live-deepseek-flash-v2-v1'}, key)
            if reused:
                raise ValueError('UNEXPECTED_TASK_REUSE')
            directory = service.root/record['task_id']
            bundle = service.bundle(record['task_id'])
            bundle['task_dir'] = str(directory)
            bundle = bind_evaluation_contract(bundle, plan['evaluation_runtime_description'], project)
            bundle = bind_format_repair_config(bundle, task['arm'], plan)
            save(directory/'evaluation-bundle.json', bundle)
            env = dict(os.environ, FINRISK_AGENT_TRIAL_LEDGER=str(model_ledger))
            started = time.monotonic()
            attempts[key]={'status':'failed','state':'started','task_id':record['task_id']}
            save(output/'current-task.json', {'task_key':key,'task_id':record['task_id'], 'state':'starting',
                                              'input_sha256':bundle['input']['input_sha256']})
            run_error=None
            returncode=None
            try:
                with (directory/'cli.stdout').open('wb') as stdout, (directory/'cli.stderr').open('wb') as stderr:
                    completed = subprocess.run([str(node_path),str(project/'agent_harness/dist/src/evaluation_cli.js'),
                        '--bundle',str(directory/'evaluation-bundle.json'),'--arm',task['arm'],
                        '--base-url',f'http://127.0.0.1:{actual_port}'], env=env, stdout=stdout, stderr=stderr,
                        timeout=300, check=False)
                    returncode=completed.returncode
            except (subprocess.TimeoutExpired, OSError, KeyboardInterrupt) as error:
                run_error=error
            finally:
                timings[key]=time.monotonic()-started
                attempts[key].update(state='returned' if run_error is None else 'interrupted',
                                     error=type(run_error).__name__ if run_error else None)
                attempt_timing = {'wall_seconds':timings[key]}
                if shared_row is not None:
                    attempt_timing.update(shared_prefetch_seconds=shared_row['elapsed_seconds'],
                        actual_task_seconds=timings[key],
                        comparable_e2e_seconds=shared_row['elapsed_seconds'] + timings[key] if task['arm'] in {'R1','A'} else None,
                        timing_interpretation='composed estimate, not measured cold start' if task['arm'] in {'R1','A'} else 'R2 diagnostic only')
                save(output/(key.replace(':','_')+'.attempt.json'),attempts[key] | attempt_timing)
            try:
                service.cancel_retrieval(record['task_id'])
            except Exception as error:
                run_error=run_error or error
            # 未知用量保留；报告和原账本永不按“失败”清零。
            try:
                known, unknown = read_costs(model_ledger, directory/'rerank-ledger.json')
            except (OSError,ValueError):
                known, unknown = 0,TASK_RESERVE_MICRO
            budget.finish(key, known, unknown)
            report_path = directory/'report.json'
            if report_path.exists():
                report = load(report_path)
                if (report.get('simulated') or report.get('execution_mode') != 'live'
                    or report.get('versions',{}).get('input_sha256') != bundle['input']['input_sha256']
                    or report.get('versions',{}).get('snapshot_id') != plan['snapshot_id']
                    or report.get('task_id') != record['task_id']):
                    raise ValueError('LIVE_REPORT_IDENTITY_MISMATCH')
                reports[key] = report
            result = {'task_key':key,'task_id':record['task_id'],'returncode':returncode,
                      'wall_seconds':timings[key],'known_micro_cny':known,'unknown_micro_cny':unknown,
                      'report':reports.get(key)}
            if shared_row is not None:
                result.update(prefetch_module.task_result_fields(task['arm'], result, shared_row))
            timings[key]=result['wall_seconds']
            save(output/(key.replace(':','_')+'.json'), result)
            service.records[record['task_id']]['task_status'] = (reports.get(key) or {}).get('task_status','failed')
            atomic_json(directory/'admission.json',service.records[record['task_id']])
            service.active_task_id = None
            events = lines(directory/'events.jsonl') if (directory/'events.jsonl').exists() else []
            circuit.observe(key,reports.get(key),events)
            save(output/'current-task.json', {'task_key':key,'task_id':record['task_id'],'state':'finished',
                                              'task_status':service.records[record['task_id']]['task_status']})
            save(output/'automatic-comparison.json' ,score_results(plan['tasks'],lines(plan['reference_file']),reports,timings,attempts))
            write_retrieval_comparison(plan_path)
            if run_error is not None:
                raise run_error
            if returncode != 0:
                raise ValueError('TASK_FAILED_STOP:no automatic replay')
            circuit.before_task()
        save(output/'status.json',{'status':'finished','tasks':len(reports)})
    except BaseException as error:
        save(output/'automatic-comparison.json',score_results(plan['tasks'],lines(plan['reference_file']),reports,timings,attempts))
        try:
            write_retrieval_comparison(plan_path)
        except Exception as diagnostic_error:
            save(output/'retrieval-reliability-error.json', {'status':'failed',
                'reason':type(diagnostic_error).__name__+':'+str(diagnostic_error)[:200]})
        save(output/'status.json',{'status':'stopped','reason':type(error).__name__+':'+str(error)[:200],
                                  'restart_policy':'read_only_no_automatic_resume'})
        raise
    finally:
        errors=cleanup_resources(service,snapshot,server,thread,sock,old_env)
        if errors:
            save(output/'cleanup-errors.json',errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[2])
    prep.add_argument('--reference-dir',type=Path,required=True)
    prep.add_argument('--snapshot',type=Path,required=True)
    prep.add_argument('--output',type=Path,required=True)
    selection=prep.add_mutually_exclusive_group()
    selection.add_argument('--full',action='store_true')
    selection.add_argument('--review-ids',nargs='+')
    prep.add_argument('--node',type=Path,required=True)
    prep.add_argument('--adaptive-format-repair-limit', type=int, choices=(0, 1), default=0)
    prep.add_argument('--snapshot-mode', choices=('full9639', 'incremental-v1'), default='full9639')
    prep.add_argument('--paired-prefetch', action='store_true')
    prep.add_argument('--dense-startup-timeout-seconds', type=float)
    run = commands.add_parser('execute')
    run.add_argument('--plan',type=Path,required=True)
    run.add_argument('--authorization',type=Path,required=True)
    run.add_argument('--node',type=Path,required=True)
    run.add_argument('--key-file',type=Path,required=True)
    run.add_argument('--dense-startup-timeout-seconds', type=float)
    args=parser.parse_args()
    if args.command=='prepare':
        plan=prepare(args.project,args.reference_dir,args.snapshot,args.output,args.node,not args.full,args.review_ids,
                     args.adaptive_format_repair_limit,args.snapshot_mode,args.paired_prefetch,
                     args.dense_startup_timeout_seconds)
        print(json.dumps({'status':'offline_prepared','tasks':plan['task_count'],'paid_api_calls':0}))
    else:
        execute(args.plan,args.authorization,args.node,args.key_file,
                dense_startup_timeout_seconds=args.dense_startup_timeout_seconds)


if __name__=='__main__':
    main()
