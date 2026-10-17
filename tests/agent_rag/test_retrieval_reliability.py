"""Read-only statistics must distinguish bound evidence from missing evidence."""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[2]


def module():
    path = PROJECT/'scripts/agentic_dev30/retrieval_reliability.py'
    assert path.is_file(), 'independent read-only retrieval reliability statistics are required'
    spec = importlib.util.spec_from_file_location('retrieval_reliability_test_module', path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':')).encode('utf-8')).hexdigest()


def frozen_plan(tmp_path, count=1):
    tasks = [{'task_key':f'S{i}:A','arm':'A','input':{'sample_id':f'S{i}',
        'text':f'待审原文{i}','user_context':{}}} for i in range(count)]
    plan = {'schema_version':'finrisk-three-arm-experiment/1.0','tasks':tasks,
        'snapshot_id':'frozen-snapshot','evaluation_protocol_sha256':'a'*64,
        'evaluation_runtime_description':{'schema_version':'finrisk-evaluation/2.4',
            'evaluation_protocol_sha256':'a'*64,
            'retrieval_reliability':{'coverage_version':'finrisk-ranking-coverage/1.0',
                'schema_version':'finrisk-retrieval-assessment/1.0','pair_attempt_limit':2,
                'rerank_protocols':['quote-copy-v1','span-ref-v2'],'configured_in_v2':True}},
        'retrieval_binding':{'rerank_protocol':'span-ref-v2','rules_sha256':'b'*64}}
    path = tmp_path/'plan.json'
    save(path, plan)
    return path, plan


def coverage(failed=0):
    return {'schema_version':'finrisk-ranking-coverage/1.0','expected_units':2,
        'scored_units':2-failed,'failed_units':failed,'ranking_complete':failed == 0,
        'topk_scope':'scored_subset' if failed else 'full_pool',
        'attempts_exhausted':bool(failed),'failure_counts':{'QUOTE_MISMATCH':failed} if failed else {},
        'failed_unit_ids':['U2'] if failed else [],'expected_set_sha256':'c'*64,
        'scored_set_sha256':'d'*64,'failed_set_sha256':'e'*64,'rerank_protocol':'span-ref-v2'}


def task_evidence(plan_path, plan, index=0, *, failed=0, assessment_status=None):
    task = plan['tasks'][index]
    task_id = f'{index+1:032x}'
    execution = plan_path.parent/'execution'
    directory = execution/'tasks'/task_id
    digest = canonical_hash({'text':task['input']['text'],'user_context':task['input']['user_context']})
    status = assessment_status or ('model_assessed_with_retrieval_limits' if failed else 'model_assessed')
    report = {'task_id':task_id,'task_status':'partial' if failed or status == 'blocked' else 'completed',
        'execution_mode':'live','versions':{'input_sha256':digest,'snapshot_id':plan['snapshot_id'],
            'evaluation_protocol_sha256':plan['evaluation_protocol_sha256'],'evaluation_arm':task['arm']},
        'risk_points':[{'risk_point_id':'P1','case_match_status':'partial_match' if failed else 'reliable_match'}],
        'resources':{},'needs_manual_review':True}
    save(directory/'report.json', report)
    assessment = {'schema_version':'finrisk-retrieval-assessment/1.0','task_id':task_id,
        'input_sha256':digest,'snapshot_id':plan['snapshot_id'],'evaluation_protocol_sha256':'a'*64,
        'report_sha256':hashlib.sha256((directory/'report.json').read_bytes()).hexdigest(),
        'retrieval_status':'partial' if failed else 'completed',
        'queries':[{'query_id':'Q1','complete':not failed,'status':'partial' if failed else 'ok',
            'ranking_coverage':coverage(failed)}],
        'points':[{'risk_point_id':'P1','assessment_status':status,
            'case_match_status':report['risk_points'][0]['case_match_status'],'needs_manual_review':True}],
        'human_semantic_review_status':'awaiting_human_review'}
    save(directory/'retrieval-assessment.json', assessment)
    save(execution/(task['task_key'].replace(':','_')+'.attempt.json'),
        {'task_id':task_id,'status':'failed','state':'returned','wall_seconds':4.25})
    save(execution/'model-ledgers'/(task['task_key'].replace(':','_')+'.json'),
        {'settled_cost_micro_cny':100,'reserved_cost_micro_cny':200})
    save(directory/'rerank-ledger.json', {'settled_cost_micro_cny':10,
        'reserved_cost_micro_cny':30,'request_reservations':[]})
    return directory, report, assessment


def attempt_rows(directory):
    rows = [
        {'ordinal':1,'pair_id':'pair1','retry_index':0,'state':'settled',
            'transport_started_at':'2026-10-17T00:00:00Z','completed_at':'2026-10-17T00:00:01Z',
            'error_code':'QUOTE_MISMATCH','will_retry':True,'usage_state':'known','http_status':200},
        {'ordinal':2,'pair_id':'pair1','retry_index':1,'state':'settled',
            'transport_started_at':'2026-10-17T00:00:02Z','completed_at':'2026-10-17T00:00:03Z',
            'error_code':None,'will_retry':False,'usage_state':'known','http_status':200},
        {'ordinal':3,'pair_id':'pair2','retry_index':0,'state':'aborted_unsent',
            'transport_started_at':None,'completed_at':'2026-10-17T00:00:03Z',
            'error_code':'ABORTED_UNSENT','will_retry':False,'usage_state':'not_sent','http_status':None}]
    save(directory/'rerank-ledger.json.attempts.json',
        {'schema_version':'finrisk-rerank-attempts/1.0','attempts':rows})
    ledger = json.loads((directory/'rerank-ledger.json').read_text(encoding='utf-8'))
    ledger['request_reservations'] = [{'pair_id':r['pair_id'],
        **({'transport_started_at':r['transport_started_at']} if r['transport_started_at'] else {})} for r in rows]
    save(directory/'rerank-ledger.json', ledger)
    return rows


def test_retry_success_still_counts_the_failed_physical_attempt(tmp_path):
    m = module()
    plan_path, plan = frozen_plan(tmp_path)
    directory, _, _ = task_evidence(plan_path, plan)
    attempt_rows(directory)
    out = m.build_comparison(plan_path)
    assert out['planned_tasks'] == out['ranking_complete_tasks'] == 1
    assert out['rerank_physical_requests'] == 2
    assert out['rerank_failures_by_code'] == {'QUOTE_MISMATCH':1}
    assert out['duplicate_paid_dispatches_prevented'] is None
    assert out['latency_seconds'] == 4.25
    assert out['known_cost_estimate_cny'] == pytest.approx(0.00011)
    assert out['unknown_reserve_cny'] == pytest.approx(0.00023)
    assert out['human_semantic_review_status'] == 'awaiting_human_review'
    assert out['not_account_statement'] is True


def test_partial_ranking_and_blocked_are_separate_from_success_and_missing(tmp_path):
    m = module()
    plan_path, plan = frozen_plan(tmp_path, 3)
    task_evidence(plan_path, plan, failed=1)
    task_evidence(plan_path, plan, 1, assessment_status='blocked')
    out = m.build_comparison(plan_path)
    assert out['planned_tasks'] == out['arms']['A']['planned_tasks'] == 3
    assert out['ranking_complete_tasks'] is None
    assert out['model_assessed_with_limits_tasks'] is None
    assert out['blocked_tasks'] is None
    assert out['items'][0]['model_assessed_with_limits_tasks'] == 1
    assert out['items'][1]['blocked_tasks'] == 1
    assert out['items'][2]['evidence_status'] == 'missing'
    assert out['items'][2]['rerank_physical_requests'] is None
    assert out['items'][2]['known_cost_estimate_cny'] is None


@pytest.mark.parametrize('field,value', [('task_id','wrong'),('input_sha256','f'*64),
    ('snapshot_id','wrong'),('evaluation_protocol_sha256',None),('report_sha256','f'*64)])
def test_assessment_identity_and_report_bytes_must_match(tmp_path, field, value):
    m = module()
    plan_path, plan = frozen_plan(tmp_path)
    directory, _, assessment = task_evidence(plan_path, plan)
    assessment[field] = value
    save(directory/'retrieval-assessment.json', assessment)
    with pytest.raises(ValueError, match='RETRIEVAL_.*MISMATCH'):
        m.build_comparison(plan_path)


def test_report_reserialization_cannot_preserve_a_stale_sidecar_hash(tmp_path):
    m = module()
    plan_path, plan = frozen_plan(tmp_path)
    directory, _, _ = task_evidence(plan_path, plan)
    with (directory/'report.json').open('ab') as stream:
        stream.write(b'\n')
    with pytest.raises(ValueError, match='RETRIEVAL_.*MISMATCH'):
        m.build_comparison(plan_path)


@pytest.mark.parametrize('change', [
    {'scored_units':3}, {'ranking_complete':False}, {'topk_scope':'scored_subset'},
    {'failure_counts':{'QUOTE_MISMATCH':1}}, {'failed_unit_ids':['UNSCORED']},
    {'rerank_protocol':'quote-copy-v1'}, {'expected_set_sha256':'bad'},
])
def test_malformed_or_unbound_ranking_coverage_is_rejected(tmp_path, change):
    m = module()
    plan_path, plan = frozen_plan(tmp_path)
    directory, _, assessment = task_evidence(plan_path, plan)
    assessment['queries'][0]['ranking_coverage'].update(change)
    save(directory/'retrieval-assessment.json', assessment)
    with pytest.raises(ValueError, match='RETRIEVAL_.*(MISMATCH|INVALID)'):
        m.build_comparison(plan_path)


def test_missing_audit_does_not_reconstruct_attempt_failures_from_final_pair_errors(tmp_path):
    m = module()
    plan_path, plan = frozen_plan(tmp_path)
    task_evidence(plan_path, plan, failed=1)
    out = m.build_comparison(plan_path)
    assert out['ranking_complete_tasks'] == 0
    assert out['model_assessed_with_limits_tasks'] == 1
    assert out['rerank_physical_requests'] is None
    assert out['rerank_failures_by_code'] is None
    assert out['duplicate_paid_dispatches_prevented'] is None


@pytest.mark.parametrize('change', ['wrong_arm','duplicate_report_point','query_without_coverage',
    'limited_without_partial_ranking','completed_with_limits'])
def test_sidecar_cannot_relabel_an_unbound_or_inconsistent_report(tmp_path, change):
    m = module()
    plan_path, plan = frozen_plan(tmp_path)
    directory, report, assessment = task_evidence(plan_path, plan, failed=1)
    if change == 'wrong_arm':
        report['versions']['evaluation_arm'] = 'R1'
    elif change == 'duplicate_report_point':
        report['risk_points'].append(dict(report['risk_points'][0]))
    elif change == 'query_without_coverage':
        assessment['queries'][0]['ranking_coverage'] = None
    elif change == 'limited_without_partial_ranking':
        assessment['queries'][0].update(complete=True,status='ok',ranking_coverage=coverage())
        assessment['retrieval_status'] = 'completed'
    else:
        report['task_status'] = 'completed'
    save(directory/'report.json', report)
    assessment['report_sha256'] = hashlib.sha256((directory/'report.json').read_bytes()).hexdigest()
    save(directory/'retrieval-assessment.json', assessment)
    with pytest.raises(ValueError, match='RETRIEVAL_.*(INVALID|MISMATCH)'):
        m.build_comparison(plan_path)


def test_read_only_statistics_do_not_release_unproven_cost_reservations(tmp_path):
    m = module()
    plan_path, plan = frozen_plan(tmp_path)
    directory, _, _ = task_evidence(plan_path, plan)
    (directory/'rerank-ledger.json').unlink()
    out = m.build_comparison(plan_path)
    assert out['known_cost_estimate_cny'] == pytest.approx(0.0001)
    assert out['unknown_reserve_cny'] == pytest.approx(0.5002)
    assert out['rerank_physical_requests'] is None


def test_ordinal_pair_binding_rejects_misattached_attempt_audit(tmp_path):
    m = module()
    plan_path, plan = frozen_plan(tmp_path)
    directory, _, _ = task_evidence(plan_path, plan)
    rows = attempt_rows(directory)
    rows[0]['pair_id'] = 'different-pair'
    save(directory/'rerank-ledger.json.attempts.json',
        {'schema_version':'finrisk-rerank-attempts/1.0','attempts':rows})
    with pytest.raises(ValueError, match='RETRIEVAL_ATTEMPT_BINDING_MISMATCH'):
        m.build_comparison(plan_path)


@pytest.mark.parametrize('without_points', [False, True])
def test_exhausted_global_or_zero_scored_retrieval_is_a_known_blocked_task(tmp_path, without_points):
    m = module()
    plan_path, plan = frozen_plan(tmp_path)
    directory, report, assessment = task_evidence(plan_path, plan, failed=1, assessment_status='blocked')
    assessment['retrieval_status'] = 'error'
    c = assessment['queries'][0]['ranking_coverage']
    c.update(scored_units=0,failed_units=2,failure_counts={'BUDGET_EXHAUSTED':2},failed_unit_ids=['U1','U2'])
    if without_points:
        report['risk_points'] = []
        assessment['points'] = []
    save(directory/'report.json', report)
    assessment['report_sha256'] = hashlib.sha256((directory/'report.json').read_bytes()).hexdigest()
    save(directory/'retrieval-assessment.json', assessment)
    out = m.build_comparison(plan_path)
    assert out['ranking_complete_tasks'] == 0
    assert out['blocked_tasks'] == 1
    assert out['model_assessed_with_limits_tasks'] == 0


def test_failed_retrieval_without_ranking_coverage_is_incomplete_not_unrun(tmp_path):
    m = module()
    plan_path, plan = frozen_plan(tmp_path)
    directory, _, assessment = task_evidence(plan_path, plan, failed=1, assessment_status='blocked')
    assessment['retrieval_status'] = 'error'
    assessment['queries'][0].update(status='error',complete=False,ranking_coverage=None)
    save(directory/'retrieval-assessment.json', assessment)
    out = m.build_comparison(plan_path)
    assert out['ranking_complete_tasks'] == 0
    assert out['blocked_tasks'] == 1


@pytest.mark.parametrize('point_status', ['model_assessed','needs_input','not_assessed',
    'model_assessed_with_retrieval_limits','blocked'])
def test_completed_report_requires_completed_retrieval_and_assessed_points(tmp_path, point_status):
    m = module()
    plan_path, plan = frozen_plan(tmp_path)
    directory, report, assessment = task_evidence(plan_path, plan,
        failed=1 if point_status == 'model_assessed' else 0, assessment_status=point_status)
    report['task_status'] = 'completed'
    save(directory/'report.json', report)
    assessment['report_sha256'] = hashlib.sha256((directory/'report.json').read_bytes()).hexdigest()
    save(directory/'retrieval-assessment.json', assessment)
    with pytest.raises(ValueError, match='RETRIEVAL_.*MISMATCH'):
        m.build_comparison(plan_path)


def completed_needs_input_evidence(tmp_path, judgment='suspected'):
    plan_path, plan = frozen_plan(tmp_path)
    directory, report, assessment = task_evidence(plan_path, plan, assessment_status='needs_input')
    report.update(overall_judgment=judgment,
        coverage={'all_submitted_segments_reviewed':True},
        validation={'schema_valid':True,'references_valid':True,'errors':[]})
    report['risk_points'][0].update(judgment=judgment,status='review_required',
        pending_tags=['合同外利益'],tag_checks=[{'tag':'合同外利益','status':'unknown',
            'case_match_status':'no_reliable_match',
            'conditions':[{'condition_id':'contract_outside','critical':True,'status':'unknown'}]}])
    return plan_path, plan, directory, report, assessment


def save_bound_assessment(directory, report, assessment):
    save(directory/'report.json', report)
    assessment['report_sha256'] = hashlib.sha256((directory/'report.json').read_bytes()).hexdigest()
    save(directory/'retrieval-assessment.json', assessment)


@pytest.mark.parametrize('judgment', ['suspected','review_required'])
def test_completed_retrieval_can_preserve_bound_unknown_current_conditions(tmp_path, judgment):
    m = module()
    plan_path, _, directory, report, assessment = completed_needs_input_evidence(tmp_path, judgment)
    save_bound_assessment(directory, report, assessment)
    out = m.build_comparison(plan_path)
    assert out['ranking_complete_tasks'] == 1
    assert out['blocked_tasks'] == out['model_assessed_with_limits_tasks'] == 0
    assert out['items'][0]['evidence_status'] == 'verified'
    assert out['human_semantic_review_status'] == 'awaiting_human_review'


@pytest.mark.parametrize('damage', ['no_unknown_tag','no_unknown_condition',
    'unknown_only_on_supported_tag','missing_tag_checks','missing_conditions',
    'manual_review_false','pending_tag_missing','bad_references','bad_schema',
    'validation_errors','incomplete_coverage','no_clear_risk','point_not_completed',
    'tag_not_completed','forged_model_assessed','not_assessed','blocked'])
def test_completed_needs_input_requires_report_bound_current_fact_gap(tmp_path, damage):
    m = module()
    plan_path, _, directory, report, assessment = completed_needs_input_evidence(tmp_path)
    point = report['risk_points'][0]
    tag = point['tag_checks'][0]
    if damage == 'no_unknown_tag':
        tag['status'] = 'not_supported'
    elif damage == 'no_unknown_condition':
        tag['conditions'][0]['status'] = 'met'
    elif damage == 'unknown_only_on_supported_tag':
        tag['conditions'][0]['status'] = 'met'
        point['tag_checks'].append({'tag':'赠送利益','status':'supported',
            'conditions':[{'status':'unknown'}]})
    elif damage == 'missing_tag_checks':
        del point['tag_checks']
    elif damage == 'missing_conditions':
        del tag['conditions']
    elif damage == 'manual_review_false':
        report['needs_manual_review'] = assessment['points'][0]['needs_manual_review'] = False
    elif damage == 'pending_tag_missing':
        point['pending_tags'] = []
    elif damage == 'bad_references':
        report['validation']['references_valid'] = False
    elif damage == 'bad_schema':
        report['validation']['schema_valid'] = False
    elif damage == 'validation_errors':
        report['validation']['errors'] = ['SOURCE_VERIFICATION_FAILED']
    elif damage == 'incomplete_coverage':
        report['coverage']['all_submitted_segments_reviewed'] = False
    elif damage == 'no_clear_risk':
        report['overall_judgment'] = 'no_clear_risk'
    elif damage == 'point_not_completed':
        point['case_match_status'] = assessment['points'][0]['case_match_status'] = 'not_completed'
    elif damage == 'tag_not_completed':
        tag['case_match_status'] = 'not_completed'
    else:
        assessment['points'][0]['assessment_status'] = (
            'model_assessed' if damage == 'forged_model_assessed' else damage)
    save_bound_assessment(directory, report, assessment)
    with pytest.raises(ValueError, match='RETRIEVAL_COMPLETED_STATUS_MISMATCH'):
        m.build_comparison(plan_path)


def test_completed_report_cannot_relabel_an_unrun_retrieval(tmp_path):
    m = module()
    plan_path, plan = frozen_plan(tmp_path)
    directory, _, assessment = task_evidence(plan_path, plan)
    assessment.update(queries=[],retrieval_status='not_run')
    save(directory/'retrieval-assessment.json', assessment)
    with pytest.raises(ValueError, match='RETRIEVAL_.*MISMATCH'):
        m.build_comparison(plan_path)


@pytest.mark.parametrize('code', ['CANDIDATE_EVIDENCE_EXCEEDS_CONTEXT','RETRIEVAL_AUDIT_WRITE_FAILED'])
@pytest.mark.parametrize('query_status', ['partial','error'])
def test_completed_ranking_with_explicit_post_ranking_failure_stays_blocked(tmp_path, code, query_status):
    m = module()
    plan_path, plan = frozen_plan(tmp_path)
    directory, _, assessment = task_evidence(plan_path, plan, assessment_status='blocked')
    assessment['retrieval_status'] = 'error'
    assessment['queries'][0].update(complete=False,status=query_status,error_code=code)
    save(directory/'retrieval-assessment.json', assessment)
    out = m.build_comparison(plan_path)
    assert out['ranking_complete_tasks'] == 1
    assert out['blocked_tasks'] == 1
    assert out['model_assessed_with_limits_tasks'] == 0


@pytest.mark.parametrize('complete,status,code', [
    (False,'partial',None), (False,'error',None), (False,'partial','UNCLASSIFIED_FAILURE'),
    (False,'error','TRANSPORT_TIMEOUT'), (True,'ok','RETRIEVAL_AUDIT_WRITE_FAILED'),
    (True,'error','CANDIDATE_EVIDENCE_EXCEEDS_CONTEXT'),
])
def test_post_ranking_exception_requires_an_explicit_allowed_failure(complete, status, code):
    m = module()
    query = {'query_id':'Q1','complete':complete,'status':status,'ranking_coverage':coverage(),
        'error_code':code}
    with pytest.raises(ValueError, match='RETRIEVAL_COVERAGE_INVALID'):
        m._coverage(query, 'span-ref-v2')


def test_two_planned_tasks_cannot_count_the_same_real_task_twice(tmp_path):
    m = module()
    plan_path, plan = frozen_plan(tmp_path, 2)
    plan['tasks'][1]['input']['text'] = plan['tasks'][0]['input']['text']
    save(plan_path, plan)
    directory, _, _ = task_evidence(plan_path, plan)
    attempt_rows(directory)
    first = plan_path.parent/'execution/S0_A.attempt.json'
    (first.parent/'S1_A.attempt.json').write_bytes(first.read_bytes())
    with pytest.raises(ValueError, match='RETRIEVAL_TASK_IDENTITY_MISMATCH'):
        m.build_comparison(plan_path)


def test_scheduler_result_task_key_must_match_the_planned_slot(tmp_path):
    m = module()
    plan_path, plan = frozen_plan(tmp_path)
    directory, _, _ = task_evidence(plan_path, plan)
    save(plan_path.parent/'execution/S0_A.json', {'task_key':'wrong:A','task_id':directory.name})
    with pytest.raises(ValueError, match='RETRIEVAL_TASK_IDENTITY_MISMATCH'):
        m.build_comparison(plan_path)


@pytest.mark.parametrize('code', ['RETRIEVAL_AUDIT_WRITE_FAILED',
    'CANDIDATE_EVIDENCE_EXCEEDS_CONTEXT', 'UNCLASSIFIED_FAILURE'])
def test_limited_gate_rejects_a_pending_query_with_any_explicit_error(tmp_path, code):
    m = module()
    plan_path, plan = frozen_plan(tmp_path)
    directory, _, assessment = task_evidence(plan_path, plan, failed=1)
    assessment['queries'][0]['error_code'] = code
    save(directory/'retrieval-assessment.json', assessment)
    with pytest.raises(ValueError, match='RETRIEVAL_LIMITED_STATUS_MISMATCH'):
        m.build_comparison(plan_path)


@pytest.mark.parametrize('code', ['STOPPED', 'CP', 'BUDGET_EXHAUSTED',
    'RETRIEVAL_TRANSPORT_CIRCUIT_OPEN'])
def test_limited_gate_rejects_global_or_checkpoint_failure_codes(tmp_path, code):
    m = module()
    plan_path, plan = frozen_plan(tmp_path)
    directory, _, assessment = task_evidence(plan_path, plan, failed=1)
    assessment['queries'][0]['ranking_coverage']['failure_counts'] = {code:1}
    save(directory/'retrieval-assessment.json', assessment)
    with pytest.raises(ValueError, match='RETRIEVAL_LIMITED_STATUS_MISMATCH'):
        m.build_comparison(plan_path)


@pytest.mark.parametrize('code', ['QUOTE_MISMATCH','SPAN_REFERENCE_MISMATCH','INVALID_JSON',
    'INVALID_COMPLETION','TRANSPORT_TIMEOUT','TRANSPORT_ERROR','HTTP_429','HTTP_5XX'])
def test_limited_gate_accepts_exactly_the_node_local_failure_allowlist(tmp_path, code):
    m = module()
    plan_path, plan = frozen_plan(tmp_path)
    directory, _, assessment = task_evidence(plan_path, plan, failed=1)
    assessment['queries'][0]['ranking_coverage']['failure_counts'] = {code:1}
    assessment['queries'][0]['error_code'] = None
    save(directory/'retrieval-assessment.json', assessment)
    out = m.build_comparison(plan_path)
    assert out['model_assessed_with_limits_tasks'] == 1
    assert out['blocked_tasks'] == 0


def test_limited_gate_preserves_an_explicitly_blocked_audit_failure(tmp_path):
    m = module()
    plan_path, plan = frozen_plan(tmp_path)
    directory, _, assessment = task_evidence(plan_path, plan, failed=1, assessment_status='blocked')
    assessment['retrieval_status'] = 'error'
    assessment['queries'][0]['error_code'] = 'RETRIEVAL_AUDIT_WRITE_FAILED'
    save(directory/'retrieval-assessment.json', assessment)
    out = m.build_comparison(plan_path)
    assert out['model_assessed_with_limits_tasks'] == 0
    assert out['blocked_tasks'] == 1


def test_prevented_duplicate_count_requires_explicit_tool_event_evidence(tmp_path):
    m = module()
    plan_path, plan = frozen_plan(tmp_path)
    directory, _, _ = task_evidence(plan_path, plan)
    event = {'sequence':1,'type':'tool.result','data':{'name':'search_cases',
        'response':{'meta':{'snapshot_id':plan['snapshot_id'],
            'duplicate_paid_dispatches_prevented':2}}}}
    (directory/'events.jsonl').write_text(json.dumps(event)+'\n', encoding='utf-8')
    assert m.build_comparison(plan_path)['duplicate_paid_dispatches_prevented'] == 2


def test_unfinished_attempt_and_unaudited_dispatch_are_not_claimed_as_zero_failures(tmp_path):
    m = module()
    plan_path, plan = frozen_plan(tmp_path)
    directory, _, _ = task_evidence(plan_path, plan)
    rows = attempt_rows(directory)
    rows[1].update(state='transport_started',completed_at=None,usage_state='unknown')
    save(directory/'rerank-ledger.json.attempts.json',
        {'schema_version':'finrisk-rerank-attempts/1.0','attempts':rows})
    out = m.build_comparison(plan_path)
    assert out['rerank_physical_requests'] == 2
    assert out['rerank_failures_by_code'] is None
    rows.pop(1)
    save(directory/'rerank-ledger.json.attempts.json',
        {'schema_version':'finrisk-rerank-attempts/1.0','attempts':rows})
    assert m.build_comparison(plan_path)['rerank_physical_requests'] is None


def test_statistics_write_only_the_independent_output_and_reject_v22_plan(tmp_path):
    m = module()
    plan_path, plan = frozen_plan(tmp_path)
    directory, _, _ = task_evidence(plan_path, plan)
    old = plan_path.parent/'execution/automatic-comparison.json'
    old.write_bytes(b'original automatic comparison stays byte-identical')
    before = {p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    target = m.write_comparison(plan_path)
    assert target == plan_path.parent/'execution/retrieval-reliability-comparison.json'
    assert json.loads(target.read_text(encoding='utf-8'))['planned_tasks'] == 1
    assert all(p.read_bytes() == data for p, data in before.items())
    completed = subprocess.run([sys.executable, str(PROJECT/'scripts/agentic_dev30/retrieval_reliability.py'),
        '--plan', str(plan_path)], cwd=tmp_path, capture_output=True, text=True, timeout=20, check=False)
    assert completed.returncode == 0, completed.stderr
    assert str(target) in completed.stdout
    assert all(p.read_bytes() == data for p, data in before.items())
    plan['evaluation_runtime_description']['schema_version'] = 'finrisk-evaluation/2.2'
    save(plan_path, plan)
    with pytest.raises(ValueError, match='EVALUATION_PROTOCOL_MISMATCH'):
        m.build_comparison(plan_path)
