import importlib.util
import json
from pathlib import Path

import pytest


def module():
    p=Path(__file__).resolve().parents[2]/'scripts/agentic_dev30/experiment.py'
    spec=importlib.util.spec_from_file_location('round_experiment',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def failed(reason='MODEL_TRANSPORT_FAILURE',diagnostic=None):
    return {'task_status':'partial','stop':{'reason':'model_failure'},'resources':{}},[
        {'type':'model.failure','data':{'reason':reason,'diagnostic':diagnostic}}]


def test_two_transport_failures_block_next_admission_and_survive_reload(tmp_path):
    m=module();gate=m.RoundCircuit(tmp_path/'health.json')
    for key in ['X:R1','X:A']:
        gate.before_task();report,events=failed();gate.observe(key,report,events)
    with pytest.raises(ValueError,match='ROUND_CIRCUIT_OPEN'):gate.before_task()
    assert gate.value['failure_streak']==2
    with pytest.raises(ValueError,match='ROUND_CIRCUIT_OPEN'):
        m.RoundCircuit(tmp_path/'health.json').before_task()


def test_semantic_review_is_not_network_failure_and_completed_resets_streak(tmp_path):
    m=module();gate=m.RoundCircuit(tmp_path/'health.json')
    gate.observe('X',{'task_status':'completed','overall_judgment':'review_required','resources':{}},[])
    assert gate.value['failure_streak']==0
    r,e=failed();gate.observe('Y',r,e)
    gate.observe('Z',{'task_status':'completed','overall_judgment':'no_clear_risk','resources':{}},[])
    assert gate.value['failure_streak']==0
    gate.observe('W',r,e);gate.before_task()
    gate.observe('W',r,e)  # Replay of one observation cannot count twice.
    assert gate.value['failure_streak']==1


def test_permanent_http_rejection_stops_immediately_without_leaking_diagnostic(tmp_path):
    m=module();gate=m.RoundCircuit(tmp_path/'health.json')
    report,events=failed('PROVIDER_HTTP_401',{'stage':'fetch','category':'connect','error_code':'sk-secret','http_status':401,'message':'Authorization: secret'})
    gate.observe('X',report,events)
    with pytest.raises(ValueError,match='ROUND_CIRCUIT_OPEN'):gate.before_task()
    saved=(tmp_path/'health.json').read_text(encoding='utf-8')
    assert 'sk-secret' not in saved and 'Authorization' not in saved


def test_rerank_circuit_blocks_whole_round_even_when_model_was_not_called(tmp_path):
    m=module();gate=m.RoundCircuit(tmp_path/'health.json')
    report={'task_status':'partial','resources':{'physical_provider_requests':0,'retrieval_accounting':{'transport_circuit_open':True,'transport_diagnostic':{'stage':'fetch','category':'timeout','error_code':'ETIMEDOUT','http_status':None}}}}
    gate.observe('X',report,[]);gate.observe('Y',report,[])
    with pytest.raises(ValueError,match='ROUND_CIRCUIT_OPEN'):gate.before_task()


def test_smoke_plan_uses_only_requested_approved_ids_and_rejects_duplicates():
    m=module();qs=[{'sample_id':f'X{i}','text':'input','user_context':{}} for i in range(3)]
    refs=[{'sample_id':f'X{i}','review_id':rid} for i,rid in enumerate(['S02','C15','C08'])]
    tasks=m.schedule(qs,refs,review_ids=['S02','C15'])
    assert len(tasks)==6 and {t['input']['sample_id'] for t in tasks}=={'X0','X1'}
    assert all('review_id' not in t for t in tasks)
    with pytest.raises(ValueError,match='INVALID_REVIEW_SELECTION'):m.schedule(qs,refs,review_ids=['S02','S02'])


def test_all_partial_reports_have_engineering_counts_but_no_accuracy():
    m=module();report={'task_status':'partial','overall_judgment':'review_required','risk_points':[],'resources':{}}
    result=m.score_results([{'task_key':'X:R1','arm':'R1','input':{'sample_id':'X'}}],
        [{'sample_id':'X','difficulty':'simple','reference':{'overall_judgment':'suspected','supported_tags':['tag']}}],{'X:R1':report})
    assert result['arms']['R1']['failed']==1
    assert result['quality_status']=='no_completed_results'
    assert result['arms']['R1']['judgment_exact'] is None


def test_opening_health_circuit_does_not_release_unknown_budget_reserve(tmp_path):
    m=module();budget=m.RoundBudget(tmp_path/'budget.json','plan',6_000_000)
    gate=m.RoundCircuit(tmp_path/'health.json')
    for key in ['X','Y']:
        gate.before_task();budget.reserve(key);budget.finish(key,0,500_000)
        r,e=failed();gate.observe(key,r,e)
    before=(tmp_path/'budget.json').read_bytes()
    with pytest.raises(ValueError,match='ROUND_CIRCUIT_OPEN'):gate.before_task()
    assert (tmp_path/'budget.json').read_bytes()==before and budget.remaining()==5_000_000


def test_explicit_content_filter_is_not_a_network_failure(tmp_path):
    m=module();gate=m.RoundCircuit(tmp_path/'health.json')
    for key in ('X','Y'):
        report,events=failed('MODEL_CONTENT_FILTER',{'stage':'stream','category':'unknown','error_code':None,'http_status':200})
        gate.observe(key,report,events)
    gate.before_task()
    assert gate.value['failure_streak']==0 and not gate.value['open']


def test_confirmed_content_filter_interrupts_consecutive_network_failures(tmp_path):
    m=module();gate=m.RoundCircuit(tmp_path/'health.json')
    report,events=failed();gate.observe('network1',report,events)
    assert gate.value['failure_streak']==1
    report,events=failed('MODEL_CONTENT_FILTER')
    events[0]['data']['physical_requests']=1
    gate.observe('filtered',report,events)
    assert gate.value['failure_streak']==0
    assert gate.value['last_failure'] is None
    report,events=failed();gate.observe('network2',report,events)
    gate.before_task()
    assert gate.value['failure_streak']==1 and not gate.value['open']

    report,events=failed('MODEL_CONTENT_FILTER')
    events[0]['data']['physical_requests']=0
    gate.observe('not_confirmed',report,events)
    assert gate.value['failure_streak']==1
