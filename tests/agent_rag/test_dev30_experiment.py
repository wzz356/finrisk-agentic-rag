import importlib.util
import itertools
import json
import tempfile
import unittest
import pytest
from unittest.mock import patch
from pathlib import Path


@pytest.mark.parametrize('protocol,filename', [
    (None, 'deepseek-rerank-rules.md'), ('quote-copy-v1', 'deepseek-rerank-rules.md'),
    ('span-ref-v2', 'deepseek-rerank-span-rules.md')])
def test_snapshot_rerank_binding_freezes_explicit_protocol_and_matching_rules(tmp_path, protocol, filename):
    m = module()
    assert callable(getattr(m, 'snapshot_rerank_binding', None)), 'snapshot protocol binding is required'
    directory = tmp_path/'resources/agent_rag'
    directory.mkdir(parents=True)
    for name in ('deepseek-rerank-rules.md','deepseek-rerank-span-rules.md'):
        (directory/name).write_text(name, encoding='utf-8')
    asset = directory/filename
    backend = {'mode':'live','assets':{'rules':{'path':str(asset),'sha256':m.file_sha(asset)}}}
    if protocol is not None:
        backend['rerank_protocol'] = protocol
    expected = {'rerank_protocol':protocol or 'quote-copy-v1','rules_sha256':m.file_sha(asset)}
    assert m.snapshot_rerank_binding({'backend':backend}, tmp_path) == expected
    backend['assets']['rules']['sha256'] = 'f'*64
    with pytest.raises(ValueError, match='RERANK_.*MISMATCH'):
        m.snapshot_rerank_binding({'backend':backend}, tmp_path)


def test_snapshot_span_protocol_cannot_bind_legacy_rules(tmp_path):
    m = module()
    assert callable(getattr(m, 'snapshot_rerank_binding', None)), 'snapshot protocol binding is required'
    directory = tmp_path/'resources/agent_rag'
    directory.mkdir(parents=True)
    for name in ('deepseek-rerank-rules.md','deepseek-rerank-span-rules.md'):
        (directory/name).write_text(name, encoding='utf-8')
    asset = directory/'deepseek-rerank-rules.md'
    snapshot = {'backend':{'mode':'live','rerank_protocol':'span-ref-v2',
        'assets':{'rules':{'path':str(asset),'sha256':m.file_sha(asset)}}}}
    with pytest.raises(ValueError, match='RERANK_.*MISMATCH'):
        m.snapshot_rerank_binding(snapshot, tmp_path)


def test_execute_rejects_pre_reliability_plan_before_runtime_or_key_access(tmp_path):
    m = module()
    plan = {'max_judgment_requests':1,'max_rerank_requests':240,
        'resilience_policy':m.RESILIENCE_POLICY,'frozen_files':[],
        'evaluation_runtime_description':{'schema_version':'finrisk-evaluation/2.2'}}
    m.save(tmp_path/'plan.json', plan)
    m.save(tmp_path/'approved.json', {'approval_status':'approved',
        'plan_sha256':m.file_sha(tmp_path/'plan.json'),'human_statement':'批准离线测试',
        'approved_at':'2026-10-17','external_data_scope':m.EXTERNAL_SCOPE,
        'max_judgment_requests':1,'max_rerank_requests':240,'total_cap_cny':1.5})
    with patch('socket.socket', side_effect=AssertionError('network before protocol validation')):
        with pytest.raises(ValueError, match='EVALUATION_PROTOCOL_MISMATCH'):
            m.execute(tmp_path/'plan.json', tmp_path/'approved.json', tmp_path/'no-node', tmp_path/'no-key')
    assert not (tmp_path/'execution').exists()


def module():
    path = Path(__file__).resolve().parents[2] / 'scripts/agentic_dev30/experiment.py'
    if not path.exists():
        path = Path(__file__).with_name('experiment.py')
    assert path.exists(), 'experiment preparation module is missing'
    spec = importlib.util.spec_from_file_location('dev30_experiment', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class ExperimentTests(unittest.TestCase):
    def test_balanced_arms_and_no_reference_in_model_input(self):
        m = module()
        queries = [{'sample_id': 'X'+str(i), 'text': 'text', 'user_context': {}} for i in range(6)]
        refs = [{'sample_id': 'X'+str(i), 'review_id': m.PILOT_IDS[i], 'difficulty': 'complex'} for i in range(6)]
        tasks = m.schedule(queries, refs, pilot=True)
        self.assertEqual(len(tasks), 18)
        orders = [tuple(t['arm'] for t in tasks[i:i+3]) for i in range(0, 18, 3)]
        self.assertEqual(set(orders), set(itertools.permutations(m.ARMS)))
        for t in tasks:
            self.assertEqual(set(t['input']), {'sample_id', 'text', 'user_context'})
            self.assertNotIn('review_id', t)
        with self.assertRaisesRegex(ValueError, 'INPUT_ALLOWLIST'):
            m.schedule([{**q, 'answer': 'leak'} for q in queries], refs, True)

    def test_unknown_cost_is_not_refunded(self):
        m = module()
        with tempfile.TemporaryDirectory() as d:
            ledger = m.RoundBudget(Path(d)/'ledger.json', 'plan', 3_000_000)
            ledger.reserve('X:R1')
            self.assertEqual(ledger.remaining(), 1_500_000)
            ledger.finish('X:R1', known_micro=100_000, unknown_micro=400_000)
            self.assertEqual(ledger.remaining(), 2_500_000)
            ledger.reserve('X:A')
            with self.assertRaisesRegex(ValueError, 'ROUND_BUDGET'):
                ledger.reserve('X:R2')
            with self.assertRaisesRegex(ValueError, 'ALREADY_STARTED'):
                m.RoundBudget(Path(d)/'ledger.json', 'plan', 3_000_000)

    def test_no_authorization_no_live_backend_import_or_calls(self):
        m = module()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            with self.assertRaisesRegex(ValueError, 'APPROVAL_REQUIRED'):
                m.require_authorization({'approval_status':'pending'}, 'sha')
            with self.assertRaisesRegex(ValueError, 'PLAN_MISMATCH'):
                m.require_authorization({'approval_status':'approved', 'plan_sha256':'wrong'}, 'sha')
            m.save(path/'plan.json', {})
            m.save(path/'pending.json', {'approval_status':'pending'})
            with patch('socket.socket', side_effect=AssertionError('network before authorization')):
                with self.assertRaisesRegex(ValueError, 'APPROVAL_REQUIRED'):
                    m.execute(path/'plan.json',path/'pending.json',path/'no-node',path/'no-key')

    def test_missing_or_unknown_task_ledgers_keep_cost_reserved(self):
        m=module()
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            self.assertEqual(m.read_costs(d/'model',d/'rerank'),(0,1_500_000))
            m.save(d/'model', {'settled_cost_micro_cny':100_000,'reserved_cost_micro_cny':200_000})
            m.save(d/'rerank', {'settled_cost_micro_cny':10_000,'reserved_cost_micro_cny':30_000})
            self.assertEqual(m.read_costs(d/'model',d/'rerank'),(110_000,230_000))

    def test_valid_empty_model_ledger_releases_only_proven_zero_allocation(self):
        m=module()
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            m.save(d/'model', {'settled_cost_micro_cny':0, 'reserved_cost_micro_cny':0,
                              'request_reservations':[]})
            m.save(d/'rerank', {'settled_cost_micro_cny':10_000,'reserved_cost_micro_cny':30_000})
            self.assertEqual(m.read_costs(d/'model',d/'rerank'), (10_000,30_000))
            m.save(d/'model', {'settled_cost_micro_cny':-1,'reserved_cost_micro_cny':0})
            self.assertEqual(m.read_costs(d/'model',d/'rerank'), (10_000,1_030_000))
            (d/'model').write_text('{bad',encoding='utf-8')
            with self.assertRaises(json.JSONDecodeError):
                m.read_costs(d/'model',d/'rerank')

    def test_safe_round_diagnostic_retains_tls_allowlist_only(self):
        m=module()
        for code in ('SSL_ERROR','UNEXPECTED_EOF_WHILE_READING','WRONG_VERSION_NUMBER',
                     'TLSV1_ALERT_PROTOCOL_VERSION','TLSV1_ALERT_INTERNAL_ERROR'):
            value={'stage':'fetch','category':'tls','error_code':code,'http_status':None}
            self.assertEqual(m.RoundCircuit._safe_diagnostic(value), value)
        result=m.RoundCircuit._safe_diagnostic({'stage':'fetch','category':'tls',
            'error_code':'UNEXPECTED_EOF_WHILE_READING sk-secret','http_status':None})
        self.assertIsNone(result['error_code'])

    def test_results_keep_failures_and_missing_tasks_in_denominator(self):
        m = module()
        tasks = [{'task_key': 'X:R1', 'arm':'R1', 'input':{'sample_id':'X'}}, {'task_key':'Y:R1','arm':'R1','input':{'sample_id':'Y'}}]
        refs = [{'sample_id':'X','difficulty':'simple','reference':{'overall_judgment':'suspected','supported_tags':['tag'],'pending_tags':[]}}, {'sample_id':'Y','difficulty':'complex','reference':{'overall_judgment':'suspected','supported_tags':['tag'],'pending_tags':[]}}]
        reports = {'X:R1':{'task_status':'completed','overall_judgment':'suspected','risk_points':[{'supported_tags':['tag'],'pending_tags':[]}],'resources':{}}}
        out = m.score_results(tasks, refs, reports)
        self.assertEqual(out['arms']['R1']['planned'], 2)
        self.assertEqual(out['arms']['R1']['missing'], 1)
        self.assertEqual(out['arms']['R1']['judgment_exact'], 1)
        self.assertEqual(out['arms']['R1']['label_false_negative'], 1)
        self.assertEqual(out['semantic_evidence_status'], 'awaiting_human_review')
        with self.assertRaisesRegex(ValueError, 'SIMULATED'):
            m.score_results(tasks, refs, {'X:R1':{**reports['X:R1'],'simulated':True}})

    def test_file_hash_drift_is_rejected(self):
        m = module()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'x'
            p.write_bytes(b'original')
            records = [{'path':str(p),'sha256':m.file_sha(p)}]
            m.verify_files(records)
            p.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'FROZEN_FILE_CHANGED'):
                m.verify_files(records)

    def test_no_model_results_have_no_quality_numbers(self):
        m=module()
        result=m.score_results([{'task_key':'X:R1','arm':'R1','input':{'sample_id':'X'}}],
            [{'sample_id':'X','difficulty':'simple','reference':{'overall_judgment':'suspected','supported_tags':['tag']}}],{})
        self.assertEqual(result['quality_status'],'not_run')
        self.assertIsNone(result['arms']['R1']['judgment_exact'])
        self.assertEqual(result['strata']['simple_risk']['R1']['planned'],1)
        self.assertIsNone(result['strata']['simple_risk']['R1']['judgment_exact'])

    def test_narrower_approval_is_not_silently_ignored(self):
        m=module()
        approval={'approval_status':'approved','plan_sha256':'x','human_statement':'yes','approved_at':'2026-10-16',
                  'total_cap_cny':6,'external_data_scope':m.EXTERNAL_SCOPE,'max_judgment_requests':10,'max_rerank_requests':4320}
        plan={'max_judgment_requests':78,'max_rerank_requests':4320}
        with self.assertRaisesRegex(ValueError,'APPROVAL_SCOPE_MISMATCH'):
            m.require_authorization(approval,'x',plan)

    def test_cleanup_continues_after_cancel_failure_and_restores_environment(self):
        m=module()
        calls=[]
        class BadControl:
            def cancel(self):
                calls.append('cancel')
                raise RuntimeError('ledger failed')
        class Dense:
            def close(self):calls.append('close')
        class Service:
            controls={'x':BadControl()}
        class Backend:
            dense=Dense()
        class Snapshot:
            backend=Backend()
        import os
        old=dict(os.environ)
        os.environ['FINRISK_UNIT_TEMPORARY']='yes'
        result=m.cleanup_resources(Service(),Snapshot(),None,None,None,old)
        self.assertEqual(calls,['cancel','close'])
        self.assertEqual(dict(os.environ),old)
        self.assertEqual(len(result),1)

    def test_snapshot_closes_even_when_service_construction_failed(self):
        m=module()
        calls=[]
        class Dense:
            def close(self):calls.append('closed')
        class Backend:
            dense=Dense()
        class Snapshot:
            backend=Backend()
        import os
        self.assertEqual(m.cleanup_resources(None,Snapshot(),None,None,None,dict(os.environ)),[])
        self.assertEqual(calls,['closed'])

    def test_started_timeout_is_failed_and_unstarted_remains_missing(self):
        m=module()
        tasks=[{'task_key':s+':R1','arm':'R1','input':{'sample_id':s}} for s in ('X','Y')]
        refs=[{'sample_id':s,'difficulty':'complex','reference':{'overall_judgment':'suspected','supported_tags':['tag']}} for s in ('X','Y')]
        result=m.score_results(tasks,refs,{}, {'X:R1':300}, {'X:R1':{'status':'failed'}})
        self.assertEqual(result['arms']['R1']['failed'],1)
        self.assertEqual(result['arms']['R1']['missing'],1)
        self.assertEqual(result['items'][0]['end_to_end_seconds'],300)
        self.assertIsNone(result['arms']['R1']['judgment_exact'])

    def test_internal_tool_app_uses_round_service_and_blocks_public_task_creation(self):
        m=module()
        from fastapi.testclient import TestClient
        class Service:
            def verify(self,task_id,evidence_ids):return {'valid_ids':evidence_ids,'rejected_ids':[]}
        with patch.dict('os.environ',{'FINRISK_AGENT_SERVICE_TOKEN':'unit-evaluation-secret'}):
            client=TestClient(m.local_tool_app(Service()))
            data={'task_id':'a'*32,'evidence_ids':['E1']}
            self.assertEqual(client.post('/internal/agent-rag/verify',json=data).status_code,401)
            response=client.post('/internal/agent-rag/verify',json=data,headers={'Authorization':'Bearer unit-evaluation-secret'})
            self.assertEqual(response.status_code,200)
            self.assertEqual(response.json()['valid_ids'],['E1'])
            self.assertEqual(client.post('/api/agent-rag/tasks',json={'text':'x'}).status_code,404)


if __name__ == '__main__':
    unittest.main()
