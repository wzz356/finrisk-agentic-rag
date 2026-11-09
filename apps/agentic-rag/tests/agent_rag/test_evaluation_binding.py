import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest


def module():
    path = Path(__file__).resolve().parents[2]/'scripts/agentic_dev30/experiment.py'
    spec = importlib.util.spec_from_file_location('evaluation_binding_experiment', path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def bindings(tmp_path):
    m = module()
    directory = tmp_path/'configs/agent_rag'
    directory.mkdir(parents=True)
    (directory/'system-prompt.txt').write_bytes(b'default prompt remains fixed')
    (directory/'evaluation-system-prompt.txt').write_bytes(b'compact experiment citation prompt')
    description = {'schema_version':'finrisk-evaluation/2.4',
        'evaluation_protocol_sha256':'a'*64,
        'system_prompt_sha256':m.file_sha(directory/'evaluation-system-prompt.txt'),
        'retrieval_reliability': {'coverage_version':'finrisk-ranking-coverage/1.0',
            'schema_version':'finrisk-retrieval-assessment/1.0','pair_attempt_limit':2,
            'rerank_protocols':['quote-copy-v1','span-ref-v2'],'configured_in_v2':True}}
    original = {'input':{'text':'current material only'},'execution_contract':{
        'prompt_sha256':m.file_sha(directory/'system-prompt.txt'),
        'model_wire_sha256':'b'*64,'criteria_sha256':'c'*64}}
    return m, original, description


def test_evaluation_adapter_binds_new_prompt_without_changing_default_task(tmp_path):
    m, original, description = bindings(tmp_path)
    before = m.canonical(original)
    bound = m.bind_evaluation_contract(original, description, tmp_path)
    assert bound['execution_contract']['prompt_sha256'] == description['system_prompt_sha256']
    assert bound['execution_contract']['evaluation_protocol_sha256'] == 'a'*64
    assert bound['execution_contract']['model_wire_sha256'] == 'b'*64
    assert m.canonical(original) == before


@pytest.mark.parametrize('mutation',['stale_protocol','previous_protocol','v22_protocol','v23_protocol','unbound_prompt','forged_original'])
def test_evaluation_adapter_rejects_stale_or_unbound_contract(tmp_path,mutation):
    m, original, description = bindings(tmp_path)
    if mutation == 'stale_protocol': description['schema_version']='finrisk-evaluation/2.0'
    elif mutation == 'previous_protocol': description['schema_version']='finrisk-evaluation/2.1'
    elif mutation == 'v22_protocol': description['schema_version']='finrisk-evaluation/2.2'
    elif mutation == 'v23_protocol': description['schema_version']='finrisk-evaluation/2.3'
    elif mutation == 'unbound_prompt': description['system_prompt_sha256']='d'*64
    else: original['execution_contract']['prompt_sha256']='e'*64
    with pytest.raises(ValueError, match='EVALUATION_.*MISMATCH'):
        m.bind_evaluation_contract(original,description,tmp_path)


def test_v2_binding_enables_retrieval_reliability_without_obligating_bm25(tmp_path):
    m, original, description = bindings(tmp_path)
    original['config'] = {'retrieval_profile':'v2-hybrid-flash-v1','engine':'fixed_rag'}
    before = m.canonical(original)
    bound = m.bind_evaluation_contract(original, description, tmp_path)
    assert bound['config']['retrieval_reliability'] is True
    assert m.canonical(original) == before
    original['config'] = {'retrieval_profile':'bm25-v1'}
    assert 'retrieval_reliability' not in m.bind_evaluation_contract(original, description, tmp_path)['config']


@pytest.mark.parametrize('change', [None, {}, {'pair_attempt_limit':3}, {'configured_in_v2':False},
    {'rerank_protocols':['span-ref-v2']}, {'unexpected':True}])
def test_binding_rejects_unfrozen_retrieval_reliability_descriptor(tmp_path, change):
    m, original, description = bindings(tmp_path)
    if change is None:
        del description['retrieval_reliability']
    elif change == {}:
        description['retrieval_reliability'] = {}
    else:
        description['retrieval_reliability'].update(change)
    with pytest.raises(ValueError, match='EVALUATION_.*MISMATCH'):
        m.bind_evaluation_contract(original, description, tmp_path)


def test_frozen_runtime_includes_both_rerank_rules_and_diagnostics_source():
    m = module()
    project = Path(__file__).resolve().parents[2]
    files = {Path(row['path']).relative_to(project).as_posix() for row in m.frozen_runtime(project)}
    assert 'resources/agent_rag/deepseek-rerank-span-rules.md' in files
    assert 'resources/agent_rag/deepseek-rerank-rules.md' in files
    assert 'scripts/agentic_dev30/retrieval_reliability.py' in files


@pytest.mark.parametrize('old_descriptor', [False, True])
def test_prepare_freezes_protocol_rules_and_recovery_without_upgrading_old_descriptor(tmp_path, old_descriptor):
    m, _, description = bindings(tmp_path)
    reference = tmp_path/'human-reference'
    queries = [{'sample_id':f'S{i:02d}','text':f'材料{i}','user_context':{}} for i in range(30)]
    refs = [{'sample_id':q['sample_id'],'review_id':f'C{i:02d}','human_approved':True,
        'input_sha256':m.obj_sha({'text':q['text'],'user_context':q['user_context']})}
        for i, q in enumerate(queries)]
    for name, rows in [('inputs/queries.jsonl',queries),('evaluation/references.jsonl',refs)]:
        path = reference/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(m.canonical(row) for row in rows)+'\n', encoding='utf-8')
    manifest = {'human_approved_count':30,'files':[{'path':name,'sha256':m.file_sha(reference/name)}
        for name in ('inputs/queries.jsonl','evaluation/references.jsonl')]}
    m.save(reference/'manifest.json', manifest)
    m.save(tmp_path/'resources/agent_rag/evaluation/dev30-human-reference-manifest.json', manifest)
    rules = tmp_path/'resources/agent_rag/deepseek-rerank-span-rules.md'
    rules.write_bytes(b'frozen span rules')
    model = tmp_path/'model'
    model.mkdir()
    (model/'config.json').write_bytes(b'{}')
    dense = tmp_path/'dense.json'
    for name in ('meta.json', 'vectors.bin'):
        (tmp_path/name).write_bytes(b'frozen')
    m.save(dense, {'model_path':str(model),'vector_meta_path':str(tmp_path/'meta.json'),
        'vector_path':str(tmp_path/'vectors.bin')})
    snapshot = tmp_path/'snapshot'
    m.save(snapshot/'manifest.json', {'snapshot_id':'snapshot','rerank_mode':'live',
        'corpus_scope':'full9639','assets':{},'backend':{'mode':'live','rerank_protocol':'span-ref-v2',
            'assets':{'rules':{'path':str(rules),'sha256':m.file_sha(rules)},
                'dense_config':{'path':str(dense),'sha256':m.file_sha(dense)}}}})
    if old_descriptor:
        description['schema_version'] = 'finrisk-evaluation/2.2'
    # This thin fixture tests protocol freezing; real asset validation has separate coverage.
    with patch.object(m, 'frozen_runtime', return_value=[]), patch.object(m.subprocess,
            'check_output', return_value=m.canonical(description).encode('utf-8')), patch.object(
                m.paired_prefetch_module(), 'validate_snapshot',
                side_effect=lambda root, mode, project: m.load(root/'manifest.json')):
        if old_descriptor:
            with pytest.raises(ValueError, match='EVALUATION_PROTOCOL_MISMATCH'):
                m.prepare(tmp_path, reference, snapshot, tmp_path/'new-output', tmp_path/'node',
                    review_ids=['C00'], adaptive_format_repair_limit=1)
            assert not (tmp_path/'new-output').exists()
        else:
            plan = m.prepare(tmp_path, reference, snapshot, tmp_path/'new-output', tmp_path/'node',
                review_ids=['C00'], adaptive_format_repair_limit=1)
            assert plan['retrieval_binding'] == {'rerank_protocol':'span-ref-v2','rules_sha256':m.file_sha(rules)}
            assert plan['evaluation_runtime_description'] == description
            assert plan['format_repair_limits'] == {'R1':0,'R2':0,'A':1}
            assert plan['max_judgment_requests'] == 13
            assert plan['max_rerank_requests'] == 720
            assert plan['limits']['task_model_requests'] == 6
            assert plan['limits']['task_rerank_cap_cny'] == 0.5
            assert plan['network_calls_during_preparation'] == 0
            assert m.load(tmp_path/'new-output/authorization.template.json')['approval_status'] == 'pending'
            m.verify_files(plan['frozen_files'])
