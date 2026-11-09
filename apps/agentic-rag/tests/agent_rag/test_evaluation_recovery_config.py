import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[2]/'scripts/agentic_dev30/experiment.py'
    spec = importlib.util.spec_from_file_location('evaluation_recovery_experiment', path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_recovery_defaults_remain_zero_and_apply_without_mutating_bundle():
    m = module()
    assert callable(getattr(m, 'format_repair_limits', None)), 'explicit frozen recovery policy is required'
    assert m.format_repair_limits({}) == {'R1': 0, 'R2': 0, 'A': 0}
    bundle = {'config': {'engine': 'controlled_agent'}, 'input': {'text': 'material'}}
    original = m.canonical(bundle)
    plan = {'format_repair_limits': {'R1': 0, 'R2': 0, 'A': 1}}
    for arm in m.ARMS:
        bound = m.bind_format_repair_config(bundle, arm, plan)
        assert bound['config']['evaluation_format_repair_limit'] == (1 if arm == 'A' else 0)
    assert m.canonical(bundle) == original


@pytest.mark.parametrize('limits', [
    {'R1': 1, 'R2': 0, 'A': 1}, {'R1': 0, 'R2': 1, 'A': 1},
    {'R1': 0, 'R2': 0, 'A': 2}, {'R1': 0, 'R2': 0, 'A': True},
    {'R1': 0, 'R2': 0, 'A': '1'}, {'R1': 0, 'A': 1}, None,
    {'R1': 0, 'R2': 0, 'A': 1, 'unknown': 1},
])
def test_invalid_frozen_recovery_policy_is_rejected(limits):
    m = module()
    assert callable(getattr(m, 'format_repair_limits', None)), 'explicit frozen recovery policy is required'
    with pytest.raises(ValueError, match='INVALID_FORMAT_REPAIR_LIMITS'):
        m.format_repair_limits({'format_repair_limits': limits})


@pytest.mark.parametrize('limit', [-1, 2, True, '1', None])
def test_invalid_prepare_recovery_limit_refused_before_reading_paths(tmp_path, limit):
    m = module()
    with pytest.raises(ValueError, match='INVALID_ADAPTIVE_FORMAT_REPAIR_LIMIT'):
        m.prepare(tmp_path/'missing', tmp_path/'missing', tmp_path/'missing', tmp_path/'output', tmp_path/'missing-node', adaptive_format_repair_limit=limit)
    assert not (tmp_path/'output').exists()
