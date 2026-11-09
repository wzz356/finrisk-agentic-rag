import importlib.util
import json
import socket
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "diagnose-agent-v2-dense.py"
REVISION = "7999e1d3359715c523056ef9478215996d62a620"


def diagnostic():
    assert SCRIPT.is_file(), "Task 5 离线差异诊断脚本尚未实现"
    spec = importlib.util.spec_from_file_location("v2_dense_diagnostic", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matrix_keeps_frozen_order_neighbors_three_repeats_and_records_differences(tmp_path):
    module = diagnostic()
    queries = [{"query_id": f"KBT{i:03}", "text": f"查询{i}"} for i in range(1, 21)]
    source = tmp_path / "frozen-source.json"
    source.write_text("不可变原资产", encoding="utf-8")
    before = source.read_bytes()
    calls = []

    class FakeEncoder:
        info = {"fake": True, "model_load_seconds": 0, "warmup_seconds": 0}

        def encode_batch(self, rows, condition):
            calls.append((condition["name"], [r["query_id"] for r in rows]))
            offset = (0.001 if condition["batch_size"] == 1 else 0)
            offset += 0.002 if condition["padding"] == "max_length" else 0
            vectors = [[1.0, int(r["query_id"][3:]) / 100 + offset] for r in rows]
            tokens = [{"query_id": r["query_id"], "input_ids": [101, i, 102],
                       "attention_mask": [1, 1, 1]} for i, r in enumerate(rows)]
            return vectors, tokens, 0.01

    frozen = [[1.0, i / 100] for i in range(1, 21)]

    def scores(vector):
        # A small real ranking where the shape-dependent change swaps the top two.
        return [vector[1], 0.201, -vector[1]]

    output = tmp_path / "diagnostic"
    report = module.diagnose(queries, frozen, scores, ["A", "B", "C"],
                             FakeEncoder(), output, [source])
    assert len(calls) == 144
    first = [r["query_id"] for r in queries[:16]]
    assert calls[0] == ("batch16_dynamic", first)
    assert calls[3] == ("batch1_dynamic", ["KBT005"])
    assert calls[12] == ("batch16_dynamic", first)
    assert calls[13] == ("batch16_dynamic", [r["query_id"] for r in queries[16:]])
    assert report["repetitions"] == 3
    assert report["query_order"] == [r["query_id"] for r in queries]
    assert report["phases"][0]["query_ids"] == ["KBT005"]
    assert report["phases"][1]["query_ids"] == report["query_order"]
    assert len(report["phases"][1]["conditions"]) == 4
    compared = report["phases"][1]["comparisons"]["batch16_dynamic__batch1_dynamic"]
    assert compared["max_abs_vector_difference"] == pytest.approx(0.001)
    assert compared["max_abs_score_difference"] == pytest.approx(0.001)
    assert compared["top100_order_identical_queries"] < 20
    assert compared["top100_set_identical_queries"] == 20
    assert all(row["repeat_bitwise_stable"] for row in report["phases"][1]["conditions"])
    assert report["source_assets_unchanged"] is True
    assert source.read_bytes() == before
    assert (output / "tokens.jsonl").is_file()
    assert (output / "vectors.jsonl").is_file()
    assert (output / "top100.jsonl").is_file()
    assert (output / "report.json").is_file()
    assert (output / "report.md").is_file()


def test_existing_output_is_rejected_before_encoder_or_source_access(tmp_path):
    module = diagnostic()
    output = tmp_path / "exists"
    output.mkdir()
    with pytest.raises(FileExistsError, match="DIAGNOSTIC_OUTPUT_EXISTS"):
        module.diagnose([], [], None, [], None, output, [tmp_path / "missing"])


def test_any_source_mutation_is_reported_and_fails_without_rewriting_source(tmp_path):
    module = diagnostic()
    source = tmp_path / "asset"
    source.write_bytes(b"before")
    queries = [{"query_id": f"KBT{i:03}", "text": str(i)} for i in range(1, 21)]

    class MutatingEncoder:
        info = {"fake": True}

        def encode_batch(self, rows, condition):
            source.write_bytes(b"changed")
            return [[1.0] for _ in rows], [], 0

    with pytest.raises(ValueError, match="SOURCE_ASSET_MUTATED"):
        module.diagnose(queries, [[1.0]] * 20, lambda vector: [1.0], ["unit"],
                         MutatingEncoder(), tmp_path / "out", [source])
    assert source.read_bytes() == b"changed"
    report = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    assert report["source_assets_unchanged"] is False


def test_network_is_blocked_even_for_injected_encoder(tmp_path):
    module = diagnostic()
    queries = [{"query_id": f"KBT{i:03}", "text": str(i)} for i in range(1, 21)]

    class NetworkEncoder:
        info = {"fake": True}

        def encode_batch(self, rows, condition):
            socket.create_connection(("127.0.0.1", 9))

    with pytest.raises(RuntimeError, match="NETWORK_FORBIDDEN"):
        module.diagnose(queries, [[1.0]] * 20, lambda vector: [1.0], ["unit"],
                         NetworkEncoder(), tmp_path / "out", [])


def test_input_binding_rejects_bad_sha_or_revision_before_encoder(tmp_path):
    module = diagnostic()
    asset = tmp_path / "asset"
    asset.write_bytes(b"frozen")
    with pytest.raises(ValueError, match="ASSET_SHA_MISMATCH"):
        module.checked_sha(asset, "0" * 64)
    with pytest.raises(ValueError, match="MODEL_REVISION_MISMATCH"):
        module.validate_profile({"model_revision": "other", "model_path": str(tmp_path / "other")})


def test_profile_never_changes_fp16_normalization_instruction_or_length():
    module = diagnostic()
    assert module.QUERY_INSTRUCTION == "为这个句子生成表示以用于检索相关文章："
    assert module.MODEL_REVISION == REVISION
    assert module.MAX_LENGTH == 512
    assert [(r["batch_size"], r["padding"]) for r in module.CONDITIONS] == [
        (16, True), (1, True), (16, "max_length"), (1, "max_length")]
    with pytest.raises(ValueError, match="ENCODER_PROFILE_MISMATCH"):
        module.validate_profile({"model_revision": REVISION, "model_path": str(Path(REVISION)),
                                 "device": "cpu", "precision": "float32"})
