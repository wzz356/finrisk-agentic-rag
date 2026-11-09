import hashlib
import json
from pathlib import Path

import pytest

TEXT = (
    "当事人：甲保险公司。\n经查，甲保险公司向投保人赠送合同外购物卡。\n对内部培训明确禁止承诺收益。"
)


def draft():
    return {
        "file_id": "WEB-1",
        "read_full_text": True,
        "insurance_relevance": {"value": True, "quotes": ["甲保险公司"], "reason": "保险主体"},
        "marketing_relevance": {
            "value": True,
            "quotes": ["赠送合同外购物卡"],
            "reason": "销售行为",
        },
        "cases": [
            {
                "local_key": "c1",
                "party_name": "甲保险公司",
                "party_quote": "当事人：甲保险公司。",
                "party_type": "institution",
                "case_summary": "向投保人赠送合同外购物卡。",
                "case_summary_quotes": ["甲保险公司向投保人赠送合同外购物卡。"],
            }
        ],
        "behaviors": [
            {
                "local_key": "b1",
                "case_key": "c1",
                "actor_role": "机构行为实施者",
                "target": "投保人",
                "action": "赠送",
                "object": "合同外购物卡",
                "summary": "向投保人赠送合同外购物卡。",
                "finding_quote": "甲保险公司向投保人赠送合同外购物卡。",
                "risk_tags": [],
                "zero_tag_reason": "本测试不分配标签",
                "conditions": [],
                "negations": [{"quote": "对内部培训明确禁止承诺收益。"}],
                "exceptions": [],
                "uncertainties": [],
            }
        ],
        "coverage": [],
        "unresolved": [],
        "source_issues": [],
    }


def module():
    from agent_rag import knowledge_updates

    return knowledge_updates


def test_valid_candidate_remains_unreviewed():
    result = module().validate_draft(draft(), "WEB-1", TEXT, {"合同外利益"})
    assert result["human_review_status"] == "pending"
    assert result["semantic_status"] == "candidate"
    assert result["behaviors"][0]["negations"][0]["quote"] in TEXT


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["behaviors"][0].update(finding_quote="机构保证年收益百分之十"),
        lambda d: d.update(human_review_status="approved"),
        lambda d: d["behaviors"][0].update(risk_tags=[{"tag": "虚构标签", "quotes": ["购物卡"]}]),
        lambda d: d["behaviors"][0].update(case_key="missing"),
        lambda d: d["behaviors"][0].update(negations=None),
        lambda d: d["cases"][0].update(party_name="乙保险公司"),
        lambda d: d["behaviors"][0].update(risk_tags=[{"tag": "合同外利益", "quotes": []}]),
        lambda d: d["behaviors"][0].update(
            risk_tags=[{"tag": "合同外利益", "quotes": ["购物卡"]}] * 4
        ),
        lambda d: d["cases"][0].update(penalties=["罚款22.5万元"]),
        lambda d: d["behaviors"][0].update(observed_scene="推断的销售现场"),
        lambda d: d["cases"][0].update(institution_hierarchy="甲公司"),
    ],
)
def test_rejects_invalid_candidate(mutate):
    value = draft()
    mutate(value)
    with pytest.raises(ValueError):
        module().validate_draft(value, "WEB-1", TEXT, {"合同外利益"})


def test_lock_rejects_overlap_and_releases_after_exception(tmp_path):
    with pytest.raises(RuntimeError):
        with module().update_lock(tmp_path):
            with pytest.raises(ValueError, match="UPDATE_LOCKED"):
                with module().update_lock(tmp_path):
                    pass
            raise RuntimeError("interrupted")
    with module().update_lock(tmp_path):
        pass


def test_append_keeps_entire_original_bytes(tmp_path):
    base, delta, dest = (tmp_path / n for n in ("base", "delta", "dest"))
    base.write_bytes(b'{"old":1}\r\n')
    delta.write_bytes(b'{"new":2}\n')
    module().append_jsonl(base, delta, dest)
    assert dest.read_bytes() == base.read_bytes() + delta.read_bytes()


def test_input_index_cannot_escape_crawl_root(tmp_path):
    root = tmp_path / "crawl"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text(TEXT, encoding="utf-8")
    (root / "index.jsonl").write_text(
        json.dumps(
            {
                "txt_path": str(outside),
                "source_url": "https://www.nfra.gov.cn/cn/view/pages/ItemDetail.html?docId=1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="SOURCE_PATH_OUTSIDE_CRAWL"):
        module().pending_documents(root, set())


def test_dedup_base_and_batch_and_untrusted_url(tmp_path):
    root = tmp_path
    p = root / "a.txt"
    p.write_bytes(TEXT.encode("utf-8"))
    row = {"txt_path": str(p), "title": "保险处罚", "source_url": "https://www.nfra.gov.cn/a"}
    (root / "index.jsonl").write_text((json.dumps(row) + "\n") * 2, encoding="utf-8")
    assert len(module().pending_documents(root, set())) == 1
    assert module().pending_documents(root, {hashlib.sha256(TEXT.encode()).hexdigest()}) == []
    row["source_url"] = "https://nfra.gov.cn.attacker.invalid/a"
    (root / "index.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="UNTRUSTED_SOURCE_URL"):
        module().pending_documents(root, set())


def test_generation_is_cached_and_repairs_at_most_once(tmp_path):
    calls = []

    class Client:
        def complete(self, system, user, *, request_key, max_tokens):
            calls.append(request_key)
            value = draft()
            if len(calls) == 1:
                value["behaviors"][0]["finding_quote"] = "不在原文中的引文"
            return json.dumps(value, ensure_ascii=False)

    source = {
        "file_id": "WEB-1",
        "full_text": TEXT,
        "text_sha256": hashlib.sha256(TEXT.encode()).hexdigest(),
    }
    defs = [{"risk_tag": "合同外利益", "description": "测试"}]

    def factory(*args):
        return Client()

    cache = tmp_path / "candidate"
    result = module().generate_candidate(source, cache, factory, tmp_path / "ledger", 1, 6, defs)
    assert result["human_review_status"] == "pending" and len(calls) == 2
    assert (
        module().generate_candidate(
            source, cache, factory, tmp_path / "different-ledger", 1, 6, defs
        )
        == result
    )
    assert len(calls) == 2
    assert json.loads((cache / "state.json").read_text())["ledger"].endswith("ledger")


def test_two_invalid_generations_are_quarantined_and_never_regenerated(tmp_path):
    calls = []

    class Client:
        def complete(self, *args, **kwargs):
            calls.append(1)
            return "{}"

    source = {"file_id": "WEB-1", "full_text": TEXT, "text_sha256": "a" * 64}
    for _ in range(2):
        with pytest.raises(ValueError, match="CANDIDATE_NEEDS_HUMAN_REVIEW"):
            module().generate_candidate(
                source,
                tmp_path / "candidate",
                lambda *args: Client(),
                tmp_path / "ledger",
                1,
                6,
                [],
            )
    assert len(calls) == 2


def test_resume_unclaimed_request_uses_current_budget_only(tmp_path):
    from agent_rag.update_model import UpdateModelClient

    source = {
        "file_id": "WEB-1",
        "full_text": TEXT,
        "text_sha256": hashlib.sha256(TEXT.encode()).hexdigest(),
    }
    old, new = tmp_path / "old.sqlite", tmp_path / "new.sqlite"
    cache = tmp_path / "candidate"
    cache.mkdir()
    module().write_json(
        cache / "state.json",
        {
            "ledger": str(old),
            "cap_cny": 1,
            "max_requests": 6,
            "attempt": 0,
            "errors": [],
            "text_sha256": source["text_sha256"],
        },
    )
    calls = []

    def factory(path, cap, count):
        def transport(payload, timeout):
            calls.append(path)
            return {
                "model": "deepseek-flash",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps(draft(), ensure_ascii=False)},
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            }

        return UpdateModelClient(path, cap, count, transport=transport)

    factory(old, 1, 6)  # 原账本确实存在，且已确认没有该请求键。
    result = module().generate_candidate(
        source, cache, factory, new, 0.5, 2, [{"risk_tag": "合同外利益"}]
    )
    assert result["file_id"] == "WEB-1" and calls == [new]
    assert factory(old, 1, 6).summary()["requests"] == 0
    assert factory(new, 0.5, 2).summary()["requests"] == 1


@pytest.mark.parametrize("damaged", ["missing", "empty", "wrong_schema"])
def test_missing_old_ledger_cannot_be_recreated_to_resend(tmp_path, damaged):
    from agent_rag.update_model import UpdateModelClient

    source = {
        "file_id": "WEB-1",
        "full_text": TEXT,
        "text_sha256": hashlib.sha256(TEXT.encode()).hexdigest(),
    }
    defs = [{"risk_tag": "合同外利益"}]
    calls = []

    def factory(path, cap, count):
        def transport(payload, timeout):
            calls.append(path)
            raise TimeoutError("unknown provider usage")

        return UpdateModelClient(path, cap, count, transport=transport)

    old = tmp_path / "old.sqlite"
    system, user = module().prompt_for(source, defs)
    with pytest.raises(RuntimeError, match="TRANSPORT_FAILED"):
        factory(old, 1, 6).complete(
            system, user, request_key=source["text_sha256"] + ":0", max_tokens=6000
        )
    cache = tmp_path / "cache"
    cache.mkdir()
    module().write_json(
        cache / "state.json",
        {
            "ledger": str(old),
            "cap_cny": 1,
            "max_requests": 6,
            "attempt": 0,
            "errors": [],
            "text_sha256": source["text_sha256"],
        },
    )
    old.unlink()
    if damaged == "empty":
        old.write_bytes(b"")
    elif damaged == "wrong_schema":
        import sqlite3

        with sqlite3.connect(old) as conn:
            conn.execute("create table other (id integer)")
    expected = (
        "UPDATE_MODEL_LEDGER_MISSING" if damaged == "missing" else "UPDATE_MODEL_LEDGER_INVALID"
    )
    with pytest.raises(RuntimeError, match=expected):
        module().generate_candidate(source, cache, factory, tmp_path / "new.sqlite", 1, 6, defs)
    assert len(calls) == 1 and not (tmp_path / "new.sqlite").exists()


def test_long_source_not_silently_truncated_or_sent(tmp_path):
    with pytest.raises(ValueError, match="LONG_SOURCE_NEEDS_REVIEW"):
        module().generate_candidate(
            {"full_text": "保" * 18001}, tmp_path, lambda *args: None, tmp_path / "ledger", 1, 6, []
        )


@pytest.mark.parametrize(
    "bad_count,total,max_docs,changed", [(1, 2, 3, False), (3, 4, 3, False), (3, 4, 3, True)]
)
def test_normalization_failure_is_isolated_and_audited(
    tmp_path, monkeypatch, bad_count, total, max_docs, changed
):
    m = module()
    source = tmp_path / "base-documents.jsonl"
    source.write_text(
        json.dumps(
            {
                "source_id": "BASE",
                "official_file_id": "BASE",
                "full_text": "原库保险文书",
                "text_sha256": hashlib.sha256("原库保险文书".encode()).hexdigest(),
                "text_version": 1,
                "origin_key": "nfra:4113:1:None" if changed else None,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    index = tmp_path / "base-index.jsonl"
    index.write_bytes(b"")
    dictionary = (
        Path(__file__).resolve().parents[2] / "contracts/agent_rag/risk_type_dictionary.csv"
    )
    base = m.V2KnowledgeSnapshot.create(source, index, dictionary, tmp_path / "base")
    rule = tmp_path / "rules.md"
    rule.write_text("test rules")
    base.manifest["backend"] = {
        "mode": "live",
        "assets": {
            "rules": {"path": str(rule)},
            "dense_config": {"path": str(tmp_path / "old-dense")},
        },
    }
    m.write_json(tmp_path / "base/manifest.json", base.manifest)
    crawl = tmp_path / "crawl"
    crawl.mkdir()
    with (crawl / "index.jsonl").open("w", encoding="utf-8") as f:
        for n in range(1, total + 1):
            path = crawl / f"{n}.txt"
            path.write_bytes((TEXT + str(n)).encode())
            f.write(
                json.dumps(
                    {
                        "txt_path": str(path),
                        "source_url": f"https://www.nfra.gov.cn/a{n}",
                        "column": 4113,
                        "doc_id": 1 if changed and n <= bad_count else n,
                    }
                )
                + "\n"
            )

    class Client:
        def complete(self, system, user, **kwargs):
            payload = json.loads(user)
            d = draft()
            d["file_id"] = payload["file_id"]
            return json.dumps(d, ensure_ascii=False)

    def compile_fake(config, accepted, out):
        if any(int(s["full_text"][-1]) <= bad_count for s, d in accepted):
            raise ValueError("V2_NORMALIZATION_FAILED")
        out.mkdir(parents=True)
        docs, units = out / "documents.jsonl", out / "structured.jsonl"
        with docs.open("w", encoding="utf-8") as ds, units.open("w", encoding="utf-8") as us:
            for s, d in accepted:
                ds.write(
                    json.dumps(
                        {**s, "source_id": "SID-" + s["file_id"], "official_file_id": s["file_id"]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                q = d["behaviors"][0]["finding_quote"]
                a = s["full_text"].index(q)
                us.write(
                    json.dumps(
                        {
                            "unit_id": "U-" + s["file_id"],
                            "file_id": s["file_id"],
                            "text": q,
                            "source_text": q,
                            "source_spans": [{"start": a, "end": a + len(q)}],
                            "source_sha256": s["text_sha256"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return {"documents": docs, "index": units, "units": len(accepted)}

    def vectors(old, index, out):
        pytest.fail("未获正式审批的候选不得构建正式向量")

    monkeypatch.setattr(m, "compile_delta", compile_fake)
    config = {
        "registry": str(tmp_path / "registry"),
        "crawl_dir": str(crawl),
        "base_snapshot": str(tmp_path / "base"),
    }
    result = m.run_update(
        config,
        generate=True,
        max_documents=max_docs,
        model_factory=lambda *args: Client(),
        vector_builder=vectors,
    )
    if bad_count == 3:
        assert result["status"] == "review_required"
        result = m.run_update(
            config,
            generate=True,
            max_documents=max_docs,
            model_factory=lambda *args: Client(),
            vector_builder=vectors,
        )
        assert result["held_documents"] == 3
    assert result["status"] == "candidate_pending_review" and result["accepted_documents"] == 1
    assert result["selectable"] is False and result["formal_build_allowed"] is False
    assert not (tmp_path / "registry/versions").exists()
    if bad_count == 1:
        assert result["review_queue"][0]["code"] == "V2_NORMALIZATION_FAILED"
    assert not (tmp_path / "registry/current.json").exists()
    assert (Path(result["run_dir"]) / "review-queue.json").exists()


def test_rollback_does_not_depend_on_builder_assets(tmp_path, monkeypatch):
    import importlib.util

    script = Path(__file__).resolve().parents[2] / "scripts/update_agent_knowledge.py"
    spec = importlib.util.spec_from_file_location("update_cli_test", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "schema_version": "finrisk-web-update/1.0",
                "registry": str(tmp_path / "registry"),
                "build_assets": {
                    "normalizer": {"path": str(tmp_path / "removed-builder.py"), "sha256": "0" * 64}
                },
                "base_snapshot": str(tmp_path / "removed-base"),
                "base_manifest_sha256": "0" * 64,
            }
        )
    )
    import agent_rag.knowledge_registry as registry

    monkeypatch.setattr(registry, "rollback_snapshot", lambda root, version: {"version": version})
    assert cli.main(["rollback", "--config", str(cfg), "--version", "old"])["version"] == "old"
