"""Scope must follow parties/facts, including notices without the full insurance word."""

import json

import pytest

from offline.insurance_relevance import classify_insurance_relevance


@pytest.mark.parametrize(
    "text, expected",
    [
        ("当事人：甲财险宁县支公司\n主要违法违规事实：虚构中介业务套取费用", "insurance"),
        ("当事人：刘某（时任甲财险支公司经理）\n主要违法违规事实：编制虚假资料", "insurance"),
        (
            "被处罚当事人\n招商仁和人寿深圳分公司及相关责任人\n主要违法违规事实\n数据不真实",
            "insurance",
        ),
        ("当事人：甲银行支行\n主要违法违规事实：违规销售保险产品\n行政处罚内容：罚款", "insurance"),
        (
            "当事人：甲银行支行\n主要违法违规事实：贷款管理不审慎\n处罚机关：中国银行保险监督管理委员会",
            "non_insurance",
        ),
        ("当事人：张某\n处罚机关：中国银行保险监督管理委员会", "uncertain"),
        ("当事人：甲银行\n主要违法违规事实：资料不真实\n处罚依据：保险法", "uncertain"),
    ],
)
def test_existing_classification_uses_grounded_fields(text, expected):
    assert (
        classify_insurance_relevance(party_name=None, violation_behavior="", original_text=text)
        == expected
    )


def test_maintenance_does_not_drop_abbreviations_or_uncertain_sources(tmp_path):
    from agent_rag.collection_store import CollectionStore
    from agent_rag.maintenance_service import MaintenanceService

    store = CollectionStore(tmp_path / "inventory")
    batch = store.create_batch("fixture")["id"]
    for index, text in enumerate(
        ["当事人：甲财险支公司\n违法事实：数据不真实", "当事人：张某\n违法事实：数据不真实"]
    ):
        store.ingest(batch, text.encode(), {"source_url": f"https://www.nfra.gov.cn/{index}"})
    service = MaintenanceService(tmp_path / "maintenance")
    result = service.import_inventory(store, batch)
    assert result["count"] == 2
    assert result["publication_allowed"] is False
    assert service.import_inventory(store, batch)["event_ids"] == result["event_ids"]


def test_incremental_entry_keeps_no_keyword_and_bank_business(tmp_path):
    from agent_rag.knowledge_updates import pending_documents

    texts = [
        "当事人：甲财险支公司\n违法事实：数据不真实",
        "当事人：甲银行支行\n违法事实：违规销售保险产品",
        "当事人：张某",
    ]
    index = []
    for i, text in enumerate(texts):
        path = tmp_path / f"{i}.txt"
        path.write_bytes(text.encode())
        index.append({"txt_path": str(path), "source_url": f"https://www.nfra.gov.cn/{i}"})
    (tmp_path / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in index))
    assert len(pending_documents(tmp_path, set())) == 3


@pytest.mark.parametrize(
    "text,category",
    [
        (
            "当事人：李明。违法事实：李明时任甲保险公司银行保险部经理，对虚假宣传负有直接责任。",
            "insurance_entity",
        ),
        ("监管机构：保险监督管理局。违法事实：当事人未明确，文中泛指保险业务监管。", "uncertain"),
        ("当事人：甲食品有限公司。违法事实：食品抽检不合格。", "non_insurance_candidate"),
        ("当事人：张某。违法事实：其与甲保险公司存在合同争议。", "uncertain"),
    ],
)
def test_scope_evidence_and_boundaries(text, category):
    from offline.insurance_relevance import evaluate_insurance_scope

    result = evaluate_insurance_scope(text)
    assert result["category"] == category
    assert result["review_status"] == "pending"
    assert not result["publication_allowed"]
    for evidence in result["evidence"]:
        assert text[evidence["start"] : evidence["end"]] == evidence["quote"]


def test_old_inventory_keeps_literal_flag_but_exposes_current_scope(tmp_path):
    from agent_rag.collection_store import CollectionStore

    store = CollectionStore(tmp_path)
    batch = store.create_batch("fixture")["id"]
    text = "当事人：甲财险支公司\r\n主要违法违规事实：数据不真实"
    key = store.ingest(batch, text.encode(), {"source_url": "https://www.nfra.gov.cn/a"})["digest"]
    record = CollectionStore(tmp_path).read_document(key)
    assert not record["contains_insurance"]
    assert record["scope_assessment"]["category"] == "insurance_entity"
    assert record["text"] == text
    assert store.documents()["items"][0]["scope_assessment"] == record["scope_assessment"]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("当事人：张某。违法事实：违反《保险销售行为管理办法》，具体事实未说明。", "uncertain"),
        ("当事人：张某。违法事实：张某系甲保险公司客户，与该公司存在合同争议。", "uncertain"),
        (
            "当事人：甲银行支行。违法事实：违规销售保险产品，经当地监管局查实。",
            "insurance_business",
        ),
        ("当事人：张某。违法事实：监管局查明张某时任甲财险支公司经理。", "insurance_entity"),
        ("当事人：张某。张某与甲保险公司存在合同纠纷。", "uncertain"),
        ("当事人：张某（系甲保险公司客户）。违法事实：资料不真实。", "uncertain"),
        (
            "当事人：张涛（时任太平人寿庆阳中支个险本部临时负责人、经理）\n违法事实：数据不真实",
            "insurance_entity",
        ),
    ],
)
def test_review_findings_do_not_use_law_names_or_customer_roles(text, expected):
    from offline.insurance_relevance import evaluate_insurance_scope

    assert evaluate_insurance_scope(text)["category"] == expected


def test_party_extraction_stops_before_unlabelled_body():
    from offline.insurance_relevance import extract_party_name

    assert extract_party_name("当事人：张某。张某与甲保险公司存在合同纠纷。") == "张某"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("当事人：张某（时任甲财险支公司客户经理）\n违法事实：编制虚假资料。", "insurance_entity"),
        (
            "当事人：张某\n违法事实：张某时任甲财险支公司客户经理，对编制虚假资料负有责任。",
            "insurance_entity",
        ),
        ("当事人：张某系甲保险公司客户\n违法事实：资料不真实。", "uncertain"),
    ],
)
def test_employment_requires_position_but_keeps_customer_manager(text, expected):
    from offline.insurance_relevance import evaluate_insurance_scope

    assert evaluate_insurance_scope(text)["category"] == expected


def test_licence_notice_is_not_a_penalty_candidate():
    from offline.insurance_relevance import evaluate_insurance_scope

    scope = evaluate_insurance_scope("当事人：人保寿险吉林省分公司\n换领保险许可证公告")
    assert scope["relevance"] == "insurance"
    assert scope["route"] == "scope_review"
    assert not scope["publication_allowed"]


def test_legacy_crawler_retains_scope_review_and_bank_insurance(tmp_path):
    from offline.penalty_crawler.filter import collect_pending

    texts = [
        "当事人：甲财险支公司\n违法事实：数据不真实",
        "当事人：甲银行\n违法事实：违规销售保险产品",
        "当事人：张某",
    ]
    rows = []
    for i, text in enumerate(texts):
        path = tmp_path / f"{i}.txt"
        path.write_bytes(text.encode())
        rows.append({"txt_path": str(path)})
    (tmp_path / "index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    pending, stats = collect_pending(tmp_path, ingested_sha1=set(), existing_case_ids=set())
    assert len(pending) == 3
    assert stats["scope_review"] == 1
    assert pending[1]["scope_assessment"]["category"] == "insurance_business"
