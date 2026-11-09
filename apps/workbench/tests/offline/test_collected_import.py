"""Collected files preserve values, subject scope, and already grounded labels."""

import hashlib
import json
import zipfile

import pytest

from offline.collected_import import (
    LEGACY_CASES,
    PREFIX,
    CollectedDocument,
    archive_documents,
    collected_payload,
    is_insurance_case,
    parse_fields,
)

TEXT = (
    "当事人: 甲保险公司及相关责任人员\n行政处罚决定书文号\n金罚〔2026〕18号\n"
    "主要违法违规事实(案由)\n虚列费用，给予投保人保险合同约定以外的利益\n"
    "行政处罚内容\n对甲保险公司罚款10万元；对李某警告并罚款2万元\n"
    "作出处罚决定的机关名称\n甲监管分局\n作出处罚决定的日期\n2026年8月17日\n"
    "行政处罚依据\n《中华人民共和国保险法》第一百一十六条\n来源公示\n处罚信息公开表"
)


class Detector:
    def __init__(self):
        self.calls = []

    def detect(self, text):
        self.calls.append(text)
        return type("Result", (), {"risk_labels": ("合同外利益",)})()


def document(text=TEXT, legacy=None):
    return CollectedDocument("source.txt", text.encode(), {}, legacy)


def test_fields_preserve_entire_group_penalties_and_only_detect_facts():
    detector = Detector()
    payload = collected_payload(document(), TEXT, {"合同外利益"}, detector)
    assert payload["party_name"] == "甲保险公司及相关责任人员"
    assert payload["penalty_content"] == "对甲保险公司罚款10万元；对李某警告并罚款2万元"
    assert payload["penalty_date"] == "2026-08-17"
    assert detector.calls == [payload["violation_behavior"]]
    assert "penalty_amounts" not in payload  # No invented allocation of group fines.


def test_multiline_headers_and_unsplit_fact_header_are_recognized():
    text = (
        TEXT.replace("当事人: ", "被处罚当事人\n")
        .replace("主要违法违规事实(案由)", "主要违法违规事实\n（案由）")
        .replace("行政处罚内容", "行政处罚决定")
        .replace("作出处罚决定的机关名称", "作出处罚决定的\n机关名称")
    )
    fields = parse_fields(text)
    assert fields["regulator"] == "甲监管分局"
    assert fields["violation_behavior"].startswith("虚列费用")
    assert fields["penalty_content"].endswith("2万元")


@pytest.mark.parametrize(
    ("party", "fact", "expected"),
    [
        ("交银金融资产投资有限公司", "未经批准变更住所", False),
        ("甲农村商业银行", "贷款审查不尽职", False),
        ("甲邮政公司", "遗失保险兼业代理业务许可证", True),
        ("甲某", "虚假理赔", True),
        ("新华资产管理股份有限公司", "关联交易管理不到位", True),
    ],
)
def test_insurance_relevance_uses_subject_and_facts(party, fact, expected):
    assert (
        is_insurance_case(
            {
                "party_name": party,
                "violation_behavior": fact,
                "regulator": "中国银行保险监督管理委员会",
                "legal_basis": "《银行业监督管理法》",
            }
        )
        is expected
    )


def test_existing_labels_reuse_only_literal_fact_quotes():
    detector = Detector()
    old = {
        "case_id": "U-old",
        "risk_tag_basis": [
            {"risk_tag": "虚列费用套取资金", "quote": "虚列费用", "reason": "原文明确"},
            {"risk_tag": "销售误导", "quote": "不存在的事实", "reason": "错误旧标签"},
        ],
    }
    payload = collected_payload(
        document(legacy=old), TEXT, {"虚列费用套取资金", "销售误导"}, detector
    )
    assert payload["import_kind"] == "legacy_structured"
    assert payload["risk_tags"] == ["虚列费用套取资金"]
    assert not detector.calls


def test_delivery_notice_has_literal_subject_facts_and_industry_ban():
    text = (
        "张某：\n经查，你在保险机构任职期间，对员工侵占保险费行为负有责任。"
        "根据《中华人民共和国保险法》，决定对你作出禁止10年进入保险业的行政处罚。\n"
        "国家金融监督管理总局甲监管分局\n2026年7月14日"
    )
    payload = collected_payload(document(text), text, {"其他"}, Detector())
    assert payload["party_name"] == "张某"
    assert payload["penalty_content"] == "决定对你作出禁止10年进入保险业的行政处罚"
    assert payload["penalty_date"] == "2026-07-14"
    assert all(quote in text for quote in payload["fact_quotes"])


def test_archive_resolves_internal_members_and_validates_old_file_hash(tmp_path):
    path = tmp_path / "old.zip"
    raw = TEXT.replace("\n", "\r\n").encode()
    sha1 = hashlib.sha1(TEXT.encode()).hexdigest()
    row = {
        "column": 4115,
        "txt_path": "C:/old-computer/untrusted-path/123.txt",
        "sha1": sha1,
        "source_url": "https://www.nfra.gov.cn/cn/view/pages/ItemDetail.html?docId=123&itemId=4115",
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(PREFIX + "index.jsonl", json.dumps(row) + "\n")
        archive.writestr(PREFIX + "ingested.jsonl", json.dumps({"sha1": sha1, "file_id": "old"}))
        archive.writestr(LEGACY_CASES, json.dumps({"file_id": "old", "case_id": "old"}))
        archive.writestr(PREFIX + "nfra/4115/123.txt", raw)
    docs = list(archive_documents(path))
    assert len(docs) == 1 and docs[0].raw == raw
    assert docs[0].legacy["case_id"] == "old"
    with zipfile.ZipFile(path, "a") as archive:
        # Change the index hash rather than extracting or following its saved local path.
        row["sha1"] = "invalid"
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr(PREFIX + "index.jsonl", json.dumps(row))
    with pytest.raises(ValueError, match="校验失败"):
        list(archive_documents(path))
