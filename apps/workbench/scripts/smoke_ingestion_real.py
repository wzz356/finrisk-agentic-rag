"""Real DeepSeek multi-party ingestion in an isolated local fixture, never the live KB."""

import asyncio
import json
import shutil
import sqlite3
from dataclasses import replace

from benchmark_support import RecordingLLM
from build_competition_dataset import ROOT, save
from dotenv import load_dotenv

from application.case_service import CaseManagementService
from llm import create_llm_client_from_env
from risk.detection import RuleRiskDetector
from settings import AppSettings
from storage.case_repository import CaseRepository


async def main():
    load_dotenv(ROOT / ".env")
    settings = AppSettings.from_env()
    base = ROOT / "outputs/competition_upgrade_20261019/ingestion_smoke"
    (base / "data").mkdir(parents=True, exist_ok=True)
    (base / "dictionary").mkdir(exist_ok=True)
    for name in ["risk_type_dictionary.csv", "risk_domain_mapping.json"]:
        shutil.copy2(settings.case_kb_root / "dictionary" / name, base / "dictionary" / name)
    if (base / "data/cases.jsonl").exists() and (base / "data/cases.jsonl").stat().st_size:
        print("Isolated real-ingestion evidence already exists; not repeating API calls.")
        return
    for name in ["cases.jsonl", "case_chunks.jsonl", "case_expansions.jsonl"]:
        (base / "data" / name).write_text("", "utf-8")
    save(base / "manifest.json", {"counts": {}, "files": {}, "data_version": "isolated-smoke"})
    with sqlite3.connect(base / "evidence_registry.sqlite3") as con:
        con.executescript(
            "CREATE TABLE IF NOT EXISTS sources(file_id TEXT PRIMARY KEY,text TEXT,sha256 TEXT);"
            "CREATE TABLE IF NOT EXISTS units(id TEXT PRIMARY KEY,file_id TEXT,data TEXT);"
        )
    document = base / "模拟公开表.txt"
    document.write_text(
        """模拟行政处罚公开表，以下名称均为测试用称谓。
当事人：甲人寿保险公司南京分公司。主要违法行为：虚列培训费用套取资金。处罚：警告，并处罚款10万元。处罚机关：某金融监管局。决定日期：2026年10月1日。
当事人：张某，甲人寿保险公司南京分公司保险代理人。主要违法行为：未经投保人本人同意，代客户签署投保文件并代填客户信息。处罚：警告，并处罚款1万元。处罚机关：某金融监管局。决定日期：2026年10月1日。
当事人：乙商业银行。主要违法行为：流动资金贷款管理不审慎，与保险业务无关。处罚：罚款30万元。处罚机关：某金融监管局。决定日期：2026年10月1日。
""",
        "utf-8",
    )
    isolated = replace(
        settings, case_kb_root=base, raw_case_upload_dir=base / "uploads", full_corpus=False
    )
    repository = CaseRepository(isolated)
    inner = create_llm_client_from_env()
    assert inner is not None
    client = RecordingLLM(inner)
    try:
        service = CaseManagementService(
            settings=isolated,
            repository=repository,
            rule_detector=RuleRiskDetector(settings.risk_keyword_path),
            llm_client=client,
        )
        result = await service.ingest_file(document, original_filename=document.name)
        assert len(result.cases) == 2
        assert all("乙商业银行" != item.party_name for item in result.cases)
        assert all(item.evidence_chunks for item in result.cases)
        save(base / "result.json", result.model_dump(mode="json"))
        save(base / "api_usage.json", client.calls)
        with sqlite3.connect(base / "evidence_registry.sqlite3") as con:
            units = con.execute("SELECT COUNT(*) FROM units").fetchone()[0]
        save(
            base / "verification.json",
            {
                "input_parties": 3,
                "insurance_cases": 2,
                "non_insurance_excluded": True,
                "registered_verbatim_units": units,
                "live_knowledge_base_modified": False,
                "real_api": True,
            },
        )
        print(
            json.dumps(
                {
                    "insurance_cases": len(result.cases),
                    "verbatim_units": units,
                    "warnings": result.warnings,
                },
                ensure_ascii=False,
            )
        )
    finally:
        await inner.aclose()


if __name__ == "__main__":
    asyncio.run(main())
