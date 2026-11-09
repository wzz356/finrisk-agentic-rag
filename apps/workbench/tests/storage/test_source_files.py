"""Legacy PDF links require matching source and file hashes inside allowed roots."""

import json
from dataclasses import replace
from hashlib import sha256

import pytest

from domain.knowledge_models import CaseDetail
from settings import AppSettings
from storage.source_files import SourceFiles


@pytest.mark.parametrize("fault", [None, "changed_pdf", "wrong_source", "outside_root"])
def test_legacy_pdf_binding_checks_both_hashes_and_directory_boundary(tmp_path, fault):
    root = tmp_path / "kb"
    root.mkdir()
    folder = tmp_path / ("private" if fault == "outside_root" else "raw_pdfs")
    folder.mkdir()
    pdf = folder / "原文.pdf"
    pdf.write_bytes(b"%PDF-1.7\nverified original")
    source_hash = sha256(b"registered source text").hexdigest()
    entry = {
        "path": pdf.relative_to(tmp_path).as_posix(),
        "source_sha256": source_hash,
        "pdf_sha256": sha256(pdf.read_bytes()).hexdigest(),
    }
    (root / "source_pdf_index.json").write_text(json.dumps({"files": {"F001": entry}}))
    if fault == "changed_pdf":
        pdf.write_bytes(b"%PDF-1.7\nchanged")
    item = CaseDetail(
        case_id="C001",
        document_id="F001",
        source_sha256="wrong" if fault == "wrong_source" else source_hash,
        violation_behavior="处罚事实",
        case_origin="original",
        retrieval_status="active",
        evidence_status="verified",
        data_version="test",
    )
    settings = replace(
        AppSettings.from_env(),
        project_root=tmp_path,
        case_kb_root=root,
        raw_case_upload_dir=tmp_path / "uploads",
    )
    assert SourceFiles(settings).resolve(item) == (pdf if fault is None else None)
