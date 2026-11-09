"""随项目交付的数据包只读 API 冒烟测试。"""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import api.app as api_module
from api.app import app, get_case_service

client = TestClient(app)


def test_case_statistics_and_detail_are_available() -> None:
    statistics = client.get("/api/cases/statistics")
    page = client.get("/api/cases", params={"page": 1, "page_size": 1})

    assert statistics.status_code == 200
    assert statistics.json()["total_cases"] >= 7_328
    assert page.status_code == 200
    assert page.json()["items"]
    case_id = page.json()["items"][0]["case_id"]
    detail = client.get(f"/api/cases/{case_id}")
    assert detail.status_code == 200
    assert detail.json()["case_id"] == case_id


@pytest.mark.parametrize(
    ("folder", "use_alias", "separator", "expected_status"),
    [
        ("raw_pdfs", False, "/", 200),
        ("raw_pdfs", False, "\\", 200),
        ("raw_pdfs", True, "/", 200),
        ("private", True, "/", 404),
    ],
)
def test_original_sources_support_portable_names_without_bypassing_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    folder: str,
    use_alias: bool,
    separator: str,
    expected_status: int,
) -> None:
    filename = "保险处罚文书" * 20 + ".pdf" if use_alias else "original.pdf"
    stored_name = sha256(filename.encode("utf-8")).hexdigest() + ".pdf" if use_alias else filename
    target = tmp_path / folder / stored_name
    target.parent.mkdir()
    target.write_bytes(b"%PDF-1.7\nportable source\n")
    settings = replace(
        api_module.get_settings(),
        project_root=tmp_path,
        raw_case_upload_dir=tmp_path / "uploads",
    )
    service = SimpleNamespace(
        get_case=lambda _: SimpleNamespace(raw_source_file=f"{folder}{separator}{filename}")
    )
    monkeypatch.setattr(api_module, "get_settings", lambda: settings)
    monkeypatch.setitem(app.dependency_overrides, get_case_service, lambda: service)

    response = client.get("/api/cases/portable-source/source")

    assert response.status_code == expected_status
    if expected_status == 200:
        from urllib.parse import unquote

        assert response.content == target.read_bytes()
        assert filename in unquote(response.headers["content-disposition"])
