"""Query 人工复核留痕服务与 API 测试。"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from api.app import app, get_review_service
from application.review_service import QueryReviewService


def test_review_api_validates_taxonomy_and_appends_trace_record(tmp_path: Path) -> None:
    output_path = tmp_path / "reviews" / "query_reviews.jsonl"
    service = QueryReviewService(output_path, ("销售误导", "夸大收益"))
    app.dependency_overrides[get_review_service] = lambda: service
    client = TestClient(app)
    try:
        response = client.post(
            "/api/query/reviews",
            json={
                "query_id": "Q-001",
                "query_checksum": "0123456789abcdef",
                "status": "corrected",
                "reviewer": "审核员甲",
                "comment": "删除过度扩展标签",
                "original_risk_labels": ["销售误导", "夸大收益"],
                "corrected_risk_labels": ["销售误导"],
                "corrected_scene": "ST3_product_intro_detail",
            },
        )
    finally:
        app.dependency_overrides.pop(get_review_service, None)

    assert response.status_code == 201
    payload = response.json()
    assert payload["review_id"].startswith("QR-")
    stored = json.loads(output_path.read_text(encoding="utf-8"))
    assert stored["reviewer"] == "审核员甲"
    assert stored["corrected_risk_labels"] == ["销售误导"]


def test_review_api_rejects_unknown_corrected_label(tmp_path: Path) -> None:
    service = QueryReviewService(tmp_path / "reviews.jsonl", ("销售误导",))
    app.dependency_overrides[get_review_service] = lambda: service
    client = TestClient(app)
    try:
        response = client.post(
            "/api/query/reviews",
            json={
                "query_id": "Q-002",
                "query_checksum": "0123456789abcdef",
                "status": "corrected",
                "corrected_risk_labels": ["不存在标签"],
            },
        )
    finally:
        app.dependency_overrides.pop(get_review_service, None)

    assert response.status_code == 422
    assert "官方标签字典" in response.json()["detail"]
