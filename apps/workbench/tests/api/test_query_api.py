"""FastAPI Query 分类、切句、风险判断与案例匹配接口测试。"""

from fastapi.testclient import TestClient

from api.app import app, get_query_service
from application.query_service import QueryProcessingService
from query.classification import QuerySceneClassifier


def _rule_only_service() -> QueryProcessingService:
    return QueryProcessingService(classifier=QuerySceneClassifier(None))


app.dependency_overrides[get_query_service] = _rule_only_service
client = TestClient(app)


def test_index_serves_workbench() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "风险查询" in response.text
    assert "案例录入" in response.text
    assert "案例管理" in response.text


def test_classifier_status_reports_rule_and_llm_state() -> None:
    response = client.get("/api/query/classifier")

    assert response.status_code == 200
    assert response.json()["llm_configured"] is False
    assert response.json()["taxonomy_version"] == "query-scene-2.0"


def test_text_query_endpoint_returns_risk_and_mapping_fields() -> None:
    response = client.post(
        "/api/query/analyze/text",
        json={"text": "营销海报，限时活动现已开启！立即扫码领取专属权益。"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["classification"]["scene"] == "ST1_poster_sms_social"
    assert len(payload["sentences"]) == 2
    assert set(payload["sentences"][0]) == {"query_id", "sentence_id", "scene", "text"}
    assert payload["sentences"][0]["sentence_id"] == payload["source_mappings"][0]["sentence_id"]
    assert payload["schema_version"] == "4.0"
    assert payload["extraction"]["checksum_sha256"]
    assert "risk_findings" in payload["risk_summary"]
    assert len(payload["risk_assessments"]) == len(payload["sentences"])
    assert 0 <= payload["risk_summary"]["overall_score"] <= 1


def test_file_query_endpoint_returns_source_regions() -> None:
    response = client.post(
        "/api/query/analyze/file",
        files={
            "file": (
                "callback.txt",
                "回访话术。坐席：您好，本次致电进行客户回访；客户：好的。",
                "text/plain",
            )
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["classification"]["scene"] == "ST5_training_telemarketing_callback"
    assert payload["source_filename"] == "callback.txt"
    assert payload["source_mappings"][0]["regions"]


def test_ambiguous_text_without_llm_uses_local_fallback() -> None:
    response = client.post(
        "/api/query/analyze/text",
        json={"text": "这是今天整理的一份材料。"},
    )

    assert response.status_code == 200
    assert response.json()["classification"]["method"] == "rule_fallback"


def test_text_query_accepts_user_selected_scene() -> None:
    response = client.post(
        "/api/query/analyze/text",
        json={
            "text": "这是一份很难仅靠文本判断载体的材料。",
            "scene": "ST4_meeting_material",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["classification"]["scene"] == "ST4_meeting_material"
    assert payload["classification"]["method"] == "user"
