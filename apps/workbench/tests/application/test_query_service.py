"""Query 应用服务和文件原文定位测试。"""

import asyncio
from pathlib import Path

from application.query_service import QueryProcessingService
from domain.common import QueryInputType
from domain.query_models import QueryScene
from query.classification import QuerySceneClassifier


def test_service_processes_file_into_risk_sentences_and_source_regions(tmp_path: Path) -> None:
    """文件读取后应同时产生 Risk 四字段输入和可回溯文本块。"""

    file_path = tmp_path / "penalty.txt"
    file_path.write_text(
        "行政处罚决定书。\n经查，该机构违反相关规定。\n决定给予警告并处罚款。",
        encoding="utf-8",
    )
    service = QueryProcessingService(classifier=QuerySceneClassifier(None))

    result = asyncio.run(service.process_file(file_path, source_uri="upload://penalty.txt"))

    assert result.input_type is QueryInputType.FILE
    assert result.source_filename == "penalty.txt"
    assert result.classification.scene is QueryScene.ST6_PENALTY_BACKEND_FACT
    assert len(result.sentences) == 3
    assert result.metadata["source_uri"] == "upload://penalty.txt"
    assert all(mapping.regions for mapping in result.source_mappings)
    for sentence, mapping in zip(result.sentences, result.source_mappings, strict=True):
        assert sentence.sentence_id == mapping.sentence_id
        assert (
            result.source_text[mapping.document_start_offset : mapping.document_end_offset]
            == sentence.text
        )
        assert mapping.regions[0].block_id.startswith("p1-b")
    assert result.source_mappings[1].context_before == result.sentences[0].text
    assert result.source_mappings[1].context_after == result.sentences[2].text
    assert result.extraction.checksum_sha256 == result.source_document_id
    assert result.extraction.block_count >= 3
