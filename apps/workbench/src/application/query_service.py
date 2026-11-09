"""文件/文本 Query 的应用层处理入口。"""

import asyncio
from collections.abc import Mapping
from pathlib import Path

from domain.common import FilePurpose, QueryInputType
from domain.file_models import FileExtractionResult
from domain.query_models import QueryProcessingResult, QueryScene
from extraction import FileExtractor
from llm import LLMClient, create_llm_client_from_env
from query.classification import ClassifierStatus, QuerySceneClassifier
from query.enhancement import prepare_query


class QueryProcessingService:
    """组织文件抽取、六类场景分类、切句和原文映射。"""

    def __init__(
        self,
        *,
        extractor: FileExtractor | None = None,
        classifier: QuerySceneClassifier | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.extractor = extractor or FileExtractor()
        self.classifier = classifier or QuerySceneClassifier(
            llm_client if llm_client is not None else create_llm_client_from_env()
        )

    @property
    def classifier_status(self) -> ClassifierStatus:
        return self.classifier.status

    async def process_text(
        self,
        text: str,
        *,
        metadata: Mapping[str, object] | None = None,
        scene: QueryScene | None = None,
    ) -> QueryProcessingResult:
        """对文字输入进行场景分类、切句和偏移映射。"""

        return await prepare_query(
            text,
            self.classifier,
            input_type=QueryInputType.TEXT,
            metadata=metadata,
            scene_override=scene,
        )

    async def process_file(
        self,
        file_path: str | Path,
        *,
        source_uri: str | None = None,
        metadata: Mapping[str, object] | None = None,
        scene: QueryScene | None = None,
    ) -> QueryProcessingResult:
        """在线程中读取文件，再异步执行规则或 LLM 场景分类。"""

        extraction = await asyncio.to_thread(
            self.extractor.extract,
            file_path,
            FilePurpose.QUERY,
            source_uri=source_uri,
            metadata=metadata,
        )
        return await self.process_extraction(extraction, metadata=metadata, scene=scene)

    async def process_extraction(
        self,
        extraction: FileExtractionResult,
        *,
        metadata: Mapping[str, object] | None = None,
        scene: QueryScene | None = None,
    ) -> QueryProcessingResult:
        """消费已有抽取结果，不再读取原始文件。"""

        if not extraction.full_text.strip():
            raise ValueError("文件未抽取到可处理文字")

        output_metadata: dict[str, object] = dict(metadata or {})
        output_metadata.update(
            {
                "source_uri": extraction.source_uri,
                "media_type": extraction.media_type,
                "extraction_status": extraction.extraction_status.value,
                "extraction_method": extraction.extraction_method.value,
                "extraction_quality_score": extraction.quality_score,
                "extraction_warnings": extraction.warnings,
                "page_count": extraction.page_count,
                "ocr_page_count": extraction.ocr_page_count,
            }
        )
        return await prepare_query(
            extraction.full_text,
            self.classifier,
            input_type=QueryInputType.FILE,
            extraction=extraction,
            metadata=output_metadata,
            scene_override=scene,
        )
