"""Query 文件读取入口。

本模块只设置 Query 用途和来源路径，具体格式解析统一由 extraction.FileExtractor 完成。
"""

from collections.abc import Mapping
from pathlib import Path

from domain.common import FilePurpose
from domain.file_models import FileExtractionResult
from extraction import FileExtractor


def extract_query_file(
    file_path: str | Path,
    *,
    extractor: FileExtractor | None = None,
    metadata: Mapping[str, object] | None = None,
) -> FileExtractionResult:
    """读取一个原始 Query 文件并返回统一结果。"""

    path = Path(file_path)
    active_extractor = extractor or FileExtractor()
    return active_extractor.extract(
        path,
        FilePurpose.QUERY,
        source_uri=f"data/raw/query/{path.name}",
        metadata=metadata,
    )
