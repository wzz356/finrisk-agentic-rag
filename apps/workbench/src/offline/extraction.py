"""案例文件读取入口。

本模块只设置案例用途和来源路径，具体格式解析统一由 extraction.FileExtractor 完成。
"""

from collections.abc import Mapping
from pathlib import Path

from domain.common import FilePurpose
from domain.file_models import FileExtractionResult
from extraction import FileExtractor


def extract_case_file(
    file_path: str | Path,
    *,
    extractor: FileExtractor | None = None,
    metadata: Mapping[str, object] | None = None,
) -> FileExtractionResult:
    """读取一个原始案例文件并返回统一结果。"""

    path = Path(file_path)
    active_extractor = extractor or FileExtractor()
    return active_extractor.extract(
        path,
        FilePurpose.CASE,
        source_uri=f"data/raw/cases/{path.name}",
        metadata=metadata,
    )
