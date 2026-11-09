"""RapidOCR 适配器。

官方文档使用 RapidOCR() 创建引擎，并通过 result.boxes、result.txts、
result.scores 和 result.word_results 读取位置、文本及置信度。
"""

from collections.abc import Iterator, Sequence
from statistics import fmean
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from domain.common import ExtractionMethod
from extraction.config import ExtractionConfig
from extraction.geometry import normalize_polygon
from extraction.types import ElementDraft, PageDraft, Polygon


class OcrEngine(Protocol):
    """RapidOCR 可调用对象的最小协议，便于测试注入假引擎。"""

    def __call__(
        self,
        image: NDArray[np.uint8],
        *,
        return_word_box: bool,
    ) -> object:
        """识别一张 RGB 图片。"""

        ...


class RapidOcrReader:
    """把 RapidOCR 输出转换为项目内部页面草稿。"""

    def __init__(
        self,
        config: ExtractionConfig,
        engine: OcrEngine | None = None,
    ) -> None:
        """保存配置和可选引擎；默认引擎在首次使用时创建。"""

        self._config = config
        self._engine = engine

    def read(self, image: Image.Image) -> PageDraft:
        """识别图片并保留行级、字级坐标和置信度。"""

        rgb_image = image.convert("RGB")
        image_array = np.asarray(rgb_image)
        result = self._get_engine()(
            image_array,
            return_word_box=self._config.include_word_boxes,
        )

        width, height = rgb_image.size
        raw_texts = _as_list(getattr(result, "txts", None))
        raw_boxes = _as_list(getattr(result, "boxes", None))
        raw_scores = _as_list(getattr(result, "scores", None))

        blocks: list[ElementDraft] = []
        scores: list[float] = []
        for index, raw_text in enumerate(raw_texts):
            text = str(raw_text).strip()
            if not text:
                continue
            confidence = _float_at(raw_scores, index)
            if confidence is not None:
                scores.append(confidence)
            blocks.append(
                ElementDraft(
                    text=text,
                    confidence=confidence,
                    polygon=_polygon_at(raw_boxes, index, width, height),
                    extraction_method=ExtractionMethod.RAPIDOCR,
                )
            )

        words = self._read_words(
            getattr(result, "word_results", None),
            width,
            height,
        )
        average_confidence = fmean(scores) if scores else None
        warnings: list[str] = []
        if not blocks:
            warnings.append("RapidOCR 未识别到文本")
        if average_confidence is not None and average_confidence < self._config.min_ocr_confidence:
            warnings.append(
                "RapidOCR 平均置信度低于配置阈值："
                f"{average_confidence:.4f} < {self._config.min_ocr_confidence:.4f}"
            )

        return PageDraft(
            width=float(width),
            height=float(height),
            blocks=blocks,
            words=words,
            used_ocr=True,
            average_confidence=average_confidence,
            quality_score=average_confidence or 0.0,
            extraction_method=ExtractionMethod.RAPIDOCR,
            warnings=warnings,
        )

    def _get_engine(self) -> OcrEngine:
        """延迟创建 RapidOCR，避免原生文本文件无意义加载模型。"""

        if self._engine is None:
            from rapidocr import RapidOCR

            self._engine = cast(OcrEngine, RapidOCR())
        return self._engine

    def _read_words(
        self,
        raw_word_results: object,
        width: int,
        height: int,
    ) -> list[ElementDraft]:
        """解析 return_word_box=True 时返回的字词级结果。"""

        words: list[ElementDraft] = []
        for values in _iter_word_values(raw_word_results):
            text = str(values[0]).strip()
            if not text:
                continue
            words.append(
                ElementDraft(
                    text=text,
                    confidence=_as_float(values[1]),
                    polygon=_normalize_box(values[2], width, height),
                    extraction_method=ExtractionMethod.RAPIDOCR,
                )
            )
        return words


def _iter_word_values(value: object) -> Iterator[list[object]]:
    """递归展开 RapidOCR 的行级嵌套字词结果。"""

    values = _as_list(value)
    if len(values) >= 3 and isinstance(values[0], str) and _as_float(values[1]) is not None:
        yield values
        return

    for child in values:
        yield from _iter_word_values(child)


def _as_list(value: object) -> list[object]:
    """把元组、列表或 NumPy 数组安全转换为普通列表。"""

    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return cast(list[object], value.tolist())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _as_float(value: object) -> float | None:
    """把置信度转换为 0 到 1 的浮点数。"""

    try:
        score = float(cast(float | int | str, value))
    except (TypeError, ValueError):
        return None
    return min(max(score, 0.0), 1.0)


def _float_at(values: list[object], index: int) -> float | None:
    """读取指定位置的置信度。"""

    if index >= len(values):
        return None
    return _as_float(values[index])


def _polygon_at(
    boxes: list[object],
    index: int,
    width: int,
    height: int,
) -> Polygon | None:
    """读取并归一化指定行的四边形。"""

    if index >= len(boxes):
        return None
    return _normalize_box(boxes[index], width, height)


def _normalize_box(value: object, width: int, height: int) -> Polygon | None:
    """把 RapidOCR 返回的四点坐标归一化。"""

    points: list[list[float]] = []
    for raw_point in _as_list(value):
        coordinates = _as_list(raw_point)
        if len(coordinates) < 2:
            return None
        x = _as_float_coordinate(coordinates[0])
        y = _as_float_coordinate(coordinates[1])
        if x is None or y is None:
            return None
        points.append([x, y])
    return normalize_polygon(points, float(width), float(height))


def _as_float_coordinate(value: object) -> float | None:
    """把坐标值转换为浮点数。"""

    try:
        return float(cast(float | int | str, value))
    except (TypeError, ValueError):
        return None
