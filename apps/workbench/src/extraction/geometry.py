"""页面坐标归一化工具。"""

from collections.abc import Sequence

from extraction.types import Polygon


def normalize_polygon(
    points: Sequence[Sequence[float]],
    width: float,
    height: float,
) -> Polygon | None:
    """把四边形坐标转换为相对页面宽高的 0 到 1 坐标。"""

    if width <= 0 or height <= 0 or len(points) != 4:
        return None

    normalized: list[tuple[float, float]] = []
    for point in points:
        if len(point) < 2:
            return None
        x = min(max(float(point[0]) / width, 0.0), 1.0)
        y = min(max(float(point[1]) / height, 0.0), 1.0)
        normalized.append((x, y))

    return (normalized[0], normalized[1], normalized[2], normalized[3])


def normalize_rectangle(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    width: float,
    height: float,
) -> Polygon | None:
    """把矩形转换为顺时针四边形并归一化。"""

    return normalize_polygon(
        ((x0, y0), (x1, y0), (x1, y1), (x0, y1)),
        width,
        height,
    )
