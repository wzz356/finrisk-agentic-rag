"""Query 人工复核结果的本地可追溯存储。"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from threading import RLock

from domain.review_models import QueryReviewRecord, QueryReviewRequest


class QueryReviewService:
    def __init__(self, output_path: Path, taxonomy: tuple[str, ...]) -> None:
        self.output_path = output_path
        self._taxonomy = set(taxonomy)
        self._lock = RLock()

    def save(self, request: QueryReviewRequest) -> QueryReviewRecord:
        unknown_labels = set(request.corrected_risk_labels) - self._taxonomy
        if unknown_labels:
            raise ValueError(f"修正标签不在官方标签字典中：{'、'.join(sorted(unknown_labels))}")
        reviewed_at = datetime.now(UTC)
        identity = f"{request.query_id}|{request.query_checksum}|{reviewed_at.isoformat()}"
        record = QueryReviewRecord(
            **request.model_dump(),
            review_id=f"QR-{sha256(identity.encode()).hexdigest()[:16].upper()}",
            reviewed_at=reviewed_at,
        )
        line = json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n"
        with self._lock:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with self.output_path.open("a", encoding="utf-8", newline="") as handle:
                handle.write(line)
                handle.flush()
        return record
