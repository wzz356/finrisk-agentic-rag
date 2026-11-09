"""Conservative source deduplication; a shared page or decision number is insufficient."""

from __future__ import annotations

import re
import unicodedata
from hashlib import sha256


def compact_source(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text.replace("\\n", "\n")))


def source_identity(text: str) -> str:
    return sha256(compact_source(text).encode("utf-8")).hexdigest()
