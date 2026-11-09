"""The published runtime bundle is sufficient and still rejects altered evidence."""

import json
import shutil
from pathlib import Path

import pytest

from application.knowledge_release import P0KnowledgeRelease

ASSETS = Path(__file__).resolve().parents[2] / "data/runtime_knowledge/p0_20261018"


def versions():
    return json.loads((ASSETS / "contracts/runtime_manifest.json").read_text("utf-8"))[
        "full_reader"
    ]["versions"]


def test_runtime_bundle_needs_neither_upstream_code_nor_retired_queries():
    release = P0KnowledgeRelease(ASSETS, versions())
    assert len(release.cards) == 12 and release.articles
    assert not (ASSETS / "code").exists()
    assert not (ASSETS / "evaluation_new36").exists()
    assert not list(ASSETS.rglob("*expected*"))
    for article in release.articles.values():
        assert release.read_article(article["article_id"])["literal_verified"]


@pytest.mark.parametrize("kind", ["snapshot", "mechanism"])
def test_runtime_bundle_rejects_changed_source_or_mechanism(tmp_path, kind):
    target = tmp_path / "runtime"
    shutil.copytree(ASSETS, target)
    if kind == "mechanism":
        path = target / "assets/expanded/knowledge/mechanisms.jsonl"
    else:
        path = next((target / "assets/legal/snapshots").glob("core_*.txt"))
    with path.open("a", encoding="utf-8") as stream:
        stream.write("modified")
    with pytest.raises(ValueError, match="payload"):
        P0KnowledgeRelease(target, versions())
