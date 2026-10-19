import hashlib
import importlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest


def make_store(root):
    try:
        module = importlib.import_module("agent_rag.collection_store")
    except ModuleNotFoundError:
        pytest.fail("CollectionStore has not been implemented")
    return module.CollectionStore(root)


def metadata(doc_id="1", **extra):
    return {
        "title": "行政处罚决定书",
        "source_url": f"https://www.nfra.gov.cn/cn/view/pages/ItemDetail.html?docId={doc_id}",
        "publish_date": "2026-10-18",
        "origin": "official_crawl",
        "official_fields": {"docId": doc_id, "itemId": "4113"},
        **extra,
    }


def test_creates_persistent_batches_and_json_settings(tmp_path):
    store = make_store(tmp_path / "inventory")
    first = store.create_batch("legacy_import", {"count": 2})
    second = store.create_batch("official_crawl")
    assert store.root == (tmp_path / "inventory").resolve()
    assert first["id"] != second["id"]
    assert first["kind"] == "legacy_import"
    assert first["metadata"] == {"count": 2}
    assert first["status"] == "pending"
    assert first["created_at"]
    store.set_batch_status(first["id"], "incomplete", {"failed_page": 3})
    store.set_value("job", {"state": "running", "cursor": [1, 2]})
    reopened = make_store(store.root)
    assert reopened.get_value("job") == {"state": "running", "cursor": [1, 2]}
    assert reopened.get_value("missing", False) is False
    batches = reopened.list_batches()
    assert [row["id"] for row in batches] == [second["id"], first["id"]]
    assert batches[1]["status"] == "incomplete"
    assert batches[1]["details"] == {"failed_page": 3}
    assert batches[1]["documents"] == 0
    assert reopened.counts()["batches"] == 2


def test_digest_uses_bom_decoded_text_without_changing_literal_newlines(tmp_path):
    store = make_store(tmp_path)
    batch = store.create_batch("official_crawl")["id"]
    text = "甲保险公司\r\n第一行\n第二行\r末行"
    raw = b"\xef\xbb\xbf" + text.encode("utf-8")
    result = store.ingest(batch, raw, metadata())
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert result == {"digest": digest, "disposition": "new"}
    document = store.read_document(digest)
    assert document["text"] == text
    assert document["metadata"] == metadata()
    assert document["contains_insurance"] is True
    assert any(path.read_bytes() == raw for path in (tmp_path / "blobs").iterdir())


def test_cross_batch_dedup_preserves_all_sources_and_raw_versions(tmp_path):
    store = make_store(tmp_path)
    first = store.create_batch("legacy_import")["id"]
    second = store.create_batch("official_crawl")["id"]
    raw = "银行处罚文书，未涉及其他业务。".encode()
    original = metadata("1", origin="legacy_import", official_fields={"original": True})
    digest = store.ingest(first, raw, original)["digest"]
    assert store.ingest(second, b"\xef\xbb\xbf" + raw, metadata("2")) == {
        "digest": digest,
        "disposition": "duplicate",
    }
    assert store.ingest(second, b"\xef\xbb\xbf" + raw, metadata("2"))["disposition"] == "duplicate"
    document = store.read_document(digest)
    assert document["contains_insurance"] is False
    assert len(document["sources"]) == 2
    assert {source["batch_id"] for source in document["sources"]} == {first, second}
    assert {source["metadata"]["source_url"] for source in document["sources"]} == {
        original["source_url"],
        metadata("2")["source_url"],
    }
    assert len(document["raw_versions"]) == 2
    assert store.counts()["documents"] == 1
    batches = {batch["id"]: batch for batch in store.list_batches()}
    assert batches[first]["new_documents"] == 1
    assert batches[second]["documents"] == 1
    assert batches[second]["duplicates"] == 1
    assert len(list((tmp_path / "blobs").iterdir())) == 2


def test_baseline_duplicates_are_registered_without_losing_originals(tmp_path):
    store = make_store(tmp_path)
    batch = store.create_batch("official_crawl")["id"]
    raw = "基线保险处罚文书".encode()
    digest = hashlib.sha256(raw).hexdigest()
    assert store.ingest(batch, raw, metadata(), {digest})["disposition"] == "baseline_duplicate"
    assert store.read_document(digest)["text"] == raw.decode()
    assert store.documents()["items"][0]["baseline_duplicate"] is True
    assert store.list_batches()[0]["baseline_duplicates"] == 1
    assert store.list_batches()[0]["new_documents"] == 0
    assert store.counts()["baseline_duplicates"] == 1


def test_documents_are_paginated_and_public_metadata_has_no_absolute_paths(tmp_path):
    store = make_store(tmp_path)
    batch = store.create_batch("official_crawl")["id"]
    other = store.create_batch("legacy_import")["id"]
    for number in range(5):
        store.ingest(
            batch,
            f"正文{number}".encode(),
            metadata(str(number), txt_path=str(tmp_path / "old.txt")),
        )
    store.ingest(other, b"separate", metadata("other"))
    page = store.documents(batch, offset=1, limit=2)
    assert page["total"] == 5
    assert len(page["items"]) == 2
    assert store.documents(batch, offset=20)["items"] == []
    assert store.documents()["total"] == 6
    assert "txt_path" not in page["items"][0]["metadata"]
    assert str(tmp_path) not in json.dumps(page, ensure_ascii=False)
    assert {"digest", "title", "source_url", "publish_date", "origin", "status"} <= page["items"][
        0
    ].keys()
    for offset, limit in [(-1, 20), (0, 0), (0, -1)]:
        with pytest.raises(ValueError):
            store.documents(offset=offset, limit=limit)


@pytest.mark.parametrize(
    "url",
    [
        "http://www.nfra.gov.cn/a",
        "https://nfra.gov.cn.attacker.invalid/a",
        "https://user@www.nfra.gov.cn/a",
        "file:///private.txt",
        "https://nfra.gov.cn:444/a",
        "https://example.com/a",
        "https://www.nfra.gov.cn\\@example.com/a",
        "",
    ],
)
def test_rejects_non_official_sources_without_creating_inventory_entries(tmp_path, url):
    store = make_store(tmp_path)
    batch = store.create_batch("official_crawl")["id"]
    with pytest.raises(ValueError, match="UNTRUSTED_SOURCE_URL"):
        store.ingest(batch, b"body", metadata(source_url=url))
    assert store.counts()["documents"] == 0
    assert not list((tmp_path / "blobs").iterdir())


@pytest.mark.parametrize("url", ["https://nfra.gov.cn/a", "https://www.nfra.gov.cn:443/a"])
def test_accepts_both_official_https_hosts(tmp_path, url):
    store = make_store(tmp_path)
    batch = store.create_batch("official_crawl")["id"]
    assert store.ingest(batch, b"body", metadata(source_url=url))["disposition"] == "new"


def test_rejects_unknown_batch_and_invalid_utf8_without_orphan_files(tmp_path):
    store = make_store(tmp_path)
    with pytest.raises(ValueError, match="UNKNOWN_BATCH"):
        store.ingest("missing", b"body", metadata())
    batch = store.create_batch("official_crawl")["id"]
    with pytest.raises(ValueError):
        store.ingest(batch, b"\xff\xfe", metadata())
    assert not list((tmp_path / "blobs").iterdir())
    with pytest.raises(ValueError, match="UNKNOWN_BATCH"):
        store.set_batch_status("missing", "done")


def test_read_verifies_raw_hash_and_never_repairs_tampered_originals(tmp_path):
    store = make_store(tmp_path)
    batch = store.create_batch("official_crawl")["id"]
    digest = store.ingest(batch, b"body", metadata())["digest"]
    blob = next((tmp_path / "blobs").iterdir())
    blob.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        store.read_document(digest)
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        store.ingest(batch, b"body", metadata())
    assert blob.read_bytes() == b"tampered"


def test_only_reads_registered_hashes_and_rejects_paths(tmp_path):
    store = make_store(tmp_path)
    unregistered = b"not registered"
    digest = hashlib.sha256(unregistered).hexdigest()
    (tmp_path / "blobs" / f"{digest}.bin").write_bytes(unregistered)
    with pytest.raises(ValueError, match="UNKNOWN_DOCUMENT"):
        store.read_document(digest)
    for path in ["../outside", "C:/private.txt", "/private.txt", "a" * 63, "A" * 64]:
        with pytest.raises(ValueError):
            store.read_document(path)


def test_rejects_corrupted_raw_reference_in_database(tmp_path):
    store = make_store(tmp_path)
    batch = store.create_batch("official_crawl")["id"]
    digest = store.ingest(batch, b"body", metadata())["digest"]
    database = next(tmp_path.glob("*.sqlite3"))
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute(
            "UPDATE documents SET raw_sha256 = ? WHERE digest = ?", ("../outside", digest)
        )
    with pytest.raises(ValueError):
        store.read_document(digest)


def test_rejects_unregistered_canonical_version_even_when_its_text_hash_matches(tmp_path):
    store = make_store(tmp_path)
    batch = store.create_batch("official_crawl")["id"]
    digest = store.ingest(batch, b"body", metadata())["digest"]
    unregistered = b"\xef\xbb\xbfbody"
    raw_hash = hashlib.sha256(unregistered).hexdigest()
    (tmp_path / "blobs" / f"{raw_hash}.bin").write_bytes(unregistered)
    with sqlite3.connect(next(tmp_path.glob("*.sqlite3"))) as connection:
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute(
            "UPDATE documents SET raw_sha256 = ? WHERE digest = ?", (raw_hash, digest)
        )
    with pytest.raises(ValueError, match="UNREGISTERED_ORIGINAL"):
        store.read_document(digest)


def test_read_also_verifies_retained_alternate_raw_versions(tmp_path):
    store = make_store(tmp_path)
    batch = store.create_batch("official_crawl")["id"]
    digest = store.ingest(batch, b"body", metadata())["digest"]
    alternate = b"\xef\xbb\xbfbody"
    store.ingest(batch, alternate, metadata("alternate"))
    path = tmp_path / "blobs" / f"{hashlib.sha256(alternate).hexdigest()}.bin"
    path.write_bytes(b"changed alternate")
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        store.read_document(digest)


def test_concurrent_ingest_across_instances_keeps_one_document_and_every_source(tmp_path):
    store = make_store(tmp_path)
    batch = store.create_batch("official_crawl")["id"]
    stores = [make_store(tmp_path) for _ in range(4)]

    def ingest(number):
        return stores[number % 4].ingest(batch, b"same original", metadata(str(number)))

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(ingest, range(16)))
    assert sum(row["disposition"] == "new" for row in results) == 1
    assert store.counts()["documents"] == 1
    assert store.list_batches()[0]["documents"] == 1
    document = store.read_document(results[0]["digest"])
    assert len(document["sources"]) == 16
    assert len(list((tmp_path / "blobs").iterdir())) == 1
