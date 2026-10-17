"""增量文书的保守质量门、原V2构建资产适配及只追加编排。

机械验证不表示语义正确或人工批准；原文、模型候选和所有版本均保留。
"""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from agent_rag.contracts import canonical_json
from agent_rag.v2_snapshot import V2KnowledgeSnapshot, file_hash, rows
from offline.penalty_crawler.filter import BANK_PARTY_RE


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical_json(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


@contextmanager
def update_lock(registry: Path):
    """操作系统持有的非阻塞锁，进程异常退出后可重新获取。"""
    registry.mkdir(parents=True, exist_ok=True)
    stream = (registry / "update.lock").open("a+b")
    stream.seek(0, os.SEEK_END)
    if not stream.tell():
        stream.write(b"0")
        stream.flush()
    stream.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise ValueError("UPDATE_LOCKED") from None
    try:
        yield
    finally:
        stream.seek(0)
        if os.name == "nt":
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def append_jsonl(base: Path, delta: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as out:
        with base.open("rb") as src:
            shutil.copyfileobj(src, out)
        if base.stat().st_size:
            with base.open("rb") as src:
                src.seek(-1, os.SEEK_END)
                if src.read(1) != b"\n":
                    out.write(b"\n")
        with delta.open("rb") as src:
            shutil.copyfileobj(src, out)


def pending_documents(crawl: Path, known: set[str]) -> list[dict]:
    crawl = crawl.resolve()
    result, seen = [], set(known)
    for row in rows(crawl / "index.jsonl") if (crawl / "index.jsonl").is_file() else []:
        path = Path(row["txt_path"])
        if not path.is_absolute():
            path = crawl / path
        path = path.resolve()
        if not path.is_relative_to(crawl):
            raise ValueError("SOURCE_PATH_OUTSIDE_CRAWL")
        if not path.is_file():
            continue
        url = row.get("source_url", "")
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in {"www.nfra.gov.cn", "nfra.gov.cn"}:
            raise ValueError("UNTRUSTED_SOURCE_URL")
        # 与原始导出一致保存换行，不能read_text的通用换行改变证据坐标。
        raw = path.read_bytes()
        text = raw.decode("utf-8-sig")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen or "保险" not in text or BANK_PARTY_RE.search(text):
            continue
        seen.add(digest)
        result.append(
            {
                **row,
                "txt_path": str(path),
                "full_text": text,
                "text_sha256": digest,
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "file_id": "WEB-" + digest[:32],
                "text_version": 1,
                "original_filename": path.name,
                "origin_key": f"nfra:{row.get('column')}:{row.get('doc_id')}:{row.get('part')}",
            }
        )
    return result


def _nonempty(value, code="INVALID_DRAFT"):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(code)


def validate_draft(value: dict, fid: str, text: str, tags: set[str]) -> dict:
    """拒绝错引文、未知标签、跨案例关联和模型自行宣布批准。"""
    if (
        not isinstance(value, dict)
        or value.get("file_id") != fid
        or value.get("read_full_text") is not True
    ):
        raise ValueError("INVALID_DRAFT_ID_OR_SCOPE")
    d = copy.deepcopy(value)

    def walk(item):
        if isinstance(item, dict):
            for name, child in item.items():
                if (
                    name.endswith("review_status") or name == "review_status"
                ) and child != "pending":
                    raise ValueError("MODEL_CANNOT_APPROVE")
                if name == "semantic_status" and child != "candidate":
                    raise ValueError("MODEL_CANNOT_APPROVE")
                if name in {"publication_status", "human_approved", "reviewer", "reviewed_at"}:
                    raise ValueError("MODEL_CANNOT_APPROVE")
                if name == "quote" or name.endswith("_quote"):
                    if child is not None and (
                        not isinstance(child, str) or not child.strip() or child not in text
                    ):
                        raise ValueError("UNBOUND_QUOTE")
                if name == "quotes" or name.endswith("_quotes"):
                    if not isinstance(child, list) or any(
                        not isinstance(q, str) or not q.strip() or q not in text for q in child
                    ):
                        raise ValueError("UNBOUND_QUOTE")
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)

    walk(d)
    cases, behaviors = d.get("cases"), d.get("behaviors")
    if not isinstance(cases, list) or not cases or not isinstance(behaviors, list) or not behaviors:
        raise ValueError("EMPTY_CASE_OR_BEHAVIOR")
    keys = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("INVALID_CASE")
        for field in ("local_key", "party_name", "party_quote", "party_type", "case_summary"):
            _nonempty(case.get(field))
        if case["local_key"] in keys or case["party_name"] not in case["party_quote"]:
            raise ValueError("INVALID_PARTY_OR_CASE_KEY")
        if "、" in case["party_name"] or "及相关" in case["party_name"]:
            raise ValueError("COMBINED_PARTIES_NEEDS_REVIEW")
        if not case.get("case_summary_quotes"):
            raise ValueError("UNSUPPORTED_CASE_SUMMARY")
        keys.add(case["local_key"])
        for field in ("penalties", "legal_refs", "related_party_names", "unknowns"):
            case.setdefault(field, [])
            if not isinstance(case[field], list):
                raise ValueError("INVALID_CASE_COLLECTION")
        case.setdefault("institution_hierarchy", {})
        if not isinstance(case["institution_hierarchy"], dict):
            raise ValueError("INVALID_INSTITUTION_HIERARCHY")
        for collection in ("penalties", "legal_refs"):
            for item in case[collection]:
                if not isinstance(item, dict) or not (item.get("quote") or item.get("quotes")):
                    raise ValueError("INVALID_PENALTY_OR_LEGAL_REF")
        for field, quotation in (
            ("penalty_doc_no", "doc_no_quote"),
            ("regulator", "regulator_quote"),
            ("penalty_date", "penalty_date_quote"),
            ("behavior_period", "behavior_period_quote"),
        ):
            if case.get(field) is not None and not case.get(quotation):
                raise ValueError("UNSUPPORTED_CASE_METADATA")
    bkeys = set()
    for behavior in behaviors:
        if not isinstance(behavior, dict) or behavior.get("case_key") not in keys:
            raise ValueError("INVALID_BEHAVIOR_CASE")
        for field in ("local_key", "actor_role", "summary", "finding_quote", "action"):
            _nonempty(behavior.get(field))
        if behavior["local_key"] in bkeys:
            raise ValueError("DUPLICATE_BEHAVIOR_KEY")
        bkeys.add(behavior["local_key"])
        risk = behavior.get("risk_tags")
        if not isinstance(risk, list) or len(risk) > 3:
            raise ValueError("INVALID_RISK_TAGS")
        used = set()
        for tag in risk:
            if (
                not isinstance(tag, dict)
                or tag.get("tag") not in tags
                or tag["tag"] in used
                or not tag.get("quotes")
            ):
                raise ValueError("INVALID_RISK_TAGS")
            used.add(tag["tag"])
            tag["assignment_scope"] = "behavior_context"
        if not risk:
            _nonempty(behavior.get("zero_tag_reason"), "ZERO_TAG_REASON_REQUIRED")
        for field in ("conditions", "negations", "exceptions", "uncertainties"):
            if not isinstance(behavior.get(field), list):
                raise ValueError("QUALIFIERS_MUST_BE_EXPLICIT_ARRAYS")
            for item in behavior[field]:
                if not isinstance(item, dict) or not (item.get("quote") or item.get("quotes")):
                    raise ValueError("UNSUPPORTED_QUALIFIER")
        # 未出现的渠道/产品不能从处罚文书载体反推出实际营销渠道。
        behavior.setdefault("observed_scene", {})
        behavior.setdefault("applicable_scenes", [])
        behavior.setdefault("material_function", "unknown")
        if not isinstance(behavior["observed_scene"], dict) or not isinstance(
            behavior["applicable_scenes"], list
        ):
            raise ValueError("INVALID_SCENE")
        for field in (
            "materials",
            "factor_values",
            "risk_mechanism_quotes",
            "material_function_quotes",
        ):
            behavior.setdefault(field, [])
            if not isinstance(behavior[field], list):
                raise ValueError("INVALID_BEHAVIOR_COLLECTION")
        if any(
            not isinstance(item, dict)
            for field in ("materials", "factor_values", "applicable_scenes")
            for item in behavior[field]
        ):
            raise ValueError("INVALID_BEHAVIOR_COLLECTION")
        if any(not item.get("factor_id") for item in behavior["factor_values"]):
            raise ValueError("INVALID_FACTOR")
    # 同一笼统案由不能被自动扩展为多个主体各自实施该行为。
    if len(cases) > 1:
        by_case = {case["local_key"]: case for case in cases}
        for behavior in behaviors:
            shared = [b for b in behaviors if b["finding_quote"] == behavior["finding_quote"]]
            if (
                len({b["case_key"] for b in shared}) > 1
                and by_case[behavior["case_key"]]["party_name"] not in behavior["finding_quote"]
            ):
                raise ValueError("CROSS_PARTY_FINDING_NEEDS_REVIEW")
    for field in ("coverage", "unresolved", "source_issues"):
        d.setdefault(field, [])
        if not isinstance(d[field], list):
            raise ValueError("INVALID_DRAFT_COLLECTION")
    for coverage in d["coverage"]:
        if not isinstance(coverage, dict) or not coverage.get("quote"):
            raise ValueError("INVALID_COVERAGE")
        if not set(coverage.get("case_keys", [])).issubset(keys) or not set(
            coverage.get("behavior_keys", [])
        ).issubset(bkeys):
            raise ValueError("INVALID_COVERAGE_LINK")
    for field in ("insurance_relevance", "marketing_relevance"):
        relevance = d.get(field)
        if not isinstance(relevance, dict) or type(relevance.get("value")) not in (
            bool,
            type(None),
        ):
            raise ValueError("INVALID_RELEVANCE")
        if relevance.get("value") is not None and not relevance.get("quotes"):
            raise ValueError("UNSUPPORTED_RELEVANCE")
    if d["insurance_relevance"]["value"] is not True:
        raise ValueError("INSURANCE_SCOPE_NEEDS_REVIEW")
    d.update(
        human_review_status="pending",
        semantic_status="candidate",
        creator="deepseek-flash",
        model_or_prompt_version="web-update-v1",
    )
    return d


def _load_asset(asset: dict, name: str):
    path = Path(asset["path"]).resolve()
    if file_hash(path) != asset["sha256"]:
        raise ValueError("BUILD_ASSET_HASH_MISMATCH")
    spec = importlib.util.spec_from_file_location("_finrisk_update_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compile_delta(config: dict, accepted: list[tuple[dict, dict]], out: Path) -> dict:
    assets = config["build_assets"]
    normalizer = _load_asset(assets["normalizer"], "normalizer")
    projector = _load_asset(assets["projector"], "projector")
    contract = Path(assets["contract"]["path"])
    if file_hash(contract) != assets["contract"]["sha256"]:
        raise ValueError("BUILD_ASSET_HASH_MISMATCH")
    normalizer.CONTRACT_PATH = contract
    candidates, sources = {}, {}
    for source, draft in accepted:
        fid = source["file_id"]
        candidates[fid] = {
            "file_id": fid,
            "draft": draft,
            "draft_json_sha256": normalizer.json_sha(draft),
            "human_review_status": "pending",
            "semantic_status": "candidate",
            "batch": out.parent.name,
        }
        sources[fid] = {
            **source,
            "document_metadata": {
                "title": source.get("title"),
                "source_url": source["source_url"],
                "origin_key": source["origin_key"],
                "original_filename": source["original_filename"],
                "extraction_method": "existing_nfra_crawler_export",
                "encoding": "utf-8",
                "text_quality_flags": ["automated_candidate_pending_human_review"],
            },
        }
    manifest = normalizer.normalize_records(candidates, sources, out)
    check = normalizer.verify(out)
    write_json(out / "mechanical-verification.json", check)
    if not check["passed"] or list(rows(out / "extensions/issues.jsonl")):
        raise ValueError("V2_NORMALIZATION_FAILED")
    delta = out.parent / "structured-delta.jsonl"
    audits, count = {}, 0
    with delta.open("x", encoding="utf-8", newline="\n") as stream:
        for fid, wrapper in candidates.items():
            units, audit = projector.project_document(
                fid, sources[fid]["full_text"], wrapper["draft"], wrapper["draft_json_sha256"]
            )
            audits[fid] = audit
            for unit in units:
                if unit["corpus"] == "structured":
                    stream.write(canonical_json(unit) + "\n")
                    count += 1
    write_json(out.parent / "projection-audit.json", audits)
    if not count:
        raise ValueError("EMPTY_PROJECTION")
    return {
        "documents": out / "data/documents.jsonl",
        "index": delta,
        "units": count,
        "normalization_counts": manifest["counts"],
    }


def prompt_for(source: dict, definitions: list[dict]) -> tuple[str, str]:
    system = (
        Path(__file__).resolve().parents[2] / "resources/agent_rag/web-update-extraction.md"
    ).read_text(encoding="utf-8")
    return system, canonical_json(
        {
            "file_id": source["file_id"],
            "title": source.get("title"),
            "official_labels": definitions,
            "full_text": source["full_text"],
        }
    )


def generate_candidate(
    source: dict,
    cache: Path,
    client_factory,
    ledger: Path,
    cap_cny: float,
    max_requests: int,
    definitions: list[dict],
) -> dict:
    if len(source["full_text"]) > 18000:
        raise ValueError("LONG_SOURCE_NEEDS_REVIEW_NO_TRUNCATION")
    cache.mkdir(parents=True, exist_ok=True)
    state_path, candidate = cache / "state.json", cache / "candidate.json"
    tags = {r["risk_tag"] for r in definitions}
    if candidate.exists():
        return validate_draft(
            json.loads(candidate.read_text(encoding="utf-8")),
            source["file_id"],
            source["full_text"],
            tags,
        )
    resuming = state_path.exists()
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {
            "ledger": str(ledger.resolve()),
            "cap_cny": cap_cny,
            "max_requests": max_requests,
            "attempt": 0,
            "errors": [],
            "text_sha256": source["text_sha256"],
        }
    )
    if state["text_sha256"] != source["text_sha256"]:
        raise ValueError("CANDIDATE_SOURCE_DRIFT")
    if state["attempt"] >= 2:
        raise ValueError("CANDIDATE_NEEDS_HUMAN_REVIEW")
    if resuming and not Path(state["ledger"]).is_file():
        raise RuntimeError("UPDATE_MODEL_LEDGER_MISSING")
    if resuming:
        # 只读打开已有审计；空文件/空schema不能被客户端当作新账本初始化。
        try:
            uri = Path(state["ledger"]).resolve().as_uri() + "?mode=ro"
            with sqlite3.connect(uri, uri=True) as connection:
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                policy = connection.execute(
                    "SELECT version,purpose FROM policy WHERE id=1"
                ).fetchone()
                if tables != {"policy", "requests"} or policy != (1, "case_generation"):
                    raise ValueError("invalid ledger")
        except (sqlite3.Error, ValueError, OSError):
            raise RuntimeError("UPDATE_MODEL_LEDGER_INVALID") from None
    write_json(cache / "source.json", source)
    write_json(state_path, state)
    client = client_factory(Path(state["ledger"]), state["cap_cny"], state["max_requests"])
    system, user = prompt_for(source, definitions)
    # 当前文书若中断，保持同一请求键；客户端只回放成功或阻止未知请求重发。
    while state["attempt"] < 2:
        attempt = state["attempt"]
        repair = "\n上次机械校验错误：" + ",".join(state["errors"]) if attempt else ""
        request_key = source["text_sha256"] + f":{attempt}"
        replay = Path(state["ledger"]).resolve() != ledger.resolve()
        try:
            raw = client.complete(
                system + repair,
                user,
                request_key=request_key,
                max_tokens=6000,
                **({"replay_only": True} if replay else {}),
            )
        except RuntimeError as error:
            if not replay or str(error) != "UPDATE_MODEL_REPLAY_NOT_AVAILABLE":
                raise
            # 只有旧账本明确证明此键从未发送，才将新请求迁移到本轮唯一预算。
            state.setdefault("previous_ledgers", []).append(state["ledger"])
            state.update(ledger=str(ledger.resolve()), cap_cny=cap_cny, max_requests=max_requests)
            write_json(state_path, state)
            client = client_factory(ledger, cap_cny, max_requests)
            raw = client.complete(system + repair, user, request_key=request_key, max_tokens=6000)
        (cache / f"response-{attempt}.txt").write_text(raw, encoding="utf-8")
        try:
            draft = validate_draft(json.loads(raw), source["file_id"], source["full_text"], tags)
        except (ValueError, TypeError, KeyError) as error:
            state["errors"].append(str(error).split(":", 1)[0][:120])
            state["attempt"] += 1
            write_json(state_path, state)
            continue
        write_json(candidate, draft)
        state["status"] = "mechanically_valid_pending_human_review"
        write_json(state_path, state)
        return draft
    raise ValueError("CANDIDATE_NEEDS_HUMAN_REVIEW")


def run_update(
    config: dict,
    *,
    max_documents=3,
    generate=False,
    publish=False,
    cap_cny=1.0,
    max_requests=6,
    key_file=None,
    vector_builder=None,
    model_factory=None,
    include_review=False,
) -> dict:
    from agent_rag.knowledge_registry import (
        current_snapshot,
        publish_snapshot,
        register_base_snapshot,
    )
    from agent_rag.update_model import UpdateModelClient
    from agent_rag.update_vectors import build_dense_extension

    registry, crawl = Path(config["registry"]).resolve(), Path(config["crawl_dir"]).resolve()
    if not 1 <= max_documents <= 100 or not 1 <= max_requests <= 200 or not 0 < cap_cny <= 100:
        raise ValueError("INVALID_UPDATE_LIMITS")
    if publish and not generate:
        raise ValueError("PUBLISH_REQUIRES_EXPLICIT_GENERATION")
    with update_lock(registry):
        base_path = (
            current_snapshot(registry)
            if (registry / "current.json").exists()
            else Path(config["base_snapshot"])
        )
        base = V2KnowledgeSnapshot(base_path)
        held = {p.parent.name for p in (registry / "candidates").glob("*/disposition.json")}
        known = {s["text_sha256"] for s in base.sources.values()}
        pending = pending_documents(crawl, known | (set() if include_review else held))
        result = {
            "pending_documents": len(pending),
            "selected": min(len(pending), max_documents),
            "held_documents": len(held - known),
            "base_snapshot_id": base.manifest["snapshot_id"],
            "published": False,
            "human_review_status": "pending",
            "semantic_coverage_verified": False,
        }
        if not generate or not pending:
            return {**result, "status": "no_new_documents" if not pending else "dry_run"}
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        run = registry / "runs" / stamp
        run.mkdir(parents=True)
        definitions = list(
            csv.DictReader(Path(base.assets["dictionary"]["path"]).open(encoding="utf-8-sig"))
        )
        factory = model_factory or (
            lambda ledger, cap, count: UpdateModelClient(
                ledger, cap_cny=cap, max_requests=count, key_file=key_file
            )
        )
        accepted, review = [], []
        origins = {s.get("origin_key"): s for s in base.sources.values() if s.get("origin_key")}
        for source in pending[:max_documents]:
            cache = registry / "candidates" / source["text_sha256"]
            if source["origin_key"] in origins:
                entry = {
                    "file_id": source["file_id"],
                    "code": "CHANGED_OFFICIAL_SOURCE_NEEDS_REVIEW",
                    "supersedes_file_id": origins[source["origin_key"]]["official_file_id"],
                    "status": "needs_manual_review",
                    "run": str(run),
                    "candidate_dir": str(cache),
                }
                write_json(cache / "source.json", source)
                write_json(cache / "disposition.json", entry)
                review.append(entry)
                continue
            try:
                draft = generate_candidate(
                    source,
                    cache,
                    factory,
                    run / "generation.sqlite",
                    cap_cny,
                    max_requests,
                    definitions,
                )
                compile_delta(
                    config,
                    [(source, draft)],
                    run / "document-checks" / source["file_id"] / "normalized",
                )
                (cache / "disposition.json").unlink(missing_ok=True)
                accepted.append((source, draft))
            except (ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
                entry = {
                    "file_id": source["file_id"],
                    "code": str(error).split(":", 1)[0][:120],
                    "candidate_dir": str(cache),
                    "status": "needs_manual_review",
                    "run": str(run),
                }
                write_json(cache / "disposition.json", entry)
                review.append(entry)
        result.update(run_dir=str(run), accepted_documents=len(accepted), review_queue=review)
        if model_factory is None and (run / "generation.sqlite").exists():
            result["generation_budget"] = UpdateModelClient(
                run / "generation.sqlite", cap_cny, max_requests, key_file=key_file
            ).summary()
        elif not (run / "generation.sqlite").exists():
            result["generation_budget"] = {
                "requests": 0,
                "settled_cost_cny": 0,
                "reserved_cost_cny": 0,
            }
        write_json(run / "review-queue.json", {"items": review})
        if not accepted:
            result["status"] = "review_required"
            write_json(run / "result.json", result)
            return result
        delta = compile_delta(config, accepted, run / "normalized")
        # 以最终路径构建；完成标记/验证前不会有current指针引用此目录。
        version = registry / "versions" / stamp
        version.mkdir(parents=True)
        try:
            sources, index = version / "documents.jsonl", version / "structured.jsonl"
            append_jsonl(Path(base.assets["sources"]["path"]), delta["documents"], sources)
            append_jsonl(Path(base.assets["index"]["path"]), delta["index"], index)
            dictionary = version / "risk_type_dictionary.csv"
            shutil.copyfile(base.assets["dictionary"]["path"], dictionary)
            old_dense = Path(base.manifest["backend"]["assets"]["dense_config"]["path"])
            dense = (vector_builder or build_dense_extension)(old_dense, index, version / "dense")
            rule = version / "rerank-rules.md"
            shutil.copyfile(base.manifest["backend"]["assets"]["rules"]["path"], rule)
            snapshot = V2KnowledgeSnapshot.create(sources, index, dictionary, version / "snapshot")
            snapshot.manifest.update(
                knowledge_version="finrisk-v2-web-update-v1",
                corpus_scope="incremental-v1",
                parent_snapshot_id=base.manifest["snapshot_id"],
                generation_run=str(run),
                human_approved=False,
                semantic_coverage_verified=False,
                rerank_mode="live",
                backend={
                    "mode": "live",
                    "rerank_protocol": "span-ref-v2",
                    "assets": {
                        "dense_config": {"path": str(dense.resolve()), "sha256": file_hash(dense)},
                        "rules": {"path": str(rule.resolve()), "sha256": file_hash(rule)},
                    },
                },
            )
            snapshot.manifest["snapshot_id"] = (
                "v2-"
                + hashlib.sha256(
                    canonical_json(
                        {
                            "core": snapshot.manifest["snapshot_id"],
                            "dense": file_hash(dense),
                            "rules": file_hash(rule),
                            "protocol": "span-ref-v2",
                        }
                    ).encode()
                ).hexdigest()[:24]
            )
            write_json(version / "snapshot/manifest.json", snapshot.manifest)
            for unit in snapshot.units:
                snapshot.candidate(unit, 1, 0.0)
            result.update(
                status="built",
                version=stamp,
                snapshot_path=str(version / "snapshot"),
                snapshot_id=snapshot.manifest["snapshot_id"],
                new_units=delta["units"],
                source_count=len(snapshot.sources),
                chunk_count=len(snapshot.units),
            )
            write_json(version / "update-complete.json", result)
            if publish:
                bootstrap = register_base_snapshot(registry, Path(config["base_snapshot"]))
                if not (registry / "current.json").exists():
                    publish_snapshot(registry, bootstrap)
                result["publication"] = publish_snapshot(registry, version / "snapshot")
                result.update(status="published", published=True)
        except BaseException:
            write_json(version / "update-failed.json", {"status": "failed_not_published"})
            raise
        write_json(run / "result.json", result)
        return result
