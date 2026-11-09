"""模型可调用的三工具；不接受文件路径，不根据历史案例填当前产品事实。"""

import copy
import json
import re
import time
from typing import Any

from agent_rag.contracts import canonical_json, tool_schema, validate_schema
from agent_rag.input_registry import InputRegistry, text_hash
from agent_rag.knowledge_snapshot import KnowledgeSnapshot, make_evidence
from agent_rag.retrieval import PROFILE
from agent_rag.v2_context import POLICY as CONTEXT_POLICY
from agent_rag.v2_context import strip_parent_context
from agent_rag.v2_snapshot import PROFILE as V2_PROFILE


def _last_text_boundary(text: str, lower: int, upper: int) -> int | None:
    """仅在允许的偏移范围内找段末，次选句末，不移动锚点到窗口外。"""
    offset = max(0, lower - 2)
    for pattern in (r"\r\n|[\r\n]", r"[。！？；!?;]"):
        boundaries = [
            offset + match.end()
            for match in re.finditer(pattern, text[offset:upper])
            if lower <= offset + match.end() <= upper
        ]
        if boundaries:
            return boundaries[-1]
    return None


def _duplicate_hint(text: str) -> dict[str, Any] | None:
    """只给有明确字段边界的完整事实段建立原文指纹；不归并事件。"""
    field_prefix = r"(?:^|[\r\n。；;])[\t ]*"
    field_separator = r"[\t ]*(?:[：:][\t ]*|\r\n|[\r\n])"
    headings = list(
        re.finditer(
            field_prefix
            + r"(?:主要违法违规事实|主要违法事实|违法违规事实|违法事实)"
            + r"(?:[（(]案由[)）])?"
            + field_separator,
            text,
        )
    )
    if len(headings) != 1:
        return None
    start = headings[0].end()
    closing = re.search(
        field_prefix
        + r"(?P<field>行政处罚依据|行政处罚决定|行政处罚内容|处罚依据|处罚决定|处罚内容)"
        + field_separator,
        text[start:],
    )
    if not closing:
        return None
    # 只除去字段分隔空白；事实内部的空格、换行和标点均参与哈希。
    end = start + closing.start("field")
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    facts = text[start:end]
    if len(facts) < 80:
        return None
    return {
        "kind": "exact_fact_text",
        "fact_text_sha256": text_hash(facts),
        "fact_start": start,
        "fact_end": end,
        "fact_chars": len(facts),
        "event_review_status": "not_assessed",
        "caution": "仅文本重复线索，未经事件审核；不得据此合并文书或主体，也不得直接跳过未知来源。",
    }


def envelope(status: str, data: Any, meta: dict[str, Any], error: Any = None) -> dict[str, Any]:
    return {
        "schema_version": "finrisk-tools/1.0",
        "status": status,
        "data": data,
        "meta": meta,
        "error": error,
    }


class KnowledgeTools:
    def __init__(
        self,
        snapshot: KnowledgeSnapshot,
        registry: InputRegistry,
        profile: str | None = None,
        *,
        control: Any = None,
        audit_dir: Any = None,
    ) -> None:
        profile = profile or snapshot.manifest.get("retrieval_profile", PROFILE)
        if profile not in {PROFILE, V2_PROFILE} or profile != snapshot.manifest.get(
            "retrieval_profile", PROFILE
        ):
            raise ValueError("PROFILE_NOT_READY")
        self.snapshot, self.registry = snapshot, registry
        self.profile, self.control, self.audit_dir = profile, control, audit_dir
        self.query_sequence = 0
        self.search_disabled = False
        self.evidence: dict[str, dict[str, Any]] = {}

    def call(self, name: str, params: dict[str, Any], query_limit: int = 6) -> dict[str, Any]:
        return self._call(name, params, query_limit)

    def preview_read(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._call("read_source", params, 0, register_evidence=False)

    def _call(
        self,
        name: str,
        params: dict[str, Any],
        query_limit: int,
        *,
        register_evidence: bool = True,
    ) -> dict[str, Any]:
        started = time.monotonic()
        meta = {
            "retrieval_profile": self.profile,
            "snapshot_id": self.snapshot.manifest["snapshot_id"],
            "backend_query_attempts": 0,
            "cache_hits": 0,
            "external_retrieval_requests": 0,
        }
        try:
            if name == "search_cases" and self.search_disabled:
                raise ValueError("RETRIEVAL_TASK_CLOSED")
            if name == "get_risk_definition" and any(
                tag not in self.snapshot.definitions for tag in params.get("tags", [])
            ):
                raise ValueError("INVALID_RISK_TAG")
            validate_schema(params, tool_schema(name, self.profile))
            if name == "search_cases":
                result = (
                    self._search_v2(params, meta, query_limit, started=started)
                    if self.profile == V2_PROFILE
                    else self._search(params, meta, query_limit)
                )
            elif name == "read_source":
                result = self._read(params, meta, register_evidence=register_evidence)
            elif name == "get_risk_definition":
                result = envelope(
                    "ok",
                    {"definitions": [self.snapshot.definitions[tag] for tag in params["tags"]]},
                    meta,
                )
            else:
                raise ValueError("UNKNOWN_TOOL")
        except (ValueError, KeyError, OSError) as error:
            code = str(error).split(":", 1)[0]
            result = envelope(
                "error", None, meta, {"code": code, "message": str(error)[:300], "retryable": False}
            )
        if "duration_ms" not in result["meta"]:
            result["meta"]["duration_ms"] = round((time.monotonic() - started) * 1000, 2)
        if self.profile == V2_PROFILE:
            result["schema_version"] = "finrisk-tools/1.1"
        return result

    def _search_v2(
        self, params: dict[str, Any], meta: dict[str, Any], query_limit: int, *, started: float
    ) -> dict[str, Any]:
        if len(params["queries"]) != 1:
            raise ValueError("V2_MAX_ONE_QUERY_PER_SEARCH")
        query = params["queries"][0]
        for ref in query["focus_refs"]:
            self.registry.resolve(ref)
        self.query_sequence += 1
        row = {
            **query,
            "query_id": f"Q{self.query_sequence:04}",
            "status": "error",
            "candidates": [],
            "complete": False,
            "returned_candidates_truncated": False,
        }
        if query_limit < 1 and not self.snapshot.backend.is_cached(query["text"], params["top_k"]):
            row["error"] = {"code": "QUERY_BUDGET_EXHAUSTED"}
            return envelope("partial", {"queries": [row]}, meta)
        result, has_context = None, False
        try:
            result = self.snapshot.search(query["text"], params["top_k"], self.control)
            has_context = "context_expansion" in result["meta"] or any(
                "context_expansion" in candidate for candidate in result["candidates"]
            )
            for candidate in result["candidates"]:
                self.snapshot.read(candidate["source_id"])
                diagnosis = candidate.get("context_expansion", {})
                added_ids = (
                    set(diagnosis.get("added_evidence_ids", []))
                    - set(candidate.get("required_evidence_ids", []))
                    if diagnosis.get("policy") == CONTEXT_POLICY else set()
                )
                for evidence in candidate["evidence"]:
                    # 基线锚点仍按旧语义登记；新增父窗口只登记最终返回的部分。
                    if evidence["evidence_id"] not in added_ids:
                        self.evidence[evidence["evidence_id"]] = evidence
            row.update(
                candidates=result["candidates"],
                complete=result["complete"],
                stage_status=result["meta"].get("stage_status", {}),
            )
            if "ranking_coverage" in result:
                row["ranking_coverage"] = result["ranking_coverage"]
            row["status"] = (
                ("ok" if row["candidates"] else "empty") if row["complete"] else "partial"
            )
            if (not row["complete"] and "ranking_coverage" not in row
                    and result["meta"].get("stop_reason")
                    and all(row["stage_status"].get(stage) == "not_started"
                            for stage in ("bm25", "bge", "rrf", "rerank"))):
                row.update(status="error", error={"code": "STOPPED"})
            meta.update(result["meta"])
        except (ValueError, OSError, RuntimeError) as error:
            row.update(complete=False, status="error")
            row["error"] = {"code": str(error).split(":", 1)[0][:100]}
        response = envelope(
            "partial" if not row["complete"] else row["status"], {"queries": [row]}, meta
        )
        if has_context:
            # 新策略把最终包装也计入预算；冻结耗时，避免_call再加字段冲破临界值。
            meta["duration_ms"] = round((time.monotonic() - started) * 1000, 2)
            response["schema_version"] = "finrisk-tools/1.1"
        omitted_ids, omitted_units = set(), set()
        omitted_diagnostics = 0

        def over_budget():
            return len(json.dumps(response, ensure_ascii=False, separators=(",", ":"))) > 12000

        def fit_payload():
            nonlocal omitted_diagnostics
            if has_context:
                for index in range(len(row["candidates"]) - 1, -1, -1):
                    if not over_budget():
                        break
                    candidate = row["candidates"][index]
                    stripped = strip_parent_context(candidate, "tool_payload_budget")
                    removed = {item["evidence_id"] for item in candidate["evidence"]} - {
                        item["evidence_id"] for item in stripped["evidence"]
                    }
                    if removed:
                        omitted_ids.update(removed)
                        omitted_units.add(candidate["unit_id"])
                    row["candidates"][index] = stripped
                if over_budget():
                    # 全部可选字段撤回后必须回到基线大小，不能为诊断淘汰原证据。
                    for candidate in row["candidates"]:
                        omitted_diagnostics += int("context_expansion" in candidate)
                        candidate.pop("context_expansion", None)
                    meta.pop("context_expansion", None)
            while over_budget() and row["candidates"]:
                row["candidates"].pop()
                row["returned_candidates_truncated"] = True
            if not row["candidates"] and row["returned_candidates_truncated"]:
                row.update(
                    status="partial", complete=False,
                    error={"code": "CANDIDATE_EVIDENCE_EXCEEDS_CONTEXT"},
                )
                response["status"] = "partial"
            if over_budget():
                raise ValueError("TOOL_PAYLOAD_TOO_LARGE")

        fit_payload()
        if result is not None and self.audit_dir is not None:
            audit = copy.deepcopy(result.get("audit", {}))
            if has_context:
                audit.setdefault("context_expansion", {}).update(
                    budget_omitted_candidates=len(omitted_units),
                    budget_omitted_evidence_ids=sorted(omitted_ids),
                    budget_omitted_diagnostics=omitted_diagnostics,
                    returned_candidates_truncated=row["returned_candidates_truncated"],
                )
            try:
                self.audit_dir.mkdir(parents=True, exist_ok=True)
                with (self.audit_dir / f"{row['query_id']}.json").open(
                    "x", encoding="utf-8"
                ) as handle:
                    json.dump(audit, handle, ensure_ascii=False, allow_nan=False)
            except (ValueError, OSError, TypeError):
                row.update(complete=False, status="error")
                row["error"] = {"code": "RETRIEVAL_AUDIT_WRITE_FAILED"}
                response["status"] = "partial"
                fit_payload()
        for candidate in row["candidates"]:
            for evidence in candidate["evidence"]:
                self.evidence[evidence["evidence_id"]] = evidence
        return response

    def _search(
        self, params: dict[str, Any], meta: dict[str, Any], query_limit: int
    ) -> dict[str, Any]:
        for query in params["queries"]:
            for ref in query["focus_refs"]:
                self.registry.resolve(ref)
        queries = []
        for idx, query in enumerate(params["queries"]):
            row = {
                "query_id": f"Q{idx + 1:02d}",
                **query,
                "status": "empty",
                "candidates": [],
                "complete": True,
                "returned_candidates_truncated": False,
                "stage_status": {
                    "bm25": "completed",
                    "bge": "not_enabled",
                    "rrf": "not_enabled",
                    "rerank": "not_enabled",
                },
            }
            # 缓存查询不重复消耗后端尝试，但调用/动作由Node登记。
            key = canonical_json(
                {
                    "text": " ".join(query["text"].split()),
                    "top_k": params["top_k"],
                    "snapshot": self.snapshot.manifest["snapshot_id"],
                    "profile": PROFILE,
                }
            )
            cached = key in self.snapshot.cache
            if not cached and meta["backend_query_attempts"] >= query_limit:
                row.update(status="error", complete=False, error={"code": "QUERY_BUDGET_EXHAUSTED"})
                queries.append(row)
                continue
            if not cached:
                meta["backend_query_attempts"] += 1
            try:
                candidates, hit = self.snapshot.search(query["text"], params["top_k"])
                meta["cache_hits"] += int(hit)
                for rank, (evidence, score) in enumerate(candidates, 1):
                    source, text = self.snapshot.read(evidence["source_id"])
                    self.evidence[evidence["evidence_id"]] = evidence
                    row["candidates"].append(
                        {
                            "source_id": source["source_id"],
                            "title": source["title"],
                            "case_id": None,
                            "behavior_id": None,
                            "rank": rank,
                            "score": round(score, 6),
                            "algorithm": PROFILE,
                            "annotation_status": "source_only",
                            "duplicate_hint": _duplicate_hint(text),
                            "evidence": [evidence],
                        }
                    )
                row["status"] = "ok" if candidates else "empty"
            except (ValueError, OSError, RuntimeError) as error:
                row.update(
                    status="error",
                    complete=False,
                    error={"code": str(error).split(":", 1)[0][:120]},
                )
            queries.append(row)
        status = (
            "partial"
            if any(q["status"] == "error" for q in queries)
            else "ok"
            if any(q["candidates"] for q in queries)
            else "empty"
        )
        result = envelope(status, {"queries": queries}, meta)
        # 丢弃整条低排位候选，不截断引文；明确告诉模型投影被缩减。
        while len(json.dumps(result, ensure_ascii=False, separators=(",", ":"))) > 5900:
            rows = [q for q in queries if q["candidates"]]
            if not rows:
                raise ValueError("TOOL_PAYLOAD_TOO_LARGE")
            row = max(rows, key=lambda q: len(q["candidates"]))
            row["candidates"].pop()
            row["returned_candidates_truncated"] = True
        return result

    def _read(
        self,
        params: dict[str, Any],
        meta: dict[str, Any],
        *,
        register_evidence: bool = True,
    ) -> dict[str, Any]:
        source_id = params["source_id"]
        if source_id.startswith("INPUT:"):
            text = self.registry.read_input(source_id)
            source = self.registry.export()
        else:
            source, text = self.snapshot.read(source_id)
        start = params.get("cursor", 0)
        limit = min(params["max_chars"], 4800)
        anchor = None
        if "anchor_evidence_id" in params:
            anchor = self.evidence.get(params["anchor_evidence_id"])
            if not anchor or anchor["source_id"] != source_id:
                raise ValueError("INVALID_EVIDENCE_REFERENCE")
            if anchor["end"] - anchor["start"] > limit:
                # 锚点本身超过上限时先给其连续前缀，由显式标记和cursor续读。
                start = anchor["start"]
            else:
                earliest_start = max(0, anchor["end"] - limit)
                desired_start = max(earliest_start, anchor["start"] - 300)
                boundary = _last_text_boundary(text, earliest_start, desired_start)
                start = boundary if boundary is not None else desired_start
        if start > len(text):
            raise ValueError("INVALID_CURSOR")
        # 给JSON包装留空间；请求的是上限，不保证每次都回满。
        end = min(len(text), start + limit)
        if end < len(text) and (anchor is None or anchor["end"] <= end):
            minimum_end = anchor["end"] if anchor else start + 1
            boundary = _last_text_boundary(text, minimum_end, end)
            if boundary is not None:
                end = boundary
        split = (start > 0 and text[start - 1] not in "\r\n") or (
            end < len(text) and text[end - 1] not in "\r\n" and text[end] not in "\r\n"
        )
        evidence = make_evidence(source, start, end, text)
        if source_id.startswith("INPUT:"):
            evidence["source_kind"] = "input"
            evidence["text_quality"] = "registered_input"
        if register_evidence:
            self.evidence[evidence["evidence_id"]] = evidence
        return envelope(
            "ok",
            {
                "source_id": source_id,
                "evidence": evidence,
                "has_more": end < len(text),
                "next_cursor": end if end < len(text) else None,
                "window_splits_long_paragraph": split,
                "anchor_not_fully_covered": bool(
                    anchor and not (start <= anchor["start"] and anchor["end"] <= end)
                ),
                "case_id": None,
                "behavior_id": None,
            },
            meta,
        )
