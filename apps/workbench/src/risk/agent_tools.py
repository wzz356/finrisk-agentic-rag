"""Request-scoped read-only tool dispatcher with explicit shared query budgets."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from application.evidence_service import KnowledgeEvidenceService
from application.retrieval_service import CaseRetrievalService


@dataclass
class QueryToolBudget:
    remaining_searches: int = 4
    remaining_reads: int = 8
    events: list[dict[str, Any]] = field(default_factory=list)


class RiskEvidenceTools:
    names = ["search_cases", "search_regulations", "read_evidence", "check_conditions"]

    def __init__(
        self,
        retrieval: CaseRetrievalService,
        evidence: KnowledgeEvidenceService,
        source_items: list[dict[str, Any]],
        budget: QueryToolBudget,
    ) -> None:
        self.retrieval = retrieval
        self.evidence = evidence
        self.source_items = {item["sentence_id"]: item for item in source_items}
        self.budget = budget
        self._seen_searches: set[tuple[str, str]] = set()

    async def dispatch(self, name: str, arguments: dict[str, object]) -> object:
        if name not in self.names:
            raise ValueError("未注册工具")
        if name.startswith("search_"):
            query = str(arguments.get("query", "")).strip()
            if not query or len(query) > 2000:
                raise ValueError("查询长度不合法")
            signature = (name, query)
            if signature in self._seen_searches:
                return {
                    "status": "no_new_evidence",
                    "reason": "已执行相同补查，请提交已有判断或待补事实",
                }
            if self.budget.remaining_searches <= 0:
                return {"status": "budget_exhausted", "next_action": "submit_pending_facts"}
            self._seen_searches.add(signature)
            self.budget.remaining_searches -= 1
            top_k = min(10, max(1, int(str(arguments.get("top_k", 3)))))
            if name == "search_cases":
                if (
                    arguments.get("sentence_id")
                    and arguments["sentence_id"] not in self.source_items
                ):
                    raise ValueError("非本批目标")
                matches = await asyncio.to_thread(
                    self.retrieval.search, query, top_k=top_k, risk_tags=None
                )
                self.budget.events.append(
                    {"tool": name, "returned_ids": [m.matched_unit_id for m in matches]}
                )
                return {
                    "matches": [m.model_dump(mode="json") for m in matches],
                    "historical_facts_only": True,
                    "remaining_searches": self.budget.remaining_searches,
                }
            result = await asyncio.to_thread(self.evidence.search_regulations, query, top_k)
            self.budget.events.append(
                {"tool": name, "article_ids": [r["id"] for r in result["articles"]]}
            )
            return result
        if name == "read_evidence":
            if self.budget.remaining_reads <= 0:
                return {"status": "budget_exhausted"}
            self.budget.remaining_reads -= 1
            result = await asyncio.to_thread(
                self.evidence.read, str(arguments.get("id", "")), str(arguments.get("layer", "hit"))
            )
            self.budget.events.append({"tool": name, "id": result["id"], "layer": result["layer"]})
            return result
        assessments = json.loads(str(arguments.get("assessments_json", "[]")))
        if not isinstance(assessments, list):
            raise ValueError("条件核查必须为列表")
        states = []
        for item in assessments:
            if not isinstance(item, dict):
                raise ValueError("条件核查条目必须为对象")
            sid = item.get("sentence_id")
            if sid not in self.source_items or item.get("status") not in {
                "met",
                "not_met",
                "unknown",
            }:
                raise ValueError("条件或目标不合法")
            proof = item.get("evidence", [])
            current = self.source_items[sid]["context"]["source_sentences"]
            if not isinstance(proof, list) or any(
                not isinstance(quote, str) or not quote.strip() for quote in proof
            ):
                raise ValueError("条件证据必须为非空引文数组")
            if item["status"] != "unknown" and (
                not proof or any(not any(quote in s["text"] for s in current) for quote in proof)
            ):
                raise ValueError("已满足或不满足的条件必须引用当前输入原文")
            states.append(item["status"])
        state = (
            "not_supported"
            if "not_met" in states
            else "unknown"
            if not states or "unknown" in states
            else "supported_under_supplied_assessments"
        )
        return {
            "state": state,
            "semantic_truth_checked": False,
            "scope": "only_supplied_selected_conditions",
            "excluded_fact_sources": ["historical_case", "regulatory_text", "similarity_score"],
        }
