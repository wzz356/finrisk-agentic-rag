import { readFileSync } from "node:fs";
import { canonical, defaults, hash, root, type JsonObject, type TaskInput } from "../src/contracts.js";
import { budgetProfileSha256, loadBudgetProfile } from "../src/budget_profile.js";
import { Budget } from "../src/budget.js";
import { TaskState } from "../src/state.js";
import { criteriaFor } from "../src/criteria.js";
import { buildCitationOptions } from "../src/citation_bundles.js";
export function fixture() {
  const text = "购买这份保险就送购物卡。\r\n😀购买第二份保险也送购物卡。";
  const definitions = JSON.parse(readFileSync(root + "/contracts/agent_rag/risk-definitions.json", "utf8"));
  const user_context = { purpose_hint: "当前保险营销宣传" };
  const profile = loadBudgetProfile("planned-evidence-loop-v1");
  const config: any = { ...defaults, ...profile, profile_id: "loop-test", retrieval_profile: "v2-hybrid-flash-v1",
    max_external_retrieval_requests: 800, budget_profile_id: profile.profile_id,
    budget_profile_sha256: budgetProfileSha256(profile), orchestration_mode: "planned_adaptive" };
  const task: TaskInput = { input: { task_id: "loop", source_id: "INPUT:loop", text, text_sha256: hash(text), text_version: "1",
    input_sha256: hash(canonical({ text, user_context })), user_context,
    segments: [{ segment_id: "S0001", source_id: "INPUT:loop", start: 0, end: Array.from(text).length, text }] },
    snapshot: { snapshot_id: "SNAPSHOT", dictionary_sha256: definitions[0].dictionary_sha256 }, definitions, config };
  const state = new TaskState(task.input), budget = new Budget(config);
  state.evaluationCitations = true; state.citationSnapshotId = "SNAPSHOT";
  const refs = ["购买这份保险就送购物卡。", "😀购买第二份保险也送购物卡。"].map(quote => ({ segment_id: "S0001", quote, occurrence: 1 }));
  const segments = [{ segment_id: "S0001", status: "reviewed", reason: "逐项核查当前赠礼宣传。" }];
  function plan(count = 1, tags = ["赠送利益"]): JsonObject {
    return { kind: "plan", segments, points: refs.slice(0, count).map(ref => ({ input_refs: [ref], facts: [
      { field: "insurance_product", value: "yes", origin: "input", input_refs: [ref], context_refs: [] },
      { field: "material_function", value: "direct_promotion", origin: "user_context", input_refs: [], context_refs: ["$.user_context.purpose_hint"] }
    ], candidate_tags: tags, search_intent: "比较保险投保赠礼机制。", query_text: "保险 投保 购物卡" })) };
  }
  function assess(request: any, cases: JsonObject[][] = []): JsonObject {
    const ref = refs[Number(request.risk_point_id.slice(2)) - 1]!;
    return { kind: "assess", assessment_id: request.assessment_id, state_revision: request.state_revision,
      risk_point_id: request.risk_point_id, tag_checks: request.candidate_tags.map((tag: string, i: number) => ({
        tag, conditions: criteriaFor(tag).map(c => ({ criterion_id: c.id, status: c.id === "contract_outside" ? "unknown" : "met",
          input_refs: c.id === "contract_outside" ? [] : [ref], context_refs: [], reason: "当前输入逐项支持；合同归属保留未知。" })),
        case_assessments: cases[i] ?? [] })) };
  }
  function addCases(count = 1, sameSource = false) {
    const quotes = Array.from({ length: count }, (_, i) => "处罚原文" + i + "：购买保险送购物卡。\r\n😀限定条件\n");
    const combined = quotes.join("");
    const candidates = quotes.map((quote, i) => {
      const source_id = sameSource ? "SHARED" : "SRC" + i, start = sameSource ? Array.from(quotes.slice(0, i).join("")).length : 0;
      const sourceText = sameSource ? combined : quote, end = start + Array.from(quote).length;
      const e = { evidence_id: "E" + i, source_id, source_kind: "penalty", text_version: "1", text_sha256: hash(sourceText),
        start, end, quote, source_verified: true };
      const w = { ...e, evidence_id: "W" + i, was_read: true };
      state.evidence.set(e.evidence_id, e); state.evidence.set(w.evidence_id, w); state.contextEvidenceIds.add(w.evidence_id);
      return { source_id, unit_id: "UNIT" + i, required_evidence_ids: [e.evidence_id], evidence: [e] };
    });
    state.searches.push({ focus_refs: [refs[0]], risk_point_ids: ["RP0001"], goal: "mechanism_search", text: "保险赠礼",
      status: "ok", complete: true, candidates });
    for (const o of buildCitationOptions("SNAPSHOT", candidates, [...state.evidence.values()])) state.citationOptions.set(o.case_ref, o);
    return [...state.citationOptions.values()];
  }
  const bridge: any = { async call() { throw new Error("NO_API"); }, async verifyEvidence(ids: string[]) { return { valid_ids: ids, rejected_ids: [] }; } };
  return { task, state, budget, refs, segments, plan, assess, addCases, bridge };
}
export function comparison(case_ref: string, status = "reliable_match", reason = "投保赠礼机制相同。"): JsonObject {
  return { case_ref, same_points: [reason], differences: ["主体与具体情节不同。"], match_status: status };
}
export function business(state: TaskState) {
  return ["phase", "analysis", "points", "final", "gaps", "caseTagBindings", "protocolVersion", "evidence", "citationOptions", "contextEvidenceIds"]
    .map(k => ({ k, identity: (state as any)[k], value: structuredClone((state as any)[k]) }));
}
