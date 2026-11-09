import { type TaskInput, type JsonObject } from "./contracts.js";
import type { ScriptStep } from "./model_gateway.js";
import { criteriaFor } from "./criteria.js";

export function mockPlan(task: TaskInput): ScriptStep[] {
  const segment = task.input.segments.find(s => s.text.trim())!;
  const quote = Array.from(segment.text).slice(0, 500).join("");
  const ref = { segment_id: segment.segment_id, quote, occurrence: 1 };
  const assessments = task.input.segments.map(s => ({ segment_id: s.segment_id, status: "reviewed", reason: "模拟协议已处理；真实语义尚未审查。" }));
  const gift = task.mock_scenario === "gift-example";
  if (gift && task.input.text !== "购买这份保险就送购物卡。") throw new Error("MOCK_GIFT_REQUIRES_EXACT_SYNTHETIC_INPUT");
  const analysis: JsonObject = gift ? { kind: "analysis", segments: assessments, points: [{ input_refs: [ref], candidate_tags: ["赠送利益", "合同外利益"], facts: [{ field: "insurance_product", value: "yes", origin: "input", input_refs: [ref], context_refs: [] }, { field: "material_function", value: "direct_promotion", origin: "inferred", input_refs: [ref], context_refs: [] }, { field: "contract_outside", value: null, origin: "unknown", input_refs: [], context_refs: [] }] }] } : { kind: "analysis", document_context: { main_scene: null, scene_status: "unknown", material_functions: ["unknown"] }, segment_assessments: assessments, risk_points: [], unprocessed_segment_ids: [] };
  const search = { queries: [{ text: Array.from(task.input.text).slice(0, 250).join(""), focus_refs: [ref], risk_point_ids: gift ? ["RP0001"] : [], goal: "mechanism_search" }], top_k: 5, filters: {} };
  const result: JsonObject = gift ? { kind: "final", segments: assessments, points: [{ risk_point_id: "RP0001", tag_checks: ["赠送利益", "合同外利益"].map(tag => ({ tag, conditions: criteriaFor(tag).map(c => ({ criterion_id: c.id, status: c.id === "contract_outside" ? "unknown" : "met", input_refs: c.id === "contract_outside" ? [] : [ref], context_refs: [], reason: c.id === "contract_outside" ? "模拟输入未提供当前合同约定。" : "合成演示fixture，仅验证程序流程。" })) })), case_matches: [] }] } : {
    kind: "final", segment_assessments: assessments, risk_points: [{ risk_point_id: null, input_refs: [ref], mechanism: "本轮仅验证工程协议，材料需真实模型审查。", judgment: "review_required", tag_checks: [], case_match_status: "no_reliable_match", case_matches: [], unknowns: ["尚未执行真实语义审查；模拟检索结果不证明违规或合规。"] }], unprocessed_segment_ids: [], summary: "模拟模式完成工程流程；风险结论待真实模型与人工核查。",
  };
  const steps: ScriptStep[] = [analysis];
  if (task.config?.engine !== "fixed_rag") {
    steps.push((_context, state) => state.phase === "finalizing" ? result : [{ type: "toolCall", id: "search", name: "search_cases", arguments: search }]);
    steps.push((_context, state) => {
      if (state.phase === "finalizing") return result;
      const evidence = [...state.evidence.values()][0];
      return evidence ? [{ type: "toolCall", id: "read", name: "read_source", arguments: { source_id: evidence.source_id, anchor_evidence_id: evidence.evidence_id, max_chars: 1000 } }] : result;
    });
  }
  if (gift && task.config?.engine !== "fixed_rag") steps.push((_context, state) => state.phase === "finalizing" ? result : { kind: "finalize" });
  steps.push(result);
  return steps;
}
