import { readFileSync } from "node:fs";
import Ajv2020 from "ajv/dist/2020.js";
import { canonical, hash, root, type Config, type JsonObject, type TaskInput } from "./contracts.js";
import { criteriaFor, criteriaProjection } from "./criteria.js";
import { candidateBundleRead, normalizeModelOutput, wireSchema, wireContract } from "./model_protocol.js";
import { ModelSchemaError, schemaIssuePolicy } from "./model_json.js";
import type { TaskState } from "./state.js";
import { citationProtocolVersion } from "./citation_bundles.js";
import { isPartialRerank, onlyPartialRerankGaps, retrievalExecutionFailed, retrievalReliabilityPolicy } from "./retrieval_quality.js";

export type EvaluationArm = "R1" | "R2" | "A";
export const evaluationInputAlias = "INPUT:EVALUATION";
const gapSchema = { type: "object", additionalProperties: false, required: ["type", "risk_point_id", "reason"], properties: { type: { enum: ["case_comparison", "unread_source", "input_fact"] }, risk_point_id: { type: "string", pattern: "^RP[0-9]{4,}$" }, reason: { type: "string", minLength: 1, maxLength: 500 } } };
export const evaluationProtocolVersion = "finrisk-evaluation/2.4";
export const evaluationSystemPrompt = readFileSync(`${root}/configs/agent_rag/evaluation-system-prompt.txt`, "utf8");
export const evaluationSystemPromptHash = hash(evaluationSystemPrompt);
const pointSchema = { type: "object", additionalProperties: false, required: ["input_refs", "facts", "tag_checks", "case_matches"], properties: {
  input_refs: wireSchema.$defs.candidate.properties.input_refs,
  facts: wireSchema.$defs.candidate.properties.facts,
  tag_checks: { ...wireSchema.$defs.finalPoint.properties.tag_checks, minItems: 1 },
  case_matches: wireSchema.$defs.finalPoint.properties.case_matches,
} };
export const evaluationSchema = { $schema: wireSchema.$schema, type: "object", additionalProperties: false, required: ["kind", "segments", "points", "continuation"], properties: { kind: { const: "evaluation_first" }, segments: wireSchema.$defs.analysis.properties.segments, points: { type: "array", maxItems: 8, items: pointSchema }, continuation: { type: "object", additionalProperties: false, required: ["gaps"], properties: { gaps: { type: "array", maxItems: 8, items: gapSchema } } } }, $defs: Object.fromEntries(["quoteRef", "refs", "contextRefs", "strings", "segmentAssessment", "caseMatch", "tag", "fact", "condition", "tagCheck"].map(name => [name, wireSchema.$defs[name]])) };
const legacyEvaluationSchema = structuredClone(evaluationSchema);
evaluationSchema.$defs.caseMatch = wireContract("final", true).$defs.caseMatch;
const ajv = new Ajv2020.default({ strict: false, allErrors: true });
const validator = ajv.compile<JsonObject>(evaluationSchema);
const legacyValidator = ajv.compile<JsonObject>(legacyEvaluationSchema);
const legacyEvaluationInstruction = "本次一次响应必须同时审阅所有输入段落、提出候选点并完成逐标签条件终判。只提交evaluation_first JSON，顶层只有kind、segments、points、continuation，禁止工具调用。每个point只提交input_refs、facts、tag_checks、case_matches；tag_checks是该点唯一标签清单，必须包含全部候选标签并填全每个标签逐项条件，不能留空清单。程序按points数组顺序注册RP0001等编号，continuation.gaps引用该编号。facts中的insurance_product和contract_outside用yes/no/null，material_function用原Schema枚举，推断不可冒充当前事实。非unknown条件须有当前输入或用户背景引用。案例只引用task_state中已读且可见的原文，并逐项说明相同与不同；V2须选实际unit_id并覆盖全部required_evidence_ids。没有可靠案例可保留空case_matches。continuation.gaps仅记录仍需补足的case_comparison、unread_source或input_fact，充分则空数组；不得仅凭confidence判充分。当前合同、主体、用途、产品等事实未知时不能用历史库补齐，应保留unknown和待复核。覆盖与引用检查仅是工程验证，语义正确性仍待人工核验。";
export const evaluationInstruction = legacyEvaluationInstruction.replace("案例只引用task_state中已读且可见的原文，并逐项说明相同与不同；V2须选实际unit_id并覆盖全部required_evidence_ids。", "案例只从task_state.citation_options选择case_ref；case_matches只含case_ref、same_points、differences、match_status、supports_tags，禁止填写source_id/unit_id/evidence_ids。选项只对应已核读且当前可见、完整覆盖必要锚点的原文；程序展开真实来源、单元及本轮可见引用窗口，模型逐项说明相同机制与差异。") + " points数组中的每个风险点对象和points数组必须完整闭合，再输出与points同级的continuation，禁止把continuation放在风险点内。input_fact只表示影响本次材料判断的必要条件未知，须对应该点至少一个unknown标签条件；不影响当前条件判断的补充调查信息不能列为阻断缺口。";
// Resolve the exact active schema, including compact/legacy refs, instead of
// maintaining another handwritten field contract beside the validator.
function objectFieldInventory(schema: JsonObject): JsonObject[] {
  const result: JsonObject[] = [];
  const visit = (node: JsonObject, path: string, ancestors: Set<JsonObject>) => {
    if (node.$ref) node = schema.$defs[node.$ref.slice("#/$defs/".length)];
    if (!node || ancestors.has(node)) return;
    const next = new Set(ancestors).add(node);
    if (node.type === "object") {
      result.push({ path, fields: Object.keys(node.properties ?? {}), required: node.required ?? [], additionalProperties: node.additionalProperties });
      for (const [name, child] of Object.entries(node.properties ?? {})) visit(child as JsonObject, `${path}.${name}`, next);
    } else if (node.type === "array" && node.items) visit(node.items, `${path}[]`, next);
  };
  visit(schema, "$", new Set());
  return result;
}
const derivedOutputInstruction = "程序派生的报告汇总信息不属于模型可自行增补的输出字段；状态、编号等仅在当前Schema明确列出的位置填写，每层只输出字段清单允许的字段，不添加解释性或工程字段。";
// A source fingerprint is deliberately explicit so the experiment controller can freeze it.
export const evaluationProtocolHash = hash(canonical({ version: evaluationProtocolVersion, retrieval_reliability: retrievalReliabilityPolicy, format_recovery: "retire_exact_admitted_error_after_acceptEvaluation_and_recovery_audit; preserve_rejected_reply_and_other_blockers", schema: evaluationSchema, final_schema: wireContract("final", true), system_prompt: evaluationSystemPrompt, citation_version: citationProtocolVersion, instruction: evaluationInstruction, query: "canonical({text,user_context}); unicode_max=500; no_truncation", prefetch: "one_search_top3; dynamic_recompute_after_actual_window; sourceReadPlan_required_anchor_order; max3_1000_char_windows; skip_already_covered_anchor; attempted_anchor_not_retried", first_payload: { input_source_alias: evaluationInputAlias, seconds_remaining: "configured_deadline_not_prefetch_wallclock", query_remaining: "min(actual_remaining, max(0, configured_query_limit_minus_prefetch_query_count)); display_only" }, object_field_inventory: objectFieldInventory(evaluationSchema), derived_output_instruction: derivedOutputInstruction, schema_feedback: { ...schemaIssuePolicy, source: "typed_Ajv_structure_only", fields: ["instancePath", "keyword", "additionalProperty", "missingProperty"], redaction: "bounded_ascii_identifiers_no_values_or_error_messages" }, gate: "unified_points_tag_checks_only; ordered_RP_ids; complete_criteria_no_synthesis; strict_report_references; unknown_current_facts_stop; actionable_case_gap_only" }));

export function evaluationConfig(arm: EvaluationArm, config: Partial<Config> = {}): Partial<Config> {
  if (!["R1", "R2", "A"].includes(arm)) throw new Error("INVALID_EVALUATION_ARM");
  return { ...config, evaluation_arm: arm, engine: arm === "R2" ? "fixed_rag" : "controlled_agent", eval_mode: arm === "R1" ? "one_shot_rag" : arm === "A" ? "adaptive_rag" : undefined, max_model_requests: 6, max_search_calls: 2, max_tool_actions: 8, task_deadline_seconds: 270 };
}
export function evaluationQuery(task: TaskInput): JsonObject {
  const text = canonical({ text: task.input.text, user_context: task.input.user_context });
  if (Array.from(text).length > 500) throw new Error("EVALUATION_QUERY_TOO_LONG");
  const focus_refs = task.input.segments.filter(s => s.text.trim()).map(s => ({ segment_id: s.segment_id, quote: s.text, occurrence: 1 }));
  if (!focus_refs.length || focus_refs.length > 8) throw new Error("EVALUATION_QUERY_SCOPE_INVALID");
  return { queries: [{ text, focus_refs, risk_point_ids: [], goal: "mechanism_search" }], top_k: 3, filters: {} };
}
export function evaluationProtocol(task: TaskInput): string {
  const compact = task.config?.retrieval_profile === "v2-hybrid-flash-v1";
  const schema = compact ? evaluationSchema : legacyEvaluationSchema;
  return `${compact ? evaluationInstruction : legacyEvaluationInstruction} ${derivedOutputInstruction}\n对象字段清单（由当前Schema派生）=${JSON.stringify(objectFieldInventory(schema))}\n官方字典=${JSON.stringify(task.definitions.map(d => ({ tag: d.risk_tag, definition: d.description })))}\n逐标签条件（内部字典解释，不是新增法规）=${JSON.stringify(criteriaProjection(task.definitions.map(d => d.risk_tag)))}\n当前输出JSON Schema=${JSON.stringify(schema)}`;
}
function completeCoverage(segments: JsonObject[], task: TaskInput): boolean {
  return segments.length === task.input.segments.length && task.input.segments.every(s => segments.some(a => a.segment_id === s.segment_id && a.status === "reviewed"));
}
export function acceptEvaluation(value: JsonObject, state: TaskState, task: TaskInput): JsonObject {
  const validate = state.evaluationCitations ? validator : legacyValidator;
  if (!validate(value)) throw new ModelSchemaError(validate.errors);
  if (!completeCoverage(value.segments, task)) throw new Error("EVALUATION_COVERAGE_INCOMPLETE");
  // Reject incomplete model checklists before the legacy normalizer can retain
  // omitted conditions as unknown. The unified wire has no second tag inventory.
  for (const point of value.points) for (const tag of point.tag_checks) {
    if (criteriaFor(tag.tag).some(c => !tag.conditions.some((v: JsonObject) => v.criterion_id === c.id))) throw new Error("EVALUATION_CRITERIA_INCOMPLETE");
  }
  const analysis = { kind: "analysis", segments: value.segments, points: value.points.map((point: JsonObject) => ({ input_refs: point.input_refs, facts: point.facts, candidate_tags: point.tag_checks.map((tag: JsonObject) => tag.tag) })) };
  const finalWire = { kind: "final", segments: value.segments, points: value.points.map((point: JsonObject, index: number) => ({ risk_point_id: `RP${String(index + 1).padStart(4, "0")}`, tag_checks: point.tag_checks, case_matches: point.case_matches })) };
  // Analysis and final normalization both write business state. Stage their
  // mutable maps so a rejected continuation cannot expose a partially accepted
  // evaluation; evidence and citation registries remain read-only inputs.
  const staged: TaskState = Object.assign(Object.create(Object.getPrototypeOf(state)), state, {
    gaps: new Map(state.gaps),
    caseTagBindings: new Map(state.caseTagBindings),
  });
  staged.acceptAnalysis(normalizeModelOutput(analysis, "analysis", staged, task));
  const final = normalizeModelOutput(finalWire, "final", staged, task);
  for (const gap of value.continuation.gaps) {
    if (!staged.points.some(point => point.risk_point_id === gap.risk_point_id)) throw new Error("INVALID_EVALUATION_GAP_REFERENCE");
    const point = final.risk_points.find((row: JsonObject) => row.risk_point_id === gap.risk_point_id)!;
    if (gap.type === "input_fact" && !point.tag_checks.some((tag: JsonObject) => tag.status === "unknown" && tag.conditions.some((condition: JsonObject) => condition.status === "unknown"))) throw new Error("EVALUATION_INPUT_GAP_WITHOUT_UNKNOWN");
  }
  staged.validateRefs(final);
  Object.assign(state, { phase: staged.phase, analysis: staged.analysis, points: staged.points, final,
    gaps: staged.gaps, caseTagBindings: staged.caseTagBindings, protocolVersion: staged.protocolVersion });
  return value.continuation;
}
export function evaluationRoute(continuation: JsonObject, state: TaskState, report: JsonObject): "sufficient" | "input_facts_missing" | "continue" | "limited_assessment" | "retrieval_failed" | "incomplete" {
  if (!report.validation.schema_valid || !report.validation.references_valid || !report.coverage.all_submitted_segments_reviewed || state.protocolInvalid) return "incomplete";
  if (state.validationErrors.includes("EVALUATION_CRITERIA_INCOMPLETE")) return "incomplete";
  const currentFactsMissing = report.risk_points.some((p: JsonObject) => p.tag_checks.some((t: JsonObject) => t.status === "unknown" && t.conditions.some((c: JsonObject) => c.status === "unknown")));
  if (currentFactsMissing || continuation.gaps.some((g: JsonObject) => g.type === "input_fact")) return "input_facts_missing";
  if (retrievalExecutionFailed(state, report)) return "retrieval_failed";
  const actionable = continuation.gaps.some((g: JsonObject) => {
    const point = report.risk_points.find((p: JsonObject) => p.risk_point_id === g.risk_point_id);
    if (!point || point.judgment === "no_clear_risk") return false;
    const searches = state.searches.filter(q => (q.complete && ["ok", "empty"].includes(q.status) || state.retrievalReliability && onlyPartialRerankGaps(state, report) && isPartialRerank(q)) && q.focus_refs.some((r: JsonObject) => point.input_refs.some((f: JsonObject) => { const a = state.resolve(r), b = state.resolve(f); return a.start < b.end && b.start < a.end; })));
    return searches.length > 0 && (g.type === "case_comparison" && ["no_reliable_match", "partial_match", "not_completed"].includes(point.case_match_status) || g.type === "unread_source" && searches.some(q => q.candidates.some((c: JsonObject) => !candidateBundleRead(c, state, true))));
  });
  if (actionable) return "continue";
  if (!continuation.gaps.length && onlyPartialRerankGaps(state, report)) return "limited_assessment";
  return report.task_status === "completed" && !report.risk_points.some((p: JsonObject) => p.judgment === "review_required") && !continuation.gaps.length ? "sufficient" : "incomplete";
}
