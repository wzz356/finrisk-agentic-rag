import type { Budget } from "./budget.js";
import type { TaskState } from "./state.js";
import type { JsonObject } from "./contracts.js";
import { candidateBundleRead } from "./model_protocol.js";

const labels: Record<string, string> = { material_analysis: "理解材料", search_cases: "搜索处罚案例", read_source: "回查处罚原文", get_risk_definition: "核对官方标签", condition_comparison: "比较适用条件", final_generation: "生成审查结论", complete: "流程完成", incomplete: "部分完成" };
export function modelBudget(budget: Budget): JsonObject {
  const used = budget.summary();
  return { model_remaining_after_current: Math.max(0, budget.config.max_model_requests - Number(used.model_request_units) - 1), tool_remaining: Math.max(0, budget.config.max_tool_actions - Number(used.tool_actions)), search_remaining: Math.max(0, budget.config.max_search_calls - Number(used.search_calls)), query_remaining: budget.remainingQueries(), seconds_remaining: Math.ceil(budget.remainingMs() / 1000) };
}
export function progressSnapshot(state: TaskState, budget: Budget, action: string, sequence: number, report?: JsonObject): JsonObject {
  const used = budget.summary();
  const limit = (key: string, maximum: number) => ({ used: Number(used[key]), limit: maximum });
  const queries = new Map<string, JsonObject[]>();
  state.searches.forEach(query => { const key = state.queryKey(query); queries.set(key, [...(queries.get(key) ?? []), query]); });
  const pendingCases = [...queries.values()].filter(rows => {
    const related = report?.risk_points.filter((point: JsonObject) => rows.some(query => query.risk_point_ids.includes(point.risk_point_id) || query.focus_refs.some((ref: JsonObject) => point.input_refs.some((p: JsonObject) => p.segment_id === ref.segment_id && p.quote === ref.quote && p.occurrence === ref.occurrence))));
    if (related?.length && related.every((point: JsonObject) => point.case_match_status === "not_required")) return false;
    const latest = rows.at(-1)!;
    return !latest.complete || latest.status === "error" || rows.some(query => query.complete && query.candidates?.length && query.candidates.every((candidate: JsonObject) => !candidateBundleRead(candidate, state)));
  }).length;
  // 查询目标与候选点的案例比较分别计数；运行期间尚无终稿比较，保守保留候选。
  const pendingCasePoints = report ? report.risk_points.filter((point: JsonObject) => point.case_match_status === "not_completed").length : state.points.length;
  // 仅统计真实条件缺口；报告unknowns也包含执行失败/漏审等程序兜底。
  const inputConditions = new Set<string>();
  for (const gap of state.gaps.values()) {
    if (gap.type !== "input_condition" || gap.status === "resolved") continue;
    const point = report?.risk_points.find((p: JsonObject) => p.risk_point_id === gap.risk_point_id);
    // 有终稿条件投影时以结构化判据为准；缺稿/漏点仍保留真实输入gap。
    if (point?.tag_checks.length) continue;
    inputConditions.add(`${gap.risk_point_id}:${gap.statement}`);
  }
  for (const point of report?.risk_points ?? []) for (const tag of point.tag_checks) {
    if (tag.status !== "unknown") continue;
    for (const condition of tag.conditions) if (condition.status === "unknown") inputConditions.add(`${point.risk_point_id}:${condition.condition_id}`);
  }
  const pendingInput = inputConditions.size;
  const taskStatus = report?.task_status ?? "running";
  const stageLabel = labels[action] ?? labels.condition_comparison;
  const message = report ? taskStatus === "partial" ? report.validation.schema_valid ? "部分审查完成，仍有未完成事项。" : "执行未形成有效终稿，不能作为完整业务判断。" : pendingInput ? `流程完成，仍有${pendingInput}项条件待复核。` : "审查流程完成。" : `正在${stageLabel}。`;
  return {
    schema_version: "finrisk-progress/1.0", task_id: state.input.task_id, sequence, updated_at: new Date().toISOString(),
    phase: state.phase, action, stage_label: stageLabel, task_status: taskStatus, execution_mode: budget.config.execution_mode,
    stop_reason: report ? state.stopReason : null, message,
    counts: { segments_total: state.input.segments.length, segments_analyzed: state.analysis?.segment_assessments.filter((segment: JsonObject) => segment.status === "reviewed").length ?? 0, segments_finalized: report?.coverage.reviewed_segment_ids.length ?? 0, risk_points: report?.risk_points.length ?? state.points.length, read_sources: new Set([...state.evidence.values()].filter(evidence => evidence.source_kind === "penalty" && evidence.was_read).map(evidence => evidence.source_id)).size, pending_input_conditions: pendingInput, pending_case_queries: pendingCases, pending_case_points: pendingCasePoints },
    limits: { model_requests: limit("model_request_units", budget.config.max_model_requests), tool_actions: limit("tool_actions", budget.config.max_tool_actions), search_calls: limit("search_calls", budget.config.max_search_calls) },
    time: { remaining_seconds: report ? 0 : Math.ceil(budget.remainingMs() / 1000), elapsed_ms: Math.max(0, Number(used.elapsed_ms)) },
  };
}
