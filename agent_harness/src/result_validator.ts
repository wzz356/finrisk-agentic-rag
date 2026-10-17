import { readFileSync } from "node:fs";
import { canonical, hash, root, type JsonObject, type TaskInput, validateOutput } from "./contracts.js";
import type { TaskState } from "./state.js";
import type { Budget } from "./budget.js";
import type { ToolBridge } from "./tool_bridge.js";
import { criteria, criteriaByteHash } from "./criteria.js";
import { isPartialRerank, partialRankingSummary } from "./retrieval_quality.js";

const promptHash = hash(readFileSync(`${root}/configs/agent_rag/system-prompt.txt`, "utf8"));
function contextWarnings(state: TaskState, id: string): JsonObject {
  if (state.protocolVersion !== "finrisk-model-wire/1.2") return {};
  const initial = state.points.find(point => point.risk_point_id === id);
  const use = initial?.observations.material_function;
  const warnings = !use || use.origin === "unknown" || use.value === null || use.value === "unknown" ? ["材料用途未提供或无法确定"] : use.origin === "model_inference" ? ["模型依据文本推断用途"] : [];
  if (!initial) warnings.push("终稿新增点缺少已接受的用途分析");
  return { context_warnings: warnings };
}

export async function assembleReport(task: TaskInput, state: TaskState, budget: Budget, bridge: ToolBridge): Promise<JsonObject> {
  const errors = [...state.validationErrors];
  let final = state.final;
  let schemaValid = Boolean(final);
  if (final) {
    try { validateOutput(final, "final"); state.validateRefs(final); }
    catch (error) { errors.push(String((error as Error).message)); final = undefined; schemaValid = false; }
  }
  const definitionMap = new Map(task.definitions.map(d => [d.risk_tag, d]));
  const assessments = final?.segment_assessments ?? [];
  const assessmentIds = assessments.map((s: JsonObject) => s.segment_id);
  if (new Set(assessmentIds).size !== assessmentIds.length) { errors.push("DUPLICATE_SEGMENT_ASSESSMENT"); final = undefined; schemaValid = false; }
  const sourcePoints = final?.risk_points ?? state.points.map(p => ({ ...p, judgment: "review_required", tag_checks: [], case_matches: [], case_match_status: "not_completed", unknowns: [...p.unknowns, "执行未形成有效终稿。"] }));
  const points: JsonObject[] = [];
  const allSpans = new Map<string, JsonObject>();
  const seenPointIds = new Set<string>();
  let referencesValid = true;
  let semanticConflict = false;
  let caseIncomplete = false;
  let newPointNumber = state.points.length;
  const selectedIds: string[] = sourcePoints.flatMap((p: JsonObject) => p.case_matches.flatMap((c: JsonObject) => c.evidence_ids));
  let verified = new Set<string>();
  if (selectedIds.length) {
    try {
      const result = await bridge.verifyEvidence([...new Set(selectedIds)], Math.min(budget.remainingMs(), budget.config.tool_request_timeout_seconds * 1000));
      verified = new Set(result.valid_ids);
      if (result.rejected_ids.length) { referencesValid = false; errors.push("SOURCE_HASH_OR_EVIDENCE_MISMATCH"); }
    } catch { referencesValid = false; errors.push("SOURCE_VERIFICATION_FAILED"); }
  }
  for (const point of sourcePoints) {
    const spans: JsonObject[] = [];
    for (const ref of point.input_refs) {
      try { const span = state.resolve(ref); spans.push(span); allSpans.set(canonical(span), span); }
      catch { referencesValid = false; errors.push("INVALID_INPUT_REFERENCE"); }
    }
    if (!spans.length) { referencesValid = false; errors.push("POINT_WITHOUT_VALID_INPUT"); continue; }
    const id = point.risk_point_id ?? `RP${String(++newPointNumber).padStart(4, "0")}`;
    if ((point.risk_point_id && !state.points.some(p => p.risk_point_id === id)) || seenPointIds.has(id)) { referencesValid = false; errors.push("INVALID_RISK_POINT_ID"); continue; }
    seenPointIds.add(id);
    const matches: JsonObject[] = [];
    for (const match of point.case_matches) {
      const valid = match.evidence_ids.every((eid: string) => verified.has(eid) && state.evidence.get(eid)?.source_id === match.source_id && state.evidence.get(eid)?.source_kind === "penalty");
      if (!valid) { referencesValid = false; errors.push("INVALID_CASE_OR_EVIDENCE_REFERENCE"); continue; }
      matches.push({ ...match, evidence: match.evidence_ids.map((eid: string) => state.evidence.get(eid)), semantic_review_status: "model_assessed" });
    }
    const relatedSearches = state.searches.filter(query => query.focus_refs.some((ref: JsonObject) => {
      const s = state.resolve(ref); return spans.some(p => s.start < p.end && p.start < s.end);
    }));
    const latestQueries = new Map<string, JsonObject>(); relatedSearches.forEach(q => latestQueries.set(state.queryKey(q), q));
    const pendingSearch = [...latestQueries.values()].some(q => !q.complete || q.status === "error");
    const completedSearch = relatedSearches.some(q => q.complete && ["ok", "empty"].includes(q.status)) && !pendingSearch;
    let caseStatus = point.case_match_status;
    if (["reliable_match", "partial_match"].includes(caseStatus) && !matches.length) caseStatus = completedSearch ? "no_reliable_match" : "not_completed";
    if (caseStatus === "no_reliable_match" && !completedSearch) { caseStatus = "not_completed"; errors.push("CASE_SEARCH_NOT_COMPLETED"); }
    if (pendingSearch && caseStatus !== "not_required") { caseStatus = "not_completed"; errors.push("CASE_QUERY_GOAL_UNFINISHED"); }
    if (caseStatus === "not_required" && (point.judgment !== "no_clear_risk" || point.tag_checks.some((t: JsonObject) => t.status !== "not_supported"))) { caseStatus = "not_completed"; errors.push("CASE_REQUIRED_FOR_CANDIDATE_RISK"); }
    if (caseStatus === "not_completed") caseIncomplete = true;
    const supported: string[] = []; const pending: string[] = [];
    const tagGroups = new Map<string, JsonObject[]>();
    for (const tag of point.tag_checks) tagGroups.set(tag.tag, [...(tagGroups.get(tag.tag) ?? []), tag]);
    const tagChecks = [...tagGroups.values()].map(group => {
      let tag = group[0]!;
      if (group.length > 1) { semanticConflict = true; errors.push(`DUPLICATE_RISK_TAG:${tag.tag}`); tag = { ...tag, status: "unknown", reason: "同一标签出现重复或矛盾判据，保留原记录供复核。", conflicting_tag_checks: group }; }
      const definition = definitionMap.get(tag.tag);
      if (!definition) { referencesValid = false; errors.push("MISSING_DICTIONARY_DEFINITION"); }
      const status = definition ? tag.status : "unknown";
      if (status === "supported") supported.push(tag.tag);
      if (status === "unknown") pending.push(tag.tag);
      const bindings = state.caseTagBindings.get(id) ?? [];
      const tagMatches = matches.filter(m => bindings.some(b => b.source_id === m.source_id && b.supports_tags.includes(tag.tag) && b.evidence_ids.every((eid: string) => m.evidence_ids.includes(eid))));
      const scopedCaseStatus = state.protocolVersion !== "finrisk-model-wire/1.2" ? caseStatus : caseStatus === "not_completed" ? "not_completed" : tagMatches.length ? tagMatches.every(m => m.match_status === "reliable_match") ? "reliable_match" : "partial_match" : completedSearch ? "no_reliable_match" : "not_completed";
      if (scopedCaseStatus === "not_completed" && status !== "not_supported") caseIncomplete = true;
      return { ...tag, status, case_match_status: status === "not_supported" && tag.case_match_status === "not_required" ? "not_required" : scopedCaseStatus, definition_ref: definition ? { risk_tag: tag.tag, description: definition.description, dictionary_sha256: task.snapshot.dictionary_sha256 } : null,
        support_input_spans: tag.support_input_refs.map((r: JsonObject) => state.resolve(r)), contra_input_spans: tag.contra_input_refs.map((r: JsonObject) => state.resolve(r)) };
    });
    let judgment = point.judgment;
    if (judgment === "no_clear_risk" && point.unknowns.length) { judgment = "review_required"; errors.push("UNKNOWNS_PREVENT_NO_CLEAR_RISK"); }
    if (judgment === "suspected" && !supported.length) judgment = "review_required";
    if (judgment === "no_clear_risk" && (supported.length || pending.length)) { judgment = supported.length ? "suspected" : "review_required"; errors.push("JUDGMENT_TAG_CONFLICT"); }
    points.push({ ...point, ...contextWarnings(state, id), risk_point_id: id, judgment, input_spans: spans, supported_tags: supported, pending_tags: pending, tag_checks: tagChecks, case_match_status: caseStatus, case_matches: matches, status: judgment === "review_required" || pending.length ? "review_required" : "resolved" });
  }
  // analysis提出的点不能被终稿静默删除；未处理点作为程序待复核保留。
  for (const initial of state.points) {
    if (points.some(p => p.risk_point_id === initial.risk_point_id || (p.mechanism === initial.mechanism && canonical(p.input_refs) === canonical(initial.input_refs)))) continue;
    const spans = initial.input_refs.map((r: JsonObject) => state.resolve(r)); spans.forEach((s: JsonObject) => allSpans.set(canonical(s), s));
    points.push({ ...initial, ...contextWarnings(state, initial.risk_point_id), input_spans: spans, supported_tags: [], pending_tags: initial.candidate_tags, tag_checks: [], judgment: "review_required", status: "review_required", case_match_status: "not_completed", case_matches: [], unknowns: [...initial.unknowns, "终稿未处理该候选风险点。"] });
    caseIncomplete = true;
  }
  const provided = task.input.segments.map(s => s.segment_id);
  const reviewed = final ? assessments.filter((s: JsonObject) => s.status === "reviewed").map((s: JsonObject) => s.segment_id) : [];
  const unresolved = provided.filter(id => assessments.some((s: JsonObject) => s.segment_id === id && s.status !== "reviewed") || final?.unprocessed_segment_ids.includes(id));
  const notReviewed = provided.filter(id => !reviewed.includes(id) && !unresolved.includes(id));
  const coverage = { provided_segment_ids: provided, reviewed_segment_ids: reviewed, unresolved_segment_ids: unresolved, not_reviewed_segment_ids: notReviewed, all_submitted_segments_reviewed: provided.length === reviewed.length && !unresolved.length && !notReviewed.length, semantic_coverage_verified: false };
  const unresolvedInputGaps = [...state.gaps.values()].filter(g => g.type === "input_condition" && g.status !== "resolved" && (state.protocolVersion !== "finrisk-model-wire/1.2" || !points.some(p => p.risk_point_id === g.risk_point_id && p.tag_checks.length && p.tag_checks.every((t: JsonObject) => t.status === "not_supported"))));
  const allQueryStates = new Map<string, JsonObject>(); state.searches.forEach(q => allQueryStates.set(state.queryKey(q), q));
  const unfinishedQueries = [...allQueryStates.values()].filter(q => !q.complete || q.status === "error");
  const partial = !final || !coverage.all_submitted_segments_reviewed || caseIncomplete || unfinishedQueries.length > 0 || !referencesValid || state.protocolInvalid || semanticConflict;
  const overall = points.some(p => p.supported_tags.length) ? "suspected" : partial || unresolvedInputGaps.length || points.some(p => p.judgment === "review_required") ? "review_required" : "no_clear_risk";
  const status = partial ? "partial" : "completed";
  const limitedRanking = state.retrievalReliability && unfinishedQueries.length > 0 && unfinishedQueries.every(isPartialRerank)
    && referencesValid && !semanticConflict && !state.protocolInvalid;
  state.phase = status;
  return {
    schema_version: "finrisk-report/1.0", task_id: task.input.task_id, execution_mode: budget.config.execution_mode,
    task_status: status, overall_judgment: overall, needs_manual_review: partial || Boolean(unresolvedInputGaps.length) || points.some(p => p.pending_tags.length || p.unknowns.length || p.judgment === "review_required") || budget.config.execution_mode === "mock",
    risk_points: points, input_spans: [...allSpans.values()], coverage,
    summary: !final ? "执行未形成有效终稿，当前待复核是执行兜底，不能作为完整业务风险判断；原文及执行记录已保留。" : limitedRanking ? partialRankingSummary(state) : partial ? (overall === "suspected" ? "已有输入依据提示部分疑似风险；仍有未完成或矛盾事项需要复核。" : "本轮仅完成部分审查，尚不能判断整篇材料是否合规。") : final.summary,
    versions: { snapshot_id: task.snapshot.snapshot_id, dictionary_sha256: task.snapshot.dictionary_sha256, input_sha256: task.input.input_sha256, text_sha256: task.input.text_sha256, contract: state.protocolVersion, requested_contract: state.requestedProtocolVersion, accepted_contract: state.analysis || state.final ? state.protocolVersion : null, prompt: "finrisk-agent/1.5", prompt_sha256: promptHash, criteria: criteria.schema_version, criteria_sha256: criteriaByteHash, criteria_review_status: criteria.review_status, config_sha256: hash(canonical(budget.config)), retrieval_profile: budget.config.retrieval_profile, pi_agent_core: "1.2.0" },
    stop: { reason: state.stopReason, unresolved: points.filter(p => p.status === "review_required").map(p => p.risk_point_id) },
    resources: { ...budget.summary(), json_repairs: state.repairs, tool_format_corrections: state.toolFormatCorrections, estimated_cost: null },
    validation: { schema_valid: schemaValid, references_valid: referencesValid, semantic_review_status: budget.config.execution_mode === "live" && final && schemaValid && referencesValid ? "model_assessed" : "not_assessed", errors },
    model_generated: Boolean(final) && budget.config.execution_mode === "live", simulated: budget.config.execution_mode === "mock",
    retrieved_evidence: [...state.evidence.values()],
  };
}
