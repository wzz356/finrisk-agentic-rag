import { type JsonObject, validateOutput, validateProgram } from "./contracts.js";
import { AGENT_LOOP_VERSION, validateAgentAssessment } from "./agent_loop_contracts.js";
import { assembleReport } from "./result_validator.js";
import { isPartialRerank, partialRankingSummary } from "./retrieval_quality.js";
import { stageLoopState, type LoopRegistry, type LoopPoint } from "./agent_loop_state.js";
import type { ToolBridge } from "./tool_bridge.js";

const assembledReports = new WeakMap<object, { registry: LoopRegistry; token: string }>();
function matchesIdentity(a: JsonObject, b: JsonObject): boolean {
  return a.source_id === b.source_id && a.unit_id === b.unit_id
    && a.evidence_ids.length === b.evidence_ids.length && a.evidence_ids.every((id: string) => b.evidence_ids.includes(id));
}
/** Assemble only currently accepted revisions. Dirty drafts never enter the public report. */
export async function assembleLoopReport(registry: LoopRegistry, bridge: ToolBridge): Promise<JsonObject> {
  const token = registry.reportToken(), staged = stageLoopState(registry.state);
  const accepted = [...registry.points.values()].flatMap(p => { const a = registry.currentAccepted(p); return a ? [{ point: p, accepted: a }] : []; });
  staged.retrievalReliability = true;
  staged.searches = [...new Set([...registry.points.keys()].flatMap(id => registry.queriesForPoint(id)))];
  staged.caseTagBindings = new Map(accepted.map(({ point, accepted: a }) => [point.risk_point_id, structuredClone(a.bindings)]));
  const segments = registry.segments.length ? registry.segments : registry.task.input.segments.map(s => ({ segment_id: s.segment_id, status: "not_reviewed", reason: "执行在接受计划或判定前停止。" }));
  const final = { kind: "final", segment_assessments: structuredClone(segments),
    unprocessed_segment_ids: segments.filter(s => s.status !== "reviewed").map(s => s.segment_id),
    risk_points: accepted.map(({ accepted: a }) => structuredClone(a.result)),
    summary: accepted.some(({ accepted: a }) => a.result.judgment === "suspected")
      ? "按当前材料与官方条件提示疑似风险，逐标签保留案例比较与未知事实。"
      : accepted.some(({ accepted: a }) => a.result.judgment === "review_required")
        ? "必要条件仍有未知，按当前材料列出待复核事项。" : "按当前材料逐点完成审阅；未发现明确风险，结论受材料范围限制。" };
  validateOutput(final, "final", { loopProjection: true, config: registry.budget.config }); staged.validateRefs(final); staged.final = final;
  const report = await assembleReport(registry.task, staged, registry.budget, bridge, { loopProjection: true, pointQueries: id => registry.queriesForPoint(id) });
  registry.assertReportToken(token);
  // Legacy report matching does not separate units or retain nested tag reliability.
  // Restore it from the accepted tag comparisons, matching all real citation identities.
  for (const p of report.risk_points) {
    const point = registry.points.get(p.risk_point_id), a = point && registry.currentAccepted(point);
    if (!a) continue;
    for (const check of p.tag_checks) {
      if (check.status === "not_supported" && check.case_match_status === "not_required") continue;
      if (p.case_match_status === "not_completed") { check.case_match_status = "not_completed"; continue; }
      const record = a.tag_assessments.find(t => t.tag === check.tag);
      const comparisons = record?.case_assessments.filter((c: JsonObject) => p.case_matches.some((m: JsonObject) => matchesIdentity(c, m))) ?? [];
      check.case_match_status = comparisons.length ? comparisons.every((c: JsonObject) => c.match_status === "reliable_match") ? "reliable_match" : "partial_match"
        : check.case_match_status;
    }
    if (point.additional_mechanisms_uncovered) {
      p.unknowns.push("候选标签容量已满，额外机制未覆盖。"); p.status = "review_required"; report.task_status = "partial"; report.needs_manual_review = true;
    }
  }
  if (registry.active_blockers.size) {
    report.task_status = "partial"; report.needs_manual_review = true;
    report.validation.errors = [...new Set([...report.validation.errors, ...[...registry.active_blockers.values()].map(d => d.code)])];
  }
  if (report.task_status === "partial") {
    if (!report.risk_points.some((p: JsonObject) => p.supported_tags.length)) report.overall_judgment = "review_required";
    report.summary = (staged.searches.some(isPartialRerank) ? partialRankingSummary(staged) : "") + "本轮仍有未完成审查或执行阻断；已接受的逐点判断与待核事项分别保留。";
  }
  validateProgram(report, "final-report"); registry.assertReportToken(token);
  registry.state.final = final; registry.state.phase = report.task_status; assembledReports.set(report, { registry, token });
  return report;
}
function relatedQueries(registry: LoopRegistry, point: LoopPoint): JsonObject[] {
  const queries = registry.queriesForPoint(point.risk_point_id);
  return [...new Map(queries.map(q => [registry.state.queryKey(q), q])).values()];
}
/** reportSha256 is the SHA-256 of the exact final report bytes written by the caller. */
export function agentAssessmentSidecar(registry: LoopRegistry, report: JsonObject, reportSha256: string): JsonObject {
  registry.assertIdentity();
  const assembled = assembledReports.get(report);
  if (!assembled || assembled.registry !== registry) throw new Error("LOOP_REPORT_IDENTITY_MISMATCH");
  registry.assertReportToken(assembled.token);
  if (!/^[0-9a-f]{64}$/.test(reportSha256) || report.task_id !== registry.identity.task_id
    || report.versions.input_sha256 !== registry.identity.input_sha256 || report.versions.snapshot_id !== registry.task.snapshot.snapshot_id) throw new Error("LOOP_REPORT_IDENTITY_MISMATCH");
  const points = [...registry.points.values()].map(p => {
    const accepted = registry.currentAccepted(p), r = report.risk_points.find((row: JsonObject) => row.risk_point_id === p.risk_point_id);
    const queries = relatedQueries(registry, p), local = queries.some(isPartialRerank);
    const requiredChecks = accepted?.result.tag_checks.filter((t: JsonObject) => t.status !== "not_supported") ?? [];
    const comparisons = requiredChecks.map((t: JsonObject) => accepted!.tag_assessments.find(a => a.tag === t.tag)?.case_assessments ?? []);
    const caseState = !accepted ? "not_assessed" : comparisons.some((cs: JsonObject[]) => !cs.length) ? "insufficient"
      : comparisons.flat().some((c: JsonObject) => c.match_status !== "reliable_match") ? "partial_match" : "matched";
    return {
      risk_point_id: p.risk_point_id, plan_revision: p.plan_revision, state_revision: p.state_revision, candidate_version: p.candidate_version,
      assessment_revision: p.assessment_revision, input_refs: structuredClone(p.input_refs), facts: structuredClone(p.facts),
      candidate_tags: [...p.candidate_tags], input_state: accepted ? accepted.result.tag_checks.some((t: JsonObject) => t.status === "unknown" && t.conditions.some((c: JsonObject) => c.status === "unknown")) ? "needs_input" : "complete"
        : registry.state.points.find(row => row.risk_point_id === p.risk_point_id)?.unknowns.length ? "needs_input" : "complete",
      risk_judgment: accepted ? r?.judgment ?? "review_required" : "review_required",
      case_state: caseState,
      retrieval_state: !queries.length ? "not_run" : queries.some(q => !q.complete && !isPartialRerank(q)) ? "failed" : local ? "local_partial" : "completed",
      processing_state: p.processing_state, initial_query_covered: p.initial_query_covered, last_assessed_revision: (accepted ?? p.draft)?.state_revision ?? null,
      accepted_assessment_id: accepted?.assessment_id ?? null, draft_assessment_id: p.draft?.assessment_id ?? null,
      pending_case_goals: [...p.pending_case_goals], read_anchor_ids: [...p.read_anchor_ids], stop_reason: p.stop_reason,
      additional_mechanisms_uncovered: p.additional_mechanisms_uncovered, tag_assessments: accepted ? structuredClone(accepted.tag_assessments) : [],
    };
  });
  const value = { schema_version: "finrisk-agent-assessment/1.0", protocol_version: AGENT_LOOP_VERSION,
    orchestration_mode: registry.budget.config.orchestration_mode, identity: { ...registry.identity, report_sha256: reportSha256 },
    plan_revision: registry.plan_revision, points, attempt_diagnostics: structuredClone(registry.attempt_diagnostics),
    active_blockers: structuredClone([...registry.active_blockers.values()]), actions: structuredClone(registry.actions),
    blocker_retirements: structuredClone(registry.blocker_retirements), local_ranking_limited: points.some(p => p.retrieval_state === "local_partial"),
    semantic_review_required: true };
  validateAgentAssessment(value); return value;
}
