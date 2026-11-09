import { readFileSync } from "node:fs";
import { hash, root, validateProgram, type JsonObject } from "./contracts.js";
import type { TaskState } from "./state.js";

export const rankingCoverageVersion = "finrisk-ranking-coverage/1.0";
const localCodes = new Set(["QUOTE_MISMATCH", "SPAN_REFERENCE_MISMATCH", "INVALID_JSON", "INVALID_COMPLETION", "TRANSPORT_TIMEOUT", "TRANSPORT_ERROR", "HTTP_429", "HTTP_5XX"]);
const incompleteCaseCodes = new Set(["CASE_QUERY_GOAL_UNFINISHED", "CASE_SEARCH_NOT_COMPLETED"]);
const postRankingErrors = new Set(["CANDIDATE_EVIDENCE_EXCEEDS_CONTEXT", "RETRIEVAL_AUDIT_WRITE_FAILED"]);
const fields = ["schema_version", "expected_units", "scored_units", "failed_units", "ranking_complete", "topk_scope", "attempts_exhausted", "failure_counts", "failed_unit_ids", "expected_set_sha256", "scored_set_sha256", "failed_set_sha256", "rerank_protocol"];
export const retrievalReliabilityDescription = { coverage_version: rankingCoverageVersion, schema_version: "finrisk-retrieval-assessment/1.0", pair_attempt_limit: 2, rerank_protocols: ["quote-copy-v1", "span-ref-v2"], configured_in_v2: true };
export const retrievalReliabilityPolicy = { ...retrievalReliabilityDescription, coverage_fields: fields, local_failure_codes: [...localCodes].sort(), limited_error_codes: [...incompleteCaseCodes].sort(), post_ranking_errors: [...postRankingErrors].sort(), schema_sha256: hash(readFileSync(`${root}/contracts/agent_rag/retrieval-assessment.schema.json`, "utf8")), ranking_complete: "all_expected_units_scored_no_zero_fill", report: "strict_report_1.0_plus_private_diagnostic", recovery: "service_internal_same_task_two_attempts_no_new_quota" };

export function validateRankingCoverage(query: JsonObject, required = false): void {
  const c = query.ranking_coverage;
  const reject = () => { throw new Error("INVALID_RANKING_COVERAGE"); };
  if (required && (typeof query.query_id !== "string" || !query.query_id)) return reject();
  if (c === undefined && (!required || query.complete === false && query.status === "error" && query.error?.code)) return;
  const postRankingFailure = query.complete === false && ["partial", "error"].includes(query.status) && postRankingErrors.has(query.error?.code);
  if (query.complete === true && query.error) return reject();
  if (!c || typeof c !== "object" || Array.isArray(c) || fields.some(k => !(k in c)) || Object.keys(c).some(k => !fields.includes(k))) return reject();
  if (c.schema_version !== rankingCoverageVersion || !["quote-copy-v1", "span-ref-v2"].includes(c.rerank_protocol)) return reject();
  if (["expected_units", "scored_units", "failed_units"].some(k => !Number.isSafeInteger(c[k]) || c[k] < 0 || c[k] > 100)) return reject();
  if (c.expected_units !== c.scored_units + c.failed_units || typeof c.ranking_complete !== "boolean" || c.ranking_complete !== (c.failed_units === 0)
    || query.complete !== c.ranking_complete && !(c.ranking_complete && postRankingFailure) || c.topk_scope !== (c.ranking_complete ? "full_pool" : "scored_subset") || typeof c.attempts_exhausted !== "boolean"
    || c.ranking_complete && c.attempts_exhausted) return reject();
  if (!c.failure_counts || typeof c.failure_counts !== "object" || Array.isArray(c.failure_counts)
    || Object.entries(c.failure_counts).some(([code, n]) => !/^[A-Z_0-9]{1,64}$/.test(code) || !Number.isSafeInteger(n) || (n as number) < 1)
    || Object.values(c.failure_counts).reduce((n: number, v) => n + (v as number), 0) !== c.failed_units) return reject();
  if (!Array.isArray(c.failed_unit_ids) || c.failed_unit_ids.length !== Math.min(3, c.failed_units)
    || c.failed_unit_ids.some((id: unknown) => typeof id !== "string" || !id || id.length > 200)
    || new Set(c.failed_unit_ids).size !== c.failed_unit_ids.length
    || ["expected_set_sha256", "scored_set_sha256", "failed_set_sha256"].some(k => typeof c[k] !== "string" || !/^[a-f0-9]{64}$/.test(c[k]))) return reject();
  const ids = (query.candidates ?? []).map((v: JsonObject) => v.unit_id);
  if (ids.some((id: unknown) => typeof id !== "string" || !id || c.failed_unit_ids.includes(id))
    || new Set(ids).size !== ids.length || ids.length > c.scored_units) return reject();
  if (!query.stage_status || ["bm25", "bge", "rrf"].some(k => query.stage_status[k] !== "completed")
    || query.stage_status.rerank !== (c.ranking_complete ? "completed" : "partial")
    || !postRankingFailure && ![c.ranking_complete ? "ok" : "partial", ...(c.ranking_complete ? ["empty"] : [])].includes(query.status)) return reject();
}

export function isPartialRerank(query: JsonObject): boolean {
  try { validateRankingCoverage(query, true); } catch { return false; }
  const c = query.ranking_coverage;
  return Boolean(c && !query.complete && query.status === "partial" && !query.error
    && c.scored_units > 0 && c.failed_units > 0 && c.attempts_exhausted
    && Object.keys(c.failure_counts).every(code => localCodes.has(code)));
}

function latestQueries(state: TaskState): JsonObject[] {
  return [...new Map(state.searches.map(q => [state.queryKey(q), q])).values()];
}

function hardFailure(state: TaskState, report: JsonObject): boolean {
  return state.protocolInvalid || !report.validation.schema_valid || !report.validation.references_valid
    || !report.coverage.all_submitted_segments_reviewed
    || report.validation.errors.some((code: string) => !incompleteCaseCodes.has(code));
}

export function onlyPartialRerankGaps(state: TaskState, report: JsonObject): boolean {
  if (!state.retrievalReliability || hardFailure(state, report)) return false;
  const pending = latestQueries(state).filter(q => !q.complete || q.status === "error");
  return pending.length > 0 && pending.every(isPartialRerank);
}

export function retrievalExecutionFailed(state: TaskState, report: JsonObject): boolean {
  if (!state.retrievalReliability || hardFailure(state, report)) return false;
  return latestQueries(state).some(q => !q.complete && !isPartialRerank(q)
    && (q.error?.code || q.ranking_coverage?.failed_units > 0));
}

export function projectRankingCoverage(query: JsonObject): JsonObject | undefined {
  const c = query.ranking_coverage;
  if (!c) return undefined;
  return Object.fromEntries(["expected_units", "scored_units", "failed_units", "ranking_complete", "topk_scope", "attempts_exhausted", "failure_counts", "failed_unit_ids", "rerank_protocol"].map(k => [k, c[k]]));
}

export function partialRankingSummary(state: TaskState): string {
  const coverages = latestQueries(state).filter(isPartialRerank).map(q => q.ranking_coverage);
  return `本轮候选评分${coverages.reduce((n, c) => n + c.scored_units, 0)}/${coverages.reduce((n, c) => n + c.expected_units, 0)}（按查询计）；现有案例来自已评分子集，排序未完成，审查结果需复核。`;
}

export function retrievalAssessment(state: TaskState, report: JsonObject, reportHash: string | null): JsonObject {
  const queries = latestQueries(state);
  const limited = onlyPartialRerankGaps(state, report);
  const value = {
    schema_version: "finrisk-retrieval-assessment/1.0", task_id: report.task_id,
    input_sha256: report.versions.input_sha256, snapshot_id: report.versions.snapshot_id,
    evaluation_protocol_sha256: report.versions.evaluation_protocol_sha256 ?? null,
    report_sha256: reportHash, retrieval_status: !queries.length ? "not_run" : retrievalExecutionFailed(state, report) || queries.some(q => q.status === "error") ? "error" : queries.every(q => q.complete) ? "completed" : "partial",
    queries: queries.map(q => ({ query_id: q.query_id, complete: q.complete, status: q.status, error_code: q.error ? /^[A-Z_0-9]{1,64}$/.test(q.error.code) ? q.error.code : "TOOL_EXECUTION_FAILED" : null, ranking_coverage: q.ranking_coverage ?? null })),
    points: report.risk_points.map((p: JsonObject) => ({
      risk_point_id: p.risk_point_id,
      assessment_status: hardFailure(state, report) ? state.final ? "blocked" : "not_assessed"
        : p.tag_checks.some((t: JsonObject) => t.status === "unknown" && t.conditions.some((c: JsonObject) => c.status === "unknown")) ? "needs_input"
        : retrievalExecutionFailed(state, report) ? "blocked"
        : limited ? "model_assessed_with_retrieval_limits" : report.task_status === "completed" ? "model_assessed" : "not_assessed",
      case_match_status: p.case_match_status, needs_manual_review: report.needs_manual_review,
    })),
    human_semantic_review_status: "awaiting_human_review",
  };
  validateProgram(value, "retrieval-assessment");
  return value;
}
