import test from "node:test";
import assert from "node:assert/strict";
import { hash, type JsonObject } from "../src/contracts.js";
import { TaskState } from "../src/state.js";

async function quality(): Promise<any> {
  const path = "../src/retrieval_quality.js";
  return import(path);
}

function query(): JsonObject {
  return { query_id: "Q1", text: "投保赠礼", focus_refs: [], risk_point_ids: [], goal: "mechanism_search",
    complete: false, status: "partial", candidates: [{ unit_id: "U1" }],
    stage_status: { bm25: "completed", bge: "completed", rrf: "completed", rerank: "partial" },
    ranking_coverage: { schema_version: "finrisk-ranking-coverage/1.0", expected_units: 100,
      scored_units: 98, failed_units: 2, ranking_complete: false, topk_scope: "scored_subset",
      attempts_exhausted: true, failure_counts: { QUOTE_MISMATCH: 2 }, failed_unit_ids: ["U99", "U100"],
      expected_set_sha256: "a".repeat(64), scored_set_sha256: "b".repeat(64), failed_set_sha256: "c".repeat(64),
      rerank_protocol: "span-ref-v2" } };
}

function state(): TaskState {
  const value = new TaskState({ task_id: "unit", source_id: "INPUT:unit", text: "投保赠礼", text_version: "unit",
    text_sha256: hash("投保赠礼"), input_sha256: "d".repeat(64), user_context: {}, segments: [] });
  (value as any).retrievalReliability = true;
  value.searches = [query()]; value.final = {};
  return value;
}

function report(): JsonObject {
  return { task_id: "unit", task_status: "partial", overall_judgment: "suspected", needs_manual_review: true,
    versions: { snapshot_id: "snapshot", input_sha256: "d".repeat(64), evaluation_protocol_sha256: "e".repeat(64) },
    validation: { schema_valid: true, references_valid: true, errors: ["CASE_QUERY_GOAL_UNFINISHED"] },
    coverage: { all_submitted_segments_reviewed: true },
    risk_points: [{ risk_point_id: "RP0001", judgment: "suspected", supported_tags: ["赠送利益"],
      case_match_status: "not_completed", tag_checks: [{ status: "supported", conditions: [{ status: "met" }] }] }] };
}

test("98/100 coverage remains partial but is an eligible local rerank failure", async () => {
  const q = await quality(); const value = query();
  assert.doesNotThrow(() => q.validateRankingCoverage(value, true));
  assert.equal(q.isPartialRerank(value), true);
  assert.equal(q.onlyPartialRerankGaps(state(), report()), true);
  assert.equal(value.complete, false);
});

test("malformed coverage and forged complete claims are rejected", async () => {
  const q = await quality();
  for (const mutation of [
    (v: JsonObject) => v.ranking_coverage.scored_units++,
    (v: JsonObject) => v.ranking_coverage.ranking_complete = true,
    (v: JsonObject) => v.ranking_coverage.failure_counts.QUOTE_MISMATCH = 1,
    (v: JsonObject) => v.ranking_coverage.failed_unit_ids.push("U1"),
    (v: JsonObject) => v.ranking_coverage.expected_units = false,
  ]) {
    const value = query(); mutation(value);
    assert.throws(() => q.validateRankingCoverage(value, true), /RANKING_COVERAGE/);
  }
  const missing = query(); delete missing.ranking_coverage;
  assert.throws(() => q.validateRankingCoverage(missing, true), /RANKING_COVERAGE/);
  assert.doesNotThrow(() => q.validateRankingCoverage(missing, false));
});

test("audit, budget, protocol and source failures cannot use local rerank fallback", async () => {
  const q = await quality();
  for (const code of ["AUDIT_WRITE_FAILED", "CHECKPOINT_UNAVAILABLE", "HTTP_4XX", "STOPPED", "UNCLASSIFIED_RERANK_FAILURE"]) {
    const value = query(); value.ranking_coverage.failure_counts = { [code]: 2 };
    assert.equal(q.isPartialRerank(value), false);
  }
  const value = state(); value.protocolInvalid = true;
  assert.equal(q.onlyPartialRerankGaps(value, report()), false);
  const invalid = report(); invalid.validation.references_valid = false;
  assert.equal(q.onlyPartialRerankGaps(state(), invalid), false);
});

test("model projection omits volatile audit metadata", async () => {
  const q = await quality(); const first = query(); first.audit_id = "random";
  first.ranking_coverage.audit_id = "also-random";
  const projected = q.projectRankingCoverage(first);
  assert.equal(JSON.stringify(projected).includes("random"), false);
  assert.equal(projected.scored_units, 98);
});

test("completed ranking with explicit projection failure is a tool failure, not malformed coverage", async () => {
  const q = await quality(); const value = query();
  Object.assign(value.ranking_coverage, { scored_units: 100, failed_units: 0, ranking_complete: true,
    topk_scope: "full_pool", attempts_exhausted: false, failure_counts: {}, failed_unit_ids: [] });
  value.stage_status.rerank = "completed"; value.candidates = [];
  value.error = { code: "CANDIDATE_EVIDENCE_EXCEEDS_CONTEXT" };
  assert.doesNotThrow(() => q.validateRankingCoverage(value, true));
  const s = state(); s.searches = [value];
  assert.equal(q.retrievalExecutionFailed(s, report()), true);
  assert.equal(q.onlyPartialRerankGaps(s, report()), false);
  assert.equal(q.retrievalAssessment(s, report(), null).retrieval_status, "error");
  delete value.error;
  assert.throws(() => q.validateRankingCoverage(value, true), /RANKING_COVERAGE/);
});

test("complete query cannot conceal a tool error", async () => {
  const q = await quality(); const value = query();
  Object.assign(value.ranking_coverage, { scored_units: 100, failed_units: 0, ranking_complete: true,
    topk_scope: "full_pool", attempts_exhausted: false, failure_counts: {}, failed_unit_ids: [] });
  value.stage_status.rerank = "completed"; value.complete = true; value.status = "ok";
  value.error = { code: "RETRIEVAL_AUDIT_WRITE_FAILED" };
  assert.throws(() => q.validateRankingCoverage(value, true), /RANKING_COVERAGE/);
});

test("sidecar distinguishes partial assessment, unknown input and hard failure without upgrading report", async () => {
  const q = await quality(); const r = report();
  const limited = q.retrievalAssessment(state(), r, "f".repeat(64));
  assert.equal(limited.points[0].assessment_status, "model_assessed_with_retrieval_limits");
  assert.equal(limited.report_sha256, "f".repeat(64));
  assert.equal(r.task_status, "partial");
  const unknown = report(); unknown.risk_points[0].tag_checks[0] = { status: "unknown", conditions: [{ status: "unknown" }] };
  assert.equal(q.retrievalAssessment(state(), unknown, null).points[0].assessment_status, "needs_input");
  const invalid = report(); invalid.validation.references_valid = false;
  assert.equal(q.retrievalAssessment(state(), invalid, null).points[0].assessment_status, "blocked");
});
