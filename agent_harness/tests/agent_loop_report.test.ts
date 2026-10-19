import test from "node:test";
import assert from "node:assert/strict";
import { fixture, comparison } from "./agent_loop_fixture.js";
import { hash, validateProgram } from "../src/contracts.js";
import { validateAgentAssessment } from "../src/agent_loop_contracts.js";
const statePath = "../src/agent_loop_state.js", reportPath = "../src/agent_loop_report.js";
async function prepared(tags = ["赠送利益"], count = 1) {
  const { LoopRegistry } = await import(statePath), f = fixture(), r = new LoopRegistry(f.task, f.state, f.budget);
  r.registerPlan(f.plan(count, tags)); return { ...f, r, ...(await import(reportPath)) };
}
test("unassessed dirty deferred blocked retain pending tags and report 1.0 schema", async () => {
  for (const status of ["dirty", "deferred", "blocked"]) {
    const f = await prepared(); if (status !== "dirty") f.r.setProcessingState("RP0001", status, "TEST_STOP");
    const report = await f.assembleLoopReport(f.r, f.bridge); validateProgram(report, "final-report");
    assert.equal(report.task_status, "partial"); assert.equal(report.risk_points[0].judgment, "review_required");
    assert.deepEqual(report.risk_points[0].pending_tags, ["赠送利益"]); assert.deepEqual(report.risk_points[0].tag_checks, []);
  }
});
test("accepted result followed by new evidence cannot appear completed or currently assessed", async () => {
  const f = await prepared(), o = f.addCases()[0]!, req = f.r.beginAssessment("RP0001");
  await f.r.acceptPointAssessment(f.assess(req, [[comparison(o.case_ref)]]), req, f.bridge);
  assert.equal((await f.assembleLoopReport(f.r, f.bridge)).task_status, "completed");
  f.r.markPointDirty("RP0001", "NEW_EVIDENCE"); const report = await f.assembleLoopReport(f.r, f.bridge);
  assert.equal(report.task_status, "partial"); assert.deepEqual(report.risk_points[0].supported_tags, []);
  const sidecar = f.agentAssessmentSidecar(f.r, report, hash(JSON.stringify(report)));
  validateAgentAssessment(sidecar); assert.equal(sidecar.points[0].processing_state, "dirty");
  assert.equal(sidecar.points[0].case_state, "not_assessed"); assert.equal(sidecar.points[0].risk_judgment, "review_required");
  assert.equal(sidecar.points[0].last_assessed_revision, req.state_revision); assert.equal(sidecar.points[0].accepted_assessment_id, null);
});
test("same case comparisons keep per tag reasons reliability and full prefixed rationale", async () => {
  const f = await prepared(["赠送利益", "合同外利益"]), o = f.addCases()[0]!, req = f.r.beginAssessment("RP0001");
  const longReason = "长".repeat(300), a = comparison(o.case_ref, "reliable_match", longReason), b = comparison(o.case_ref, "partial_match", "合同归属不同，不能互相背书。");
  a.same_points = Array(16).fill(longReason); b.same_points = Array(16).fill("合同机制比较。");
  await f.r.acceptPointAssessment(f.assess(req, [[a], [b]]), req, f.bridge);
  const report = await f.assembleLoopReport(f.r, f.bridge); validateProgram(report, "final-report");
  assert.equal(report.risk_points[0].case_matches.length, 1); assert.equal(report.risk_points[0].case_matches[0].same_points.length, 32);
  assert.equal(report.risk_points[0].case_matches[0].same_points[0], "[赠送利益] " + longReason);
  const checks = report.risk_points[0].tag_checks; assert.equal(checks[0].case_match_status, "reliable_match"); assert.equal(checks[1].case_match_status, "partial_match");
  const sidecar = f.agentAssessmentSidecar(f.r, report, hash(JSON.stringify(report)));
  validateAgentAssessment(sidecar); assert.equal(sidecar.points[0].tag_assessments[0].case_assessments[0].same_points[0], longReason);
  assert.equal(sidecar.points[0].tag_assessments[1].case_assessments[0].match_status, "partial_match");
});
test("same source distinct units cannot spread reliable status across tag bindings", async () => {
  const f = await prepared(["赠送利益", "合同外利益"]), options = f.addCases(2, true), req = f.r.beginAssessment("RP0001");
  await f.r.acceptPointAssessment(f.assess(req, [[comparison(options[0]!.case_ref)], [comparison(options[1]!.case_ref, "partial_match")]]), req, f.bridge);
  const report = await f.assembleLoopReport(f.r, f.bridge);
  assert.equal(report.risk_points[0].tag_checks[0].case_match_status, "reliable_match");
  assert.equal(report.risk_points[0].tag_checks[1].case_match_status, "partial_match");
  assert.deepEqual(report.risk_points[0].case_matches.map((m: any) => m.unit_id), ["UNIT0", "UNIT1"]);
});
test("report stays strict old external protocol and sidecar binds explicit report byte hash", async () => {
  const f = await prepared(), req = f.r.beginAssessment("RP0001"); await f.r.acceptPointAssessment(f.assess(req), req, f.bridge);
  const report = await f.assembleLoopReport(f.r, f.bridge); assert.equal(report.schema_version, "finrisk-report/1.0"); validateProgram(report, "final-report");
  const byteHash = hash(JSON.stringify(report, null, 2) + "\r\n"), sidecar = f.agentAssessmentSidecar(f.r, report, byteHash);
  validateAgentAssessment(sidecar); assert.equal(sidecar.identity.report_sha256, byteHash);
  assert.equal(sidecar.protocol_version, "finrisk-agent-loop/1.0"); assert.equal(report.versions.contract, "finrisk-model-wire/1.2");
  const bad = structuredClone(sidecar); bad.points[0].tag_assessments[0].unknown_field = true;
  assert.throws(() => validateAgentAssessment(bad), /LOOP_SCHEMA/);
});
test("report assembly rejects revisions changed across source verification await", async () => {
  const f = await prepared(), o = f.addCases()[0]!, req = f.r.beginAssessment("RP0001");
  await f.r.acceptPointAssessment(f.assess(req, [[comparison(o.case_ref)]]), req, f.bridge);
  const beforeFinal = f.state.final, phase = f.state.phase;
  const bridge = { ...f.bridge, async verifyEvidence(ids: string[]) { f.r.markPointDirty("RP0001", "NEW_EVIDENCE"); return { valid_ids: ids, rejected_ids: [] }; } };
  await assert.rejects(f.assembleLoopReport(f.r, bridge), /LOOP_STALE_REPORT/);
  assert.strictEqual(f.state.final, beforeFinal); assert.equal(f.state.phase, phase);
});

test("sidecar refuses a report assembled before the current point became dirty", async () => {
  const f = await prepared(), req = f.r.beginAssessment("RP0001"); await f.r.acceptPointAssessment(f.assess(req), req, f.bridge);
  const report = await f.assembleLoopReport(f.r, f.bridge); f.r.markPointDirty("RP0001", "NEW_EVIDENCE");
  assert.throws(() => f.agentAssessmentSidecar(f.r, report, hash(JSON.stringify(report))), /LOOP_STALE_REPORT/);
});
test("controlled normalized comparison projection is gated and leaves legacy limits intact", async () => {
  const f = await prepared(["赠送利益", "合同外利益"]), o = f.addCases()[0]!, req = f.r.beginAssessment("RP0001");
  await f.r.acceptPointAssessment(f.assess(req, [[comparison(o.case_ref, "reliable_match", "长".repeat(300))], [comparison(o.case_ref)]]), req, f.bridge);
  const { validateOutput, loopNormalizedContractHash } = await import("../src/contracts.js");
  const accepted = f.r.points.get("RP0001").accepted.result;
  const final = { kind: "final", segment_assessments: f.segments, unprocessed_segment_ids: [], risk_points: [accepted], summary: "逐点核查。" };
  assert.throws(() => validateOutput(final, "final"), /INVALID_SCHEMA/);
  assert.throws(() => validateOutput(final, "final", { loopProjection: true }), /LOOP_PROFILE_REQUIRED/);
  assert.doesNotThrow(() => validateOutput(final, "final", { loopProjection: true, config: f.budget.config }));
  assert.match(loopNormalizedContractHash(), /^[0-9a-f]{64}$/);
  const tooMany = structuredClone(final); tooMany.risk_points[0].case_matches[0].same_points = Array(17).fill("短理由");
  assert.throws(() => validateOutput(tooMany, "final"), /INVALID_SCHEMA/);
  assert.doesNotThrow(() => validateOutput(tooMany, "final", { loopProjection: true, config: f.budget.config }));
});

test("same sentence P1 failed query cannot contaminate P2 completed case or retrieval", async () => {
  const { LoopRegistry } = await import(statePath), f = fixture(), r = new LoopRegistry(f.task, f.state, f.budget);
  const plan = f.plan(2); plan.points[1].input_refs = structuredClone(plan.points[0].input_refs); plan.points[1].candidate_tags = ["合同外利益"];
  r.registerPlan(plan); const options = f.addCases(2);
  f.state.searches[0]!.complete = false; f.state.searches[0]!.status = "error"; f.state.searches[0]!.error = { code: "LOCAL_FAILURE" };
  f.state.searches.push({ ...structuredClone(f.state.searches[0]), risk_point_ids: ["RP0002"], text: "仅查合同",
    complete: true, status: "ok", error: undefined });
  for (const id of r.points.keys()) { const req = r.beginAssessment(id), value = f.assess(req, [[comparison(options[id === "RP0001" ? 0 : 1]!.case_ref)]]);
    value.tag_checks.forEach((t: any) => t.conditions.forEach((c: any) => { if (c.input_refs.length) c.input_refs = [f.refs[0]]; }));
    await r.acceptPointAssessment(value, req, f.bridge);
  }
  const { assembleLoopReport, agentAssessmentSidecar } = await import(reportPath), report = await assembleLoopReport(r, f.bridge);
  const p2 = report.risk_points.find((p: any) => p.risk_point_id === "RP0002");
  assert.equal(p2.case_match_status, "reliable_match"); assert.equal(p2.tag_checks[0].case_match_status, "reliable_match");
  assert.deepEqual(r.projectPointState("RP0002").searches.map((q: any) => q.risk_point_ids), [["RP0002"]]);
  const sidecar = agentAssessmentSidecar(r, report, hash(JSON.stringify(report)));
  assert.equal(sidecar.points[1].retrieval_state, "completed"); assert.equal(sidecar.points[0].retrieval_state, "failed");
});

test("registered points with identical refs and candidates are never collapsed when one is unassessed", async () => {
  const { LoopRegistry } = await import(statePath), f = fixture(), r = new LoopRegistry(f.task, f.state, f.budget);
  const plan = f.plan(2); plan.points[1] = structuredClone(plan.points[0]); r.registerPlan(plan);
  const req = r.beginAssessment("RP0001"); await r.acceptPointAssessment(f.assess(req), req, f.bridge);
  const { assembleLoopReport } = await import(reportPath), report = await assembleLoopReport(r, f.bridge);
  assert.deepEqual(report.risk_points.map((p: any) => p.risk_point_id), ["RP0001", "RP0002"]);
  assert.equal(report.risk_points[1].judgment, "review_required"); assert.deepEqual(report.risk_points[1].pending_tags, ["赠送利益"]);
  assert.equal(report.task_status, "partial");
});
test("local partial ranking retains accepted case applicability independently and global stops remain failed", async () => {
  for (const failure of ["QUOTE_MISMATCH", "STOPPED"]) {
    const f = await prepared(), o = f.addCases()[0]!, q = f.state.searches[0]!;
    Object.assign(q, { query_id: "Q1", complete: false, status: "partial",
      stage_status: { bm25: "completed", bge: "completed", rrf: "completed", rerank: "partial" },
      ranking_coverage: { schema_version: "finrisk-ranking-coverage/1.0", expected_units: 2, scored_units: 1, failed_units: 1,
        ranking_complete: false, topk_scope: "scored_subset", attempts_exhausted: true, failure_counts: { [failure]: 1 }, failed_unit_ids: ["FAILED"],
        expected_set_sha256: "a".repeat(64), scored_set_sha256: "b".repeat(64), failed_set_sha256: "c".repeat(64), rerank_protocol: "span-ref-v2" } });
    const { isPartialRerank } = await import("../src/retrieval_quality.js"); assert.equal(isPartialRerank(q), failure === "QUOTE_MISMATCH");
    const req = f.r.beginAssessment("RP0001"); await f.r.acceptPointAssessment(f.assess(req, [[comparison(o.case_ref)]]), req, f.bridge);
    const report = await f.assembleLoopReport(f.r, f.bridge), sidecar = f.agentAssessmentSidecar(f.r, report, hash(JSON.stringify(report)));
    assert.equal(report.task_status, "partial"); assert.equal(sidecar.points[0].case_state, "matched");
    assert.equal(sidecar.points[0].retrieval_state, failure === "QUOTE_MISMATCH" ? "local_partial" : "failed");
    if (failure === "QUOTE_MISMATCH") assert.match(report.summary, /排序未完成/);
  }
});

test("report snapshot also expires when a new global blocker appears", async () => {
  const f = await prepared(), req = f.r.beginAssessment("RP0001"); await f.r.acceptPointAssessment(f.assess(req), req, f.bridge);
  const report = await f.assembleLoopReport(f.r, f.bridge);
  f.r.recordDiagnostic({ scope: f.r.scope("report"), code: "BUDGET_EXHAUSTED", category: "program", blocker_class: "budget" });
  assert.throws(() => f.agentAssessmentSidecar(f.r, report, hash(JSON.stringify(report))), /LOOP_STALE_REPORT/);
});
test("controlled report options cannot be enabled for an unbound legacy profile", async () => {
  const { assembleReport } = await import("../src/result_validator.js"), { Budget } = await import("../src/budget.js"),
    { defaults } = await import("../src/contracts.js"), f = fixture();
  await assert.rejects(assembleReport(f.task, f.state, new Budget(defaults), f.bridge, { loopProjection: true }), /LOOP_PROFILE_REQUIRED/);
});

test("quality: candidate overflow expires completed report and fresh report keeps known tags with explicit partial", async () => {
  const tags = ["赠送利益", "合同外利益", "诱导投保"], f = await prepared(tags), o = f.addCases()[0]!, req = f.r.beginAssessment("RP0001");
  await f.r.acceptPointAssessment(f.assess(req, tags.map(() => [comparison(o.case_ref)])), req, f.bridge);
  const report = await f.assembleLoopReport(f.r, f.bridge); assert.equal(report.task_status, "completed");
  const token = f.r.reportToken(), accepted = f.r.points.get("RP0001").accepted;
  assert.throws(() => f.r.expandCandidates("RP0001", ["虚假宣传"], ["虚假宣传"]), /LOOP_CANDIDATE_LIMIT/);
  assert.notEqual(f.r.reportToken(), token);
  assert.throws(() => f.agentAssessmentSidecar(f.r, report, hash(JSON.stringify(report))), /LOOP_STALE_REPORT/);
  assert.strictEqual(f.r.points.get("RP0001").accepted, accepted); assert.deepEqual(f.r.points.get("RP0001").candidate_tags, tags);
  const fresh = await f.assembleLoopReport(f.r, f.bridge), sidecar = f.agentAssessmentSidecar(f.r, fresh, hash(JSON.stringify(fresh)));
  assert.equal(fresh.task_status, "partial"); assert.match(fresh.risk_points[0].unknowns.join(" "), /额外机制未覆盖/);
  assert.deepEqual(fresh.risk_points[0].tag_checks.map((t: any) => t.tag), tags); assert.equal(sidecar.points[0].additional_mechanisms_uncovered, true);
});

test("excluded tag with another unknown condition does not require sidecar input", async () => {
  const f = await prepared(["合同外利益"]), request = f.r.beginAssessment("RP0001"), value = f.assess(request);
  value.tag_checks[0].conditions.find((c: any) => c.criterion_id === "actual_conduct").status = "not_met";
  await f.r.acceptPointAssessment(value, request, f.bridge);
  const report = await f.assembleLoopReport(f.r, f.bridge), sidecar = f.agentAssessmentSidecar(f.r, report, hash(JSON.stringify(report)));
  assert.equal(report.risk_points[0].judgment, "no_clear_risk"); assert.deepEqual(report.risk_points[0].pending_tags, []);
  assert.equal(sidecar.points[0].input_state, "complete");
});
