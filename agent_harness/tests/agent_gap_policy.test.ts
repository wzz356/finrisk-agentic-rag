import test from "node:test";
import assert from "node:assert/strict";
import { LoopRegistry } from "../src/agent_loop_state.js";
import { fixture, comparison } from "./agent_loop_fixture.js";
import { hash, type JsonObject } from "../src/contracts.js";

const modulePath = "../src/agent_gap_policy.js";
const api = () => import(modulePath);
const allowance = { reassessment_reserved: true };
function setup(count = 1, tags = ["赠送利益"]) {
  const f = fixture(), r = new LoopRegistry(f.task, f.state, f.budget);
  r.registerPlan(f.plan(count, tags)); return { f, r };
}
async function accept(r: LoopRegistry, f: ReturnType<typeof fixture>, id: string, cases: JsonObject[][] = [], edit?: (v: JsonObject) => void) {
  const request = r.beginAssessment(id), value = f.assess(request, cases); edit?.(value);
  return r.acceptPointAssessment(value, request, f.bridge);
}
function partial(query: JsonObject) {
  Object.assign(query, { query_id: "Q1", complete: false, status: "partial",
    stage_status: { bm25: "completed", bge: "completed", rrf: "completed", rerank: "partial" },
    ranking_coverage: { schema_version: "finrisk-ranking-coverage/1.0", expected_units: 100,
      scored_units: 99, failed_units: 1, ranking_complete: false, topk_scope: "scored_subset", attempts_exhausted: true,
      failure_counts: { QUOTE_MISMATCH: 1 }, failed_unit_ids: ["U100"], expected_set_sha256: "a".repeat(64),
      scored_set_sha256: "b".repeat(64), failed_set_sha256: "c".repeat(64), rerank_protocol: "span-ref-v2" } });
}
function searches(decision: any) { return decision.actions.filter((a: any) => a.tool === "search_cases"); }

test("G1 unknown contract fact does not block another tag or independent point", async () => {
  const p = await api(), { f, r } = setup(2, ["赠送利益", "合同外利益"]), option = f.addCases()[0]!;
  await accept(r, f, "RP0001", [[comparison(option.case_ref)], []]);
  const gaps = p.derivePointGaps(r), decision = p.chooseAllowedActions(r, allowance);
  assert.ok(gaps.some((g: any) => g.point_id === "RP0001" && g.tag === "合同外利益" && g.type === "input_fact"));
  assert.ok(searches(decision).some((a: any) => a.point_id === "RP0002"));
  assert.equal(searches(decision).some((a: any) => a.point_id === "RP0001"), false);
  assert.equal(r.currentAccepted(r.points.get("RP0001")!)!.result.judgment, "suspected");
  assert.deepEqual(decision.unfinished_point_ids, ["RP0001", "RP0002"]);
});
test("G2 necessary unread source has priority over an already compared independent point", async () => {
  const p = await api(), { f, r } = setup(2), options = f.addCases(2), query = f.state.searches[0]!;
  query.candidates = [query.candidates[0]];
  f.state.searches.push({ ...query, text: "第二点案例", risk_point_ids: ["RP0002"], focus_refs: [f.refs[1]], candidates: [{
    source_id: "SRC1", unit_id: "UNIT1", required_evidence_ids: ["E1"], evidence: [f.state.evidence.get("E1")] }] });
  f.state.evidence.delete("W0"); f.state.contextEvidenceIds.delete("W0");
  await accept(r, f, "RP0002", [[comparison(options[1]!.case_ref)]]);
  const decision = p.chooseAllowedActions(r, allowance);
  assert.ok(decision.actions.length > 0);
  assert.ok(decision.actions.every((a: any) => a.point_id === "RP0001" && a.tool === "read_source"));
  assert.deepEqual(decision.actions[0].targets[0].required_evidence_ids, ["E0"]);
});
test("G3 current contract unknown alone stops historical retrieval for manual input", async () => {
  const p = await api(), { f, r } = setup(1, ["合同外利益"]); await accept(r, f, "RP0001");
  const decision = p.chooseAllowedActions(r, allowance);
  assert.deepEqual(decision.actions, []); assert.equal(decision.stop_reason, "input_facts_missing");
  assert.ok(decision.gaps.every((g: any) => g.type === "input_fact"));
});
test("G4 lawful 99/100 ranking retains local_partial without repeating its query", async () => {
  const p = await api(), { f, r } = setup(), option = f.addCases()[0]!; partial(f.state.searches[0]!);
  await accept(r, f, "RP0001", [[comparison(option.case_ref)]]);
  const decision = p.chooseAllowedActions(r, allowance);
  assert.equal(decision.local_ranking_limited, true); assert.deepEqual(searches(decision), []);
  assert.equal(decision.stop_reason, "retrieval_partial_exhausted");
  assert.ok(decision.gaps.some((g: any) => g.type === "ranking_limited" && g.local_partial === true));
  assert.equal(decision.gaps.some((g: any) => g.type === "case_comparison"), false);
});
test("G5 global STOPPED prevents new search and preserves unfinished points", async () => {
  const p = await api(), { f, r } = setup(2); f.addCases();
  Object.assign(f.state.searches[0]!, { complete: false, status: "error", error: { code: "STOPPED" } });
  const before = [...r.points.values()], decision = p.chooseAllowedActions(r, allowance);
  assert.deepEqual(decision.actions, []); assert.equal(decision.stop_reason, "execution_blocked");
  assert.deepEqual(decision.unfinished_point_ids, ["RP0001", "RP0002"]);
  assert.ok(decision.gaps.some((g: any) => g.type === "execution_blocked"));
  assert.equal(decision.local_ranking_limited, false); assert.deepEqual([...r.points.values()], before);
});
test("G6 two rounds without evidence or semantic assessment progress stop as no_progress", async () => {
  const p = await api(), { f, r } = setup(), option = f.addCases()[0]!;
  r.observeEvidence(["W0"]); await accept(r, f, "RP0001", [[comparison(option.case_ref)]]);
  const baseline = p.progressFingerprint(r);
  await accept(r, f, "RP0001", [[comparison(option.case_ref, "reliable_match", "仅改写同一理由。")]], v => {
    for (const c of v.tag_checks[0].conditions) c.reason = "无语义变化的重新措辞。";
  });
  f.state.searches[0]!.candidates.push({ source_id: "UNREAD", unit_id: "UNREAD", evidence: [], required_evidence_ids: [] });
  assert.equal(p.progressFingerprint(r), baseline);
  const first = p.updateNoProgress(undefined, baseline), second = p.updateNoProgress(first, p.progressFingerprint(r));
  const third = p.updateNoProgress(second, p.progressFingerprint(r));
  assert.equal(second.stagnant_rounds, 1); assert.equal(third.stagnant_rounds, 2);
  assert.equal(p.chooseAllowedActions(r, { ...allowance, no_progress: third }).stop_reason, "no_progress");
});
test("G7 empty cases retain input-supported suspected risk while permitting a different search", async () => {
  const p = await api(), { f, r } = setup();
  f.state.searches.push({ risk_point_ids: ["RP0001"], focus_refs: [f.refs[0]], goal: "mechanism_search", text: "旧查询", complete: true, status: "empty", candidates: [] });
  await accept(r, f, "RP0001"); const before = r.points.get("RP0001"), decision = p.chooseAllowedActions(r, allowance);
  assert.ok(decision.gaps.some((g: any) => g.type === "case_search"));
  assert.ok(searches(decision).some((a: any) => a.point_id === "RP0001" && a.excluded_query_texts.includes("旧查询") && a.max_queries === 1));
  assert.strictEqual(r.points.get("RP0001"), before); assert.equal(before!.accepted!.result.judgment, "suspected");
});
test("same sentence independent points use query ownership instead of overlapping refs", async () => {
  const p = await api(), f = fixture(), r = new LoopRegistry(f.task, f.state, f.budget), plan = f.plan(2);
  plan.points[1].input_refs = structuredClone(plan.points[0].input_refs); plan.points[1].facts = structuredClone(plan.points[0].facts);
  r.registerPlan(plan); const option = f.addCases()[0]!; await accept(r, f, "RP0001", [[comparison(option.case_ref)]]);
  assert.ok(p.derivePointGaps(r).some((g: any) => g.point_id === "RP0002" && g.type === "case_search"));
  assert.equal(searches(p.chooseAllowedActions(r, allowance)).some((a: any) => a.point_id === "RP0001"), false);
});
test("per-tag and per-unit comparisons cannot borrow another accepted case applicability", async () => {
  const p = await api(), { f, r } = setup(1, ["赠送利益", "诱导投保"]), options = f.addCases(2, true);
  await accept(r, f, "RP0001", [[comparison(options[0]!.case_ref)], [comparison(options[1]!.case_ref, "partial_match")]]);
  const gaps = p.derivePointGaps(r);
  assert.ok(gaps.some((g: any) => g.type === "case_comparison" && g.tag === "诱导投保"));
  assert.equal(gaps.some((g: any) => g.type === "case_comparison" && g.tag === "赠送利益"), false);
});
test("unknown tag does not prevent same-point supported tag case search", async () => {
  const p = await api(), { f, r } = setup(1, ["合同外利益", "赠送利益"]); await accept(r, f, "RP0001");
  const actions = searches(p.chooseAllowedActions(r, allowance));
  assert.ok(actions.some((a: any) => a.tag === "赠送利益")); assert.equal(actions.some((a: any) => a.tag === "合同外利益"), false);
});
test("global circuits audit identity permissions budget and program blockers cannot be local_partial", async () => {
  const p = await api();
  for (const kind of ["transport", "identity", "permission", "budget", "program"]) {
    const { f, r } = setup(), option = f.addCases()[0]!; partial(f.state.searches[0]!);
    await accept(r, f, "RP0001", [[comparison(option.case_ref)]]);
    r.recordDiagnostic({ scope: r.scope("act"), code: "BLOCKED_TEST", category: "program", blocker_class: kind });
    const decision = p.chooseAllowedActions(r, allowance);
    assert.deepEqual(decision.actions, []); assert.equal(decision.local_ranking_limited, false);
    assert.equal(decision.gaps.some((g: any) => g.local_partial), false);
  }
  const { r } = setup();
  assert.equal(p.chooseAllowedActions(r, { ...allowance, transport_circuit_open: true }).stop_reason, "execution_blocked");
});
test("lawful partial ranking can search a distinct independent business gap but not its ranking", async () => {
  const p = await api(), { f, r } = setup(2), option = f.addCases()[0]!; partial(f.state.searches[0]!);
  await accept(r, f, "RP0001", [[comparison(option.case_ref)]]);
  const decision = p.chooseAllowedActions(r, allowance);
  assert.ok(searches(decision).some((a: any) => a.point_id === "RP0002"));
  assert.equal(searches(decision).some((a: any) => a.point_id === "RP0001"), false);
});
test("read coverage uses exact union windows rather than a single span or read_anchor_ids claim", async () => {
  const p = await api(), { f, r } = setup(); f.addCases(); const full = f.state.evidence.get("W0")!;
  f.state.evidence.delete("W0"); f.state.contextEvidenceIds.clear(); const middle = Math.floor(full.end / 2);
  const left = { ...full, evidence_id: "LEFT", end: middle, quote: Array.from(full.quote).slice(0, middle).join("") };
  const right: JsonObject = { ...full, evidence_id: "RIGHT", start: middle, quote: Array.from(full.quote).slice(middle).join("") };
  f.state.evidence.set("LEFT", left); f.state.evidence.set("RIGHT", right);
  assert.equal(p.derivePointGaps(r).some((g: any) => g.type === "unread_source"), false);
  right.text_sha256 = hash("wrong");
  assert.ok(p.derivePointGaps(r).some((g: any) => g.type === "unread_source"));
});
test("read candidate without current assessment is scheduled for assessment without claiming comparison", async () => {
  const p = await api(), { f, r } = setup(); f.addCases();
  const decision = p.chooseAllowedActions(r, allowance);
  assert.deepEqual(decision.assessment_required_point_ids, ["RP0001"]); assert.deepEqual(decision.actions, []);
  assert.equal(decision.gaps.some((g: any) => g.type === "case_comparison"), false);
});
test("only actual new read source or required anchor and accepted semantic coverage count as progress", async () => {
  const p = await api(), { f, r } = setup(); f.addCases();
  const initial = p.progressFingerprint(r); r.observeEvidence(["W0"]); const read = p.progressFingerprint(r);
  assert.notEqual(read, initial); r.markPointDirty("RP0001", "FORMAT_REPAIR"); assert.equal(p.progressFingerprint(r), read);
  await accept(r, f, "RP0001"); assert.notEqual(p.progressFingerprint(r), read);
  const changed = p.updateNoProgress({ fingerprint: initial, stagnant_rounds: 1 }, p.progressFingerprint(r));
  assert.equal(changed.stagnant_rounds, 0);
});
test("act admission uses explicit reserve and count limits without changing the task budget", async () => {
  const p = await api(), { f, r } = setup(), before = f.budget.summary();
  for (const limits of [{ reassessment_reserved: false }, { ...allowance, feedback_rounds_used: 2 },
    { ...allowance, searches_this_round: 1 }, { ...allowance, remaining_query_attempts: 0 },
    { ...allowance, remaining_search_calls: 0 }, { ...allowance, remaining_tool_actions: 0 }]) {
    assert.deepEqual(p.chooseAllowedActions(r, limits).actions, []);
  }
  const after = f.budget.summary(); delete before.elapsed_ms; delete after.elapsed_ms; assert.deepEqual(after, before);
});
test("gaps have deterministic program ids current assessment basis and ignore model repeated gaps", async () => {
  const p = await api(), { f, r } = setup(); await accept(r, f, "RP0001");
  const before = p.derivePointGaps(r); f.state.gaps.set("MODEL", { type: "unread_source", risk_point_id: "RP0009" });
  assert.deepEqual(p.derivePointGaps(r), before);
  assert.ok(before.every((g: any) => g.gap_id && g.basis.assessment_revision === 1 && Array.isArray(g.allowed_actions)));
});

test("gap assessment basis refers to the accepted revision rather than an in-flight request", async () => {
  const p = await api(), { f, r } = setup(); await accept(r, f, "RP0001");
  r.beginAssessment("RP0001");
  assert.ok(p.derivePointGaps(r).every((g: any) => g.basis.assessment_revision === 1));
});
test("reordering equivalent support references is not semantic progress", async () => {
  const p = await api(), { f, r } = setup(), option = f.addCases()[0]!;
  const refs = [{ segment_id: "S0001", quote: "保险", occurrence: 1 }, f.refs[0]];
  await accept(r, f, "RP0001", [[comparison(option.case_ref)]], v => {
    for (const c of v.tag_checks[0].conditions) c.input_refs = refs;
  });
  const before = p.progressFingerprint(r);
  await accept(r, f, "RP0001", [[comparison(option.case_ref)]], v => {
    for (const c of v.tag_checks[0].conditions) c.input_refs = [...refs].reverse();
  });
  assert.equal(p.progressFingerprint(r), before);
});
test("actual accepted applicability change and a newly read required anchor are progress", async () => {
  const p = await api(), { f, r } = setup(), option = f.addCases()[0]!;
  await accept(r, f, "RP0001", [[comparison(option.case_ref, "partial_match")]]);
  const partial = p.progressFingerprint(r);
  await accept(r, f, "RP0001", [[comparison(option.case_ref)]]); const matched = p.progressFingerprint(r);
  assert.notEqual(partial, matched); r.observeEvidence(["W0"]); assert.notEqual(p.progressFingerprint(r), matched);
});
test("a remaining unread candidate is read before replacing a partial compared case", async () => {
  const p = await api(), { f, r } = setup(), options = f.addCases(2);
  f.state.evidence.delete("W1"); f.state.contextEvidenceIds.delete("W1");
  await accept(r, f, "RP0001", [[comparison(options[0]!.case_ref, "partial_match")]]);
  const decision = p.chooseAllowedActions(r, allowance);
  assert.ok(decision.actions.length); assert.ok(decision.actions.every((a: any) => a.tool === "read_source"));
});
test("identity changes and nonlocal ranking failure never open the local acceptable path", async () => {
  const p = await api(), { f, r } = setup(), option = f.addCases()[0]!; partial(f.state.searches[0]!);
  await accept(r, f, "RP0001", [[comparison(option.case_ref)]]);
  f.task.snapshot.snapshot_id = "ALTERED";
  const changed = p.chooseAllowedActions(r, allowance);
  assert.equal(changed.stop_reason, "execution_blocked"); assert.equal(changed.local_ranking_limited, false);
  const other = setup(); other.f.addCases(); partial(other.f.state.searches[0]!);
  other.f.state.searches[0]!.ranking_coverage.failure_counts = { UNCLASSIFIED_RERANK_FAILURE: 1 };
  assert.equal(p.chooseAllowedActions(other.r, allowance).stop_reason, "execution_blocked");
});

test("a not-met criterion excludes the tag without demanding unrelated unknown input facts", async () => {
  const p = await api(), f = fixture(), text = "内部培训反例：禁止购买保险就送购物卡。";
  Object.assign(f.task.input, { text, text_sha256: hash(text), input_sha256: hash(text), segments: [{ segment_id: "S0001", source_id: "INPUT:loop", start: 0, end: Array.from(text).length, text }] });
  f.refs[0] = { segment_id: "S0001", quote: text, occurrence: 1 };
  const r = new LoopRegistry(f.task, f.state, f.budget); r.registerPlan(f.plan(1, ["合同外利益"]));
  await accept(r, f, "RP0001", [], v => {
    v.tag_checks[0].conditions.find((c: any) => c.criterion_id === "actual_conduct").status = "not_met";
  });
  const decision = p.chooseAllowedActions(r, allowance);
  assert.equal(r.points.get("RP0001")!.accepted!.result.tag_checks[0].status, "not_supported");
  assert.equal(decision.gaps.some((g: any) => g.type === "input_fact"), false);
  assert.equal(decision.stop_reason, "evidence_sufficient"); assert.deepEqual(decision.actions, []);
});
test("global blocker stops even before points exist", async () => {
  const p = await api(), f = fixture(), empty = new LoopRegistry(f.task, f.state, f.budget);
  empty.recordDiagnostic({ scope: empty.scope("plan"), code: "BUDGET_EXHAUSTED", category: "program", blocker_class: "budget" });
  assert.equal(p.chooseAllowedActions(empty, allowance).stop_reason, "execution_blocked");
});
test("completed query cannot conceal a tool error behind its success flag", async () => {
  const p = await api(), other = setup(), option = other.f.addCases()[0]!;
  await accept(other.r, other.f, "RP0001", [[comparison(option.case_ref)]]);
  other.f.state.searches[0]!.error = { code: "HTTP_4XX" };
  assert.equal(p.chooseAllowedActions(other.r, allowance).stop_reason, "execution_blocked");
});
test("same citation reference with a different window identifier is not semantic progress", async () => {
  const p = await api(), { buildCitationOptions } = await import("../src/citation_bundles.js"), { f, r } = setup(), option = f.addCases()[0]!;
  await accept(r, f, "RP0001", [[comparison(option.case_ref)]]); const before = p.progressFingerprint(r);
  f.state.evidence.set("A", { ...f.state.evidence.get("W0"), evidence_id: "A" }); f.state.contextEvidenceIds.add("A");
  const replacement = buildCitationOptions("SNAPSHOT", f.state.searches[0]!.candidates, [...f.state.evidence.values()])[0]!;
  assert.equal(replacement.case_ref, option.case_ref); assert.deepEqual(replacement.citation_evidence_ids, ["A"]);
  f.state.citationOptions.set(replacement.case_ref, replacement);
  await accept(r, f, "RP0001", [[comparison(replacement.case_ref)]]);
  assert.equal(p.progressFingerprint(r), before);
});

test("candidate capacity overflow remains unfinished without inventing another gap or action", async () => {
  const p = await api(), tags = ["赠送利益", "诱导投保", "虚假宣传"], { f, r } = setup(1, tags), option = f.addCases()[0]!;
  await accept(r, f, "RP0001", tags.map(() => [comparison(option.case_ref)]));
  assert.equal(p.chooseAllowedActions(r, allowance).stop_reason, "evidence_sufficient");
  assert.throws(() => r.expandCandidates("RP0001", ["合同外利益"], ["合同外利益"]), /LOOP_CANDIDATE_LIMIT/);
  const decision = p.chooseAllowedActions(r, allowance);
  assert.deepEqual(decision.unfinished_point_ids, ["RP0001"]); assert.equal(decision.stop_reason, "additional_mechanisms_uncovered");
  assert.deepEqual(decision.actions, []); assert.deepEqual(decision.gaps, []);
});
