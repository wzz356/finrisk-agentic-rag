import test from "node:test";
import assert from "node:assert/strict";
import { fixture, comparison, business } from "./agent_loop_fixture.js";
import { canonical, hash } from "../src/contracts.js";
import { criteriaFor } from "../src/criteria.js";
import { normalizeModelOutput } from "../src/model_protocol.js";
const modulePath = "../src/agent_loop_state.js";
const api = () => import(modulePath);
function unchanged(state: any, before: ReturnType<typeof business>) {
  for (const row of before) { assert.strictEqual(state[row.k], row.identity, row.k); assert.deepEqual(state[row.k], row.value, row.k); }
}
test("invalid second plan point leaves registry maps entries and evidence identities untouched", async () => {
  const { LoopRegistry } = await api(), f = fixture(), r = new LoopRegistry(f.task, f.state, f.budget);
  r.registerPlan(f.plan()); const old = r.points.get("RP0001"), points = r.points, reads = r.readEvidence;
  const before = business(f.state), value = f.plan(2); value.points[1].facts[1].context_refs = ["$.user_context.additional_notes"];
  assert.throws(() => r.registerPlan(value), /INVALID_CONTEXT_REFERENCE/);
  unchanged(f.state, before); assert.strictEqual(r.points, points); assert.strictEqual(r.points.get("RP0001"), old);
  assert.strictEqual(r.readEvidence, reads); assert.equal(r.plan_revision, 1);
});
test("plan numbers points in order and binds one task budget and exact identities", async () => {
  const { LoopRegistry } = await api(), f = fixture(), r = new LoopRegistry(f.task, f.state, f.budget);
  r.registerPlan(f.plan(2)); assert.deepEqual([...r.points.keys()], ["RP0001", "RP0002"]);
  assert.strictEqual(r.budget, f.budget); assert.equal(f.state.protocolVersion, "finrisk-model-wire/1.2");
  assert.equal(r.identity.input_sha256, f.task.input.input_sha256);
  f.task.snapshot.snapshot_id = "ALTERED"; assert.throws(() => r.beginAssessment("RP0001"), /LOOP_IDENTITY_CHANGED/);
});
test("late assessment and revision cannot overwrite accepted point", async () => {
  const { LoopRegistry } = await api(), f = fixture(), r = new LoopRegistry(f.task, f.state, f.budget);
  r.registerPlan(f.plan()); const old = r.beginAssessment("RP0001"), next = r.beginAssessment("RP0001");
  await assert.rejects(r.acceptPointAssessment(f.assess(old), old, f.bridge), /LOOP_STALE_ASSESSMENT/);
  await r.acceptPointAssessment(f.assess(next), next, f.bridge);
  r.markPointDirty("RP0001", "NEW_EVIDENCE");
  await assert.rejects(r.acceptPointAssessment(f.assess(next), next, f.bridge), /LOOP_STALE_ASSESSMENT/);
  assert.equal(r.points.get("RP0001").processing_state, "dirty"); assert.ok(r.points.get("RP0001").draft);
});
test("late normalization error never leaks bindings or business state", async () => {
  const { LoopRegistry } = await api(), f = fixture(), r = new LoopRegistry(f.task, f.state, f.budget);
  r.registerPlan(f.plan()); const option = f.addCases()[0]!, request = r.beginAssessment("RP0001");
  const value = f.assess(request, [[comparison(option.case_ref)]]); value.tag_checks[0].conditions.at(-1).input_refs = [];
  const before = business(f.state), point = r.points.get("RP0001");
  await assert.rejects(r.acceptPointAssessment(value, request, f.bridge), /CONDITION_WITHOUT_CURRENT_SUPPORT/);
  unchanged(f.state, before); assert.strictEqual(r.points.get("RP0001"), point); assert.equal(f.state.caseTagBindings.size, 0);
});
test("revision changing while verify awaits rejects atomic commit", async () => {
  const { LoopRegistry } = await api(), f = fixture(), r = new LoopRegistry(f.task, f.state, f.budget);
  r.registerPlan(f.plan()); const option = f.addCases()[0]!, request = r.beginAssessment("RP0001"), before = business(f.state);
  const bridge = { ...f.bridge, async verifyEvidence(ids: string[]) { r.markPointDirty("RP0001", "NEW_EVIDENCE"); return { valid_ids: ids, rejected_ids: [] }; } };
  await assert.rejects(r.acceptPointAssessment(f.assess(request, [[comparison(option.case_ref)]]), request, bridge), /LOOP_STALE_ASSESSMENT/);
  unchanged(f.state, before); assert.equal(r.points.get("RP0001").accepted, undefined);
});
test("accepting P1 preserves P2 accepted identity and pending report phase", async () => {
  const { LoopRegistry } = await api(), f = fixture(), r = new LoopRegistry(f.task, f.state, f.budget);
  r.registerPlan(f.plan(2)); const p2req = r.beginAssessment("RP0002"); await r.acceptPointAssessment(f.assess(p2req), p2req, f.bridge);
  const p2 = r.points.get("RP0002"), p1req = r.beginAssessment("RP0001");
  await r.acceptPointAssessment(f.assess(p1req), p1req, f.bridge);
  assert.strictEqual(r.points.get("RP0002"), p2); assert.equal(f.state.phase, "reviewing");
});
test("request freezes citation map and evidence visibility across later point contexts", async () => {
  const { LoopRegistry } = await api(), f = fixture(), r = new LoopRegistry(f.task, f.state, f.budget);
  r.registerPlan(f.plan(2)); const option = f.addCases()[0]!, projection = r.projectPointState("RP0001");
  const request = r.beginAssessment("RP0001", projection);
  projection.citationOptions.clear(); projection.contextEvidenceIds.clear(); f.state.citationOptions.clear(); f.state.contextEvidenceIds.clear();
  assert.equal(request.citationOptions.size, 1); assert.equal(request.contextEvidenceIds.has("W0"), true);
  await r.acceptPointAssessment(f.assess(request, [[comparison(option.case_ref)]]), request, f.bridge);
  assert.deepEqual(r.points.get("RP0001").accepted.result.case_matches[0].evidence_ids, ["W0"]);
});
test("unread withdrawn and unknown case references are rejected", async () => {
  const { LoopRegistry } = await api();
  for (const failure of ["unread", "withdrawn", "unknown"]) {
    const f = fixture(), r = new LoopRegistry(f.task, f.state, f.budget); r.registerPlan(f.plan());
    const option = f.addCases()[0]!;
    if (failure === "withdrawn") f.state.citationOptions.clear();
    const request = r.beginAssessment("RP0001"); if (failure === "unread") f.state.evidence.get("W0")!.was_read = false;
    const value = f.assess(request, [[comparison(failure === "unknown" ? "C" + "f".repeat(20) : option.case_ref)]]);
    await assert.rejects(r.acceptPointAssessment(value, request, f.bridge), /CASE|CITATION|EVIDENCE/);
    assert.equal(r.points.get("RP0001").accepted, undefined);
  }
});
test("new evidence dirties only linked point and retains old accepted solely as draft", async () => {
  const { LoopRegistry } = await api(), f = fixture(), r = new LoopRegistry(f.task, f.state, f.budget);
  r.registerPlan(f.plan(2)); for (const id of r.points.keys()) { const req = r.beginAssessment(id); await r.acceptPointAssessment(f.assess(req), req, f.bridge); }
  const p2 = r.points.get("RP0002"), old = r.points.get("RP0001").accepted;
  f.addCases(); r.observeEvidence(["W0"]);
  assert.equal(r.points.get("RP0001").accepted, undefined); assert.strictEqual(r.points.get("RP0001").draft, old);
  assert.strictEqual(r.points.get("RP0002"), p2); assert.equal(r.points.get("RP0002").processing_state, "assessed");
});
test("candidate extension retains prior tags versions and maximum three", async () => {
  const { LoopRegistry } = await api(), f = fixture(), r = new LoopRegistry(f.task, f.state, f.budget);
  r.registerPlan(f.plan()); const req = r.beginAssessment("RP0001");
  r.expandCandidates("RP0001", ["合同外利益"], ["合同外利益"]); assert.deepEqual(r.points.get("RP0001").candidate_tags, ["赠送利益", "合同外利益"]);
  assert.equal(r.points.get("RP0001").candidate_version, 2);
  await assert.rejects(r.acceptPointAssessment(f.assess(req), req, f.bridge), /LOOP_STALE_ASSESSMENT/);
  assert.throws(() => r.expandCandidates("RP0001", ["诱导投保"], []), /LOOP_DEFINITION_REQUIRED/);
  r.expandCandidates("RP0001", ["诱导投保"], ["诱导投保"]);
  assert.throws(() => r.expandCandidates("RP0001", ["虚假宣传"], ["虚假宣传"]), /LOOP_CANDIDATE_LIMIT/);
  assert.equal(r.points.get("RP0001").additional_mechanisms_uncovered, true);
});
test("controlled six case projection accepts new mode while legacy default rejects six", async () => {
  const { LoopRegistry } = await api(), f = fixture(), r = new LoopRegistry(f.task, f.state, f.budget);
  const tags = ["赠送利益", "合同外利益", "诱导投保"]; r.registerPlan(f.plan(1, tags)); const options = f.addCases(6);
  const req = r.beginAssessment("RP0001"), cases = [0, 1, 2].map(i => options.slice(i * 2, i * 2 + 2).map(o => comparison(o.case_ref)));
  await r.acceptPointAssessment(f.assess(req, cases), req, f.bridge); assert.equal(r.points.get("RP0001").accepted.result.case_matches.length, 6);
  const checks = f.assess(req).tag_checks.map((c: any) => ({ tag: c.tag, conditions: c.conditions }));
  const wire = { kind: "final", segments: f.segments, points: [{ risk_point_id: "RP0001", tag_checks: checks,
    case_matches: options.map(o => ({ ...comparison(o.case_ref), supports_tags: ["赠送利益"] })) }] };
  assert.throws(() => normalizeModelOutput(wire, "final", f.state, f.task), /INVALID_WIRE/);
});
test("one shot accepts all points atomically and rejects late second point without mutation", async () => {
  const { LoopRegistry } = await api(), f = fixture(), r = new LoopRegistry(f.task, f.state, f.budget);
  const make = () => ({ kind: "one_shot", segments: f.segments, points: f.plan(2).points.map((p: any, i: number) => ({
    input_refs: p.input_refs, facts: p.facts, tag_checks: f.assess({ assessment_id: "AS0001", state_revision: 1, risk_point_id: "RP000" + (i + 1), candidate_tags: p.candidate_tags }).tag_checks })) });
  const request = r.beginOneShot(), bad = make(); bad.points[1].tag_checks[0].conditions[0].input_refs = [];
  const before = business(f.state); await assert.rejects(r.acceptOneShot(bad, request, f.bridge), /CONDITION_WITHOUT_CURRENT_SUPPORT/); unchanged(f.state, before);
  assert.equal(r.points.size, 0);
  await r.acceptOneShot(make(), request, f.bridge); assert.deepEqual([...r.points.keys()], ["RP0001", "RP0002"]);
  assert.ok([...r.points.values()].every((p: any) => p.processing_state === "assessed"));
});
test("Unicode LF CRLF and emoji input and source spans retain exact offsets hashes", async () => {
  const { LoopRegistry } = await api(), f = fixture(), r = new LoopRegistry(f.task, f.state, f.budget);
  r.registerPlan(f.plan(2)); f.addCases(); const req = r.beginAssessment("RP0001");
  await r.acceptPointAssessment(f.assess(req, [[comparison([...req.citationOptions.keys()][0])]]), req, f.bridge);
  const resolved = f.state.resolve(f.refs[1]!); assert.equal(resolved.start, Array.from("购买这份保险就送购物卡。\r\n").length);
  assert.equal(resolved.quote[0], "😀"[0]); assert.equal(resolved.text_sha256, hash(f.task.input.text));
  assert.equal(canonical(r.points.get("RP0001").accepted.evidence[0]), canonical(f.state.evidence.get("W0")));
});
test("diagnostics stay append only and retirement requires exact accepted typed scope", async () => {
  const { LoopRegistry } = await api(), f = fixture(), r = new LoopRegistry(f.task, f.state, f.budget); r.registerPlan(f.plan());
  const req = r.beginAssessment("RP0001"), scope = r.scope("assess", "RP0001", req.assessment_id);
  r.recordDiagnostic({ scope, code: "LOOP_SCHEMA", category: "schema", blocker_class: "typed_output" });
  r.recordDiagnostic({ scope, code: "BUDGET_EXHAUSTED", category: "program", blocker_class: "budget" });
  assert.throws(() => r.retireTypedBlockers(scope), /LOOP_RECOVERY_NOT_ACCEPTED/);
  await r.acceptPointAssessment(f.assess(req), req, f.bridge); r.retireTypedBlockers(scope);
  assert.equal(r.attempt_diagnostics.length, 2); assert.equal(r.active_blockers.size, 1);
  assert.equal([...r.active_blockers.values()][0].blocker_class, "budget");
  assert.throws(() => (r.attempt_diagnostics as any).push({}), TypeError);
});

test("one shot new evidence during verification invalidates the whole request", async () => {
  const { LoopRegistry } = await api(), f = fixture(), r = new LoopRegistry(f.task, f.state, f.budget);
  const o = f.addCases()[0]!, req = r.beginOneShot(), base = f.plan().points[0];
  const checks = f.assess({ ...req, risk_point_id: "RP0001", candidate_tags: ["赠送利益"] }, [[comparison(o.case_ref)]]).tag_checks;
  const value = { kind: "one_shot", segments: f.segments, points: [{ input_refs: base.input_refs, facts: base.facts, tag_checks: checks }] };
  const bridge = { ...f.bridge, async verifyEvidence(ids: string[]) { r.observeEvidence(["W0"]); return { valid_ids: ids, rejected_ids: [] }; } };
  const before = business(f.state);
  await assert.rejects(r.acceptOneShot(value, req, bridge), /LOOP_STALE_ASSESSMENT/); unchanged(f.state, before); assert.equal(r.points.size, 0);
});
test("global identity budget permission transport diagnostics cannot masquerade as retireable branch output", async () => {
  const { LoopRegistry } = await api(), f = fixture(), r = new LoopRegistry(f.task, f.state, f.budget);
  for (const code of ["SOURCE_HASH_OR_EVIDENCE_MISMATCH", "BUDGET_EXHAUSTED", "PERMISSION_DENIED", "GLOBAL_TRANSPORT_CIRCUIT"]) {
    assert.throws(() => r.recordDiagnostic({ scope: r.scope("one_shot"), code, category: "semantic", blocker_class: "branch_output" }), /LOOP_INVALID_DIAGNOSTIC/);
  }
  assert.throws(() => r.recordDiagnostic({ scope: { ...r.scope("one_shot"), extra: true }, code: "LOOP_SCHEMA", category: "schema", blocker_class: "branch_output" }), /LOOP_INVALID_DIAGNOSTIC/);
});
test("one shot escalation keeps trusted reads queries budget diagnostics and retires only exact failed branch after full assessment", async () => {
  const { LoopRegistry } = await api(), f = fixture(), r = new LoopRegistry(f.task, f.state, f.budget), option = f.addCases()[0]!;
  r.recordQuery(f.state.searches[0]!, "one_shot"); const q = [...r.queries.values()][0], req = r.beginOneShot(), base = f.plan().points[0];
  const tag_checks = f.assess({ ...req, risk_point_id: "RP0001", candidate_tags: ["赠送利益"] }, [[comparison(option.case_ref)]]).tag_checks;
  r.recordDiagnostic({ scope: r.scope("one_shot"), code: "LOOP_SCHEMA", category: "schema", blocker_class: "branch_output" });
  r.recordDiagnostic({ scope: r.scope("one_shot"), code: "BUDGET_EXHAUSTED", category: "program", blocker_class: "budget" });
  await r.acceptOneShot({ kind: "one_shot", segments: f.segments, points: [{ input_refs: base.input_refs, facts: base.facts, tag_checks }] }, req, f.bridge);
  const evidence = f.state.evidence, e = evidence.get("W0"), searches = f.state.searches, attempts = r.attempt_diagnostics;
  assert.throws(() => r.registerPlan(f.plan(2)), /LOOP_PLAN_ALREADY_REGISTERED/);
  r.registerPlan(f.plan(2), { replace_one_shot: true });
  assert.strictEqual(f.state.evidence, evidence); assert.strictEqual(evidence.get("W0"), e); assert.strictEqual(f.state.searches, searches);
  assert.strictEqual([...r.queries.values()][0], q); assert.strictEqual(r.budget, f.budget); assert.equal(r.plan_revision, 2);
  assert.deepEqual(r.attempt_diagnostics, attempts); assert.equal(r.points.get("RP0001").accepted, undefined);
  assert.throws(() => r.retireSupersededOneShotBlockers(), /LOOP_RECOVERY_NOT_ACCEPTED/);
  for (const id of r.points.keys()) { const next = r.beginAssessment(id); await r.acceptPointAssessment(f.assess(next), next, f.bridge); }
  r.retireSupersededOneShotBlockers(); assert.equal(r.active_blockers.size, 1); assert.equal([...r.active_blockers.values()][0].blocker_class, "budget");
  assert.equal(r.attempt_diagnostics.length, 2);
});
test("accepted sidecar conditions reflect the fact locks applied by normalization", async () => {
  const { LoopRegistry } = await api(), f = fixture(), r = new LoopRegistry(f.task, f.state, f.budget);
  const plan = f.plan(); plan.points[0].facts[0] = { field: "insurance_product", value: null, origin: "unknown", input_refs: [], context_refs: [] };
  r.registerPlan(plan); const req = r.beginAssessment("RP0001"); await r.acceptPointAssessment(f.assess(req), req, f.bridge);
  const a = r.points.get("RP0001").accepted;
  assert.equal(a.result.tag_checks[0].conditions.find((c: any) => c.condition_id === "insurance_product").status, "unknown");
  assert.equal(a.tag_assessments[0].conditions.find((c: any) => c.criterion_id === "insurance_product").status, "unknown");
});

test("same sentence independent points use explicit query IDs and evidence associations", async () => {
  const { LoopRegistry } = await api(), f = fixture(), r = new LoopRegistry(f.task, f.state, f.budget);
  const plan = f.plan(2); plan.points[1].input_refs = structuredClone(plan.points[0].input_refs);
  plan.points[1].candidate_tags = ["合同外利益"]; r.registerPlan(plan);
  for (const id of r.points.keys()) { const req = r.beginAssessment(id); await r.acceptPointAssessment(f.assess(req), req, f.bridge); }
  const p2 = r.points.get("RP0002"), query = { focus_refs: [f.refs[0]], risk_point_ids: ["RP0001"], goal: "mechanism_search",
    text: "仅查赠礼", complete: false, status: "error", candidates: [] };
  r.recordQuery(query); assert.strictEqual(r.points.get("RP0002"), p2);
  assert.equal(r.projectPointState("RP0002").searches.includes(query), false);
});
test("already globally read evidence first associated with another point still dirties that point", async () => {
  const { LoopRegistry } = await api(), f = fixture(), r = new LoopRegistry(f.task, f.state, f.budget);
  r.registerPlan(f.plan(2)); f.addCases(); r.observeEvidence(["W0"], ["RP0001"]);
  for (const id of r.points.keys()) { const req = r.beginAssessment(id); await r.acceptPointAssessment(f.assess(req), req, f.bridge); }
  const p1 = r.points.get("RP0001"), p2 = r.points.get("RP0002");
  r.observeEvidence(["W0"], ["RP0002"]);
  assert.strictEqual(r.points.get("RP0001"), p1); assert.equal(r.points.get("RP0002").processing_state, "dirty");
  assert.equal(r.points.get("RP0002").state_revision, p2.state_revision + 1);
  assert.equal(r.readEvidence.size, 1); const current = r.points.get("RP0002"); r.observeEvidence(["W0"], ["RP0002"]); assert.strictEqual(r.points.get("RP0002"), current);
});

test("a request can commit only once even when an earlier duplicate reply is still verifying", async () => {
  const { LoopRegistry } = await api(), f = fixture(), r = new LoopRegistry(f.task, f.state, f.budget);
  r.registerPlan(f.plan()); const o = f.addCases()[0]!, req = r.beginAssessment("RP0001");
  let release!: () => void, started!: () => void; const entered = new Promise<void>(resolve => { started = resolve; });
  const blocked = new Promise<void>(resolve => { release = resolve; });
  const bridge = { ...f.bridge, async verifyEvidence(ids: string[]) { started(); await blocked; return { valid_ids: ids, rejected_ids: [] }; } };
  const late = r.acceptPointAssessment(f.assess(req, [[comparison(o.case_ref, "partial_match")]]), req, bridge); await entered;
  await r.acceptPointAssessment(f.assess(req, [[comparison(o.case_ref)]]), req, f.bridge); const accepted = r.points.get("RP0001").accepted;
  release(); await assert.rejects(late, /LOOP_STALE_ASSESSMENT/); assert.strictEqual(r.points.get("RP0001").accepted, accepted);
});

test("typed plan recovery retires only the exact accepted plan scope and never act or global blockers", async () => {
  const { LoopRegistry } = await api(), f = fixture(), r = new LoopRegistry(f.task, f.state, f.budget), scope = r.scope("plan");
  r.recordDiagnostic({ scope, code: "LOOP_SCHEMA", category: "schema", blocker_class: "typed_output" });
  assert.throws(() => r.retireTypedBlockers(scope), /LOOP_RECOVERY_NOT_ACCEPTED/);
  r.registerPlan(f.plan()); r.retireTypedBlockers(scope); assert.equal(r.active_blockers.size, 0); assert.equal(r.attempt_diagnostics.length, 1);
  const req = r.beginAssessment("RP0001"), actScope = r.scope("act", "RP0001", req.assessment_id);
  r.recordDiagnostic({ scope: actScope, code: "LOOP_SCHEMA", category: "schema", blocker_class: "typed_output" });
  await r.acceptPointAssessment(f.assess(req), req, f.bridge);
  assert.throws(() => r.retireTypedBlockers(actScope), /LOOP_RECOVERY_NOT_ACCEPTED/); assert.equal(r.active_blockers.size, 1);
});
test("registry rejects initially divergent task and evidence state inputs", async () => {
  const { LoopRegistry } = await api(), f = fixture(); f.state.input = structuredClone(f.task.input); f.state.input.source_id = "OTHER";
  assert.throws(() => new LoopRegistry(f.task, f.state, f.budget), /LOOP_INPUT_STATE_MISMATCH/);
});

test("quality: union-read anchors are audited only for matching source version hash and quote", async () => {
  const { LoopRegistry } = await api();
  for (const invalid of ["none", "source", "version", "hash", "quote"]) {
    const f = fixture(), r = new LoopRegistry(f.task, f.state, f.budget); r.registerPlan(f.plan()); f.addCases();
    const candidate = f.state.searches[0]!.candidates[0], anchor = candidate.evidence[0], chars = Array.from(anchor.quote as string),
      middle = Math.floor(chars.length / 2);
    f.state.evidence.delete("W0"); f.state.contextEvidenceIds.delete("W0"); f.state.citationOptions.clear();
    const left = { ...anchor, evidence_id: "LEFT", start: anchor.start, end: anchor.start + middle, quote: chars.slice(0, middle).join(""), was_read: true };
    const right = { ...anchor, evidence_id: "RIGHT", start: anchor.start + middle, end: anchor.end, quote: chars.slice(middle).join(""), was_read: true };
    if (invalid === "source") right.source_id = "OTHER";
    if (invalid === "version") right.text_version = "OTHER";
    if (invalid === "hash") right.text_sha256 = "0".repeat(64);
    if (invalid === "quote") right.quote = "伪".repeat(Array.from(right.quote as string).length);
    f.state.evidence.set("LEFT", left); f.state.contextEvidenceIds.add("LEFT"); r.observeEvidence(["LEFT"], ["RP0001"]);
    assert.deepEqual(r.points.get("RP0001").read_anchor_ids, []);
    f.state.evidence.set("RIGHT", right); f.state.contextEvidenceIds.add("RIGHT"); r.observeEvidence(["RIGHT"], ["RP0001"]);
    assert.deepEqual(r.points.get("RP0001").read_anchor_ids, invalid === "none" ? [anchor.evidence_id] : [], invalid);
    if (invalid === "none") {
      const { buildCitationOptions } = await import("../src/citation_bundles.js");
      const { candidateBundleRead } = await import("../src/model_protocol.js");
      const options = buildCitationOptions("SNAPSHOT", [candidate], [...f.state.evidence.values()]);
      assert.equal(candidateBundleRead(candidate, f.state, true), true); assert.equal(options.length, 1);
      for (const o of options) f.state.citationOptions.set(o.case_ref, o);
      const req = r.beginAssessment("RP0001"); await r.acceptPointAssessment(f.assess(req, [[comparison(options[0]!.case_ref)]]), req, f.bridge);
      assert.deepEqual(r.points.get("RP0001").read_anchor_ids, [anchor.evidence_id]);
    }
  }
});
test("quality: automatic mixed batch records only each point's relevant windows", async () => {
  const { LoopRegistry } = await api(), f = fixture(), r = new LoopRegistry(f.task, f.state, f.budget);
  r.registerPlan(f.plan(2)); f.addCases(2);
  const q = f.state.searches[0]!; q.candidates = [q.candidates[0]];
  f.state.searches.push({ ...q, risk_point_ids: ["RP0002"], focus_refs: [f.refs[1]], text: "P2 case",
    candidates: [{ source_id: "SRC1", unit_id: "UNIT1", required_evidence_ids: ["E1"], evidence: [f.state.evidence.get("E1")] }] });
  r.observeEvidence(["W0", "W1"]);
  assert.deepEqual(r.points.get("RP0001").read_anchor_ids, ["E0"]); assert.deepEqual(r.points.get("RP0002").read_anchor_ids, ["E1"]);
  for (const id of r.points.keys()) { const req = r.beginAssessment(id); await r.acceptPointAssessment(f.assess(req), req, f.bridge); }
  const p1 = r.points.get("RP0001"), p2 = r.points.get("RP0002");
  r.observeEvidence(["W1"], ["RP0001"]);
  assert.equal(r.points.get("RP0001").state_revision, p1.state_revision + 1); assert.equal(r.points.get("RP0001").processing_state, "dirty");
  assert.strictEqual(r.points.get("RP0002"), p2); assert.equal(r.readEvidence.size, 2);
  const current = r.points.get("RP0001"); r.observeEvidence(["W1"], ["RP0001"]); assert.strictEqual(r.points.get("RP0001"), current);
});

test("quality: real backend source verification failure cannot be retired by branch fallback", async () => {
  const { LoopRegistry } = await api(), f = fixture(), r = new LoopRegistry(f.task, f.state, f.budget), option = f.addCases()[0]!,
    req = r.beginOneShot(), scope = r.scope("one_shot"), base = f.plan().points[0];
  const value = { kind: "one_shot", segments: f.segments, points: [{ input_refs: base.input_refs, facts: base.facts,
    tag_checks: f.assess({ ...req, risk_point_id: "RP0001", candidate_tags: ["赠送利益"] }, [[comparison(option.case_ref)]]).tag_checks }] };
  const bridge = { ...f.bridge, async verifyEvidence() { return { valid_ids: [], rejected_ids: ["W0"] }; } };
  await assert.rejects(r.acceptOneShot(value, req, bridge), /LOOP_REPORT_REFERENCE_INVALID/);
  assert.throws(() => r.recordDiagnostic({ scope, code: "LOOP_REPORT_REFERENCE_INVALID", category: "reference", blocker_class: "branch_output" }), /LOOP_INVALID_DIAGNOSTIC/);
  r.recordDiagnostic({ scope, code: "LOOP_REPORT_REFERENCE_INVALID", category: "reference", blocker_class: "identity" });
  r.registerPlan(f.plan());
  const next = r.beginAssessment("RP0001"); await r.acceptPointAssessment(f.assess(next), next, f.bridge);
  r.retireSupersededOneShotBlockers();
  assert.equal(r.active_blockers.size, 1); assert.equal([...r.active_blockers.values()][0].blocker_class, "identity");
  assert.equal(r.attempt_diagnostics.length, 1);
});
