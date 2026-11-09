import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { canonical, hash, root, type JsonObject } from "../src/contracts.js";
import { criteriaFor } from "../src/criteria.js";
import { runPlannedTask } from "../src/agent_loop_runner.js";
import { LoopRegistry } from "../src/agent_loop_state.js";
import { Trace } from "../src/trace.js";
import { setup, stage, ledger } from "./agent_loop_runner_fixture.js";
const text = "购买本保险即向投保人赠送合同约定之外的购物卡。";
function fast(config: JsonObject = {}) {
 const s = setup("planned_adaptive", { fast_path_enabled: true, ...config }), i = s.f.task.input;
 i.text = text; i.text_sha256 = hash(text); i.input_sha256 = hash(canonical({ text, user_context: i.user_context }));
 i.segments = [{ ...i.segments[0]!, text, end: Array.from(text).length }];
 s.f.refs[0]!.quote = text; s.f.refs[1]!.quote = text; return s;
}
function plan(s: ReturnType<typeof fast>, count = 1) {
 const p = s.f.plan(count);
 for (const point of p.points) { point.facts.push({ field: "contract_outside", value: "yes", origin: "input", input_refs: [s.f.refs[0]], context_refs: [] }); point.query_text = "新计划 赠礼"; }
 return p;
}
function checks(s: ReturnType<typeof fast>, context: any, match = "reliable_match", unknown = false): JsonObject {
 const options = ledger(context).citation_options;
 return [{ tag: "赠送利益", conditions: criteriaFor("赠送利益").map(c => ({ criterion_id: c.id, status: unknown && c.id === "actual_conduct" ? "unknown" : "met", input_refs: unknown && c.id === "actual_conduct" ? [] : [s.f.refs[0]], context_refs: [], reason: "当前材料具有对应依据。" })),
 case_assessments: options.length ? [{ case_ref: options.at(-1).case_ref, same_points: ["投保赠礼机制相同。"], differences: ["具体主体不同。"], match_status: match }] : [] }];
}
function shot(s: ReturnType<typeof fast>, context: any, count = 1, match = "reliable_match", unknown = false): JsonObject {
 const p = plan(s).points[0]; return { kind: "one_shot", segments: s.f.segments, points: Array.from({ length: count }, () => ({ input_refs: p.input_refs, facts: p.facts, tag_checks: checks(s, context, match, unknown) })) };
}
function judge(s: ReturnType<typeof fast>, context: any, match = "reliable_match") { const r = stage(context).request; return { kind: "assess", assessment_id: r.assessment_id, state_revision: r.state_revision, risk_point_id: r.risk_point_id, tag_checks: checks(s, context, match) }; }
function side(s: ReturnType<typeof fast>) { return JSON.parse(readFileSync(join(s.f.task.task_dir!, "agent-assessment.json"), "utf8")); }
function dir(s: ReturnType<typeof fast>) { s.f.task.task_dir = mkdtempSync(join(root, ".test-temp-fast-")); }
test("fast success uses one fixed query and one determination with old report and final bound sidecars", async () => {
 const s = fast(); dir(s); const saved: string[] = [], original = Trace.prototype.save;
 Trace.prototype.save = function(name, value) { saved.push(name); return original.call(this, name, value); };
 try {
 const r = await runPlannedTask(s.f.task, s.bridge, [c => { assert.equal(stage(c).kind, "one_shot"); return shot(s, c); }]);
 assert.equal(r.report.resources.model_request_units, 1); assert.equal(r.report.resources.search_query_attempts, 1);
 assert.equal(r.report.task_status, "completed"); assert.equal(r.report.validation.references_valid, true);
 assert.equal(r.events.filter(e => e.type === "fast.selected").length, 1); assert.equal(r.events.filter(e => e.type === "fast.completed" && e.data.complete === true).length, 1);
 assert.equal(saved.filter(n => n === "report.json").length, 1); assert.equal(saved.filter(n => n === "agent-assessment.json").length, 1);
 assert.equal(side(s).identity.report_sha256, hash(readFileSync(join(s.f.task.task_dir!, "report.json"), "utf8")));
 assert.equal(side(s).actions[0].origin, "initial_plan");
 } finally { Trace.prototype.save = original; }
});
test("multiple output points upgrade once then keep only new plan facts and rebind trusted reads", async () => {
 const s = fast(); dir(s); const p = plan(s); p.points[0].facts[0].value = "no"; p.points[0].query_text = canonical({ text, user_context: s.f.task.input.user_context });
 const r = await runPlannedTask(s.f.task, s.bridge, [c => shot(s, c, 2), p, c => { const v = judge(s, c); v.tag_checks[0].conditions.find((x: any) => x.criterion_id === "insurance_product").status = "not_met"; return v; }]);
 assert.equal(r.events.filter(e => e.type === "fast.escalated").length, 1); assert.equal(r.report.resources.model_request_units, 3);
 assert.equal(r.report.resources.search_query_attempts, 1); assert.equal(side(s).points[0].facts[0].value, "no"); assert.equal(side(s).points[0].initial_query_covered, true);
 assert.ok(side(s).points[0].read_anchor_ids.length); assert.ok(s.calls.filter(c => c.name === "read_source").length <= 2);
});
test("actionable case gap upgrades with one new initial query and still permits two feedback searches", async () => {
 const s = fast(), p = plan(s); p.points[0].alternative_query = "不应占反馈额度";
 const act = (n: number) => (c: any) => { assert.equal(stage(c).kind, "act"); return [{ type: "toolCall", id: "s" + n, name: "search_cases", arguments: { queries: [{ text: "反馈赠礼" + n, risk_point_ids: ["RP0001"], focus_refs: [s.f.refs[0]], goal: "mechanism_search" }], top_k: 3, filters: {} } }]; };
 const r = await runPlannedTask(s.f.task, s.bridge, [c => shot(s, c, 1, "partial_match"), p, c => judge(s, c, "partial_match"), act(1), c => judge(s, c, "partial_match"), act(2), c => judge(s, c)]);
 assert.equal(r.report.resources.search_query_attempts, 4); assert.equal(r.report.resources.model_request_units, 7);
 assert.equal(r.events.filter(e => e.type === "loop.feedback_completed").length, 2); assert.equal(r.events.find(e => e.type === "loop.initial_queries_frozen")!.data.queries.length, 1);
});
test("input unknown only remains pending and never counts a complete fast solution", async () => {
 const s = fast(); const r = await runPlannedTask(s.f.task, s.bridge, [c => shot(s, c, 1, "reliable_match", true)]);
 assert.equal(r.report.resources.model_request_units, 1); assert.equal(r.report.stop.reason, "input_facts_missing");
 assert.equal(r.events.find(e => e.type === "fast.completed")!.data.complete, false); assert.equal(r.events.filter(e => e.type === "fast.escalated").length, 0);
});
test("schema and fabricated input/case rejection preserve diagnostics until audited full replacement", async () => {
 for (const failure of ["{", "case", "input"]) {
 const s = fast(); dir(s);
 const bad = (c: any) => { if (failure === "{") return "{"; const v = structuredClone(shot(s, c)); if (failure === "case") v.points[0].tag_checks[0].case_assessments[0].case_ref = "C" + "0".repeat(20); else v.points[0].input_refs[0].quote = "模型编造引用"; return v; };
 const r = await runPlannedTask(s.f.task, s.bridge, [bad, plan(s), c => judge(s, c)]), a = side(s);
 assert.equal(r.report.task_status, "completed", failure + JSON.stringify(r.report.validation));
 assert.ok(a.attempt_diagnostics.some((d: any) => d.scope.branch === "one_shot")); assert.equal(a.active_blockers.length, 0);
 assert.equal(a.blocker_retirements.filter((d: any) => d.reason === "branch_superseded").length, 1);
 assert.equal(r.events.filter(e => e.type === "branch.superseded").length, 1);
 assert.equal(r.events.filter(e => e.type === "loop.typed_format_recovered").length, 0);
 }
});
test("failed replacement or supersession audit cannot retire rejected fast blocker", async () => {
 for (const auditFailure of [false, true]) {
 const s = fast(); dir(s); const original = Trace.prototype.event;
 if (auditFailure) Trace.prototype.event = function(type, data) { if (type === "branch.superseded") throw new Error("AUDIT_WRITE_FAILED"); return original.call(this, type, data); };
 try { const r = await runPlannedTask(s.f.task, s.bridge, ["{", auditFailure ? plan(s) : "{", c => judge(s, c)]), a = side(s);
 assert.equal(r.report.task_status, "partial"); assert.ok(a.active_blockers.some((d: any) => d.blocker_class === "branch_output")); assert.equal(a.blocker_retirements.some((d: any) => d.reason === "branch_superseded"), false);
 } finally { Trace.prototype.event = original; }
 }
});
test("evidence identity permission budget transport and arbitrary program errors never start a replacement plan", async () => {
 for (const code of ["LOOP_EVIDENCE_CHANGED", "LOOP_STALE_ASSESSMENT", "LOOP_PAYLOAD_BINDING_MISMATCH", "LOOP_PERMISSION_DENIED", "budget_exhausted", "TRANSPORT_FAILURE", "UNEXPECTED_PROGRAM_BUG", "ACCEPTED_WITH_BLOCKER"]) {
 const s = fast(), original = LoopRegistry.prototype.acceptOneShot;
 LoopRegistry.prototype.acceptOneShot = async function(value, request, bridge) { if (code === "ACCEPTED_WITH_BLOCKER") { await original.call(this, value, request, bridge); this.recordDiagnostic({ scope: this.scope("one_shot"), code: "TRANSPORT_FAILURE", category: "program", blocker_class: "transport" }); return; } throw new Error(code); };
 try { const r = await runPlannedTask(s.f.task, s.bridge, [c => shot(s, c), plan(s)]);
 assert.equal(r.report.resources.model_request_units, 1, code); assert.equal(r.events.filter(e => e.type === "fast.escalated").length, 0, code); assert.equal(r.report.task_status, "partial", code);
 } finally { LoopRegistry.prototype.acceptOneShot = original; }
 }
});
test("non adaptive modes reject explicit fast config and ordinary material uses a zero query plan", async () => {
 for (const mode of ["staged_one_shot", "planned_fixed"]) { const s = fast(); s.f.task.config!.orchestration_mode = mode as "staged_one_shot" | "planned_fixed"; await assert.rejects(runPlannedTask(s.f.task, s.bridge, []), /FAST_PATH_MODE_CONFLICT/); }
 const s = setup("planned_adaptive", { fast_path_enabled: true }); const p = s.f.plan(0);
 const r = await runPlannedTask(s.f.task, s.bridge, [p]); assert.equal(r.report.resources.model_request_units, 1); assert.equal(r.report.resources.search_query_attempts, 0);
});

test("upgrade keeps the same budget token ledger and shared typed/native correction allowance", async () => {
 const { Budget } = await import("../src/budget.js"), s = fast(), original = Budget.prototype.reserveModel;
 const budgets: any[] = [], before: JsonObject[] = [];
 Budget.prototype.reserveModel = function(input, final) { budgets.push(this); before.push(this.summary()); return original.call(this, input, final); };
 try {
 const r = await runPlannedTask(s.f.task, s.bridge, ["{", "{", plan(s), c => judge(s, c, "partial_match"),
  [{ type: "toolCall", id: "invalid", name: "search_cases", arguments: { queries: [], top_k: 3, filters: {} } }]]);
 assert.equal(new Set(budgets).size, 1); assert.equal(r.report.resources.model_request_units, 5);
 assert.equal(r.report.resources.json_repairs, 1); assert.equal(r.report.resources.tool_format_corrections, 0);
 assert.ok(before.at(-1)!.settled_tokens > before[1]!.settled_tokens); assert.ok(before.at(-1)!.elapsed_ms >= before[1]!.elapsed_ms);
 assert.equal(r.report.resources.search_query_attempts, 2); assert.equal(r.report.task_status, "partial");
 } finally { Budget.prototype.reserveModel = original; }
});
test("spent fast request is retained when the same model budget rejects replacement plan admission", async () => {
 const s = fast({ max_model_requests: 2 }); dir(s);
 const r = await runPlannedTask(s.f.task, s.bridge, ["{", plan(s), c => judge(s, c)]);
 assert.equal(r.report.resources.model_request_units, 1); assert.equal(r.report.resources.search_query_attempts, 1);
 assert.equal(r.events.filter(e => e.type === "plan.accepted").length, 0); assert.equal(r.report.task_status, "partial");
 assert.ok(side(s).active_blockers.some((d: any) => d.blocker_class === "branch_output"));
});

test("a failed query hard stops globally even when the fast output has zero points", async () => {
 const s = fast(), original = s.bridge.call;
 s.bridge.call = async (name: string, args: JsonObject, ...rest: any[]) => {
  const response = await original(name, args, ...rest);
  if (name === "search_cases") {
   response.status = "partial";
   for (const q of response.data.queries) { Object.assign(q, { status: "error", complete: false, candidates: [], error: { code: "TRANSPORT_FAILURE" } }); delete q.ranking_coverage; }
  }
  return response;
 };
 const r = await runPlannedTask(s.f.task, s.bridge, [{ kind: "one_shot", segments: s.f.segments, points: [] }, s.f.plan(0)]);
 assert.equal(r.events.filter(e => e.type === "fast.escalated").length, 0);
 assert.equal(r.events.filter(e => e.type === "plan.accepted").length, 0);
 assert.ok(r.report.resources.model_request_units <= 1); assert.equal(r.report.task_status, "partial");
});
