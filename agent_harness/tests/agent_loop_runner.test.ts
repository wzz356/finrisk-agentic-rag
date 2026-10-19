import { readFileSync, mkdirSync, mkdtempSync } from "node:fs";
import { join } from "node:path";
import test from "node:test";
import assert from "node:assert/strict";
import { fixture } from "./agent_loop_fixture.js";
import { canonical, hash, root, type JsonObject } from "../src/contracts.js";
import type { ScriptStep } from "../src/model_gateway.js";
import { runTask } from "../src/runner.js";

const modulePath = "../src/agent_loop_runner.js";
async function run(f: ReturnType<typeof fixture>, bridge: any, script: ScriptStep[]) {
  const api = await import(modulePath).catch(() => ({}));
  assert.equal(typeof api.runPlannedTask, "function", "planned mode requires a real runnable Pi loop");
  return api.runPlannedTask(f.task, bridge, script);
}
import { stage, setup, assess } from "./agent_loop_runner_fixture.js";
test("adaptive observes gain cases then changes query, reads gift source and reassesses only its point", async () => {
  const { f, bridge, calls } = setup(), assessed: string[] = [], queries: string[] = [];
  const p = f.plan(2); p.points[0].query_text = "收益 案例"; p.points[1].query_text = "收益 比较";
  const judge: ScriptStep = context => { const id = stage(context).request.risk_point_id; assessed.push(id); return assess(f, context, id === "RP0002" && !calls.some(c => c.args.source_id === "GIFT") ? "partial_match" : "reliable_match"); };
  const act: ScriptStep = context => {
    const s = stage(context); assert.equal(s.kind, "act"); assert.ok(s.allowed_actions.some((a: any) => a.point_id === "RP0002"));
    queries.push("观察后 赠礼");
    return [{ type: "toolCall", id: "gift", name: "search_cases", arguments: { queries: [{ text: queries[0], focus_refs: [f.refs[1]], risk_point_ids: ["RP0002"], goal: "mechanism_search" }], top_k: 3, filters: {} } }];
  };
  const result = await run(f, bridge, [p, judge, judge, act, judge]);
  assert.deepEqual(assessed, ["RP0001", "RP0002", "RP0002"], JSON.stringify(result.events.filter((e: any) => ["loop.stopped", "tool.failure"].includes(e.type))));
  assert.equal(calls.filter(c => c.name === "search_cases").length, 3);
  assert.ok(calls.some(c => c.name === "read_source" && c.args.source_id === "GIFT"));
  assert.equal(result.report.risk_points.length, 2); assert.equal(result.report.resources.model_request_units, 5);
  assert.ok(result.events.some((e: any) => e.type === "loop.assessment_accepted" && e.data.risk_point_id === "RP0002"));
  assert.equal(result.report.validation.references_valid, true);
});
test("counterfactual sufficient first assessments perform no feedback search", async () => {
  const { f, bridge, calls } = setup(), p = f.plan(2); p.points[1].query_text += "第二点";
  const judge: ScriptStep = c => assess(f, c);
  const result = await run(f, bridge, [p, judge, judge]);
  assert.equal(calls.filter(c => c.name === "search_cases").length, 2);
  assert.equal(result.report.resources.model_request_units, 3); assert.equal(result.report.risk_points.length, 2);
});
test("two native searches in one reply dispatch only first before finishTurn", async () => {
  const { f, bridge, calls } = setup();
  const params = { queries: [{ text: "反馈 赠礼", focus_refs: [f.refs[0]], risk_point_ids: ["RP0001"], goal: "mechanism_search" }], top_k: 3, filters: {} };
  const result = await run(f, bridge, [f.plan(), c => assess(f, c, "partial_match"),
    [{ type: "toolCall", id: "s1", name: "search_cases", arguments: params }, { type: "toolCall", id: "s2", name: "search_cases", arguments: { ...params, queries: [{ ...params.queries[0], text: "第二个不应执行" }] } }],
    c => assess(f, c)]);
  assert.equal(calls.filter(c => c.name === "search_cases").length, 2);
  assert.ok(result.events.some((e: any) => e.type === "loop.tool_blocked" && e.data.code === "LOOP_ROUND_SEARCH_LIMIT"));
});
test("new runTask dispatches orchestration mode before legacy live/evaluation routing", async () => {
  const { f, bridge } = setup();
  const result = await runTask(f.task, bridge, [f.plan(), c => assess(f, c)]);
  assert.equal(result.report.risk_points.length, 1); assert.ok(result.events.some(e => e.type === "loop.plan_accepted"));
});
test("R1 is strictly one typed determination and never repairs or upgrades", async () => {
  const { f, bridge } = setup("staged_one_shot");
  const result = await run(f, bridge, ["broken", f.plan()]);
  assert.equal(result.report.resources.model_request_units, 1); assert.equal(result.report.task_status, "partial");
});
test("fast path stays off by default for existing planned tasks", async () => {
  const { f, bridge } = setup("planned_adaptive");
  const result = await run(f, bridge, [f.plan(), c => assess(f, c)]);
  assert.equal(result.events.some((e: any) => e.type === "fast.selected"), false);
});

test("typed syntax repair shares the only correction with native argument recovery", async () => {
  const { f, bridge, calls } = setup(), params = { queries: [], top_k: 3, filters: {} };
  const result = await run(f, bridge, ["{", f.plan(), c => assess(f, c, "partial_match"),
    [{ type: "toolCall", id: "bad", name: "search_cases", arguments: params }],
    [{ type: "toolCall", id: "should-not-run", name: "search_cases", arguments: params }]]);
  assert.equal(calls.filter(c => c.name === "search_cases").length, 1);
  assert.equal(result.report.resources.model_request_units, 4);
  assert.equal(result.report.task_status, "partial");
});
test("preview mismatch cannot publish a bad source as usable evidence", async () => {
  const { f, bridge } = setup(), original = bridge.previewRead;
  bridge.previewRead = async (args: JsonObject) => { const r = await original(args); r.data.evidence.quote = "被".repeat(Array.from(r.data.evidence.quote).length); return r; };
  const result = await run(f, bridge, [f.plan(), c => assess(f, c)]);
  assert.equal(result.report.task_status, "partial");
  assert.ok(!result.report.retrieved_evidence.some((e: any) => e.was_read));
  assert.ok(result.events.some((e: any) => e.type === "tool.failure" && String(e.data.reason).includes("PREVIEW")));
});
test("fixed queries are fully frozen before the first search observation", async () => {
  const { f, bridge, calls } = setup("planned_fixed", { engine: "fixed_rag" }), plan = f.plan(2);
  plan.points[0].alternative_query = "备选一"; plan.points[1].alternative_query = "备选二"; plan.points[1].query_text = "主二";
  const expected = [plan.points[0].query_text, "主二", "备选一", "备选二"];
  const original = bridge.call; bridge.call = async (name: string, args: JsonObject) => {
    if (name === "search_cases") { plan.points[1].query_text = "观察后变造"; plan.points[0].alternative_query = "新查询"; }
    return original(name, args);
  };
  const result = await run(f, bridge, [plan, c => assess(f, c), c => assess(f, c)]);
  assert.deepEqual(calls.filter(c => c.name === "search_cases").map(c => c.args.queries[0].text), expected);
  assert.equal(result.report.resources.model_request_units, 3);
});

test("first native schema error is corrected once under the same permission and retired after actual acceptance", async () => {
  const { f, bridge, calls } = setup(), params = { queries: [{ text: "新 赠礼", focus_refs: [f.refs[0]], risk_point_ids: ["RP0001"], goal: "mechanism_search" }], top_k: 3, filters: {} };
  const result = await run(f, bridge, [f.plan(), c => assess(f, c, "partial_match"),
    [{ type: "toolCall", id: "bad", name: "search_cases", arguments: { queries: [], top_k: 3, filters: {} } }],
    [{ type: "toolCall", id: "corrected", name: "search_cases", arguments: params }], c => assess(f, c)]);
  assert.equal(result.report.resources.model_request_units, 5);
  assert.equal(calls.filter(c => c.name === "search_cases").length, 2);
  assert.ok(result.events.some((e: any) => e.type === "loop.native_format_recovered"));
  assert.equal(result.report.task_status, "completed");
});

test("initial duplicate normalized queries merge all owners before applying adaptive limit", async () => {
  const { f, bridge, calls } = setup(), plan = f.plan(2);
  plan.points[0].query_text = "  共同 查询 "; plan.points[1].query_text = "共同   查询"; plan.points[0].alternative_query = "另一 查询";
  const result = await run(f, bridge, [plan, c => assess(f, c), c => assess(f, c)]);
  const searches = calls.filter(c => c.name === "search_cases");
  assert.equal(searches.length, 2);
  assert.deepEqual(searches[0]!.args.queries[0].risk_point_ids, ["RP0001", "RP0002"]);
  assert.equal(searches[0]!.args.queries[0].focus_refs.length, 2);
  assert.equal(searches[1]!.args.queries[0].text, "另一 查询");
  assert.equal(result.report.risk_points.length, 2);
});
test("fixed identical primary and alternative query is not dispatched twice", async () => {
  const { f, bridge, calls } = setup("planned_fixed", { engine: "fixed_rag" }), plan = f.plan();
  plan.points[0].alternative_query = "  " + plan.points[0].query_text + " ";
  await run(f, bridge, [plan, c => assess(f, c)]);
  assert.equal(calls.filter(c => c.name === "search_cases").length, 1);
});

test("all planned arms freeze top_k three and adaptive cannot increase candidate count", async () => {
  for (const mode of ["planned_fixed", "planned_adaptive"]) {
    const { f, bridge, calls } = setup(mode, mode === "planned_fixed" ? { engine: "fixed_rag" } : {});
    const bad = { queries: [{ text: "更多 赠礼", focus_refs: [f.refs[0]], risk_point_ids: ["RP0001"], goal: "mechanism_search" }], top_k: 10, filters: {} };
    await run(f, bridge, [f.plan(), c => assess(f, c, "partial_match"),
      [{ type: "toolCall", id: "top10", name: "search_cases", arguments: bad }], { kind: "finalize" }]);
    assert.ok(calls.filter(c => c.name === "search_cases").every(c => c.args.top_k === 3 && canonical(c.args.filters) === "{}"));
    assert.equal(calls.filter(c => c.name === "search_cases").length, 1);
  }
});

test("two feedback rounds stop before a third native search even while comparison remains partial", async () => {
  const { f, bridge, calls } = setup();
  const native = (text: string): ScriptStep => [{ type: "toolCall", id: text, name: "search_cases", arguments: { queries: [{ text, focus_refs: [f.refs[0]], risk_point_ids: ["RP0001"], goal: "mechanism_search" }], top_k: 3, filters: {} } }];
  const result = await run(f, bridge, [f.plan(), c => assess(f, c, "partial_match"), native("第一次 赠礼"), c => assess(f, c, "partial_match"), native("第二次 赠礼"), c => assess(f, c, "partial_match"), native("第三次不可执行")]);
  assert.equal(calls.filter(c => c.name === "search_cases").length, 3);
  assert.equal(result.events.filter((e: any) => e.type === "loop.feedback_completed").length, 2);
  assert.equal(result.report.resources.model_request_units, 6);
});
test("search calls query attempts external800 tools and model limits independently stop new dispatch", async () => {
  for (const [field, limit] of [["max_search_calls", 1], ["max_search_query_attempts", 1], ["max_tool_actions", 1], ["max_model_requests", 2]] as const) {
    const { f, bridge, calls } = setup("planned_adaptive", { [field]: limit, ...(field === "max_search_query_attempts" ? { initial_query_limit: 1, feedback_query_reserve: 0 } : {}) });
    const p = f.plan(field === "max_model_requests" || field === "max_tool_actions" ? 1 : 2); p.points[1] && (p.points[1].query_text += "第二点");
    const result = await run(f, bridge, [p, c => assess(f, c, "partial_match"), { kind: "finalize" }]);
    assert.ok(result.report.resources[field === "max_search_calls" ? "search_calls" : field === "max_search_query_attempts" ? "search_query_attempts" : field === "max_tool_actions" ? "tool_actions" : "model_request_units"] <= limit, field);
    assert.equal(calls.filter(c => c.name === "search_cases").length, 1, field);
  }
  const { f, bridge, calls } = setup(), original = bridge.call, p = f.plan(2); p.points[1].query_text += "二";
  bridge.call = async (name: string, args: JsonObject) => { const r = await original(name, args); if (name === "search_cases") r.meta.external_retrieval_requests = 800; return r; };
  const result = await run(f, bridge, [p]);
  assert.equal(calls.filter(c => c.name === "search_cases").length, 1); assert.equal(result.report.resources.external_retrieval_requests, 800);
});
test("one second deadline cancels and late model response cannot start tools", async () => {
  const { f, bridge, calls } = setup("planned_adaptive", { task_deadline_seconds: 1, model_request_timeout_seconds: 1, tool_request_timeout_seconds: 1 });
  let release!: (value: unknown) => void, cancelTimeout = -1;
  bridge.cancelRetrieval = async (ms: number) => { cancelTimeout = ms; return { external_retrieval_requests: 0, usage_known: false, cancellation_confirmed: false }; };
  const result = await run(f, bridge, [() => new Promise(resolve => { release = resolve; })]);
  release(f.plan()); await new Promise(resolve => setTimeout(resolve, 5));
  assert.equal(calls.length, 0); assert.equal(result.report.stop.reason, "deadline_exceeded");
  assert.ok(cancelTimeout <= 1); assert.equal(result.report.resources.unknown_usage_requests, 1);
});
test("local99 query is excluded under a new native call id but independent query remains permitted", async () => {
  const { f, bridge, calls } = setup(), original = bridge.call;
  bridge.call = async (name: string, args: JsonObject) => {
    const r = await original(name, args);
    if (name === "search_cases") {
      const q = r.data.queries[0]; q.complete = false; q.status = "partial"; q.stage_status.rerank = "partial";
      Object.assign(q.ranking_coverage, { expected_units: 100, scored_units: 99, failed_units: 1, ranking_complete: false, topk_scope: "scored_subset", attempts_exhausted: true,
        failure_counts: { QUOTE_MISMATCH: 1 }, failed_unit_ids: ["FAILED"], expected_set_sha256: "c".repeat(64), scored_set_sha256: "a".repeat(64) });
      r.status = "partial";
    }
    return r;
  };
  const params = { queries: [{ text: f.plan().points[0].query_text, focus_refs: [f.refs[0]], risk_point_ids: ["RP0001"], goal: "mechanism_search" }], top_k: 3, filters: {} };
  const result = await run(f, bridge, [f.plan(), c => assess(f, c, "partial_match"), [{ type: "toolCall", id: "new-id", name: "search_cases", arguments: params }]]);
  assert.equal(calls.filter(c => c.name === "search_cases").length, 1); assert.equal(result.report.task_status, "partial");
});
test("native request cannot escape its licensed point", async () => {
  const { f, bridge, calls } = setup(), params = { queries: [{ text: "不许可 赠礼", focus_refs: [f.refs[0]], risk_point_ids: ["RP9999"], goal: "mechanism_search" }], top_k: 3, filters: {} };
  const result = await run(f, bridge, [f.plan(), c => assess(f, c, "partial_match"), [{ type: "toolCall", id: "escape", name: "search_cases", arguments: params }]]);
  assert.equal(calls.filter(c => c.name === "search_cases").length, 1); assert.equal(result.report.task_status, "partial");
});
test("global stopped or circuit response prevents all reads and further models", async () => {
  for (const circuit of [false, true]) {
    const { f, bridge, calls } = setup(), original = bridge.call;
    bridge.call = async (name: string, args: JsonObject) => {
      const r = await original(name, args);
      if (name === "search_cases") {
        if (circuit) r.meta.transport_circuit_open = true;
        else Object.assign(r.data.queries[0], { complete: false, status: "error", error: { code: "STOPPED" } });
      }
      return r;
    };
    const result = await run(f, bridge, [f.plan(), c => assess(f, c)]);
    assert.equal(calls.filter(c => c.name === "read_source").length, 0); assert.equal(result.report.resources.model_request_units, 1);
    assert.equal(result.report.task_status, "partial");
  }
});

test("phase mutation during a request cannot accept its plan or dispatch any tool", async () => {
  const { f, bridge, calls } = setup();
  const result = await run(f, bridge, [(_context, state) => { state.phase = "reviewing"; return f.plan(); }, c => assess(f, c)]);
  assert.equal(calls.length, 0); assert.equal(result.report.task_status, "partial");
  assert.ok(result.events.some((e: any) => e.type === "loop.stopped" && e.data.code === "LOOP_PAYLOAD_BINDING_MISMATCH"));
});

test("report bytes bind the persisted assessment sidecar and retain raw assistant messages", async () => {
  const { f, bridge } = setup(), dir = mkdtempSync(join(root, ".test-temp-loop-"));
  f.task.task_dir = dir;
  const result = await run(f, bridge, [f.plan(), c => assess(f, c)]);
  const reportBytes = readFileSync(join(dir, "report.json"), "utf8"), sidecar = JSON.parse(readFileSync(join(dir, "agent-assessment.json"), "utf8"));
  assert.equal(sidecar.identity.report_sha256, hash(reportBytes)); assert.equal(sidecar.points[0].processing_state, "assessed");
  assert.ok(readFileSync(join(dir, "model_messages.jsonl"), "utf8").includes('"assistant"'));
  assert.equal(JSON.parse(reportBytes).task_status, result.report.task_status);
  const receipts = result.events.filter((e: any) => e.type === "report.persisted");
  assert.equal(receipts.length, 1, "only successful report persistence can emit a receipt");
  assert.equal(receipts[0].data.report_sha256, hash(reportBytes));
  assert.equal(receipts[0].data.task_id, f.task.input.task_id);
  assert.equal(receipts[0].data.input_sha256, f.task.input.input_sha256);
  assert.equal(receipts[0].data.profile_sha256, sidecar.identity.profile_sha256);
  assert.equal(receipts[0].data.schema_version, "finrisk-loop-report-persistence/1.0");
  assert.ok(Number.isSafeInteger(receipts[0].data.persisted_at_epoch_ms));
  assert.ok(receipts[0].data.persisted_at_epoch_ms <= Date.now());
});
test("sidecar storage failure returns and persists partial without claiming completion", async () => {
  const { f, bridge } = setup(), dir = mkdtempSync(join(root, ".test-temp-loop-"));
  f.task.task_dir = dir; mkdirSync(join(dir, "agent-assessment.json"));
  const result = await run(f, bridge, [f.plan(), c => assess(f, c)]);
  assert.equal(result.report.task_status, "partial"); assert.equal(result.report.stop.reason, "diagnostic_write_failed");
  assert.equal(JSON.parse(readFileSync(join(dir, "report.json"), "utf8")).task_status, "partial");
  const receipts = result.events.filter((e: any) => e.type === "report.persisted");
  assert.equal(receipts.length, 2, "sidecar failure rewrites need a new final receipt");
  assert.notEqual(receipts[0].data.report_sha256, receipts[1].data.report_sha256);
  assert.equal(receipts[1].data.report_sha256, hash(readFileSync(join(dir, "report.json"), "utf8")));
  assert.ok(receipts[1].sequence > result.events.find((e: any) => e.type === "loop.sidecar_failed").sequence);
});
test("definition response must match the real dictionary before candidates can expand", async () => {
  const { f, bridge } = setup(), original = bridge.call;
  bridge.call = async (name: string, args: JsonObject) => {
    const r = await original(name, args); if (name === "get_risk_definition") r.data.definitions[0].description = "伪造定义"; return r;
  };
  const native: ScriptStep = [{ type: "toolCall", id: "definition", name: "get_risk_definition", arguments: { tags: ["合同外利益"] } }];
  const result = await run(f, bridge, [f.plan(), c => assess(f, c, "partial_match"), native, c => assess(f, c), { kind: "finalize" }]);
  assert.equal(result.report.task_status, "partial");
  assert.ok(result.events.some((e: any) => e.type === "loop.stopped" && e.data.code === "LOOP_DEFINITION_REQUIRED"));
});

test("read_source retries count toward the same per-tool action cap", async () => {
  const { f, bridge, calls } = setup("planned_adaptive", { max_read_source_actions: 1 }), original = bridge.call;
  let reads = 0;
  bridge.call = async (name: string, args: JsonObject) => {
    if (name === "read_source") { reads++; throw new Error("TOOL_TRANSPORT_HTTP_503"); }
    return original(name, args);
  };
  const result = await run(f, bridge, [f.plan()]);
  assert.equal(reads, 1); assert.equal(result.report.task_status, "partial");
  assert.equal(calls.filter(c => c.name === "search_cases").length, 1);
});

test("hard failure latches before later tools in the same native reply while recoverable schema stays separate", async () => {
  for (const failure of ["definition_identity", "transport", "permission", "definition_budget"]) {
    const { f, bridge, calls } = setup("planned_adaptive", { max_tool_transport_retries: 0, ...(failure === "definition_budget" ? { max_definition_actions: 1 } : {}) });
    const original = bridge.call;
    bridge.call = async (name: string, args: JsonObject) => {
      if (failure === "transport" && name === "get_risk_definition") throw new Error("TOOL_TRANSPORT_HTTP_503");
      const response = await original(name, args);
      if (failure === "definition_identity" && name === "get_risk_definition") response.data.definitions[0].description = "伪造定义";
      return response;
    };
    const search = (id: string, point = "RP0001") => ({ type: "toolCall", id, name: "search_cases", arguments: {
      queries: [{ text: "后续 新赠礼 查询", focus_refs: [f.refs[0]], risk_point_ids: [point], goal: "mechanism_search" }], top_k: 3, filters: {},
    } });
    const definition = (id: string, tag = "合同外利益") => ({ type: "toolCall", id, name: "get_risk_definition", arguments: { tags: [tag] } });
    const first = failure === "permission" ? [search("not-allowed", "RP9999")]
      : failure === "definition_budget" ? [definition("within-budget", "赠送利益"), definition("over-budget")]
      : [definition("hard-failure")];
    const result = await run(f, bridge, [f.plan(), c => assess(f, c, "partial_match"), [...first, search("after-hard-failure")]]);
    assert.equal(calls.filter(c => c.name === "search_cases").length, 1, failure + ": later search must have zero dispatch");
    assert.equal(result.report.task_status, "partial", failure);
    assert.ok(result.events.some((e: any) => e.type === "loop.stopped"), failure + ": retain the original hard blocker");
    assert.ok(!result.events.some((e: any) => e.type === "loop.action" && e.data.query_text === "后续 新赠礼 查询"), failure + ": no admitted later action");
  }
});
