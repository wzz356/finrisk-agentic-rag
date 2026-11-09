import test, { after } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { canonical, hash, root, type JsonObject, type TaskInput } from "../src/contracts.js";
import { evaluationConfig } from "../src/evaluation.js";
import { runTask } from "../src/runner.js";
import { Trace } from "../src/trace.js";
import { buildProviderPayload } from "../src/prompt_counter.js";
import type { TaskState } from "../src/state.js";
import { offlineCounter } from "./offline_counter.js";

const local = offlineCounter(); after(() => local.close());
process.env.FINRISK_AGENT_ALLOW_EVALUATION = "1";
const text = "购买这份保险就送购物卡。";
const ref = { segment_id: "S0001", quote: text, occurrence: 1 };
const segments = [{ segment_id: "S0001", status: "reviewed", reason: "核查投保赠礼条件。" }];
function task(config: JsonObject = {}): TaskInput {
  const definitions = JSON.parse(readFileSync(root + "/contracts/agent_rag/risk-definitions.json", "utf8"));
  const user_context = { purpose_hint: "面向客户的保险营销宣传" };
  return { input: { task_id: "stage-recovery", source_id: "INPUT:stage-recovery", text, text_version: "1", text_sha256: hash(text),
    input_sha256: hash(canonical({ text, user_context })), user_context,
    segments: [{ segment_id: "S0001", source_id: "INPUT:stage-recovery", start: 0, end: Array.from(text).length, text }] },
    definitions, snapshot: { snapshot_id: "SNAPSHOT", dictionary_sha256: definitions[0].dictionary_sha256 },
    config: { ...evaluationConfig("R2"), execution_mode: "mock", retrieval_profile: "v2-hybrid-flash-v1",
      max_queries_per_search: 1, retrieval_reliability: true, ...config } as any };
}
function analysis(): JsonObject {
  return { kind: "analysis", segments, points: [{ input_refs: [ref], candidate_tags: ["赠送利益"],
    facts: [{ field: "insurance_product", value: "yes", origin: "input", input_refs: [ref], context_refs: [] },
      { field: "material_function", value: "direct_promotion", origin: "user_context", input_refs: [], context_refs: ["$.user_context.purpose_hint"] }] }] };
}
function final(): JsonObject {
  return { kind: "final", segments, points: [{ risk_point_id: "RP0001", case_matches: [],
    tag_checks: [{ tag: "赠送利益", conditions: ["actual_conduct", "insurance_product", "gift", "purchase_link"].map(criterion_id =>
      ({ criterion_id, status: "met", input_refs: [ref], context_refs: [], reason: "当前材料明确支持。" })) }] }] };
}
function bridge() {
  const calls: JsonObject[] = [];
  return { calls, countTokens: local.count, async call(name: string, args: JsonObject): Promise<any> {
    calls.push({ name, args });
    return { schema_version: "finrisk-tools/1.0", status: "empty",
      data: { queries: (args.queries ?? []).map((q: JsonObject, index: number) => ({ ...q, query_id: "Q" + (calls.length + index), status: "empty", complete: true, candidates: [],
        stage_status: { bm25: "completed", bge: "completed", rrf: "completed", rerank: "completed" },
        ranking_coverage: { schema_version: "finrisk-ranking-coverage/1.0", expected_units: 1, scored_units: 1, failed_units: 0,
          ranking_complete: true, topk_scope: "full_pool", attempts_exhausted: false, failure_counts: {}, failed_unit_ids: [],
          expected_set_sha256: "a".repeat(64), scored_set_sha256: "a".repeat(64), failed_set_sha256: "b".repeat(64), rerank_protocol: "span-ref-v2" } })) },
      meta: { snapshot_id: "SNAPSHOT", backend_query_attempts: (args.queries ?? []).length, external_retrieval_requests: 0, cache_hits: 0 }, error: null };
  }, async verifyEvidence() { return { valid_ids: [], rejected_ids: [] }; } };
}
function diagnostic(result: Awaited<ReturnType<typeof runTask>>) {
  const event = result.events.find(e => e.type === "retrieval.assessment"); assert.ok(event);
  return event.data;
}
function blocked(result: Awaited<ReturnType<typeof runTask>>): void {
  assert.equal(result.report.task_status, "partial", JSON.stringify(result.report.validation));
  assert.equal(diagnostic(result).points[0].assessment_status, "blocked");
}
const malformed = [["syntax", "{broken", "MODEL_JSON_SYNTAX"], ["root", "[]", "MODEL_JSON_ROOT_NOT_OBJECT"]] as const;
for (const phase of ["analyzing", "finalizing"] as const) for (const [category, bad, code] of malformed) {
  test("R2 retires only a recovered typed " + category + " error in " + phase, async () => {
    const b = bridge(); let acceptedState: TaskState | undefined;
    const corrected = (_context: any, state: TaskState) => { acceptedState = state; return phase === "analyzing" ? analysis() : final(); };
    const script = phase === "analyzing" ? [bad, corrected, final()] : [analysis(), bad, corrected];
    const result = await runTask(task(), b, script);
    assert.equal(result.report.task_status, "completed", JSON.stringify(result.report.validation));
    assert.deepEqual(result.report.validation.errors, []);
    assert.equal(diagnostic(result).points[0].assessment_status, "model_assessed");
    assert.equal(result.report.resources.model_request_units, 3);
    assert.equal(result.report.resources.json_repairs, 1); assert.equal(result.report.resources.tool_format_corrections, 0);
    assert.equal(result.report.resources.search_calls, 1); assert.equal(result.report.resources.tool_actions, 1);
    assert.equal(acceptedState!.repairing, false);
    const rejected = result.events.filter(e => e.type === "output.rejected"), recovered = result.events.filter(e => e.type === "output.format_recovered");
    assert.equal(rejected.length, 1); assert.equal(rejected[0]!.data.code, code);
    assert.equal(recovered.length, 1); assert.equal(recovered[0]!.data.code, code);
    assert.equal(recovered[0]!.data.scope, "stage_format"); assert.equal(recovered[0]!.data.phase, phase);
    assert.equal(recovered[0]!.data.rejected_reply_sha256, hash(bad));
    assert.equal(recovered[0]!.data.accepted_reply_sha256, hash(JSON.stringify(phase === "analyzing" ? analysis() : final())));
    assert.ok(recovered[0]!.sequence > rejected[0]!.sequence);
  });
}

test("stage recovery preserves the original rejected reply and durable audit", async () => {
  const directory = mkdtempSync(join(tmpdir(), "finrisk-stage-recovery-"));
  try {
    const t = task(); t.task_dir = directory;
    const result = await runTask(t, bridge(), [analysis(), "{broken", final()]);
    assert.equal(result.report.task_status, "completed");
    assert.ok(readFileSync(join(directory, "model_messages.jsonl"), "utf8").includes("{broken"));
    const events = readFileSync(join(directory, "events.jsonl"), "utf8").trim().split("\n").map(row => JSON.parse(row));
    assert.equal(events.filter(e => e.type === "output.rejected").length, 1);
    assert.equal(events.filter(e => e.type === "output.format_recovered").length, 1);
    assert.deepEqual(JSON.parse(readFileSync(join(directory, "validation.json"), "utf8")).errors, []);
    assert.equal(JSON.parse(readFileSync(join(directory, "retrieval-assessment.json"), "utf8")).points[0].assessment_status, "model_assessed");
  } finally { rmSync(directory, { recursive: true, force: true }); }
});

test("successful correction retains unrelated blockers and identical independent errors", async () => {
  const result = await runTask(task(), bridge(), [analysis(),
    (_context, state) => { state.validationErrors.push("MODEL_JSON_SYNTAX", "SOURCE_IDENTITY_FAILURE"); return "{broken"; },
    (_context, state) => { state.validationErrors.push("MODEL_JSON_SYNTAX"); return final(); }]);
  blocked(result);
  assert.equal(result.report.validation.errors.filter((e: string) => e === "MODEL_JSON_SYNTAX").length, 2);
  assert.ok(result.report.validation.errors.includes("SOURCE_IDENTITY_FAILURE"));
  assert.equal(result.events.filter(e => e.type === "output.format_recovered").length, 1);
});

test("stage recovery refuses to retire an error whose original index identity changed", async () => {
  const result = await runTask(task(), bridge(), [analysis(), "{broken",
    (_context, state) => { state.validationErrors.unshift("INDEPENDENT_FAILURE"); return final(); }]);
  blocked(result);
  assert.ok(result.report.validation.errors.includes("MODEL_JSON_SYNTAX"));
  assert.ok(result.report.validation.errors.includes("FORMAT_RECOVERY_ERROR_IDENTITY_MISMATCH"));
  assert.equal(result.events.some(e => e.type === "output.format_recovered"), false);
  assert.equal(result.report.resources.model_request_units, 3); assert.equal(result.report.resources.json_repairs, 1);
});

for (const phase of ["analyzing", "finalizing"] as const) {
  test("failed " + phase + " recovery audit preserves blockers without another repair", async () => {
    const original = Trace.prototype.event;
    Trace.prototype.event = function(type, data) {
      if (type === "output.format_recovered") throw new Error("FORMAT_RECOVERY_AUDIT_UNAVAILABLE");
      return original.call(this, type, data);
    };
    try {
      const script = phase === "analyzing" ? ["{broken", analysis()] : [analysis(), "{broken", final()];
      const result = await runTask(task(), bridge(), script);
      assert.equal(result.report.task_status, "partial");
      assert.ok(result.report.validation.errors.includes("MODEL_JSON_SYNTAX"));
      assert.ok(result.report.validation.errors.includes("FORMAT_RECOVERY_AUDIT_UNAVAILABLE"));
      assert.equal(result.events.some(e => e.type === "output.format_recovered"), false);
      assert.equal(result.report.resources.model_request_units, phase === "analyzing" ? 2 : 3);
      assert.equal(result.report.resources.json_repairs, 1);
    } finally { Trace.prototype.event = original; }
  });
}

for (const correction of ["syntax", "reference", "semantic"] as const) {
  test("unsuccessful stage correction retains syntax blocker: " + correction, async () => {
    const invalid = final();
    if (correction === "reference") invalid.points[0].tag_checks[0].conditions[0].input_refs = [{ ...ref, quote: "不存在的输入原话" }];
    if (correction === "semantic") invalid.points[0].tag_checks[0].conditions[0].input_refs = [];
    const result = await runTask(task(), bridge(), [analysis(), "{broken", correction === "syntax" ? "{broken-again" : invalid, final()]);
    assert.equal(result.report.task_status, "partial");
    assert.ok(result.report.validation.errors.includes("MODEL_JSON_SYNTAX"));
    assert.equal(result.events.some(e => e.type === "output.format_recovered"), false);
    assert.equal(result.report.resources.model_request_units, 3); assert.equal(result.report.resources.json_repairs, 1);
  });
}

for (const failure of ["wire", "semantic", "tool"] as const) {
  test("non-typed " + failure + " correction keeps the error and report diagnostic blocked", async () => {
    const invalid = final();
    if (failure === "wire") invalid.extra = true;
    if (failure === "semantic") invalid.points[0].tag_checks[0].conditions[0].input_refs = [];
    const bad = failure === "tool" ? [{ type: "toolCall", id: "forbidden", name: "search_cases", arguments: { queries: [{ text: "投保赠礼机制", focus_refs: [ref], risk_point_ids: ["RP0001"], goal: "mechanism_search" }], top_k: 3, filters: {} } }] : invalid;
    const result = await runTask(task(), bridge(), [analysis(), bad, final()]);
    blocked(result);
    assert.ok(result.report.validation.errors.some((e: string) => e.startsWith(failure === "wire" ? "INVALID_WIRE" : failure === "semantic" ? "CONDITION_WITHOUT_CURRENT_SUPPORT" : "TOOLS_FORBIDDEN_IN_THIS_PHASE")));
    assert.equal(result.events.some(e => e.type === "output.format_recovered"), false);
    assert.equal(result.report.resources.json_repairs, 1);
  });
}

for (const config of [{ max_model_requests: 2 }, { max_json_repairs_per_task: 0 }]) {
  test("non-first format recovery cannot bypass existing limits " + JSON.stringify(config), async () => {
    const result = await runTask(task(config), bridge(), [analysis(), "{broken", final()]);
    assert.equal(result.report.task_status, "partial");
    assert.equal(result.report.resources.model_request_units, 2); assert.equal(result.report.resources.json_repairs, 0);
    assert.equal(result.events.some(e => e.type === "output.format_recovered"), false);
  });
}

test("corrected controlled analysis releases JSON mode before legitimate reviewing tools", async () => {
  const b = bridge(); let reviewed = false;
  const result = await runTask(task({ engine: "controlled_agent", evaluation_arm: "A" }), b, ["{broken", analysis(), (context, state) => {
    reviewed = true; assert.equal(state.phase, "reviewing"); assert.equal(state.repairing, false);
    const payload = buildProviderPayload(context, state.phase, state.repairing);
    assert.equal(payload.response_format, undefined); assert.ok(payload.tools.length > 0);
    return [{ type: "toolCall", id: "recovered-search", name: "search_cases", arguments: {
      queries: [{ text: "投保赠礼机制", focus_refs: [ref], risk_point_ids: ["RP0001"], goal: "mechanism_search" }], top_k: 3, filters: {} } }];
  }, { kind: "finalize" }, final()]);
  assert.equal(reviewed, true); assert.equal(result.report.task_status, "completed", JSON.stringify(result.report.validation));
  assert.equal(result.report.resources.model_request_units, 5); assert.equal(result.report.resources.json_repairs, 1);
  assert.equal(b.calls.filter(c => c.name === "search_cases").length, 1);
});

test("A continuation final uses stage recovery without changing first-turn budgets", async () => {
  const a = analysis().points[0], f = final().points[0];
  const first = { kind: "evaluation_first", segments, points: [{ input_refs: a.input_refs, facts: a.facts, tag_checks: f.tag_checks, case_matches: [] }],
    continuation: { gaps: [{ type: "case_comparison", risk_point_id: "RP0001", reason: "追加案例机制对照。" }] } };
  const result = await runTask(task({ ...evaluationConfig("A") }), bridge(), [first, { kind: "finalize" }, "[]", final()]);
  assert.equal(result.report.task_status, "completed", JSON.stringify(result.report.validation));
  assert.deepEqual(result.report.validation.errors, []); assert.equal(diagnostic(result).points[0].assessment_status, "model_assessed");
  assert.equal(result.report.resources.model_request_units, 4); assert.equal(result.report.resources.json_repairs, 1);
  assert.equal(result.events.find(e => e.type === "output.format_recovered")!.data.phase, "finalizing");
});

test("stage format recovery cannot strengthen an unknown current contract", async () => {
  const a = analysis(), f = final();
  a.points[0].candidate_tags.push("合同外利益");
  f.points[0].tag_checks.push({ tag: "合同外利益", conditions: ["actual_conduct", "insurance_product", "benefit", "contract_outside"].map(criterion_id =>
    ({ criterion_id, status: "met", input_refs: [ref], context_refs: [], reason: "历史案例不能证明当前合同。" })) });
  const result = await runTask(task(), bridge(), [a, "{broken", f]);
  assert.equal(result.report.task_status, "completed", JSON.stringify(result.report.validation));
  assert.deepEqual(result.report.risk_points[0].pending_tags, ["合同外利益"]);
  assert.equal(result.report.stop.reason, "input_facts_missing");
  assert.equal(diagnostic(result).points[0].assessment_status, "needs_input");
  assert.deepEqual(result.report.validation.errors, []);
});

test("a later semantic correction cannot retire the earlier failed syntax attempt", async () => {
  const invalid = final(); invalid.points[0].tag_checks[0].conditions[0].input_refs = [];
  const result = await runTask(task({ max_json_repairs_per_task: 2 }), bridge(), [analysis(), "{broken", invalid, final()]);
  blocked(result);
  assert.ok(result.report.validation.errors.includes("MODEL_JSON_SYNTAX"));
  assert.ok(result.report.validation.errors.includes("CONDITION_WITHOUT_CURRENT_SUPPORT"));
  assert.equal(result.events.some(e => e.type === "output.format_recovered"), false);
  assert.equal(result.report.resources.model_request_units, 4); assert.equal(result.report.resources.json_repairs, 2);
});

test("success after two typed errors retires only the latest admitted attempt", async () => {
  const result = await runTask(task({ max_json_repairs_per_task: 2 }), bridge(), [analysis(), "{broken-first", "{broken-second", final()]);
  blocked(result);
  assert.equal(result.report.validation.errors.filter((e: string) => e === "MODEL_JSON_SYNTAX").length, 1);
  const recovered = result.events.filter(e => e.type === "output.format_recovered");
  assert.equal(recovered.length, 1);
  assert.equal(recovered[0]!.data.attempt, 2);
  assert.equal(recovered[0]!.data.rejected_reply_sha256, hash("{broken-second"));
  assert.equal(result.report.resources.model_request_units, 4); assert.equal(result.report.resources.json_repairs, 2);
});

test("partial ranking summary cannot hide an unresolved output blocker", async () => {
  const b = bridge(), original = b.call.bind(b);
  b.call = async (name, args) => {
    const response = await original(name, args);
    if (name === "search_cases") {
      response.status = "partial";
      for (const q of response.data.queries) {
        q.status = "partial"; q.complete = false; q.stage_status.rerank = "partial";
        Object.assign(q.ranking_coverage, { expected_units: 100, scored_units: 98, failed_units: 2,
          ranking_complete: false, topk_scope: "scored_subset", attempts_exhausted: true,
          failure_counts: { INVALID_COMPLETION: 2 }, failed_unit_ids: ["U99", "U100"],
          scored_set_sha256: "b".repeat(64), failed_set_sha256: "c".repeat(64) });
      }
    }
    return response;
  };
  const invalid = final(); invalid.extra = true;
  const result = await runTask(task(), b, [analysis(), invalid, final()]);
  blocked(result);
  assert.ok(result.report.validation.errors.some((e: string) => e.startsWith("INVALID_WIRE")));
  assert.match(result.report.summary, /仍有未完成或矛盾事项需要复核/);
  assert.doesNotMatch(result.report.summary, /^本轮候选评分/);
});
