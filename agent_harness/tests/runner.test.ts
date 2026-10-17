import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { runTask } from "../src/runner.js";
import { canonical, hash, root, type JsonObject, type TaskInput } from "../src/contracts.js";
import { getCurrentTools } from "@earendil-works/pi-ai";

const text = "购买这份保险就送购物卡。";
export const ref = { segment_id: "S0001", quote: text, occurrence: 1 };
export function task(extra: any = {}): TaskInput {
  const definitions = JSON.parse(readFileSync(`${root}/contracts/agent_rag/risk-definitions.json`, "utf8"));
  return { input: { task_id: "unit-task", text, source_id: "INPUT:unit-task", text_sha256: hash(text), text_version: "registered-input/1.0", input_sha256: hash(canonical({ text, user_context: {} })), user_context: {}, segments: [{ segment_id: "S0001", start: 0, end: Array.from(text).length, text, source_id: "INPUT:unit-task" }] }, snapshot: { snapshot_id: "unit-snapshot", dictionary_sha256: definitions[0].dictionary_sha256 }, definitions, ...extra };
}
export function analysis(): JsonObject {
  return { kind: "analysis", document_context: { main_scene: null, scene_status: "unknown", material_functions: ["unknown"] }, segment_assessments: [{ segment_id: "S0001", status: "reviewed", reason: "测试理解" }], risk_points: [], unprocessed_segment_ids: [] };
}
export function final(): JsonObject {
  return JSON.parse(readFileSync(`${root}/contracts/agent_rag/fixtures/valid_final.json`, "utf8"));
}
export const search = { queries: [{ text: "购买保险赠送购物卡", focus_refs: [ref], risk_point_ids: [], goal: "mechanism_search" }], top_k: 5, filters: {} };
export function empty(): any { return { schema_version: "finrisk-tools/1.0", status: "empty", data: { queries: [{ ...search.queries[0], query_id: "Q01", status: "empty", complete: true, candidates: [] }] }, meta: { snapshot_id: "unit-snapshot", backend_query_attempts: 1, cache_hits: 0, external_retrieval_requests: 0 }, error: null }; }
export function bridge(response = empty()): any {
  return { async call() { return response; }, async verifyEvidence(ids: string[]) { return { valid_ids: ids, rejected_ids: [] }; } };
}
function call(name: string, args: any): any { return { type: "toolCall", id: "tool-1", name, arguments: args }; }

test("fixed RAG final context receives full source quote without marking semantic acceptance", async () => {
  const quote = "处罚原文：赠送利益的事实及完整适用条件。";
  const evidence = { evidence_id: "fixed-evidence", source_id: "fixed-source", quote, start: 0, end: Array.from(quote).length, source_kind: "penalty", source_verified: true, text_version: "unit", text_sha256: hash(quote), context_has_more: false };
  const response = empty(); response.status = "ok"; response.data.queries[0].status = "ok";
  response.data.queries[0].candidates = [{ source_id: evidence.source_id, evidence: [evidence] }];
  const sourceResponse = { ...response, data: { evidence }, meta: { ...response.meta, backend_query_attempts: 0 } };
  let sawQuote = false;
  const sourceBridge = { async call(name: string) { return name === "read_source" ? sourceResponse : response; }, async verifyEvidence(ids: string[]) { return { valid_ids: ids, rejected_ids: [] }; } };
  const result = await runTask(task({ config: { engine: "fixed_rag" } }), sourceBridge, [analysis(), (context) => { sawQuote = JSON.stringify(context.messages).includes(quote); return final(); }]);
  assert.equal(sawQuote, true);
  assert.equal(result.report.task_status, "completed");
  assert.ok(!result.report.retrieved_evidence[0].accepted_as_relevant);
});

test("H01/H16/H19: real Pi loop uses analysis, tool and final, empty is completed, mock zero physical", async () => {
  const response = await runTask(task(), bridge(), [analysis(), [call("search_cases", search)], final()]);
  assert.equal(response.report.task_status, "completed");
  assert.equal(response.report.risk_points[0].supported_tags[0], "赠送利益");
  assert.equal(response.report.risk_points[0].pending_tags[0], "合同外利益");
  assert.equal(response.report.resources.physical_provider_requests, 0);
  assert.equal(response.report.resources.model_request_units, 3);
  assert.equal(response.report.model_generated, false);
  assert.equal(response.report.input_spans[0].quote, text);
  assert.ok(response.events.some((e: any) => e.type === "pi.tool_execution_end"));
});
test("H03/H05/H22: repeating mixed batches cannot bypass task budget", async () => {
  const calls = [call("search_cases", search), { ...call("read_source", { source_id: "INPUT:unit-task", max_chars: 500 }), id: "tool-2" }];
  const response = await runTask(task({ config: { max_model_requests: 3, max_tool_actions: 1 } }), bridge(), [analysis(), calls, calls, calls]);
  assert.ok(response.report.resources.model_request_units <= 3);
  assert.ok(response.report.resources.tool_actions <= 1);
  assert.equal(response.report.task_status, "partial");
});
test("H21/H27: one repair shared between analysis and final; owned fields rejected", async () => {
  const response = await runTask(task(), bridge(), [{ ...analysis(), task_id: "forged" }, analysis(), { ...final(), budget: { max_model_requests: 100 } }, final()]);
  assert.equal(response.report.task_status, "partial");
  assert.equal(response.report.resources.json_repairs, 1);
  assert.equal(response.report.task_id, "unit-task");
  assert.ok(response.report.resources.model_request_units <= 3);
});
test("H25: missing segment must prevent whole-document no-risk", async () => {
  const input = task(); const second = "\n其他材料";
  input.input.text += second; input.input.text_sha256 = hash(input.input.text);
  input.input.input_sha256 = hash(canonical({ text: input.input.text, user_context: {} }));
  input.input.segments.push({ segment_id: "S0002", start: Array.from(text).length, end: Array.from(input.input.text).length, text: second, source_id: input.input.source_id });
  const clean = { ...final(), risk_points: [] };
  const response = await runTask(input, bridge(), [analysis(), clean]);
  assert.equal(response.report.task_status, "partial");
  assert.equal(response.report.overall_judgment, "review_required");
  assert.deepEqual(response.report.coverage.not_reviewed_segment_ids, ["S0002"]);
});
test("H26: invented case/evidence discarded but valid input support retained", async () => {
  const value = final(); value.risk_points[0].case_match_status = "reliable_match";
  value.risk_points[0].case_matches = [{ source_id: "invented", evidence_ids: ["fake"], same_points: ["相同"], differences: ["未知"], match_status: "reliable_match" }];
  const response = await runTask(task(), bridge(), [analysis(), [call("search_cases", search)], value]);
  assert.equal(response.report.risk_points[0].supported_tags[0], "赠送利益");
  assert.equal(response.report.risk_points[0].case_matches.length, 0);
  assert.equal(response.report.validation.references_valid, false);
  assert.equal(response.report.needs_manual_review, true);
});
test("H04: live profile cannot pass without verified model adapter", async () => {
  await assert.rejects(() => runTask(task({ config: { execution_mode: "live" } }), bridge()), /LIVE_PROFILE_NOT_ADMITTED/);
});
test("H18: timed out mock generation retains unknown reservation", async () => {
  const response = await runTask(task({ config: { model_request_timeout_seconds: 0.02 } }), bridge(), [() => new Promise(() => {})]);
  assert.equal(response.report.task_status, "partial");
  assert.equal(response.report.resources.unknown_usage_requests, 1);
  assert.ok(response.report.resources.reserved_tokens > 0);
});

test("H02: reviewed prohibition reminder can end with not_required and no retrieval", async () => {
  const value = final(); const point = value.risk_points[0];
  point.judgment = "no_clear_risk"; point.tag_checks = []; point.case_match_status = "not_required";
  point.mechanism = "模拟：这是禁止性引用的排除理由，需要实际语义样本验证。"; point.unknowns = [];
  const response = await runTask(task(), bridge(), [analysis(), value]);
  assert.equal(response.report.task_status, "completed"); assert.equal(response.report.overall_judgment, "no_clear_risk");
  assert.equal(response.report.resources.tool_actions, 0); assert.equal(response.report.validation.semantic_review_status, "not_assessed");
});
test("H10: re-verification failure removes case support and preserves current-input labels", async () => {
  const evidence = { evidence_id: "E1", source_id: "P1", source_kind: "penalty", start: 0, end: 2, quote: "历史", text_sha256: "wrong", source_verified: true };
  const response = empty(); response.status = "ok"; response.data.queries[0].status = "ok";
  response.data.queries[0].candidates = [{ evidence: [evidence] }];
  const b = bridge(response); b.verifyEvidence = async () => ({ valid_ids: [], rejected_ids: ["E1"] });
  const value = final(); value.risk_points[0].case_match_status = "reliable_match";
  value.risk_points[0].case_matches = [{ source_id: "P1", evidence_ids: ["E1"], same_points: ["模拟相同点"], differences: ["合同未知"], match_status: "reliable_match" }];
  const result = await runTask(task(), b, [analysis(), [call("search_cases", search)], value]);
  assert.equal(result.report.validation.references_valid, false); assert.equal(result.report.risk_points[0].case_matches.length, 0);
  assert.deepEqual(result.report.risk_points[0].supported_tags, ["赠送利益"]);
});
test("H16/H17: failed query cannot be mislabeled genuine no-reliable-match", async () => {
  const failure = empty(); failure.status = "partial"; failure.data.queries[0].status = "error"; failure.data.queries[0].complete = false;
  const result = await runTask(task(), bridge(failure), [analysis(), [call("search_cases", search)], final()]);
  assert.equal(result.report.task_status, "partial"); assert.equal(result.report.risk_points[0].case_match_status, "not_completed");
  assert.equal(result.report.resources.search_query_attempts, 1);
});
test("H11: invented label never becomes official other", async () => {
  const value = final(); value.risk_points[0].tag_checks[0].tag = "赠送礼物";
  const result = await runTask(task(), bridge(), [analysis(), value, value]);
  assert.equal(result.report.task_status, "partial");
  assert.equal(result.report.risk_points.some((p: JsonObject) => p.supported_tags.includes("其他")), false);
});

test("supported label cannot retain not_required case when model says no_clear_risk", async () => {
  const value = final(); value.risk_points[0].judgment = "no_clear_risk";
  value.risk_points[0].case_match_status = "not_required";
  value.risk_points[0].tag_checks.forEach((tag: JsonObject) => tag.case_match_status = "not_required");
  const result = await runTask(task(), bridge(), [analysis(), value]);
  assert.equal(result.report.task_status, "partial");
  assert.equal(result.report.risk_points[0].case_match_status, "not_completed");
  assert.equal(result.report.risk_points[0].tag_checks[0].case_match_status, "not_completed");
});

test("duplicate native tool IDs cannot bypass action budget within a batch", async () => {
  let actual = 0; const b = bridge(); b.call = async () => { actual++; return empty(); };
  const calls = Array.from({ length: 5 }, () => call("get_risk_definition", { tags: ["赠送利益"] }));
  const value = final(); value.risk_points = [];
  const result = await runTask(task({ config: { max_tool_actions: 1 } }), b, [analysis(), calls, value]);
  assert.ok(actual <= 1, `actual backend calls ${actual}`);
  assert.equal(result.report.task_status, "partial");
});

test("Pi transcript exposes tools on first reviewing request and removes them before finalizing", async () => {
  const seen: string[][] = [];
  const result = await runTask(task({ config: { max_model_requests: 3 } }), bridge(), [
    context => { seen.push(getCurrentTools(context.messages).map(t => t.name)); return analysis(); },
    context => { seen.push(getCurrentTools(context.messages).map(t => t.name)); return [call("search_cases", search)]; },
    context => { seen.push(getCurrentTools(context.messages).map(t => t.name)); return final(); },
  ]);
  assert.deepEqual(seen, [[], ["search_cases", "read_source", "get_risk_definition"], []]);
  assert.equal(result.report.task_status, "completed");
});
test("unresolved condition query cannot be hidden by a successful mechanism query", async () => {
  const response = empty(); response.status = "partial";
  response.data.queries.push({ ...response.data.queries[0], query_id: "Q02", goal: "condition_check", status: "error", complete: false });
  const result = await runTask(task(), bridge(response), [analysis(), [call("search_cases", { ...search, queries: [search.queries[0], { ...search.queries[0], goal: "condition_check" }] })], final()]);
  assert.equal(result.report.task_status, "partial");
  assert.equal(result.report.risk_points[0].case_match_status, "not_completed");
});
test("key unknowns prevent whole-document no-risk even without pending labels", async () => {
  const value = final(); const point = value.risk_points[0];
  point.judgment = "no_clear_risk"; point.tag_checks = []; point.case_match_status = "not_required";
  point.unknowns = ["说话者和用途未知，可能改变结论。"];
  const result = await runTask(task(), bridge(), [analysis(), value]);
  assert.equal(result.report.overall_judgment, "review_required");
});
test("transport retry cannot exceed search_calls cap", async () => {
  let calls = 0; const b = bridge(); b.call = async () => { calls++; throw new Error("TOOL_TRANSPORT_HTTP_503"); };
  const result = await runTask(task({ config: { max_search_calls: 1 } }), b, [analysis(), [call("search_cases", search)], final()]);
  assert.equal(calls, 1); assert.equal(result.report.resources.search_calls, 1);
});

test("conflicting duplicate tag becomes pending while independent supported tags are retained", async () => {
  const value = final(); const point = value.risk_points[0];
  const duplicate = structuredClone(point.tag_checks[0]); duplicate.status = "unknown";
  duplicate.conditions[0].status = "unknown"; duplicate.support_input_refs = [];
  const independent = structuredClone(point.tag_checks[0]); independent.tag = "诱导投保";
  point.tag_checks = [point.tag_checks[0], duplicate, independent];
  const result = await runTask(task(), bridge(), [analysis(), [call("search_cases", search)], value]);
  assert.equal(result.report.task_status, "partial");
  assert.deepEqual(result.report.risk_points[0].supported_tags, ["诱导投保"]);
  assert.deepEqual(result.report.risk_points[0].pending_tags, ["赠送利益"]);
});

test("H08: closed no-progress query in a batch does not block a new input span", async () => {
  const input = task(); const text = "甲。乙。";
  input.input.text = text; input.input.text_sha256 = hash(text);
  input.input.input_sha256 = hash(canonical({ text, user_context: {} }));
  input.input.segments = [{ segment_id: "S0001", source_id: input.input.source_id, start: 0, end: 4, text }];
  const q = (word: string) => ({ text: word, focus_refs: [{ segment_id: "S0001", quote: word, occurrence: 1 }], risk_point_ids: [], goal: "mechanism_search" });
  const received: string[][] = []; const b = bridge();
  b.call = async (_name: string, args: JsonObject) => {
    received.push(args.queries.map((row: JsonObject) => row.text));
    const value = empty(); value.data.queries = args.queries.map((row: JsonObject) => ({ ...row, status: "empty", complete: true, candidates: [] }));
    value.meta.backend_query_attempts = args.queries.length; return value;
  };
  const query = (queries: JsonObject[]) => [call("search_cases", { queries, top_k: 5, filters: {} })];
  const value = final(); value.risk_points = [];
  await runTask(input, b, [analysis(), query([q("甲。")]), query([q("甲。")]), query([q("甲。"), q("乙。")]), value]);
  assert.deepEqual(received, [["甲。"], ["甲。"], ["乙。"]]);
});

test("same goal success cannot erase a different failed query", async () => {
  const response = empty(); response.status = "partial";
  response.data.queries = [
    { ...search.queries[0], text: "合同是否约定赠礼", goal: "condition_check", status: "error", complete: false, candidates: [] },
    { ...search.queries[0], text: "赠礼与投保关系", goal: "condition_check", status: "empty", complete: true, candidates: [] },
  ];
  const result = await runTask(task(), bridge(response), [analysis(), [call("search_cases", search)], final()]);
  assert.equal(result.report.task_status, "partial");
});
test("transport-failed condition query is retained after a successful prior search", async () => {
  const b = bridge(); b.call = async (_name: string, params: JsonObject) => {
    if (params.queries[0].goal === "condition_check") throw new Error("TOOL_TRANSPORT_HTTP_503");
    return empty();
  };
  const condition = { ...search, queries: [{ ...search.queries[0], text: "核对当前合同是否约定赠礼", goal: "condition_check" }] };
  const result = await runTask(task(), b, [analysis(), [call("search_cases", search)], [call("search_cases", condition)], final()]);
  assert.equal(result.report.task_status, "partial");
  assert.equal(result.report.risk_points[0].case_match_status, "not_completed");
  assert.equal(result.report.stop.reason, "tool_failure");
});
