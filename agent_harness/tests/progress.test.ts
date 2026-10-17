import test from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { canonical, hash, root, type JsonObject, type TaskInput } from "../src/contracts.js";
import { runTask } from "../src/runner.js";
import { criteriaFor } from "../src/criteria.js";
import { progressSnapshot } from "../src/progress.js";
import { TaskState } from "../src/state.js";
import { Budget } from "../src/budget.js";
import { defaults } from "../src/contracts.js";
import { normalizeModelOutput } from "../src/model_protocol.js";

const text = "购买这份保险就送购物卡。";
const ref = { segment_id: "S0001", quote: text, occurrence: 1 };
function task(config: JsonObject = {}): TaskInput {
  const definitions = JSON.parse(readFileSync(`${root}/contracts/agent_rag/risk-definitions.json`, "utf8"));
  return { input: { task_id: "status-unit", text, source_id: "INPUT:status-unit", user_context: {}, text_sha256: hash(text), input_sha256: hash(canonical({ text, user_context: {} })), text_version: "unit", segments: [{ segment_id: "S0001", source_id: "INPUT:status-unit", text, start: 0, end: Array.from(text).length }] }, snapshot: { snapshot_id: "unit", dictionary_sha256: definitions[0].dictionary_sha256 }, definitions, config, mock_scenario: "gift-example" };
}
const bridge: any = { async call(_name: string, args: JsonObject) { return { schema_version: "finrisk-tools/1.0", status: "empty", data: { queries: args.queries.map((query: JsonObject) => ({ ...query, complete: true, status: "empty", candidates: [] })) }, meta: { snapshot_id: "unit", backend_query_attempts: args.queries.length, cache_hits: 0, external_retrieval_requests: 0 }, error: null }; }, async verifyEvidence() { return { valid_ids: [], rejected_ids: [] }; } };
const analysis = { kind: "analysis", segments: [{ segment_id: "S0001", status: "reviewed", reason: "审阅实际赠礼表述，保留合同条件未知。" }], points: [{ input_refs: [ref], candidate_tags: ["赠送利益"], facts: [{ field: "insurance_product", value: "yes", origin: "input", input_refs: [ref], context_refs: [] }, { field: "material_function", value: "direct_promotion", origin: "inferred", input_refs: [ref], context_refs: [] }] }] };

test("critical execution steps emit factual progress and terminal budget counts", async () => {
  const result = await runTask(task({ engine: "fixed_rag" }), bridge);
  const snapshots = result.events.filter(event => event.type === "progress.snapshot");
  assert.ok(snapshots.length >= 3);
  const latest = snapshots.at(-1)!.data;
  assert.equal(latest.task_status, "completed");
  assert.equal(latest.counts.segments_analyzed, 1);
  assert.equal(latest.counts.segments_finalized, 1);
  assert.equal(latest.counts.pending_input_conditions, 1);
  assert.equal(latest.counts.pending_case_points, 0);
  assert.equal(latest.limits.model_requests.used, 2);
  assert.equal(latest.limits.tool_actions.used, 1);
  assert.equal(latest.execution_mode, "mock");
  assert.ok(!JSON.stringify(snapshots).includes(text));
  assert.equal(result.report.resources.physical_provider_requests, 0);
});
test("actual model context contains compact remaining budget after current request", async () => {
  const result = await runTask(task({ engine: "fixed_rag" }), bridge, [analysis, (context) => {
    const message = context.messages.find(m => m.role === "system" && typeof m.content === "string" && m.content.startsWith("<task_state>"))!;
    const ledger = JSON.parse(String(message.content).slice("<task_state>".length, -"</task_state>".length));
    assert.ok(ledger.budget);
    assert.equal(ledger.budget.model_remaining_after_current, 4);
    assert.equal(ledger.budget.tool_remaining, 7);
    assert.ok(JSON.stringify(ledger.budget).length < 220);
    return { kind: "final", segments: analysis.segments, points: [{ risk_point_id: "RP0001", tag_checks: [], case_matches: [] }] };
  }]);
  assert.equal(result.report.validation.schema_valid, true);
});
test("partial source window is counted as read but remains a pending case query", async () => {
  const evidence = { evidence_id: "HIT", source_id: "P1", source_kind: "penalty", start: 100, end: 1300, quote: "处罚原文", source_verified: true, text_version: "unit", text_sha256: "unit" };
  const b: any = { ...bridge, async call(name: string, args: JsonObject) {
    if (name === "read_source") return { schema_version: "finrisk-tools/1.0", status: "ok", data: { evidence: { ...evidence, evidence_id: "PREFIX", end: 1100 }, anchor_not_fully_covered: true }, meta: { snapshot_id: "unit", backend_query_attempts: 0, cache_hits: 0, external_retrieval_requests: 0 }, error: null };
    const response = await bridge.call(name, args); response.status = "ok"; response.data.queries[0].status = "ok"; response.data.queries[0].candidates = [{ source_id: "P1", evidence: [evidence] }]; return response;
  } };
  const result = await runTask(task({ engine: "fixed_rag" }), b);
  const snapshots = result.events.filter(event => event.type === "progress.snapshot"); assert.ok(snapshots.length);
  const latest = snapshots.at(-1)!.data;
  assert.equal(latest.task_status, "partial"); assert.equal(latest.counts.read_sources, 1); assert.equal(latest.counts.pending_case_queries, 1);
  assert.equal(latest.counts.pending_case_points, 1);
});

test("unlaunched case search has zero unfinished queries and one unverified point", () => {
  const input = task(); const state = new TaskState(input.input);
  state.acceptAnalysis(normalizeModelOutput(analysis, "analysis", state, input));
  const budget = new Budget(defaults);
  const running = progressSnapshot(state, budget, "condition_comparison", 1);
  assert.equal(running.counts.pending_case_queries, 0);
  assert.equal(running.counts.pending_case_points, 1);
  state.searches.push({ text: "赠礼", focus_refs: [ref], risk_point_ids: ["RP0001"], goal: "mechanism_search", complete: true, status: "empty", candidates: [] });
  const searched = progressSnapshot(state, budget, "condition_comparison", 2);
  assert.equal(searched.counts.pending_case_queries, 0);
  assert.equal(searched.counts.pending_case_points, 1);
  const base = { task_status: "partial", validation: { schema_valid: true }, coverage: { reviewed_segment_ids: ["S0001"] } };
  for (const [case_match_status, expected] of [["not_completed", 1], ["not_required", 0], ["no_reliable_match", 0]]) {
    const terminal = progressSnapshot(state, budget, "incomplete", 2, { ...base, risk_points: [{ risk_point_id: "RP0001", tag_checks: [], case_match_status, input_refs: [ref] }] });
    assert.equal(terminal.counts.pending_case_queries, 0);
    assert.equal(terminal.counts.pending_case_points, expected);
  }
});
test("latest progress and dedicated history persist without model messages", async () => {
  const input = task({ engine: "fixed_rag" }); input.task_dir = join(root, "outputs/agent_rag_status_20261014/unit-progress"); mkdirSync(input.task_dir, { recursive: true });
  const result = await runTask(input, bridge);
  assert.ok(result.events.some(event => event.type === "progress.snapshot"));
  const latest = JSON.parse(readFileSync(join(input.task_dir, "progress.json"), "utf8"));
  const history = readFileSync(join(input.task_dir, "progress_events.jsonl"), "utf8").trim().split("\n").map(line => JSON.parse(line));
  assert.equal(latest.sequence, history.at(-1)!.sequence);
  assert.ok(history.every(event => event.type === "progress.snapshot"));
  assert.ok(!JSON.stringify(history).includes(text));
});
test("closed repeated query cannot hide an earlier unread candidate in progress", async () => {
  const evidence = { evidence_id: "HIT", source_id: "P1", source_kind: "penalty", start: 0, end: 6, quote: "处罚原文", source_verified: true, text_version: "unit", text_sha256: "unit" };
  const b: any = { ...bridge, async call(name: string, args: JsonObject) { const response = await bridge.call(name, args); response.status = "ok"; response.data.queries[0].status = "ok"; response.data.queries[0].candidates = [{ source_id: "P1", evidence: [evidence] }]; return response; } };
  const search = () => [{ type: "toolCall", id: "S", name: "search_cases", arguments: { queries: [{ text: "投保赠礼", focus_refs: [ref], risk_point_ids: ["RP0001"], goal: "mechanism_search" }], top_k: 5, filters: {} } }];
  const result = await runTask(task({ max_tool_actions: 2, no_progress_limit: 1 }), b, [analysis, search(), search(), { kind: "finalize" }, { kind: "final", segments: analysis.segments, points: [{ risk_point_id: "RP0001", tag_checks: [], case_matches: [] }] }]);
  assert.equal(result.report.task_status, "partial");
  assert.equal(result.events.filter(event => event.type === "progress.snapshot").at(-1)!.data.counts.pending_case_queries, 1);
});

test("execution fallback does not become a missing business condition", async () => {
  for (const candidate_tags of [["赠送利益"], ["合同外利益"]]) {
    const output = { ...analysis, points: [{ ...analysis.points[0]!, candidate_tags }] };
    const result = await runTask(task(), bridge, [output]);
    assert.equal(result.report.validation.schema_valid, false);
    assert.ok(result.report.risk_points[0].unknowns.includes("执行未形成有效终稿。"));
    const expected = candidate_tags[0] === "合同外利益" ? 1 : 0;
    const snapshots = result.events.filter(event => event.type === "progress.snapshot");
    assert.equal(snapshots.at(-1)!.data.counts.pending_input_conditions, expected);
  }
});

test("one missing insurance fact shared by analysis and multiple final tags counts once", async () => {
  const candidate_tags = ["销售误导", "夸大收益"];
  const output = { ...analysis, points: [{ ...analysis.points[0]!, candidate_tags, facts: analysis.points[0]!.facts.filter(fact => fact.field === "material_function") }] };
  const final = { kind: "final", segments: analysis.segments, points: [{ risk_point_id: "RP0001", tag_checks: candidate_tags.map(tag => ({ tag, conditions: criteriaFor(tag).map(condition => ({ criterion_id: condition.id, status: condition.id === "insurance_product" ? "unknown" : "met", input_refs: condition.id === "insurance_product" ? [] : [ref], context_refs: [], reason: "合成计数回归，不作为业务判断。" })) })), case_matches: [] }] };
  const result = await runTask(task({ engine: "fixed_rag" }), bridge, [output, final]);
  assert.equal(result.report.validation.schema_valid, true);
  assert.equal(result.report.risk_points[0].unknowns.length, 1);
  assert.equal(result.events.filter(event => event.type === "progress.snapshot").at(-1)!.data.counts.pending_input_conditions, 1);
});
