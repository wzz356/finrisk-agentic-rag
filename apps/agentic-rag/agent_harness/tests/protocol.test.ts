import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { runTask } from "../src/runner.js";
import { canonical, hash, root, type JsonObject, type TaskInput } from "../src/contracts.js";
import { normalizeModelOutput } from "../src/model_protocol.js";
import { assembleReport } from "../src/result_validator.js";
import { TaskState } from "../src/state.js";
import { Budget } from "../src/budget.js";
import { defaults } from "../src/contracts.js";
import { estimateTokens } from "../src/context_builder.js";

const text = "购买这份保险就送购物卡。";
const ref = { segment_id: "S0001", quote: text, occurrence: 1 };
function task(config: JsonObject = {}): TaskInput {
  const definitions = JSON.parse(readFileSync(`${root}/contracts/agent_rag/risk-definitions.json`, "utf8"));
  return { input: { task_id: "repair", text, source_id: "INPUT:repair", text_sha256: hash(text), input_sha256: hash(canonical({ text, user_context: {} })), text_version: "unit", user_context: {}, segments: [{ segment_id: "S0001", source_id: "INPUT:repair", start: 0, end: Array.from(text).length, text }] }, snapshot: { snapshot_id: "unit", dictionary_sha256: definitions[0].dictionary_sha256 }, definitions, config };
}
const segments = [{ segment_id: "S0001", status: "reviewed", reason: "区分实际赠礼和合同条件未知。" }];
const analysis = (): JsonObject => ({ kind: "analysis", segments, points: [{ input_refs: [ref], candidate_tags: ["赠送利益", "合同外利益"], facts: [{ field: "insurance_product", value: "yes", origin: "input", input_refs: [ref], context_refs: [] }, { field: "material_function", value: "direct_promotion", origin: "inferred", input_refs: [ref], context_refs: [] }] }] });
const condition = (criterion_id: string, status = "met") => ({ criterion_id, status, input_refs: status === "unknown" ? [] : [ref], context_refs: [], reason: status === "unknown" ? "材料没有说明。" : "当前材料表达了该条件。" });
const final = () => ({ kind: "final", segments, points: [{ risk_point_id: "RP0001", tag_checks: [ { tag: "赠送利益", conditions: [condition("actual_conduct"), condition("insurance_product"), condition("gift"), condition("purchase_link")] }, { tag: "合同外利益", conditions: [condition("actual_conduct"), condition("insurance_product"), condition("benefit"), condition("contract_outside", "unknown")] } ], case_matches: [] }] });
const bridge: any = { async call(_name: string, args: JsonObject) { return { schema_version: "finrisk-tools/1.0", status: "empty", data: { queries: args.queries.map((q: JsonObject) => ({ ...q, status: "empty", complete: true, candidates: [] })) }, meta: { snapshot_id: "unit", backend_query_attempts: args.queries.length, cache_hits: 0, external_retrieval_requests: 0 }, error: null }; }, async verifyEvidence() { return { valid_ids: [], rejected_ids: [] }; } };

test("compact wire completes fixed RAG with per-tag unknown and preserved source references", async () => {
  const result = await runTask(task({ engine: "fixed_rag" }), bridge, [analysis(), final()]);
  assert.equal(result.report.task_status, "completed");
  assert.deepEqual(result.report.risk_points[0].supported_tags, ["赠送利益"]);
  assert.deepEqual(result.report.risk_points[0].pending_tags, ["合同外利益"]);
  assert.equal(result.report.resources.model_request_units, 2);
  assert.equal(result.report.resources.physical_provider_requests, 0);
  assert.equal(result.report.input_spans[0].quote, text);
});
test("missing dictionary-required contract condition stays unknown rather than supported", async () => {
  const value = final(); value.points[0]!.tag_checks[1]!.conditions.pop();
  const result = await runTask(task({ engine: "fixed_rag" }), bridge, [analysis(), value]);
  assert.deepEqual(result.report.risk_points[0].pending_tags, ["合同外利益"]);
  assert.ok(result.report.risk_points[0].tag_checks[1].conditions.some((c: JsonObject) => c.condition_id === "contract_outside" && c.status === "unknown"));
});
test("unspecified current contract cannot become outside-contract after retrieval", async () => {
  const value = final(); value.points[0]!.tag_checks[1]!.conditions.at(-1)!.status = "met"; value.points[0]!.tag_checks[1]!.conditions.at(-1)!.input_refs = [ref];
  const result = await runTask(task({ engine: "fixed_rag" }), bridge, [analysis(), value]);
  assert.deepEqual(result.report.risk_points[0].pending_tags, ["合同外利益"]);
});
test("unknown product cannot become known from the same input during finalization", async () => {
  const value = analysis(); value.points[0]!.facts[0] = { field: "insurance_product", value: null, origin: "unknown", input_refs: [], context_refs: [] };
  const result = await runTask(task({ engine: "fixed_rag" }), bridge, [value, final()]);
  assert.deepEqual(result.report.risk_points[0].supported_tags, []);
  assert.ok(result.report.risk_points[0].tag_checks.every((tag: JsonObject) => tag.conditions.some((c: JsonObject) => c.condition_id === "insurance_product" && c.status === "unknown")));
});
test("null risk point ID cannot replace an existing unknown product finding", async () => {
  const value = analysis(); value.points[0]!.facts[0] = { field: "insurance_product", value: null, origin: "unknown", input_refs: [], context_refs: [] };
  const response: any = final(); response.points[0].risk_point_id = null; response.points[0].input_refs = [ref];
  const result = await runTask(task({ engine: "fixed_rag" }), bridge, [value, response, response]);
  assert.equal(result.report.risk_points.some((p: JsonObject) => p.supported_tags.length), false);
  assert.equal(result.report.task_status, "partial");
});
test("unlisted company permission cannot be introduced as a critical criterion", async () => {
  const value = final(); value.points[0]!.tag_checks[0]!.conditions.push(condition("company_permission"));
  const result = await runTask(task({ engine: "fixed_rag" }), bridge, [analysis(), value, value]);
  assert.equal(result.report.task_status, "partial");
  assert.ok(result.report.validation.errors.some((e: string) => e.includes("INVALID_CRITERION")));
});
test("unsupported condition requires current input or user background evidence", async () => {
  const value = final(); const c = value.points[0]!.tag_checks[0]!.conditions[0]!; c.status = "not_met"; c.input_refs = [];
  const result = await runTask(task({ engine: "fixed_rag" }), bridge, [analysis(), value, value]);
  assert.equal(result.report.task_status, "partial");
  assert.ok(result.report.validation.errors.some((e: string) => e.includes("CONDITION_WITHOUT_CURRENT_SUPPORT")));
});
test("reviewed compact prohibition with zero candidates needs no case search", async () => {
  let calls = 0; const b = { ...bridge, async call(name: string, args: JsonObject) { calls++; return bridge.call(name, args); } };
  const result = await runTask(task({ engine: "fixed_rag" }), b, [{ kind: "analysis", segments, points: [] }, { kind: "final", segments, points: [] }]);
  assert.equal(result.report.overall_judgment, "no_clear_risk");
  assert.equal(result.report.task_status, "completed");
  assert.equal(calls, 0);
});
test("compact agent completes native search-read-final with unique evidence and final reservation", async () => {
  const quote = "处罚原文：以投保为条件赠送购物卡，但不说明当前输入的合同内容。";
  const evidence = { evidence_id: "E1", source_id: "P1", source_kind: "penalty", quote, start: 0, end: Array.from(quote).length, text_sha256: hash(quote), text_version: "unit", source_verified: true, context_has_more: false };
  const b: any = { ...bridge, async call(name: string, args: JsonObject) { const response = await bridge.call("search_cases", { queries: [{ text: "购物卡", focus_refs: [ref], risk_point_ids: ["RP0001"], goal: "mechanism_search" }] }); return { ...response, status: "ok", data: name === "read_source" ? { evidence } : { queries: [{ ...response.data.queries[0], status: "ok", candidates: [{ source_id: "P1", rank: 1, evidence: [evidence] }] }] }, meta: { ...response.meta, backend_query_attempts: name === "read_source" ? 0 : 1 } }; } };
  const call = (id: string, name: string, args: JsonObject) => [{ type: "toolCall", id, name, arguments: args }];
  let contexts: string[] = [];
  const result = await runTask(task({ max_model_requests: 4 }), b, [analysis(), call("S", "search_cases", { queries: [{ text: "投保赠送购物卡", focus_refs: [ref], risk_point_ids: ["RP0001"], goal: "mechanism_search" }], top_k: 5, filters: {} }), call("R", "read_source", { source_id: "P1", anchor_evidence_id: "E1", max_chars: 1000 }), c => { contexts = c.messages.map(m => JSON.stringify(m)); return final(); }]);
  assert.equal(result.report.task_status, "completed");
  assert.equal(result.report.resources.tool_actions, 2);
  assert.equal(result.report.resources.model_request_units, 4);
  assert.equal(result.report.resources.physical_provider_requests, 0);
  assert.equal(contexts.join("").split(quote).length - 1, 1);
  assert.equal(contexts.join("").split(text).length - 1, 1);
});
test("new insurance tag cannot bypass product unknown absent from the original candidate tags", async () => {
  const initial: any = analysis(); initial.points[0].candidate_tags = ["培训材料违规"]; initial.points[0].facts = [];
  const result = await runTask(task({ engine: "fixed_rag" }), bridge, [initial, final()]);
  assert.equal(result.report.risk_points[0].supported_tags.length, 0);
});
function caseBridge(): any {
  const evidence = { evidence_id: "CASE1", source_id: "SOURCE1", source_kind: "penalty", quote: "原文事实：以购买保险为条件赠送购物卡。", start: 0, end: 23, source_verified: true, text_version: "unit", text_sha256: "unit", context_has_more: false };
  return { async call(name: string, args: JsonObject) { const meta = { snapshot_id: "unit", backend_query_attempts: name === "search_cases" ? args.queries.length : 0, cache_hits: 0, external_retrieval_requests: 0 }; return { schema_version: "finrisk-tools/1.0", status: "ok", meta, error: null, data: name === "read_source" ? { evidence } : { queries: args.queries.map((q: JsonObject) => ({ ...q, status: "ok", complete: true, candidates: [{ source_id: "SOURCE1", rank: 1, evidence: [evidence] }] })) } }; }, async verifyEvidence(ids: string[]) { return { valid_ids: ids, rejected_ids: [] }; } };
}
function caseFinal(): any {
  const value: any = final(); value.points[0].case_matches = [{ source_id: "SOURCE1", evidence_ids: ["CASE1"], supports_tags: ["赠送利益"], same_points: ["赠礼与购买联系"], differences: ["当前合同外条件未知"], match_status: "reliable_match" }]; return value;
}
test("one gift case cannot become reliable support for a different contract-outside tag", async () => {
  const result = await runTask(task({ engine: "fixed_rag" }), caseBridge(), [analysis(), caseFinal()]);
  assert.equal(result.report.task_status, "completed");
  assert.equal(result.report.risk_points[0].tag_checks[0].case_match_status, "reliable_match");
  assert.equal(result.report.risk_points[0].tag_checks[1].case_match_status, "no_reliable_match");
});
test("final cannot cite source IDs whose original text is absent from final context", async () => {
  const search = [{ type: "toolCall", id: "S", name: "search_cases", arguments: { queries: [{ text: "购物卡", focus_refs: [ref], risk_point_ids: ["RP0001"], goal: "mechanism_search" }], top_k: 5, filters: {} } }];
  const original = caseBridge();
  const unavailableRead = { ...original, async call(name: string, args: JsonObject) {
    if (name === "read_source") return { schema_version: "finrisk-tools/1.0", status: "error", data: null, meta: { snapshot_id: "unit", backend_query_attempts: 0, cache_hits: 0, external_retrieval_requests: 0 }, error: { code: "SOURCE_UNAVAILABLE" } };
    return original.call(name, args);
  } };
  const result = await runTask(task({ max_model_requests: 3 }), unavailableRead, [analysis(), search, caseFinal()]);
  assert.equal(result.report.task_status, "partial");
  assert.ok(result.report.validation.errors.some((e: string) => e.includes("PRESENTED_EVIDENCE_REQUIRED")));
});
test("missing tag-specific search keeps partial even if an independent tag has a case", async () => {
  const input = task(); const s = new TaskState(input.input); s.acceptAnalysis(normalizeModelOutput(analysis(), "analysis", s, input));
  s.evidence.set("CASE1", { evidence_id: "CASE1", source_id: "SOURCE1", source_kind: "penalty", quote: "赠礼原文", source_verified: true }); s.contextEvidenceIds.add("CASE1");
  s.final = normalizeModelOutput(caseFinal(), "final", s, input);
  const result = await assembleReport(input, s, new Budget(defaults), caseBridge());
  assert.equal(result.task_status, "partial");
  assert.equal(result.risk_points[0].tag_checks[1].case_match_status, "not_completed");
});
test("case tag binding uses same new point numbering as report assembly", () => {
  const input = task(); input.input.text += "独立新事实。"; input.input.text_sha256 = hash(input.input.text); input.input.input_sha256 = hash(canonical({ text: input.input.text, user_context: {} })); input.input.segments[0]!.text = input.input.text; input.input.segments[0]!.end = Array.from(input.input.text).length;
  const s = new TaskState(input.input); s.acceptAnalysis(normalizeModelOutput(analysis(), "analysis", s, input)); s.contextEvidenceIds.add("CASE1");
  const response = caseFinal(); response.points.push({ risk_point_id: null, input_refs: [{ segment_id: "S0001", quote: "独立新事实。", occurrence: 1 }], tag_checks: [{ tag: "培训材料违规", conditions: [] }], case_matches: [{ ...response.points[0].case_matches[0], supports_tags: ["培训材料违规"] }] });
  normalizeModelOutput(response, "final", s, input);
  assert.equal(s.caseTagBindings.get("RP0002")![0]!.supports_tags[0], "培训材料违规");
  assert.equal(s.caseTagBindings.has("RP0003"), false);
});
test("execution contract mismatch is rejected before any generation or tool execution", async () => {
  const input: any = task(); input.execution_contract = { model_wire_sha256: "wrong", criteria_sha256: "wrong", prompt_sha256: "wrong" };
  let calls = 0;
  await assert.rejects(() => runTask(input, { ...bridge, async call() { calls++; throw new Error("unexpected tool"); } }, [analysis(), final()]), /EXECUTION_CONTRACT_MISMATCH/);
  assert.equal(calls, 0);
});
test("fixed source reads reserve final context room before adding a third long quote", async () => {
  let reads = 0; let finalInput = 0;
  const evidence = (id: string) => ({ evidence_id: `E${id}`, source_id: id, quote: "这是完整条件原话。".repeat(112).slice(0, 1000), source_verified: true, source_kind: "penalty" });
  const b: any = { async call(name: string, args: JsonObject) { const meta = { snapshot_id: "unit", backend_query_attempts: name === "search_cases" ? args.queries.length : 0, external_retrieval_requests: 0, cache_hits: 0 }; if (name === "read_source") { reads++; return { schema_version: "finrisk-tools/1.0", status: "ok", meta, error: null, data: { evidence: evidence(args.source_id) } }; } return { schema_version: "finrisk-tools/1.0", status: "ok", meta, error: null, data: { queries: args.queries.map((q: JsonObject) => ({ ...q, status: "ok", complete: true, candidates: ["A", "B", "C"].map(id => ({ source_id: id, evidence: [evidence(id)] })) })) } }; }, async verifyEvidence() { return { valid_ids: [], rejected_ids: [] }; } };
  const result = await runTask(task({ engine: "fixed_rag" }), b, [analysis(), context => { finalInput = estimateTokens(context.messages); return final(); }]);
  assert.equal(result.report.task_status, "completed");
  assert.ok(reads >= 1 && reads < 3);
  assert.ok(finalInput <= defaults.service_max_input_tokens - 800);
});
test("default gift demo exercises the same compact protocol as real mode", async () => {
  const result = await runTask({ ...task({ engine: "fixed_rag" }), mock_scenario: "gift-example" }, bridge);
  assert.equal(result.report.versions.contract, "finrisk-model-wire/1.2");
  assert.deepEqual(result.report.risk_points[0].supported_tags, ["赠送利益"]);
  assert.deepEqual(result.report.risk_points[0].pending_tags, ["合同外利益"]);
  assert.equal(result.report.simulated, true);
});
test("reading a source title does not complete comparison of a later matched fact", () => {
  const input = task(); const s = new TaskState(input.input); s.acceptAnalysis(normalizeModelOutput(analysis(), "analysis", s, input));
  s.searches.push({ focus_refs: [ref], risk_point_ids: ["RP0001"], goal: "counter_context", complete: true, status: "ok", candidates: [{ source_id: "SOURCE1", evidence: [{ evidence_id: "HIT", source_id: "SOURCE1", start: 5000, end: 5020 }] }] });
  s.evidence.set("HEADER", { evidence_id: "HEADER", source_id: "SOURCE1", source_kind: "penalty", start: 0, end: 6, quote: "文首处罚标题", was_read: true, source_verified: true }); s.contextEvidenceIds.add("HEADER");
  const value = normalizeModelOutput(final(), "final", s, input);
  assert.equal(value.risk_points[0].case_match_status, "not_completed");
});
