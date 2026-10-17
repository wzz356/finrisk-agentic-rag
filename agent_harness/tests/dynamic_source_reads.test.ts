import test, { after } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { canonical, hash, root, type JsonObject, type TaskInput } from "../src/contracts.js";
import { candidateBundleRead } from "../src/model_protocol.js";
import { runTask } from "../src/runner.js";
import type { ScriptStep } from "../src/model_gateway.js";
import { offlineCounter } from "./offline_counter.js";
const counter = offlineCounter();
after(() => counter.close());

process.env.FINRISK_AGENT_ALLOW_EVALUATION = "1";
const text = "购买这份保险就送购物卡。";
const ref = { segment_id: "S0001", quote: text, occurrence: 1 };
const segments = [{ segment_id: "S0001", status: "reviewed", reason: "逐段检查赠礼条件。" }];
type Phase = "fixed" | "final" | "R1" | "A";
const phases: Phase[] = ["fixed", "final", "R1", "A"];
function task(phase: Phase, config: JsonObject = {}): TaskInput {
  const definitions = JSON.parse(readFileSync(`${root}/contracts/agent_rag/risk-definitions.json`, "utf8"));
  const user_context = { purpose_hint: "面向客户的保险销售宣传" };
  return { input: { task_id: "dynamic-read", source_id: "INPUT:dynamic-read", text, user_context, text_sha256: hash(text), input_sha256: hash(canonical({ text, user_context })), text_version: "unit", segments: [{ segment_id: "S0001", source_id: "INPUT:dynamic-read", text, start: 0, end: Array.from(text).length }] }, definitions, snapshot: { snapshot_id: "unit", dictionary_sha256: definitions[0].dictionary_sha256 }, config: { engine: phase === "fixed" ? "fixed_rag" : "controlled_agent", ...(phase === "R1" || phase === "A" ? { eval_mode: phase === "R1" ? "one_shot_rag" : "adaptive_rag" } : {}), retrieval_profile: "v2-hybrid-flash-v1", max_queries_per_search: 1, max_model_requests: 6, max_tool_actions: 8, max_search_calls: 2, service_max_input_tokens: 50000, ...config } as any };
}
function outputs() {
  const conditions = ["actual_conduct", "insurance_product", "gift", "purchase_link"].map(criterion_id => ({ criterion_id, status: "met", input_refs: [ref], context_refs: [], reason: "当前材料明确支持。" }));
  const point = { input_refs: [ref], facts: [{ field: "insurance_product", value: "yes", origin: "input", input_refs: [ref], context_refs: [] }, { field: "material_function", value: "direct_promotion", origin: "user_context", input_refs: [], context_refs: ["$.user_context.purpose_hint"] }], tag_checks: [{ tag: "赠送利益", conditions }], case_matches: [] };
  return { analysis: { kind: "analysis", segments, points: [{ input_refs: point.input_refs, facts: point.facts, candidate_tags: ["赠送利益"] }] }, final: { kind: "final", segments, points: [{ risk_point_id: "RP0001", tag_checks: point.tag_checks, case_matches: [] }] }, evaluation: { kind: "evaluation_first", segments, points: [point], continuation: { gaps: [] } } };
}
function evidence(source_id: string, evidence_id: string, start: number, end: number): JsonObject {
  const quote = "脱".repeat(end - start);
  return { source_id, evidence_id, source_kind: "penalty", start, end, quote, source_verified: true, text_version: "unit", text_sha256: hash(source_id), context_has_more: false };
}
function candidate(source_id: string, coordinates: number[][]): JsonObject {
  const anchors = coordinates.map(([start, end], index) => evidence(source_id, `${source_id}:E${index}`, start!, end!));
  return { source_id, unit_id: `UNIT:${source_id}`, evidence: anchors, required_evidence_ids: anchors.map(e => e.evidence_id) };
}
// 保留真实脱敏坐标及返回窗口，正文只使用占位字符。
const fixtureA = () => [candidate("F390", [[198, 225], [279, 387]]), candidate("F422", [[505, 596], [44, 85]])];
const fixtureC15 = () => [candidate("F423", [[551, 703], [42, 83], [109, 116], [176, 184], [258, 266]])];
function bridge(candidates: JsonObject[], windows: Record<string, number[][]>, mode = "full") {
  const reads: JsonObject[] = []; let searches = 0;
  const b: any = { reads, searches: () => searches, async call(name: string, args: JsonObject) {
    const meta = { snapshot_id: "unit", backend_query_attempts: name === "search_cases" ? args.queries.length : 0, cache_hits: 0, external_retrieval_requests: 0, ...(mode === "circuit" && name === "search_cases" ? { transport_circuit_open: true, transport_failure_streak: 3 } : {}) };
    if (name === "search_cases") { searches++; return { schema_version: "finrisk-tools/1.0", status: "ok", data: { queries: args.queries.map((q: JsonObject) => ({ ...q, complete: true, status: "ok", candidates: structuredClone(candidates) })) }, meta, error: null }; }
    reads.push(structuredClone(args));
    if (mode === "timeout") return new Promise(() => {});
    if (mode === "error") return { schema_version: "finrisk-tools/1.0", status: "error", data: null, meta, error: { code: "SOURCE_UNAVAILABLE" } };
    const ordinal = reads.filter(r => r.source_id === args.source_id).length - 1;
    const range = mode === "no-progress" ? [0, 1] : windows[args.source_id]![Math.min(ordinal, windows[args.source_id]!.length - 1)]!;
    return { schema_version: "finrisk-tools/1.0", status: "ok", data: { evidence: evidence(args.source_id, `${args.source_id}:W${ordinal}`, range[0]!, range[1]!), anchor_not_fully_covered: mode === "no-progress" || mode === "long" }, meta, error: null };
  }, async verifyEvidence(ids: string[]) { return { valid_ids: ids, rejected_ids: [] }; } };
  return b;
}
async function run(phase: Phase, b: any, config: JsonObject = {}) {
  if (phase === "R1" || phase === "A") b.countTokens = counter.count;
  const value = outputs(); let complete = false;
  const final: ScriptStep = (_context, state) => { complete = state.searches.every(q => q.candidates.every((c: JsonObject) => candidateBundleRead(c, state))); return phase === "R1" || phase === "A" ? value.evaluation : value.final; };
  const search = [{ type: "toolCall", id: "model-search", name: "search_cases", arguments: { queries: [{ text: "投保赠礼", focus_refs: [ref], risk_point_ids: ["RP0001"], goal: "mechanism_search" }], top_k: 5, filters: {} } }];
  const script: ScriptStep[] = phase === "fixed" ? [value.analysis, final] : phase === "final" ? [value.analysis, search, { kind: "finalize" }, final] : [final];
  return { ...await runTask(task(phase, config), b, script), complete };
}
for (const phase of phases) {
  test(`${phase} replaces covered A anchors with F422 header in the third logical read`, async () => {
    const b = bridge(fixtureA(), { F390: [[0, 973]], F422: [[202, 1161], [0, 945]] });
    const result = await run(phase, b);
    assert.deepEqual(b.reads.map((r: JsonObject) => r.anchor_evidence_id), ["F390:E0", "F422:E0", "F422:E1"]);
    assert.ok(result.complete, JSON.stringify(result.report.validation));
    assert.ok(b.reads.every((r: JsonObject) => r.max_chars === 1000));
    assert.equal(result.report.resources.model_request_units, phase === "final" ? 4 : phase === "fixed" ? 2 : 1);
    assert.equal(result.report.resources.tool_actions, 4);
    assert.ok(result.events.some(e => e.type === "retrieval.read_skipped" && e.data.anchor_evidence_id === "F390:E1" && e.data.reason === "already_covered"));
    assert.equal(result.events.filter(e => e.type === "retrieval.read_selected").length, 3);
    assert.ok(result.events.some(e => e.type === "retrieval.read_phase_ended" && e.data.logical_reads === 3));
  });
  test(`${phase} covers all five C15 anchors using only two full windows`, async () => {
    const b = bridge(fixtureC15(), { F423: [[237, 1221], [0, 945]] });
    const result = await run(phase, b);
    assert.deepEqual(b.reads.map((r: JsonObject) => r.anchor_evidence_id), ["F423:E0", "F423:E1"]);
    assert.ok(result.complete);
    assert.ok(result.events.some(e => e.type === "retrieval.read_phase_ended" && e.data.logical_reads === 2 && e.data.reason === "no_unattempted_anchors"));
  });
  for (const mode of ["error", "no-progress", "long", "timeout"]) {
    test(`${phase} bounds ${mode} reads and never treats attempted anchors as covered`, async () => {
      const candidates = [candidate("BOUND", [[100, 1800], [2000, 2050], [3000, 3050], [4000, 4050]])];
      const b = bridge(candidates, { BOUND: [[100, 1100], [2000, 2050], [3000, 3050]] }, mode);
      const result = await run(phase, b, { max_tool_transport_retries: 0, tool_request_timeout_seconds: 0.01 });
      assert.deepEqual(b.reads.map((r: JsonObject) => r.anchor_evidence_id), ["BOUND:E0", "BOUND:E1", "BOUND:E2"]);
      assert.equal(new Set(b.reads.map((r: JsonObject) => `${r.source_id}:${r.anchor_evidence_id}`)).size, 3);
      assert.equal(result.complete, false);
      assert.ok(result.report.resources.tool_actions <= 8);
      assert.ok(result.events.some(e => e.type === "retrieval.read_phase_ended" && e.data.logical_reads === 3 && e.data.reason === "read_limit"));
    });
  }
  test(`${phase} stops dynamic reads at the shared tool limit`, async () => {
    const b = bridge(fixtureA(), { F390: [[0, 973]], F422: [[202, 1161], [0, 945]] });
    const result = await run(phase, b, { max_tool_actions: 2 });
    assert.equal(b.reads.length, 1);
    assert.equal(result.report.resources.tool_actions, 2);
    assert.equal(result.report.stop.reason, "budget_exhausted");
    assert.ok(result.events.some(e => e.type === "retrieval.read_phase_ended" && e.data.logical_reads === 1 && e.data.reason === "tool_budget"));
  });
}
test("R1 and A dynamic prefetch never admits a fourth search candidate", async () => {
  for (const phase of ["R1", "A"] as const) {
    const candidates = [candidate("FIRST", [[100, 200], [300, 400]]), candidate("SECOND", [[500, 600]]), candidate("THIRD", [[700, 800]]), candidate("FOURTH", [[900, 1000]])];
    const b = bridge(candidates, { FIRST: [[0, 450]], SECOND: [[450, 650]], THIRD: [[650, 850]], FOURTH: [[850, 1050]] });
    const result = await run(phase, b);
    assert.deepEqual(b.reads.map((r: JsonObject) => r.source_id), ["FIRST", "SECOND", "THIRD"]);
    assert.equal(result.report.resources.model_request_units, 1);
  }
});
test("explicit evaluation retrieval circuit ends prefetch before any dynamic read", async () => {
  for (const phase of ["fixed", "R1", "A"] as const) {
    const b = bridge(fixtureA(), {}, "circuit");
    const result = await run(phase, b, phase === "fixed" ? { evaluation_arm: "R2" } : {});
    assert.equal(b.reads.length, 0);
    assert.equal(result.report.stop.reason, "tool_failure");
    assert.equal(result.report.resources.model_request_units, phase === "fixed" ? 1 : 0);
    assert.ok(result.events.some(e => e.type === "retrieval.circuit_open"));
  }
});
