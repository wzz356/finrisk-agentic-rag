import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { runTask } from "../src/runner.js";
import { canonical, hash, root, type JsonObject, type TaskInput } from "../src/contracts.js";

test("fixed retrieval uses input and tags, reads diverse candidates across points", async () => {
  const text = "购买保险送购物卡。收益有保障。";
  const refs = ["购买保险送购物卡。", "收益有保障。"].map(quote => ({ segment_id: "S0001", quote, occurrence: 1 }));
  const definitions = JSON.parse(readFileSync(`${root}/contracts/agent_rag/risk-definitions.json`, "utf8"));
  const task: TaskInput = { input: { task_id: "selection", text, source_id: "INPUT:selection", text_sha256: hash(text), input_sha256: hash(canonical({ text, user_context: {} })), text_version: "unit", user_context: {}, segments: [{ segment_id: "S0001", source_id: "INPUT:selection", start: 0, end: Array.from(text).length, text }] }, snapshot: { snapshot_id: "unit", dictionary_sha256: definitions[0].dictionary_sha256 }, definitions, config: { engine: "fixed_rag", service_max_input_tokens: 20000 } };
  const segments = [{ segment_id: "S0001", status: "reviewed", reason: "两个独立候选点。" }];
  const analysis = { kind: "analysis", segments, points: refs.map((r, i) => ({ input_refs: [r], candidate_tags: [i ? "保证收益" : "赠送利益"], facts: [{ field: "material_function", value: i ? null : "direct_promotion", origin: i ? "unknown" : "inferred", input_refs: i ? [] : [r], context_refs: [] }] })) };
  const received: JsonObject[] = []; const readSources: string[] = [];
  const evidence = (id: string) => ({ evidence_id: `E${id}`, source_id: id, quote: `来源${id}的原文及条件。`, start: 0, end: 10, source_verified: true, source_kind: "penalty" });
  const b: any = { async call(name: string, args: JsonObject) {
    received.push({ name, args }); const meta = { snapshot_id: "unit", backend_query_attempts: name === "search_cases" ? args.queries.length : 0, cache_hits: 0, external_retrieval_requests: 0 };
    const candidates = (ids: string[]) => ids.map((id, index) => ({ source_id: id, rank: index + 1, duplicate_hint: id === "A" || id === "B" ? { kind: "exact_fact_text", fact_text_sha256: "same-fact" } : null, evidence: [evidence(id)] }));
    if (name === "search_cases") return { schema_version: "finrisk-tools/1.0", status: "ok", meta, error: null, data: { queries: args.queries.map((q: JsonObject, i: number) => ({ ...q, complete: true, status: "ok", candidates: candidates(i ? ["D"] : ["A", "B", "C"]) })) } };
    readSources.push(args.source_id); return { schema_version: "finrisk-tools/1.0", status: "ok", meta, error: null, data: { evidence: evidence(args.source_id) } };
  }, async verifyEvidence() { return { valid_ids: [], rejected_ids: [] }; } };
  await runTask(task, b, [analysis, { kind: "final", segments, points: refs.map((_, i) => ({ risk_point_id: `RP000${i + 1}`, tag_checks: [], case_matches: [] })) }]);
  const queries = received.find(c => c.name === "search_cases")!.args.queries;
  assert.equal(queries.length, 2);
  assert.ok(queries[0].text.includes("赠送利益"));
  assert.ok(!queries[0].text.includes("待核机制"));
  assert.deepEqual(readSources, ["A", "D", "C"]);
});
