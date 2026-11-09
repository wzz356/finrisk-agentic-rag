import test from "node:test";
import assert from "node:assert/strict";
import { Budget } from "../src/budget.js";
import { defaults, hash, type Config, type JsonObject, type TaskInput, type ToolEnvelope } from "../src/contracts.js";
import { TaskState } from "../src/state.js";
import { Trace } from "../src/trace.js";
import type { ToolBridge } from "../src/tool_bridge.js";

import { createToolExecutor } from "../src/tool_executor.js";
import { createSourceReadSchedule } from "../src/source_read_schedule.js";

const input = { task_id: "shared-tool", source_id: "INPUT:shared-tool", text: "保险", user_context: {}, text_sha256: hash("保险"), input_sha256: "unused", text_version: "unit", segments: [{ segment_id: "S0001", source_id: "INPUT:shared-tool", text: "保险", start: 0, end: 2 }] };
const query = { text: "投保赠礼", focus_refs: [{ segment_id: "S0001", quote: "保险", occurrence: 1 }], risk_point_ids: ["RP0001"], goal: "mechanism_search" };
const searchParams = () => ({ queries: [structuredClone(query)], top_k: 5, filters: {} });
function evidence(source_id: string, evidence_id: string, start = 0, end = 2): JsonObject {
  return { source_id, evidence_id, source_kind: "penalty", start, end, quote: "脱".repeat(end - start), source_verified: true, text_version: "unit", text_sha256: hash(source_id), context_has_more: false };
}
function envelope(data: JsonObject | null, meta: JsonObject = {}): ToolEnvelope {
  return { schema_version: "finrisk-tools/1.0", status: "ok", data, meta: { snapshot_id: "unit", backend_query_attempts: 0, external_retrieval_requests: 0, cache_hits: 0, ...meta }, error: null };
}
function searched(params: JsonObject, candidates: JsonObject[] = []): ToolEnvelope {
  return envelope({ queries: params.queries.map((q: JsonObject) => ({ ...q, status: "ok", complete: true, candidates })) }, { backend_query_attempts: params.queries.length, external_retrieval_requests: params.queries.length });
}
async function setup(call: ToolBridge["call"], overrides: Partial<Config> = {}, extra: JsonObject = {}) {
  const config = { ...defaults, retrieval_profile: "v2-hybrid-flash-v1", max_external_retrieval_requests: 16, max_tool_transport_retries: 0, ...overrides } as Config;
  const state = new TaskState(input, config.no_progress_limit);
  const budget = new Budget(config);
  const trace = new Trace();
  const progress: string[] = [];
  let finalRequests = 0;
  const bridge: ToolBridge = { call, async verifyEvidence(ids) { return { valid_ids: ids, rejected_ids: [] }; }, ...extra.bridge };
  const executor = createToolExecutor({ task: { input, definitions: [], snapshot: { snapshot_id: "unit" } } as unknown as TaskInput, config, state, budget, trace, bridge, auditEnabled: false, emitProgress: (name: string) => progress.push(name), requestFinal: () => finalRequests++, ...extra, ...(extra.bridge ? { bridge } : {}) });
  return { executor, config, state, budget, trace, progress, finalRequests: () => finalRequests };
}

test("shared executor accepts a successful search without charging the caller action twice", async () => {
  let dispatches = 0;
  const context = await setup(async (_name, params) => { dispatches++; return searched(params); }, {}, { auditEnabled: true });
  context.budget.chargeTool("evaluation:search", "search_cases");
  const result = await context.executor.execute("search_cases", "evaluation:search", searchParams());
  assert.equal(result.isError, false);
  assert.equal(result.terminate, false);
  assert.equal(dispatches, 1);
  assert.equal(context.budget.summary().tool_actions, 1);
  assert.equal(context.budget.queryAttempts, 1);
  assert.equal(context.budget.externalRequests, 1);
  assert.equal(context.state.searches.length, 1);
  assert.deepEqual(context.progress, ["search_cases", "condition_comparison"]);
  assert.equal(context.trace.events.find(e => e.type === "evaluation.program_tool")?.data.origin, "shared_first_prefetch");
  assert.equal(context.trace.events.filter(e => e.type === "tool.result").length, 1);
});

test("shared executor rejects another snapshot before accepting read evidence", async () => {
  let afterReads = 0;
  const context = await setup(async () => envelope({ evidence: evidence("CASE", "E1") }, { snapshot_id: "wrong" }), {}, { afterRead: () => { afterReads++; } });
  const result = await context.executor.execute("read_source", "read", { source_id: "CASE", anchor_evidence_id: "E1", max_chars: 1000 });
  assert.equal(result.isError, true);
  assert.equal(context.state.protocolInvalid, true);
  assert.ok(context.state.validationErrors.includes("TOOL_SNAPSHOT_OR_SCHEMA_MISMATCH"));
  assert.equal(context.state.evidence.size, 0);
  assert.equal(afterReads, 0);
  assert.equal(context.finalRequests(), 1);
  assert.equal(context.trace.events.filter(e => e.type === "tool.failure").length, 1);
});

test("shared executor rejects malformed envelopes before accepting state", async () => {
  const context = await setup(async () => ({ ...envelope({ evidence: evidence("CASE", "E1") }), unexpected: true }) as ToolEnvelope);
  const result = await context.executor.execute("read_source", "read", { source_id: "CASE" });
  assert.equal(result.isError, true);
  assert.equal(context.state.evidence.size, 0);
  assert.match(context.state.validationErrors[0]!, /^INVALID_PROGRAM_CONTRACT:tool-envelope:/);
});

test("unknown search outcomes conservatively retain attempted query usage", async () => {
  const context = await setup(async () => { throw new Error("lost response"); });
  context.budget.chargeTool("search", "search_cases");
  const result = await context.executor.execute("search_cases", "search", searchParams());
  assert.equal(result.isError, true);
  assert.equal(context.budget.queryAttempts, 1);
  assert.equal(context.budget.externalRequests, 0);
  assert.equal(context.state.noProgress.size, 0);
  assert.equal(context.state.searches[0]?.complete, false);
  assert.equal(context.trace.events.find(e => e.type === "tool.failure")?.data.query_usage_unknown, true);
});

test("transport retry charges one additional action and clips request timeout to remaining time", async () => {
  let dispatches = 0;
  const timeouts: number[] = [];
  const context = await setup(async (_name, params, _limit, timeout) => {
    timeouts.push(timeout);
    if (++dispatches === 1) throw new Error("TOOL_TRANSPORT_HTTP_503");
    return searched(params);
  }, { max_tool_transport_retries: 1, task_deadline_seconds: 1, tool_request_timeout_seconds: 30 });
  context.budget.chargeTool("search", "search_cases");
  const result = await context.executor.execute("search_cases", "search", searchParams());
  assert.equal(result.isError, false);
  assert.equal(dispatches, 2);
  assert.equal(context.budget.summary().tool_actions, 2);
  assert.equal(context.budget.queryAttempts, 2);
  assert.equal(context.budget.externalRequests, 1);
  assert.ok(timeouts.every(ms => ms > 0 && ms <= 1000));
  assert.equal(context.state.searches.length, 1);
  assert.equal(context.state.searches[0]?.complete, true);
});

test("cancellation reconciles physical retrieval usage and opens the live shared circuit before retry", async () => {
  let dispatches = 0;
  const context = await setup(async () => { dispatches++; throw new Error("TOOL_TIMEOUT"); }, { max_tool_transport_retries: 1 }, { retrievalCircuitEnabled: true, bridge: { async cancelRetrieval() { return { external_retrieval_requests: 3, usage_known: true, cancellation_confirmed: true, transport_circuit_open: true, transport_failure_streak: 3 }; } } });
  context.budget.chargeTool("search", "search_cases");
  assert.equal(context.executor.isRetrievalCircuitOpen(), false);
  await context.executor.execute("search_cases", "search", searchParams());
  assert.equal(dispatches, 1);
  assert.equal(context.executor.isRetrievalCircuitOpen(), true);
  assert.equal(context.executor.getRetrievalAccounting()!.external_retrieval_requests, 3);
  assert.equal(context.state.stopReason, "tool_failure");
  assert.equal(context.budget.queryAttempts, 1);
  assert.equal(context.budget.externalRequests, 3);
  await context.executor.cancelBackend();
  assert.equal(context.budget.externalRequests, 3);
  assert.equal(context.trace.events.filter(e => e.type === "retrieval.circuit_open").length, 1);
});

test("legacy circuit activation defaults to the audit flag and can be explicitly enabled without audit", async () => {
  for (const flags of [{ auditEnabled: false }, { auditEnabled: true }, { auditEnabled: false, retrievalCircuitEnabled: true }]) {
    const context = await setup(async (_name, params) => ({ ...searched(params), meta: { ...searched(params).meta, transport_circuit_open: true, transport_failure_streak: 3 } }), {}, flags);
    const result = await context.executor.execute("search_cases", "model", searchParams());
    assert.equal(context.executor.isRetrievalCircuitOpen(), flags.auditEnabled || Boolean(flags.retrievalCircuitEnabled));
    assert.equal(result.terminate, flags.auditEnabled || Boolean(flags.retrievalCircuitEnabled));
    assert.equal(context.trace.events.some(e => e.type === "evaluation.model_tool"), flags.auditEnabled);
  }
});

test("afterRead receives the original accepted response and runs before tool result audit", async () => {
  const response = envelope({ evidence: evidence("CASE", "WINDOW", 0, 5) });
  const params = { source_id: "CASE", anchor_evidence_id: "ANCHOR", max_chars: 1000 };
  const seen: unknown[] = [];
  const context = await setup(async () => response, {}, { afterRead: async (readParams: JsonObject, actual: ToolEnvelope) => {
    await Promise.resolve();
    assert.equal(context.state.evidence.get("WINDOW")?.was_read, true);
    assert.equal(context.trace.events.some(e => e.type === "tool.result"), false);
    seen.push(readParams, actual);
  } });
  const result = await context.executor.execute("read_source", "read", params);
  assert.equal(result.details, response);
  assert.equal(seen[0], params);
  assert.equal(seen[1], response);
});

test("closed no-progress groups skip dispatch and preserve prior completed query semantics", async () => {
  let dispatches = 0;
  const context = await setup(async (_name, params) => { dispatches++; return searched(params); });
  const params = searchParams();
  context.state.acceptTool("search_cases", params, searched(params), "prior");
  context.state.noProgress.set(context.state.group(query), context.config.no_progress_limit);
  const result = await context.executor.execute("search_cases", "closed", params);
  assert.equal(dispatches, 0);
  assert.equal(result.details.data.queries[0].no_progress_blocked, true);
  assert.equal(result.details.data.queries[0].complete, true);
  assert.equal(context.budget.queryAttempts, 0);
  assert.equal(context.state.noProgress.get(context.state.group(query)), context.config.no_progress_limit);
});

test("replaying a shared search action retains TaskState action idempotency", async () => {
  const context = await setup(async (_name, params) => searched(params));
  await context.executor.execute("search_cases", "same", searchParams());
  await context.executor.execute("search_cases", "same", searchParams());
  assert.equal(context.state.searches.length, 1);
  assert.equal(context.state.noProgress.get(context.state.group(query)), 1);
});

function candidate(source_id: string, ranges: number[][], dynamic = true): JsonObject {
  const anchors = ranges.map(([start, end], index) => evidence(source_id, `${source_id}:E${index}`, start, end));
  return { source_id, evidence: anchors, ...(dynamic ? { required_evidence_ids: anchors.map(e => e.evidence_id) } : {}) };
}

test("shared V2 schedule recalculates coverage after each actual read and skips covered anchors", async () => {
  const state = new TaskState(input); const trace = new Trace();
  const queries = [{ candidates: [candidate("A", [[10, 20], [30, 40]]), candidate("B", [[50, 60]])] }];
  const schedule = createSourceReadSchedule({ state, trace }, queries, "fixed");
  assert.equal(schedule.planned[0]!.evidence[0].evidence_id, "A:E0");
  const first = schedule.next(0)!; schedule.attempt(first);
  state.acceptTool("read_source", {}, envelope({ evidence: evidence("A", "A:WINDOW", 0, 45) }));
  const second = schedule.next(1)!; schedule.attempt(second);
  assert.equal(second.source_id, "B");
  state.acceptTool("read_source", {}, envelope({ evidence: evidence("B", "B:WINDOW", 45, 65) }));
  assert.equal(schedule.next(2), undefined);
  schedule.end("no_unattempted_anchors");
  assert.ok(trace.events.some(e => e.type === "retrieval.read_skipped" && e.data.anchor_evidence_id === "A:E1" && e.data.reason === "already_covered"));
  assert.deepEqual(trace.events.filter(e => e.type === "retrieval.read_selected").map(e => e.data.logical_read), [1, 2]);
  assert.equal(trace.events.at(-1)?.data.logical_reads, 2);
});

test("shared V2 schedule keeps a three-read phase and does not retry uncovered attempted anchors", async () => {
  const state = new TaskState(input); const trace = new Trace();
  const schedule = createSourceReadSchedule({ state, trace }, [{ candidates: [candidate("A", [[10, 20], [30, 40], [50, 60], [70, 80]])] }], "final");
  for (let index = 0; index < 3; index++) schedule.attempt(schedule.next(index)!);
  schedule.end("read_limit");
  assert.deepEqual(trace.events.filter(e => e.type === "retrieval.read_selected").map(e => e.data.anchor_evidence_id), ["A:E0", "A:E1", "A:E2"]);
  assert.ok(trace.events.filter(e => e.type === "retrieval.read_selected").every(e => e.data.max_logical_reads === 3));
  assert.equal(trace.events.at(-1)?.data.logical_reads, 3);
});

test("shared V1 schedule retains its initial diverse ordering and three-source plan", async () => {
  const state = new TaskState(input); const trace = new Trace();
  const schedule = createSourceReadSchedule({ state, trace }, [{ candidates: [candidate("A", [[0, 2]], false), candidate("B", [[0, 2]], false), candidate("C", [[0, 2]], false), candidate("D", [[0, 2]], false)] }], "fixed");
  assert.deepEqual(schedule.planned.map((c: JsonObject) => c.source_id), ["A", "B", "C"]);
  state.acceptTool("read_source", {}, envelope({ evidence: evidence("B", "B:WINDOW") }));
  assert.equal(schedule.next(1)!.source_id, "B");
  assert.equal(schedule.next(3), undefined);
});
test("new guarded executor rejects a late bad snapshot before mutating any state or health", async () => {
  let release!: (value: ToolEnvelope) => void, enabled = true;
  const context = await setup(async () => new Promise<ToolEnvelope>(resolve => { release = resolve; }), {},
    { assertActive: () => { if (!enabled) throw new Error("LOOP_TASK_STOPPED"); } });
  const pending = context.executor.execute("search_cases", "late", searchParams());
  enabled = false;
  release(envelope({ queries: [] }, { snapshot_id: "wrong", transport_circuit_open: true, external_retrieval_requests: 2 }));
  await assert.rejects(pending, /LOOP_TASK_STOPPED/);
  assert.equal(context.state.protocolInvalid, false); assert.deepEqual(context.state.validationErrors, []);
  assert.equal(context.state.stopReason, "evidence_sufficient"); assert.equal(context.state.searches.length, 0);
  assert.equal(context.executor.isRetrievalCircuitOpen(), false);
});

test("new executor protects a reserved reassessment time slice before dispatch", async () => {
  let dispatches = 0;
  const context = await setup(async (_name, params) => { dispatches++; return searched(params); }, {}, { availableTimeMs: () => 0 });
  await assert.rejects(context.executor.execute("search_cases", "time-reserved", searchParams()), /REASSESSMENT_TIME_RESERVED/);
  assert.equal(dispatches, 0);
});

test("new cancellation may reconcile usage after close without changing business state or circuit", async () => {
  const context = await setup(async (_n, p) => searched(p), {}, {
    retrievalCircuitEnabled: true, assertActive: () => { throw new Error("LOOP_TASK_STOPPED"); },
    bridge: { async cancelRetrieval() { return { external_retrieval_requests: 1, transport_circuit_open: true, usage_known: false }; } },
  });
  await context.executor.cancelBackend();
  assert.equal(context.budget.externalRequests, 1); assert.equal(context.state.stopReason, "evidence_sufficient");
  assert.equal(context.executor.isRetrievalCircuitOpen(), false);
});
