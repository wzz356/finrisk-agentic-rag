import test from "node:test";
import assert from "node:assert/strict";
import { TaskState } from "../src/state.js";
import { buildContext, estimateTokens } from "../src/context_builder.js";
import { defaults, hash, type InputRecord } from "../src/contracts.js";
import type { AgentMessage } from "@earendil-works/pi-agent-core";

test("context projection includes actual runtime tool declarations in its ceiling", () => {
  const state = new TaskState(input("当前材料。"));
  const timestamp = Date.now();
  const response = { data: { queries: [{ candidates: [{ evidence: [{ quote: "完整条件原话。".repeat(500) }] }] }] }, meta: {} };
  const messages: AgentMessage[] = [
    { role: "system", content: "固定规则", timestamp },
    { role: "user", content: "当前材料。", timestamp },
    { role: "assistant", api: "openai-completions", provider: "test", model: "test", content: [{ type: "toolCall", id: "native1", name: "search_cases", arguments: {} }], stopReason: "toolUse", usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } }, timestamp },
    { role: "toolResult", toolCallId: "native1", toolName: "search_cases", content: [{ type: "text", text: JSON.stringify(response) }], isError: false, timestamp },
  ];
  const stage: AgentMessage = { role: "system", content: "<runtime_stage>当前阶段=reviewing</runtime_stage>", toolsAdded: [{ name: "search_cases", description: "工具说明".repeat(300), parameters: { type: "object" } }], timestamp };
  const config = { ...defaults, service_max_input_tokens: 9000 };
  const projected = buildContext(messages, state, config, [stage]);
  assert.ok(estimateTokens(projected) <= config.service_max_input_tokens - 800);
  assert.equal(projected.at(-1), stage);
  const tool = projected.find(m => m.role === "toolResult")!;
  assert.equal(tool.role, "toolResult");
  if (tool.role === "toolResult" && tool.content[0]?.type === "text") {
    const returned = JSON.parse(tool.content[0].text);
    assert.equal(returned.data.queries[0].candidates.length, 0);
    assert.equal(returned.data.queries[0].returned_candidates_truncated, true);
  }
});

function input(text = "A😀é同句。同句。"): InputRecord {
  return { task_id: "state-test", text, source_id: "INPUT:state-test", text_sha256: hash(text), text_version: "test-v1", input_sha256: "unit", user_context: {}, segments: [{ segment_id: "S0001", source_id: "INPUT:state-test", start: 0, end: Array.from(text).length, text }] };
}
test("wire final carries each full read quote once and removes historical tool transcript", () => {
  const state = new TaskState(input("当前输入。")); state.protocolVersion = "finrisk-model-wire/1.2"; state.phase = "finalizing";
  const quote = "关键条件：该利益在保险合同之外，且不是培训中的禁止性引用。";
  state.evidence.set("E1", { evidence_id: "E1", source_id: "P1", quote, start: 0, end: Array.from(quote).length, was_read: true, source_verified: true });
  const messages: AgentMessage[] = [{ role: "system", content: "规则", timestamp: 0 }, { role: "user", content: "当前输入。", timestamp: 0 }, { role: "toolResult", toolCallId: "old", toolName: "read_source", content: [{ type: "text", text: JSON.stringify({ data: { evidence: { evidence_id: "E1", quote } } }) }], isError: false, timestamp: 0 }];
  const projected = buildContext(messages, state, defaults);
  assert.equal(JSON.stringify(projected).split(quote).length - 1, 1);
  assert.ok(projected.every(m => m.role !== "assistant" && m.role !== "toolResult"));
});
test("wire reviewing drops whole unread candidates to leave tool planning room", () => {
  const state = new TaskState(input("原文")); state.protocolVersion = "finrisk-model-wire/1.2"; state.phase = "reviewing";
  for (let i = 0; i < 6; i++) state.evidence.set(`E${i}`, { evidence_id: `E${i}`, source_id: `P${i}`, quote: `候选${i}条件原话。`.repeat(100), source_verified: true });
  const stage: AgentMessage = { role: "system", content: "<runtime_stage>reviewing</runtime_stage>", toolsAdded: [{ name: "search_cases", description: "工具", parameters: { type: "object", description: "参数说明".repeat(100) } }], timestamp: 0 };
  const projected = buildContext([{ role: "system", content: "规则", timestamp: 0 }, { role: "user", content: "原文", timestamp: 0 }], state, { ...defaults, service_max_input_tokens: 6000 }, [stage]);
  assert.ok(estimateTokens(projected) <= 5200);
  assert.ok(projected.some(m => JSON.stringify(m).includes("unread_candidates_omitted")));
});
test("final context removes duplicate accepted analysis while ledger retains scope and critical conditions", () => {
  const state = new TaskState(input("原文"));
  const unknown = { value: null, origin: "unknown", input_refs: [], context_refs: [], unknown_reason: "角色未知" };
  const condition = { condition_id: "C1", statement: "必须核对是否约定在当前合同中", critical: true, status: "unknown", input_refs: [], context_refs: [], reason: "合同条件未知" };
  const value = { kind: "analysis", document_context: { main_scene: null, scene_status: "unknown", material_functions: ["unknown"] }, segment_assessments: [{ segment_id: "S0001", status: "reviewed", reason: "已审阅输入" }], risk_points: [{ point_key: "P1", input_refs: [{ segment_id: "S0001", quote: "原文", occurrence: 1 }], mechanism: "赠礼条件待核", candidate_tags: ["合同外利益"], observations: { actor: unknown, audience: unknown, material_function: unknown, conditions: [condition] }, unknowns: ["合同条件未知"] }], unprocessed_segment_ids: [] };
  state.acceptAnalysis(value); state.phase = "finalizing";
  const messages: AgentMessage[] = [
    { role: "system", content: "规则", timestamp: 0 }, { role: "user", content: "原文", timestamp: 0 },
    { role: "assistant", api: "openai-completions", provider: "mock", model: "mock", content: [{ type: "text", text: JSON.stringify(value) }], stopReason: "stop", usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } }, timestamp: 0 },
  ];
  const projected = buildContext(messages, state, defaults);
  assert.ok(!projected.some(m => m.role === "assistant"));
  const ledger = projected.find(m => m.role === "system" && String(m.content).startsWith("<task_state>"))!;
  assert.ok(JSON.stringify(ledger).includes(condition.statement));
  assert.ok(JSON.stringify(ledger).includes("segment_assessments"));
  assert.ok(JSON.stringify(ledger).includes("document_context"));
});

test("H13/H14/H15: Node code-point reference matches Python, rejects altered quote", () => {
  const state = new TaskState(input());
  const span = state.resolve({ segment_id: "S0001", quote: "同句。", occurrence: 2 });
  assert.equal(span.start, 7); assert.equal(span.end, 10);
  assert.throws(() => state.resolve({ segment_id: "S0001", quote: "同句！", occurrence: 1 }), /INVALID_INPUT_REFERENCE/);
});
test("H07/H08: new evidence ranges deduplicate; irrelevant text does not count semantic progress", () => {
  const state = new TaskState(input("甲。乙。"));
  const query = { focus_refs: [{ segment_id: "S0001", quote: "甲。", occurrence: 1 }], risk_point_ids: [], complete: true, status: "ok" };
  const response: any = { data: { queries: [query], evidence: { evidence_id: "E1", quote: "历史", source_id: "P1", source_verified: true, start: 0, end: 2 } } };
  state.acceptTool("search_cases", {}, response); state.acceptTool("search_cases", {}, response);
  assert.equal(state.evidence.size, 1); assert.equal(state.noProgress.get(state.group(query)), 2);
  const other = { ...query, focus_refs: [{ segment_id: "S0001", quote: "乙。", occurrence: 1 }] };
  assert.equal(state.noProgress.get(state.group(other)), undefined);
  state.acceptDecision({ focus_refs: query.focus_refs, accepted_evidence_ids: ["E1"], gap_updates: [] });
  assert.equal(state.noProgress.get(state.group(query)), 0);
  state.acceptTool("search_cases", {}, response);
  state.acceptDecision({ focus_refs: query.focus_refs, accepted_evidence_ids: ["E1"], gap_updates: [] });
  assert.equal(state.noProgress.get(state.group(query)), 1);
});
test("H09: historical evidence cannot resolve current contract gap", () => {
  const state = new TaskState(input()); state.gaps.set("G0001", { type: "input_condition", status: "open" });
  assert.throws(() => state.acceptDecision({ focus_refs: [], accepted_evidence_ids: [], gap_updates: [{ gap_id: "G0001", status: "resolved", input_refs: [], evidence_ids: ["historical"], reason: "历史存在" }] }), /HISTORICAL_CASE_CANNOT_FILL_INPUT_FACT/);
});
test("H28: context pruning retains tool-call/result pairs and exact final evidence", () => {
  const state = new TaskState(input("原文"));
  const usage = { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };
  const messages: AgentMessage[] = [{ role: "system", content: "指令", timestamp: 0 }, { role: "user", content: "原文", timestamp: 0 }];
  for (let i = 0; i < 5; i++) {
    messages.push({ role: "assistant", content: [{ type: "toolCall", id: `T${i}`, name: "read_source", arguments: { source_id: "P1", max_chars: 500 } }], api: "openai-completions", provider: "mock", model: "mock", stopReason: "toolUse", usage, timestamp: 0 });
    messages.push({ role: "toolResult", toolCallId: `T${i}`, toolName: "read_source", content: [{ type: "text", text: i === 4 ? "关键条件：不满足购买条件时不赠送。" : "旧内容".repeat(500) }], isError: false, timestamp: 0 });
  }
  const result = buildContext(messages, state, { ...defaults, service_max_input_tokens: 3000 });
  assert.ok(result.some(m => m.role === "toolResult" && JSON.stringify(m).includes("不满足购买条件")));
  for (const message of result) if (message.role === "toolResult") assert.ok(result.some(m => m.role === "assistant" && m.content.some(c => c.type === "toolCall" && c.id === message.toolCallId)));
  assert.ok(result.length < messages.length);
});

test("accepted evidence quote remains visible after later large reads", () => {
  const state = new TaskState(input("原文"));
  state.evidence.set("critical", { evidence_id: "critical", source_id: "P1", start: 0, end: 20, quote: "关键例外：明确禁止赠送合同外利益。", source_verified: true, accepted_as_relevant: true });
  const messages: AgentMessage[] = [{ role: "system", content: "指令", timestamp: 0 }, { role: "user", content: "原文", timestamp: 0 }];
  for (let i = 0; i < 4; i++) {
    messages.push({ role: "assistant", content: [{ type: "text", text: "读源" }], api: "openai-completions", provider: "mock", model: "mock", stopReason: "stop", usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } }, timestamp: 0 });
    messages.push({ role: "toolResult", toolName: "read_source", toolCallId: `R${i}`, content: [{ type: "text", text: i === 0 ? "关键例外：明确禁止赠送合同外利益。" : "其他材料".repeat(1000) }], isError: false, timestamp: 0 });
  }
  const result = buildContext(messages, state, { ...defaults, service_max_input_tokens: 10000 });
  assert.ok(JSON.stringify(result).includes("关键例外：明确禁止赠送合同外利益。"));
});
