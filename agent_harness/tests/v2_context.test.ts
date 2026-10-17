import test from "node:test";
import assert from "node:assert/strict";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import { buildContext, buildCountedContext, estimateTokens } from "../src/context_builder.js";
import { defaults, hash, type InputRecord, type JsonObject } from "../src/contracts.js";
import { TaskState } from "../src/state.js";

const config = { ...defaults, retrieval_profile: "v2-hybrid-flash-v1" };
const messages: AgentMessage[] = [
  { role: "system", content: "完整规则", timestamp: 0 },
  { role: "user", content: "购买保险送购物卡。", timestamp: 0 },
];

function stateFor(bundles: string[][] = [["E1", "E2"]]): TaskState {
  const text = "购买保险送购物卡。";
  const input: InputRecord = {
    task_id: "v2-context", source_id: "INPUT:v2-context", text, text_sha256: hash(text),
    text_version: "1", input_sha256: "unit", user_context: {},
    segments: [{ segment_id: "S1", text, source_id: "INPUT:v2-context", start: 0, end: text.length }],
  };
  const state = new TaskState(input);
  state.protocolVersion = "finrisk-model-wire/1.2";
  state.phase = "reviewing";
  for (const id of new Set(bundles.flat())) state.evidence.set(id, {
    evidence_id: id, source_id: "SRC", quote: `${id}完整主体行为及合同外必要条件。`.repeat(30),
    start: Number(id.slice(1)) * 1000, end: Number(id.slice(1)) * 1000 + 100,
    source_verified: true,
  });
  state.searches.push({ focus_refs: [], risk_point_ids: [], status: "ok", complete: true,
    candidates: bundles.map((ids, index) => ({
      source_id: "SRC", file_id: "F1", unit_id: `U${index + 1}`, rank: index + 1,
      required_evidence_ids: ids, evidence: ids.map(id => state.evidence.get(id)),
    })),
  });
  return state;
}

function ledger(projected: AgentMessage[]): JsonObject {
  const message = projected.find(m => m.role === "system" && typeof m.content === "string"
    && m.content.startsWith("<task_state>"));
  assert.ok(message && typeof message.content === "string");
  return JSON.parse(message.content.slice("<task_state>".length, -"</task_state>".length));
}

function assertComplete(projected: AgentMessage[]): void {
  const data = ledger(projected);
  const visible = new Set(data.evidence.map((e: JsonObject) => e.evidence_id));
  for (const query of data.searches) for (const candidate of query.candidates) {
    if (!candidate.required_evidence_ids) continue;
    assert.ok(candidate.required_evidence_ids.every((id: string) => visible.has(id)),
      `候选 ${candidate.unit_id} 的必要证据被拆开`);
  }
}

function recentSearch(state: TaskState): AgentMessage[] {
  return [...messages, {
    role: "assistant", api: "openai-completions", provider: "mock", model: "mock",
    stopReason: "toolUse", timestamp: 0,
    content: [{ type: "toolCall", id: "search1", name: "search_cases", arguments: {} }],
    usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
  }, {
    role: "toolResult", toolCallId: "search1", toolName: "search_cases", timestamp: 0,
    isError: false, content: [{ type: "text", text: JSON.stringify({
      data: { queries: state.searches }, meta: {},
    }) }],
  }];
}

test("V2 character pruning removes a whole candidate and its necessary unread evidence", () => {
  const state = stateFor();
  const full = buildContext(messages, state, { ...config, service_max_input_tokens: 1000000 });
  const result = buildContext(messages, state, {
    ...config, service_max_input_tokens: estimateTokens(full) + 799,
  });
  assertComplete(result);
  assert.deepEqual(ledger(result).searches[0].candidates, []);
  assert.deepEqual(ledger(result).evidence, []);
  assert.equal(state.evidence.size, 2, "上下文投影不能删除原账本");
  assert.equal(state.searches[0]!.candidates.length, 1);
});

test("V2 counted pruning never sends a partial candidate to the token counter", async () => {
  const state = stateFor();
  let calls = 0;
  const result = await buildCountedContext(messages, state, config, [], async projected => {
    calls++;
    assertComplete(projected);
    return { prompt_tokens: ledger(projected).searches[0].candidates.length ? 20000 : 1000 };
  });
  assert.equal(calls, 2);
  assert.equal(result.count.prompt_tokens, 1000);
  assert.deepEqual(ledger(result.messages).evidence, []);
  assert.equal(state.contextEvidenceIds.size, 0);
});

test("V2 pruning keeps shared anchors for every surviving candidate", async () => {
  const state = stateFor([["E1", "E2"], ["E2", "E3"]]);
  const result = await buildCountedContext(messages, state, config, [], async projected => {
    assertComplete(projected);
    return { prompt_tokens: ledger(projected).searches[0].candidates.length > 1 ? 20000 : 1000 };
  });
  const data = ledger(result.messages);
  assert.deepEqual(data.searches[0].candidates.map((c: JsonObject) => c.unit_id), ["U1"]);
  assert.deepEqual(data.evidence.map((e: JsonObject) => e.evidence_id), ["E1", "E2"]);
  assert.deepEqual([...state.contextEvidenceIds], ["E1", "E2"]);
});

test("V2 candidates sharing their entire evidence bundle can still be pruned separately", async () => {
  const state = stateFor([["E1", "E2"], ["E1", "E2"]]);
  const result = await buildCountedContext(messages, state, config, [], async projected => ({
    prompt_tokens: ledger(projected).searches[0].candidates.length > 1 ? 20000 : 1000,
  }));
  assertComplete(result.messages);
  assert.deepEqual(ledger(result.messages).searches[0].candidates.map((c: JsonObject) => c.unit_id),
    ["U1"]);
  assert.equal(ledger(result.messages).evidence.length, 2);
});

for (const protection of ["was_read", "accepted_as_relevant", "presented_in_fixed_context"]) {
  test(`V2 ${protection} protects the full required bundle in both budget paths`, async () => {
    const state = stateFor();
    state.evidence.get("E1")![protection] = true;
    const originalQuote = state.evidence.get("E1")!.quote;
    const full = buildContext(messages, state, { ...config, service_max_input_tokens: 1000000 });
    const characters = buildContext(messages, state, {
      ...config, service_max_input_tokens: estimateTokens(full) + 799,
    });
    assertComplete(characters);
    assert.equal(ledger(characters).evidence.length, 2);
    const counted = await buildCountedContext(messages, state, config, [], async projected => {
      assertComplete(projected);
      return { prompt_tokens: 20000 };
    });
    assert.equal(counted.count.prompt_tokens, 20000, "保护内容超限须保留并交由网关拒绝");
    assert.equal(ledger(counted.messages).evidence.length, 2);
    assert.equal(ledger(counted.messages).evidence[0].quote, originalQuote);
  });
}

test("V2 recent tool results remove the same whole candidates while preserving call pairs", async () => {
  const state = stateFor([["E1", "E2"], ["E2", "E3"]]);
  const history = recentSearch(state);
  const original = JSON.stringify(history);
  const result = await buildCountedContext(history, state, config, [], async projected => ({
    prompt_tokens: ledger(projected).searches[0].candidates.length > 1 ? 20000 : 1000,
  }));
  assertComplete(result.messages);
  const tool = result.messages.find(m => m.role === "toolResult");
  assert.ok(tool?.role === "toolResult" && tool.content[0]?.type === "text");
  const query = JSON.parse(tool.content[0].text).data.queries[0];
  assert.deepEqual(query.candidates.map((c: JsonObject) => c.unit_id), ["U1"]);
  assert.deepEqual(query.candidates[0].required_evidence_ids, ["E1", "E2"]);
  assert.equal(query.returned_candidates_truncated, true);
  assert.ok(result.messages.some(m => m.role === "assistant"
    && m.content.some(c => c.type === "toolCall" && c.id === tool.toolCallId)));
  assert.equal(JSON.stringify(history), original);
});

test("V2 final context hides incomplete unread bundles and keeps complete read bundles", () => {
  const state = stateFor([["E1", "E2"], ["E3"]]);
  state.phase = "finalizing";
  state.evidence.get("E1")!.was_read = true;
  state.evidence.get("E3")!.was_read = true;
  const result = buildContext(messages, state, config);
  assertComplete(result);
  assert.deepEqual(ledger(result).searches[0].candidates.map((c: JsonObject) => c.unit_id), ["U2"]);
  assert.equal(ledger(result).evidence.find((e: JsonObject) => e.evidence_id === "E1").quote,
    state.evidence.get("E1")!.quote);
});

test("V2 final context keeps units whose original anchors are fully covered by a read window", () => {
  const state = stateFor();
  state.phase = "finalizing";
  state.evidence.set("WINDOW", { evidence_id: "WINDOW", source_id: "SRC", start: 900, end: 2200,
    quote: "已完整核读的原文窗口", was_read: true, source_verified: true });
  const result = buildContext(messages, state, config);
  const data = ledger(result);
  assert.equal(data.searches[0].candidates.length, 1);
  assert.deepEqual(data.evidence.map((e: JsonObject) => e.evidence_id), ["WINDOW"]);
  const unit = data.searches[0].candidates[0];
  assert.deepEqual(unit.read_evidence_ids, ["WINDOW"]);
  assert.deepEqual(unit.required_evidence_locators.map((e: JsonObject) => [e.evidence_id,e.covered_by]),
    [["E1",["WINDOW"]],["E2",["WINDOW"]]]);
  assert.deepEqual([...state.contextEvidenceIds], ["WINDOW"]);
});
