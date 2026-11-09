import test from "node:test";
import assert from "node:assert/strict";
import { deepseekGateway, loadDeepSeekKey } from "../src/deepseek_gateway.js";
import { mkdirSync, mkdtempSync, writeFileSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { TrialBudget } from "../src/trial_budget.js";
import { Budget } from "../src/budget.js";
import { TaskState } from "../src/state.js";
import { Trace } from "../src/trace.js";
import { defaults, root, type InputRecord } from "../src/contracts.js";

test("explicit key file takes precedence over inherited environment key", () => {
  const beforeKey = process.env.DEEPSEEK_API_KEY;
  const beforeFile = process.env.DEEPSEEK_API_KEY_FILE;
  const temp = join(root, ".test-temp-key"); mkdirSync(temp, { recursive: true });
  const file = join(mkdtempSync(join(temp, "key-")), "key.txt");
  writeFileSync(file, "sk-file-key-for-offline-test", "utf8");
  try {
    process.env.DEEPSEEK_API_KEY = "sk-inherited-key-for-offline-test";
    process.env.DEEPSEEK_API_KEY_FILE = file;
    assert.equal(loadDeepSeekKey(), "sk-file-key-for-offline-test");
  } finally {
    if (beforeKey === undefined) delete process.env.DEEPSEEK_API_KEY; else process.env.DEEPSEEK_API_KEY = beforeKey;
    if (beforeFile === undefined) delete process.env.DEEPSEEK_API_KEY_FILE; else process.env.DEEPSEEK_API_KEY_FILE = beforeFile;
  }
});

test("trial egress refusal never reaches HTTP transport or counts as physical request", async () => {
  const beforeLedger = process.env.FINRISK_AGENT_TRIAL_LEDGER;
  const temp = join(root, ".test-temp-trial"); mkdirSync(temp, { recursive: true });
  const path = join(mkdtempSync(join(temp, "exhausted-")), "ledger.json");
  const guard = new TrialBudget(path, 1);
  const body = JSON.stringify({ model: "deepseek-flash", max_tokens: 4096, thinking: { type: "disabled" }, messages: [] });
  for (let i = 0; i < 18; i++) guard.reserve(body);
  process.env.FINRISK_AGENT_TRIAL_LEDGER = path;
  try {
    const budget = new Budget({ ...defaults, execution_mode: "live" }); let calls = 0; const trace = new Trace();
    const { streamFn, model } = deepseekGateway(budget, state(), trace, "unit-test-key", async () => { calls++; return sse(undefined); });
    const message = await streamFn(model, context).result();
    assert.equal(message.stopReason, "error"); assert.equal(message.errorMessage, "TRIAL_CNY_BUDGET_EXHAUSTED");
    assert.equal(calls, 0); assert.equal(budget.summary().physical_provider_requests, 0);
    const failure = trace.events.find(e => e.type === "model.failure")!;
    assert.equal(failure.data.physical_requests, 0);
  } finally {
    if (beforeLedger === undefined) delete process.env.FINRISK_AGENT_TRIAL_LEDGER; else process.env.FINRISK_AGENT_TRIAL_LEDGER = beforeLedger;
  }
});

function state(): TaskState {
  return new TaskState({ task_id: "adapter-test", text: "资料", user_context: {}, source_id: "INPUT:adapter-test", text_sha256: "unit", input_sha256: "unit", text_version: "test", segments: [] } as InputRecord);
}
const context: any = { messages: [{ role: "system", content: "只返回JSON", timestamp: 0 }, { role: "user", content: "资料", timestamp: 0 }] };
const reply = { id: "test-reply", object: "chat.completion.chunk", model: "deepseek-flash", created: 1, choices: [{ index: 0, delta: { role: "assistant", content: "{\"kind\":\"analysis\"}" }, finish_reason: null }] };
function sse(usage: any): Response {
  const done = { ...reply, choices: [{ index: 0, delta: {}, finish_reason: "stop" }], usage };
  return new Response(`data: ${JSON.stringify(reply)}\n\ndata: ${JSON.stringify(done)}\n\ndata: [DONE]\n\n`, { headers: { "content-type": "text/event-stream" } });
}

test("gateway independently records wire text and verifies SDK assembly", async () => {
  const budget = new Budget({ ...defaults, execution_mode: "live" }); const trace = new Trace();
  const { streamFn, model } = deepseekGateway(budget, state(), trace, "fake-key", async () => sse({ prompt_tokens: 100, completion_tokens: 40, total_tokens: 140 }));
  const message = await streamFn(model, context).result();
  assert.equal(message.stopReason, "stop");
  const observation = trace.events.find(e => e.type === "model.response_observed")?.data;
  assert.ok(observation, "wire response diagnostic must exist");
  assert.equal(observation.capture_complete, true);
  assert.equal(observation.text_matches, true);
  assert.equal(observation.raw_json_valid, true);
  assert.equal(observation.request_id, "MR0001");
  assert.equal(budget.summary().physical_provider_requests, 1);
});

test("invalid provider JSON is diagnosed without changing raw output or known usage", async () => {
  const broken = '{"points":[{"case_matches":[],"continuation":{"gaps":[]}}';
  const data = { ...reply, choices: [{ index: 0, delta: { content: broken }, finish_reason: null }] };
  const done = { ...reply, choices: [{ index: 0, delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 100, completion_tokens: 40, total_tokens: 140 } };
  const budget = new Budget({ ...defaults, execution_mode: "live" }); const trace = new Trace();
  const { streamFn, model } = deepseekGateway(budget, state(), trace, "fake-key", async () => new Response(`data: ${JSON.stringify(data)}\n\ndata: ${JSON.stringify(done)}\n\ndata: [DONE]\n\n`, { headers: { "content-type": "text/event-stream" } }));
  const message = await streamFn(model, context).result();
  assert.equal(message.content.filter(c => c.type === "text").map(c => c.type === "text" ? c.text : "").join(""), broken);
  const observation = trace.events.find(e => e.type === "model.response_observed")?.data;
  assert.ok(observation, "invalid wire JSON must have independent diagnostics");
  assert.equal(observation.raw_json_valid, false); assert.equal(observation.text_matches, true);
  assert.equal(budget.summary().unknown_usage_requests, 0);
  assert.equal(budget.summary().settled_tokens, 140);
});

test("a timeout during a partial stream preserves incomplete observation without blaming assembly", async () => {
  const budget = new Budget({ ...defaults, execution_mode: "live", model_request_timeout_seconds: 0.08 }); const trace = new Trace();
  const { streamFn, model } = deepseekGateway(budget, state(), trace, "fake-key", async () => new Response(new ReadableStream({
    start(controller) { controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify(reply)}\n\n`)); },
    pull() { return new Promise(() => {}); },
  }), { headers: { "content-type": "text/event-stream" } }));
  const message = await streamFn(model, context).result();
  assert.equal(message.errorMessage, "model_request_timeout");
  const observed = trace.events.find(e => e.type === "model.response_observed")?.data;
  assert.ok(observed, "partial response must remain diagnosable after timeout");
  assert.equal(observed.capture_complete, false); assert.equal(observed.text_matches, null);
  assert.equal(observed.fragment_count, 1);
  assert.equal(budget.summary().unknown_usage_requests, 1);
});

test("response audit write failure terminates safely after settling known usage", async () => {
  class FailingTrace extends Trace { override save(name: string, value: unknown): void { if (name.startsWith("model-response-")) throw new Error("private disk error marker"); super.save(name, value); } }
  const trace = new FailingTrace(); const budget = new Budget({ ...defaults, execution_mode: "live" });
  const { streamFn, model } = deepseekGateway(budget, state(), trace, "fake-key", async () => sse({ prompt_tokens: 100, completion_tokens: 40, total_tokens: 140 }));
  const message = await Promise.race([streamFn(model, context).result(), new Promise<never>((_, reject) => { const timer = setTimeout(() => reject(new Error("stream did not terminate after audit failure")), 500); timer.unref(); })]);
  assert.equal(message.errorMessage, "MODEL_RESPONSE_AUDIT_FAILED");
  assert.equal(budget.summary().unknown_usage_requests, 0); assert.equal(budget.summary().physical_provider_requests, 1);
  assert.ok(!JSON.stringify(trace.events).includes("private disk error marker"));
});

test("unknown named SSE events cannot create a false SDK assembly failure", async () => {
  const unknown = { choices: [{ index: 0, delta: { content: "unrecognized" }, finish_reason: null }] };
  const trace = new Trace(); const budget = new Budget({ ...defaults, execution_mode: "live" });
  const normal = await sse({ prompt_tokens: 100, completion_tokens: 40, total_tokens: 140 }).text();
  const { streamFn, model } = deepseekGateway(budget, state(), trace, "fake-key", async () => new Response(`event: thread.unrecognized\ndata: ${JSON.stringify(unknown)}\n\n${normal}`, { headers: { "content-type": "text/event-stream" } }));
  const message = await streamFn(model, context).result();
  assert.equal(message.stopReason, "stop", "unknown wire events must not be guessed as assistant content");
  const observed = trace.events.find(e => e.type === "model.response_observed")!.data;
  assert.equal(observed.capture_complete, false); assert.equal(observed.text_matches, null);
  assert.equal(observed.incomplete_reason, "unsupported_event");
  assert.equal(budget.summary().settled_tokens, 140); assert.equal(budget.summary().physical_provider_requests, 1);
});
test("gateway settles trial reservation only from verified successful provider usage", async () => {
  const previous = process.env.FINRISK_AGENT_TRIAL_LEDGER;
  const temp = join(root, ".test-temp-trial"); mkdirSync(temp, { recursive: true });
  const path = join(mkdtempSync(join(temp, "gateway-settle-")), "ledger.json"); process.env.FINRISK_AGENT_TRIAL_LEDGER = path;
  try {
    const budget = new Budget({ ...defaults, execution_mode: "live" });
    const usage = { prompt_tokens: 100, completion_tokens: 40, total_tokens: 140, prompt_cache_hit_tokens: 60, prompt_cache_miss_tokens: 40 };
    const { streamFn, model } = deepseekGateway(budget, state(), new Trace(), "fake-key", async () => sse(usage));
    await streamFn(model, context).result();
    const saved = JSON.parse(readFileSync(path, "utf8"));
    assert.equal(saved.request_reservations.length, 1);
    assert.equal(saved.reserved_cost_micro_cny, 0);
    assert.equal(saved.settled_cost_micro_cny, 403);
  } finally { if (previous === undefined) delete process.env.FINRISK_AGENT_TRIAL_LEDGER; else process.env.FINRISK_AGENT_TRIAL_LEDGER = previous; }
});
test("H04/H23: DeepSeek transport invokes once, disables thinking/retries, usage cache not doubled", async () => {
  const budget = new Budget({ ...defaults, execution_mode: "live" });
  let calls = 0;
  const fetcher: typeof fetch = async (_url, init) => {
    calls++; const body = JSON.parse(String(init?.body));
    assert.equal(body.model, "deepseek-flash"); assert.equal(body.thinking.type, "disabled");
    assert.equal(body.max_tokens, 4096); assert.equal(body.stream, true);
    assert.deepEqual(body.tools, []); assert.equal(body.tool_choice, "none");
    return sse({ prompt_tokens: 100, completion_tokens: 40, total_tokens: 140, prompt_cache_hit_tokens: 60, prompt_cache_miss_tokens: 40, completion_tokens_details: { reasoning_tokens: 10 } });
  };
  const { streamFn, model } = deepseekGateway(budget, state(), new Trace(), "unit-test-key", fetcher);
  const message = await streamFn(model, context).result();
  assert.equal(message.stopReason, "stop"); assert.equal(message.usage.totalTokens, 140);
  assert.equal(message.usage.input, 40); assert.equal(message.usage.cacheRead, 60);
  assert.equal(budget.summary().settled_tokens, 140);
  assert.equal(budget.summary().physical_provider_requests, 1); assert.equal(calls, 1);
});
test("H04: 503 must not trigger hidden retry; unknown usage reserved", async () => {
  let calls = 0; const budget = new Budget({ ...defaults, execution_mode: "live" });
  const { streamFn, model } = deepseekGateway(budget, state(), new Trace(), "unit-test-key", async () => { calls++; return new Response(JSON.stringify({ error: { message: "service unavailable" } }), { status: 503, headers: { "content-type": "application/json" } }); });
  const message = await streamFn(model, context).result();
  assert.equal(message.stopReason, "error"); assert.equal(calls, 1);
  assert.equal(budget.summary().physical_provider_requests, 1);
  assert.equal(budget.summary().unknown_usage_requests, 1);
  assert.ok(!message.errorMessage?.includes("unit-test-key"));
});
test("H18: successful response without usage does not pretend zero usage", async () => {
  const budget = new Budget({ ...defaults, execution_mode: "live" });
  const { streamFn, model } = deepseekGateway(budget, state(), new Trace(), "unit-test-key", async () => sse(undefined));
  await streamFn(model, context).result();
  assert.equal(budget.summary().unknown_usage_requests, 1);
  assert.ok(Number(budget.summary().reserved_tokens) > 0);
});
test("a past repaired output does not disable tools during later reviewing requests", async () => {
  const budget = new Budget({ ...defaults, execution_mode: "live" }); const s = state(); s.phase = "reviewing"; s.repairs = 1;
  const messages = [{ ...context.messages[0], toolsAdded: [{ name: "search_cases", description: "检索", parameters: { type: "object", properties: {}, additionalProperties: false } }] }, context.messages[1]];
  const { streamFn, model } = deepseekGateway(budget, s, new Trace(), "fake-key", async (_url, init) => {
    const body = JSON.parse(String(init?.body)); assert.equal(body.tools.length, 1); assert.notEqual(body.tool_choice, "none");
    return sse({ prompt_tokens: 100, completion_tokens: 40, total_tokens: 140 });
  });
  const message = await streamFn(model, { messages } as any).result();
  assert.equal(message.stopReason, "stop");
});

test("actual DeepSeek payload keeps native tools during format correction and forbids them in JSON stages", async () => {
  for (const [phase, repairing, allowTools] of [
    ["analyzing", false, false], ["analyzing", true, false],
    ["reviewing", false, true], ["reviewing", true, false],
    ["finalizing", false, false], ["finalizing", true, false],
  ] as const) {
    const budget = new Budget({ ...defaults, execution_mode: "live" });
    const s = state(); s.phase = phase; s.repairing = repairing; s.toolFormatCorrections = 1;
    const messages = [{ ...context.messages[0], toolsAdded: [{ name: "search_cases", description: "检索", parameters: { type: "object", properties: {}, additionalProperties: false } }] }, context.messages[1]];
    let calls = 0;
    const { streamFn, model } = deepseekGateway(budget, s, new Trace(), "fake-key", async (_url, init) => {
      calls++;
      const body = JSON.parse(String(init?.body));
      assert.equal(body.thinking.type, "disabled"); assert.equal(body.stream, true);
      if (allowTools) {
        assert.equal(body.tools.length, 1); assert.notEqual(body.tool_choice, "none");
        assert.equal(body.response_format, undefined);
      } else {
        assert.deepEqual(body.tools, []); assert.equal(body.tool_choice, "none");
        assert.deepEqual(body.response_format, { type: "json_object" });
      }
      return sse({ prompt_tokens: 100, completion_tokens: 40, total_tokens: 140 });
    });
    const message = await streamFn(model, { messages } as any).result();
    assert.equal(message.stopReason, "stop"); assert.equal(calls, 1);
    assert.equal(budget.summary().physical_provider_requests, 1);
    assert.equal(budget.config.provider_auto_retries, 0); assert.equal(budget.config.max_model_transport_retries, 0);
  }
});

const secretDiagnosticMarker = "sk-DO-NOT-LOG https://secret.invalid Authorization: Bearer hidden";
for (const [code, category] of [["ENOTFOUND", "dns"], ["ECONNREFUSED", "connect"], ["CERT_HAS_EXPIRED", "tls"], ["UND_ERR_CONNECT_TIMEOUT", "timeout"], ["ABORT_ERR", "abort"], ["UNTRUSTED_SECRET_CODE", "unknown"]]) {
  test(`gateway reports only allowlisted ${category} fetch diagnostics without retry`, async () => {
    let calls = 0; const budget = new Budget({ ...defaults, execution_mode: "live" }); const trace = new Trace();
    const { streamFn, model } = deepseekGateway(budget, state(), trace, "fake-key", async () => { calls++; throw new TypeError(secretDiagnosticMarker, { cause: { code, message: secretDiagnosticMarker } }); });
    const message = await streamFn(model, context).result();
    assert.equal(message.stopReason, "error"); assert.equal(calls, 1);
    const failure = trace.events.find(event => event.type === "model.failure")!.data;
    assert.deepEqual(failure.diagnostic, { stage: "fetch", category, error_code: category === "unknown" ? null : code, http_status: null });
    assert.equal(budget.summary().physical_provider_requests, 1); assert.equal(budget.summary().unknown_usage_requests, 1);
    assert.ok(!JSON.stringify(trace.events).includes(secretDiagnosticMarker));
    assert.ok(!JSON.stringify(message).includes(secretDiagnosticMarker));
    assert.ok(!JSON.stringify(failure).includes("UNTRUSTED_SECRET_CODE"));
  });
}

test("gateway records safe HTTP status without response bodies or headers", async () => {
  const budget = new Budget({ ...defaults, execution_mode: "live" }); const trace = new Trace(); let calls = 0;
  const { streamFn, model } = deepseekGateway(budget, state(), trace, "fake-key", async () => { calls++; return new Response(JSON.stringify({ error: { message: secretDiagnosticMarker } }), { status: 403, headers: { "content-type": "application/json", "x-secret": secretDiagnosticMarker } }); });
  const message = await streamFn(model, context).result();
  const failure = trace.events.find(event => event.type === "model.failure")!.data;
  assert.equal(message.errorMessage, "PROVIDER_HTTP_403"); assert.equal(calls, 1);
  assert.deepEqual(failure.diagnostic, { stage: "response", category: "http", error_code: null, http_status: 403 });
  assert.ok(!JSON.stringify(trace.events).includes(secretDiagnosticMarker));
});

test("stream failure stays stream unknown even when its error resembles a DNS error", async () => {
  const budget = new Budget({ ...defaults, execution_mode: "live" }); const trace = new Trace(); let calls = 0;
  const { streamFn, model } = deepseekGateway(budget, state(), trace, "fake-key", async () => {
    calls++;
    return new Response(new ReadableStream({ start(controller) { controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify(reply)}\n\n`)); }, pull(controller) { controller.error(Object.assign(new Error(secretDiagnosticMarker), { code: "ENOTFOUND" })); } }), { headers: { "content-type": "text/event-stream" } });
  });
  const message = await streamFn(model, context).result();
  const failure = trace.events.find(event => event.type === "model.failure")!.data;
  assert.equal(message.stopReason, "error"); assert.equal(calls, 1);
  assert.deepEqual(failure.diagnostic, { stage: "stream", category: "unknown", error_code: null, http_status: 200 });
  assert.ok(!JSON.stringify(trace.events).includes(secretDiagnosticMarker));
  assert.equal(budget.summary().unknown_usage_requests, 1);
});

test("gateway timeout records safe stage and leaves an in-flight reservation unknown", async () => {
  const budget = new Budget({ ...defaults, execution_mode: "live", model_request_timeout_seconds: 0.03 }); const trace = new Trace(); let calls = 0;
  const { streamFn, model } = deepseekGateway(budget, state(), trace, "fake-key", async () => { calls++; return new Promise(() => {}); });
  const message = await streamFn(model, context).result();
  assert.equal(message.errorMessage, "model_request_timeout"); assert.equal(calls, 1);
  assert.deepEqual(trace.events.find(event => event.type === "model.failure")!.data.diagnostic, { stage: "fetch", category: "timeout", error_code: null, http_status: null });
  assert.equal(budget.summary().unknown_usage_requests, 1);
});

function finishChunk(reason: string): string {
  return `data: ${JSON.stringify({ ...reply, choices: [{ index: 0, delta: { refusal: secretDiagnosticMarker }, finish_reason: reason }] })}\n\n`;
}

test("SDK content_filter is a safe semantic refusal with no retry or released unknown fees", async () => {
  const previous = process.env.FINRISK_AGENT_TRIAL_LEDGER;
  const temp = join(root, ".test-temp-trial"); mkdirSync(temp, { recursive: true });
  const path = join(mkdtempSync(join(temp, "content-filter-")), "ledger.json"); process.env.FINRISK_AGENT_TRIAL_LEDGER = path;
  try {
    const budget = new Budget({ ...defaults, execution_mode: "live" }); const trace = new Trace(); let calls = 0;
    const { streamFn, model } = deepseekGateway(budget, state(), trace, "fake-key", async () => {
      calls++;
      return new Response(`${finishChunk("content_filter")}data: [DONE]\n\n`, { headers: { "content-type": "text/event-stream", "x-private": secretDiagnosticMarker } });
    });
    const message = await streamFn(model, context).result();
    const failure = trace.events.find(event => event.type === "model.failure")!.data;
    assert.equal(message.stopReason, "error"); assert.equal(message.errorMessage, "MODEL_CONTENT_FILTER");
    assert.equal(failure.reason, "MODEL_CONTENT_FILTER"); assert.equal(failure.diagnostic, null);
    assert.equal(calls, 1); assert.equal(budget.summary().physical_provider_requests, 1);
    assert.equal(budget.summary().unknown_usage_requests, 1); assert.ok(Number(budget.summary().reserved_tokens) > 0);
    const saved = JSON.parse(readFileSync(path, "utf8"));
    assert.equal(saved.request_reservations.length, 1); assert.ok(saved.reserved_cost_micro_cny > 0); assert.equal(saved.settled_cost_micro_cny, 0);
    assert.ok(!JSON.stringify(trace.events).includes(secretDiagnosticMarker)); assert.ok(!JSON.stringify(message).includes(secretDiagnosticMarker));
  } finally { if (previous === undefined) delete process.env.FINRISK_AGENT_TRIAL_LEDGER; else process.env.FINRISK_AGENT_TRIAL_LEDGER = previous; }
});

test("SDK network_error and unknown finish reasons retain infrastructure classification", async () => {
  for (const reason of ["network_error", `content_filter ${secretDiagnosticMarker}`]) {
    const budget = new Budget({ ...defaults, execution_mode: "live" }); const trace = new Trace(); let calls = 0;
    const { streamFn, model } = deepseekGateway(budget, state(), trace, "fake-key", async () => {
      calls++; return new Response(`${finishChunk(reason)}data: [DONE]\n\n`, { headers: { "content-type": "text/event-stream" } });
    });
    const message = await streamFn(model, context).result();
    const failure = trace.events.find(event => event.type === "model.failure")!.data;
    assert.equal(message.errorMessage, "MODEL_TRANSPORT_FAILURE"); assert.equal(calls, 1);
    assert.deepEqual(failure.diagnostic, { stage: "stream", category: "unknown", error_code: null, http_status: 200 });
    assert.equal(budget.summary().unknown_usage_requests, 1);
    assert.ok(!JSON.stringify(trace.events).includes(secretDiagnosticMarker)); assert.ok(!JSON.stringify(message).includes(secretDiagnosticMarker));
  }
});

test("a real stream interruption after content_filter is still an infrastructure failure", async () => {
  const budget = new Budget({ ...defaults, execution_mode: "live" }); const trace = new Trace(); let calls = 0;
  const { streamFn, model } = deepseekGateway(budget, state(), trace, "fake-key", async () => {
    calls++;
    return new Response(new ReadableStream({ start(controller) { controller.enqueue(new TextEncoder().encode(finishChunk("content_filter"))); }, async pull(controller) { await new Promise(resolve => setTimeout(resolve, 5)); controller.error(new Error(secretDiagnosticMarker)); } }), { headers: { "content-type": "text/event-stream" } });
  });
  const message = await streamFn(model, context).result();
  const failure = trace.events.find(event => event.type === "model.failure")!.data;
  assert.equal(message.errorMessage, "MODEL_TRANSPORT_FAILURE"); assert.equal(calls, 1);
  assert.deepEqual(failure.diagnostic, { stage: "stream", category: "unknown", error_code: null, http_status: 200 });
  assert.equal(budget.summary().unknown_usage_requests, 1);
  assert.ok(!JSON.stringify(trace.events).includes(secretDiagnosticMarker)); assert.ok(!JSON.stringify(message).includes(secretDiagnosticMarker));
});
