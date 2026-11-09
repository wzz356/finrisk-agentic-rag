import test from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync, cpSync, readdirSync, symlinkSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { fixture } from "./agent_loop_fixture.js";
import { root, canonical, hash } from "../src/contracts.js";
import { TrialBudget } from "../src/trial_budget.js";
import { deepseekGateway } from "../src/deepseek_gateway.js";
import { Budget } from "../src/budget.js";
import { Trace } from "../src/trace.js";
import { buildProviderPayload } from "../src/prompt_counter.js";
const directory = join(root, ".test-temp-trial"); mkdirSync(directory, { recursive: true });
function setup() {
  const f = fixture(), config: any = { ...f.task.config, execution_mode: "live", request_admission_policy: "v2-v1" };
  const path = join(mkdtempSync(join(directory, "planned-gateway-")), "ledger.json");
  const binding = { budget_profile_id: config.budget_profile_id, budget_profile_sha256: config.budget_profile_sha256 };
  new TrialBudget(path, 1, "v2-v1", binding).initializeEmpty();
  const context: any = { messages: [{ role: "system", content: "JSON", timestamp: 0 }, { role: "user", content: "材料", timestamp: 0 }] };
  return { ...f, config, path, binding, context };
}
function sse(value: any = {}): Response {
  const base = { id: "offline", object: "chat.completion.chunk", model: "deepseek-flash", created: 1 };
  return new Response("data: " + JSON.stringify({ ...base, choices: [{ index: 0, delta: { content: JSON.stringify(value) }, finish_reason: null }] }) + "\n\n" +
    "data: " + JSON.stringify({ ...base, choices: [{ index: 0, delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } }) + "\n\ndata: [DONE]\n\n",
    { headers: { "content-type": "text/event-stream" } });
}
async function withLedger(path: string, fn: () => Promise<void>) {
  const previous = process.env.FINRISK_AGENT_TRIAL_LEDGER; process.env.FINRISK_AGENT_TRIAL_LEDGER = path;
  try { await fn(); } finally { if (previous === undefined) delete process.env.FINRISK_AGENT_TRIAL_LEDGER; else process.env.FINRISK_AGENT_TRIAL_LEDGER = previous; }
}
test("planned real gateway binds ledger and permits final assessment at model slot twelve only", async () => {
  const f = setup(), budget = new Budget(f.config); f.state.phase = "finalizing"; let calls = 0;
  await withLedger(f.path, async () => {
    const gateway = deepseekGateway(budget, f.state, new Trace(), "offline-fake-key", async () => { calls++; return sse(); });
    for (let i = 0; i < 12; i++) assert.equal((await gateway.streamFn(gateway.model, f.context).result()).stopReason, "stop");
    assert.equal((await gateway.streamFn(gateway.model, f.context).result()).stopReason, "error");
    assert.equal(calls, 12); assert.equal(budget.summary().model_request_units, 12);
    const data = JSON.parse(readFileSync(f.path, "utf8")); assert.equal(data.budget_profile_sha256, f.binding.budget_profile_sha256);
    assert.equal(data.request_reservations.length, 12);
  });
});
test("same cached bound ledger disk hash tampering blocks next actual fetch", async () => {
  const f = setup(), budget = new Budget(f.config); let calls = 0;
  await withLedger(f.path, async () => {
    const gateway = deepseekGateway(budget, f.state, new Trace(), "offline-fake-key", async () => { calls++; return sse(); });
    assert.equal((await gateway.streamFn(gateway.model, f.context).result()).stopReason, "stop");
    const data = JSON.parse(readFileSync(f.path, "utf8")); delete data.budget_profile_sha256; writeFileSync(f.path, JSON.stringify(data));
    assert.equal((await gateway.streamFn(gateway.model, f.context).result()).stopReason, "error"); assert.equal(calls, 1);
  });
});
test("bound tampering between payload preview and physical reserve still blocks zero fetch", async () => {
  const f = setup(), budget = new Budget(f.config), trace = new Trace(); let calls = 0;
  await withLedger(f.path, async () => {
    trace.onEvent = e => { if (e.type === "context.admission") { const data = JSON.parse(readFileSync(f.path, "utf8")); data.budget_profile_sha256 = "0".repeat(64); writeFileSync(f.path, JSON.stringify(data)); } };
    const gateway = deepseekGateway(budget, f.state, trace, "offline-fake-key", async () => { calls++; return sse(); });
    assert.equal((await gateway.streamFn(gateway.model, f.context).result()).stopReason, "error"); assert.equal(calls, 0);
  });
});
test("post-prepare phase or payload mutation is refused before real fetch", async () => {
  for (const change of ["phase", "payload"]) {
    const f = setup(), budget = new Budget(f.config); let calls = 0;
    const expected = hash(canonical(buildProviderPayload(f.context, f.state.phase, false)));
    await withLedger(f.path, async () => {
      const guard = () => { if (hash(canonical(buildProviderPayload(f.context, f.state.phase, false))) !== expected || f.state.phase !== "analyzing") throw new Error("LOOP_PAYLOAD_BINDING_MISMATCH"); };
      const counter: any = async () => { if (change === "phase") f.state.phase = "reviewing"; else f.context.messages[1].content = "已变更"; return {}; };
      const gateway = deepseekGateway(budget, f.state, new Trace(), "offline-fake-key", async () => { calls++; return sse(); }, counter, guard);
      assert.equal((await gateway.streamFn(gateway.model, f.context).result()).stopReason, "error"); assert.equal(calls, 0);
    });
  }
});
test("transport failure preserves unknown bound reservation across reopening", async () => {
  const f = setup(), budget = new Budget(f.config);
  await withLedger(f.path, async () => {
    const gateway = deepseekGateway(budget, f.state, new Trace(), "offline-fake-key", async () => { throw new Error("OFFLINE_NETWORK_FAILURE"); });
    assert.equal((await gateway.streamFn(gateway.model, f.context).result()).stopReason, "error");
    new TrialBudget(f.path, 1, "v2-v1", f.binding).initializeEmpty();
    const data = JSON.parse(readFileSync(f.path, "utf8")); assert.equal(data.request_reservations.length, 1);
    assert.equal(data.request_reservations[0].settlement, undefined); assert.ok(data.reserved_cost_micro_cny > 0);
    assert.equal(budget.summary().unknown_usage_requests, 1);
  });
});
test("changed physical profile catalog is reread before the next fetch in an isolated runtime", async () => {
  const f = setup(), isolated = mkdtempSync(join(directory, "profile-runtime-"));
  for (const folder of ["contracts/agent_rag", "configs/agent_rag"]) {
    mkdirSync(join(isolated, folder), { recursive: true });
    for (const name of readdirSync(join(root, folder))) if (name.endsWith(".json")) cpSync(join(root, folder, name), join(isolated, folder, name));
  }
  cpSync(join(root, "agent_harness/dist/src"), join(isolated, "agent_harness/dist/src"), { recursive: true });
  cpSync(join(root, "agent_harness/package.json"), join(isolated, "agent_harness/package.json"));
  symlinkSync(join(root, "agent_harness/node_modules"), join(isolated, "agent_harness/node_modules"), "junction");
  const isolatedGateway = await import(pathToFileURL(join(isolated, "agent_harness/dist/src/deepseek_gateway.js")).href);
  const budget = new Budget(f.config); let calls = 0;
  await withLedger(f.path, async () => {
    const gateway = isolatedGateway.deepseekGateway(budget, f.state, new Trace(), "offline-fake-key", async () => { calls++; return sse(); });
    assert.equal((await gateway.streamFn(gateway.model, f.context).result()).stopReason, "stop");
    const catalogPath = join(isolated, "configs/agent_rag/budget-profiles.json"), catalog = JSON.parse(readFileSync(catalogPath, "utf8"));
    catalog.profiles["planned-evidence-loop-v1"].max_model_requests = 11; writeFileSync(catalogPath, JSON.stringify(catalog));
    assert.equal((await gateway.streamFn(gateway.model, f.context).result()).stopReason, "error"); assert.equal(calls, 1);
  });
});
test("Unicode windows and critical model balance deny act before spending the reassessment reserve", async () => {
  const f = setup(), config: any = { ...f.config, profile_id: "live-deepseek-flash-planned-v1", fast_path_enabled: false };
  const runner = await import("../src/agent_loop_runner.js"), tokens = (await import("./offline_counter.js")).offlineCounter();
  const guard = new TrialBudget(f.path, 1, "v2-v1", f.binding);
  const payload = (content: string) => JSON.stringify({ model: "deepseek-flash", thinking: { type: "disabled" }, max_tokens: 4096, messages: [{ role: "user", content }] });
  const id = guard.reserve(payload("a")); guard.settle(id, { prompt_tokens: 400000, completion_tokens: 0, total_tokens: 400000 });
  // Eight 1000-codepoint supplementary-character windows exceed the old ASCII increment.
  assert.equal(guard.previewSequence([payload("a".repeat(20000)), payload("a".repeat(20000) + " ".repeat(12000))]).fits, true);
  assert.equal(guard.previewSequence([payload("a".repeat(20000)), payload("a".repeat(20000) + "😀".repeat(8 * 1000))]).fits, false);
  let acts = 0;
  const previous = { fetch: globalThis.fetch, allow: process.env.FINRISK_AGENT_ALLOW_LIVE, key: process.env.DEEPSEEK_API_KEY, keyFile: process.env.DEEPSEEK_API_KEY_FILE };
  process.env.FINRISK_AGENT_ALLOW_LIVE = "1"; process.env.DEEPSEEK_API_KEY = "sk-offline-fake"; process.env.DEEPSEEK_API_KEY_FILE = "";
  globalThis.fetch = async (_input, init) => {
    const body = JSON.parse(String(init?.body)), text = JSON.stringify(body.messages);
    const decoded = body.messages.map((m: any) => String(m.content)).join("\n");
    const match = [...decoded.matchAll(/<loop_stage>(.*?)<\/loop_stage>/gs)].at(-1); assert.ok(match, text);
    const stage = JSON.parse(match![1]!);
    if (stage.kind === "act") acts++;
    return sse(stage.kind === "plan" ? f.plan() : stage.kind === "assess" ? f.assess(stage.request) : { kind: "finalize" });
  };
  const bridge: any = {
    countTokens: tokens.count, async verifyEvidence(ids: string[]) { return { valid_ids: ids, rejected_ids: [] }; },
    async call(_name: string, args: any) { return { schema_version: "finrisk-tools/1.0", status: "empty", error: null,
      meta: { snapshot_id: "SNAPSHOT", backend_query_attempts: 1, external_retrieval_requests: 0, cache_hits: 0 },
      data: { queries: args.queries.map((q: any) => ({ ...q, query_id: "Q1", status: "empty", complete: true, candidates: [],
        stage_status: { bm25: "completed", bge: "completed", rrf: "completed", rerank: "completed" },
        ranking_coverage: { schema_version: "finrisk-ranking-coverage/1.0", expected_units: 0, scored_units: 0, failed_units: 0,
          ranking_complete: true, topk_scope: "full_pool", attempts_exhausted: false, failure_counts: {}, failed_unit_ids: [],
          expected_set_sha256: "a".repeat(64), scored_set_sha256: "a".repeat(64), failed_set_sha256: "a".repeat(64), rerank_protocol: "span-ref-v2" } })) } }; },
  };
  try {
    await withLedger(f.path, async () => {
      const task = { ...f.task, config, execution_contract: runner.plannedExecutionContract(config) };
      const result = await runner.runPlannedTask(task, bridge);
      assert.equal(acts, 0); assert.equal(result.report.resources.physical_provider_requests, 2);
      assert.ok(result.events.some((e: any) => e.type === "loop.reassessment_reserve" && e.data.fee_reserved === false));
    });
  } finally {
    tokens.close(); globalThis.fetch = previous.fetch;
    for (const [key, value] of [["FINRISK_AGENT_ALLOW_LIVE", previous.allow], ["DEEPSEEK_API_KEY", previous.key], ["DEEPSEEK_API_KEY_FILE", previous.keyFile]] as const) {
      if (value === undefined) delete process.env[key]; else process.env[key] = value;
    }
  }
});

test("a bound gateway keeps one task ledger when the environment switches to another ledger", async () => {
  const f = setup(), other = setup(), budget = new Budget(f.config); let calls = 0;
  await withLedger(f.path, async () => {
    const gateway = deepseekGateway(budget, f.state, new Trace(), "offline-fake-key", async () => { if (++calls === 1) throw new Error("UNKNOWN_FIRST"); return sse(); });
    await gateway.streamFn(gateway.model, f.context).result();
    process.env.FINRISK_AGENT_TRIAL_LEDGER = other.path;
    await gateway.streamFn(gateway.model, f.context).result();
    const original = JSON.parse(readFileSync(f.path, "utf8")), switched = JSON.parse(readFileSync(other.path, "utf8"));
    assert.equal(original.request_reservations.length, 2); assert.equal(switched.request_reservations.length, 0);
    assert.equal(original.request_reservations[0].settlement, undefined);
  });
});

test("a tightened effective model cap cannot borrow the registered one-CNY ledger", async () => {
  const f = setup(), config = { ...f.config, model_trial_cap_micro_cny: 500000, task_total_cost_cap_micro_cny: 2000000 }, budget = new Budget(config);
  let calls = 0;
  await withLedger(f.path, async () => {
    const gateway = deepseekGateway(budget, f.state, new Trace(), "offline-fake-key", async () => { calls++; return sse(); });
    assert.equal((await gateway.streamFn(gateway.model, f.context).result()).stopReason, "error"); assert.equal(calls, 0);
  });
});
