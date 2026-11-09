import test from "node:test";
import assert from "node:assert/strict";
import { canonical, hash } from "../src/contracts.js";
import { plannedExecutionContract } from "../src/agent_loop_runner.js";
const path = "../src/agent_loop_evaluation.js";
async function api() { const m = await import(path).catch(() => ({})); assert.equal(typeof m.loopEvaluationDescription, "function", "new offline evaluation description is required"); return m; }
test("new description binds actual planned configs and never legacy budget overrides", async () => {
  const m = await api(), d = m.loopEvaluationDescription();
  assert.equal(d.schema_version, "finrisk-loop-evaluation/1.0");
  for (const stage of ["smoke", "development", "independent"]) for (const arm of ["R1", "R2", "A"]) {
    const row = d.stages[stage][arm], c = row.config;
    assert.deepEqual([c.max_model_requests,c.max_search_calls,c.max_search_query_attempts,c.max_tool_actions,c.task_deadline_seconds,c.rerank_max_requests], [12,4,4,14,600,800]);
    assert.equal(c.profile_id, "live-deepseek-flash-planned-v1");
    assert.equal(c.fast_path_enabled, arm === "A" && stage !== "smoke");
    assert.equal(row.config_sha256, hash(canonical(c)));
    assert.deepEqual(row.planned_execution_contract, plannedExecutionContract(c));
    assert.equal(row.base_config_node_sha256, hash(canonical(row.base_config)));
    assert.equal(row.base_config.orchestration_mode, arm === "R2" ? "planned_fixed" : "planned_adaptive");
  }
  assert.ok(d.loop_contract_inventory && d.planned_system_prompt && d.planned_tool_schemas && d.fast_path_rule_sha256);
});
test("CLI description is offline even without live permission", async () => {
  const cliPath = "../src/agent_loop_cli.js", c = await import(cliPath).catch(() => ({})); assert.equal(typeof c.agentLoopCli,"function");
  const old = process.env.FINRISK_AGENT_ALLOW_LIVE; delete process.env.FINRISK_AGENT_ALLOW_LIVE;
  try { assert.equal((await c.agentLoopCli(["--describe"])).schema_version,"finrisk-loop-evaluation/1.0"); }
  finally { if (old === undefined) delete process.env.FINRISK_AGENT_ALLOW_LIVE; else process.env.FINRISK_AGENT_ALLOW_LIVE = old; }
});
test("derived bundle rejects arm/config/contract drift before running", async () => {
  const m = await api(), d = m.loopEvaluationDescription(), r = d.stages.development.A;
  const b = { config: structuredClone(r.config), execution_contract: r.planned_execution_contract,
    loop_evaluation: { protocol_sha256:d.evaluation_protocol_sha256, stage:"development", arm:"A", base_config_node_sha256:r.base_config_node_sha256 } };
  m.validateLoopEvaluationBundle(b,"A","development");
  b.config.max_model_requests = 6;
  assert.throws(() => m.validateLoopEvaluationBundle(b,"A","development"), /BINDING/);
});


test("all loop arms share explicit current-fact origin guidance with purpose inference exception", async () => {
  const { plannedSystemPrompt } = await import("../src/agent_loop_runner.js");
  for (const rule of ["origin=input", "input_refs", "origin=user_context", "context_refs", "material_function", "inferred", "unknown"]) assert.ok(plannedSystemPrompt.includes(rule), rule);
  const d = (await api()).loopEvaluationDescription();
  for (const arm of ["R1", "R2", "A"]) assert.equal(d.stages.smoke[arm].planned_execution_contract.prompt_sha256, hash(plannedSystemPrompt));
});

test("sourced facts and inferred purpose remain distinct; inferred product and absent contract stay unknown", async () => {
  const { fixture } = await import("./agent_loop_fixture.js"), { LoopRegistry } = await import("../src/agent_loop_state.js");
  for (const origin of ["input", "inferred", "unknown"]) {
    const f = fixture(), registry = new LoopRegistry(f.task, f.state, f.budget), plan = f.plan(1, ["赠送利益", "合同外利益"]);
    plan.points[0].facts[0].origin = origin;
    plan.points[0].facts[1] = { field: "material_function", value: "direct_promotion", origin: "inferred", input_refs: [f.refs[0]], context_refs: [] };
    registry.registerPlan(plan);
    assert.equal(f.state.points[0]!.observations.material_function.origin, "model_inference");
    const request = registry.beginAssessment("RP0001"), assessment = f.assess(request);
    for (const check of assessment.tag_checks) for (const condition of check.conditions) if (condition.criterion_id === "contract_outside") Object.assign(condition, { status: "met", input_refs: [f.refs[0]] });
    await registry.acceptPointAssessment(assessment, request, f.bridge);
    const conditions = registry.points.get("RP0001")!.accepted!.result.tag_checks.flatMap((check: any) => check.conditions);
    assert.equal(conditions.find((c: any) => c.condition_id === "insurance_product").status, origin === "input" ? "met" : "unknown");
    assert.equal(conditions.find((c: any) => c.condition_id === "contract_outside").status, "unknown");
    assert.equal(registry.points.get("RP0001")!.facts[0]!.origin, origin);
  }
});
