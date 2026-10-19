import test from "node:test";
import assert from "node:assert/strict";
import { fixture, comparison } from "./agent_loop_fixture.js";
import { LoopRegistry } from "../src/agent_loop_state.js";

test("final assessment context replaces frozen mapping and immediately expires previous request", async () => {
  const f = fixture(), registry = new LoopRegistry(f.task, f.state, f.budget);
  registry.registerPlan(f.plan()); const option = f.addCases()[0]!;
  const projection = registry.projectPointState("RP0001"), old = registry.beginAssessment("RP0001", projection);
  projection.citationOptions.clear(); projection.contextEvidenceIds.clear();
  const finalize = (registry as any).finalizeAssessmentContext;
  assert.equal(typeof finalize, "function", "final context must be frozen after the last count/prune");
  const current = finalize.call(registry, old, projection);
  assert.equal(current.assessment_id, old.assessment_id); assert.equal(current.citationOptions.size, 0);
  await assert.rejects(registry.acceptPointAssessment(f.assess(old), old, f.bridge), /LOOP_STALE_ASSESSMENT/);
  await assert.rejects(registry.acceptPointAssessment(f.assess(current, [[comparison(option.case_ref)]]), current, f.bridge), /CASE_REF/);
  await registry.acceptPointAssessment(f.assess(current), current, f.bridge);
});

test("one shot finalization preserves AS identity and does not adopt projection identity", async () => {
  const f = fixture(), registry = new LoopRegistry(f.task, f.state, f.budget), old = registry.beginOneShot();
  const projection = registry.projectPointState;
  const finalize = (registry as any).finalizeAssessmentContext;
  assert.equal(typeof finalize, "function");
  const current = finalize.call(registry, old, f.state);
  assert.equal(current.kind, "one_shot"); assert.equal(current.assessment_id, old.assessment_id);
  assert.throws(() => finalize.call(registry, old, f.state), /LOOP_STALE_ASSESSMENT/);
  assert.equal(typeof projection, "function");
});

test("typed recovery audit failure cannot retire a blocker or append retirement", async () => {
  const f = fixture(), registry = new LoopRegistry(f.task, f.state, f.budget); registry.registerPlan(f.plan());
  const req = registry.beginAssessment("RP0001"), scope = registry.scope("assess", "RP0001", req.assessment_id);
  registry.recordDiagnostic({ scope, code: "LOOP_SCHEMA", category: "schema", blocker_class: "typed_output" });
  await registry.acceptPointAssessment(f.assess(req), req, f.bridge);
  const before = [...registry.active_blockers], retirements = [...registry.blocker_retirements];
  assert.throws(() => (registry.retireTypedBlockers as any)(scope, () => { throw new Error("AUDIT_WRITE_FAILED"); }), /AUDIT_WRITE_FAILED/);
  assert.deepEqual([...registry.active_blockers], before); assert.deepEqual(registry.blocker_retirements, retirements);
});
test("native recovery audit failure cannot retire even an accepted replacement", () => {
  const f = fixture(), registry = new LoopRegistry(f.task, f.state, f.budget); registry.registerPlan(f.plan());
  const scope = registry.scope("act", "RP0001"), d = registry.recordNativeArgumentFailure(scope, "A1", "get_risk_definition");
  registry.actions.push({ action_id: "A1", action_type: "get_risk_definition", scope, status: "blocked" }, { action_id: "A2", action_type: "get_risk_definition", scope, status: "completed" });
  const response = { schema_version: "finrisk-tools/1.0", status: "ok", data: {}, error: null, meta: { snapshot_id: "SNAPSHOT" } };
  const schema = { get_risk_definition: { type: "object", properties: { tags: { type: "array" } }, required: ["tags"] } };
  assert.throws(() => (registry.retireNativeArgumentBlocker as any)(d, scope, "A1", "A2", { tags: ["赠送利益"] }, response, schema, () => { throw new Error("AUDIT_WRITE_FAILED"); }), /AUDIT_WRITE_FAILED/);
  assert.ok(registry.active_blockers.has(d)); assert.equal(registry.blocker_retirements.length, 0);
});

test("native retirement binds the rejected action identity rather than any blocked action in scope", async () => {
  const f = fixture(), registry = new LoopRegistry(f.task, f.state, f.budget); registry.registerPlan(f.plan());
  const scope = registry.scope("act", "RP0001"), record = (registry as any).recordNativeArgumentFailure;
  assert.equal(typeof record, "function");
  const d = record.call(registry, scope, "A1", "get_risk_definition");
  registry.actions.push({ action_id: "A1", action_type: "get_risk_definition", scope, status: "blocked" },
    { action_id: "A2", action_type: "get_risk_definition", scope, status: "blocked" },
    { action_id: "A3", action_type: "get_risk_definition", scope, status: "completed" });
  const response: any = { schema_version: "finrisk-tools/1.0", status: "ok", data: {}, error: null, meta: { snapshot_id: "SNAPSHOT" } };
  const schemas = { get_risk_definition: { type: "object", properties: { tags: { type: "array" } }, required: ["tags"] } };
  assert.throws(() => registry.retireNativeArgumentBlocker(d, scope, "A2", "A3", { tags: [] }, response, schemas), /NOT_ACCEPTED/);
  assert.ok(registry.active_blockers.has(d));
});
