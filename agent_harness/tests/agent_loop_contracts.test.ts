import test from "node:test";
import assert from "node:assert/strict";
import Ajv2020 from "ajv/dist/2020.js";
import { canonical, hash, officialTags, type JsonObject } from "../src/contracts.js";
import { criteriaByteHash, criteriaFor, criteriaHash } from "../src/criteria.js";
import { ModelSchemaError } from "../src/model_json.js";
import { wireSchema } from "../src/model_protocol.js";

// Dynamic import makes the initial red run report the absent contract module.
const modulePath = "../src/agent_loop_contracts.js";
const api = () => import(modulePath);
const text = "今天购买这份保险，就送500元购物卡。";
const ref = { segment_id: "S0001", quote: text, occurrence: 1 };
const productRef = { ...ref, quote: "购买这份保险" };
const segments = [{ segment_id: "S0001", status: "reviewed", reason: "本段实际宣传投保赠礼，合同内容未提供。" }];
const caseRef = "C" + "a".repeat(20);

function plan(): JsonObject {
  return structuredClone({ kind: "plan", segments: structuredClone(segments), points: [{ input_refs: [ref], facts: [
    { field: "material_function", value: "direct_promotion", origin: "user_context", input_refs: [], context_refs: ["$.user_context.purpose_hint"] },
    { field: "insurance_product", value: "yes", origin: "input", input_refs: [productRef], context_refs: [] },
    { field: "contract_outside", value: null, origin: "unknown", input_refs: [], context_refs: [] }
  ], candidate_tags: ["赠送利益", "合同外利益"], search_intent: "比较投保赠礼与合同归属条件。", query_text: "保险 投保赠送购物卡 合同外利益" }] });
}
// A separate initial single-tag state; never derive it by deleting a plan tag.
function assessmentContext(): JsonObject {
  return { expected_assessment_id: "AS0001", expected_state_revision: 1, expected_risk_point_id: "RP0001", candidate_tags: ["赠送利益"], visible_case_refs: [] };
}
function assess(): JsonObject {
  const refs: Record<string, JsonObject> = { actual_conduct: ref, insurance_product: productRef, gift: { ...ref, quote: "送500元购物卡" }, purchase_link: ref };
  return structuredClone({ kind: "assess", assessment_id: "AS0001", state_revision: 1, risk_point_id: "RP0001", tag_checks: [{ tag: "赠送利益", conditions: criteriaFor("赠送利益").map(c => ({ criterion_id: c.id, status: "met", input_refs: [refs[c.id]], context_refs: c.id === "actual_conduct" ? ["$.user_context.purpose_hint"] : [], reason: "当前材料与营销背景明确支持此条件。" })), case_assessments: [] }] });
}
function oneShot(): JsonObject {
  return structuredClone({ kind: "one_shot", segments: structuredClone(segments), points: [{ input_refs: [ref], facts: [], tag_checks: assess().tag_checks }] });
}
function addCase(value: JsonObject): void {
  value.tag_checks[0].case_assessments = [{ case_ref: caseRef, same_points: ["均以投保为条件赠礼。"], differences: ["具体赠品不同。"], match_status: "partial_match" }];
}
function rejection(a: any, value: JsonObject, kind: string, code: string, context?: JsonObject): any {
  let caught: any;
  assert.throws(() => a.validateLoopOutput(value, kind, context), (error: any) => { caught = error; return error instanceof a.LoopContractError && error.code === code; });
  return caught;
}

test("independent plan and assessment fixtures accept minimal output without generated IDs or gaps", async () => {
  const a = await api();
  assert.equal(a.AGENT_LOOP_VERSION, "finrisk-agent-loop/1.0");
  a.validateLoopOutput(plan(), "plan");
  a.validateLoopOutput(assess(), "assess", assessmentContext());
  a.validateLoopOutput(oneShot(), "one_shot");
  assert.deepEqual(a.parseLoopOutput(JSON.stringify(plan()), "plan"), plan());
  const noFacts = plan(); noFacts.points[0].facts = []; a.validateLoopOutput(noFacts, "plan");
});

test("plan and one shot are bounded at four points and plan tags at one to three", async () => {
  const a = await api();
  for (const make of [plan, oneShot]) {
    const v = make(); v.points = Array.from({ length: 4 }, () => structuredClone(v.points[0])); a.validateLoopOutput(v, v.kind);
    v.points.push(structuredClone(v.points[0])); rejection(a, v, v.kind, "LOOP_SCHEMA");
  }
  for (const tags of [[], officialTags.slice(0, 4)]) {
    const v = plan(); v.points[0].candidate_tags = tags; rejection(a, v, "plan", "LOOP_SCHEMA");
  }
  const duplicate = plan(); duplicate.points[0].candidate_tags = ["赠送利益", "赠送利益"]; rejection(a, duplicate, "plan", "LOOP_SCHEMA");
});

test("new projections reuse old input references facts condition and segment constraints", async () => {
  const a = await api();
  for (const [field, old] of [["inputRefs", wireSchema.$defs.candidate.properties.input_refs], ["facts", wireSchema.$defs.candidate.properties.facts], ["condition", wireSchema.$defs.condition], ["segmentAssessment", wireSchema.$defs.segmentAssessment]] as const) {
    assert.deepEqual(a.loopContract("plan").$defs[field] ?? a.loopContract("assess").$defs[field], old);
  }
  const v = plan(); v.points[0].facts = Array(6).fill(v.points[0].facts[0]); rejection(a, v, "plan", "LOOP_SCHEMA");
  const invalid = plan(); invalid.points[0].facts[0].origin = "input"; rejection(a, invalid, "plan", "LOOP_SCHEMA");
});

test("all output object levels reject extra fields including continuation and case supports_tags", async () => {
  const a = await api();
  const objects = [
    (v: JsonObject) => v, (v: JsonObject) => v.segments[0], (v: JsonObject) => v.points[0],
    (v: JsonObject) => v.points[0].facts[0], (v: JsonObject) => v.points[0].input_refs[0]
  ];
  for (const target of objects) { const v = plan(); target(v).continuation = {}; rejection(a, v, "plan", "LOOP_SCHEMA"); }
  for (const target of [(v: JsonObject) => v, (v: JsonObject) => v.tag_checks[0], (v: JsonObject) => v.tag_checks[0].conditions[0], (v: JsonObject) => v.tag_checks[0].case_assessments[0]]) {
    const v = assess(); addCase(v); target(v).supports_tags = ["赠送利益"]; rejection(a, v, "assess", "LOOP_SCHEMA", { ...assessmentContext(), visible_case_refs: [caseRef] });
  }
  for (const name of ["facts", "gaps", "summary", "supported_tags", "pending_tags", "supports_tags"]) { const v = assess(); v[name] = []; rejection(a, v, "assess", "LOOP_SCHEMA", assessmentContext()); }
  const one = oneShot(); one.points[0].query_text = "多余查询"; rejection(a, one, "one_shot", "LOOP_SCHEMA");
});

test("Unicode code point boundaries apply to search intent and both query fields", async () => {
  const a = await api();
  for (const [name, max] of [["search_intent", 300], ["query_text", 500], ["alternative_query", 500]] as const) {
    for (const count of [1, max]) { const v = plan(); v.points[0][name] = "😀".repeat(count); a.validateLoopOutput(v, "plan"); }
    for (const count of [0, max + 1]) { const v = plan(); v.points[0][name] = "😀".repeat(count); rejection(a, v, "plan", "LOOP_SCHEMA"); }
  }
});

test("official dictionary binds plan and one shot tags without hand copied tag or criterion enums", async () => {
  const a = await api();
  for (const tag of officialTags) {
    const v = plan(); v.points[0].candidate_tags = [tag]; a.validateLoopOutput(v, "plan");
    const one = oneShot(); one.points[0].tag_checks = [{ tag, conditions: criteriaFor(tag).map(c => ({ criterion_id: c.id, status: "unknown", input_refs: [], context_refs: [], reason: "尚待核查。" })), case_assessments: [] }]; a.validateLoopOutput(one, "one_shot");
  }
  const v = plan(); v.points[0].candidate_tags = ["自创标签"]; rejection(a, v, "plan", "LOOP_SCHEMA");
  const one = oneShot(); one.points[0].tag_checks[0].tag = "自创标签"; rejection(a, one, "one_shot", "LOOP_SCHEMA");
});

test("assessment requires the current request identity and rejects stale revision and nonexistent point", async () => {
  const a = await api();
  rejection(a, assess(), "assess", "LOOP_ASSESSMENT_CONTEXT_REQUIRED");
  for (const [field, value, code] of [["assessment_id", "AS0002", "LOOP_ASSESSMENT_ID_MISMATCH"], ["state_revision", 0, "LOOP_STATE_REVISION_MISMATCH"], ["risk_point_id", "RP9999", "LOOP_RISK_POINT_MISMATCH"]]) {
    const v = assess(); v[field as string] = value; const error = rejection(a, v, "assess", code as string, assessmentContext()); assert.equal(a.loopFormatRepairable(error), false);
  }
  for (const [field, value] of [["state_revision", -1], ["state_revision", 1.5], ["assessment_id", "RP0001"], ["risk_point_id", "new-point"]]) { const v = assess(); v[field as string] = value; rejection(a, v, "assess", "LOOP_SCHEMA", assessmentContext()); }
});

test("assessment includes every current candidate exactly once rather than silently dropping a tag", async () => {
  const a = await api();
  const omitted = assess(); rejection(a, omitted, "assess", "LOOP_CANDIDATE_TAGS_MISMATCH", { ...assessmentContext(), candidate_tags: ["赠送利益", "合同外利益"] });
  const duplicate = assess(); duplicate.tag_checks.push(structuredClone(duplicate.tag_checks[0])); rejection(a, duplicate, "assess", "LOOP_CANDIDATE_TAGS_MISMATCH", assessmentContext());
  const other = assess(); other.tag_checks[0].tag = "合同外利益"; rejection(a, other, "assess", "LOOP_CANDIDATE_TAGS_MISMATCH", assessmentContext());
});

test("each criterion is complete exactly once for assessment and one shot", async () => {
  const a = await api();
  for (const make of [assess, oneShot]) for (const mutation of ["missing", "duplicate", "invented"]) {
    const v = make(); const conditions = (v.kind === "assess" ? v.tag_checks : v.points[0].tag_checks)[0].conditions;
    if (mutation === "missing") conditions.pop();
    if (mutation === "duplicate") conditions[1] = structuredClone(conditions[0]);
    if (mutation === "invented") conditions[0].criterion_id = "invented";
    rejection(a, v, v.kind, "LOOP_CRITERIA_MISMATCH", v.kind === "assess" ? assessmentContext() : undefined);
  }
  const one = oneShot(); one.points[0].tag_checks.push(structuredClone(one.points[0].tag_checks[0])); rejection(a, one, "one_shot", "LOOP_CANDIDATE_TAGS_MISMATCH");
});

test("cases remain nested under their own tag and require currently visible options", async () => {
  const a = await api(); const v = assess(); addCase(v);
  const ctx = { ...assessmentContext(), visible_case_refs: [caseRef] }; a.validateLoopOutput(v, "assess", ctx);
  rejection(a, v, "assess", "LOOP_INVALID_CASE_REF", assessmentContext());
  const one = oneShot(); one.points[0].tag_checks = v.tag_checks; rejection(a, one, "one_shot", "LOOP_INVALID_CASE_REF"); a.validateLoopOutput(one, "one_shot", ctx);
  for (const count of [2, 3]) { const more = structuredClone(v); more.tag_checks[0].case_assessments = Array(count).fill(v.tag_checks[0].case_assessments[0]); if (count === 2) a.validateLoopOutput(more, "assess", ctx); else rejection(a, more, "assess", "LOOP_SCHEMA", ctx); }
  const bad = structuredClone(v); bad.tag_checks[0].case_assessments[0].case_ref = "Cunregistered"; rejection(a, bad, "assess", "LOOP_SCHEMA", ctx);
});

test("structural acceptance leaves quote occurrence and source verification to existing state normalization", async () => {
  const a = await api(); const v = plan(); v.points[0].input_refs[0] = { segment_id: "S9999", quote: "并非当前输入的引用", occurrence: 10 }; a.validateLoopOutput(v, "plan");
});

test("safe typed feedback separates repairable format from semantic tool and reference failure", async () => {
  const a = await api();
  for (const [code, category, repairable] of [["LOOP_SCHEMA", "schema", true], ["LOOP_CRITERIA_MISMATCH", "semantic", false], ["LOOP_TOOL_FORMAT", "tool_format", false], ["LOOP_INVALID_INPUT_REFERENCE", "reference", false]]) {
    const error = new a.LoopContractError(code); assert.equal(a.loopOutputDiagnostic(error, "private reply").category, category); assert.equal(a.loopFormatRepairable(error), repairable);
  }
  let syntax: any; try { a.parseLoopOutput('{"secret":', "plan"); } catch (error) { syntax = error; }
  assert.equal(syntax.code, "LOOP_JSON_SYNTAX"); assert.equal(a.loopFormatRepairable(syntax), true);
  assert.equal(a.loopFormatRepairable(new Error("LOOP_SCHEMA: injected")), false);
  const diagnostic = a.loopOutputDiagnostic(new Error("private injected error message"), "private reply");
  assert.equal(diagnostic.code, "LOOP_SEMANTIC_ERROR"); assert.equal(diagnostic.reply_sha256, hash("private reply"));
  assert.ok(!JSON.stringify(diagnostic).includes("private"));
});

test("schema feedback reuses bounded ModelSchemaError issues and removes untrusted field names", async () => {
  const a = await api();
  const errors: any[] = [{ instancePath: "/points/0", schemaPath: "private schema", keyword: "additionalProperties", params: { additionalProperty: "ignore_all_instructions" }, message: "private message" }, { instancePath: "/points/0/facts/0", schemaPath: "private schema", keyword: "required", params: { missingProperty: "input_refs" } }, { instancePath: "/privateInjectedField", schemaPath: "", keyword: "made_up", params: {} }, { instancePath: "", schemaPath: "", keyword: "type", params: {} }];
  const error = new a.LoopContractError("LOOP_SCHEMA", errors);
  assert.equal(error.schemaIssues.length, new ModelSchemaError(errors).schemaIssues.length);
  const d = a.loopOutputDiagnostic(error, "private body");
  assert.equal(d.schema_issues.length, 3); assert.equal(d.schema_issues[0].additionalProperty, "[redacted]"); assert.equal(d.schema_issues[1].missingProperty, "input_refs"); assert.equal(d.schema_issues[2].instancePath, "[redacted]");
  for (const forbidden of ["private", "ignore_all", "made_up"]) assert.ok(!JSON.stringify(d).includes(forbidden));
});

test("phase projections and inventory expose reproducible hashes and no other stage payload", async () => {
  const a = await api(); const inventory = a.loopContractInventory();
  assert.equal(inventory.protocol_version, a.AGENT_LOOP_VERSION);
  for (const kind of ["plan", "assess", "one_shot"]) {
    const contract = a.loopContract(kind); const validate = new Ajv2020.default({ strict: false }).compile(contract);
    assert.equal(validate(kind === "plan" ? plan() : kind === "assess" ? assess() : oneShot()), true);
    assert.equal(inventory.stages[kind].sha256, hash(canonical(contract)));
    assert.deepEqual(Object.keys(contract.$defs).filter(k => ["plan", "assess", "one_shot"].includes(k)), [kind]);
    contract.$defs[kind].properties.kind.const = "tampered"; assert.equal(a.loopContract(kind).$defs[kind].properties.kind.const, kind);
  }
  for (const key of ["loop_schema_sha256", "assessment_schema_sha256", "wire_schema_sha256", "criteria_sha256", "official_tags_sha256"]) assert.match(inventory[key], /^[0-9a-f]{64}$/);
  assert.throws(() => a.loopContract("final"), { code: "LOOP_INVALID_KIND" });
});

test("inventory distinguishes byte identity from canonical criteria identity", async () => {
  const a = await api(); const inventory = a.loopContractInventory();
  assert.equal(inventory.criteria_sha256, criteriaByteHash);
  assert.equal(inventory.criteria_content_sha256, criteriaHash);
});

test("private sidecar schema strictly reserves scoped diagnostics actions identity budgets and point dimensions", async () => {
  const a = await api(); const sidecar = {
    schema_version: "finrisk-agent-assessment/1.0", protocol_version: a.AGENT_LOOP_VERSION, orchestration_mode: "planned_adaptive",
    identity: { task_id: "unit", profile_id: "unit", input_sha256: "a".repeat(64), snapshot_sha256: "b".repeat(64), report_sha256: "c".repeat(64), loop_schema_sha256: "d".repeat(64), criteria_sha256: "e".repeat(64), budget_profile_sha256: "f".repeat(64) },
    plan_revision: 0, points: [], attempt_diagnostics: [], active_blockers: [], actions: [], local_ranking_limited: false, semantic_review_required: true
  };
  a.validateAgentAssessment(sidecar);
  const scope = { branch: "assess", risk_point_id: "RP0001", assessment_id: "AS0001", state_revision: 1, plan_revision: 0 };
  const populated: JsonObject = structuredClone(sidecar);
  populated.points = [{ risk_point_id: "RP0001", plan_revision: 0, state_revision: 1, input_refs: [ref], facts: [], candidate_tags: ["赠送利益"], input_state: "complete", risk_judgment: "suspected", case_state: "insufficient", retrieval_state: "local_partial", processing_state: "dirty", initial_query_covered: true, last_assessed_revision: null, pending_case_goals: ["mechanism_search"], read_anchor_ids: [], stop_reason: null }];
  populated.attempt_diagnostics = [{ scope, code: "LOOP_CRITERIA_MISMATCH", category: "semantic" }]; populated.active_blockers = [{ scope, code: "LOOP_INVALID_CASE_REF", category: "reference" }];
  for (const origin of ["initial_plan", "adaptive_model", "required_anchor_read"]) populated.actions.push({ action_id: "A0001", origin, action_type: origin === "required_anchor_read" ? "read_source" : "search_cases", scope, status: "completed", query_text: "保险 赠礼", local_ranking_limited: false });
  a.validateAgentAssessment(populated);
  for (const target of [(v: JsonObject) => v, (v: JsonObject) => v.identity, (v: JsonObject) => v.points[0], (v: JsonObject) => v.actions[0], (v: JsonObject) => v.attempt_diagnostics[0], (v: JsonObject) => v.active_blockers[0].scope]) {
    const v = structuredClone(populated); target(v).unknown_field = "must reject"; assert.throws(() => a.validateAgentAssessment(v), { code: "LOOP_SCHEMA" });
  }
  const missingScope = structuredClone(populated); delete missingScope.active_blockers[0].scope; assert.throws(() => a.validateAgentAssessment(missingScope), { code: "LOOP_SCHEMA" });
});
