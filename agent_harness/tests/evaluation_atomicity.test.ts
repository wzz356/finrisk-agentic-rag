import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { canonical, hash, root, type JsonObject, type TaskInput } from "../src/contracts.js";
import { acceptEvaluation } from "../src/evaluation.js";
import { buildCitationOptions } from "../src/citation_bundles.js";
import { TaskState } from "../src/state.js";

const text = "购买这份保险就送购物卡。";
const ref = { segment_id: "S0001", quote: text, occurrence: 1 };
const segments = [{ segment_id: "S0001", status: "reviewed", reason: "核查当前保险赠礼条件。" }];
function task(): TaskInput {
  const definitions = JSON.parse(readFileSync(root + "/contracts/agent_rag/risk-definitions.json", "utf8"));
  const user_context = { purpose_hint: "当前保险营销宣传" };
  return { input: { task_id: "atomic", source_id: "INPUT:atomic", text, text_sha256: hash(text), text_version: "1",
    input_sha256: hash(canonical({ text, user_context })), user_context,
    segments: [{ segment_id: "S0001", source_id: "INPUT:atomic", start: 0, end: Array.from(text).length, text }] },
    snapshot: { snapshot_id: "SNAPSHOT", dictionary_sha256: definitions[0].dictionary_sha256 }, definitions,
    config: { retrieval_profile: "v2-hybrid-flash-v1" } as any };
}
function envelope(): JsonObject {
  return { kind: "evaluation_first", segments, points: [{ input_refs: [ref],
    facts: [{ field: "insurance_product", value: "yes", origin: "input", input_refs: [ref], context_refs: [] },
      { field: "material_function", value: "direct_promotion", origin: "user_context", input_refs: [], context_refs: ["$.user_context.purpose_hint"] }],
    tag_checks: [{ tag: "赠送利益", conditions: ["actual_conduct", "insurance_product", "gift", "purchase_link"].map(criterion_id =>
      ({ criterion_id, status: "met", input_refs: [ref], context_refs: [], reason: "当前材料逐项支持。" })) }], case_matches: [] }],
    continuation: { gaps: [] } };
}
function addUnknownContract(value: JsonObject): void {
  value.points[0].tag_checks.push({ tag: "合同外利益", conditions: ["actual_conduct", "insurance_product", "benefit", "contract_outside"].map(criterion_id =>
    ({ criterion_id, status: criterion_id === "contract_outside" ? "unknown" : "met",
      input_refs: criterion_id === "contract_outside" ? [] : [ref], context_refs: [], reason: "当前合同约定未提供。" })) });
  value.continuation.gaps = [{ type: "input_fact", risk_point_id: "RP0001", reason: "当前合同约定未知。" }];
}
function prepared(compact: boolean, existing: boolean): { state: TaskState; task: TaskInput; selection: JsonObject } {
  const t = task(), state = new TaskState(t.input);
  state.evaluationCitations = compact; state.citationSnapshotId = t.snapshot.snapshot_id;
  const quote = "处罚原文：购买保险附赠购物卡。", end = Array.from(quote).length;
  const anchor = { evidence_id: "E1", source_id: "SRC", source_kind: "penalty", text_version: "1", text_sha256: hash(quote),
    start: 0, end, quote, source_verified: true };
  const window = { ...anchor, evidence_id: "W1", was_read: true };
  const candidate = { source_id: "SRC", unit_id: "UNIT", required_evidence_ids: ["E1"], evidence: [anchor] };
  state.evidence.set("E1", anchor); state.evidence.set("W1", window);
  state.contextEvidenceIds.add("W1");
  state.searches.push({ focus_refs: [ref], risk_point_ids: ["RP0001"], goal: "mechanism_search", text: "投保赠礼", complete: true, status: "ok", candidates: [candidate] });
  const option = buildCitationOptions("SNAPSHOT", [candidate], [window])[0]!;
  state.citationOptions.set(option.case_ref, option);
  const assessment = { supports_tags: ["赠送利益"], same_points: ["购买保险附赠利益"], differences: ["机构不同"], match_status: "reliable_match" };
  const selection = compact ? { case_ref: option.case_ref, ...assessment } : { source_id: "SRC", unit_id: "UNIT", evidence_ids: ["W1"], ...assessment };
  if (existing) {
    const prior = envelope(); addUnknownContract(prior); prior.points[0].case_matches = [selection];
    acceptEvaluation(prior, state, t);
    state.phase = "finalizing";
    state.gaps.set("G0099", { gap_id: "G0099", risk_point_id: "RP0001", type: "case_comparison", statement: "保留既有缺口。", status: "unresolved" });
    state.caseTagBindings.set("RP0099", [{ source_id: "OLD", evidence_ids: ["OLD"], supports_tags: ["赠送利益"] }]);
  }
  return { state, task: t, selection };
}
const businessKeys = ["phase", "analysis", "points", "final", "gaps", "caseTagBindings", "protocolVersion"] as const;
const readKeys = ["input", "evidence", "citationOptions", "contextEvidenceIds", "searches", "noProgress", "appliedSearchActions"] as const;
function snapshot(state: TaskState, keys: readonly (keyof TaskState)[]) {
  return keys.map(key => ({ key, identity: state[key], value: structuredClone(state[key]) }));
}
function unchanged(state: TaskState, before: ReturnType<typeof snapshot>): void {
  for (const { key, identity, value } of before) {
    assert.strictEqual(state[key], identity, String(key) + " identity changed");
    assert.deepEqual(state[key], value, String(key) + " content changed");
  }
}

const failures: [string, (value: JsonObject) => void, RegExp][] = [
  ["input gap without unknown", value => { value.continuation.gaps = [{ type: "input_fact", risk_point_id: "RP0001", reason: "全部条件明确时不得凭空声明缺事实。" }]; }, /EVALUATION_INPUT_GAP_WITHOUT_UNKNOWN/],
  ["unregistered gap", value => { addUnknownContract(value); value.continuation.gaps[0].risk_point_id = "RP9999"; }, /INVALID_EVALUATION_GAP_REFERENCE/],
  ["second point normalization", value => {
    value.points.push(structuredClone(value.points[0]));
    value.points[1].tag_checks[0].conditions[0].input_refs = [];
  }, /CONDITION_WITHOUT_CURRENT_SUPPORT/],
  ["second point citation", value => {
    value.points.push(structuredClone(value.points[0]));
    const match = value.points[1].case_matches[0];
    if (match.case_ref) match.case_ref = "C" + "f".repeat(20); else match.unit_id = "MISSING";
  }, /CITATION_REFERENCE_NOT_VISIBLE|CASE_UNIT_BINDING_MISMATCH/],
];
for (const compact of [false, true]) for (const existing of [false, true]) for (const [name, mutate, error] of failures) {
  test("rejected evaluation preserves " + (existing ? "existing" : "empty") + " state: " + (compact ? "compact " : "legacy ") + name, () => {
    const { state, task: t, selection } = prepared(compact, existing);
    const value = envelope(); value.points[0].case_matches = [selection]; mutate(value);
    const business = snapshot(state, businessKeys), reads = snapshot(state, readKeys), original = structuredClone(value);
    assert.throws(() => acceptEvaluation(value, state, t), error);
    unchanged(state, business); unchanged(state, reads);
    assert.deepEqual(value, original);
    assert.ok(state instanceof TaskState);
    assert.equal(state.resolve(ref).quote, text);
  });
}
for (const compact of [false, true]) {
  test("successful evaluation commits complete business state and preserves read registries: " + (compact ? "compact" : "legacy"), () => {
    const { state, task: t, selection } = prepared(compact, true);
    const before = snapshot(state, readKeys), oldAnalysis = state.analysis, oldFinal = state.final, oldGaps = structuredClone(state.gaps);
    const value = envelope(); addUnknownContract(value); value.points[0].case_matches = [selection];
    const continuation = acceptEvaluation(value, state, t);
    assert.strictEqual(continuation, value.continuation);
    assert.notStrictEqual(state.analysis, oldAnalysis); assert.notStrictEqual(state.final, oldFinal);
    assert.equal(state.phase, "reviewing"); assert.equal(state.protocolVersion, "finrisk-model-wire/1.2");
    assert.deepEqual(state.points.map(p => p.risk_point_id), ["RP0001"]);
    assert.deepEqual(state.analysis!.risk_points.map((p: JsonObject) => p.candidate_tags), [["赠送利益", "合同外利益"]]);
    assert.equal(state.final!.risk_points[0].tag_checks.find((t: JsonObject) => t.tag === "合同外利益").status, "unknown");
    assert.deepEqual(state.caseTagBindings.get("RP0001")![0]!.evidence_ids, ["W1"]);
    assert.deepEqual(state.caseTagBindings.get("RP0099"), [{ source_id: "OLD", evidence_ids: ["OLD"], supports_tags: ["赠送利益"] }]);
    for (const [id, gap] of oldGaps) assert.deepEqual(state.gaps.get(id), gap);
    assert.equal(state.gaps.size, oldGaps.size + 1);
    assert.equal([...state.gaps.values()].at(-1)!.type, "input_condition");
    unchanged(state, before);
  });
}
