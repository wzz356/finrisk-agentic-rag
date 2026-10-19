import test, { after } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import { canonical, defaults, hash, root, type JsonObject, type TaskInput } from "../src/contracts.js";
import { buildContext, buildCountedContext } from "../src/context_builder.js";
import { TaskState } from "../src/state.js";
import { normalizeModelOutput, stageProtocol, wireContract } from "../src/model_protocol.js";
import { evaluationCli } from "../src/evaluation_cli.js";
import { evaluationConfig, evaluationProtocol, evaluationProtocolHash } from "../src/evaluation.js";
import { runTask } from "../src/runner.js";
import { buildProviderPayload } from "../src/prompt_counter.js";
import { offlineCounter } from "./offline_counter.js";
const local = offlineCounter(); after(() => local.close());
process.env.FINRISK_AGENT_ALLOW_EVALUATION = "1";
const text = "购买这份保险就送购物卡。", source = "处罚：机构购买保险向客户赠送购物卡，并以赠礼诱导投保。";
const ref = { segment_id: "S0001", quote: text, occurrence: 1 };
const segments = [{ segment_id: "S0001", status: "reviewed", reason: "审阅当前投保赠礼表述。" }];
function task(arm: "R1" | "R2" | "A" = "R1"): TaskInput {
  const definitions = JSON.parse(readFileSync(root + "/contracts/agent_rag/risk-definitions.json", "utf8"));
  const user_context = { purpose_hint: "当前保险营销宣传" };
  return { input: { task_id: "cite", source_id: "INPUT:cite", text, text_sha256: hash(text), text_version: "1", input_sha256: hash(canonical({ text, user_context })), user_context, segments: [{ segment_id: "S0001", source_id: "INPUT:cite", start: 0, end: Array.from(text).length, text }] },
    snapshot: { snapshot_id: "SNAPSHOT", dictionary_sha256: definitions[0].dictionary_sha256 }, definitions,
    config: evaluationConfig(arm, { retrieval_profile: "v2-hybrid-flash-v1", max_queries_per_search: 1 }) };
}
function data(sourceText = source) {
  const span = (id: string, start: number, end: number, extra = {}) => ({ evidence_id: id, source_id: "SRC", source_kind: "penalty", text_version: "1", text_sha256: hash(sourceText), quote: Array.from(sourceText).slice(start, end).join(""), start, end, source_verified: true, context_has_more: false, ...extra });
  const anchors = [[0,3],[3,8],[8,12],[12,20],[20,Array.from(sourceText).length]].map(([a,b], i) => span("E" + i, a!, b!));
  const candidate = { source_id: "SRC", file_id: "FILE", unit_id: "UNIT", rank: 1, required_evidence_ids: anchors.map(e => e.evidence_id), evidence: anchors };
  const windows = [span("W1",0,20,{ was_read: true }), span("W2",20,Array.from(sourceText).length,{ was_read: true })];
  return { candidate, windows };
}
function bridge(sourceText = source): any {
  const { candidate, windows } = data(sourceText); const calls: JsonObject[] = [];
  return { calls, countTokens: local.count,
    async previewRead(args: JsonObject) { return { schema_version: "finrisk-tools/1.0", status: "ok", data: { evidence: windows[args.anchor_evidence_id === "E4" ? 1 : 0] }, meta: { snapshot_id: "SNAPSHOT", backend_query_attempts: 0, cache_hits: 0, external_retrieval_requests: 0 }, error: null }; },
    async call(name: string, args: JsonObject) {
      calls.push({ name, args });
      return { schema_version: "finrisk-tools/1.0", status: "ok", data: name === "read_source" ? { evidence: windows[args.anchor_evidence_id === "E4" ? 1 : 0] } : { queries: args.queries.map((q: JsonObject) => ({ ...q, status: "ok", complete: true, candidates: [candidate] })) }, meta: { snapshot_id: "SNAPSHOT", backend_query_attempts: name === "search_cases" ? 1 : 0, cache_hits: 0, external_retrieval_requests: 0 }, error: null };
    }, async verifyEvidence(ids: string[]) { return { valid_ids: ids, rejected_ids: [] }; } };
}
function envelope(): JsonObject {
  return { kind: "evaluation_first", segments, points: [{ input_refs: [ref], facts: [{ field: "insurance_product", value: "yes", origin: "input", input_refs: [ref], context_refs: [] }, { field: "material_function", value: "direct_promotion", origin: "user_context", input_refs: [], context_refs: ["$.user_context.purpose_hint"] }], tag_checks: [{ tag: "赠送利益", conditions: ["actual_conduct","insurance_product","gift","purchase_link"].map(criterion_id => ({ criterion_id, status: "met", input_refs: [ref], context_refs: [], reason: "当前输入依据。" })) }], case_matches: [] }], continuation: { gaps: [] } };
}
function final(value = envelope()): JsonObject { return { kind: "final", segments, points: value.points.map((p: JsonObject) => ({ risk_point_id: "RP0001", tag_checks: p.tag_checks, case_matches: p.case_matches })) }; }
function analysis(): JsonObject { const p = envelope().points[0]; return { kind: "analysis", segments, points: [{ input_refs: p.input_refs, facts: p.facts, candidate_tags: ["赠送利益"] }] }; }
function select(case_ref: string): JsonObject { return { case_ref, supports_tags: ["赠送利益"], same_points: ["购买保险附赠利益"], differences: ["主体不同"], match_status: "reliable_match" }; }
function ledger(messages: AgentMessage[]): JsonObject {
  const m = messages.find(m => m.role === "system" && typeof m.content === "string" && m.content.startsWith("<task_state>"));
  assert.ok(m && typeof m.content === "string"); return JSON.parse(m.content.slice(12, -13));
}
function preparedState(): { state: TaskState; task: TaskInput; messages: AgentMessage[] } {
  const t = task(); const state = new TaskState(t.input); Object.assign(state, { evaluationCitations: true, citationSnapshotId: t.snapshot.snapshot_id });
  state.protocolVersion = "finrisk-model-wire/1.2"; state.phase = "finalizing";
  const d = data(); d.candidate.evidence.forEach(e => state.evidence.set(e.evidence_id, e)); d.windows.forEach(e => state.evidence.set(e.evidence_id,e));
  state.searches.push({ focus_refs: [ref], risk_point_ids: ["RP0001"], goal: "mechanism_search", status: "ok", complete: true, candidates: [d.candidate] });
  state.acceptAnalysis(normalizeModelOutput(analysis(),"analysis",state,t)); state.phase = "finalizing";
  return { state, task: t, messages: [{ role: "system", content: "规则", timestamp: 0 }, { role: "user", content: text, timestamp: 0 }] };
}

test("R1 and A compact first payloads are identical and one judgement expands five anchors without repair", async () => {
  const payloads: JsonObject[] = [];
  for (const arm of ["R1","A"] as const) {
    const t = task(arm); t.input.task_id = "random-" + arm; t.input.source_id = "INPUT:" + arm; t.input.segments[0]!.source_id = t.input.source_id;
    const b = bridge();
    const result = await runTask(t,b,[(context) => {
      payloads.push(buildProviderPayload(context, "analyzing", false)); const options = ledger(context.messages).citation_options;
      assert.equal(options.length, 1); const v = envelope(); v.points[0].case_matches = [select(options[0].case_ref)]; return v;
    }]);
    assert.equal(result.report.task_status, "completed", JSON.stringify(result.report.validation));
    assert.equal(result.report.resources.model_request_units,1); assert.equal(result.report.resources.json_repairs,0);
    assert.equal(result.report.versions.prompt_sha256,hash(readFileSync(root + "/configs/agent_rag/evaluation-system-prompt.txt","utf8")));
    assert.deepEqual(result.report.risk_points[0].case_matches[0].evidence_ids,["W1","W2"]);
    assert.equal(result.report.risk_points[0].case_matches[0].unit_id,"UNIT");
    assert.equal(b.calls.filter((c: JsonObject) => c.name === "read_source").length,2);
  }
  assert.deepEqual(payloads[0],payloads[1]);
});

test("R2 final schema and actual model output use compact references", async () => {
  const b = bridge(); const result = await runTask(task("R2"),b,[analysis(),context => {
    const options = ledger(context.messages).citation_options; const v = final(); v.points[0].case_matches = [select(options[0].case_ref)]; return v;
  }]);
  assert.equal(result.report.task_status,"completed",JSON.stringify({ validation: result.report.validation, calls: b.calls, events: result.events.filter(e => e.type.startsWith("retrieval.")) }));
  assert.deepEqual(result.report.risk_points[0].case_matches[0].evidence_ids,["W1","W2"]);
  assert.equal(result.report.versions.prompt_sha256,hash(readFileSync(root + "/configs/agent_rag/evaluation-system-prompt.txt","utf8")));
});

test("registry rebuild retains old visible refs across A searches and excludes new unread candidates", () => {
  const { state, task: t, messages } = preparedState();
  const first = ledger(buildContext(messages,state,{...defaults,...t.config})); assert.equal(first.citation_options.length,1);
  const oldRef = first.citation_options[0].case_ref; const newer = data().candidate;
  newer.unit_id = "NEW"; newer.source_id = "OTHER"; newer.evidence = newer.evidence.map(e => ({ ...e, source_id: "OTHER" }));
  state.searches.push({ focus_refs: [ref], risk_point_ids: ["RP0001"], status: "ok", complete: true, candidates: [newer] });
  newer.evidence.forEach(e => state.evidence.set("OTHER"+e.evidence_id,{...e,evidence_id:"OTHER"+e.evidence_id}));
  const next = ledger(buildContext(messages,state,{...defaults,...t.config}));
  assert.deepEqual(next.citation_options.map((o: JsonObject) => o.case_ref),[oldRef]);
  const v = final(); v.points[0].case_matches = [select(oldRef)];
  assert.equal(normalizeModelOutput(v,"final",state,t).risk_points[0].case_matches[0].unit_id,"UNIT");
});

test("final context hides original anchor IDs but expands visible read IDs and revokes options when quote disappears", async () => {
  const { state, task: t, messages } = preparedState();
  const full = ledger(buildContext(messages,state,{...defaults,...t.config}));
  assert.deepEqual(full.evidence.map((e: JsonObject) => e.evidence_id),["W1","W2"]);
  const caseRef = full.citation_options[0].case_ref;
  state.evidence.get("W2")!.was_read = false;
  await buildCountedContext(messages,state,{...defaults,...t.config},[], async m => ({ prompt_tokens: ledger(m).citation_options.length ? 20000 : 10 }));
  const v = final(); v.points[0].case_matches = [select(caseRef)];
  assert.throws(() => normalizeModelOutput(v,"final",state,t), /CITATION_REFERENCE_NOT_VISIBLE/);
});

test("compact schemas/instructions/prompt hash agree and default final contract remains unchanged", async () => {
  const { state, task: t } = preparedState(); const describe: any = await evaluationCli(["--describe"]);
  assert.equal(describe.schema_version,"finrisk-evaluation/2.4");
  assert.equal(describe.system_prompt_sha256,hash(readFileSync(root + "/configs/agent_rag/evaluation-system-prompt.txt","utf8")));
  assert.ok(!Object.hasOwn(describe.envelope_schema.$defs.caseMatch.properties,"source_id"));
  assert.ok(Object.hasOwn(describe.envelope_schema.$defs.caseMatch.properties,"case_ref"));
  assert.ok(stageProtocol(t,state,"final").includes('"case_ref"'));
  assert.ok(!evaluationProtocol(t).includes("V2须选实际unit_id"));
  assert.ok(Object.hasOwn(wireContract("final").$defs.caseMatch.properties,"source_id"));
  assert.equal(describe.evaluation_protocol_sha256,evaluationProtocolHash);
});

test("missing frozen snapshot rejects V2 evaluation before model or tools", async () => {
  const t = task(); delete t.snapshot.snapshot_id; const b = bridge();
  await assert.rejects(() => runTask(t,b,[envelope()]), /CITATION_SNAPSHOT_REQUIRED/); assert.equal(b.calls.length,0);
});

test("unknown current contract remains unknown even when compact selection is valid", () => {
  const { state, task: t, messages } = preparedState(); const data = ledger(buildContext(messages,state,{...defaults,...t.config}));
  const v = final(); v.points[0].case_matches = [select(data.citation_options[0].case_ref)];
  v.points[0].tag_checks.push({ tag: "合同外利益", conditions: ["actual_conduct","insurance_product","benefit","contract_outside"].map(criterion_id => ({ criterion_id, status: "met", input_refs: [ref], context_refs: [], reason: "不得通过案例强化合同事实。" })) });
  const normalized = normalizeModelOutput(v,"final",state,t); const contract = normalized.risk_points[0].tag_checks.find((t: JsonObject) => t.tag === "合同外利益");
  assert.equal(contract.conditions.find((c: JsonObject) => c.condition_id === "contract_outside").status,"unknown");
});

test("A continuation final still accepts the old visible case_ref after another search", async () => {
  let oldRef = ""; const b = bridge();
  const result = await runTask(task("A"),b,[
    context => { const v = envelope(); oldRef = ledger(context.messages).citation_options[0].case_ref; v.points[0].case_matches = [{ ...select(oldRef), match_status: "partial_match" }]; v.continuation.gaps = [{ type: "case_comparison", risk_point_id: "RP0001", reason: "补充适用差异。" }]; return v; },
    [{ type: "toolCall", id: "NEXT", name: "search_cases", arguments: { queries: [{ text: "赠礼适用差异", focus_refs: [ref], risk_point_ids: ["RP0001"], goal: "counter_context" }], top_k: 3, filters: {} } }],
    { kind: "finalize" },
    context => { assert.ok(ledger(context.messages).citation_options.some((o: JsonObject) => o.case_ref === oldRef)); const v = final(); v.points[0].case_matches = [select(oldRef)]; return v; },
  ]);
  assert.equal(result.report.task_status,"completed",JSON.stringify(result.report.validation));
  assert.equal(result.report.resources.model_request_units,4);
  assert.equal(result.report.resources.json_repairs,0);
  assert.deepEqual(result.report.risk_points[0].case_matches[0].evidence_ids,["W1","W2"]);
});

test("fabricated references and source/unit/citation overrides cannot qualify as completed", async () => {
  for (const override of [{ case_ref: "C" + "f".repeat(20) }, { source_id: "OTHER" }, { unit_id: "OTHER" }, { evidence_ids: ["E0"] }, { quote: "伪造处罚原文" }, { unknown: true }]) {
    const result = await runTask(task(),bridge(),[context => {
      const v = envelope(); const o = ledger(context.messages).citation_options[0]; v.points[0].case_matches = [{ ...select(o.case_ref), ...override }]; return v;
    }]);
    assert.equal(result.report.task_status,"partial");
    assert.equal(result.report.resources.model_request_units,1);
    assert.equal(result.report.resources.json_repairs,0);
  }
});

test("incomplete read candidate with empty matches never masquerades as normal empty retrieval", async () => {
  const b = bridge(), original = b.call.bind(b);
  b.call = async (name: string, args: JsonObject) => {
    const r = await original(name,args);
    if (name === "read_source" && args.anchor_evidence_id === "E4") return { ...r, status: "error", data: null, error: { code: "UNREAD", retryable: false } };
    return r;
  };
  const result = await runTask(task(),b,[context => { assert.deepEqual(ledger(context.messages).citation_options,[]); return envelope(); }]);
  assert.equal(result.report.task_status,"partial");
  assert.equal(result.report.risk_points[0].case_match_status,"not_completed");
  assert.notEqual(result.report.stop.reason,"normal_empty");
});

test("nonexperimental V2 keeps the original wire and default system prompt", async () => {
  const t = task("R2"); delete t.config!.evaluation_arm;
  const b = bridge();
  const result = await runTask(t,b,[analysis(), context => {
    const l = ledger(context.messages); assert.equal(l.citation_options,undefined);
    const system = context.messages.filter(m => m.role === "system").map(m => m.content).join("\n");
    assert.ok(system.includes("case_matches只引用本轮已呈现原话的source_id及evidence_id"));
    const v = final(); v.points[0].case_matches = [{ source_id: "SRC", unit_id: "UNIT", evidence_ids: ["W1","W2"], same_points: ["投保赠礼"], differences: ["主体不同"], supports_tags: ["赠送利益"], match_status: "reliable_match" }]; return v;
  }]);
  assert.equal(result.report.task_status,"completed",JSON.stringify(result.report.validation));
});

test("V2 experiment mock output cannot bypass compact validation through legacy full-report fallback", () => {
  const { state, task: t } = preparedState();
  const full = normalizeModelOutput(final(),"final",state,t);
  assert.throws(() => normalizeModelOutput(full,"final",state,t), /MODEL_WIRE_REQUIRED/);
});

test("evaluation bundle read status requires source identity and actual quoted coverage", async () => {
  const { candidateBundleRead } = await import("../src/model_protocol.js");
  for (const mutate of [
    (w: JsonObject) => { w.text_version = "OTHER"; },
    (w: JsonObject) => { w.text_sha256 = "OTHER"; },
    (w: JsonObject) => { w.quote = "伪".repeat(w.end-w.start); },
  ]) {
    const { state, task: t, messages } = preparedState(); mutate(state.evidence.get("W1")!);
    buildContext(messages,state,{...defaults,...t.config});
    assert.equal(candidateBundleRead(state.searches[0]!.candidates[0],state,true),false);
  }
});

test("live V2 experiment requires a trial ledger regardless of accepted model profile pairing", async () => {
  const { wireSchemaHash } = await import("../src/model_protocol.js");
  const { criteriaByteHash } = await import("../src/criteria.js");
  const { admissionPolicyHash } = await import("../src/request_admission.js");
  const { toolSchemasForProfile } = await import("../src/contracts.js");
  const keys = ["FINRISK_AGENT_ALLOW_LIVE","FINRISK_AGENT_TRIAL_LEDGER","DEEPSEEK_API_KEY","DEEPSEEK_API_KEY_FILE"];
  const saved = new Map(keys.map(k => [k,process.env[k]]));
  const oldFetch = globalThis.fetch; let fetches = 0;
  process.env.FINRISK_AGENT_ALLOW_LIVE = "1"; delete process.env.FINRISK_AGENT_TRIAL_LEDGER;
  process.env.DEEPSEEK_API_KEY = "INVALID_OFFLINE_DUMMY"; delete process.env.DEEPSEEK_API_KEY_FILE;
  globalThis.fetch = async () => { fetches++; throw new Error("NO_EXTERNAL_EGRESS"); };
  try {
    const t = task(); Object.assign(t.config!,{ execution_mode: "live", profile_id: "live-deepseek-flash-bm25-v1", request_admission_policy: "v2-v1" });
    const tools = toolSchemasForProfile("v2-hybrid-flash-v1"); tools.search_cases.properties.queries.maxItems=1;
    t.execution_contract = { model_wire_sha256: wireSchemaHash, criteria_sha256: criteriaByteHash,
      prompt_sha256: hash(readFileSync(root + "/configs/agent_rag/evaluation-system-prompt.txt","utf8")),
      tool_contract_sha256: hash(canonical(tools)), request_admission_policy_sha256: admissionPolicyHash("v2-v1"), evaluation_protocol_sha256: evaluationProtocolHash };
    const b = bridge(); await assert.rejects(() => runTask(t,b),/MODEL_TRIAL_LEDGER_REQUIRED/);
    assert.equal(b.calls.length,0); assert.equal(fetches,0);
  } finally {
    globalThis.fetch=oldFetch;
    for (const key of keys) { const v=saved.get(key); if(v === undefined) delete process.env[key]; else process.env[key]=v; }
  }
});

const incompleteChecklistCases = ["empty_tags", "missing_candidate_tag", "missing_criterion", "empty_conditions"] as const;
function omitChecklist(value: JsonObject, mode: typeof incompleteChecklistCases[number]) {
  if (mode === "empty_tags") value.points[0].tag_checks = [];
  else if (mode === "missing_criterion") value.points[0].tag_checks[0].conditions.pop();
  else if (mode === "empty_conditions") value.points[0].tag_checks[0].conditions = [];
}
for (const mode of incompleteChecklistCases) {
  test("compact final normalizer rejects incomplete checklist: " + mode, () => {
    const { state, task: t } = preparedState();
    if (mode === "missing_candidate_tag") state.points[0]!.candidate_tags.push("诱导投保");
    const value = final(); omitChecklist(value,mode);
    assert.throws(() => normalizeModelOutput(value,"final",state,t), /EVALUATION_CRITERIA_INCOMPLETE|INVALID_WIRE/);
  });
}
for (const arm of ["R2","A"] as const) for (const mode of incompleteChecklistCases) {
  test(arm + " final cannot complete an incomplete checklist: " + mode, async () => {
    const { criteriaFor } = await import("../src/criteria.js");
    const t = task(arm); t.config!.max_json_repairs_per_task = 0;
    const initial = envelope();
    if (mode === "missing_candidate_tag") initial.points[0].tag_checks.push({ tag: "诱导投保", conditions: criteriaFor("诱导投保").map(c => ({ criterion_id: c.id, status: "met", input_refs: [ref], context_refs: [], reason: "当前材料逐项评估。" })) });
    const a = analysis();
    if (mode === "missing_candidate_tag") a.points[0].candidate_tags.push("诱导投保");
    const rejectFinal = (context: any) => {
      const value = final(); value.points[0].case_matches = [select(ledger(context.messages).citation_options[0].case_ref)];
      omitChecklist(value,mode); return value;
    };
    const script = arm === "R2" ? [a,rejectFinal] : [
      (context: any) => { const first = structuredClone(initial); first.points[0].case_matches = [{ ...select(ledger(context.messages).citation_options[0].case_ref), match_status: "partial_match" }]; first.continuation.gaps = [{ type: "case_comparison", risk_point_id: "RP0001", reason: "继续核对案例差异。" }]; return first; },
      { kind: "finalize" }, rejectFinal,
    ];
    const result = await runTask(t,bridge(),script);
    assert.equal(result.report.task_status,"partial",JSON.stringify(result.report.validation));
    assert.equal(result.report.stop.reason,"invalid_output");
    assert.equal(result.report.resources.model_request_units,arm === "R2" ? 2 : 3);
    assert.equal(result.report.resources.json_repairs,0);
    assert.ok(result.report.validation.errors.some((e: string) => /EVALUATION_CRITERIA_INCOMPLETE|INVALID_WIRE/.test(e)));
  });
}

function evidenceEnvelope(evidence: JsonObject[], search = false): any {
  return { schema_version: "finrisk-tools/1.0", status: "ok", data: search ? { queries: [{ focus_refs: [ref], risk_point_ids: ["RP0001"], status: "ok", complete: true, candidates: [{ source_id: evidence[0]!.source_id, evidence }] }] } : { evidence: evidence[0] }, meta: { snapshot_id: "SNAPSHOT", backend_query_attempts: search ? 1 : 0, cache_hits: 0, external_retrieval_requests: 0 }, error: null };
}
const identityMutations: [string,(row: JsonObject) => void][] = [
  ["source",(row) => { row.source_id = "OTHER"; }],
  ["version",(row) => { row.text_version = "OTHER"; }],
  ["hash",(row) => { row.text_sha256 = "OTHER"; }],
  ["quote",(row) => { row.quote = "伪".repeat(Array.from(row.quote).length); }],
  ["start",(row) => { row.start++; }],
  ["end",(row) => { row.end++; }],
  ["kind",(row) => { row.source_kind = "input"; }],
  ["verification",(row) => { row.source_verified = false; }],
];
for (const [field,mutate] of identityMutations) {
  test("evaluation evidence ID rebinding fails closed for " + field, () => {
    const t = task(), state = new TaskState(t.input); state.evaluationCitations = true; state.citationSnapshotId = "SNAPSHOT";
    const old = data().windows[0]!; state.acceptTool("read_source",{},evidenceEnvelope([old]));
    const before = canonical(state.evidence.get(old.evidence_id)); const changed = structuredClone(old); mutate(changed);
    assert.throws(() => state.acceptTool("search_cases",{},evidenceEnvelope([changed],true),"CONFLICT"),/EVIDENCE_IDENTITY_CONFLICT/);
    assert.equal(canonical(state.evidence.get(old.evidence_id)),before);
    assert.equal(state.searches.length,0); assert.equal(state.appliedSearchActions.has("CONFLICT"),false);
  });
}
test("identity preflight preserves every prior entry and rejects intra-response aliases atomically", () => {
  const t = task(), state = new TaskState(t.input); state.evaluationCitations = true;
  const old = data().windows[0]!; state.acceptTool("read_source",{},evidenceEnvelope([old]));
  const changed = { ...old, quote: "伪".repeat(Array.from(old.quote).length) };
  assert.throws(() => state.acceptTool("search_cases",{},evidenceEnvelope([{ ...old,evidence_id:"NEW" },changed],true),"ATOMIC"),/EVIDENCE_IDENTITY_CONFLICT/);
  assert.equal(state.evidence.has("NEW"),false);
  const fresh = new TaskState(t.input); fresh.evaluationCitations = true;
  assert.throws(() => fresh.acceptTool("search_cases",{},evidenceEnvelope([old,changed],true),"DUPLICATE"),/EVIDENCE_IDENTITY_CONFLICT/);
  assert.equal(fresh.evidence.size,0);
});
test("same identity search revisits preserve read state and default behavior stays compatible", () => {
  for (const compact of [false,true]) {
    const t = task(), state = new TaskState(t.input); state.evaluationCitations = compact;
    const old = data().windows[0]!; state.acceptTool("read_source",{},evidenceEnvelope([old]));
    state.acceptTool("search_cases",{},evidenceEnvelope([{ ...old,was_read:false }],true),"SAME");
    assert.equal(state.evidence.get(old.evidence_id)!.was_read,true);
    if (!compact) {
      state.acceptTool("search_cases",{},evidenceEnvelope([{ ...old,source_id:"LEGACY" }],true),"LEGACY");
      assert.equal(state.evidence.get(old.evidence_id)!.source_id,"LEGACY");
    }
  }
});


test("explicit A format recovery cannot retry a fabricated case reference", async () => {
  const t = task("A"); t.config = { ...t.config, evaluation_format_repair_limit: 1 };
  const value = envelope(); value.points[0].case_matches = [select("C" + "f".repeat(20))];
  const result = await runTask(t, bridge(), [value, envelope()]);
  assert.equal(result.report.stop.reason, "invalid_output");
  assert.equal(result.report.resources.model_request_units, 1);
  assert.equal(result.report.resources.json_repairs, 0);
  assert.ok(result.report.validation.errors.includes("CITATION_REFERENCE_NOT_VISIBLE"));
});

test("complete report preserves literal line breaks CRLF and supplementary characters in penalty quotes", async () => {
  const raw = "处罚原文𠮷：违反第八\n条的规定。\r\n机构购买保险赠送购物卡🎁，并以赠礼诱导投保。";
  const expected = data(raw).windows;
  const result = await runTask(task(), bridge(raw), [context => {
    const current = ledger(context.messages);
    assert.equal(current.citation_options.length, 1);
    const value = envelope(); value.points[0].case_matches = [select(current.citation_options[0].case_ref)];
    return value;
  }]);
  assert.equal(result.report.task_status, "completed", JSON.stringify(result.report.validation));
  assert.equal(result.report.validation.references_valid, true);
  const match = result.report.risk_points[0].case_matches[0];
  assert.equal(match.source_id, "SRC"); assert.equal(match.unit_id, "UNIT");
  assert.deepEqual(match.evidence_ids, ["W1", "W2"]);
  assert.equal(match.evidence.map((e: JsonObject) => e.quote).join(""), raw);
  assert.ok(match.evidence[0].quote.includes("第八\n条"));
  assert.ok(match.evidence.map((e: JsonObject) => e.quote).join("").includes("\r\n"));
  assert.notEqual(raw.length, Array.from(raw).length);
  for (let i = 0; i < expected.length; i++) {
    const actual = match.evidence[i], original: JsonObject = expected[i]!;
    for (const field of ["evidence_id", "source_id", "source_kind", "text_version", "text_sha256", "start", "end", "quote", "source_verified"]) {
      assert.equal(actual[field], original[field], field + " must be backfilled from the read window");
    }
    assert.equal(actual.quote, Array.from(raw).slice(actual.start, actual.end).join(""));
    assert.equal(Array.from(actual.quote).length, actual.end - actual.start);
  }
  assert.equal(match.evidence.at(-1).end, Array.from(raw).length);
  assert.equal(result.report.resources.model_request_units, 1);
  assert.equal(result.report.resources.json_repairs, 0);
});
