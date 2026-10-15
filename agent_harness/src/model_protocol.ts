import { readFileSync } from "node:fs";
import Ajv2020 from "ajv/dist/2020.js";
import { root, hash, validateOutput, type JsonObject, type TaskInput } from "./contracts.js";
import type { TaskState } from "./state.js";
import { checkCriteriaBinding, criteriaFor, criteriaProjection } from "./criteria.js";

const wireBytes = readFileSync(`${root}/contracts/agent_rag/model-wire.schema.json`, "utf8");
export const wireSchemaHash = hash(wireBytes);
export const wireSchema = JSON.parse(wireBytes);
const ajv = new Ajv2020.default({ strict: false, allErrors: true });
const validators = new Map<string, ReturnType<typeof ajv.compile>>();
export function wireContract(kind: string): JsonObject {
  const definitions: JsonObject = {};
  const add = (name: string): void => { if (definitions[name]) return; definitions[name] = wireSchema.$defs[name]; visit(definitions[name]); };
  const visit = (v: any): void => { if (Array.isArray(v)) v.forEach(visit); else if (v && typeof v === "object") { if (v.$ref?.startsWith("#/$defs/")) add(v.$ref.slice(8)); Object.values(v).forEach(visit); } };
  if (!["analysis", "final", "finalize"].includes(kind)) throw new Error("INVALID_OUTPUT_KIND");
  add(kind); return { $schema: wireSchema.$schema, $ref: `#/$defs/${kind}`, $defs: definitions };
}
function validateWire(value: JsonObject, kind: string): void {
  if (value.kind !== kind) throw new Error(`INVALID_OUTPUT_KIND:${kind}`);
  if (!validators.has(kind)) validators.set(kind, ajv.compile(wireContract(kind)));
  const validator = validators.get(kind)!;
  if (!validator(value)) throw new Error(`INVALID_WIRE:${(validator.errors ?? []).slice(0, 6).map(e => {
    const location = e.instancePath.match(/^\/points\/(\d+)\/facts\/(\d+)(?:\/|$)/);
    const fact = location ? value.points?.[Number(location[1])]?.facts?.[Number(location[2])] : undefined;
    const origin = fact ? `field=${wireSchema.$defs.fact.properties.field.enum.includes(fact.field) ? fact.field : "invalid"},origin=${wireSchema.$defs.fact.properties.origin.enum.includes(fact.origin) ? fact.origin : "invalid"};` : "";
    const guidance = e.keyword === "minItems" && fact?.origin === "input" && e.instancePath.endsWith("/input_refs") ? "origin=input必须有input_refs；仅有背景证据时用user_context并提供context_refs；" : e.keyword === "minItems" && fact?.origin === "user_context" && e.instancePath.endsWith("/context_refs") ? "origin=user_context必须有context_refs；" : "";
    return `${e.instancePath}:${origin}${guidance}${e.message}:${JSON.stringify(e.params)}`;
  }).join(";").slice(0, 1000)}`);
}
const unknownObservation = () => ({ value: null, origin: "unknown", input_refs: [], context_refs: [], unknown_reason: "当前材料及用户背景未提供直接依据。" });
function observation(fact?: JsonObject): JsonObject {
  if (!fact || fact.origin === "unknown") return unknownObservation();
  if (!fact.value || (!fact.input_refs.length && !fact.context_refs.length)) throw new Error(`FACT_WITHOUT_CURRENT_SUPPORT:${fact.field}`);
  if (fact.origin === "input" && !fact.input_refs.length || fact.origin === "user_context" && !fact.context_refs.length) throw new Error(`FACT_ORIGIN_CONFLICT:${fact.field}.${fact.origin === "input" ? "input_refs" : "context_refs"}:origin=${fact.origin}`);
  return { ...fact, origin: fact.origin === "inferred" ? "model_inference" : fact.origin, unknown_reason: null };
}
function hypothesis(tags: string[], refs: JsonObject[]): string {
  return `待核机制：${tags.join("、")}；输入表述：${Array.from(refs[0]?.quote ?? "").slice(0, 160).join("")}`.slice(0, 300);
}
function segments(value: JsonObject): JsonObject {
  const ids = value.segments.map((s: JsonObject) => s.segment_id);
  if (new Set(ids).size !== ids.length) throw new Error("DUPLICATE_SEGMENT_ASSESSMENT");
  return { segment_assessments: value.segments, unprocessed_segment_ids: value.segments.filter((s: JsonObject) => s.status !== "reviewed").map((s: JsonObject) => s.segment_id) };
}
function candidatePresented(evidence: JsonObject, state: TaskState): boolean {
  if (state.contextEvidenceIds.has(evidence.evidence_id)) return true;
  if (!Number.isInteger(evidence.start) || !Number.isInteger(evidence.end)) return false;
  const windows = [...state.evidence.values()].filter(e => e.source_id === evidence.source_id && state.contextEvidenceIds.has(e.evidence_id) && Number.isInteger(e.start) && Number.isInteger(e.end)).sort((a, b) => a.start - b.start);
  let covered = evidence.start;
  for (const window of windows) { if (window.end <= covered) continue; if (window.start > covered) break; covered = window.end; if (covered >= evidence.end) return true; }
  return false;
}
export function candidateRead(evidence: JsonObject, state: TaskState): boolean {
  if (state.evidence.get(evidence.evidence_id)?.was_read) return true;
  if (!Number.isInteger(evidence.start) || !Number.isInteger(evidence.end)) return false;
  const windows = [...state.evidence.values()].filter(e => e.source_id === evidence.source_id && e.was_read && Number.isInteger(e.start) && Number.isInteger(e.end)).sort((a, b) => a.start - b.start);
  let covered = evidence.start;
  for (const window of windows) { if (window.end <= covered) continue; if (window.start > covered) break; covered = window.end; if (covered >= evidence.end) return true; }
  return false;
}
export function validateFinalize(value: JsonObject): void { validateWire(value, "finalize"); }
export function candidateBundleRead(candidate: JsonObject, state: TaskState, presented = false): boolean {
  const evidence: JsonObject[] = candidate.evidence ?? [];
  const usable = (e: JsonObject) => candidateRead(e, state) && (!presented || candidatePresented(e, state));
  if (candidate.required_evidence_ids === undefined) return evidence.some(usable);
  const required = candidate.required_evidence_ids;
  return Array.isArray(required) && required.length > 0 && required.every((id: string) => {
    const e = evidence.find(row => row.evidence_id === id);
    return Boolean(e && usable(e));
  });
}
function validateUnitCitations(match: JsonObject, candidate: JsonObject, state: TaskState): void {
  const required: JsonObject[] = candidate.required_evidence_ids.map((id: string) => candidate.evidence.find((e: JsonObject) => e.evidence_id === id));
  const citations: JsonObject[] = match.evidence_ids.map((id: string) => state.evidence.get(id));
  if (citations.some(e => !e || !candidateRead(e, state))) throw new Error('CASE_CITATION_NOT_READ');
  if (citations.some(e => e.source_id !== match.source_id || !Number.isInteger(e.start) || !Number.isInteger(e.end) || e.end <= e.start || !required.some(r => r.source_id === e.source_id && r.start < e.end && e.start < r.end))) throw new Error('CASE_EVIDENCE_UNIT_BINDING_MISMATCH');
  const windows = [...citations].sort((a, b) => a.start - b.start);
  for (const anchor of required) {
    let covered = anchor.start;
    for (const window of windows) {
      if (window.end <= covered) continue;
      if (window.start > covered) break;
      covered = window.end;
      if (covered >= anchor.end) break;
    }
    if (covered < anchor.end) throw new Error('CASE_CITATION_BUNDLE_INCOMPLETE');
  }
}
export function normalizeModelOutput(value: JsonObject, kind: string, state: TaskState, task: TaskInput): JsonObject {
  // 旧fixture/历史协议仍可离线验收，真实运行只接受精简协议。
  if (!Array.isArray(value.segments)) {
    if (task.config?.execution_mode === "live") throw new Error("MODEL_WIRE_REQUIRED:使用segments与points精简结构");
    validateOutput(value, kind); return value;
  }
  validateWire(value, kind); state.validateRefs(value);
  checkCriteriaBinding(task.definitions, task.snapshot.dictionary_sha256);
  let normalized: JsonObject;
  if (kind === "analysis") {
    const points = value.points.map((p: JsonObject, index: number) => {
      if (new Set(p.candidate_tags).size !== p.candidate_tags.length) throw new Error("DUPLICATE_RISK_TAG");
      if (new Set(p.facts.map((f: JsonObject) => f.field)).size !== p.facts.length) throw new Error("DUPLICATE_FACT_FIELD");
      const facts = new Map<string, JsonObject>(p.facts.map((f: JsonObject) => [f.field, f]));
      const obs = (field: string) => { const fact = facts.get(field); const o: JsonObject = field === "material_function" && (!fact || fact.origin === "unknown" || fact.value === null || fact.value === "unknown") ? unknownObservation() : observation(fact); delete o.field; return o; };
      const materialFunction = obs("material_function");
      const product = facts.get("insurance_product");
      if (product && !["yes", "no", null, ""].includes(product.value)) throw new Error("INVALID_INSURANCE_FACT_VALUE:使用yes/no/null");
      const productKnown = product && ["input", "user_context"].includes(product.origin) && ["yes", "no"].includes(product.value);
      if (productKnown) observation(product);
      // 所有点保留此事实，避免终稿新增保险标签绕过最初未知状态。
      const conditions = [{ condition_id: "insurance_product", statement: "表述指向保险产品或当前保险交易。", critical: true, status: productKnown ? product!.value === "yes" ? "met" : "not_met" : "unknown", input_refs: productKnown ? product!.input_refs : [], context_refs: productKnown ? product!.context_refs : [], reason: productKnown ? "模型依据当前材料/背景评估，仍需语义核验。" : "产品类型未知或仅为推断，不能当作事实。" }];
      const contract = facts.get("contract_outside");
      if (contract && !["yes", "no", null, ""].includes(contract.value)) throw new Error("INVALID_CONTRACT_FACT_VALUE:使用yes/no/null");
      const contractKnown = contract && ["input", "user_context"].includes(contract.origin) && ["yes", "no"].includes(contract.value);
      if (contractKnown) observation(contract);
      conditions.push({ condition_id: "contract_outside", statement: "利益在当前保险合同约定之外。", critical: true, status: contractKnown ? contract!.value === "yes" ? "met" : "not_met" : "unknown", input_refs: contractKnown ? contract!.input_refs : [], context_refs: contractKnown ? contract!.context_refs : [], reason: contractKnown ? "模型依据当前合同事实评估，仍需语义核验。" : "当前合同约定未提供；历史案例不能补此事实。" });
      // 用途未知只锁定当前行为，合法文本推断仍由终稿逐标签审查，不预判met/not_met。
      if (materialFunction.origin === "unknown") conditions.push({ condition_id: "actual_conduct", statement: "当前材料表达实际实施或指导该行为，而非禁止提醒、他人引用或风险教育。", critical: true, status: "unknown", input_refs: [], context_refs: [], reason: "材料用途未提供或无法确定；承诺内容不能单独证明宣传用途。" });
      return { point_key: `P${index + 1}`, input_refs: p.input_refs, mechanism: hypothesis(p.candidate_tags, p.input_refs), candidate_tags: p.candidate_tags,
        observations: { actor: obs("actor"), audience: obs("audience"), material_function: materialFunction, conditions },
        unknowns: conditions.filter(c => c.status === "unknown" && p.candidate_tags.some((t: string) => criteriaFor(t).some(x => x.id === c.condition_id))).map(c => c.statement) };
    });
    const declared = task.input.user_context.declared_scene;
    const scenes = ["ST1_poster_sms_social", "ST2_live_short_video", "ST3_product_intro_detail", "ST4_meeting_material", "ST5_training_telemarketing_callback", "ST6_penalty_backend_fact"];
    const functions = [...new Set(points.filter((p: JsonObject) => ["input", "user_context"].includes(p.observations.material_function.origin)).map((p: JsonObject) => p.observations.material_function.value))];
    normalized = { kind, document_context: { main_scene: scenes.includes(declared ?? "") ? declared : null, scene_status: scenes.includes(declared ?? "") ? "provided" : "unknown", material_functions: functions.length ? functions : ["unknown"] }, ...segments(value), risk_points: points };
  } else {
    let newPointNumber = state.points.length;
    const points = value.points.map((p: JsonObject) => {
      const initial = p.risk_point_id ? state.points.find(row => row.risk_point_id === p.risk_point_id) : undefined;
      if (p.risk_point_id && !initial) throw new Error("INVALID_RISK_POINT_ID");
      const refs = initial?.input_refs ?? p.input_refs ?? [];
      if (!refs.length) throw new Error("POINT_WITHOUT_VALID_INPUT");
      if (!initial && state.points.some(prior => prior.input_refs.some((r: JsonObject) => refs.some((f: JsonObject) => { const a = state.resolve(r), b = state.resolve(f); return a.start < b.end && b.start < a.end; })))) throw new Error("UNREGISTERED_OVERLAPPING_POINT:使用已有RP编号，不能借null替换未知事实");
      const tags = [...new Set([...(initial?.candidate_tags ?? []), ...p.tag_checks.map((t: JsonObject) => t.tag)])] as string[];
      for (const match of p.case_matches) {
        if (task.config?.retrieval_profile === 'v2-hybrid-flash-v1') {
          if (!match.unit_id) throw new Error('CASE_UNIT_REQUIRED');
          const selected = state.searches.flatMap(q => q.candidates ?? []).find((c: JsonObject) => c.source_id === match.source_id && c.unit_id === match.unit_id);
          if (!selected) throw new Error('CASE_UNIT_BINDING_MISMATCH');
          if (!candidateBundleRead(selected, state, true)) throw new Error('CASE_BUNDLE_NOT_FULLY_READ');
          validateUnitCitations(match, selected, state);
        }
        if (match.evidence_ids.some((id: string) => !state.contextEvidenceIds.has(id))) throw new Error("PRESENTED_EVIDENCE_REQUIRED:终稿必须看到所引原话，不能仅凭来源定位");
        if (match.supports_tags.some((tag: string) => !tags.includes(tag))) throw new Error("INVALID_CASE_TAG_BINDING");
      }
      if (new Set(p.tag_checks.map((t: JsonObject) => t.tag)).size !== p.tag_checks.length) throw new Error("DUPLICATE_RISK_TAG");
      const checks = tags.map(tag => {
        const supplied = p.tag_checks.find((t: JsonObject) => t.tag === tag)?.conditions ?? [];
        const template = criteriaFor(tag);
        if (new Set(supplied.map((c: JsonObject) => c.criterion_id)).size !== supplied.length || supplied.some((c: JsonObject) => !template.some(t => t.id === c.criterion_id))) throw new Error(`INVALID_CRITERION:${tag}`);
        const conditions = template.map(t => {
          let c = supplied.find((row: JsonObject) => row.criterion_id === t.id);
          if (!c) c = { status: "unknown", input_refs: [], context_refs: [], reason: "模型未填写本标签所需条件，程序保留未知。" };
          if (c.status !== "unknown" && !c.input_refs.length && !c.context_refs.length) throw new Error("CONDITION_WITHOUT_CURRENT_SUPPORT");
          const prior = initial?.observations.conditions.find((row: JsonObject) => row.condition_id === t.id);
          if (!initial && t.id === "actual_conduct") c = { ...c, status: "unknown", reason: "终稿新增点缺少已接受的用途分析，当前材料用途无法确定。" };
          if (prior && prior.status === "unknown" && c.status !== "unknown") c = { ...c, status: "unknown", reason: "分析阶段此当前事实未知；本轮没有新增用户事实，禁止借历史案例或推测强化。" };
          if (prior && prior.status !== "unknown" && c.status !== "unknown" && c.status !== prior.status) throw new Error(`FACT_STATUS_CONFLICT:${t.id}`);
          return { condition_id: t.id, statement: t.statement, critical: true, status: c.status, input_refs: c.input_refs, context_refs: c.context_refs, reason: c.reason };
        });
        const status = conditions.some(c => c.status === "not_met") ? "not_supported" : conditions.every(c => c.status === "met") ? "supported" : "unknown";
        const support = conditions.filter(c => c.status === "met").flatMap(c => c.input_refs);
        // 对外旧报告要求supported附当前输入；背景支持不替代被审材料自身。
        return { tag, status, reason: status === "supported" ? "当前输入与官方定义条件对应，提示疑似风险。" : status === "not_supported" ? "至少一个必要条件有当前依据予以排除。" : "必要条件存在未知，保留待复核。", conditions,
          support_input_refs: status === "supported" ? refs : support.slice(0, 16), contra_input_refs: conditions.filter(c => c.status === "not_met").flatMap(c => c.input_refs).slice(0, 16), case_match_status: status === "not_supported" ? "not_required" : "not_completed" };
      });
      const judgment = checks.some(c => c.status === "supported") ? "suspected" : checks.some(c => c.status === "unknown") || !checks.length ? "review_required" : "no_clear_risk";
      const searches = state.searches.filter(q => (p.risk_point_id && q.risk_point_ids.includes(p.risk_point_id)) || q.focus_refs.some((r: JsonObject) => refs.some((f: JsonObject) => r.segment_id === f.segment_id && r.quote === f.quote)));
      const unreadGoal = searches.some(q => q.candidates?.length && q.candidates.every((c: JsonObject) => !candidateBundleRead(c, state, true)));
      const caseStatus = judgment === "no_clear_risk" ? "not_required" : unreadGoal ? "not_completed" : p.case_matches.length ? p.case_matches.every((c: JsonObject) => c.match_status === "reliable_match") ? "reliable_match" : "partial_match" : searches.some(q => q.complete) ? "no_reliable_match" : "not_completed";
      state.caseTagBindings.set(p.risk_point_id ?? `RP${String(++newPointNumber).padStart(4, "0")}`, p.case_matches);
      return { risk_point_id: p.risk_point_id, input_refs: refs, mechanism: hypothesis(tags, refs), judgment, tag_checks: checks.map(c => ({ ...c, case_match_status: c.status === "not_supported" ? "not_required" : caseStatus })),
        case_match_status: caseStatus, case_matches: p.case_matches.map((m: JsonObject) => { const { supports_tags, ...record } = m; return record; }), unknowns: [...new Set(checks.filter(t => t.status === "unknown").flatMap(t => t.conditions.filter(c => c.status === "unknown").map(c => c.statement)))].slice(0, 16) };
    });
    normalized = { kind, ...segments(value), risk_points: points, summary: points.some((p: JsonObject) => p.judgment === "suspected") ? "按当前材料与官方标签定义提示疑似风险；案例支持程度和未知条件逐项列示。" : points.some((p: JsonObject) => p.judgment === "review_required") ? "材料关键条件不足，按现有文本列出待复核事项。" : "按现有材料未发现明确风险；本结论受材料范围限制。审查说明：" + value.segments.slice(0, 3).map((s: JsonObject) => s.reason).join("；") };
  }
  validateOutput(normalized, kind); state.protocolVersion = "finrisk-model-wire/1.2";
  return normalized;
}

export function wireExample(kind: string): JsonObject {
  const segments = [{ segment_id: "S0001", status: "reviewed", reason: "说明当前段落用途、否定与判断依据。" }];
  const refs = [{ segment_id: "S0001", quote: "示例片段原话", occurrence: 1 }];
  const points = kind === "analysis" ? [{ input_refs: refs, candidate_tags: ["合同外利益"], facts: [{ field: "insurance_product", value: null, origin: "unknown", input_refs: [], context_refs: [] }] }] : [{ risk_point_id: "RP0001", tag_checks: [{ tag: "合同外利益", conditions: criteriaFor("合同外利益").map(c => ({ criterion_id: c.id, status: "unknown", input_refs: [], context_refs: [], reason: "示例：未提供充分事实。" })) }], case_matches: [] }];
  return { kind, segments, points };
}

export function stageProtocol(task: TaskInput, state: TaskState, kind?: string): string {
  let dictionary = state.phase === "analyzing" ? `\n官方字典=${JSON.stringify(task.definitions.map(d => ({ tag: d.risk_tag, definition: d.description })))}` : `\n逐标签条件（内部字典解释，不是新增法规）=${JSON.stringify(criteriaProjection(state.points.flatMap(p => p.candidate_tags)))}`;
  if (task.config?.retrieval_profile === 'v2-hybrid-flash-v1') dictionary += '\nV2案例参照须填写实际命中unit_id，核读该单元required_evidence_ids对应的行为、主体及条件；evidence_ids须引用本轮已核读且可见的原话，覆盖全部必要锚点，可用覆盖它们的回源窗口；同一文书的其它行为不能补当前参照。读全仅证明证据可查看，不自动证明语义适用。';
  return dictionary + (kind ? `\n当前输出JSON Schema=${JSON.stringify(wireContract(kind))}\n结构示例（仅演示结构，必须审阅实际材料）=${JSON.stringify(wireExample(kind))}${kind === "analysis" ? "\nfacts记录actor/audience/material_function/insurance_product/contract_outside；后两者value必须为yes/no/null。insurance_product只指保险业务指向，不要求具体险种；原文明确购买保险可支持yes。material_function必须用枚举；从文字推断用途用inferred，背景明确用途才用user_context。origin=input必须有input_refs；仅有背景证据时用user_context并提供context_refs，未知value=null。" : ""}` : `\n本阶段仅使用已声明的原生工具调用选择必要工具，不在正文JSON中写工具名或参数，不必附加decision JSON。准备结束时只输出${JSON.stringify({ kind: "finalize" })}；不得提交final或重复analysis，正式终稿由下一阶段的完整Schema约束。搜索候选不等于已核对案例，必要时用read_source覆盖命中锚点。结束信号Schema=${JSON.stringify(wireContract("finalize"))}`);
}
