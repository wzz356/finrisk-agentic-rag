import { canonical, hash, MODEL_WIRE_VERSION, officialTags, type JsonObject, type TaskInput, type ToolEnvelope, validateToolArguments } from "./contracts.js";
import { criteriaByteHash } from "./criteria.js";
import { validateBudgetConfig } from "./budget_profile.js";
import { AGENT_LOOP_VERSION, loopSchemaHash, validateLoopOutput, LoopModelDataError } from "./agent_loop_contracts.js";
import { buildCitationOptions, expandCaseSelections, type CitationOption } from "./citation_bundles.js";
import { normalizeModelOutput } from "./model_protocol.js";
import { assembleReport } from "./result_validator.js";
import { validateProgram } from "./contracts.js";
import type { TaskState } from "./state.js";
import type { Budget } from "./budget.js";
import type { ToolBridge } from "./tool_bridge.js";

export type ProcessingState = "dirty" | "assessed" | "deferred" | "blocked";
export type BlockerClass = "typed_output" | "branch_output" | "identity" | "permission" | "budget" | "transport" | "program";
export interface AcceptedPoint {
  assessment_id: string; state_revision: number; candidate_version: number; assessment_revision: number;
  result: JsonObject; bindings: JsonObject[]; tag_assessments: JsonObject[]; evidence: JsonObject[];
}
export interface LoopPoint extends JsonObject {
  risk_point_id: string; plan_revision: number; state_revision: number; candidate_version: number; assessment_revision: number;
  candidate_tags: string[]; input_refs: JsonObject[]; facts: JsonObject[]; processing_state: ProcessingState;
  accepted?: AcceptedPoint; draft?: AcceptedPoint;
}
export interface AssessmentRequest {
  readonly kind: "assess" | "one_shot"; readonly assessment_id: string; readonly risk_point_id: string;
  readonly state_revision: number; readonly plan_revision: number; readonly candidate_version: number;
  readonly assessment_revision: number; readonly candidate_tags: readonly string[];
  readonly citationOptions: ReadonlyMap<string, CitationOption>; readonly contextEvidenceIds: ReadonlySet<string>;
  readonly evidenceBindings: ReadonlyMap<string, string>;
}
function freeze<T>(value: T): T {
  if (value && typeof value === "object" && !Object.isFrozen(value)) { Object.values(value).forEach(freeze); Object.freeze(value); }
  return value;
}
function immutableMap<K, V>(source: Map<K, V>): ReadonlyMap<K, V> {
  const data = new Map(source);
  const view: ReadonlyMap<K, V> = Object.freeze({ get size() { return data.size; }, get: (k: K) => data.get(k), has: (k: K) => data.has(k),
    keys: () => data.keys(), values: () => data.values(), entries: () => data.entries(), [Symbol.iterator]: () => data[Symbol.iterator](),
    forEach: (fn: (v: V, k: K, m: ReadonlyMap<K, V>) => void) => data.forEach((v, k) => fn(v, k, view)) } as ReadonlyMap<K, V>);
  return view;
}
function immutableSet<T>(source: Set<T>): ReadonlySet<T> {
  const data = new Set(source);
  const view: ReadonlySet<T> = Object.freeze({ get size() { return data.size; }, has: (v: T) => data.has(v),
    keys: () => data.keys(), values: () => data.values(), entries: () => data.entries(), [Symbol.iterator]: () => data[Symbol.iterator](),
    forEach: (fn: (v: T, k: T, s: ReadonlySet<T>) => void) => data.forEach(v => fn(v, v, view)) });
  return view;
}
export function stageLoopState(state: TaskState): TaskState {
  return Object.assign(Object.create(Object.getPrototypeOf(state)), state, {
    gaps: new Map([...state.gaps].map(([k, v]) => [k, structuredClone(v)])),
    caseTagBindings: new Map([...state.caseTagBindings].map(([k, v]) => [k, structuredClone(v)])),
    points: structuredClone(state.points), analysis: state.analysis && structuredClone(state.analysis),
    final: state.final && structuredClone(state.final), citationOptions: new Map(state.citationOptions),
    contextEvidenceIds: new Set(state.contextEvidenceIds), evidence: new Map([...state.evidence].map(([k, v]) => [k, structuredClone(v)])),
    searches: [...state.searches], validationErrors: [...state.validationErrors],
  });
}
function evidenceIdentity(e: JsonObject): string {
  return canonical(Object.fromEntries(["evidence_id", "source_id", "source_kind", "text_version", "text_sha256", "start", "end", "quote", "source_verified", "was_read"].map(k => [k, e[k] ?? null])));
}
function mergedCases(checks: JsonObject[], options: ReadonlyMap<string, CitationOption>) {
  const cases = new Map<string, JsonObject>(), tag_assessments: JsonObject[] = [];
  for (const check of checks) {
    const expanded = expandCaseSelections(check.case_assessments.map((c: JsonObject) => ({ ...c, supports_tags: [check.tag] })), options);
    tag_assessments.push({ tag: check.tag, conditions: structuredClone(check.conditions), case_assessments: check.case_assessments.map((c: JsonObject, i: number) => ({
      ...structuredClone(c), source_id: expanded[i]!.source_id, unit_id: expanded[i]!.unit_id, evidence_ids: expanded[i]!.evidence_ids })) });
    for (const c of check.case_assessments) {
      const prior = cases.get(c.case_ref);
      const reasons = (field: string) => c[field].map((s: string) => "[" + check.tag + "] " + s);
      if (!prior) cases.set(c.case_ref, { case_ref: c.case_ref, same_points: reasons("same_points"), differences: reasons("differences"), match_status: c.match_status, supports_tags: [check.tag] });
      else { prior.same_points.push(...reasons("same_points")); prior.differences.push(...reasons("differences")); prior.supports_tags.push(check.tag);
        if (c.match_status !== "reliable_match") prior.match_status = "partial_match"; }
    }
  }
  return { cases: [...cases.values()], tag_assessments };
}

// Only these errors, thrown while normalizing model data, can be superseded by a fresh plan.
// Citation source/window binding, CAS, transport, payload and unexpected errors stay hard failures.
function normalizeLoopModel(...args: Parameters<typeof normalizeModelOutput>): JsonObject {
  try { return normalizeModelOutput(...args); }
  catch (error) {
    const code = error instanceof Error ? error.message.split(":", 1)[0]! : "";
    if (error instanceof Error && error.constructor === Error && ["INVALID_INPUT_REFERENCE", "INVALID_CONTEXT_REFERENCE", "INVALID_SEGMENT_REFERENCE",
      "DUPLICATE_SEGMENT_ASSESSMENT", "DUPLICATE_RISK_TAG", "DUPLICATE_FACT_FIELD", "FACT_WITHOUT_CURRENT_SUPPORT", "FACT_ORIGIN_CONFLICT",
      "INVALID_INSURANCE_FACT_VALUE", "INVALID_CONTRACT_FACT_VALUE", "CONDITION_WITHOUT_CURRENT_SUPPORT", "FACT_STATUS_CONFLICT",
      "INVALID_CRITERION", "DUPLICATE_CASE_REFERENCE", "INVALID_CASE_SELECTION"].includes(code)) throw new LoopModelDataError(code);
    throw error;
  }
}

/** Task-scoped registry. Tool observations stay in TaskState; no second evidence authority is created. */
export class LoopRegistry {
  readonly protocol_version = AGENT_LOOP_VERSION;
  readonly identity: Readonly<JsonObject>;
  readonly points = new Map<string, LoopPoint>();
  readonly readEvidence = new Map<string, JsonObject>();
  readonly queries = new Map<string, JsonObject>();
  readonly active_blockers = new Map<string, JsonObject>();
  readonly actions: JsonObject[] = [];
  readonly blocker_retirements: JsonObject[] = [];
  plan_revision = 0;
  segments: JsonObject[] = [];
  branch: "plan" | "one_shot" | null = null;
  private nativeFailures = new Map<string, { action_id: string; tool: string }>();
  private attemptRecords: JsonObject[] = [];
  private requests = new WeakSet<object>();
  private activeRequests = new Map<string, AssessmentRequest>();
  private requestNumber = 0;
  private observationRevision = 0;
  private queryOrigins = new WeakMap<object, { plan_revision: number; risk_point_ids: string[] }>();
  private pointReads = new Map<string, Map<string, string>>();
  private oneShotRequest?: AssessmentRequest;
  private acceptedPlanScope?: JsonObject;
  private binding: string;
  constructor(readonly task: TaskInput, readonly state: TaskState, readonly budget: Budget) {
    if (canonical(state.input) !== canonical(task.input)) throw new Error("LOOP_INPUT_STATE_MISMATCH");
    const config = { ...budget.config, ...task.config };
    if (!validateBudgetConfig(config)) throw new Error("LOOP_PROFILE_REQUIRED");
    if (canonical(config) !== canonical(budget.config)) throw new Error("LOOP_BUDGET_CONFIG_MISMATCH");
    this.identity = freeze({ task_id: task.input.task_id, profile_id: config.profile_id, input_sha256: task.input.input_sha256,
      snapshot_sha256: hash(canonical(task.snapshot)), criteria_sha256: criteriaByteHash,
      budget_profile_sha256: config.budget_profile_sha256, profile_sha256: hash(canonical(config)), loop_schema_sha256: loopSchemaHash });
    this.binding = this.bindingNow();
  }
  get attempt_diagnostics(): readonly JsonObject[] { return Object.freeze([...this.attemptRecords]); }
  private bindingNow(): string {
    return canonical({ input: this.task.input, state_input: this.state.input, snapshot: this.task.snapshot, definitions: this.task.definitions,
      config: this.budget.config, task_config: this.task.config, criteria_sha256: criteriaByteHash });
  }
  assertIdentity(): void { if (this.bindingNow() !== this.binding) throw new Error("LOOP_IDENTITY_CHANGED"); }
  private point(id: string): LoopPoint { const p = this.points.get(id); if (!p) throw new Error("LOOP_UNKNOWN_POINT"); return p; }
  private stageAnalysis(points: JsonObject[], segments: JsonObject[]) {
    const staged = stageLoopState(this.state);
    const owned = new Set(this.points.keys());
    for (const [id, g] of staged.gaps) if (owned.has(g.risk_point_id) && g.type === "input_condition") staged.gaps.delete(id);
    staged.evaluationCitations = true; staged.citationSnapshotId = this.task.snapshot.snapshot_id;
    const analysis = normalizeLoopModel({ kind: "analysis", segments, points: points.map(p => ({
      input_refs: p.input_refs, facts: p.facts, candidate_tags: p.candidate_tags })) }, "analysis", staged, this.task);
    staged.acceptAnalysis(analysis); return staged;
  }
  private planPoints(value: JsonObject, revision: number): LoopPoint[] {
    return value.points.map((p: JsonObject, i: number) => freeze({ ...structuredClone(p), risk_point_id: "RP" + String(i + 1).padStart(4, "0"),
      plan_revision: revision, state_revision: 1, candidate_version: 1, assessment_revision: 0, processing_state: "dirty",
      initial_query_covered: false, pending_case_goals: [], read_anchor_ids: [], stop_reason: null, additional_mechanisms_uncovered: false }));
  }
  private commitPlan(value: JsonObject, staged: TaskState, points: LoopPoint[], branch: "plan" | "one_shot") {
    for (const query of this.state.searches) this.queryOrigin(query);
    const priorIds = new Set(this.points.keys());
    // Preserve global registries and unrelated entries; retire only this plan's model-owned bindings.
    for (const id of priorIds) this.state.caseTagBindings.delete(id);
    this.points.clear(); for (const p of points) this.points.set(p.risk_point_id, p);
    this.plan_revision++; this.segments = structuredClone(value.segments); this.branch = branch; this.activeRequests.clear(); this.pointReads.clear();
    Object.assign(this.state, { analysis: staged.analysis, points: staged.points, gaps: staged.gaps, final: undefined,
      phase: "reviewing", protocolVersion: MODEL_WIRE_VERSION, evaluationCitations: true, citationSnapshotId: this.task.snapshot.snapshot_id });
  }
  registerPlan(value: JsonObject, options: { replace_one_shot?: boolean } = {}): void {
    this.assertIdentity(); validateLoopOutput(value, "plan");
    const staged = this.stageAnalysis(value.points, value.segments), points = this.planPoints(value, this.plan_revision + 1);
    if (this.branch && !(this.branch === "one_shot" && options.replace_one_shot === true)) throw new Error("LOOP_PLAN_ALREADY_REGISTERED");
    const acceptedScope = this.scope("plan");
    this.commitPlan(value, staged, points, "plan"); this.acceptedPlanScope = freeze(acceptedScope);
  }
  projectPointState(id: string): TaskState {
    this.assertIdentity(); this.point(id);
    const staged = stageLoopState(this.state);
    staged.points = staged.points.filter(p => p.risk_point_id === id);
    staged.searches = staged.searches.filter(q => this.queryRelated(q, this.point(id)));
    const candidates = staged.searches.flatMap(q => q.candidates ?? []);
    const anchors = candidates.flatMap(c => c.evidence ?? []), anchorIds = new Set(anchors.map(e => e.evidence_id));
    const fixed = new Set((this.point(id).accepted ?? this.point(id).draft)?.evidence.map(e => e.evidence_id) ?? []);
    staged.evidence = new Map([...staged.evidence].filter(([key, e]) => anchorIds.has(key) || fixed.has(key) || e.was_read && anchors.some(a =>
      a.source_id === e.source_id && a.text_version === e.text_version && a.text_sha256 === e.text_sha256 && a.start < e.end && e.start < a.end)));
    staged.contextEvidenceIds = new Set([...staged.contextEvidenceIds].filter(key => staged.evidence.has(key)));
    const options = buildCitationOptions(this.task.snapshot.snapshot_id, candidates, [...staged.evidence.values()]);
    staged.citationOptions = new Map(options.map(o => [o.case_ref, o]));
    staged.gaps = new Map([...staged.gaps].filter(([, g]) => g.risk_point_id === id));
    if (staged.analysis) staged.analysis = { ...staged.analysis, risk_points: staged.analysis.risk_points?.filter((p: JsonObject) => p.risk_point_id === id),
      segment_assessments: [], unprocessed_segment_ids: [] };
    staged.protocolVersion = MODEL_WIRE_VERSION; staged.evaluationCitations = true; staged.citationSnapshotId = this.task.snapshot.snapshot_id;
    return staged;
  }
  private request(kind: "assess" | "one_shot", projection: TaskState, point?: LoopPoint): AssessmentRequest {
    const req = Object.freeze({ kind, assessment_id: "AS" + String(++this.requestNumber).padStart(4, "0"),
      risk_point_id: point?.risk_point_id ?? "", state_revision: point?.state_revision ?? this.observationRevision,
      plan_revision: this.plan_revision, candidate_version: point?.candidate_version ?? 0,
      assessment_revision: (point?.assessment_revision ?? 0) + 1, candidate_tags: Object.freeze([...(point?.candidate_tags ?? [])]),
      citationOptions: immutableMap(new Map([...projection.citationOptions].map(([k, v]) => [k, freeze(structuredClone(v))]))),
      contextEvidenceIds: immutableSet(new Set(projection.contextEvidenceIds)),
      evidenceBindings: immutableMap(new Map([...projection.contextEvidenceIds].filter(id => projection.evidence.has(id)).map(id => [id, evidenceIdentity(projection.evidence.get(id)!)]))) });
    this.requests.add(req); return req;
  }
  beginAssessment(id: string, projection: TaskState = this.state): AssessmentRequest {
    this.assertIdentity(); const point = this.point(id), req = this.request("assess", projection, point);
    this.points.set(id, freeze({ ...point, assessment_revision: req.assessment_revision })); this.activeRequests.set(id, req); return req;
  }
  beginOneShot(projection: TaskState = this.state): AssessmentRequest {
    this.assertIdentity(); const req = this.request("one_shot", projection); this.oneShotRequest = req; return req;
  }
  /** Freeze only the context that survived the final request count/prune. No await may split this CAS. */
  finalizeAssessmentContext(request: AssessmentRequest, projection: TaskState): AssessmentRequest {
    this.assertRequest(request);
    if (canonical(projection.input) !== canonical(this.state.input)) throw new Error("LOOP_INPUT_STATE_MISMATCH");
    for (const id of projection.contextEvidenceIds) {
      const actual = this.state.evidence.get(id), projected = projection.evidence.get(id);
      if (!actual || !projected || evidenceIdentity(actual) !== evidenceIdentity(projected)) throw new Error("LOOP_EVIDENCE_CHANGED");
    }
    const next: AssessmentRequest = Object.freeze({ ...request,
      citationOptions: immutableMap(new Map([...projection.citationOptions].map(([k, v]) => [k, freeze(structuredClone(v))]))),
      contextEvidenceIds: immutableSet(new Set(projection.contextEvidenceIds)),
      evidenceBindings: immutableMap(new Map([...projection.contextEvidenceIds].map(id => [id, evidenceIdentity(projection.evidence.get(id)!)]))),
    });
    this.requests.add(next);
    if (request.kind === "one_shot") this.oneShotRequest = next;
    else this.activeRequests.set(request.risk_point_id, next);
    return next;
  }
  private assertRequest(request: AssessmentRequest): void {
    this.assertIdentity();
    if (!this.requests.has(request) || request.plan_revision !== this.plan_revision) throw new Error("LOOP_STALE_ASSESSMENT");
    if (request.kind === "one_shot") { if (this.oneShotRequest !== request || request.state_revision !== this.observationRevision) throw new Error("LOOP_STALE_ASSESSMENT"); return; }
    const p = this.point(request.risk_point_id);
    if (this.activeRequests.get(p.risk_point_id) !== request || p.state_revision !== request.state_revision
      || p.candidate_version !== request.candidate_version || p.assessment_revision !== request.assessment_revision) throw new Error("LOOP_STALE_ASSESSMENT");
  }
  private requestState(request: AssessmentRequest, base = this.state): TaskState {
    const staged = stageLoopState(base); staged.citationOptions = new Map(request.citationOptions);
    staged.contextEvidenceIds = new Set(request.contextEvidenceIds); staged.evaluationCitations = true; staged.citationSnapshotId = this.task.snapshot.snapshot_id;
    staged.protocolVersion = MODEL_WIRE_VERSION; return staged;
  }
  private checkEvidence(request: AssessmentRequest, result: JsonObject): JsonObject[] {
    const ids = [...new Set<string>(result.case_matches.flatMap((c: JsonObject) => c.evidence_ids))];
    return ids.map(id => {
      const e = this.state.evidence.get(id);
      if (!e || !e.was_read || !request.contextEvidenceIds.has(id) || request.evidenceBindings.get(id) !== evidenceIdentity(e)) throw new Error("LOOP_EVIDENCE_CHANGED");
      return structuredClone(e);
    });
  }
  private normalizePoint(checks: JsonObject[], id: string, request: AssessmentRequest, staged: TaskState) {
    const merged = mergedCases(checks, request.citationOptions);
    const wire = { kind: "final", segments: this.segments, points: [{ risk_point_id: id,
      tag_checks: checks.map(c => ({ tag: c.tag, conditions: c.conditions })), case_matches: merged.cases }] };
    staged.searches = this.queriesForPoint(id);
    const normalized = normalizeLoopModel(wire, "final", staged, this.task, { loopProjection: true });
    const result = normalized.risk_points[0]!;
    this.normalizeTagConditions(merged.tag_assessments, result);
    return { normalized, result, bindings: staged.caseTagBindings.get(id)!, tag_assessments: merged.tag_assessments };
  }
  private normalizeTagConditions(assessments: JsonObject[], result: JsonObject): void {
    for (const tag of assessments) tag.conditions = result.tag_checks.find((t: JsonObject) => t.tag === tag.tag).conditions.map((c: JsonObject) => ({
      criterion_id: c.condition_id, status: c.status, input_refs: structuredClone(c.input_refs), context_refs: structuredClone(c.context_refs), reason: c.reason,
    }));
  }
  private async verify(staged: TaskState, bridge: ToolBridge): Promise<void> {
    const report = await assembleReport(this.task, staged, this.budget, bridge, { loopProjection: true });
    if (!report.validation.schema_valid || !report.validation.references_valid) throw new Error("LOOP_REPORT_REFERENCE_INVALID");
    validateProgram(report, "final-report");
  }
  async acceptPointAssessment(value: JsonObject, request: AssessmentRequest, bridge: ToolBridge): Promise<AcceptedPoint> {
    this.assertRequest(request); if (request.kind !== "assess") throw new Error("LOOP_INVALID_REQUEST_KIND");
    validateLoopOutput(value, "assess", { expected_assessment_id: request.assessment_id, expected_state_revision: request.state_revision,
      expected_risk_point_id: request.risk_point_id, candidate_tags: request.candidate_tags, visible_case_refs: [...request.citationOptions.keys()] });
    value = structuredClone(value);
    const staged = this.requestState(request), data = this.normalizePoint(value.tag_checks, request.risk_point_id, request, staged);
    this.checkEvidence(request, data.result); staged.final = data.normalized;
    await this.verify(staged, bridge);
    this.assertRequest(request);
    const evidence = this.checkEvidence(request, data.result);
    const accepted: AcceptedPoint = freeze({ assessment_id: request.assessment_id, state_revision: request.state_revision,
      candidate_version: request.candidate_version, assessment_revision: request.assessment_revision,
      result: data.result, bindings: data.bindings, tag_assessments: data.tag_assessments, evidence });
    const point = this.point(request.risk_point_id);
    this.points.set(point.risk_point_id, freeze({ ...point, accepted, processing_state: "assessed", stop_reason: null }));
    this.state.caseTagBindings.set(point.risk_point_id, structuredClone(data.bindings));
    this.activeRequests.delete(point.risk_point_id);
    return accepted;
  }
  async acceptOneShot(value: JsonObject, request: AssessmentRequest, bridge: ToolBridge): Promise<void> {
    this.assertRequest(request); if (request.kind !== "one_shot" || this.branch) throw new Error("LOOP_INVALID_REQUEST_KIND");
    validateLoopOutput(value, "one_shot", { visible_case_refs: [...request.citationOptions.keys()] });
    value = structuredClone(value);
    const plan = { segments: value.segments, points: value.points.map((p: JsonObject) => ({ input_refs: p.input_refs, facts: p.facts, candidate_tags: p.tag_checks.map((c: JsonObject) => c.tag) })) };
    const staged = this.requestState(request, this.stageAnalysis(plan.points, plan.segments));
    // normalizePoint uses plan segments without publishing any registry state.
    const results: ReturnType<LoopRegistry["normalizePoint"]>[] = [];
    for (let i = 0; i < value.points.length; i++) {
      const checks = value.points[i].tag_checks, id = "RP" + String(i + 1).padStart(4, "0"), merged = mergedCases(checks, request.citationOptions);
      const normalized = normalizeLoopModel({ kind: "final", segments: value.segments, points: [{ risk_point_id: id,
        tag_checks: checks.map((c: JsonObject) => ({ tag: c.tag, conditions: c.conditions })), case_matches: merged.cases }] }, "final", staged, this.task, { loopProjection: true });
      const result = normalized.risk_points[0]; this.checkEvidence(request, result); this.normalizeTagConditions(merged.tag_assessments, result);
      results.push({ normalized, result, bindings: staged.caseTagBindings.get(id)!, tag_assessments: merged.tag_assessments });
    }
    staged.final = { kind: "final", segment_assessments: value.segments, unprocessed_segment_ids: value.segments.filter((s: JsonObject) => s.status !== "reviewed").map((s: JsonObject) => s.segment_id),
      risk_points: results.map(d => d.result), summary: "按当前材料逐点列示风险条件与案例比较，未知事实保留待复核。" };
    await this.verify(staged, bridge); this.assertRequest(request);
    const points = this.planPoints(plan, this.plan_revision + 1).map((p, i) => {
      const d = results[i]!, evidence = this.checkEvidence(request, d.result);
      return freeze({ ...p, assessment_revision: 1, processing_state: "assessed" as const, accepted: freeze({
        assessment_id: request.assessment_id, state_revision: p.state_revision, candidate_version: p.candidate_version,
        assessment_revision: 1, result: d.result, bindings: d.bindings, tag_assessments: d.tag_assessments, evidence }) });
    });
    this.commitPlan({ ...plan, segments: value.segments }, staged, points, "one_shot");
    for (const p of points) this.state.caseTagBindings.set(p.risk_point_id, structuredClone(p.accepted!.bindings));
  }
  markPointDirty(id: string, reason: string): void {
    this.assertIdentity(); const p = this.point(id);
    this.points.set(id, freeze({ ...p, state_revision: p.state_revision + 1, processing_state: "dirty", accepted: undefined, draft: p.accepted ?? p.draft, stop_reason: reason }));
    this.activeRequests.delete(id);
  }
  setProcessingState(id: string, status: "deferred" | "blocked", reason: string): void {
    this.markPointDirty(id, reason); this.points.set(id, freeze({ ...this.point(id), processing_state: status }));
  }
  expandCandidates(id: string, tags: readonly string[], verifiedDefinitionTags: readonly string[]): void {
    this.assertIdentity(); const p = this.point(id);
    if (tags.some(t => !officialTags.includes(t) || !verifiedDefinitionTags.includes(t))) throw new Error("LOOP_DEFINITION_REQUIRED");
    const merged = [...new Set([...p.candidate_tags, ...tags])];
    if (merged.length > 3) { this.points.set(id, freeze({ ...p, additional_mechanisms_uncovered: true })); throw new Error("LOOP_CANDIDATE_LIMIT"); }
    if (merged.length === p.candidate_tags.length) return;
    this.markPointDirty(id, "CANDIDATES_CHANGED");
    this.points.set(id, freeze({ ...this.point(id), candidate_tags: merged, candidate_version: p.candidate_version + 1 }));
    this.state.points = this.state.points.map(row => row.risk_point_id === id ? { ...row, candidate_tags: [...merged] } : row);
  }
  private queryOrigin(query: JsonObject) {
    let origin = this.queryOrigins.get(query);
    if (!origin) {
      const ids = this.branch ? [...(query.risk_point_ids ?? [])] : [];
      if (ids.some(id => !this.points.has(id))) throw new Error("LOOP_UNKNOWN_QUERY_POINT");
      origin = { plan_revision: this.plan_revision, risk_point_ids: ids };
      this.queryOrigins.set(query, origin);
    }
    return origin;
  }
  private queryRelated(query: JsonObject, p: LoopPoint): boolean {
    const origin = this.queryOrigin(query);
    if (origin.risk_point_ids.length) return origin.plan_revision === p.plan_revision && origin.risk_point_ids.includes(p.risk_point_id);
    const spans = p.input_refs.map(r => this.state.resolve(r));
    return (query.focus_refs ?? []).some((r: JsonObject) => { const s = this.state.resolve(r); return spans.some(a => a.start < s.end && s.start < a.end); });
  }
  queriesForPoint(id: string): JsonObject[] {
    this.assertIdentity(); const p = this.point(id);
    return this.state.searches.filter(q => this.queryRelated(q, p));
  }
  recordQuery(query: JsonObject, origin: "initial_plan" | "adaptive_model" | "one_shot" = "initial_plan"): void {
    this.assertIdentity(); this.state.validateRefs(query);
    const ids: string[] = this.branch ? [...(query.risk_point_ids ?? [])] : [];
    if (ids.some(id => !this.points.has(id))) throw new Error("LOOP_UNKNOWN_QUERY_POINT");
    this.queryOrigins.set(query, { plan_revision: this.plan_revision, risk_point_ids: ids });
    const key = this.plan_revision + ":" + this.state.queryKey(query);
    const current = this.queries.get(key), next = freeze({ plan_revision: this.plan_revision, origin, query: structuredClone(query),
      risk_point_ids: [...this.points.values()].filter(p => this.queryRelated(query, p)).map(p => p.risk_point_id) });
    if (current && canonical(current) === canonical(next)) return;
    this.queries.set(key, next); this.observationRevision++;
    for (const id of next.risk_point_ids) {
      this.markPointDirty(id, "QUERY_CHANGED"); const p = this.point(id);
      this.points.set(id, freeze({ ...p, initial_query_covered: origin === "initial_plan" || p.initial_query_covered }));
    }
  }
  observeEvidence(ids: readonly string[], explicitPointIds?: readonly string[]): void {
    this.assertIdentity();
    const entries = ids.map(id => { const e = this.state.evidence.get(id); if (!e || !e.was_read) throw new Error("LOOP_EVIDENCE_NOT_READ"); return [id, e] as const; });
    const relatedEntries = new Map<LoopPoint, typeof entries>();
    for (const p of explicitPointIds ? explicitPointIds.map(id => this.point(id)) : this.points.values()) {
      const queries = this.queriesForPoint(p.risk_point_id);
      const selected = explicitPointIds ? entries : entries.filter(([, e]) => queries.some(q => (q.candidates ?? []).some((c: JsonObject) =>
        c.source_id === e.source_id && (c.evidence ?? []).some((anchor: JsonObject) =>
          anchor.source_id === e.source_id && anchor.text_version === e.text_version && anchor.text_sha256 === e.text_sha256
          && anchor.start < e.end && e.start < anchor.end))));
      if (selected.length) relatedEntries.set(p, selected);
    }
    const changed = entries.filter(([id, e]) => !this.readEvidence.has(id) || evidenceIdentity(this.readEvidence.get(id)!) !== evidenceIdentity(e));
    const pointChanges = [...relatedEntries].filter(([p, selected]) => selected.some(([id, e]) => this.pointReads.get(p.risk_point_id)?.get(id) !== evidenceIdentity(e)));
    if (!changed.length && !pointChanges.length) return;
    this.observationRevision++;
    for (const [id, e] of changed) this.readEvidence.set(id, freeze(structuredClone(e)));
    for (const [p, selected] of pointChanges) {
      const reads = this.pointReads.get(p.risk_point_id) ?? new Map<string, string>();
      for (const [id, e] of selected) reads.set(id, evidenceIdentity(e));
      this.pointReads.set(p.risk_point_id, reads);
      this.markPointDirty(p.risk_point_id, "NEW_EVIDENCE");
      const point = this.point(p.risk_point_id);
      const windows = [...reads].flatMap(([id, identity]) => {
        const e = this.state.evidence.get(id);
        return e?.was_read && evidenceIdentity(e) === identity ? [e] : [];
      });
      // The existing compact bundle validator checks union coverage, exact
      // source/version/hash, Unicode offsets and overlapping quote identity.
      const anchors = this.queriesForPoint(p.risk_point_id).flatMap(q => (q.candidates ?? []).flatMap((c: JsonObject) =>
        (c.evidence ?? []).filter((anchor: JsonObject) => (c.required_evidence_ids ?? []).includes(anchor.evidence_id)
          && buildCitationOptions(this.task.snapshot.snapshot_id, [{ ...c, required_evidence_ids: [anchor.evidence_id], evidence: [anchor] }], windows).length === 1)
          .map((anchor: JsonObject) => anchor.evidence_id)));
      this.points.set(p.risk_point_id, freeze({ ...point, read_anchor_ids: [...new Set(anchors)] }));
    }
  }

  scope(branch: string, pointId: string | null = null, assessmentId: string | null = null): JsonObject {
    return { branch, risk_point_id: pointId, assessment_id: assessmentId, state_revision: pointId ? this.point(pointId).state_revision : 0, plan_revision: this.plan_revision };
  }
  recordDiagnostic(diagnostic: JsonObject): string {
    this.assertIdentity(); const d = structuredClone(diagnostic);
    const classes = ["typed_output", "branch_output", "identity", "permission", "budget", "transport", "program"];
    const categories = ["syntax", "schema", "semantic", "tool_format", "reference", "program"];
    const scopeFields = ["branch", "risk_point_id", "assessment_id", "state_revision", "plan_revision"];
    const protectedClass = d.code === "LOOP_REPORT_REFERENCE_INVALID" || /IDENTITY|SOURCE_HASH|EVIDENCE_MISMATCH/.test(d.code) ? "identity"
      : /BUDGET|DEADLINE|COST_CAP/.test(d.code) ? "budget" : /PERMISSION|AUTHORIZATION|FORBIDDEN/.test(d.code) ? "permission"
      : /TRANSPORT/.test(d.code) ? "transport" : null;
    if (Object.keys(d).some(k => !["scope", "code", "category", "blocker_class"].includes(k))
      || protectedClass && d.blocker_class !== protectedClass
      || !d.scope || scopeFields.some(k => !(k in d.scope)) || Object.keys(d.scope).some(k => !scopeFields.includes(k))
      || d.scope.risk_point_id !== null && !/^RP[0-9]+$/.test(d.scope.risk_point_id)
      || d.scope.assessment_id !== null && !/^AS[0-9]+$/.test(d.scope.assessment_id)
      || d.scope.state_revision < 0 || d.scope.plan_revision < 0
      || !classes.includes(d.blocker_class) || !categories.includes(d.category) || !/^[A-Z][A-Z0-9_]{0,79}$/.test(d.code)
      || !d.scope || !["plan", "initial_search", "assess", "act", "one_shot", "required_anchor_read", "final_read_guard", "report"].includes(d.scope.branch)
      || !Number.isSafeInteger(d.scope.state_revision) || !Number.isSafeInteger(d.scope.plan_revision)) throw new Error("LOOP_INVALID_DIAGNOSTIC");
    const id = hash(canonical(d)); this.attemptRecords.push(freeze(d)); this.active_blockers.set(id, freeze(d)); return id;
  }
  retireTypedBlockers(scope: JsonObject, beforeCommit?: () => void): void {
    this.assertIdentity(); const p = scope.risk_point_id && this.points.get(scope.risk_point_id), a = p && p.processing_state === "assessed" && p.accepted;
    const planAccepted = scope.branch === "plan" && this.acceptedPlanScope && canonical(scope) === canonical(this.acceptedPlanScope);
    const assessmentAccepted = scope.branch === "assess" && a && a.assessment_id === scope.assessment_id
      && a.state_revision === scope.state_revision && p.plan_revision === scope.plan_revision;
    if (!planAccepted && !assessmentAccepted) throw new Error("LOOP_RECOVERY_NOT_ACCEPTED");
    const matched = [...this.active_blockers].filter(([, d]) => d.blocker_class === "typed_output" && ["syntax", "schema"].includes(d.category) && canonical(d.scope) === canonical(scope));
    beforeCommit?.();
    for (const [id] of matched) {
      this.active_blockers.delete(id); this.blocker_retirements.push(freeze({ diagnostic_id: id, reason: "typed_recovered", replacement_plan_revision: this.plan_revision }));
    }
  }
  recordNativeArgumentFailure(scope: JsonObject, actionId: string, tool: string): string {
    this.assertIdentity();
    if (scope.branch !== "act" || scope.assessment_id !== null || !/^A[0-9]+$/.test(actionId)
      || !["search_cases", "read_source", "get_risk_definition"].includes(tool)) throw new Error("LOOP_INVALID_DIAGNOSTIC");
    const id = this.recordDiagnostic({ scope, code: "LOOP_NATIVE_ARGUMENT_SCHEMA_" + actionId, category: "tool_format", blocker_class: "typed_output" });
    this.nativeFailures.set(id, freeze({ action_id: actionId, tool })); return id;
  }
  /** Retire one native schema blocker only after a valid replacement tool and its audit exist. */
  retireNativeArgumentBlocker(diagnosticId: string, scope: JsonObject, rejectedActionId: string, acceptedActionId: string,
    params: JsonObject, response: ToolEnvelope, schemas: JsonObject, beforeCommit?: () => void): void {
    this.assertIdentity();
    const diagnostic = this.active_blockers.get(diagnosticId);
    const binding = this.nativeFailures.get(diagnosticId);
    const rejected = this.actions.find(a => a.action_id === rejectedActionId), accepted = this.actions.find(a => a.action_id === acceptedActionId);
    if (!binding || binding.action_id !== rejectedActionId || binding.tool !== accepted?.action_type
      || !diagnostic || diagnostic.blocker_class !== "typed_output" || diagnostic.category !== "tool_format" || diagnostic.scope.branch !== "act"
      || diagnostic.scope.assessment_id !== null || canonical(diagnostic.scope) !== canonical(scope)
      || !rejected || rejected.status !== "blocked" || !accepted || !["completed", "empty", "partial"].includes(accepted.status)
      || acceptedActionId === rejectedActionId || rejected.action_type !== accepted.action_type
      || canonical(rejected.scope) !== canonical(scope) || canonical(accepted.scope) !== canonical(scope)
      || response.status === "error" || response.meta.snapshot_id !== this.task.snapshot.snapshot_id) throw new Error("LOOP_NATIVE_RECOVERY_NOT_ACCEPTED");
    validateToolArguments(accepted.action_type, params, schemas); validateProgram(response, "tool-envelope");
    beforeCommit?.();
    this.active_blockers.delete(diagnosticId); this.nativeFailures.delete(diagnosticId);
    this.blocker_retirements.push(freeze({ diagnostic_id: diagnosticId, reason: "native_recovered", replacement_plan_revision: this.plan_revision }));
  }
  retireSupersededOneShotBlockers(options: { scope?: JsonObject; beforeCommit?: (diagnosticIds: readonly string[]) => void } = {}): void {
    this.assertIdentity();
    if (this.branch !== "plan" || ![...this.points.values()].every(p => this.currentAccepted(p))
      || !this.task.input.segments.every(s => this.segments.some(row => row.segment_id === s.segment_id && row.status === "reviewed"))) throw new Error("LOOP_RECOVERY_NOT_ACCEPTED");
    const matched = [...this.active_blockers].filter(([, d]) => d.blocker_class === "branch_output" && d.scope.branch === "one_shot"
      && d.scope.plan_revision < this.plan_revision && (!options.scope || canonical(d.scope) === canonical(options.scope)));
    options.beforeCommit?.(Object.freeze(matched.map(([id]) => id)));
    for (const [id] of matched) {
      this.active_blockers.delete(id); this.blocker_retirements.push(freeze({ diagnostic_id: id, reason: "branch_superseded", replacement_plan_revision: this.plan_revision }));
    }
  }
  currentAccepted(p: LoopPoint): AcceptedPoint | undefined {
    return p.processing_state === "assessed" && p.accepted?.state_revision === p.state_revision && p.accepted.candidate_version === p.candidate_version ? p.accepted : undefined;
  }
  reportToken(): string { this.assertIdentity(); return canonical({ plan: this.plan_revision, observations: this.observationRevision, active_blockers: [...this.active_blockers.keys()].sort(), stop_reason: this.state.stopReason, validation_errors: this.state.validationErrors, protocol_invalid: this.state.protocolInvalid, points: [...this.points.values()].map(p => [p.risk_point_id, p.state_revision, p.candidate_version, p.processing_state, p.accepted?.assessment_id ?? null, p.additional_mechanisms_uncovered]) }); }
  assertReportToken(token: string): void { if (this.reportToken() !== token) throw new Error("LOOP_STALE_REPORT"); }
}
