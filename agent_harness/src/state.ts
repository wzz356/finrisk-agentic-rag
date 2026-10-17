import type { CitationOption } from "./citation_bundles.js";
import { validateRankingCoverage } from "./retrieval_quality.js";
import { hash, canonical, MODEL_WIRE_VERSION, type InputRecord, type JsonObject, type Phase, type ToolEnvelope } from "./contracts.js";
export class TaskState {
  phase: Phase = "analyzing";
  analysis?: JsonObject;
  final?: JsonObject;
  stopReason = "evidence_sufficient";
  repairs = 0;
  toolFormatCorrections = 0;
  repairing = false;
  validationErrors: string[] = [];
  protocolInvalid = false;
  protocolVersion = "finrisk-model/1.0";
  requestedProtocolVersion = MODEL_WIRE_VERSION;
  runtimeBudget?: JsonObject;
  contextEvidenceIds = new Set<string>();
  evaluationCitations = false;
  retrievalReliability = false;
  expectedRerankProtocol?: string;
  citationSnapshotId?: string;
  citationOptions = new Map<string, CitationOption>();
  caseTagBindings = new Map<string, JsonObject[]>();
  points: JsonObject[] = [];
  evidence = new Map<string, JsonObject>();
  searches: JsonObject[] = [];
  noProgress = new Map<string, number>();
  appliedSearchActions = new Set<string>();
  gaps = new Map<string, JsonObject>();
  constructor(public input: InputRecord, public noProgressLimit = 2) {}
  resolve(ref: JsonObject): JsonObject {
    const segment = this.input.segments.find(s => s.segment_id === ref.segment_id);
    if (!segment || typeof ref.quote !== "string" || !ref.quote.length || !Number.isInteger(ref.occurrence) || ref.occurrence < 1) throw new Error("INVALID_INPUT_REFERENCE");
    const chars = Array.from(segment.text); const quote = Array.from(ref.quote); let seen = 0;
    for (let i = 0; i <= chars.length - quote.length; i++) {
      if (chars.slice(i, i + quote.length).join("") === ref.quote && ++seen === ref.occurrence) {
        const start = segment.start + i;
        return { source_id: this.input.source_id, source_kind: "input", segment_id: segment.segment_id,
          text_version: this.input.text_version, text_sha256: this.input.text_sha256, start, end: start + quote.length, quote: ref.quote };
      }
    }
    throw new Error("INVALID_INPUT_REFERENCE");
  }
  group(query: JsonObject): string {
    // ID不独立提供免费额度；无RP的查询按相同输入区间分组。
    return query.risk_point_ids.length ? [...query.risk_point_ids].sort().join(",") : hash(canonical(query.focus_refs.map((r: JsonObject) => this.resolve(r))));
  }
  queryKey(query: JsonObject): string {
    return canonical({ group: this.group(query), goal: query.goal, text: String(query.text).trim().replace(/\s+/g, " ") });
  }
  acceptAnalysis(value: JsonObject): void {
    this.validateRefs(value);
    this.analysis = value;
    this.points = value.risk_points.map((p: JsonObject, i: number) => ({ ...p, risk_point_id: `RP${String(i + 1).padStart(4, "0")}`, status: "pending" }));
    for (const point of this.points) for (const unknown of point.unknowns) {
      const id = `G${String(this.gaps.size + 1).padStart(4, "0")}`;
      this.gaps.set(id, { gap_id: id, risk_point_id: point.risk_point_id, type: "input_condition", statement: unknown, status: "unresolved" });
    }
    this.phase = "reviewing";
  }
  validateRefs(value: unknown): void {
    if (Array.isArray(value)) { for (const item of value) this.validateRefs(item); return; }
    if (!value || typeof value !== "object") return;
    const row = value as JsonObject;
    if ("segment_id" in row && "quote" in row) this.resolve(row);
    if ("context_refs" in row) for (const ref of row.context_refs) {
      const key = String(ref).replace("$.user_context.", "");
      if (!String(ref).startsWith("$.user_context.") || !this.input.user_context[key]) throw new Error("INVALID_CONTEXT_REFERENCE");
    }
    if ("segment_id" in row && !("quote" in row) && !this.input.segments.some(s => s.segment_id === row.segment_id)) throw new Error("INVALID_SEGMENT_REFERENCE");
    if ("unprocessed_segment_ids" in row && row.unprocessed_segment_ids.some((id: string) => !this.input.segments.some(s => s.segment_id === id))) throw new Error("INVALID_SEGMENT_REFERENCE");
    for (const child of Object.values(row)) this.validateRefs(child);
  }
  acceptTool(name: string, params: JsonObject, value: ToolEnvelope, actionId?: string): void {
    if (name === 'search_cases' && actionId) {
      if (this.appliedSearchActions.has(actionId)) return;
    }
    if (name === "search_cases") {
      for (const query of value.data?.queries ?? []) {
        validateRankingCoverage(query, this.retrievalReliability);
        if (this.retrievalReliability && query.ranking_coverage && this.expectedRerankProtocol
          && query.ranking_coverage.rerank_protocol !== this.expectedRerankProtocol) throw new Error("RERANK_PROTOCOL_BINDING_MISMATCH");
      }
    }
    if (this.evaluationCitations) {
      const identity = (row: JsonObject) => canonical(Object.fromEntries(["source_id", "source_kind", "text_version", "text_sha256", "start", "end", "quote", "source_verified"].map(key => [key, row[key]])));
      const registered = new Map([...this.evidence].map(([id, row]) => [id, identity(row)]));
      const preflight = (data: unknown): void => {
        if (Array.isArray(data)) { data.forEach(preflight); return; }
        if (!data || typeof data !== "object") return;
        const row = data as JsonObject;
        if (typeof row.evidence_id === "string" && typeof row.quote === "string") {
          const binding = identity(row);
          if (registered.has(row.evidence_id) && registered.get(row.evidence_id) !== binding) throw new Error("EVIDENCE_IDENTITY_CONFLICT");
          registered.set(row.evidence_id, binding);
        }
        Object.values(row).forEach(preflight);
      };
      // 先验证整个结果，冲突不得留下已读继承、半更新证据或搜索幂等标记。
      preflight(value.data);
    }
    if (name === 'search_cases' && actionId) this.appliedSearchActions.add(actionId);
    const visit = (data: unknown): void => {
      if (Array.isArray(data)) { data.forEach(visit); return; }
      if (!data || typeof data !== "object") return;
      const row = data as JsonObject;
      if (row.evidence_id && row.quote && row.source_verified) {
        const prior = this.evidence.get(row.evidence_id);
        this.evidence.set(row.evidence_id, { ...prior, ...row, was_read: name === "read_source" || Boolean(prior?.was_read), anchor_not_fully_covered: name === "read_source" ? value.data?.anchor_not_fully_covered ?? false : prior?.anchor_not_fully_covered ?? false });
      }
      Object.values(row).forEach(visit);
    };
    visit(value.data);
    if (name === "search_cases" && value.data?.queries) {
      const completedGroups = new Set<string>();
      for (const query of value.data.queries) {
        // 新引文是否语义相关由DecisionNote确认，单纯新增来源不清零无进展计数。
        if (query.complete === true && ['ok', 'empty'].includes(query.status) && !query.no_progress_blocked) completedGroups.add(this.group(query));
        this.searches.push(query);
      }
      for (const key of completedGroups) this.noProgress.set(key, (this.noProgress.get(key) ?? 0) + 1);
    }
  }
  acceptDecision(value: JsonObject): void {
    this.validateRefs(value);
    const focused = this.searches.filter(q => q.focus_refs.some((r: JsonObject) => value.focus_refs.some((f: JsonObject) => canonical(f) === canonical(r))));
    // 所有附加笔记先验证完毕；非法后续项不能留下已接受证据或gap的半更新。
    for (const id of value.accepted_evidence_ids) {
      if (!this.evidence.has(id)) throw new Error("INVALID_EVIDENCE_REFERENCE");
    }
    for (const update of value.gap_updates) {
      const gap = this.gaps.get(update.gap_id);
      if (!gap) throw new Error("INVALID_GAP_REFERENCE");
      if (gap.type === "input_condition" && update.status === "resolved" && !update.input_refs.length) throw new Error("HISTORICAL_CASE_CANNOT_FILL_INPUT_FACT");
      if (update.evidence_ids.some((id: string) => !this.evidence.has(id))) throw new Error("INVALID_EVIDENCE_REFERENCE");
    }
    for (const id of value.accepted_evidence_ids) {
      // 同一证据重复接受不算新进展。
      const evidence = this.evidence.get(id)!;
      if (!evidence.accepted_as_relevant) {
        evidence.accepted_as_relevant = true;
        for (const query of focused) this.noProgress.set(this.group(query), 0);
      }
    }
    for (const update of value.gap_updates) {
      Object.assign(this.gaps.get(update.gap_id)!, update);
    }
  }
}
