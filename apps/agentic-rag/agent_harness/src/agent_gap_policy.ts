import { canonical, hash, type JsonObject } from "./contracts.js";
import { candidateBundleRead } from "./model_protocol.js";
import { isPartialRerank, retrievalReliabilityPolicy } from "./retrieval_quality.js";
import type { AcceptedPoint, LoopPoint, LoopRegistry } from "./agent_loop_state.js";

export type PointGapType = "case_search" | "case_comparison" | "unread_source" | "input_fact" | "ranking_limited" | "execution_blocked";
export type AllowedTool = "read_source" | "search_cases";
export interface PointGap {
  gap_id: string;
  point_id: string;
  tag?: string;
  type: PointGapType;
  basis: { plan_revision: number; state_revision: number; candidate_version: number; assessment_revision: number };
  allowed_actions: AllowedTool[];
  condition_ids?: string[];
  targets?: JsonObject[];
  query_keys?: string[];
  blocker_codes?: string[];
  local_partial?: boolean;
}
export interface NoProgressState { fingerprint: string; stagnant_rounds: number }
export interface PolicyControls {
  transport_circuit_open?: boolean;
  execution_blockers?: readonly string[];
}
/** Counters and the reevaluation reserve come from the caller's existing ledger. */
export interface ActionAllowance extends PolicyControls {
  reassessment_reserved: boolean;
  feedback_rounds_used?: number;
  searches_this_round?: number;
  remaining_search_calls?: number;
  remaining_query_attempts?: number;
  remaining_tool_actions?: number;
  no_progress?: NoProgressState;
}
export interface AllowedAction {
  point_id: string;
  tag?: string;
  tool: AllowedTool;
  gap_ids: string[];
  targets?: JsonObject[];
  max_queries?: 1;
  excluded_query_texts?: string[];
}
export type PolicyStopReason = "execution_blocked" | "no_progress" | "budget_reserve_missing" | "feedback_limit" | "search_limit"
  | "input_facts_missing" | "retrieval_partial_exhausted" | "case_evidence_insufficient" | "additional_mechanisms_uncovered" | "evidence_sufficient";
export interface PolicyDecision {
  gaps: PointGap[];
  actions: AllowedAction[];
  stop_reason: PolicyStopReason | null;
  unfinished_point_ids: string[];
  assessment_required_point_ids: string[];
  local_ranking_limited: boolean;
}
function latestQueries(registry: LoopRegistry, point: LoopPoint): JsonObject[] {
  return [...new Map(registry.queriesForPoint(point.risk_point_id).map(q => [registry.state.queryKey(q), q])).values()];
}
function blockers(registry: LoopRegistry, controls: PolicyControls): string[] {
  const codes = [...(controls.execution_blockers ?? []), ...[...registry.active_blockers.values()].map(d => String(d.code))];
  try { registry.assertIdentity(); } catch { codes.push("LOOP_IDENTITY_CHANGED"); }
  if (controls.transport_circuit_open) codes.push("TRANSPORT_CIRCUIT_OPEN");
  if (registry.state.protocolInvalid) codes.push("PROTOCOL_INVALID");
  codes.push(...registry.state.validationErrors.filter(c => !retrievalReliabilityPolicy.limited_error_codes.includes(c)));
  if (["budget_exhausted", "deadline_exceeded", "tool_failure", "model_failure", "invalid_output", "diagnostic_write_failed", "context_capacity_reached", "STOPPED"].includes(registry.state.stopReason)) codes.push(registry.state.stopReason);
  for (const q of registry.state.searches) if (q.error?.code && /STOPPED|CIRCUIT|AUDIT|CHECKPOINT|HASH|IDENTITY|EVIDENCE.*MISMATCH|BUDGET|DEADLINE|PERMISSION|FORBIDDEN|AUTHORIZATION|TRANSPORT|PROGRAM/.test(q.error.code)) codes.push(q.error.code);
  return [...new Set(codes)].sort();
}
function gap(point: LoopPoint, type: PointGapType, allowed_actions: AllowedTool[], details: Partial<PointGap> = {}): PointGap {
  const basis = { plan_revision: point.plan_revision, state_revision: point.state_revision,
    candidate_version: point.candidate_version, assessment_revision: point.assessment_revision };
  const identity = { point_id: point.risk_point_id, type, tag: details.tag ?? null, basis,
    condition_ids: details.condition_ids ?? [], query_keys: details.query_keys ?? [], targets: details.targets ?? [], blocker_codes: details.blocker_codes ?? [] };
  return { gap_id: "G" + hash(canonical(identity)).slice(0, 20), point_id: point.risk_point_id, type, basis, allowed_actions, ...details };
}
function target(c: JsonObject): JsonObject {
  return { source_id: c.source_id, unit_id: c.unit_id, required_evidence_ids: [...(c.required_evidence_ids ?? [])],
    evidence: structuredClone(c.evidence ?? []) };
}
function comparisons(accepted: AcceptedPoint, tag: string): JsonObject[] {
  return accepted.tag_assessments.find(t => t.tag === tag)?.case_assessments ?? [];
}
/** Derive only from current program state; model-authored repeated gaps are never read. */
export function derivePointGaps(registry: LoopRegistry, controls: PolicyControls = {}): PointGap[] {
  const result: PointGap[] = [], global = blockers(registry, controls);
  for (const point of registry.points.values()) {
    const accepted = registry.currentAccepted(point);
    const gapPoint = { ...point, assessment_revision: accepted?.assessment_revision ?? 0 };
    if (global.length) result.push(gap(gapPoint, "execution_blocked", [], { blocker_codes: global }));
    // An identity failure must not invoke query ownership on the changed registry.
    const queries = global.includes("LOOP_IDENTITY_CHANGED") ? [] : latestQueries(registry, point);
    const failures = queries.filter(q => q.error || q.status === "error" || !q.complete && !isPartialRerank(q));
    if (!global.length && failures.length) result.push(gap(gapPoint, "execution_blocked", [], {
      query_keys: failures.map(q => registry.state.queryKey(q)), blocker_codes: failures.map(q => q.error?.code ?? "SEARCH_NOT_COMPLETED") }));
    const limited = queries.filter(isPartialRerank);
    if (limited.length && !global.length && !failures.length) result.push(gap(gapPoint, "ranking_limited", [], {
      query_keys: limited.map(q => registry.state.queryKey(q)), local_partial: true }));
    for (const tag of point.candidate_tags) {
      const check = accepted?.result.tag_checks.find((t: JsonObject) => t.tag === tag);
      if (check?.status === "not_supported") continue;
      const unknowns = check?.conditions.filter((c: JsonObject) => c.status === "unknown") ?? [];
      if (unknowns.length) result.push(gap(gapPoint, "input_fact", [], { tag,
        condition_ids: unknowns.map((c: JsonObject) => c.condition_id).sort() }));
      // Historical cases cannot resolve a condition about the current input.
      if (unknowns.length) continue;
      const cases = accepted ? comparisons(accepted, tag) : [];
      if (cases.length && cases.every(c => c.match_status === "reliable_match")) continue;
      const candidates: JsonObject[] = queries.flatMap(q => q.candidates ?? []);
      const unread = candidates.filter(c => !candidateBundleRead(c, registry.state));
      const read = candidates.some(c => candidateBundleRead(c, registry.state));
      const permitted = !global.length && !failures.length;
      if (!accepted && read) continue;
      if (unread.length) result.push(gap(gapPoint, "unread_source", permitted ? ["read_source"] : [], { tag,
        targets: [...new Map(unread.map(c => [canonical(target(c)), target(c)])).values()] }));
      else result.push(gap(gapPoint, cases.length ? "case_comparison" : "case_search", permitted ? ["search_cases"] : [], { tag }));
    }
  }
  return result;
}
function exhausted(value: number | undefined): boolean { return value !== undefined && (!Number.isSafeInteger(value) || value <= 0); }
/** Candidates are permissions, not instructions or a generated query. One action is chosen by the caller. */
export function chooseAllowedActions(registry: LoopRegistry, allowance: ActionAllowance): PolicyDecision {
  const gaps = derivePointGaps(registry, allowance);
  const decision: PolicyDecision = { gaps, actions: [], stop_reason: null,
    unfinished_point_ids: [...registry.points.values()].filter(p => p.additional_mechanisms_uncovered || !registry.currentAccepted(p) || gaps.some(g => g.point_id === p.risk_point_id)).map(p => p.risk_point_id),
    assessment_required_point_ids: [],
    local_ranking_limited: gaps.some(g => g.type === "ranking_limited" && g.local_partial) };
  if (blockers(registry, allowance).length || gaps.some(g => g.type === "execution_blocked")) {
    decision.local_ranking_limited = false;
    for (const g of gaps) if (g.type === "ranking_limited") g.local_partial = false;
    decision.stop_reason = "execution_blocked"; return decision;
  }
  if ((allowance.no_progress?.stagnant_rounds ?? 0) >= 2) { decision.stop_reason = "no_progress"; return decision; }
  decision.assessment_required_point_ids = [...registry.points.values()].filter(p => !registry.currentAccepted(p)
    && (p.draft || latestQueries(registry, p).some(q => (q.candidates ?? []).some((c: JsonObject) => candidateBundleRead(c, registry.state))))).map(p => p.risk_point_id);
  const actionable = gaps.filter(g => g.allowed_actions.length);
  if ((actionable.length || decision.assessment_required_point_ids.length) && !allowance.reassessment_reserved) { decision.stop_reason = "budget_reserve_missing"; return decision; }
  if (decision.assessment_required_point_ids.length) return decision;
  if (actionable.length && (allowance.feedback_rounds_used ?? 0) >= 2) { decision.stop_reason = "feedback_limit"; return decision; }
  if (actionable.length && exhausted(allowance.remaining_tool_actions)) { decision.stop_reason = "search_limit"; return decision; }
  const priority = (g: PointGap) => g.type === "unread_source" ? 0 : g.type === "case_search" ? 1 : 2;
  const first = actionable.length ? Math.min(...actionable.map(priority)) : -1;
  for (const g of actionable.filter(g => priority(g) === first)) for (const tool of g.allowed_actions) {
    if (tool === "search_cases" && ((allowance.searches_this_round ?? 0) >= 1 || exhausted(allowance.remaining_search_calls) || exhausted(allowance.remaining_query_attempts))) continue;
    const action: AllowedAction = { point_id: g.point_id, ...(g.tag ? { tag: g.tag } : {}), tool, gap_ids: [g.gap_id] };
    if (tool === "read_source") action.targets = structuredClone(g.targets ?? []);
    if (tool === "search_cases") {
      action.max_queries = 1;
      action.excluded_query_texts = [...new Set(latestQueries(registry, registry.points.get(g.point_id)!).map(q => String(q.text).trim().replace(/\s+/g, " ")))];
    }
    decision.actions.push(action);
  }
  if (decision.actions.length) return decision;
  decision.stop_reason = actionable.length ? "search_limit" : gaps.some(g => g.type === "input_fact") ? "input_facts_missing"
    : decision.local_ranking_limited ? "retrieval_partial_exhausted" : gaps.some(g => ["case_search", "case_comparison", "unread_source"].includes(g.type)) ? "case_evidence_insufficient" : [...registry.points.values()].some(p => p.additional_mechanisms_uncovered) ? "additional_mechanisms_uncovered" : "evidence_sufficient";
  return decision;
}
/** Revision counters, candidate counts and free text are deliberately absent. */
export function progressFingerprint(registry: LoopRegistry): string {
  registry.assertIdentity();
  const readSources = [...new Set([...registry.readEvidence.values()].filter(e => e.was_read).map(e =>
    canonical({ source_id: e.source_id, text_version: e.text_version, text_sha256: e.text_sha256 })))].sort();
  const points = [...registry.points.values()].map(p => {
    const accepted = registry.currentAccepted(p) ?? p.draft;
    return { point_id: p.risk_point_id, read_anchor_ids: [...(p.read_anchor_ids ?? [])].sort(),
      tags: (accepted?.result.tag_checks ?? []).map((t: JsonObject) => ({ tag: t.tag, status: t.status,
        conditions: t.conditions.map((c: JsonObject) => ({ id: c.condition_id, status: c.status,
          input_refs: [...new Set(c.input_refs.map((r: JsonObject) => canonical(r)))].sort(), context_refs: [...new Set(c.context_refs)].sort() })).sort((a: JsonObject, b: JsonObject) => a.id.localeCompare(b.id)),
        cases: comparisons(accepted!, t.tag).map(c => ({ source_id: c.source_id, unit_id: c.unit_id,
          case_ref: c.case_ref, match_status: c.match_status })).sort((a: JsonObject, b: JsonObject) => canonical(a).localeCompare(canonical(b)))
      })).sort((a: JsonObject, b: JsonObject) => a.tag.localeCompare(b.tag)) };
  });
  return hash(canonical({ read_sources: readSources, points, coverage: registry.segments.map(s => ({ segment_id: s.segment_id, status: s.status })) }));
}
/** Call once after a completed feedback round; establishing the baseline is not a round. */
export function updateNoProgress(previous: NoProgressState | undefined, fingerprint: string): NoProgressState {
  return { fingerprint, stagnant_rounds: previous?.fingerprint === fingerprint ? previous.stagnant_rounds + 1 : 0 };
}
