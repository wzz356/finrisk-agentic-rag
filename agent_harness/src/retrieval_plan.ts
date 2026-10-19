import type { JsonObject } from "./contracts.js";

export function mechanismQueries(points: JsonObject[]): JsonObject[] {
  return points.map(p => ({ text: Array.from([...p.input_refs.map((r: JsonObject) => r.quote), ...(p.candidate_tags ?? [])].join(" ")).slice(0, 500).join(""), focus_refs: p.input_refs.slice(0, 8), risk_point_ids: p.risk_point_id ? [p.risk_point_id] : [], goal: "mechanism_search" }));
}

function validContextSpan(e: JsonObject): boolean {
  return typeof e.evidence_id === "string" && e.evidence_id.length > 0 && e.source_verified === true
    && ["source_id", "text_version"].every(k => typeof e[k] === "string" && e[k].length > 0)
    && typeof e.text_sha256 === "string" && /^[0-9a-f]{64}$/i.test(e.text_sha256)
    && Number.isSafeInteger(e.start) && Number.isSafeInteger(e.end) && e.start >= 0 && e.end > e.start
    && typeof e.quote === "string" && Array.from(e.quote).length === e.end - e.start;
}

function verifiedParent(c: JsonObject, anchor: JsonObject): JsonObject | undefined {
  const meta = c.context_expansion;
  const ids = (value: unknown) => Array.isArray(value) && value.every(id => typeof id === "string" && id.length > 0);
  if (!meta || typeof meta !== "object" || Array.isArray(meta) || meta.policy !== "verified-parent-v1"
    || typeof meta.policy_sha256 !== "string" || !/^[0-9a-f]{64}$/i.test(meta.policy_sha256)
    || meta.status !== "expanded" || meta.reason !== "VERIFIED_PARENT"
    || !ids(meta.parent_evidence_ids) || !meta.parent_evidence_ids.length || !ids(meta.added_evidence_ids)
    || meta.added_evidence_ids.some((id: string) => !meta.parent_evidence_ids.includes(id))
    || !Number.isInteger(meta.added_chars) || meta.added_chars < 0 || meta.added_chars > 600
    || !validContextSpan(anchor) || anchor.source_id !== c.source_id) return undefined;
  return meta.parent_evidence_ids.map((id: string) => c.evidence.find((row: JsonObject) => row.evidence_id === id))
    .find((parent: JsonObject | undefined) => parent && validContextSpan(parent) && parent.source_id === anchor.source_id
      && parent.text_version === anchor.text_version && parent.text_sha256 === anchor.text_sha256
      && parent.end - parent.start <= 900 && parent.start <= anchor.start && parent.end >= anchor.end
      && Array.from(parent.quote).slice(anchor.start - parent.start, anchor.end - parent.start).join("") === anchor.quote);
}

export function sourceReadPlan(queries: JsonObject[], maxSources = 3, isRead?: (evidence: JsonObject) => boolean): JsonObject[] {
  if (queries.some(q => (q.candidates ?? []).some((c: JsonObject) => c.required_evidence_ids !== undefined))) {
    const selected: JsonObject[] = []; const seen = new Set<string>();
    const groups = queries.map(q => (q.candidates ?? []).flatMap((c: JsonObject) => {
      const required: string[] = c.required_evidence_ids ?? (c.evidence ?? []).slice(0, 1).map((e: JsonObject) => e.evidence_id);
      return required.flatMap(id => {
        const e = (c.evidence ?? []).find((row: JsonObject) => row.evidence_id === id);
        if (!e || isRead?.(e)) return [];
        const parent = c.required_evidence_ids !== undefined ? verifiedParent(c, e) : undefined;
        const target = parent && !isRead?.(parent) ? parent : e;
        return [{ ...c, evidence: [target, ...(c.evidence ?? []).filter((row: JsonObject) => row.evidence_id !== target.evidence_id)] }];
      });
    }));
    const take = (c?: JsonObject) => {
      if (!c || selected.length >= maxSources) return;
      const key = `${c.source_id}:${c.evidence[0].evidence_id}`;
      if (!seen.has(key)) { seen.add(key); selected.push(c); }
    };
    for (const group of groups) take(group[0]);
    for (const group of groups) for (const c of group) take(c);
    return selected;
  }
  const chosen: JsonObject[] = []; const sources = new Set<string>(); const fingerprints = new Set<string>();
  const group = (c: JsonObject) => c.duplicate_hint?.kind === "exact_fact_text" ? c.duplicate_hint.fact_text_sha256 : undefined;
  const available = (c: JsonObject, diverse = true) => c.source_id && c.evidence?.length && !sources.has(c.source_id) && (!diverse || !group(c) || !fingerprints.has(group(c)));
  const take = (c?: JsonObject) => { if (!c || chosen.length >= maxSources) return; chosen.push(c); sources.add(c.source_id); if (group(c)) fingerprints.add(group(c)); };
  // 每个输入点先读一条；完全相同事实文字只作排序线索，不合并案件或删除来源。
  for (const q of queries) take((q.candidates ?? []).find((c: JsonObject) => available(c)) ?? (q.candidates ?? []).find((c: JsonObject) => available(c, false)));
  for (const q of queries) for (const c of q.candidates ?? []) if (available(c)) take(c);
  for (const q of queries) for (const c of q.candidates ?? []) if (available(c, false)) take(c);
  return chosen;
}
