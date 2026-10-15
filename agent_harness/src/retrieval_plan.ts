import type { JsonObject } from "./contracts.js";

export function mechanismQueries(points: JsonObject[]): JsonObject[] {
  return points.map(p => ({ text: Array.from([...p.input_refs.map((r: JsonObject) => r.quote), ...(p.candidate_tags ?? [])].join(" ")).slice(0, 500).join(""), focus_refs: p.input_refs.slice(0, 8), risk_point_ids: p.risk_point_id ? [p.risk_point_id] : [], goal: "mechanism_search" }));
}

export function sourceReadPlan(queries: JsonObject[], maxSources = 3, isRead?: (evidence: JsonObject) => boolean): JsonObject[] {
  if (queries.some(q => (q.candidates ?? []).some((c: JsonObject) => c.required_evidence_ids !== undefined))) {
    const selected: JsonObject[] = []; const seen = new Set<string>();
    const groups = queries.map(q => (q.candidates ?? []).flatMap((c: JsonObject) => {
      const required: string[] = c.required_evidence_ids ?? (c.evidence ?? []).slice(0, 1).map((e: JsonObject) => e.evidence_id);
      return required.flatMap(id => {
        const e = (c.evidence ?? []).find((row: JsonObject) => row.evidence_id === id);
        return e && !isRead?.(e) ? [{ ...c, evidence: [e, ...(c.evidence ?? []).filter((row: JsonObject) => row.evidence_id !== id)] }] : [];
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
