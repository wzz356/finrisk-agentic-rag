import { canonical, hash, type JsonObject } from "./contracts.js";

export const citationProtocolVersion = "finrisk-citation/1.0";
export interface CitationOption {
  case_ref: string;
  unit_id: string;
  source_id: string;
  required_evidence_ids: string[];
  citation_evidence_ids: string[];
}
function validSpan(e: JsonObject): boolean {
  return typeof e.evidence_id === "string" && e.evidence_id.length > 0 && e.source_verified === true && ["source_id", "text_version", "text_sha256"].every(k => typeof e[k] === "string" && e[k].length > 0)
    && Number.isInteger(e.start) && Number.isInteger(e.end) && e.start >= 0 && e.end > e.start
    && typeof e.quote === "string" && Array.from(e.quote).length === e.end - e.start;
}
function identity(e: JsonObject): JsonObject {
  return { source_id: e.source_id, text_version: e.text_version, text_sha256: e.text_sha256, start: e.start, end: e.end, quote: e.quote };
}
function overlapMatches(a: JsonObject, b: JsonObject): boolean {
  if (a.source_id !== b.source_id || a.text_version !== b.text_version || a.text_sha256 !== b.text_sha256) return false;
  const start = Math.max(a.start, b.start), end = Math.min(a.end, b.end);
  return start < end && Array.from(a.quote).slice(start - a.start, end - a.start).join("") === Array.from(b.quote).slice(start - b.start, end - b.start).join("");
}
export function buildCitationOptions(snapshotId: string, candidates: readonly JsonObject[], visibleEvidence: readonly JsonObject[]): CitationOption[] {
  if (typeof snapshotId !== "string" || !snapshotId.trim()) throw new Error("CITATION_SNAPSHOT_REQUIRED");
  const windows = visibleEvidence.filter(e => e.was_read === true && validSpan(e)).sort((a,b) => a.start - b.start || b.end - a.end || String(a.evidence_id).localeCompare(String(b.evidence_id)));
  const windowIds = new Map<string, string>();
  for (const w of windows) {
    const binding = canonical(identity(w));
    if (windowIds.has(w.evidence_id) && windowIds.get(w.evidence_id) !== binding) throw new Error("CITATION_EVIDENCE_BINDING_CONFLICT");
    windowIds.set(w.evidence_id, binding);
  }
  const units = new Map<string, string>(), refs = new Map<string, string>();
  const invalidUnits = new Set<string>();
  const options = new Map<string, CitationOption>();
  for (const c of candidates) {
    if (typeof c.unit_id !== "string" || !c.unit_id || typeof c.source_id !== "string" || !c.source_id
      || !Array.isArray(c.required_evidence_ids) || !c.required_evidence_ids.length || !Array.isArray(c.evidence)) { if (typeof c.unit_id === "string") invalidUnits.add(c.unit_id); continue; }
    const required: JsonObject[] = [];
    let valid = true;
    for (const id of c.required_evidence_ids) {
      const rows = c.evidence.filter((e: JsonObject) => e.evidence_id === id);
      const e = rows[0];
      if (!e || rows.some((row: JsonObject) => canonical(identity(row)) !== canonical(identity(e))) || !validSpan(e) || e.source_id !== c.source_id) { valid = false; break; }
      if (!required.some(r => canonical(identity(r)) === canonical(identity(e)))) required.push(e);
    }
    if (!valid || !required.length) { invalidUnits.add(c.unit_id); continue; }
    const first = required[0]!;
    if (required.some(e => e.text_version !== first.text_version || e.text_sha256 !== first.text_sha256)) { invalidUnits.add(c.unit_id); continue; }
    required.sort((a,b) => a.start - b.start || a.end - b.end || a.quote.localeCompare(b.quote));
    const binding = canonical({ snapshot_id: snapshotId, unit_id: c.unit_id, source_id: c.source_id,
      text_version: first.text_version, text_sha256: first.text_sha256, anchors: required.map(identity) });
    if (units.has(c.unit_id) && units.get(c.unit_id) !== binding) throw new Error("CITATION_UNIT_BINDING_CONFLICT");
    units.set(c.unit_id, binding);
    const case_ref = "C" + hash(binding).slice(0, 20);
    if (refs.has(case_ref) && refs.get(case_ref) !== binding) throw new Error("CITATION_REFERENCE_CONFLICT");
    refs.set(case_ref, binding);
    const selected = new Set<string>();
    for (const anchor of required) {
      const matching = windows.filter(w => overlapMatches(anchor, w));
      let covered = anchor.start;
      for (const window of matching) {
        if (window.end <= covered) continue;
        if (window.start > covered) break;
        covered = window.end; selected.add(window.evidence_id);
        if (covered >= anchor.end) break;
      }
      if (covered < anchor.end) { valid = false; break; }
    }
    if (!valid) continue;
    const selectedWindows = windows.filter(w => selected.has(w.evidence_id));
    if (selectedWindows.some((w, i) => selectedWindows.slice(0, i).some(prior => prior.start < w.end && w.start < prior.end && !overlapMatches(prior, w)))) continue;
    const ids = selectedWindows.map(w => w.evidence_id);
    options.set(case_ref, { case_ref, unit_id: c.unit_id, source_id: c.source_id,
      required_evidence_ids: required.map(e => e.evidence_id), citation_evidence_ids: [...new Set(ids)] });
  }
  return [...options.values()].filter(option => !invalidUnits.has(option.unit_id));
}
export function expandCaseSelections(matches: readonly JsonObject[], options: ReadonlyMap<string, CitationOption>): JsonObject[] {
  const seen = new Set<string>();
  const fields = ["case_ref", "same_points", "differences", "match_status", "supports_tags"];
  return matches.map(match => {
    if (!match || typeof match !== "object" || Object.keys(match).length !== fields.length || Object.keys(match).some(k => !fields.includes(k))
      || typeof match.case_ref !== "string" || !/^C[0-9a-f]{20}$/.test(match.case_ref)
      || !["reliable_match", "partial_match"].includes(match.match_status)
      || !["same_points", "differences", "supports_tags"].every(k => Array.isArray(match[k]) && match[k].every((v: unknown) => typeof v === "string" && v.length > 0))) throw new Error("INVALID_CASE_SELECTION");
    if (seen.has(match.case_ref)) throw new Error("DUPLICATE_CASE_REFERENCE");
    seen.add(match.case_ref);
    const option = options.get(match.case_ref);
    if (!option || option.case_ref !== match.case_ref || !option.citation_evidence_ids.length) throw new Error("CITATION_REFERENCE_NOT_VISIBLE");
    const { case_ref, ...assessment } = match;
    return { source_id: option.source_id, unit_id: option.unit_id, evidence_ids: [...option.citation_evidence_ids], ...assessment };
  });
}
