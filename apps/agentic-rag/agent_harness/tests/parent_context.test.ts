import test from "node:test";
import assert from "node:assert/strict";
import { hash, type JsonObject } from "../src/contracts.js";
import { buildCitationOptions } from "../src/citation_bundles.js";
import { candidateRead } from "../src/model_protocol.js";
import { sourceReadPlan } from "../src/retrieval_plan.js";

const text = "甲😀乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉。".repeat(50);
function fixture(source = "SOURCE", ranges = [[2, 5], [12, 15]], parentRanges = [[0, 20]]) {
  const evidence = (id: string, start: number, end: number): JsonObject => ({
    evidence_id: `${source}:${id}`, source_id: source, start, end,
    quote: Array.from(text).slice(start, end).join(""), text_version: "unit",
    text_sha256: hash(text), source_verified: true,
  });
  const anchors = ranges.map(([start, end], i) => evidence(`A${i}`, start!, end!));
  const parents = parentRanges.map(([start, end], i) => evidence(`P${i}`, start!, end!));
  const candidate: JsonObject = {
    source_id: source, unit_id: `UNIT:${source}`, rank: 1, match_reason: { tags: ["赠送利益"] },
    required_evidence_ids: anchors.map(e => e.evidence_id), evidence: [...anchors, ...parents],
    context_expansion: { policy: "verified-parent-v1", policy_sha256: hash("policy"), status: "expanded",
      reason: "VERIFIED_PARENT", parent_evidence_ids: parents.map(e => e.evidence_id),
      added_evidence_ids: parents.map(e => e.evidence_id), added_chars: 14 },
  };
  return { candidate, anchors, parents, evidence };
}
const queries = (...candidates: JsonObject[]) => [{ candidates }];
const plannedIds = (rows: JsonObject[]) => rows.map(c => c.evidence[0].evidence_id);

test("one verified parent covers multiple required anchors without changing candidate bindings", () => {
  const { candidate, anchors, parents } = fixture();
  const before = structuredClone(candidate);
  const plan = sourceReadPlan(queries(candidate));
  assert.deepEqual(plannedIds(plan), [parents[0]!.evidence_id]);
  assert.notEqual(plan[0], candidate);
  assert.deepEqual(candidate, before);
  assert.equal(plan[0]!.required_evidence_ids, candidate.required_evidence_ids);
  assert.equal(plan[0]!.context_expansion, candidate.context_expansion);
  assert.deepEqual(plan[0]!.evidence, [parents[0], ...anchors]);
  assert.deepEqual({ ...plan[0], evidence: candidate.evidence }, candidate);
});

test("parent metadata remains unread and case_ref still binds the original required anchors", () => {
  const { candidate, anchors, parents } = fixture();
  const state = { evidence: new Map(candidate.evidence.map((e: JsonObject) => [e.evidence_id, e])) } as Parameters<typeof candidateRead>[1];
  assert.equal(candidateRead(parents[0]!, state), false);
  assert.deepEqual(buildCitationOptions("SNAPSHOT", [candidate], candidate.evidence), []);
  assert.deepEqual(plannedIds(sourceReadPlan(queries(candidate), 3, e => candidateRead(e, state))), [parents[0]!.evidence_id]);
  const baseline = buildCitationOptions("SNAPSHOT", [candidate], anchors.map(e => ({ ...e, was_read: true })))[0]!;
  const readWindow = { ...parents[0]!, evidence_id: "REAL_READ_WINDOW", was_read: true };
  state.evidence.set(readWindow.evidence_id, readWindow);
  assert.deepEqual(sourceReadPlan(queries(candidate), 3, e => candidateRead(e, state)), []);
  const accepted = buildCitationOptions("SNAPSHOT", [candidate], [readWindow]);
  assert.equal(accepted.length, 1);
  assert.equal(accepted[0]!.case_ref, baseline.case_ref);
  assert.deepEqual(accepted[0]!.required_evidence_ids, candidate.required_evidence_ids);
  assert.deepEqual(accepted[0]!.citation_evidence_ids, [readWindow.evidence_id]);
});

test("an existing required parent can be preferred when no evidence was added", () => {
  const { candidate, anchors, parents } = fixture("EXISTING", [[2, 5]], [[0, 20]]);
  candidate.required_evidence_ids.push(parents[0]!.evidence_id);
  candidate.context_expansion.added_evidence_ids = [];
  candidate.context_expansion.added_chars = 0;
  const before = structuredClone(candidate);
  assert.deepEqual(plannedIds(sourceReadPlan(queries(candidate))), [parents[0]!.evidence_id]);
  assert.deepEqual(candidate, before);
  assert.deepEqual(candidate.required_evidence_ids, [anchors[0]!.evidence_id, parents[0]!.evidence_id]);
});

test("invalid parent identity, coordinates, containment or quote falls back to the original anchor", () => {
  const invalid: Array<[string, (c: JsonObject, p: JsonObject) => void]> = [
    ["hash", (_c, p) => { p.text_sha256 = hash("different source"); }],
    ["version", (_c, p) => { p.text_version = "other"; }],
    ["source", (_c, p) => { p.source_id = "OTHER"; }],
    ["unverified", (_c, p) => { p.source_verified = false; }],
    ["fractional offset", (_c, p) => { p.start = 0.5; }],
    ["string offset", (_c, p) => { p.start = "0"; }],
    ["negative offset", (_c, p) => { p.start = -1; }],
    ["empty span", (_c, p) => { p.end = p.start; p.quote = ""; }],
    ["character length", (_c, p) => { p.quote += "字"; }],
    ["over 900", (_c, p) => { p.end = 901; p.quote = Array.from(text).slice(0, 901).join(""); }],
    ["partial coverage", (_c, p) => { p.end = 4; p.quote = Array.from(p.quote).slice(0, 4).join(""); }],
    ["overlap text", (_c, p) => { p.quote = "伪".repeat(p.end - p.start); }],
    ["missing parent evidence", c => { c.evidence.pop(); }],
    ["anchor unverified", c => { c.evidence[0].source_verified = false; }],
    ["anchor character length", c => { c.evidence[0].quote += "字"; }],
    ["candidate source", c => { c.source_id = "OTHER"; }],
  ];
  const boundary = fixture("BOUNDARY", [[2, 5]], [[0, 900]]);
  assert.deepEqual(plannedIds(sourceReadPlan(queries(boundary.candidate))), [boundary.parents[0]!.evidence_id]);
  for (const [name, mutate] of invalid) {
    const { candidate, anchors, parents } = fixture("INVALID", [[2, 5]]);
    mutate(candidate, parents[0]!);
    assert.deepEqual(plannedIds(sourceReadPlan(queries(candidate))), [anchors[0]!.evidence_id], name);
  }
});

test("missing or invalid expansion metadata retains anchor planning", () => {
  const invalid: Array<[string, (c: JsonObject) => void]> = [
    ["missing", c => { delete c.context_expansion; }],
    ["null", c => { c.context_expansion = null; }],
    ["policy", c => { c.context_expansion.policy = "other-policy"; }],
    ["policy hash", c => { c.context_expansion.policy_sha256 = "not-sha256"; }],
    ["skipped", c => { c.context_expansion.status = "skipped"; }],
    ["budget omitted", c => { c.context_expansion.status = "budget_omitted"; }],
    ["reason", c => { c.context_expansion.reason = "UNVERIFIED_PARENT"; }],
    ["parent ids", c => { c.context_expansion.parent_evidence_ids = "SOURCE:P0"; }],
    ["empty parent ids", c => { c.context_expansion.parent_evidence_ids = []; }],
    ["malformed parent id", c => { c.context_expansion.parent_evidence_ids.push(null); }],
    ["added ids", c => { c.context_expansion.added_evidence_ids = null; }],
    ["unbound added id", c => { c.context_expansion.added_evidence_ids = ["OTHER"]; }],
    ["added chars", c => { c.context_expansion.added_chars = -1; }],
  ];
  for (const field of ["policy", "policy_sha256", "status", "reason", "parent_evidence_ids", "added_evidence_ids", "added_chars"]) {
    invalid.push([`missing ${field}`, c => { delete c.context_expansion[field]; }]);
  }
  for (const [name, mutate] of invalid) {
    const { candidate, anchors } = fixture("BAD_META", [[2, 5]]);
    mutate(candidate);
    assert.deepEqual(plannedIds(sourceReadPlan(queries(candidate))), [anchors[0]!.evidence_id], name);
  }
});

test("Unicode code point coordinates admit emoji parents and reject UTF-16 offsets", () => {
  const { candidate, anchors, parents } = fixture("EMOJI", [[1, 4]], [[0, 8]]);
  assert.equal(anchors[0]!.quote, "😀乙丙");
  assert.deepEqual(plannedIds(sourceReadPlan(queries(candidate))), [parents[0]!.evidence_id]);
  parents[0]!.end = parents[0]!.quote.length;
  assert.deepEqual(plannedIds(sourceReadPlan(queries(candidate))), [anchors[0]!.evidence_id]);
});

test("read checks inspect original anchors first and skip already covered anchors", () => {
  const { candidate, anchors } = fixture();
  const checks: string[] = [];
  const plan = sourceReadPlan(queries(candidate), 3, e => { checks.push(e.evidence_id); return true; });
  assert.deepEqual(plan, []);
  assert.deepEqual(checks, anchors.map(e => e.evidence_id));
});

test("attempted parent failure falls back to unattempted anchors without repeating the parent", () => {
  const { candidate, anchors, parents } = fixture();
  const attempted = new Set<string>();
  const isRead = (e: JsonObject) => attempted.has(e.evidence_id);
  const first = sourceReadPlan(queries(candidate), 1, isRead);
  assert.deepEqual(plannedIds(first), [parents[0]!.evidence_id]);
  attempted.add(parents[0]!.evidence_id);
  assert.deepEqual(plannedIds(sourceReadPlan(queries(candidate), 3, isRead)), anchors.map(e => e.evidence_id));
  attempted.add(anchors[0]!.evidence_id);
  assert.deepEqual(plannedIds(sourceReadPlan(queries(candidate), 1, isRead)), [anchors[1]!.evidence_id]);
  attempted.add(anchors[1]!.evidence_id);
  assert.deepEqual(sourceReadPlan(queries(candidate), 3, isRead), []);
});

test("distinct parents keep candidate order, query fairness and maxSources", () => {
  const a = fixture("A", [[2, 5], [22, 25]], [[0, 10], [20, 30]]);
  const b = fixture("B"); const c = fixture("C"); const d = fixture("D");
  const grouped = [{ candidates: [a.candidate, b.candidate] }, ...queries(c.candidate), ...queries(d.candidate)];
  assert.deepEqual(plannedIds(sourceReadPlan(grouped, 5)), ["A:P0", "C:P0", "D:P0", "A:P1", "B:P0"]);
  assert.deepEqual(plannedIds(sourceReadPlan(grouped, 2)), ["A:P0", "C:P0"]);
  assert.deepEqual(sourceReadPlan(grouped, 0), []);
});

test("V1 candidates retain their first evidence and source diversity", () => {
  const a = fixture("V1A").candidate; const b = fixture("V1B").candidate;
  delete a.required_evidence_ids; delete b.required_evidence_ids;
  assert.deepEqual(plannedIds(sourceReadPlan(queries(a, b))), ["V1A:A0", "V1B:A0"]);
  const v2 = fixture("V2").candidate;
  assert.deepEqual(plannedIds(sourceReadPlan(queries(a, v2))), ["V1A:A0", "V2:P0"]);
});