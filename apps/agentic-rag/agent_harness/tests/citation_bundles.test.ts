import test from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { canonical, hash, type JsonObject } from "../src/contracts.js";

async function api(): Promise<any> {
  assert.ok(existsSync(new URL("../src/citation_bundles.js", import.meta.url)), "citation bundle implementation must exist");
  return import("../src/" + "citation_bundles.js");
}
export function fixture() {
  const text = "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉";
  const evidence = (id: string, start: number, end: number, extra = {}) => ({ evidence_id: id,
    source_id: "SOURCE", text_version: "1", text_sha256: hash(text), start, end,
    quote: Array.from(text).slice(start, end).join(""), source_verified: true, ...extra });
  const anchors = [[0, 3], [2, 5], [5, 8], [11, 15], [15, 19]].map(([a,b], i) => evidence("A" + i, a!, b!));
  const candidate = { source_id: "SOURCE", unit_id: "UNIT", required_evidence_ids: anchors.map(e => e.evidence_id), evidence: anchors };
  const windows = [evidence("W1", 0, 9, { was_read: true }), evidence("W2", 10, 20, { was_read: true })];
  return { candidate, windows, evidence };
}
const selection = (case_ref: string) => ({ case_ref, same_points: ["相同机制"], differences: ["当前条件仍需核验"], match_status: "reliable_match", supports_tags: ["赠送利益"] });

test("five necessary anchors covered by two real visible windows expand from one case_ref", async () => {
  const { buildCitationOptions, expandCaseSelections } = await api();
  const { candidate, windows } = fixture();
  const options = buildCitationOptions("SNAPSHOT", [candidate], windows);
  assert.equal(options.length, 1);
  assert.match(options[0].case_ref, /^C[0-9a-f]{20}$/);
  assert.deepEqual(options[0].citation_evidence_ids, ["W1", "W2"]);
  assert.deepEqual(expandCaseSelections([selection(options[0].case_ref)], new Map(options.map((o: JsonObject) => [o.case_ref, o]))),
    [{ source_id: "SOURCE", unit_id: "UNIT", evidence_ids: ["W1", "W2"], ...Object.fromEntries(Object.entries(selection("")).filter(([key]) => key !== "case_ref")) }]);
});

test("unread, hidden, forged quote, wrong source/version/hash and gapped windows never yield a reference", async () => {
  const { buildCitationOptions } = await api();
  for (const mutate of [
    (w: JsonObject[]) => { w[0]!.was_read = false; },
    (w: JsonObject[]) => { w.pop(); },
    (w: JsonObject[]) => { w[0]!.quote = "伪".repeat(9); },
    (w: JsonObject[]) => { w[0]!.source_id = "OTHER"; },
    (w: JsonObject[]) => { w[0]!.text_version = "2"; },
    (w: JsonObject[]) => { w[0]!.text_sha256 = "other"; },
    (w: JsonObject[]) => { w[0]!.quote = w[0]!.quote.slice(1); },
    (w: JsonObject[]) => { w[0]!.end = 7; w[0]!.quote = w[0]!.quote.slice(0, 7); },
  ]) {
    const { candidate, windows } = fixture(); mutate(windows);
    assert.deepEqual(buildCitationOptions("SNAPSHOT", [candidate], windows), []);
  }
});

test("binding ignores random window IDs and metadata but binds snapshot/unit/source/identity", async () => {
  const { buildCitationOptions } = await api(); const { candidate, windows } = fixture();
  const ref = buildCitationOptions("SNAPSHOT", [candidate], windows)[0].case_ref;
  windows.forEach((w, i) => Object.assign(w, { evidence_id: "RANDOM" + i, cache_hit: true, task_id: "random", timestamp: 123 }));
  candidate.evidence.reverse(); candidate.required_evidence_ids.reverse();
  assert.equal(buildCitationOptions("SNAPSHOT", [candidate], windows)[0].case_ref, ref);
  assert.notEqual(buildCitationOptions("OTHER", [candidate], windows)[0].case_ref, ref);
  assert.notEqual(buildCitationOptions("SNAPSHOT", [{ ...candidate, unit_id: "OTHER" }], windows)[0].case_ref, ref);
  const duplicate = { ...candidate.evidence[0]!, evidence_id: "ALIAS" };
  candidate.evidence.push(duplicate); candidate.required_evidence_ids.push("ALIAS");
  assert.equal(buildCitationOptions("SNAPSHOT", [candidate], windows)[0].case_ref, ref);
});

test("different binding of one unit fails closed and distinct units may share real text", async () => {
  const { buildCitationOptions } = await api(); const { candidate, windows } = fixture();
  assert.throws(() => buildCitationOptions("SNAPSHOT", [candidate, { ...candidate, required_evidence_ids: candidate.required_evidence_ids.slice(1) }], windows), /CITATION_UNIT_BINDING_CONFLICT/);
  assert.equal(buildCitationOptions("SNAPSHOT", [candidate, { ...candidate, unit_id: "OTHER" }], windows).length, 2);
  assert.throws(() => buildCitationOptions("", [candidate], windows), /CITATION_SNAPSHOT_REQUIRED/);
});

test("fabricated/hidden references, legacy or unknown mixed fields and duplicate refs are rejected", async () => {
  const { buildCitationOptions, expandCaseSelections } = await api(); const { candidate, windows } = fixture();
  const options = buildCitationOptions("SNAPSHOT", [candidate], windows); const ref = options[0].case_ref;
  const map = new Map(options.map((o: JsonObject) => [o.case_ref, o]));
  assert.throws(() => expandCaseSelections([selection("C" + "0".repeat(20))], map), /CITATION_REFERENCE_NOT_VISIBLE/);
  for (const field of ["source_id", "unit_id", "evidence_ids", "unknown"]) assert.throws(() => expandCaseSelections([{ ...selection(ref), [field]: "fabricated" }], map), /INVALID_CASE_SELECTION/);
  assert.throws(() => expandCaseSelections([selection(ref), selection(ref)], map), /DUPLICATE_CASE_REFERENCE/);
  assert.deepEqual(expandCaseSelections([], map), []);
  assert.equal(canonical(candidate), canonical(candidate), "source fixtures remain plain data");
});

test("Unicode code point offsets handle supplementary characters and reject UTF16-shaped windows", async () => {
  const { buildCitationOptions } = await api();
  const quote = "甲😀乙𠮷丙"; const base = { source_id: "UNICODE", text_version: "1", text_sha256: hash(quote), source_verified: true };
  const anchor = { ...base, evidence_id: "A", start: 1, end: 4, quote: "😀乙𠮷" };
  const candidate = { source_id: "UNICODE", unit_id: "U", required_evidence_ids: ["A"], evidence: [anchor] };
  const windows = [{ ...base, evidence_id: "W", start: 0, end: 5, quote, was_read: true }];
  assert.equal(buildCitationOptions("S", [candidate], windows).length,1);
  windows[0]!.end = quote.length;
  assert.deepEqual(buildCitationOptions("S", [candidate], windows),[]);
});

test("a malformed rebinding of a recalled unit cannot leave its first option usable", async () => {
  const { buildCitationOptions } = await api(); const { candidate, windows } = fixture();
  assert.deepEqual(buildCitationOptions("S",[candidate,{ ...candidate, required_evidence_ids: ["MISSING"] }],windows),[]);
});

test("overlapping selected windows must agree even between necessary anchors", async () => {
  const { buildCitationOptions } = await api(); const base = { source_id: "S", text_version: "1", text_sha256: hash("abcdefghij"), source_verified: true };
  const anchors = [{ ...base, evidence_id: "A", start: 0, end: 2, quote: "ab" }, { ...base, evidence_id: "B", start: 8, end: 10, quote: "ij" }];
  const candidate = { source_id: "S", unit_id: "U", required_evidence_ids: ["A","B"], evidence: anchors };
  const windows = [{ ...base, evidence_id: "W1", start: 0, end: 7, quote: "abcdefg", was_read: true }, { ...base, evidence_id: "W2", start: 5, end: 10, quote: "XXhij", was_read: true }];
  assert.deepEqual(buildCitationOptions("SNAPSHOT",[candidate],windows),[]);
});

test("read windows without an authoritative evidence ID cannot produce citations", async () => {
  const { buildCitationOptions } = await api(); const {candidate,windows}=fixture();
  windows[0]!.evidence_id = "";
  assert.deepEqual(buildCitationOptions("S",[candidate],windows),[]);
});
