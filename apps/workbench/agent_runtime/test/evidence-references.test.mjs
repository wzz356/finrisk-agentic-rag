import test from "node:test";
import assert from "node:assert/strict";
import { resolveEvidenceReferences } from "../runtime.mjs";

const request = {review_context: {evidence_mode: "source_ids", compact_payload: {
  source_sentences: [{id: "s1", text: "客户本人如实回答回访问题。"}, {id: "s2", text: "另一产品保证收益。"}],
  targets: [{sentence_id: "target1", source_ids: ["s1"]}],
}}};

test("source references rehydrate every nested evidence array from the original text", () => {
  const value = {items: [{sentence_id: "target1", evidence: ["s1"],
    mechanism_assessments: [{evidence: ["s1"], condition_assessments: [{evidence: ["s1"]}]}]}]};
  const result = resolveEvidenceReferences(value, request);
  assert.equal(result.items[0].evidence[0], "客户本人如实回答回访问题。");
  assert.equal(result.items[0].mechanism_assessments[0].condition_assessments[0].evidence[0], result.items[0].evidence[0]);
});

test("references cannot borrow another product's source", () => {
  assert.throws(() => resolveEvidenceReferences({items: [{sentence_id: "target1", evidence: ["s2"]}]}, request), /超出/);
});

test("already literal evidence remains compatible", () => {
  assert.equal(resolveEvidenceReferences({items: [{sentence_id: "target1", evidence: ["如实回答"]}]}, request).items[0].evidence[0], "如实回答");
});
