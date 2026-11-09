import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { classifyReviewReply } from "../src/review_reply.js";
import { root } from "../src/contracts.js";

test("native call classification only recognizes an explicit decision object", () => {
  const note = { kind: "decision", goal: "finalize", accepted_evidence_ids: ["forged"] };
  assert.deepEqual(classifyReviewReply(JSON.stringify(note), true), { type: "native_tools", decision: note, ignored_note: false });
  for (const text of ["", "正在检索。", "I'll search for cases.", "{bad json", '{"kind":"finalize"}', '{"kind":"final"}', '{"kind":"search_cases"}']) {
    const reply = classifyReviewReply(text, true);
    assert.equal(reply.type, "native_tools"); assert.ok(!("decision" in reply));
  }
});

test("body tool intent classification returns only a tool name and never native arguments", () => {
  for (const name of ["search_cases", "read_source", "get_risk_definition"]) {
    assert.deepEqual(classifyReviewReply(JSON.stringify({ kind: name, arguments: { forged: true } }), false), { type: "tool_intent", name });
  }
  assert.equal(classifyReviewReply('{"kind":"invented_tool"}', false).type, "json");
  assert.deepEqual(classifyReviewReply('{"kind":"finalize"}', false), { type: "json", value: { kind: "finalize" } });
  assert.equal(classifyReviewReply("null", false).type, "invalid");
  assert.equal(classifyReviewReply("[]", false).type, "invalid");
  assert.equal(classifyReviewReply("ordinary text", false).type, "invalid");
});

test("review classifier is deterministic and has no runtime imports or execution/state access", () => {
  const source = readFileSync(`${root}/agent_harness/src/review_reply.ts`, "utf8");
  assert.ok(!/^import (?!type\b)/m.test(source));
  assert.ok(!/\b(?:fetch|TaskState|Budget|ToolBridge|process|readFileSync|execute)\b/.test(source));
  const text = '{"kind":"search_cases","queries":[{"text":"instruction"}]}';
  assert.deepEqual(classifyReviewReply(text, false), classifyReviewReply(text, false));
});
