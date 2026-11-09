import assert from "node:assert/strict";
import { createRequire } from "node:module";
import test from "node:test";

const { buildView } = createRequire(import.meta.url)("../../src/web/static/source-map.js");

function fixture(lines, levels = []) {
  let cursor = 2;
  const source_mappings = [];
  const risk_assessments = [];
  lines.forEach((text, index) => {
    const sentence_id = `query_s${index + 1}`;
    const end = cursor + Array.from(text).length;
    source_mappings.push({ sentence_id, document_start_offset: cursor, document_end_offset: end });
    risk_assessments.push({ sentence_id, text, risk_level: levels[index] || "none",
      risk_level_name: levels[index] || "无明显风险", risk_score: index / 10, review_status: "local" });
    cursor = end + 2;
  });
  return { source_offset_unit: "unicode_code_point", source_text: `  ${lines.join("\r\n")}  `,
    source_mappings, risk_assessments };
}

test("emoji, rare CJK, ZWJ sequences and CRLF preserve all source characters", () => {
  const result = fixture(["📢 客户𠀀：保证8%吗？", "顾问👩‍💻：不能保证。", "🇨🇳 实际分红可能为零。"]);
  const view = buildView(result);
  assert.equal(view.parts.map(part => part.text).join(""), result.source_text);
  assert.deepEqual(view.entries.map(entry => entry.text), result.risk_assessments.map(item => item.text));
  assert.equal(view.entries[1].text[0], "顾");
});

test("each repeated sentence keeps its own ID, index and risk level", () => {
  const result = fixture(["保证收益。", "保证收益。"], ["none", "high"]);
  result.risk_assessments.reverse();
  const view = buildView(result, { sentenceId: "query_s2", sourceSentenceId: "query_s2" });
  assert.deepEqual(view.entries.map(entry => [entry.id, entry.label, entry.level]),
    [["query_s1", "S01", "none"], ["query_s2", "S02", "high"]]);
  assert.deepEqual(view.entries.map(entry => entry.selected), [false, true]);
});

test("a risky answer can cite a safe question without transferring risk to the question", () => {
  const result = fixture(["客户：📢保证8%吗？", "顾问：是的。"], ["none", "medium"]);
  const question = result.source_mappings[0];
  const start = question.document_start_offset + Array.from("客户：📢").length;
  const view = buildView(result, { sentenceId: "query_s2", sourceSentenceId: "query_s1", start, end: start + 4 });
  assert.equal(view.selected.level, "medium");
  assert.equal(view.cited.level, "none");
  assert.equal(view.cited.cited, true);
  assert.equal(view.cited.selected, false);
  assert.deepEqual(view.cited.pieces.filter(piece => piece.evidence).map(piece => piece.text), ["保证8%"]);
  assert.equal(view.parts.flatMap(part => part.pieces || [{ text: part.text }]).map(part => part.text).join(""), result.source_text);
});

test("selecting a safe sentence keeps its original severity", () => {
  const result = fixture(["不能保证收益。"]);
  const view = buildView(result, { sentenceId: "query_s1", sourceSentenceId: "query_s1" });
  assert.equal(view.selected.level, "none");
  assert.equal(view.selected.selected, true);
  assert.equal(view.selected.pieces.some(piece => piece.evidence), false);
});

test("unknown business facts use pending styling even with a high attention score", () => {
  const result = fixture(["产品停售公告待补。"], ["high"]);
  result.risk_assessments[0].decision_state = "insufficient_information";
  result.risk_assessments[0].risk_level_name = "信息不足";
  result.risk_assessments[0].risk_score = .8;
  const view = buildView(result);
  assert.equal(view.entries[0].level, "unknown");
  assert.equal(view.entries[0].levelName, "信息不足");
  assert.equal(view.entries[0].score, .8);
});

for (const fault of ["utf16", "overlap", "bounds", "changed_text", "duplicate", "missing_text", "missing_assessment"]) {
  test(`rejects ${fault} instead of painting another sentence`, () => {
    const result = fixture(["📢 内容。", "后一句。"]);
    if (fault === "utf16") result.source_offset_unit = "utf16";
    if (fault === "overlap") result.source_mappings[1].document_start_offset = 3;
    if (fault === "bounds") result.source_mappings[1].document_end_offset = 999;
    if (fault === "changed_text") result.risk_assessments[0].text = "另一句。";
    if (fault === "duplicate") result.source_mappings[1].sentence_id = "query_s1";
    if (fault === "missing_text") result.source_text += "遗漏。";
    if (fault === "missing_assessment") result.risk_assessments.pop();
    assert.throws(() => buildView(result), /原文/);
  });
}

test("rejects evidence offsets outside the cited sentence", () => {
  const result = fixture(["问题？", "回答。"]);
  assert.throws(() => buildView(result, { sentenceId: "query_s2", sourceSentenceId: "query_s1", start: 1, end: 7 }), /证据/);
});
