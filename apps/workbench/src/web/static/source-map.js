/* All API offsets count Unicode code points, with an exclusive end offset. */
const FinRiskSource = (() => {
  const levels = new Set(["none", "low", "review", "medium", "high"]);
  const label = index => `S${String(index + 1).padStart(2, "0")}`;

  function buildView(result, selection = null) {
    if (result.source_offset_unit && result.source_offset_unit !== "unicode_code_point") {
      throw new Error("原文位置的字符单位不受支持");
    }
    const characters = Array.from(result.source_text);
    const slice = (start, end) => characters.slice(start, end).join("");
    const assessments = new Map(result.risk_assessments.map(item => [item.sentence_id, item]));
    const mappings = result.source_mappings;
    if (assessments.size !== mappings.length || assessments.size !== result.risk_assessments.length) {
      throw new Error("原文位置与逐句判断未能一一对应");
    }
    const ids = new Set();
    let cursor = 0;
    const parts = [];
    const entries = mappings.map((mapping, index) => {
      const id = mapping.sentence_id;
      const start = mapping.document_start_offset;
      const end = mapping.document_end_offset;
      const assessment = assessments.get(id);
      if (!Number.isInteger(start) || !Number.isInteger(end) || start < cursor || end <= start
          || end > characters.length || ids.has(id) || !assessment || !levels.has(assessment.risk_level)
          || slice(start, end) !== assessment.text || slice(cursor, start).trim()) {
        throw new Error("原文位置校验未通过，已停止着色以避免标错文字");
      }
      ids.add(id);
      if (start > cursor) parts.push({ text: slice(cursor, start) });
      const entry = {
        id, start, end, index, label: label(index), text: assessment.text,
        level: assessment.decision_state === "insufficient_information" ? "unknown" : assessment.risk_level,
        levelName: assessment.risk_level_name,
        score: assessment.risk_score, needsReview: assessment.review_status === "needs_review",
        selected: selection?.sentenceId === id,
        cited: selection?.sourceSentenceId === id && selection.sentenceId !== id,
        pieces: [{ text: assessment.text, evidence: false }],
      };
      parts.push(entry);
      cursor = end;
      return entry;
    });
    if (slice(cursor).trim()) throw new Error("原文位置遗漏了尾部文字");
    if (cursor < characters.length) parts.push({ text: slice(cursor) });
    const selected = entries.find(entry => entry.id === selection?.sentenceId);
    const cited = entries.find(entry => entry.id === selection?.sourceSentenceId);
    if (selection && (!selected || !cited)) throw new Error("所选语句不属于本次审查");
    if (selection?.start != null || selection?.end != null) {
      const { start, end } = selection;
      if (!cited || !Number.isInteger(start) || !Number.isInteger(end)
          || start < cited.start || end > cited.end || end <= start) {
        throw new Error("证据位置未落在引用原句内");
      }
      cited.pieces = [
        { text: slice(cited.start, start), evidence: false },
        { text: slice(start, end), evidence: true },
        { text: slice(end, cited.end), evidence: false },
      ].filter(piece => piece.text);
    }
    return { parts, entries, selected, cited };
  }

  return { buildView, label };
})();

if (typeof module !== "undefined" && module.exports) module.exports = FinRiskSource;
