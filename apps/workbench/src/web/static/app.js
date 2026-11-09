const state = {
  activeView: "risk",
  queryMode: "text",
  queryFile: null,
  caseFile: null,
  riskResult: null,
  riskFilter: "all",
  sourceSelection: null,
  sourceView: null,
  casePage: 1,
  casePageSize: 12,
  caseResult: null,
  analysisTimer: null,
};

const sceneLabels = {
  ST1_poster_sms_social: "海报 / 短信 / 社交平台文案",
  ST2_live_short_video: "直播 / 短视频 / 口播",
  ST3_product_intro_detail: "产品介绍 / 详情页",
  ST4_meeting_material: "产品说明会 / 客户活动材料",
  ST5_training_telemarketing_callback: "培训 / 电销 / 回访话术",
  ST6_penalty_backend_fact: "监管处罚 / 后台业务事实",
};

const classificationMethodLabels = {
  user: "用户指定场景",
  rule: "本地规则分类",
  rule_fallback: "本地规则回退",
  llm: "大模型复核分类",
};

const el = (selector) => document.querySelector(selector);
const roleLabels = {
  customer: "客户", sales: "销售 / 顾问", training: "培训发言",
  question: "提问", answer: "回答", host: "主持人", unknown: "未标记角色",
};
const stanceLabels = {
  statement: "陈述", question: "提问", denial: "否定 / 纠正",
  prohibition: "禁止性要求", negative_example: "反面引用", unknown: "待核对",
};
const samples = {
  dialogue: {
    scene: "ST5_training_telemarketing_callback",
    text: "客户：这款保险保证每年 8% 的收益吗？\n顾问：是的，可以这样理解。\n顾问：实际分红以公司经营情况为准，可能为零。",
  },
  quotation: {
    scene: "ST5_training_telemarketing_callback",
    text: "培训材料：以下是反面话术，严禁使用。\n“本金绝对安全，保证年化收益8%。”\n正确话术：非保证利益存在不确定性，实际权益以合同为准。",
  },
  products: {
    scene: "ST3_product_intro_detail",
    text: "产品A：分红是不保证的，可能为零。\n\n产品B：保证年化收益8%，本金绝对安全。",
  },
};
document.querySelectorAll("[data-sample]").forEach(button => {
  button.addEventListener("click", () => {
    const sample = samples[button.dataset.sample];
    switchQueryMode("text");
    el("#query-text").value = sample.text;
    el("#query-count").textContent = Array.from(sample.text).length.toLocaleString("zh-CN");
    el("#query-scene").value = sample.scene;
    el("#query-text").focus();
    toast("示例已载入，可编辑后审查");
  });
});

document.querySelectorAll("[data-view]").forEach((button) => {
  button.addEventListener("click", () => switchView(button.dataset.view));
});
document.querySelectorAll("[data-route]").forEach((link) => {
  link.addEventListener("click", (event) => {
    event.preventDefault();
    switchView(link.dataset.route);
  });
});
document.querySelectorAll("[data-query-mode]").forEach((button) => {
  button.addEventListener("click", () => switchQueryMode(button.dataset.queryMode));
});

setupDropZone("#query-drop-zone", "#query-file", (file) => {
  state.queryFile = file;
  el("#query-file-name").textContent = fileLabel(file);
});
setupDropZone("#case-drop-zone", "#case-file", (file) => {
  state.caseFile = file;
  el("#case-file-name").textContent = fileLabel(file);
});

el("#query-text").addEventListener("input", (event) => {
  el("#query-count").textContent = Array.from(event.target.value).length.toLocaleString("zh-CN");
});
el("#analyze-button").addEventListener("click", analyzeQuery);
el("#copy-risk-result").addEventListener("click", copyRiskResult);
el("#export-risk-report").addEventListener("click", exportRiskReport);
el("#export-submission").addEventListener("click", exportSubmission);
el("#save-review").addEventListener("click", saveQueryReview);
el("#risk-filter").addEventListener("change", (event) => {
  state.riskFilter = event.target.value;
  renderRiskSentences();
});
el("#risk-list").addEventListener("click", handleCaseLink);
el("#top-cases").addEventListener("click", handleCaseLink);
el("#source-highlight").addEventListener("click", handleSourceHighlightClick);
el("#source-highlight").addEventListener("keydown", event => {
  if ((event.key === "Enter" || event.key === " ") && event.target.closest("[data-sentence-target]")) {
    event.preventDefault();
    handleSourceHighlightClick(event);
  }
});
el("#ingest-button").addEventListener("click", ingestCase);
el("#case-filters").addEventListener("submit", (event) => {
  event.preventDefault();
  state.casePage = 1;
  loadCases();
});
el("#refresh-cases").addEventListener("click", () => loadCases());
el("#case-list").addEventListener("click", handleCaseListClick);
el("#case-pagination").addEventListener("click", handlePagination);
el("#drawer-close").addEventListener("click", closeDrawer);
el("#drawer-backdrop").addEventListener("click", closeDrawer);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeDrawer();
});

const routeViews = new Set(["risk", "case-entry", "cases", "evaluation"]);
function syncHashView() {
  const view = window.location.hash.slice(1);
  if (routeViews.has(view) && view !== state.activeView) switchView(view);
}
window.addEventListener("hashchange", syncHashView);
syncHashView();
loadHealth();
renderWorkspaceSummary(null);
loadCollectionStatus();
setInterval(loadCollectionStatus, 60_000);
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) loadCollectionStatus();
});

function switchView(view) {
  state.activeView = view;
  document.querySelectorAll("[data-view]").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.view === view);
  });
  document.querySelectorAll(".view").forEach((panel) => panel.classList.remove("is-visible"));
  el(`#${view}-view`)?.classList.add("is-visible");
  el("#current-view-label").textContent = {
    risk: "材料审查", "case-entry": "案例录入", cases: "案例管理", evaluation: "效果评测",
  }[view] || "工作台";
  if (view === "cases") {
    loadStatistics();
    loadCases();
  }
  if (view === "evaluation") loadEvaluation();
  window.location.hash = view;
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function switchQueryMode(mode) {
  state.queryMode = mode;
  document.querySelectorAll("[data-query-mode]").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.queryMode === mode);
    button.setAttribute("aria-selected", String(button.dataset.queryMode === mode));
  });
  el("#query-text-panel").classList.toggle("is-visible", mode === "text");
  el("#query-file-panel").classList.toggle("is-visible", mode === "file");
  showFormError("#query-error", "");
}

function setupDropZone(zoneSelector, inputSelector, onSelect) {
  const zone = el(zoneSelector);
  const input = el(inputSelector);
  input.addEventListener("change", () => onSelect(input.files?.[0] ?? null));
  ["dragenter", "dragover"].forEach((name) => {
    zone.addEventListener(name, (event) => {
      event.preventDefault();
      zone.classList.add("is-dragging");
    });
  });
  ["dragleave", "drop"].forEach((name) => {
    zone.addEventListener(name, (event) => {
      event.preventDefault();
      zone.classList.remove("is-dragging");
    });
  });
  zone.addEventListener("drop", (event) => {
    const file = event.dataTransfer?.files?.[0] ?? null;
    if (file) onSelect(file);
  });
}

async function loadHealth() {
  try {
    const data = await api("/api/health");
    el("#health-state").classList.add("is-ready");
    el("#health-text").textContent = `${data.active_case_count.toLocaleString("zh-CN")} ${data.case_listing_unit === "document" ? "份文书" : "条案例"}可用`;
    const runtime = data.agent_runtime;
    const rerankNames = {none:"多路融合",cloud:"Qwen 重排",deepseek:"DeepSeek 重排"};
    const timingNames = {post:"Post",pre:"Pre",adaptive:"自适应融合"};
    el("#query-rerank").options[0].textContent = `推荐 · ${rerankNames[data.retrieval?.rerank_provider] || "当前配置"}`;
    el("#query-timing").options[0].textContent = `推荐 · ${timingNames[data.retrieval?.timing] || "当前配置"}`;
    const mode = !data.llm_model ? "本地规则模式"
      : runtime?.available ? `Pi Agent · ${data.llm_model}` : "Agent 未就绪 · 规则模式";
    el("#health-state").title = `${mode} · 数据版本 ${data.data_version || "—"}`;
  } catch (error) {
    el("#health-state").classList.add("is-error");
    el("#health-text").textContent = "服务连接失败";
  }
}

async function analyzeQuery() {
  showFormError("#query-error", "");
  let path;
  let options;
  if (state.queryMode === "text") {
    const text = el("#query-text").value;
    const scene = el("#query-scene").value || null;
    if (!text.trim()) return showFormError("#query-error", "请输入需要审查的文字。");
    path = "/api/query/analyze/text";
    options = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, scene, rerank_provider: el("#query-rerank").value || null,
        retrieval_timing: el("#query-timing").value || null }),
    };
  } else {
    if (!state.queryFile) return showFormError("#query-error", "请先选择 Query 文件。");
    const form = new FormData();
    form.append("file", state.queryFile);
    const scene = el("#query-scene").value;
    if (scene) form.append("scene", scene);
    if (el("#query-rerank").value) form.append("rerank_provider", el("#query-rerank").value);
    if (el("#query-timing").value) form.append("retrieval_timing", el("#query-timing").value);
    path = "/api/query/analyze/file";
    options = { method: "POST", body: form };
  }
  setBusy("#analyze-button", true);
  state.riskResult = null;
  state.sourceSelection = null;
  state.sourceView = null;
  state.riskFilter = "all";
  el("#risk-filter").value = "all";
  el("#risk-content").hidden = true;
  el("#risk-empty").hidden = false;
  el("#risk-results").setAttribute("aria-busy", "true");
  const started = Date.now();
  const status = el("#analysis-state");
  status.hidden = false;
  const updateStatus = () => {
    status.textContent = `正在结合上下文审查材料，已用时 ${Math.floor((Date.now() - started) / 1000)} 秒。`;
  };
  updateStatus();
  state.analysisTimer = window.setInterval(updateStatus, 1000);
  try {
    state.riskResult = await api(path, options);
    renderRiskResult();
  } catch (error) {
    showFormError("#query-error", error.message);
  } finally {
    setBusy("#analyze-button", false);
    window.clearInterval(state.analysisTimer);
    state.analysisTimer = null;
    status.hidden = true;
    el("#risk-results").setAttribute("aria-busy", "false");
  }
}

function renderRiskResult() {
  const result = state.riskResult;
  const summary = result.risk_summary;
  el("#risk-empty").hidden = true;
  el("#risk-content").hidden = false;
  el("#risk-summary").innerHTML = `
    <article class="summary-card level-${summary.decision_state === "insufficient_information" ? "unknown" : escapeHtml(summary.overall_level)}">
      <span>整体风险</span><strong>${escapeHtml(summary.overall_level_name)}</strong>
      <small>${escapeHtml(summary.action)}</small>
    </article>
    <article class="summary-card"><span>风险关注分</span><strong>${formatScore(summary.overall_score)}</strong><small>评分范围 0–1</small></article>
    <article class="summary-card"><span>需关注语句</span><strong>${summary.risky_sentence_count} / ${summary.sentence_count}</strong><small>共 ${summary.retrieval_sentence_count} 句检索类案</small></article>
  `;
  el("#review-status").value = "confirmed";
  el("#review-scene").value = "";
  el("#review-labels").value = summary.risk_labels.join("，");
  el("#review-comment").value = "";
  el("#review-state").textContent = "尚未保存人工复核";
  el("#risk-warnings").innerHTML = summary.warnings
    .map((warning) => `<p>提示：${escapeHtml(warning)}</p>`)
    .join("") + (summary.missing_facts || [])
    .map((fact) => `<p>待补事实：${escapeHtml(fact)}</p>`).join("");
  const classification = result.classification;
  const methodLabel = classificationMethodLabels[classification.method] || classification.method;
  el("#classification-note").textContent = `${sceneLabels[classification.scene] || classification.scene} · ${formatPercent(classification.confidence)} · ${methodLabel}`;
  const extraction = result.extraction;
  const quality = extraction.quality_score == null ? "—" : formatPercent(extraction.quality_score);
  el("#extraction-note").textContent = `${extraction.extraction_method} · ${extraction.page_count} 页 · OCR ${extraction.ocr_page_count} 页 · 质量 ${quality} · SHA-256 ${extraction.checksum_sha256.slice(0, 10)}…`;
const timing = result.stage_timings_ms || {};
  const riskTiming = summary.stage_timings_ms || {};
  el("#timing-note").textContent = `总耗时 ${formatDuration(timing.total)} · LLM ${formatDuration(riskTiming.llm_review)} · 前检索 ${formatDuration(riskTiming.pre_retrieval)} · 后检索 ${formatDuration(riskTiming.case_retrieval)}`;
  const runs = summary.agent_runs || [];
  el("#agent-trace-note").textContent = runs.length
    ? `Pi Agent ${runs[0].version} · ${runs.length} 次语义审查 · ${runs.reduce((n, run) => n + run.turns, 0)} 轮 · ${runs.reduce((n, run) => n + run.tool_calls, 0)} 次工具调用 · ${runs.reduce((n, run) => n + run.tool_failures, 0)} 次提交校验失败`
    : "本次无语义复核结果";
  const candidates = classification.rule_candidates.filter((item) => item.score > 0).slice(0, 4);
  el("#classification-candidates").innerHTML = candidates.length
    ? candidates.map((item, index) => `
      <span class="candidate-chip ${index === 0 ? "is-top" : ""}">
        ${escapeHtml(sceneLabels[item.scene] || item.scene)} · ${formatPercent(item.confidence)}
      </span>
    `).join("")
    : '<span class="candidate-chip">本地规则无明确候选，使用模型判定</span>';
  renderSourceHighlight();
  renderTopCases();
  renderRiskSentences();
  el("#risk-results").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderRiskSentences() {
  if (!state.riskResult) return;
  let assessments = state.riskResult.risk_assessments.map((item, index) => ({ item, index }));
  if (state.riskFilter === "risky") {
    assessments = assessments.filter(({ item }) => item.risk_score >= state.riskResult.risk_summary.risk_display_threshold);
  }
  if (state.riskFilter === "retrieved") {
    assessments = assessments.filter(({ item }) => item.matched_cases.length > 0);
  }
  if (!assessments.length) {
    el("#risk-list").innerHTML = '<div class="inline-empty">当前筛选下没有风险语句。</div>';
    return;
  }
  const mappings = new Map(state.riskResult.source_mappings.map((item) => [item.sentence_id, item]));
  const contexts = new Map((state.riskResult.sentence_contexts || []).map(item => [item.sentence_id, item]));
  el("#risk-list").innerHTML = assessments.map(({ item, index }) => {
    const mapping = mappings.get(item.sentence_id);
    const context = contexts.get(item.sentence_id);
    const pageNumbers = [...new Set((mapping?.regions || []).map((region) => region.page_number))];
    const location = pageNumbers.length ? `第 ${pageNumbers.join("、")} 页` : "文本输入";
    const unknown = item.decision_state === "insufficient_information";
    const tags = (unknown ? (item.candidate_labels || []) : item.risk_labels)
      .map((tag) => `<span class="tag">${unknown ? "候选：" : ""}${escapeHtml(tag)}</span>`).join("");
    const missingFacts = (item.missing_facts || [])
      .map((fact) => `<p class="reasoning">待补事实：${escapeHtml(fact)}</p>`).join("");
    const evidence = item.evidence.map(value => `<li>${escapeHtml(value.description)} ${value.sentence_id
      ? `<button class="source-link" type="button" data-source-sentence="${escapeHtml(value.sentence_id)}" ${sourceRangeAttributes(value)}>定位证据 · ${sourceLabel(value.sentence_id)}</button>` : ""}</li>`).join("");
    const matches = item.matched_cases.map((matched) => `
      <button class="matched-case" type="button" data-case-id="${escapeHtml(matched.case_id)}">
        <span><b>${escapeHtml(matched.case_id)}</b><em>排序分 ${formatScore(matched.score)}</em></span>
        <strong>${escapeHtml(matched.case_title || matched.violation_behavior.slice(0, 72))}</strong>
        <small>${escapeHtml(matched.matched_risk_tags.join(" · ") || matched.risk_tags.join(" · ") || "未标注")}</small>
        <small class="score-breakdown">BM25 ${formatScore(matched.bm25_score)} · 向量 ${formatScore(matched.vector_score)} · RRF ${formatScore(matched.rrf_score)}</small>
        <small>${matched.literal_verified ? "原文已核对" : "类案匹配"} · ${escapeHtml(matched.file_id || "")}</small>
        <small>${escapeHtml(matched.match_reason || ("命中证据切片 " + matched.matched_unit_id))}</small>
      </button>
    `).join("");
    return `
      <article class="risk-card level-${unknown ? "unknown" : escapeHtml(item.risk_level)} ${state.sourceSelection?.sentenceId === item.sentence_id ? "is-selected" : ""}" id="sentence-${escapeHtml(item.sentence_id)}" data-sentence-id="${escapeHtml(item.sentence_id)}" tabindex="-1">
        <div class="risk-body">
          <div class="risk-card-head">
            <div><button class="sentence-number source-link" type="button" data-source-sentence="${escapeHtml(item.sentence_id)}" aria-label="${FinRiskSource.label(index)} 定位当前句原文">${FinRiskSource.label(index)} ↗</button><span class="level-badge">${escapeHtml(item.risk_level_name)}</span>${tags}</div>
            <strong>风险分 ${formatScore(item.risk_score)}</strong>
          </div>
          <p class="risk-text">${escapeHtml(item.text)}</p>
          <p class="reasoning">${escapeHtml(item.reasoning)}</p>
          ${(item.remediation_suggestions || []).length ? `<div class="context-details"><strong>修改建议</strong><ul>${item.remediation_suggestions.map(s => `<li>${escapeHtml(s)}</li>`).join("")}</ul></div>` : ""}
          ${missingFacts}
          <div class="sentence-meta">
            <span>${escapeHtml(location)}</span>
            <span>偏移 ${mapping?.document_start_offset ?? "—"}–${mapping?.document_end_offset ?? "—"}</span>
            <span>${escapeHtml(roleLabels[context?.speaker_role] || "未标记角色")}</span>
            <span>${item.llm_reviewed ? (item.review_status === "needs_review" ? "语义已复核 · 上下文待补" : "语义已复核") : item.review_status === "needs_review" ? "语义仍待复核" : "本地规则判断"}</span>
          </div>
          ${renderSentenceContext(context)}
          <details>
            <summary>查看 ${item.evidence.length} 条判断依据与 ${item.matched_cases.length} 条相似案例</summary>
            <div class="evidence-grid">
              <div><h4>判断依据</h4><ul>${evidence || "<li>无额外证据</li>"}</ul></div>
              <div><h4>相关案例</h4><p class="muted">${escapeHtml(retrievalDescription(item))}</p><div class="matched-list">${matches}</div></div>
            </div>
          </details>
        </div>
      </article>
    `;
  }).join("");
}

function renderSourceHighlight() {
  const result = state.riskResult;
  const note = el("#source-selection");
  note.classList.remove("is-error");
  try {
    const view = FinRiskSource.buildView(result, state.sourceSelection);
    state.sourceView = view;
    el("#source-highlight").innerHTML = view.parts.map(part => {
      if (!part.id) return escapeHtml(part.text);
      const description = `${part.label} · ${part.levelName} · 风险分 ${formatScore(part.score)}${part.needsReview ? " · 语义仍待复核" : ""}`;
      const content = part.pieces.map(piece => piece.evidence
        ? `<span class="source-evidence">${escapeHtml(piece.text)}</span>` : escapeHtml(piece.text)).join("");
      return `<span class="source-sentence level-${escapeHtml(part.level)} ${part.selected ? "is-selected" : ""} ${part.cited ? "is-cited" : ""}" data-sentence-target="${escapeHtml(part.id)}" data-label="${part.label}" role="button" tabindex="0" aria-label="${escapeHtml(description)}，查看对应判断" aria-pressed="${part.selected}" title="${escapeHtml(description)}">${content}</span>`;
    }).join("");
    if (!view.selected) {
      note.textContent = "底色：风险等级 · 蓝框：当前句 · 下划线：引用证据";
    } else {
      const selected = view.selected;
      const citation = view.cited.id !== selected.id
        ? `；引用 ${view.cited.label}（${view.cited.levelName}）`
        : "";
      note.innerHTML = `<strong>当前判断 ${selected.label} · ${escapeHtml(selected.levelName)} · ${formatScore(selected.score)}</strong>${escapeHtml(citation)}${selected.needsReview ? " 语义仍待复核。" : ""}`;
    }
  } catch (error) {
    state.sourceView = null;
    el("#source-highlight").textContent = result.source_text;
    note.classList.add("is-error");
    note.textContent = `${error.message}。请对照下方逐句结果重新审查材料。`;
  }
}

function renderTopCases() {
  const cases = state.riskResult.risk_summary.top_matched_cases;
  el("#top-cases").innerHTML = cases.length ? cases.map((item, index) => `
    <button class="top-case-item" type="button" data-case-id="${escapeHtml(item.case_id)}">
      <span><b>${String(index + 1).padStart(2, "0")} · ${escapeHtml(item.case_id)}</b><em>排序分 ${formatScore(item.score)}</em></span>
      <strong>${escapeHtml(item.case_title || item.violation_behavior.slice(0, 78))}</strong>
      <small>${escapeHtml(item.matched_risk_tags.join(" · ") || item.risk_tags.join(" · "))}</small>
      <small>${escapeHtml(item.match_reason || ("命中证据切片 " + item.matched_unit_id))}</small>
    </button>
  `).join("") : `<div class="inline-empty compact">${escapeHtml(
    state.riskResult.risk_assessments.some(item => item.retrieval_status === "unavailable")
      ? "类案检索暂不可用"
      : state.riskResult.risk_summary.retrieval_sentence_count
        ? "暂无相似案例。"
        : "本次无需检索类案"
  )}</div>`;
}

function handleSourceHighlightClick(event) {
  const target = event.target.closest("[data-sentence-target]");
  if (!target) return;
  const id = target.dataset.sentenceTarget;
  if (!document.getElementById(`sentence-${id}`)) {
    state.riskFilter = "all";
    el("#risk-filter").value = "all";
    renderRiskSentences();
    toast("已切换为全部语句，显示所选原文的判断");
  }
  selectSourceSentence({ sentenceId: id, sourceSentenceId: id });
  const sentence = document.getElementById(`sentence-${id}`);
  sentence?.scrollIntoView({ behavior: "smooth", block: "center" });
  sentence?.focus({ preventScroll: true });
}

function selectSourceSentence(selection) {
  state.sourceSelection = selection;
  renderSourceHighlight();
  el("#risk-list").querySelectorAll("[data-sentence-id]").forEach(card => {
    card.classList.toggle("is-selected", card.dataset.sentenceId === selection.sentenceId);
  });
}

function sourceRangeAttributes(source) {
  return Number.isInteger(source.document_start_offset) && Number.isInteger(source.document_end_offset)
    ? `data-source-start="${source.document_start_offset}" data-source-end="${source.document_end_offset}"` : "";
}

function sourceLabel(id) {
  const index = state.riskResult.source_mappings.findIndex(item => item.sentence_id === id);
  return index >= 0 ? FinRiskSource.label(index) : "原文";
}

async function copyRiskResult() {
  if (!state.riskResult) return;
  const summary = state.riskResult.risk_summary;
  const lines = [
    `整体风险：${summary.overall_level_name}（风险程度分 ${formatScore(summary.overall_score)} / 1）`,
    `风险标签：${summary.risk_labels.join("、") || "无"}`,
    `建议动作：${summary.action}`,
    "",
    ...state.riskResult.risk_assessments.map((item, index) =>
      `${index + 1}. [${item.risk_level_name} ${formatScore(item.risk_score)}] ${item.text}\n   ${item.reasoning}\n   ${retrievalDescription(item)}`
    ),
    "",
    ...summary.warnings.map(warning => `复核提示：${warning}`),
  ];
  await navigator.clipboard.writeText(lines.join("\n"));
  toast("结果摘要已复制");
}

async function saveQueryReview() {
  if (!state.riskResult) return;
  const result = state.riskResult;
  const rawLabels = el("#review-labels").value.trim();
  const correctedLabels = rawLabels
    ? [...new Set(rawLabels.split(/[,，、/]/).map((value) => value.trim()).filter(Boolean))]
    : [];
  const payload = {
    query_id: result.query_id,
    query_checksum: result.extraction.checksum_sha256,
    status: el("#review-status").value,
    reviewer: el("#reviewer-name").value.trim() || "现场审核员",
    comment: el("#review-comment").value.trim(),
    original_risk_labels: result.risk_summary.risk_labels,
    corrected_risk_labels: correctedLabels,
    corrected_scene: el("#review-scene").value || null,
    analysis_schema_version: result.schema_version,
  };
  setBusy("#save-review", true);
  try {
    const saved = await api("/api/query/reviews", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    el("#review-state").textContent = `已保存 ${saved.review_id} · ${new Date(saved.reviewed_at).toLocaleString("zh-CN")}`;
    toast("人工复核记录已保存");
  } catch (error) {
    toast(`复核保存失败：${error.message}`);
  } finally {
    setBusy("#save-review", false);
  }
}

function exportRiskReport() {
  if (!state.riskResult) return;
  const result = state.riskResult;
  const summary = result.risk_summary;
  const contexts = new Map((result.sentence_contexts || []).map(item => [item.sentence_id, item]));
  const findings = result.risk_assessments.map((item, index) => {
    const context = contexts.get(item.sentence_id);
    const sources = context?.source_sentences.map(source =>
      `<p><b>${source.is_target ? "当前句" : "上下文"}</b> [${source.document_start_offset}–${source.document_end_offset}] ${escapeHtml(source.text)}</p>`
    ).join("") || "";
    const evidence = item.evidence.map(value => `<li>${escapeHtml(value.description)}${value.document_start_offset != null
      ? ` <span class="muted">[原文偏移 ${value.document_start_offset}–${value.document_end_offset}]</span>` : ""}</li>`).join("");
    return `<section class="finding"><h3>${String(index + 1).padStart(2, "0")} · ${escapeHtml(item.risk_level_name)} <small>风险程度分 ${formatScore(item.risk_score)} / 1</small></h3>
      <p class="original">${escapeHtml(item.text)}</p>
      <p>判断理由：${escapeHtml(item.reasoning)}</p>
      ${(item.remediation_suggestions || []).length ? `<p>修改建议：${item.remediation_suggestions.map(escapeHtml).join("；")}</p>` : ""}
      <p>标签：${escapeHtml(item.risk_labels.join("、") || "未标注")} · ${item.llm_reviewed ? (item.review_status === "needs_review" ? "语义已复核 · 上下文待补" : "语义已复核") : item.review_status === "needs_review" ? "语义仍待复核" : "本地规则判断"}</p>
      <p>说话人：${escapeHtml(roleLabels[context?.speaker_role] || "未标记")} · 立场：${escapeHtml(stanceLabels[context?.stance] || "待核对")}</p>
      <div class="context">${sources}</div>
      ${context?.warnings.length ? `<p class="warning">${escapeHtml(context.warnings.join("；"))}</p>` : ""}
      <ul>${evidence}</ul><p class="muted">${escapeHtml(retrievalDescription(item))}</p>
      <p>类案候选：${escapeHtml(item.matched_cases.map(c => c.case_id).join("、") || "无")}</p></section>`;
  }).join("");
  const cases = summary.top_matched_cases.map((item, index) => `
    <tr><td>${index + 1}</td><td>${escapeHtml(item.case_id)}<br>${escapeHtml(item.penalty_doc_no || "")}</td><td>${escapeHtml(item.party_name || item.case_title || "—")}</td><td>${escapeHtml(item.matched_risk_tags.join("、") || item.risk_tags.join("、"))}</td><td>${formatScore(item.score)}</td><td>${escapeHtml(item.match_reason || "相似事实匹配")}<p>${escapeHtml(item.penalty_content || "")}</p><small>${escapeHtml(item.regulator || "")}</small></td><td>${escapeHtml(item.file_id || item.matched_source)}<p>${escapeHtml(item.matched_text.slice(0,280))}</p><small>${(item.source_spans || []).map(s=>`${s.start}–${s.end}`).join("，")}</small></td></tr>
  `).join("");
  const retrievalNote = result.risk_assessments.some(item => item.retrieval_status === "unavailable")
    ? "部分类案检索暂不可用"
    : summary.retrieval_sentence_count ? "暂无相似案例。"
      : "本次无需检索类案";
  const warnings = summary.warnings.map(warning => `<p>${escapeHtml(warning)}</p>`).join("");
  const runs = summary.agent_runs || [];
  const trace = runs.length ? `Pi Agent ${escapeHtml(runs[0].version)} · ${runs.length} 次审查 · ${runs.reduce((n, run) => n + run.turns, 0)} 轮 · ${runs.reduce((n, run) => n + run.tool_calls, 0)} 次工具调用` : "本次未取得 Agent 语义复核结果";
  const html = `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>FinRisk审查报告-${escapeHtml(result.query_id)}</title>
    <style>body{max-width:1040px;margin:40px auto;padding:0 24px;color:#27272a;font:14px/1.85 "Segoe UI","Microsoft YaHei",sans-serif}header{border-bottom:1px solid #d4d4d8;padding-bottom:20px}h1{font-size:26px;font-weight:600}h2{font-size:19px;margin:32px 0 16px}h3{font-size:16px;font-weight:550}.meta{display:grid;grid-template-columns:repeat(3,1fr);gap:15px;margin:22px 0}.meta div{padding:12px;background:#fafafa;border:1px solid #e4e4e7}table{width:100%;border-collapse:collapse;margin:12px 0 24px}th,td{padding:10px;border:1px solid #e4e4e7;text-align:left;vertical-align:top;font-size:12px}th{background:#f4f4f5;font-weight:500}.source{padding:18px;background:#fafafa;white-space:pre-wrap;overflow-wrap:anywhere}.finding{padding:18px 0;border-bottom:1px solid #e4e4e7}.original{font-weight:500;white-space:pre-wrap}.context{padding:8px 14px;background:#fafafa;border-left:2px solid #d4d4d8}.context p{margin:6px 0;white-space:pre-wrap}.muted,small{color:#71717a;font-size:12px}.warning{background:#fffbf4;color:#85652c;padding:12px 16px}.foot{color:#71717a;font-size:11px;margin-top:30px}@media print{body{margin:0}.finding{break-inside:avoid}table{font-size:11px}}@media(max-width:600px){.meta{grid-template-columns:1fr}body{padding:0 14px}}</style>
    </head><body><header><h1>保险营销合规审查报告</h1><p>FinRisk · 结合上下文核对材料与原文依据</p></header>
    <section class="meta"><div>整体判断<br><strong>${escapeHtml(summary.overall_level_name)}</strong></div><div>风险程度分<br><strong>${formatScore(summary.overall_score)} / 1</strong></div><div>材料场景<br><strong>${escapeHtml(sceneLabels[result.classification.scene] || result.classification.scene)}</strong></div><div>风险标签<br>${escapeHtml(summary.risk_labels.join("、") || "未标注")}</div><div>建议动作<br>${escapeHtml(summary.action)}</div><div>仍需语义或人工复核<br>${summary.pending_review_sentence_count || 0} 句</div></section>
    <p class="muted">生成时间：${new Date().toLocaleString("zh-CN")} · 耗时 ${formatDuration(result.stage_timings_ms?.total)} · ${trace}</p>
    ${warnings ? `<div class="warning"><strong>复核提示</strong>${warnings}</div>` : ""}
    <h2>原始材料</h2><div class="source">${escapeHtml(result.source_text)}</div>
    <h2>逐句判断与上下文</h2>${findings}
    <h2>相关处罚案例</h2>
    <table><thead><tr><th>#</th><th>案例 / 文号</th><th>主体</th><th>共同标签</th><th>排序分</th><th>匹配理由与处罚</th><th>来源与原文摘录</th></tr></thead><tbody>${cases || `<tr><td colspan="7">${escapeHtml(retrievalNote)}</td></tr>`}</tbody></table>
    <p class="foot">Query ID：${escapeHtml(result.query_id)} · SHA-256：${escapeHtml(result.extraction.checksum_sha256)} · Schema：${escapeHtml(result.schema_version)}</p></body></html>`;
  downloadText(`FinRisk审查报告-${result.query_id}.html`, html, "text/html;charset=utf-8");
  toast("审查报告已导出，可用浏览器打印为 PDF");
}

function downloadText(filename, content, mediaType) {
  const blob = new Blob([content], { type: mediaType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

async function loadEvaluation() {
  const target = el("#evaluation-content");
  try {
    const data = await api("/api/evaluation", { cache: "no-store" });
    if (!data.available) {
      target.innerHTML = '<div class="inline-empty">暂无已发布的评测结果。</div>';
      return;
    }
    window.FinRiskEvaluation.mount(target, data);
  } catch (error) {
    target.innerHTML = `<div class="inline-empty error">${escapeHtml(error.message)}</div>`;
  }
}

function exportSubmission() {
  const result = state.riskResult;
  if (!result) return;
  const payload = { question_id: result.query_id,
    risk_type: result.risk_summary.risk_labels.join("；"),
    retrieved_cases: result.risk_summary.top_matched_cases.map((c, i) => ({case_id: c.case_id, rank: i + 1, reason: c.match_reason})),
    suggestion: [...new Set(result.risk_assessments.flatMap(a => a.remediation_suggestions || []))].join("；") };
  downloadText("submission.jsonl", JSON.stringify(payload) + "\n", "application/x-ndjson;charset=utf-8");
}

async function ingestCase() {
  showFormError("#case-error", "");
  if (!state.caseFile) return showFormError("#case-error", "请先选择案例原始文件。");
  const form = new FormData();
  form.append("file", state.caseFile);
  setBusy("#ingest-button", true);
  try {
    const result = await api("/api/cases/upload", { method: "POST", body: form });
    renderIngestionResult(result);
    state.caseFile = null;
    el("#case-file").value = "";
    el("#case-file-name").textContent = "尚未选择文件";
    await loadHealth();
  } catch (error) {
    showFormError("#case-error", error.message);
  } finally {
    setBusy("#ingest-button", false);
  }
}

function renderIngestionResult(result) {
  const record = result.case;
  const warnings = result.warnings.map((value) => `<li>${escapeHtml(value)}</li>`).join("");
  const container = el("#ingestion-result");
  container.hidden = false;
  container.innerHTML = `
    <div class="success-mark">✓</div>
    <div class="ingestion-copy">
      <p class="kicker">INGESTION COMPLETE</p>
      <h2>${escapeHtml(record.case_title || record.case_id)}${(result.cases || []).length > 1 ? ` · ${result.cases.length}个主体` : ""}</h2>
      <p>${escapeHtml(record.violation_behavior || "案例已完成结构化录入。")}</p>
      <div class="tag-row">${record.risk_tags.map((tag) => `<span class="tag">${escapeHtml(tag)}</span>`).join("")}</div>
      <small>案例编号 ${escapeHtml(record.case_id)} · ${result.llm_used ? "DeepSeek 结构化" : "规则降级结构化"} · ${escapeHtml(result.extraction_method)}</small>
      ${warnings ? `<ul class="warning-items">${warnings}</ul>` : ""}
      <button class="quiet-button" type="button" data-open-ingested="${escapeHtml(record.case_id)}">查看案例详情</button>
    </div>
  `;
  container.querySelector("[data-open-ingested]").addEventListener("click", () => openCase(record.case_id));
  container.scrollIntoView({ behavior: "smooth", block: "center" });
}

async function loadStatistics() {
  try {
    const value = await api("/api/cases/statistics");
    const items = [value.total_cases, value.active_cases, value.subject_case_count ?? value.total_cases, value.risk_tag_count];
    el("#case-stats").querySelectorAll("strong").forEach((node, index) => {
      node.textContent = items[index].toLocaleString("zh-CN");
    });
    el("#case-source-counts").textContent = `原始知识库 ${value.base_cases.toLocaleString("zh-CN")} · 官网采集 ${(value.crawled_cases || 0).toLocaleString("zh-CN")} · 网页录入 ${value.user_cases.toLocaleString("zh-CN")}`;
    await loadCollectionStatus();
  } catch (error) {
    toast(`统计读取失败：${error.message}`);
  }
}

function renderWorkspaceSummary(value) {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Shanghai", year: "numeric", month: "2-digit", day: "2-digit",
  }).formatToParts(new Date());
  const part = (type) => parts.find((item) => item.type === type).value;
  const fallbackDate = `${part("year")}-${part("month")}-${part("day")}`;
  const today = /^\d{4}-\d{2}-\d{2}$/.test(value?.today || "") ? value.today : fallbackDate;
  const [year, month, day] = today.split("-");
  const dateNode = el("#workspace-date");
  dateNode.dateTime = today;
  dateNode.textContent = `${year}年${month}月${day}日`;
  const count = value?.today_crawled_cases;
  const available = Number.isInteger(count) && count >= 0;
  el("#today-crawled-count").textContent = available ? count.toLocaleString("zh-CN") : "—";
  el(".workspace-summary").title = available
    ? "北京时间今日成功入库的官网案例，按文书去重；不含网页手动录入"
    : "今日采集数暂不可用，正在等待服务更新";
}

async function loadCollectionStatus() {
  try {
    const value = await api("/api/collection/status");
    renderWorkspaceSummary(value);
    const node = el("#collection-status");
    const statusNames = { running: "采集中", success: "已更新", partial: "待续更新", paused: "官网访问暂停", interrupted: "待下次续跑", waiting: "等待新案例", collected: "已采集" };
    const next = value.next_run_at ? new Date(value.next_run_at).toLocaleString("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false }) : "";
    node.textContent = value.enabled ? `每日 ${value.daily_time} · ${statusNames[value.status] || "已启用"}${next ? ` · 下次 ${next}` : ""}` : "每日更新未启用";
    node.title = [value.start_date ? `起始日期：${value.start_date}` : "", `待处理 ${(value.pending_parts || 0) + (value.pending_documents || 0)} 项`, `索引：${value.index_status}`].filter(Boolean).join("；");
  } catch {
    renderWorkspaceSummary(null);
    el("#collection-status").textContent = "采集状态暂不可用";
  }
}

async function loadCases() {
  const params = new URLSearchParams({
    page: String(state.casePage),
    page_size: String(state.casePageSize),
  });
  const filters = {
    q: el("#case-search").value.trim(),
    risk_tag: el("#case-risk-tag").value.trim(),
    retrieval_status: el("#case-status").value,
    origin: el("#case-origin").value,
regulator: el("#case-regulator").value.trim(),
    party_name: el("#case-party-name").value.trim(),
    institution_type: el("#case-institution-type").value.trim(),
    penalty_date_from: el("#case-date-from").value,
    penalty_date_to: el("#case-date-to").value,
  };
  Object.entries(filters).forEach(([key, value]) => value && params.set(key, value));
  el("#case-list").innerHTML = '<div class="inline-empty">正在读取案例……</div>';
  try {
    state.caseResult = await api(`/api/cases?${params}`);
    renderCaseList();
  } catch (error) {
    el("#case-list").innerHTML = `<div class="inline-empty error">${escapeHtml(error.message)}</div>`;
  }
}

function caseSourceName(item) {
  return ({ original: "原始知识库", official_crawl: "官网采集", web_ingestion: "网页录入" })[item.source_group] || "原始知识库";
}

function caseRetrievalStatus(item) {
  if (item.retrieval_disposition === "archive_only") return "仅归档";
  if (item.retrieval_disposition === "scope_review") return "范围待核";
  return item.retrieval_status === "active" ? "启用" : "已停用";
}

function caseDocumentKind(item) {
  return ({ penalty: "处罚材料", pre_penalty_notice: "拟处罚公告", non_penalty: "许可等非处罚材料", penalty_delivery_notice: "处罚送达公告" })[item.document_kind] || item.document_kind;
}

function renderCaseList() {
  const result = state.caseResult;
  el("#case-count-label").textContent = `共 ${result.total.toLocaleString("zh-CN")} 份文书 · 第 ${result.page} / ${Math.max(result.total_pages, 1)} 页`;
  if (!result.items.length) {
    el("#case-list").innerHTML = '<div class="inline-empty">没有符合条件的案例。</div>';
  } else {
    el("#case-list").innerHTML = result.items.map((item) => `
      <article class="case-row ${item.retrieval_status === "inactive" ? "is-inactive" : ""}">
        <button class="case-main" type="button" data-open-case="${escapeHtml(item.case_id)}">
          <span class="case-id">${escapeHtml(item.document_id || item.case_id)}<small>${item.case_origin === "source_document_only" ? "主体尚未归属" : `${item.subject_case_count || 1} 个主体案例`}</small></span>
          <span class="case-title">${escapeHtml(item.case_title || item.penalty_doc_no || "未命名案例")}</span>
          <span class="case-fact">${escapeHtml(item.violation_behavior || "暂无违法事实摘要")}</span>
          <span class="tag-row">${item.risk_tags.slice(0, 4).map((tag) => `<i>${escapeHtml(tag)}</i>`).join("")}</span>
        </button>
        <div class="case-meta">
          <span>${escapeHtml(item.regulator || "监管机构未记录")}</span>
          <span>${formatDate(item.penalty_date)}</span>
          <span class="origin-chip">${caseSourceName(item)}</span>
          <span class="status-chip ${escapeHtml(item.retrieval_status)}">${caseRetrievalStatus(item)}</span>
        </div>
      </article>
    `).join("");
  }
  renderPagination(result);
}

function renderPagination(result) {
  if (result.total_pages <= 1) {
    el("#case-pagination").replaceChildren();
    return;
  }
  const pages = new Set([1, result.total_pages, result.page - 1, result.page, result.page + 1]);
  const valid = [...pages].filter((page) => page >= 1 && page <= result.total_pages).sort((a, b) => a - b);
  let previous = 0;
  const parts = [];
  valid.forEach((page) => {
    if (page - previous > 1) parts.push("<span>…</span>");
    parts.push(`<button type="button" data-page="${page}" class="${page === result.page ? "is-active" : ""}">${page}</button>`);
    previous = page;
  });
  el("#case-pagination").innerHTML = parts.join("");
}

function handleCaseListClick(event) {
  const button = event.target.closest("[data-open-case]");
  if (button) openCase(button.dataset.openCase);
}

function handleCaseLink(event) {
  const source = event.target.closest("[data-source-sentence]");
  if (source) {
    const owner = source.closest("[data-sentence-id]")?.dataset.sentenceId || source.dataset.sourceSentence;
    selectSourceSentence({
      sentenceId: owner,
      sourceSentenceId: source.dataset.sourceSentence,
      ...(source.hasAttribute("data-source-start") ? {
        start: Number(source.dataset.sourceStart), end: Number(source.dataset.sourceEnd),
      } : {}),
    });
    const target = [...el("#source-highlight").querySelectorAll("[data-sentence-target]")]
      .find(mark => mark.dataset.sentenceTarget === source.dataset.sourceSentence);
    el("#source-highlight").scrollIntoView({ behavior: "smooth", block: "center" });
    target?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    return;
  }
  const button = event.target.closest("[data-case-id]");
  if (button) openCase(button.dataset.caseId);
}

function handlePagination(event) {
  const button = event.target.closest("[data-page]");
  if (!button) return;
  state.casePage = Number(button.dataset.page);
  loadCases();
  el("#cases-view").scrollIntoView({ behavior: "smooth" });
}

async function openCase(caseId) {
  el("#case-detail").innerHTML = '<div class="inline-empty">正在读取案例详情……</div>';
  el("#drawer-backdrop").hidden = false;
  el("#case-drawer").classList.add("is-open");
  el("#case-drawer").setAttribute("aria-hidden", "false");
  document.body.classList.add("drawer-open");
  try {
    const item = await api(`/api/cases/${encodeURIComponent(caseId)}`);
    renderCaseDetail(item);
  } catch (error) {
    el("#case-detail").innerHTML = `<div class="inline-empty error">${escapeHtml(error.message)}</div>`;
  }
}

function renderCaseDetail(item) {
  const fieldLabels = { party_name: "当事人", penalty_doc_no: "处罚文号", penalty_date: "处罚日期",
    regulator: "监管机构", case_summary: "案例摘要", penalty_content: "处罚内容", behavior_period: "行为期间" };
  const evidence = item.evidence_chunks.slice(0, 12).map((chunk) => `
    <article><span>${escapeHtml(chunk.chunk_type)}${chunk.is_verbatim ? " · 原文" : ""}</span><p>${escapeHtml(chunk.chunk_text)}</p></article>
  `).join("");
  el("#case-detail").innerHTML = `
    <p class="kicker">CASE DETAIL</p>
    <span class="drawer-case-id">${escapeHtml(item.case_id)}</span>
    <h2>${escapeHtml(item.case_title || item.penalty_doc_no || "未命名案例")}</h2>
    <div class="tag-row">${item.risk_tags.map((tag) => `<span class="tag">${escapeHtml(tag)}</span>`).join("")}</div>
    <div class="detail-section">
      ${item.original_pdf_available ? `<a class="quiet-button" data-original-pdf href="/api/cases/${encodeURIComponent(item.case_id)}/source" target="_blank" rel="noopener">打开原文 PDF ↗</a>` : '<button class="quiet-button" type="button" disabled>暂无原文 PDF</button>'}
      ${item.source_url && /^https?:\/\//i.test(item.source_url) ? `<a class="quiet-button" href="${escapeHtml(item.source_url)}" target="_blank" rel="noopener">查看来源网页 ↗</a>` : ""}
    </div>
    <dl class="case-fields">
      <div><dt>当事人</dt><dd>${escapeHtml(item.party_name || "—")}</dd></div>
      <div><dt>处罚文号</dt><dd>${escapeHtml(item.penalty_doc_no || "—")}</dd></div>
      <div><dt>处罚机关</dt><dd>${escapeHtml(item.regulator || "—")}</dd></div>
      <div><dt>处罚日期</dt><dd>${formatDate(item.penalty_date)}</dd></div>
      <div><dt>机构类型</dt><dd>${escapeHtml(item.institution_type || "—")}</dd></div>
      <div><dt>发布日期</dt><dd>${formatDate(item.publish_date)}</dd></div>
      <div><dt>数据来源</dt><dd>${caseSourceName(item)}</dd></div>
      <div><dt>检索状态</dt><dd>${caseRetrievalStatus(item)}</dd></div>
      <div><dt>文书类型</dt><dd>${escapeHtml(caseDocumentKind(item))}</dd></div>
      <div><dt>审核状态</dt><dd>模型候选 · 待人工核对</dd></div>
      ${item.decision_status && item.decision_status !== "not_individually_verified" ? `<div><dt>决定状态</dt><dd>${escapeHtml(item.decision_status)}</dd></div>` : ""}
    </dl>
    <section class="detail-section"><h3>违法事实</h3><p>${escapeHtml(item.violation_behavior || "—")}</p></section>
    <section class="detail-section"><h3>处罚内容</h3><p>${escapeHtml(item.penalty_content || "—")}</p></section>
    ${item.case_summary ? `<section class="detail-section"><h3>案例摘要</h3><p>${escapeHtml(item.case_summary)}</p></section>` : ""}
    ${(item.entity_relations || []).length ? `<section class="detail-section"><h3>主体关联</h3>${item.entity_relations.map(r => `<p>${escapeHtml(r.source || "")} · ${escapeHtml(r.relation || "")} · ${escapeHtml(r.target || "")}</p>`).join("")}</section>` : ""}
    ${Object.keys(item.field_quality || {}).length ? `<details class="detail-section"><summary>字段来源</summary>${Object.entries(item.field_quality).map(([field, quality]) => `<p>${escapeHtml(fieldLabels[field] || field)} · ${escapeHtml(quality)}</p>`).join("")}</details>` : ""}
    ${item.raw_source_file && !item.original_pdf_available ? `<section class="detail-section"><a class="quiet-button" href="/api/cases/${encodeURIComponent(item.case_id)}/source" target="_blank" rel="noopener">查看来源原件 ↗</a></section>` : ""}
    ${(item.related_cases || []).length ? `<section class="detail-section"><h3>同文书其他主体</h3>${item.related_cases.map(subject => `<article><button class="quiet-button" type="button" data-related-case="${escapeHtml(subject.case_id)}">${escapeHtml(subject.party_name || subject.case_title || subject.case_id)}</button><p>${escapeHtml(subject.violation_behavior)}</p></article>`).join("")}</section>` : ""}
    <section class="detail-section"><h3>证据切片 <small>${item.evidence_chunks.length}</small></h3><div class="evidence-list">${evidence || '<p class="muted">暂无证据切片。</p>'}</div></section>
    <div class="drawer-actions">
      <button class="${item.retrieval_status === "active" ? "danger-button" : "primary-button"}" id="toggle-case-status" type="button" ${["archive_only", "scope_review"].includes(item.retrieval_disposition) ? "disabled" : ""}>
        ${item.retrieval_status === "active" ? "停用案例" : "恢复案例"}
      </button>
      <small>${item.retrieval_status === "active" ? "停用后不再参与检索，但数据和原文仍保留。" : "恢复后将重新参与 Query 案例匹配。"}</small>
    </div>
  `;
  el("#toggle-case-status").addEventListener("click", () => toggleCaseStatus(item));
  el("#case-detail").querySelectorAll("[data-related-case]").forEach(button => {
    button.addEventListener("click", () => openCase(button.dataset.relatedCase));
  });
}

async function toggleCaseStatus(item) {
  const active = item.retrieval_status !== "active";
  if (!active && !window.confirm("确认停用该案例？记录和原始文件会保留，可随时恢复。")) return;
  setBusy("#toggle-case-status", true);
  try {
    const updated = await api(`/api/cases/${encodeURIComponent(item.case_id)}/status`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ active }),
    });
    renderCaseDetail(updated);
    toast(active ? "案例已恢复" : "案例已停用，可随时恢复");
    if (state.activeView === "cases") {
      await Promise.all([loadStatistics(), loadCases()]);
    }
  } catch (error) {
    toast(error.message);
  }
}

function closeDrawer() {
  el("#case-drawer").classList.remove("is-open");
  el("#case-drawer").setAttribute("aria-hidden", "true");
  el("#drawer-backdrop").hidden = true;
  document.body.classList.remove("drawer-open");
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  let payload;
  try {
    payload = await response.json();
  } catch (error) {
    throw new Error(`服务返回了无法解析的响应（HTTP ${response.status}）`);
  }
  if (!response.ok) throw new Error(readApiError(payload));
  return payload;
}

function readApiError(payload) {
  if (typeof payload?.detail === "string") return payload.detail;
  if (Array.isArray(payload?.detail)) {
    return payload.detail.map((item) => item.msg || "字段校验失败").join("；");
  }
  return "请求失败，请查看服务日志。";
}

function setBusy(selector, busy) {
  const button = el(selector);
  if (!button) return;
  button.disabled = busy;
  button.classList.toggle("is-loading", busy);
  const label = button.querySelector("span");
  if (label) {
    if (!button.dataset.idleLabel) button.dataset.idleLabel = label.textContent;
    label.textContent = busy ? (selector === "#analyze-button" ? "审查中" : "处理中") : button.dataset.idleLabel;
  }
}

function showFormError(selector, message) {
  el(selector).textContent = message;
}

function toast(message) {
  const node = el("#toast");
  node.textContent = message;
  node.classList.add("is-visible");
  window.clearTimeout(toast.timer);
  toast.timer = window.setTimeout(() => node.classList.remove("is-visible"), 2800);
}

function fileLabel(file) {
  return file ? `${file.name} · ${formatFileSize(file.size)}` : "尚未选择文件";
}

function formatFileSize(size) {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / 1024 / 1024).toFixed(1)} MB`;
}

function formatPercent(value) {
  return `${Math.round(Number(value) * 100)}%`;
}

function formatScore(value) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(2) : "—";
}

function retrievalDescription(item) {
  const status = item.retrieval_status || (item.retrieval_performed
    ? item.matched_cases.length ? "candidates" : "no_sufficient_match" : "not_requested");
  if (status === "unavailable") return "类案检索暂不可用。";
  if (status === "not_requested") return "本句未触发类案检索。";
  if (status === "no_sufficient_match") return "暂无相似案例。";
  return `匹配 ${item.matched_cases.length} 条相关案例。`;
}

function renderSentenceContext(context) {
  if (!context) return "";
  const lines = context.source_sentences.map(source => `
    <div class="context-line ${source.is_target ? "is-target" : ""}">
      <button class="source-link" type="button" data-source-sentence="${escapeHtml(source.sentence_id)}" ${sourceRangeAttributes(source)}>${source.is_target ? "当前" : "上下文"} ${sourceLabel(source.sentence_id)}</button>
      <p>${escapeHtml(source.text)}</p>
    </div>
  `).join("");
  return `<details class="context-details" ${context.requires_review ? "open" : ""}>
    <summary>审查语境 · ${escapeHtml(stanceLabels[context.stance] || "待核对")}${context.requires_review ? " · 需结合上下文" : ""}</summary>
    <div class="context-lines">${lines}</div>
    ${context.review_reasons.map(reason => `<p class="muted">${escapeHtml(reason)}</p>`).join("")}
    ${context.warnings.map(warning => `<p class="context-warning">${escapeHtml(warning)}</p>`).join("")}
  </details>`;
}

function formatDate(value) {
  return value ? new Intl.DateTimeFormat("zh-CN").format(new Date(`${value}T00:00:00`)) : "—";
}

function formatDuration(milliseconds) {
  if (milliseconds == null || Number.isNaN(Number(milliseconds))) return "—";
  const value = Number(milliseconds);
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${Math.round(value)} ms`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;")
    .replaceAll("\r", "&#13;");
}

const initialView = window.location.hash.slice(1);
if (["risk", "case-entry", "cases"].includes(initialView)) switchView(initialView);
