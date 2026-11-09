/* Dated, source-bound evaluation results. All displayed text is escaped. */
(function (root) {
  "use strict";
  const escape = value => String(value ?? "—").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const pct = value => Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : "—";
  const fixed = value => Number.isFinite(value) ? value.toFixed(3) : "—";
  const seconds = value => Number.isFinite(value) ? `${value.toFixed(2)} s` : "—";
  const states = {risk:"明确风险", no_clear_risk:"未见明确风险", insufficient_information:"信息不足"};
  const modes = {deepseek_adaptive:"当前版本", rules_only:"无大模型（规则与轻量模型）"};
  const scenes = {ST1_poster_sms_social:"海报 / 短信 / 社交文案", ST2_live_short_video:"直播 / 短视频 / 口播", ST3_product_intro_detail:"产品介绍 / 详情页", ST4_meeting_material:"说明会 / 客户活动", ST5_training_telemarketing_callback:"培训 / 电销 / 回访", ST6_penalty_backend_fact:"监管处罚 / 后台事实"};
  const metricNames = {
    queries:"样本数", state_accuracy:"三态准确率", risk_binary_accuracy:"风险二分类准确率",
    risk_precision:"风险精确率", risk_recall:"风险召回率", risk_f1:"风险 F1",
    scene_accuracy:"场景准确率", label_micro_f1:"标签 Micro-F1", label_macro_f1:"标签 Macro-F1（27 类）",
    label_supported_macro_f1:"有支持样本标签 Macro-F1", median_seconds:"中位耗时", p95_seconds:"P95 耗时",
    warnings:"含警告 Query", errors:"查询错误", missing_model_targets:"未完成模型目标",
    long_queries:"900 字以上 Query", long_median_seconds:"900 字以上中位耗时",
    judged_queries:"已判定相关性 Query", positive_queries:"有强相关案例 Query", no_positive_queries:"无强相关案例 Query",
    hit1:"Hit@1", hit5:"Hit@5", mrr:"MRR@20", recall_at_3:"Recall@3", recall_at_5:"Recall@5",
    recall_at_10:"Recall@10", recall_at_20:"Recall@20", ndcg_at_10:"NDCG@10",
  };
  function metric(key, value) {
    if (value === null || value === undefined) return "—";
    if (key.includes("seconds")) return seconds(value);
    if (key === "mrr" || key === "ndcg_at_10") return fixed(value);
    if (/accuracy|precision|recall|f1|^hit/.test(key)) return pct(value);
    return escape(value);
  }
  function table(headers, rows, cls = "") {
    return `<div class="evaluation-scroll"><table class="evaluation-table ${cls}"><thead><tr>${headers.map(h=>`<th scope="col">${escape(h)}</th>`).join("")}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(cell=>`<td>${cell}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  }
  function comparisonChart(run, kind) {
    const quality = kind === "quality";
    const metrics = quality
      ? [["risk_binary_accuracy", "准确率"], ["risk_recall", "召回率"], ["risk_f1", "F1"]]
      : [["median_seconds", "中位耗时"], ["p95_seconds", "P95 耗时"]];
    const series = [
      { key: "rules_only", name: "无大模型", color: "#9aa3ad" },
      { key: "deepseek_adaptive", name: "当前版本", color: "#278464" },
    ];
    const width = 820, height = 310, left = 62, right = 28, top = 35, bottom = 258;
    const plotWidth = width - left - right, plotHeight = bottom - top;
    const maximum = Math.max(...metrics.flatMap(([key]) => series.map(s => run.risk[s.key][key])));
    const ceiling = quality ? 1 : Math.max(1, Math.ceil(maximum / 4) * 4);
    const ticks = Array.from({ length: 5 }, (_, i) => ceiling * i / 4);
    const format = value => quality ? pct(value) : seconds(value);
    const grid = ticks.map(value => {
      const y = bottom - value / ceiling * plotHeight;
      return `<g class="evaluation-chart-tick"><line x1="${left}" x2="${width-right}" y1="${y}" y2="${y}"/><text x="${left-12}" y="${y+4}" text-anchor="end">${quality ? `${Math.round(value*100)}%` : `${Number(value.toFixed(2))}`}</text></g>`;
    }).join("");
    const groupWidth = plotWidth / metrics.length, barWidth = Math.min(64, groupWidth * .22), gap = 12;
    const bars = metrics.map(([key, label], index) => {
      const center = left + groupWidth * (index + .5);
      return series.map((s, si) => {
        const value = run.risk[s.key][key], barHeight = value / ceiling * plotHeight;
        const x = center + (si === 0 ? -gap/2-barWidth : gap/2), y = bottom-barHeight;
        return `<g class="evaluation-chart-bar" data-series="${s.key}" data-metric="${key}" data-value="${value}"><rect x="${x}" y="${y}" width="${barWidth}" height="${barHeight}" fill="${s.color}" rx="3"><title>${s.name} · ${escape(label)}：${format(value)}</title></rect><text x="${x+barWidth/2}" y="${y-9}" text-anchor="middle">${format(value)}</text></g>`;
      }).join("") + `<text class="evaluation-chart-category" x="${center}" y="${bottom+28}" text-anchor="middle">${escape(label)}</text>`;
    }).join("");
    const title = quality ? "准确率、召回率与 F1" : "处理耗时";
    const description = quality
      ? "风险二分类指标，越高越好。F1 综合衡量精确率与召回率；标签 Micro-F1 见完整对比数值。"
      : "同一批 200 条样本，中位耗时与 P95 分开比较，越低越好。纵轴单位为秒，从 0 开始。";
    return `<figure class="evaluation-chart"><figcaption><h4>${title}</h4><p class="muted">${description}</p></figcaption><div class="evaluation-chart-legend">${series.map(s=>`<span><i style="background:${s.color}"></i>${s.name}</span>`).join("")}</div><div class="evaluation-chart-scroll"><svg viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="evaluation-chart-${kind}-title evaluation-chart-${kind}-desc"><title id="evaluation-chart-${kind}-title">${title}：无大模型与当前版本对比</title><desc id="evaluation-chart-${kind}-desc">${metrics.map(([key,label])=>`${label}，${series.map(s=>`${s.name} ${format(run.risk[s.key][key])}`).join("，")}`).join("；")}</desc><text class="evaluation-chart-unit" x="${left}" y="17">${quality ? "百分比（%）" : "秒（s）"}</text>${grid}${bars}</svg></div></figure>`;
  }
  function labelsTable(values) {
    return table(["标签", "参考数量", "TP 命中", "FP 误报", "FN 漏报", "精确率", "召回率", "F1"], Object.entries(values).map(([name,v])=>[
      escape(name), v.support, v.tp, v.fp, v.fn, v.tp+v.fp ? pct(v.tp/(v.tp+v.fp)) : "—", v.support ? pct(v.tp/v.support) : "—", pct(v.f1),
    ]));
  }
  function groupTable(values, names = {}) {
    return table(["分组", "样本数", "二分类准确率", "三态准确率", "风险召回率", "标签 Micro-F1", "场景准确率", "中位耗时", "P95"], Object.entries(values).map(([k,v])=>[
      escape(names[k]||k), v.queries, pct(v.risk_binary_accuracy), pct(v.state_accuracy), pct(v.risk_recall), pct(v.label_micro_f1), pct(v.scene_accuracy), seconds(v.median_seconds), seconds(v.p95_seconds),
    ]));
  }
  function outcomes(row, reference) {
    const a = new Set(reference.risk_labels), p = new Set(row.risk_labels || []);
    return {
      state: row.decision_state === reference.decision_state,
      scene: row.scene === reference.scene,
      fp: [...p].filter(x=>!a.has(x)), fn: [...a].filter(x=>!p.has(x)),
      falsePositive: row.decision_state === "risk" && reference.decision_state !== "risk",
      falseNegative: row.decision_state !== "risk" && reference.decision_state === "risk",
    };
  }
  function filterPredictions(predictions, inputs, query, filter) {
    const q = query.trim().toLocaleLowerCase();
    return predictions.filter(row=>{
      const input=inputs[row.query_id], o=outcomes(row,input.reference);
      const matches = !q || `${row.query_id} ${input.text} ${row.risk_labels.join(" ")} ${input.reference.risk_labels.join(" ")}`.toLocaleLowerCase().includes(q);
      return matches && (filter==="all" || (filter==="state"&&!o.state) || (filter==="labels"&&(o.fp.length||o.fn.length)) || (filter==="scene"&&!o.scene) || (filter==="fp"&&o.falsePositive) || (filter==="fn"&&o.falseNegative));
    });
  }
  function renderDashboard(data) {
    const run=data.runs.find(r=>r.id===data.selected_run), risk=run.risk[data.selected_mode];
    const stale=data.freshness?.status!=="current";
    const cards=[["风险识别准确率",pct(risk.risk_binary_accuracy),"200 条测试样本"],["风险精确率",pct(risk.risk_precision),"判为风险的样本中实际有风险的比例"],["风险召回率",pct(risk.risk_recall),"识别出已有风险的比例"],["风险 F1",pct(risk.risk_f1),"综合衡量精确率与召回率"]];
    const comparisonKeys=["risk_binary_accuracy","state_accuracy","risk_precision","risk_recall","risk_f1","label_micro_f1","label_macro_f1","scene_accuracy","median_seconds","p95_seconds"];
    const comparison=table(["指标",modes.rules_only,modes.deepseek_adaptive],comparisonKeys.map(key=>[escape(metricNames[key]),metric(key,run.risk.rules_only[key]),metric(key,risk[key])]),"evaluation-comparison-table");
    const retrievalKeys=["hit1","hit5","recall_at_10","mrr","ndcg_at_10","median_seconds"];
    return `<div class="evaluation-intro"><span class="evaluation-badge">当前启用版本 · 发布验收</span><span class="evaluation-badge">评测日期 ${escape(data.evaluated_on)}</span><p>200 条测试样本 · 27 类风险标签 · 六类场景</p></div>
      <div class="summary-grid">${cards.map(([a,b,c])=>`<article class="summary-card"><span>${a}</span><strong>${b}</strong><small>${c}</small></article>`).join("")}</div>
      <p class="muted evaluation-context">DeepSeek 自适应融合 · 三态准确率 ${pct(risk.state_accuracy)} · 中位耗时 ${seconds(risk.median_seconds)}。本页统一采用当前算法的发布验收记录。</p>
      ${stale?'<p class="evaluation-warning">当前版本与评测记录的绑定已变化，请核对评测结果。</p>':""}
      <section id="evaluation-comparison"><h3>无大模型与当前版本对比</h3>${comparisonChart(run,"quality")}${comparisonChart(run,"time")}<details class="evaluation-details"><summary>查看完整对比数值（含精确率与标签 F1）</summary>${comparison}</details></section>
      <section id="evaluation-retrieval"><h3>当前版本 · 案例检索</h3><p class="muted">500 案例闭池，187 条 Query 有强相关案例。Hit 为至少命中一条，Recall 为相关案例集合的覆盖率。</p>${table(retrievalKeys.map(k=>metricNames[k]),[retrievalKeys.map(k=>metric(k,run.retrieval[k]))])}</section>
      <section id="evaluation-detail"><h3>详细测试结果</h3><p class="muted">以下均为当前启用版本的结果，可按需展开。</p>${renderRun(run,data.selected_mode)}
        <details class="evaluation-details"><summary>200 条测试样本 · 逐条查看</summary><div class="evaluation-controls"><label class="evaluation-search">搜索原文、编号或标签<input id="evaluation-search" type="search" placeholder="输入编号、关键词或风险标签" /></label><label>筛选结果<select id="evaluation-filter"><option value="all">全部结果</option><option value="state">三态不一致</option><option value="labels">标签误报或漏报</option><option value="scene">场景不一致</option><option value="fp">风险误报</option><option value="fn">风险漏报</option></select></label></div><div id="evaluation-query-results"></div></details>
      </section>
      <details class="evaluation-details"><summary>评测说明</summary><ul class="evaluation-notes">${data.notes.map(n=>`<li>${escape(n)}</li>`).join("")}</ul></details>`;
  }
  function renderRun(run, mode) {
    const detail=run.modes[mode], risk=run.risk[mode];
    return `<details class="evaluation-details"><summary>27 类风险标签 · 识别表现</summary><p class="muted">TP 为正确检出，FP 为误报，FN 为漏报。</p>${labelsTable(risk.per_label)}</details>
      <details class="evaluation-details"><summary>六类场景与文本长度 · 分组表现</summary><h4>六类场景</h4>${groupTable(detail.scenes,scenes)}<h4>文本长度</h4><p class="muted">900 字及以上仅 1 条样本。</p>${groupTable(detail.lengths)}</details>
      <details class="evaluation-details"><summary>三态判断 · 混淆矩阵</summary><p class="muted">行是参考状态，列是预测状态。</p>${table(["参考 ＼ 预测",...Object.values(states)],Object.entries(detail.confusion).map(([s,v])=>[states[s],...Object.keys(states).map(p=>v[p])]))}</details>`;
  }
  function renderSamples(run, mode, inputs, query="", filter="all", page=1) {
    const all=filterPredictions(run.modes[mode].predictions,inputs,query,filter);
    const pages=Math.max(1,Math.ceil(all.length/20)), current=Math.min(page,pages);
    const items=all.slice((current-1)*20,current*20).map(row=>{
      const input=inputs[row.query_id], ref=input.reference, o=outcomes(row,ref), rank=run.rankings[row.query_id];
      const badges=[!o.state?"三态不一致":"三态一致",...(o.fp.length?[`标签误报 ${o.fp.length}`]:[]),...(o.fn.length?[`标签漏报 ${o.fn.length}`]:[]),...(!o.scene?["场景不一致"]:[])];
      return `<details class="evaluation-sample"><summary><strong>${escape(row.query_id)}</strong><span>${escape(states[ref.decision_state])} → ${escape(states[row.decision_state]||row.decision_state)}</span><span class="muted">${escape(badges.join(" · "))} · ${seconds(row.seconds)}</span></summary><div class="evaluation-sample-body"><h4>测试原文</h4><p class="evaluation-query-text">${escape(input.text)}</p>${table(["项目","参考","预测"],[["状态",states[ref.decision_state],escape(states[row.decision_state]||row.decision_state)],["场景",escape(scenes[ref.scene]),escape(scenes[row.scene]||row.scene)],["标签",escape(ref.risk_labels.join("、")||"无"),escape(row.risk_labels.join("、")||"无")]])}<p>标签误报：${escape(o.fp.join("、")||"无")}；标签漏报：${escape(o.fn.join("、")||"无")}</p><p class="muted">参考理由：${escape(ref.reason)}</p><p class="muted">参考证据：${escape((ref.evidence||[]).join("；"))}</p><p class="muted">未完成目标 ${row.missing_model_targets??0} · 警告 ${escape((row.warnings||[]).join("；")||"无")} · 错误 ${escape(row.error||"无")}</p><h4>独立检索 Top20</h4><p class="muted">强相关案例总数 ${rank.relevant_count} · 原文校验 ${rank.literal_verified?"通过":"未通过"} · ${seconds(rank.seconds)}${rank.metrics?` · Hit@5 ${pct(rank.metrics.hit5)} · Recall@10 ${pct(rank.metrics.recall_at_10)} · NDCG@10 ${fixed(rank.metrics.ndcg_at_10)}`:" · 无强相关案例，不纳入排序质量均值"}</p>${table(["排名","案例 ID","相关等级（≥2 为强相关）"],rank.case_ids.map((id,i)=>[i+1,escape(id),rank.grades[i]]))}</div></details>`;
    }).join("");
    return `<p class="muted">${modes[mode]} · 符合条件 ${all.length} / 200 条 · 第 ${current} / ${pages} 页</p>${items||'<div class="inline-empty">没有符合条件的结果</div>'}<div class="evaluation-pagination"><button type="button" class="quiet-button" data-evaluation-page="${current-1}" ${current===1?"disabled":""}>上一页</button><span>${current} / ${pages}</span><button type="button" class="quiet-button" data-evaluation-page="${current+1}" ${current===pages?"disabled":""}>下一页</button></div>`;
  }
  function mount(target,data) {
    target.innerHTML=renderDashboard(data);
    const find=s=>target.querySelector(s);
    const run=data.runs.find(r=>r.id===data.selected_run);
    let page=1;
    const samples=()=>{find("#evaluation-query-results").innerHTML=renderSamples(run,data.selected_mode,data.inputs,find("#evaluation-search").value,find("#evaluation-filter").value,page);};
    find("#evaluation-search").addEventListener("input",()=>{page=1;samples();});
    find("#evaluation-filter").addEventListener("change",()=>{page=1;samples();});
    find("#evaluation-query-results").addEventListener("click",event=>{
      const button=event.target.closest("[data-evaluation-page]");
      if(button){page=Number(button.dataset.evaluationPage);samples();}
    });
    samples();
  }
  const exported={mount,renderDashboard,renderRun,renderSamples,filterPredictions,outcomes,comparisonChart};
  if(typeof module!=="undefined"&&module.exports) module.exports=exported;
  else root.FinRiskEvaluation=exported;
})(typeof window!=="undefined"?window:globalThis);
