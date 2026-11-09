(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.FinRiskProgress = api;
})(typeof globalThis === "object" ? globalThis : this, function () {
  const terminal = { completed: "流程完成", partial: "部分完成", failed: "执行失败", interrupted: "执行中断", rejected: "任务被拒绝" };
  function caseCounts(counts) { return `未完成查询${counts.pending_case_queries}项 · ` + (Number.isInteger(counts.pending_case_points) ? `案例待核${counts.pending_case_points}处` : "案例待核未记录"); }
  function statusLine(snapshot, taskStatus) {
    if (!snapshot) return terminal[taskStatus] || "任务已登记，等待开始处理……";
    const counts = snapshot.counts, limits = snapshot.limits;
    const pieces = [terminal[taskStatus] || snapshot.stage_label,
      `已分析${counts.segments_analyzed}/${counts.segments_total}段`,
      `候选${counts.risk_points}处`, `已读${counts.read_sources}份`,
      `${snapshot.execution_mode === "mock" ? "模拟调用" : "模型调用"}${limits.model_requests.used}/${limits.model_requests.limit}`,
      `工具${limits.tool_actions.used}/${limits.tool_actions.limit}`];
    if (!terminal[taskStatus]) pieces.push(`剩余${snapshot.time.remaining_seconds}秒`);
    if (counts.pending_input_conditions) pieces.push(`待核条件${counts.pending_input_conditions}项`);
    pieces.push(caseCounts(counts));
    return pieces.join(" · ");
  }
  function completionText(task) {
    if (task.task_status === "partial") return task.report?.validation?.schema_valid === false
      ? "执行未完成，尚未形成有效终稿；请查看运行记录。"
      : "部分审查完成，仍有未完成事项；请查看结论和运行记录。";
    if (task.task_status === "completed") return "审查流程完成；风险意见和待复核条件见下方。";
    return `${terminal[task.task_status] || "执行未完成"}；未自动重新运行。`;
  }
  function historyLine(snapshot) { return `${snapshot.stage_label} · 模型${snapshot.limits.model_requests.used}/${snapshot.limits.model_requests.limit} · 已读${snapshot.counts.read_sources}份 · ${caseCounts(snapshot.counts)}`; }
  return { statusLine, completionText, historyLine };
});
