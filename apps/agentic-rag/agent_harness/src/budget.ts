import type { Config } from "./contracts.js";
export class Budget {
  private started: number;
  private model = new Map<string, { reserved: number; total?: number; unknown: boolean }>();
  private tools = new Map<string, string>();
  private rejected = new Set<string>();
  private searches = 0;
  queryAttempts = 0;
  externalRequests = 0;
  physicalRequests = 0;
  constructor(public config: Config, public clock: () => number = () => performance.now()) {
    this.started = clock();
    if (config.max_model_requests < 2 || config.provider_auto_retries !== 0 || config.max_model_transport_retries !== 0) throw new Error("INVALID_BUDGET_CONFIG");
  }
  private checkTime(): void { if (this.remainingMs() <= 0) throw new Error("deadline_exceeded"); }
  reserveModel(input: number, final: boolean): string {
    this.checkModel(input, final);
    const output = this.config.max_output_tokens_per_request;
    const id = `MR${String(this.model.size + 1).padStart(4, "0")}`;
    this.model.set(id, { reserved: input + output, unknown: false });
    return id;
  }
  private checkModel(input: number, final: boolean): void {
    this.checkTime();
    const output = this.config.max_output_tokens_per_request;
    if (input > this.config.service_max_input_tokens || input + output + this.config.context_safety_margin_tokens > this.config.model_context_window) throw new Error("context_window_exceeded");
    const finalReserve = final ? 0 : this.finalReserve(input);
    if (this.model.size >= this.config.max_model_requests || (!final && this.model.size >= this.config.max_model_requests - 1) || this.tokens() + input + output + finalReserve > this.config.task_token_budget) throw new Error("budget_exhausted");
  }
  canReserveModel(input: number, final: boolean): boolean {
    try { this.checkModel(input, final); return true; } catch { return false; }
  }
  private tokens(): number { return [...this.model.values()].reduce((n, r) => n + (r.total ?? r.reserved), 0); }
  private finalReserve(input: number): number {
    // 保留当前终稿输入及下一次工具最大载荷带来的保守增量，受服务窗口上限约束。
    return Math.min(this.config.service_max_input_tokens, input + 2 * this.config.search_return_max_chars) + this.config.max_output_tokens_per_request;
  }
  settle(id: string, total?: number): void {
    const request = this.model.get(id);
    if (!request || request.total !== undefined) return;
    if (total === undefined) request.unknown = true;
    else {
      if (!Number.isFinite(total) || total < 0) throw new Error("INVALID_USAGE");
      request.total = total; request.unknown = false;
    }
  }
  chargeTool(id: string, name: string): void {
    if (this.tools.has(id)) return;
    this.checkTime();
    if (this.tools.size >= this.config.max_tool_actions || (name === "search_cases" && this.searches >= this.config.max_search_calls)) {
      this.rejected.add(id); throw new Error("budget_exhausted");
    }
    this.tools.set(id, name); if (name === "search_cases") this.searches++;
  }
  toolAdmitted(id: string): boolean { return this.tools.has(id) && !this.rejected.has(id); }
  recordQueries(count: number, external = 0): void {
    if (!Number.isInteger(count) || count < 0 || count > this.remainingQueries()) throw new Error("QUERY_BUDGET_VIOLATION");
    this.queryAttempts += count; this.externalRequests += external;
    if (this.externalRequests > this.config.max_external_retrieval_requests) throw new Error("EXTERNAL_RETRIEVAL_BUDGET_VIOLATION");
  }
  remainingQueries(): number { return this.config.max_search_query_attempts - this.queryAttempts; }
  recordPhysicalRequest(): void {
    if (this.config.execution_mode !== "live" || this.physicalRequests >= this.model.size) throw new Error("HIDDEN_PROVIDER_RETRY_BLOCKED");
    this.physicalRequests++;
  }
  summary(): Record<string, number | string> {
    const rows = [...this.model.values()];
    return {
      model_request_units: rows.length, physical_provider_requests: this.physicalRequests,
      settled_tokens: rows.reduce((n, r) => n + (r.total ?? 0), 0),
      reserved_tokens: rows.reduce((n, r) => n + (r.total === undefined ? r.reserved : 0), 0),
      unknown_usage_requests: rows.filter(r => r.unknown).length, tool_actions: this.tools.size,
      rejected_tool_actions: this.rejected.size, search_calls: this.searches, search_query_attempts: this.queryAttempts,
      external_retrieval_requests: this.externalRequests, usage_kind: this.config.execution_mode === "mock" ? "simulated" : "provider_reported",
      elapsed_ms: Math.round(this.clock() - this.started),
    };
  }
  shouldFinalize(input: number): boolean {
    return this.model.size >= this.config.max_model_requests - 1 || this.tokens() + input + this.config.max_output_tokens_per_request + this.finalReserve(input) > this.config.task_token_budget;
  }
  remainingMs(): number { return Math.max(0, this.config.task_deadline_seconds * 1000 - (this.clock() - this.started)); }
}
