import type { Budget } from "./budget.js";
import { validateProgram, type Config, type JsonObject, type TaskInput, type ToolEnvelope } from "./contracts.js";
import type { TaskState } from "./state.js";
import type { Trace } from "./trace.js";
import type { ToolBridge } from "./tool_bridge.js";

export interface ToolExecutorOptions {
  task: TaskInput;
  config: Config;
  state: TaskState;
  budget: Budget;
  trace: Trace;
  bridge: ToolBridge;
  auditEnabled: boolean;
  /** Legacy audit enables the circuit by default; another runner can enable it explicitly. */
  retrievalCircuitEnabled?: boolean;
  emitProgress: (action: string) => void;
  requestFinal: () => void;
  assertActive?: () => void;
  groupNoProgressEnabled?: boolean;
  availableTimeMs?: () => number;
  beforeRetry?: (name: string) => void;
  cancellationTimeoutMs?: () => number;
  beforeAccept?: (name: string, params: JsonObject, response: ToolEnvelope) => void | Promise<void>;
  afterRead?: (params: JsonObject, response: ToolEnvelope) => void | Promise<void>;
}

/** Uses the caller's state and budget. The caller charges the initial action; only retries charge here. */
export function createToolExecutor(options: ToolExecutorOptions) {
  const { task, config, state, budget, trace, bridge, auditEnabled, retrievalCircuitEnabled = auditEnabled, emitProgress, requestFinal, afterRead } = options;
  const bounded = async <T>(promise: Promise<T>, ms: number): Promise<T> => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    try { return await Promise.race([promise, new Promise<never>((_, reject) => { timer = setTimeout(() => reject(new Error("TOOL_TIMEOUT")), Math.max(1, ms)); })]); }
    finally { clearTimeout(timer); }
  };
  let retrievalAccounting: JsonObject | undefined;
  let retrievalCircuitOpen = false;
  const observeRetrievalHealth = (meta: JsonObject) => {
    if (!retrievalCircuitEnabled || meta.transport_circuit_open !== true) return;
    const newlyOpen = !retrievalCircuitOpen;
    retrievalCircuitOpen = true; state.stopReason = "tool_failure";
    retrievalAccounting = { ...retrievalAccounting, transport_circuit_open: true, transport_failure_streak: meta.transport_failure_streak, transport_diagnostic: meta.transport_diagnostic ?? null };
    if (newlyOpen) trace.event("retrieval.circuit_open", retrievalAccounting);
  };
  const cancelBackend = async () => {
    if (config.retrieval_profile !== 'v2-hybrid-flash-v1' || !bridge.cancelRetrieval) return;
    try {
      retrievalAccounting = { ...retrievalAccounting, ...await (options.cancellationTimeoutMs ? bounded(bridge.cancelRetrieval(Math.max(1, options.cancellationTimeoutMs())), Math.max(1, options.cancellationTimeoutMs())) : bridge.cancelRetrieval()) };
      let mayUpdateHealth = true;
      try { options.assertActive?.(); } catch { mayUpdateHealth = false; }
      if (mayUpdateHealth) observeRetrievalHealth(retrievalAccounting);
      if (retrievalCircuitOpen) retrievalAccounting.transport_circuit_open = true;
      const count = retrievalAccounting.external_retrieval_requests;
      if (Number.isSafeInteger(count) && count >= budget.externalRequests) budget.recordQueries(0, count - budget.externalRequests);
      trace.event('retrieval.cancelled', retrievalAccounting);
    } catch (error) {
      retrievalAccounting = { ...retrievalAccounting, usage_known: false, cancellation_confirmed: false };
      trace.event('retrieval.cancel_failed', { reason: String((error as Error).message).split(':', 1)[0] });
    }
  };
  const execute = async (name: string, id: string, params: JsonObject) => {
    options.assertActive?.();
    if ((options.availableTimeMs?.() ?? budget.remainingMs()) <= 0) throw new Error("LOOP_REASSESSMENT_TIME_RESERVED");
    if (auditEnabled) {
      const origin = id.startsWith("evaluation:") ? "shared_first_prefetch" : id.startsWith("final:") ? "final_read_guard" : id.startsWith("fixed:") ? "fixed_pipeline" : "model_native_call";
      trace.event(origin === "model_native_call" ? "evaluation.model_tool" : "evaluation.program_tool", { name, id, params, origin });
    }
    emitProgress(name);
    let attempts = 0;
    const timeout = () => Math.min(budget.remainingMs(), config.tool_request_timeout_seconds * 1000, options.availableTimeMs?.() ?? Infinity);
    const closedQueries: JsonObject[] = name === "search_cases" && options.groupNoProgressEnabled !== false ? params.queries.filter((q: JsonObject) => (state.noProgress.get(state.group(q)) ?? 0) >= config.no_progress_limit) : [];
    const activeParams = closedQueries.length ? { ...params, queries: params.queries.filter((q: JsonObject) => !closedQueries.includes(q)) } : params;
    const closedRows = closedQueries.map((q, index) => {
      const prior = [...state.searches].reverse().find(previous => state.queryKey(previous) === state.queryKey(q));
      const done = Boolean(prior?.complete && ["ok", "empty"].includes(prior.status));
      return { ...(done ? prior : {}), ...q, query_id: `BLOCKED${index + 1}`, status: done ? prior!.status : "error", complete: done, candidates: [], no_progress_blocked: true, prior_completed_query: done, error: done ? null : { code: "GROUP_NO_PROGRESS" } };
    });
    while (true) {
      const limit = budget.remainingQueries();
      let accountedQueries = false;
      try {
        const response = name === "search_cases" && !activeParams.queries.length ? { schema_version: "finrisk-tools/1.0", status: "empty", data: { queries: [] }, meta: { snapshot_id: task.snapshot.snapshot_id, backend_query_attempts: 0, cache_hits: 0, external_retrieval_requests: 0 }, error: null } as any : await bounded(bridge.call(name, activeParams, limit, timeout()), timeout());
        options.assertActive?.();
        try {
          validateProgram(response, "tool-envelope");
          if (!["finrisk-tools/1.0", "finrisk-tools/1.1"].includes(response.schema_version) || response.meta.snapshot_id !== task.snapshot.snapshot_id) throw new Error("TOOL_SNAPSHOT_OR_SCHEMA_MISMATCH");
          if (name === "search_cases" && !Array.isArray(response.data?.queries) && response.status !== "error") throw new Error("TOOL_SEARCH_RESULT_MISSING");
        } catch (error) { state.protocolInvalid = true; state.validationErrors.push(String((error as Error).message)); throw error; }
        accountedQueries = true;
        budget.recordQueries(response.meta.backend_query_attempts, response.meta.external_retrieval_requests);
        if (name === "search_cases") observeRetrievalHealth(response.meta);
        if (name === "search_cases" && closedRows.length) {
          response.data ??= { queries: [] }; response.data.queries ??= [];
          response.data.queries = [...closedRows, ...response.data.queries];
          response.meta.no_progress_blocked_queries = closedRows.length;
          if (closedRows.some(row => !row.complete)) response.status = "partial";
        }
        if (name === "search_cases" && !response.data?.queries) response.data = { queries: activeParams.queries.map((q: JsonObject, index: number) => ({ ...q, query_id: `LOCAL:${id}:${index + 1}`, status: "error", complete: false, candidates: [], error: { code: response.error?.code ?? "TOOL_SEARCH_FAILED" } })) };
        await options.beforeAccept?.(name, params, response); options.assertActive?.();
        try { state.acceptTool(name, params, response, id); }
        catch (error) { state.protocolInvalid = true; state.validationErrors.push(String((error as Error).message)); throw error; }
        if (name === "read_source") await afterRead?.(params, response);
        options.assertActive?.();
        trace.event("tool.result", { id, name, params, response });
        emitProgress("condition_comparison");
        return { content: [{ type: "text" as const, text: JSON.stringify(response) }], details: response, isError: response.status === "error", terminate: retrievalCircuitOpen || budget.remainingMs() <= 0 };
      } catch (error) {
        const reason = String((error as Error).message);
        if (name === 'search_cases') await cancelBackend();
        // 传输结果未知时保守保留可能已发出的query额度，不填0。
        if (name === "search_cases" && !accountedQueries) budget.recordQueries(Math.min(activeParams.queries.length, limit));
        options.assertActive?.();
        trace.event("tool.failure", { id, name, reason, query_usage_unknown: name === "search_cases" });
        const retryable = /TOOL_TRANSPORT_HTTP_50[234]|fetch failed|TOOL_TIMEOUT|TimeoutError/.test(reason);
        if (!retrievalCircuitOpen && retryable && attempts < config.max_tool_transport_retries && budget.remainingMs() > 100 && (options.availableTimeMs?.() ?? Infinity) > 100 && (name !== "search_cases" || budget.remainingQueries() >= activeParams.queries.length)) {
          try { options.beforeRetry?.(name); budget.chargeTool(`${id}:retry:${attempts + 1}`, name); attempts++; continue; }
          catch { state.stopReason = "budget_exhausted"; requestFinal(); }
        }
        const response = { schema_version: "finrisk-tools/1.0", status: "error", data: null, meta: { snapshot_id: task.snapshot.snapshot_id }, error: { code: reason, retryable: false } };
        if (name === "search_cases") state.acceptTool(name, params, { ...response, data: { queries: [...closedRows, ...activeParams.queries.map((q: JsonObject, index: number) => ({ ...q, query_id: `LOCAL:${id}:${index + 1}`, status: "error", complete: false, candidates: [], error: { code: reason } }))] } } as any);
        state.stopReason = budget.remainingMs() <= 0 ? "deadline_exceeded" : state.stopReason === "budget_exhausted" ? "budget_exhausted" : "tool_failure";
        requestFinal();
        emitProgress("condition_comparison");
        return { content: [{ type: "text" as const, text: JSON.stringify(response) }], details: response, isError: true, terminate: budget.remainingMs() <= 0 };
      }
    }
  };
  return {
    execute, cancelBackend,
    isRetrievalCircuitOpen: () => retrievalCircuitOpen,
    getRetrievalAccounting: () => retrievalAccounting,
  };
}