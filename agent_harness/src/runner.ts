import { readFileSync, rmSync } from "node:fs";
import { Agent, type AgentTool } from "@earendil-works/pi-agent-core";
import { getCurrentTools, getToolStateChanges, toToolDeclaration } from "@earendil-works/pi-ai";
import { Type, type TSchema } from "typebox";
import { Budget } from "./budget.js";
import { canonical, contractForKind, defaults, hash, root, toolSchemasForProfile, validateOutput, validateProgram, validateToolArguments, type JsonObject, type TaskInput } from "./contracts.js";
import { buildContext, buildCountedContext, estimateTokens } from "./context_builder.js";
import { countPayload, buildProviderPayload, type PromptCounter } from "./prompt_counter.js";
import { activeTrialBudget } from './trial_budget.js';
import { admissionPolicyHash, assessRequest } from './request_admission.js';
import { TaskState } from "./state.js";
import { Trace } from "./trace.js";
import { mockGateway, mockModel, type ScriptStep } from "./model_gateway.js";
import { mockPlan } from "./mock_plan.js";
import { deepseekGateway, loadDeepSeekKey } from "./deepseek_gateway.js";
import type { ToolBridge } from "./tool_bridge.js";
import { assembleReport } from "./result_validator.js";
import { candidateRead, candidateBundleRead, normalizeModelOutput, stageProtocol, validateFinalize, wireSchemaHash } from "./model_protocol.js";
import { checkCriteriaBinding, criteriaByteHash } from "./criteria.js";
import { mechanismQueries, sourceReadPlan } from "./retrieval_plan.js";
import { modelBudget, progressSnapshot } from "./progress.js";
import { classifyReviewReply } from "./review_reply.js";
import { acceptEvaluation, evaluationInputAlias, evaluationProtocol, evaluationProtocolHash, evaluationQuery, evaluationRoute, evaluationSystemPrompt } from "./evaluation.js";
import { parseModelJson, outputDiagnostic, evaluationFormatRepairable } from "./model_json.js";
import { onlyPartialRerankGaps, retrievalAssessment } from "./retrieval_quality.js";

export async function runTask(task: TaskInput, bridge: ToolBridge, script?: ScriptStep[]): Promise<{ report: JsonObject; events: JsonObject[] }> {
  const config = { ...defaults, ...task.config };
  const evaluationEnabled = Boolean(config.eval_mode || config.evaluation_arm);
  const evaluationCitations = evaluationEnabled && config.retrieval_profile === "v2-hybrid-flash-v1";
  if (config.retrieval_reliability !== undefined && typeof config.retrieval_reliability !== "boolean") throw new Error("INVALID_RETRIEVAL_RELIABILITY_CONFIG");
  if (config.retrieval_reliability && config.retrieval_profile !== "v2-hybrid-flash-v1") throw new Error("RETRIEVAL_RELIABILITY_REQUIRES_V2");
  const formatRepairLimit = config.evaluation_format_repair_limit === undefined ? 0 : config.evaluation_format_repair_limit;
  if (formatRepairLimit !== 0 && formatRepairLimit !== 1) throw new Error("INVALID_EVALUATION_FORMAT_REPAIR_LIMIT");
  if (formatRepairLimit && (config.eval_mode === "one_shot_rag" || config.evaluation_arm === "R1")) throw new Error("ONE_SHOT_FORMAT_REPAIR_FORBIDDEN");
  if (formatRepairLimit && config.eval_mode !== "adaptive_rag") throw new Error("EVALUATION_FORMAT_REPAIR_REQUIRES_ADAPTIVE");
  if (config.evaluation_arm && !["R1", "R2", "A"].includes(config.evaluation_arm)) throw new Error("INVALID_EVALUATION_ARM");
  if (config.eval_mode && !["one_shot_rag", "adaptive_rag"].includes(config.eval_mode)) throw new Error("INVALID_EVALUATION_MODE");
  if (evaluationEnabled && process.env.FINRISK_AGENT_ALLOW_EVALUATION !== "1") throw new Error("EVALUATION_NOT_ADMITTED");
  if (config.eval_mode && config.engine !== "controlled_agent") throw new Error("INVALID_EVALUATION_ENGINE");
  const firstQuery = config.eval_mode ? evaluationQuery(task) : undefined;
  if (config.execution_mode !== "mock" && process.env.FINRISK_AGENT_ALLOW_LIVE !== "1") throw new Error("LIVE_PROFILE_NOT_ADMITTED:需要显式开启真实调用");
  if (config.execution_mode === "live" && !["live-deepseek-flash-bm25-v1", "live-deepseek-flash-v2-v1"].includes(config.profile_id)) throw new Error("LIVE_PROFILE_NOT_ADMITTED:模型profile不匹配");
  if (config.profile_id === 'live-deepseek-flash-v2-v1' && !process.env.FINRISK_AGENT_TRIAL_LEDGER) throw new Error('MODEL_TRIAL_LEDGER_REQUIRED');
  if (!["bm25-source-v1", "v2-hybrid-flash-v1"].includes(config.retrieval_profile)) throw new Error("PROFILE_NOT_READY");
  if (task.snapshot.retrieval_profile && task.snapshot.retrieval_profile !== config.retrieval_profile) throw new Error("TOOL_SNAPSHOT_OR_SCHEMA_MISMATCH");
  checkCriteriaBinding(task.definitions, task.snapshot.dictionary_sha256);
  if (!Number.isInteger(config.max_queries_per_search) || config.max_queries_per_search < 1 || config.max_queries_per_search > 3) throw new Error("INVALID_BUDGET_CONFIG:max_queries_per_search");
  const toolSchemas = toolSchemasForProfile(config.retrieval_profile);
  if (config.max_queries_per_search > toolSchemas.search_cases.properties.queries.maxItems) throw new Error('TOOL_CONTRACT_CONFIG_MISMATCH');
  toolSchemas.search_cases.properties.queries.maxItems = config.max_queries_per_search;
  const validateArgs = (name: string, args: unknown) => validateToolArguments(name, args, toolSchemas);
  const systemPrompt = evaluationCitations ? evaluationSystemPrompt : readFileSync(`${root}/configs/agent_rag/system-prompt.txt`, "utf8");
  const expected = task.execution_contract;
  if (evaluationEnabled && (expected?.evaluation_protocol_sha256 && expected.evaluation_protocol_sha256 !== evaluationProtocolHash || config.execution_mode === "live" && !expected?.evaluation_protocol_sha256)) throw new Error("EXECUTION_CONTRACT_MISMATCH:evaluation_protocol");
  if (config.execution_mode === "live" && !expected) throw new Error("EXECUTION_CONTRACT_BINDING_REQUIRED");
  if (expected && (expected.model_wire_sha256 !== wireSchemaHash || expected.criteria_sha256 !== criteriaByteHash || expected.prompt_sha256 !== hash(systemPrompt))) throw new Error("EXECUTION_CONTRACT_MISMATCH");
  if ((expected?.tool_contract_sha256 && expected.tool_contract_sha256 !== hash(canonical(toolSchemas)))
    || (config.execution_mode === 'live' && config.retrieval_profile === 'v2-hybrid-flash-v1' && !expected?.tool_contract_sha256)) throw new Error('EXECUTION_CONTRACT_MISMATCH:tool_contract');
  if ((expected?.request_admission_policy_sha256 && expected.request_admission_policy_sha256 !== admissionPolicyHash(config.request_admission_policy))
    || (config.execution_mode === 'live' && config.retrieval_profile === 'v2-hybrid-flash-v1' && !expected?.request_admission_policy_sha256)) throw new Error('EXECUTION_CONTRACT_MISMATCH:request_admission_policy');
  if (hash(task.input.text) !== task.input.text_sha256 || hash(canonical({ text: task.input.text, user_context: task.input.user_context })) !== task.input.input_sha256) throw new Error("INPUT_HASH_MISMATCH");
  const chars = Array.from(task.input.text);
  let offset = 0;
  for (const segment of task.input.segments) {
    if (segment.start !== offset || chars.slice(segment.start, segment.end).join("") !== segment.text || segment.source_id !== task.input.source_id) throw new Error("INPUT_SEGMENT_MISMATCH");
    offset = segment.end;
  }
  if (offset !== chars.length) throw new Error("INPUT_SEGMENT_COVERAGE_MISMATCH");
  if (evaluationCitations && (typeof task.snapshot.snapshot_id !== "string" || !task.snapshot.snapshot_id.trim())) throw new Error("CITATION_SNAPSHOT_REQUIRED");
  if (evaluationCitations && config.execution_mode === "live") {
    const trial = activeTrialBudget(config.request_admission_policy);
    if (!trial) throw new Error("MODEL_TRIAL_LEDGER_REQUIRED");
    if (config.retrieval_reliability !== true) throw new Error("RETRIEVAL_RELIABILITY_BINDING_REQUIRED");
    trial.initializeEmpty();
  }
  const budget = new Budget(config);
  const state = new TaskState(task.input, config.no_progress_limit);
  state.evaluationCitations = evaluationCitations;
  state.retrievalReliability = config.retrieval_reliability === true;
  if (state.retrievalReliability) {
    const protocol = task.snapshot.rerank_protocol ?? task.snapshot.backend?.rerank_protocol ?? (config.execution_mode === "live" ? "quote-copy-v1" : undefined);
    if (protocol !== undefined && !["quote-copy-v1", "span-ref-v2"].includes(protocol)) throw new Error("RERANK_PROTOCOL_NOT_READY");
    state.expectedRerankProtocol = protocol;
  }
  if (evaluationCitations) state.citationSnapshotId = task.snapshot.snapshot_id;
  let evaluationFirstPending = Boolean(config.eval_mode);
  let evaluationReport: JsonObject | undefined;
  if (evaluationFirstPending) state.protocolVersion = "finrisk-model-wire/1.2";
  const trace = new Trace(task.task_dir);
  let currentAction = "material_analysis";
  const emitProgress = (action = currentAction, report?: JsonObject) => {
    currentAction = action;
    const snapshot = progressSnapshot(state, budget, action, trace.events.length + 1, report);
    validateProgram(snapshot, "progress");
    trace.event("progress.snapshot", snapshot); trace.save("progress.json", snapshot);
  };
  trace.onEvent = event => { if (["model.request", "model.settled", "model.failure"].includes(event.type)) emitProgress(); };
  const promptCounter: PromptCounter | undefined = bridge.countTokens ? (payload, timeoutMs) => bridge.countTokens!(payload, Math.max(1, Math.min(timeoutMs ?? config.tool_request_timeout_seconds * 1000, budget.remainingMs()))) : undefined;
  const usePayloadCount = config.execution_mode === "live" || Boolean(promptCounter);
  const gateway = config.execution_mode === "mock" ? { streamFn: mockGateway(budget, state, trace, script ?? mockPlan({ ...task, config }), promptCounter), model: mockModel } : deepseekGateway(budget, state, trace, loadDeepSeekKey(), fetch, promptCounter);
  trace.event("task.started", { task_id: task.input.task_id, snapshot: task.snapshot, config });
  emitProgress();
  let repairInstruction = "";
  let pendingEvaluationRepair: { errorIndex: number; message: string; diagnostic: JsonObject; attempt: number } | undefined;
  let toolFormatInstruction = "";
  let forceFinal = false;
  let admissionStopped = false;
  let finalReadsChecked = false;
  const blocked = new Set<string>();
  const actionKeys = new Map<string, string>();
  const invalidNativeIds = new Set<string>();
  const invalidArgumentErrors = new Map<string, string>();
  const seenMessages = new WeakSet<object>();
  let actionOrdinal = 0;
  let agent: Agent;
  const readPreviews = new Map<string, JsonObject>();
  const projectContext = async (messages: any[], target: TaskState, trailing: any[] = []) => {
    if (usePayloadCount) return buildCountedContext(messages, target, config, trailing, async projected => {
      const payload = buildProviderPayload({ messages: projected } as any, target.phase, target.repairing, config.max_output_tokens_per_request);
      const count = await countPayload(payload, promptCounter, Math.min(budget.remainingMs(), config.tool_request_timeout_seconds * 1000));
      const body = JSON.stringify(payload);
      const trial = config.execution_mode === 'live' ? activeTrialBudget(config.request_admission_policy) : undefined;
      const admission = trial?.preview(body) ?? (config.request_admission_policy ? assessRequest(body, config.request_admission_policy) : { fits: true, reason: null });
      return { ...count, body_bytes: Buffer.byteLength(body, 'utf8'), admission_fits: admission.fits, admission_reason: admission.reason };
    });
    const projected = buildContext(messages, target, config, trailing);
    return { messages: projected, count: { prompt_tokens: estimateTokens(projected), counter_kind: "conservative_character_estimate" } };
  };
  const finalProjection = async (target: TaskState) => {
    target.phase = "finalizing"; target.repairing = Boolean(repairInstruction);
    const stage = { role: "system", content: `<runtime_stage>当前阶段=finalizing；${repairInstruction || "立即提交final JSON，不再检索。"}${stageProtocol(task, target, "final")}</runtime_stage>`, timestamp: Date.now() };
    return projectContext(agent.state.messages, target, [stage]);
  };
  const cloneState = () => Object.assign(new TaskState(state.input, state.noProgressLimit), structuredClone(state));
  const finalReadFits = async (params: JsonObject, alreadyCharged = false): Promise<boolean> => {
    if (state.protocolVersion !== "finrisk-model-wire/1.2") return true;
    const projectedState = cloneState();
    projectedState.runtimeBudget = modelBudget(budget);
    if (!alreadyCharged) projectedState.runtimeBudget.tool_remaining = Math.max(0, projectedState.runtimeBudget.tool_remaining - 1);
    try {
      if (!bridge.previewRead) {
        if (usePayloadCount) { trace.event("context.read_preview", { source_id: params.source_id, fits: false, reason: "SOURCE_PREVIEW_UNAVAILABLE" }); return false; }
        // Historical offline bridges retain their conservative contract until they provide preview.
        const projected = await finalProjection(projectedState);
        return projected.count.prompt_tokens + Math.min(params.max_chars, 4800) * 2 + 300 <= config.service_max_input_tokens - 800;
      }
      validateArgs("read_source", params);
      const preview = await bounded(bridge.previewRead(params, Math.min(budget.remainingMs(), config.tool_request_timeout_seconds * 1000)), Math.min(budget.remainingMs(), config.tool_request_timeout_seconds * 1000));
      validateProgram(preview, "tool-envelope");
      if (!["finrisk-tools/1.0", "finrisk-tools/1.1"].includes(preview.schema_version) || preview.meta.snapshot_id !== task.snapshot.snapshot_id || preview.status !== "ok" || !preview.data?.evidence) throw new Error(preview.error?.code ?? "SOURCE_PREVIEW_UNAVAILABLE");
      projectedState.acceptTool("read_source", params, preview);
      const projected = await finalProjection(projectedState);
      const capacity = projected.count.prompt_tokens <= config.service_max_input_tokens - config.context_safety_margin_tokens && (projected.count as JsonObject).admission_fits !== false;
      const fits = capacity && budget.canReserveModel(projected.count.prompt_tokens, true);
      trace.event("context.read_preview", { source_id: params.source_id, evidence_id: preview.data.evidence.evidence_id, ...projected.count, fits, full_window_chars: Array.from(preview.data.evidence.quote).length, before_read: true });
      if (fits) readPreviews.set(canonical(params), preview);
      return fits;
    } catch (error) { trace.event("context.read_preview", { source_id: params.source_id, fits: false, reason: String((error as Error).message).split(":", 1)[0] }); return false; }
  };
  const saveState = () => trace.save("state.json", { task_id: task.input.task_id, phase: state.phase, stop_reason: state.stopReason, points: state.points, gaps: [...state.gaps.values()], resources: budget.summary() });
  const bounded = async <T>(promise: Promise<T>, ms: number): Promise<T> => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    try { return await Promise.race([promise, new Promise<never>((_, reject) => { timer = setTimeout(() => reject(new Error("TOOL_TIMEOUT")), Math.max(1, ms)); })]); }
    finally { clearTimeout(timer); }
  };
  let retrievalAccounting: JsonObject | undefined;
  let retrievalCircuitOpen = false;
  const observeRetrievalHealth = (meta: JsonObject) => {
    if (!evaluationEnabled || meta.transport_circuit_open !== true) return;
    const newlyOpen = !retrievalCircuitOpen;
    retrievalCircuitOpen = true; state.stopReason = "tool_failure";
    retrievalAccounting = { ...retrievalAccounting, transport_circuit_open: true, transport_failure_streak: meta.transport_failure_streak, transport_diagnostic: meta.transport_diagnostic ?? null };
    if (newlyOpen) trace.event("retrieval.circuit_open", retrievalAccounting);
  };
  const cancelBackend = async () => {
    if (config.retrieval_profile !== 'v2-hybrid-flash-v1' || !bridge.cancelRetrieval) return;
    try {
      retrievalAccounting = { ...retrievalAccounting, ...await bridge.cancelRetrieval() };
      observeRetrievalHealth(retrievalAccounting);
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
    if (evaluationEnabled) {
      const origin = id.startsWith("evaluation:") ? "shared_first_prefetch" : id.startsWith("final:") ? "final_read_guard" : id.startsWith("fixed:") ? "fixed_pipeline" : "model_native_call";
      trace.event(origin === "model_native_call" ? "evaluation.model_tool" : "evaluation.program_tool", { name, id, params, origin });
    }
    emitProgress(name);
    let attempts = 0;
    const timeout = () => Math.min(budget.remainingMs(), config.tool_request_timeout_seconds * 1000);
    const closedQueries: JsonObject[] = name === "search_cases" ? params.queries.filter((q: JsonObject) => (state.noProgress.get(state.group(q)) ?? 0) >= config.no_progress_limit) : [];
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
        try { state.acceptTool(name, params, response, id); }
        catch (error) { state.protocolInvalid = true; state.validationErrors.push(String((error as Error).message)); throw error; }
        if (name === "read_source" && readPreviews.has(canonical(params))) {
          const preview = readPreviews.get(canonical(params))!; readPreviews.delete(canonical(params));
          const sameWindow = canonical(response.data) === canonical(preview.data);
          const actualState = cloneState(); actualState.runtimeBudget = modelBudget(budget);
          const actual = await finalProjection(actualState);
          const fits = actual.count.prompt_tokens <= config.service_max_input_tokens - config.context_safety_margin_tokens && (actual.count as JsonObject).admission_fits !== false && budget.canReserveModel(actual.count.prompt_tokens, true);
          trace.event("context.read_verified", { source_id: params.source_id, same_window: sameWindow, ...actual.count, fits, after_read: true });
          if (!sameWindow || !fits) { state.stopReason = "context_capacity_reached"; forceFinal = true; }
          if (!sameWindow) { state.protocolInvalid = true; state.validationErrors.push("SOURCE_PREVIEW_WINDOW_MISMATCH"); }
        }
        trace.event("tool.result", { id, name, params, response });
        emitProgress("condition_comparison");
        return { content: [{ type: "text" as const, text: JSON.stringify(response) }], details: response, isError: response.status === "error", terminate: retrievalCircuitOpen || budget.remainingMs() <= 0 };
      } catch (error) {
        const reason = String((error as Error).message);
        if (name === 'search_cases') await cancelBackend();
        // 传输结果未知时保守保留可能已发出的query额度，不填0。
        if (name === "search_cases" && !accountedQueries) budget.recordQueries(Math.min(activeParams.queries.length, limit));
        trace.event("tool.failure", { id, name, reason, query_usage_unknown: name === "search_cases" });
        const retryable = /TOOL_TRANSPORT_HTTP_50[234]|fetch failed|TOOL_TIMEOUT|TimeoutError/.test(reason);
        if (!retrievalCircuitOpen && retryable && attempts < config.max_tool_transport_retries && budget.remainingMs() > 100 && (name !== "search_cases" || budget.remainingQueries() >= activeParams.queries.length)) {
          try { budget.chargeTool(`${id}:retry:${attempts + 1}`, name); attempts++; continue; }
          catch { state.stopReason = "budget_exhausted"; forceFinal = true; }
        }
        const response = { schema_version: "finrisk-tools/1.0", status: "error", data: null, meta: { snapshot_id: task.snapshot.snapshot_id }, error: { code: reason, retryable: false } };
        if (name === "search_cases") state.acceptTool(name, params, { ...response, data: { queries: [...closedRows, ...activeParams.queries.map((q: JsonObject, index: number) => ({ ...q, query_id: `LOCAL:${id}:${index + 1}`, status: "error", complete: false, candidates: [], error: { code: reason } }))] } } as any);
        state.stopReason = budget.remainingMs() <= 0 ? "deadline_exceeded" : state.stopReason === "budget_exhausted" ? "budget_exhausted" : "tool_failure";
        forceFinal = true;
        emitProgress("condition_comparison");
        return { content: [{ type: "text" as const, text: JSON.stringify(response) }], details: response, isError: true, terminate: budget.remainingMs() <= 0 };
      }
    }
  };
  const tools: AgentTool[] = Object.keys(toolSchemas).map(name => ({ name, label: name, description: ({ search_cases: "查询冻结处罚原文，返回真实来源、排名和引文，不输出违规判断。", read_source: "读取本任务输入或注册处罚来源的限定上下文。", get_risk_definition: "读取官方风险名称与定义；不接受自创同义词。" } as Record<string, string>)[name]!, parameters: Type.Unsafe(toolSchemas[name] as TSchema), executionMode: "sequential" as const, replay: "safe" as const, execute: (id: string, params: unknown) => execute(name, actionKeys.get(id) ?? id, params as JsonObject) }));
  const sourceReadSchedule = (queries: JsonObject[], phase: string) => {
    const dynamic = queries.some(q => (q.candidates ?? []).some((c: JsonObject) => c.required_evidence_ids !== undefined));
    // V1 保留一次性多样性排序；V2 每次按真实窗口覆盖重算下一个锚点。
    const planned = sourceReadPlan(queries, dynamic ? 1 : 3, e => candidateRead(e, state));
    const attempted = new Set<string>(); const skipped = new Set<string>();
    const key = (e: JsonObject) => canonical([e.source_id, e.evidence_id]);
    let logicalReads = 0;
    return {
      planned,
      next: (index: number) => dynamic ? sourceReadPlan(queries, 1, e => {
        const covered = candidateRead(e, state);
        if (covered && !skipped.has(key(e))) {
          skipped.add(key(e));
          trace.event("retrieval.read_skipped", { phase, source_id: e.source_id, anchor_evidence_id: e.evidence_id, reason: "already_covered" });
        }
        return covered || attempted.has(key(e));
      })[0] : planned[index],
      attempt: (candidate: JsonObject) => {
        attempted.add(key(candidate.evidence[0])); logicalReads++;
        trace.event("retrieval.read_selected", { phase, source_id: candidate.source_id, anchor_evidence_id: candidate.evidence[0].evidence_id, logical_read: logicalReads, max_logical_reads: 3 });
      },
      end: (reason: string) => trace.event("retrieval.read_phase_ended", { phase, logical_reads: logicalReads, reason }),
    };
  };
  const fixedRetrieval = async () => {
    if (state.protocolVersion === "finrisk-model-wire/1.2" && !state.points.length) return;
    const points = state.points.length ? state.points : task.input.segments.filter(s => s.text.trim()).map(s => ({ input_refs: [{ segment_id: s.segment_id, quote: Array.from(s.text).slice(0, 500).join(""), occurrence: 1 }], mechanism: Array.from(s.text).slice(0, 250).join(""), risk_point_id: undefined }));
    const queries = mechanismQueries(points);
    const batchSize = Math.min(config.max_queries_per_search, 3);
    for (let offset = 0; offset < queries.length && budget.remainingQueries() > 0; offset += batchSize) {
      const batch = queries.slice(offset, offset + Math.min(batchSize, budget.remainingQueries()));
      try { budget.chargeTool(`fixed:search:${offset}`, "search_cases"); }
      catch { state.stopReason = "budget_exhausted"; break; }
      await execute("search_cases", `fixed:search:${offset}`, { queries: batch, top_k: 5, filters: {} });
      if (retrievalCircuitOpen) return;
    }
    const schedule = sourceReadSchedule(state.searches, "fixed");
    trace.event("retrieval.read_plan", { sources: schedule.planned.map(c => c.source_id), selection: "point_coverage_then_exact_fact_text_diversity", event_deduplication_verified: false });
    let endReason = "read_limit";
    try {
      for (let index = 0; index < 3; index++) {
        if (budget.remainingMs() <= 0) { state.stopReason = "deadline_exceeded"; endReason = "deadline_exceeded"; break; }
        const candidate = schedule.next(index);
        if (!candidate) { endReason = "no_unattempted_anchors"; break; }
        const anchor = candidate.evidence[0];
        const params = { source_id: candidate.source_id, anchor_evidence_id: anchor.evidence_id, max_chars: 1000 };
        if (!await finalReadFits(params)) { endReason = "reserved_final_context"; trace.event("retrieval.read_omitted", { source_id: candidate.source_id, reason: endReason, remaining_candidates_not_assessed: true }); break; }
        try { budget.chargeTool(`fixed:read:${index}`, "read_source"); }
        catch { state.stopReason = "budget_exhausted"; endReason = "tool_budget"; break; }
        schedule.attempt(candidate);
        const read = await execute("read_source", `fixed:read:${index}`, params);
        const fetched = (read.details as JsonObject).data?.evidence;
        if (fetched && state.evidence.has(fetched.evidence_id)) state.evidence.get(fetched.evidence_id)!.presented_in_fixed_context = true;
        if (read.terminate) { endReason = retrievalCircuitOpen ? "retrieval_circuit_open" : "deadline_exceeded"; break; }
      }
    } finally { schedule.end(endReason); }
  };
  const ensureFinalReads = async () => {
    if (finalReadsChecked || config.engine !== "controlled_agent" || state.protocolVersion !== "finrisk-model-wire/1.2") return;
    finalReadsChecked = true;
    const unread = state.searches.filter(q => q.candidates?.length && q.candidates.every((c: JsonObject) => !candidateBundleRead(c, state)));
    const schedule = sourceReadSchedule(unread, "final");
    let endReason = "read_limit";
    try {
      for (let index = 0; index < 3; index++) {
        if (budget.remainingMs() <= 0) { state.stopReason = "deadline_exceeded"; endReason = "deadline_exceeded"; break; }
        const candidate = schedule.next(index);
        if (!candidate) { endReason = "no_unattempted_anchors"; break; }
        const params = { source_id: candidate.source_id, anchor_evidence_id: candidate.evidence[0].evidence_id, max_chars: 1000 };
        if (!await finalReadFits(params)) { state.stopReason = "context_capacity_reached"; endReason = "reserved_final_context"; trace.event("retrieval.read_omitted", { source_id: candidate.source_id, reason: endReason, remaining_candidates_not_assessed: true }); break; }
        const id = `final:read:${index}`;
        try { budget.chargeTool(id, "read_source"); }
        catch { state.stopReason = "budget_exhausted"; endReason = "tool_budget"; trace.event("retrieval.read_omitted", { source_id: candidate.source_id, reason: endReason, remaining_candidates_not_assessed: true }); break; }
        schedule.attempt(candidate);
        trace.event("retrieval.final_read_guard", { source_id: candidate.source_id, anchor_evidence_id: params.anchor_evidence_id });
        const read = await execute("read_source", id, params);
        if (read.isError && !["budget_exhausted", "deadline_exceeded", "context_capacity_reached"].includes(state.stopReason)) state.stopReason = "tool_failure";
        if (read.terminate) { endReason = retrievalCircuitOpen ? "retrieval_circuit_open" : "deadline_exceeded"; break; }
      }
    } finally { schedule.end(endReason); }
    if (state.stopReason === "evidence_sufficient" && unread.some(q => q.candidates.every((c: JsonObject) => !candidateBundleRead(c, state)))) state.stopReason = "case_read_incomplete";
  };
  const invalid = (error: unknown, text = "") => {
    const message = String((error as Error).message);
    const errorIndex = state.validationErrors.push(message) - 1;
    const diagnostic = outputDiagnostic(error, text);
    trace.event("output.rejected", { phase: state.phase, error: message, ...diagnostic });
    if (evaluationFirstPending) {
      if (evaluationFormatRepairable(error) && state.repairs < formatRepairLimit && state.repairs + state.toolFormatCorrections < config.max_json_repairs_per_task && Number(budget.summary().model_request_units) < config.max_model_requests - 1 && budget.remainingMs() > 0) {
        state.repairs++;
        pendingEvaluationRepair = { errorIndex, message, diagnostic, attempt: state.repairs };
        repairInstruction = `上一回复格式校验失败：${diagnostic.code}；位置=${diagnostic.position ?? "未定位"}；Schema问题（仅校验器路径/关键字/字段）=${JSON.stringify(diagnostic.schema_issues ?? [])}。重新提交同一evaluation_first Schema的完整JSON；保留当前输入与已核读证据，不调用工具，不新增事实。continuation与points同级。所有条件与缺口仍须通过校验。`;
        trace.event("output.format_repair", { attempt: state.repairs, ...diagnostic, additional_retrieval: false });
        return { action: "continue" as const };
      }
      state.protocolInvalid = true; state.stopReason = "invalid_output"; return { action: "end" as const };
    }
    if (state.repairs + state.toolFormatCorrections < config.max_json_repairs_per_task && Number(budget.summary().model_request_units) < config.max_model_requests && budget.remainingMs() > 0) {
      state.repairs++; if (state.phase !== "analyzing") state.phase = "finalizing";
      repairInstruction = `上一输出不符合协议：${message}。请只修复本阶段JSON，保留已知事实，不调用新工具。`;
      return { action: "continue" as const };
    }
    state.stopReason = "invalid_output"; return { action: "end" as const };
  };
  const correctToolFormat = async (name: string) => {
    const message = `NATIVE_TOOL_CALL_REQUIRED:${name}`;
    state.validationErrors.push(message);
    trace.event("review.format_rejected", { error: message, implicit_tool_execution: false });
    if (budget.remainingMs() <= 0) { state.stopReason = "deadline_exceeded"; return { action: "end" as const }; }
    const instruction = "上一正文JSON只是工具意图，不能执行。保持reviewing，必须改用已声明的原生工具调用；不要在正文写search_cases/read_source/get_risk_definition JSON。若不再需要工具，只输出{\"kind\":\"finalize\"}。";
    state.runtimeBudget = modelBudget(budget);
    const projected = await projectContext(agent.state.messages, state, [{ role: "system", content: `<runtime_stage>当前阶段=reviewing；${instruction}${stageProtocol(task, state)}</runtime_stage>`, toolsAdded: tools.map(toToolDeclaration), timestamp: Date.now() }]);
    const finalRequired = budget.shouldFinalize(projected.count.prompt_tokens);
    if (state.repairs + state.toolFormatCorrections < config.max_json_repairs_per_task && !finalRequired) {
      state.toolFormatCorrections++; toolFormatInstruction = instruction;
      trace.event("review.format_correction", { correction: state.toolFormatCorrections, shared_repairs_used: state.repairs + state.toolFormatCorrections });
    } else {
      state.protocolInvalid = true;
      state.stopReason = finalRequired ? "budget_exhausted" : "invalid_output";
      forceFinal = true; toolFormatInstruction = "";
    }
    saveState(); return { action: "continue" as const };
  };
  agent = new Agent({
    initialState: { model: gateway.model, systemPrompt, tools: [], thinkingLevel: "off" },
    streamFn: gateway.streamFn, toolExecution: "sequential",
    prepareRequest: async ({ context }) => {
      if (retrievalCircuitOpen) throw new Error("RETRIEVAL_CIRCUIT_OPEN");
      state.runtimeBudget = modelBudget(budget);
      // This is a shared experimental prompt value; Budget still enforces actual elapsed time.
      if (evaluationFirstPending) {
        state.runtimeBudget.seconds_remaining = config.task_deadline_seconds;
        // Equal first-turn display for a cold/cache prefetch; never raise the
        // actual remaining allowance or change Budget accounting/admission.
        state.runtimeBudget.query_remaining = Math.min(budget.remainingQueries(), Math.max(0, config.max_search_query_attempts - (firstQuery?.queries.length ?? 0)));
      }
      state.repairing = Boolean(repairInstruction);
      const assembleStage = async () => {
        if (evaluationFirstPending) {
          const stage = { role: "system", content: `<runtime_stage>${repairInstruction ? `${repairInstruction}\n` : ""}${evaluationProtocol(task)}</runtime_stage>`, timestamp: Date.now() };
          return { ...await projectContext(context.messages, state, [stage]), available: [] as AgentTool[] };
        }
        const base = buildContext(context.messages, state, config, [], { pruneByCharacters: !usePayloadCount });
        const available = state.phase === "reviewing" && !repairInstruction && config.engine === "controlled_agent" ? tools : [];
        const changes = getToolStateChanges(getCurrentTools(base), available.map(toToolDeclaration));
        const kind = state.phase === "analyzing" ? "analysis" : state.phase === "finalizing" || repairInstruction ? "final" : undefined;
        const stage = { role: "system", content: `<runtime_stage>当前阶段=${state.phase}；${repairInstruction || (state.phase === "reviewing" ? toolFormatInstruction : "") || (state.phase === "analyzing" ? "只提交analysis JSON，无工具。" : state.phase === "finalizing" ? "立即提交final JSON，不再检索。" : "选择必要原生工具，或提交finalize信号进入正式终稿阶段。")}${stageProtocol(task, state, kind)}</runtime_stage>`, ...(changes.toolsAdded.length ? { toolsAdded: changes.toolsAdded } : {}), ...(changes.toolsRemoved.length ? { toolsRemoved: changes.toolsRemoved } : {}), timestamp: Date.now() };
        return { ...await projectContext(context.messages, state, [stage]), available };
      };
      if (state.phase !== "analyzing" && forceFinal) state.phase = "finalizing";
      let prepared = await assembleStage();
      if (state.phase === 'reviewing' && (prepared.count as JsonObject).admission_fits === false) {
        state.stopReason = (prepared.count as JsonObject).admission_reason === 'TRIAL_CNY_BUDGET_EXHAUSTED' ? 'budget_exhausted' : 'context_capacity_reached';
        state.phase = 'finalizing';
      }
      if (state.phase === "reviewing" && budget.shouldFinalize(prepared.count.prompt_tokens)) state.phase = "finalizing";
      if (state.phase === "finalizing") await ensureFinalReads();
      state.runtimeBudget = modelBudget(budget);
      if (state.phase === "finalizing") prepared = await assembleStage();
      emitProgress(state.phase === "analyzing" ? "material_analysis" : state.phase === "finalizing" ? "final_generation" : "condition_comparison");
      const { messages, available } = prepared;
      if ((prepared.count as JsonObject).admission_fits === false) {
        admissionStopped = true;
        state.stopReason = (prepared.count as JsonObject).admission_reason === 'TRIAL_CNY_BUDGET_EXHAUSTED' ? 'budget_exhausted' : 'context_capacity_reached';
        trace.event('context.admission_stopped', { phase: state.phase, ...prepared.count, physical_request: false });
        saveState();
        throw new Error('REQUEST_ADMISSION_STOP');
      }
      agent.state.tools = available;
      // Pi的自动声明早于prepareRequest；必须在本次真实transcript同步声明，不能只改执行集合。
      trace.message({ role: "request_context", phase: state.phase, request_index: Number(budget.summary().model_request_units) + 1, messages });
      trace.event("context.prepared", { phase: state.phase, estimated_tokens: prepared.count.prompt_tokens, ...prepared.count, message_count: messages.length, tools: available.map(t => t.name) });
      saveState(); return { context: { ...context, messages, tools: available } };
    },
    beforeToolCall: async ({ toolCall, args }) => {
      const params = args as JsonObject;
      if (config.eval_mode && toolCall.name === "read_source" && params.source_id === evaluationInputAlias) params.source_id = task.input.source_id;
      if (invalidArgumentErrors.has(toolCall.id)) return { block: true, reason: invalidArgumentErrors.get(toolCall.id)! };
      const actionKey = actionKeys.get(toolCall.id);
      if (!actionKey || invalidNativeIds.has(toolCall.id) || blocked.has(actionKey) || !budget.toolAdmitted(actionKey) || state.phase !== "reviewing" || repairInstruction || budget.remainingMs() <= 0) return { block: true, reason: "工具已被阶段、重复ID或全局预算阻止；应提交终稿。", terminate: budget.remainingMs() <= 0 };
      if (toolCall.name === "read_source" && !await finalReadFits(params, true)) { forceFinal = true; state.stopReason = "context_capacity_reached"; return { block: true, reason: "继续读取会挤占终稿空间，保留已有完整原话并立即终稿；未读候选不能声称已比较。" }; }
      if (toolCall.name === "search_cases") {
        for (const query of params.queries) {
          query.focus_refs.forEach((r: JsonObject) => state.resolve(r));
          if (query.risk_point_ids.some((id: string) => !state.points.some(p => p.risk_point_id === id))) return { block: true, reason: "INVALID_RISK_POINT_ID" };
        }
      }
      return undefined;
    },
    finishTurn: async ({ message }) => {
      if (admissionStopped) return { action: 'end' };
      if (["error", "aborted"].includes(message.stopReason)) { state.stopReason = budget.remainingMs() <= 0 || message.errorMessage === "deadline_exceeded" ? "deadline_exceeded" : ['context_window_exceeded','TRIAL_REQUEST_BODY_TOO_LARGE'].includes(message.errorMessage ?? '') ? "context_capacity_reached" : ["budget_exhausted", "TRIAL_CNY_BUDGET_EXHAUSTED"].includes(message.errorMessage ?? "") ? "budget_exhausted" : "model_failure"; state.validationErrors.push(message.errorMessage ?? message.stopReason); return { action: "end" }; }
      const text = message.content.filter(c => c.type === "text").map(c => c.type === "text" ? c.text : "").join("");
      const hasTools = message.content.some(c => c.type === "toolCall");
      try {
        if (hasTools) {
          if (state.phase === "analyzing" || state.phase === "finalizing" || repairInstruction) throw new Error("TOOLS_FORBIDDEN_IN_THIS_PHASE");
          const reply = classifyReviewReply(text, true);
          if (reply.type === "native_tools" && reply.decision) {
            try { validateOutput(reply.decision, "decision"); state.acceptDecision(reply.decision); if (reply.decision.goal === "finalize") forceFinal = true; }
            catch (error) { const errorText = String((error as Error).message); state.validationErrors.push(errorText); trace.event("review.note_rejected", { error: errorText, state_updated: false }); }
          } else if (reply.type === "native_tools" && reply.ignored_note) trace.event("review.note_ignored", { reason: "not_explicit_decision", state_updated: false });
          if (!message.content.some(c => c.type === 'toolCall' && invalidArgumentErrors.has(c.id))) toolFormatInstruction = "";
          if (blocked.size) { forceFinal = true; state.stopReason = "budget_exhausted"; }
          saveState(); return { action: "continue" };
        }
        if (state.phase === "reviewing") {
          const reply = classifyReviewReply(text, false);
          if (reply.type === "tool_intent") return correctToolFormat(reply.name);
        }
        const raw = parseModelJson(text);
        if (evaluationFirstPending) {
          const continuation = acceptEvaluation(raw, state, task);
          if (pendingEvaluationRepair) {
            const pending = pendingEvaluationRepair;
            if (state.validationErrors[pending.errorIndex] !== pending.message) throw new Error("FORMAT_RECOVERY_ERROR_IDENTITY_MISMATCH");
            trace.event("output.format_recovered", { attempt: pending.attempt, code: pending.diagnostic.code,
              category: pending.diagnostic.category, rejected_reply_sha256: pending.diagnostic.reply_sha256,
              accepted_reply_sha256: hash(text), scope: "evaluation_first_format", semantic_sufficiency_verified: false });
            // Keep the rejected reply and its audit; retire only this recovered blocker.
            state.validationErrors.splice(pending.errorIndex, 1);
            pendingEvaluationRepair = undefined;
          }
          repairInstruction = ""; state.repairing = false;
          const report = await assembleReport(task, state, budget, bridge);
          const route = evaluationRoute(continuation, state, report);
          const continuing = config.eval_mode === "adaptive_rag" && route === "continue" && !forceFinal && budget.remainingMs() > 0;
          trace.event("evaluation.first_decision", { route, continuing, semantic_sufficiency_verified: false, protocol_sha256: evaluationProtocolHash });
          evaluationFirstPending = false;
          if (continuing) {
            for (const gap of continuation.gaps) {
              const id = `G${String(state.gaps.size + 1).padStart(4, "0")}`;
              state.gaps.set(id, { gap_id: id, risk_point_id: gap.risk_point_id, type: "case_evidence", statement: gap.reason, status: "unresolved", origin: "evaluation_first" });
            }
            state.phase = "reviewing"; state.final = undefined; saveState(); return { action: "continue" };
          }
          if (route === "input_facts_missing") state.stopReason = "input_facts_missing";
          else if (route === "limited_assessment") state.stopReason = "retrieval_partial_exhausted";
          else if (route === "retrieval_failed") state.stopReason = "tool_failure";
          else if (route === "incomplete" && (state.stopReason === "evidence_sufficient" || state.protocolInvalid && state.stopReason === "tool_failure")) { state.protocolInvalid = true; state.stopReason = "invalid_output"; }
          // assembleReport must see any final routing flags before saving its result.
          evaluationReport = route === "incomplete" ? undefined : report;
          if (evaluationReport) evaluationReport.stop.reason = state.stopReason;
          return { action: "end" };
        }
        if (state.phase === "analyzing") {
          const value = normalizeModelOutput(raw, "analysis", state, task); state.acceptAnalysis(value); repairInstruction = "";
          emitProgress("condition_comparison");
          if (config.engine === "fixed_rag") { await fixedRetrieval(); if (retrievalCircuitOpen) return { action: "end" }; forceFinal = true; }
          else if (state.protocolVersion === "finrisk-model-wire/1.2" && !state.points.length) forceFinal = true;
          saveState(); return { action: "continue" };
        }
        if (state.phase === "reviewing" && state.protocolVersion === "finrisk-model-wire/1.2") {
          if (raw.kind !== "final") validateFinalize(raw);
          trace.event("final.handoff", { submitted_kind: raw.kind, early_final_not_accepted: raw.kind === "final" });
          toolFormatInstruction = ""; forceFinal = true; saveState(); return { action: "continue" };
        }
        const value = normalizeModelOutput(raw, "final", state, task); state.validateRefs(value); state.final = value; repairInstruction = "";
        if (state.stopReason === "evidence_sufficient" && value.risk_points.some((p: JsonObject) => p.unknowns.length)) state.stopReason = "input_facts_missing";
        return { action: "end" };
      } catch (error) { return invalid(error, text); }
    },
  });
  agent.subscribe(event => {
    if (event.type === "message_end") {
      if (seenMessages.has(event.message)) return;
      seenMessages.add(event.message);
      trace.message(event.message);
      if (event.message.role === "assistant") {
        const ids = event.message.content.filter(c => c.type === "toolCall").map(c => c.type === "toolCall" ? c.id : "");
        for (const id of ids) if (ids.filter(item => item === id).length > 1 || actionKeys.has(id)) {
          invalidNativeIds.add(id); state.protocolInvalid = true; forceFinal = true;
          state.stopReason = "invalid_output";
          if (!state.validationErrors.includes("DUPLICATE_NATIVE_TOOL_ID")) state.validationErrors.push("DUPLICATE_NATIVE_TOOL_ID");
        }
        for (const call of event.message.content) {
        if (call.type !== "toolCall") continue;
        const key = `ACTION${String(++actionOrdinal).padStart(4, "0")}`;
        actionKeys.set(call.id, key);
        try { budget.chargeTool(key, call.name); } catch { blocked.add(key); forceFinal = true; }
        try { validateArgs(call.name, call.arguments); } catch (error) {
          const reason = String((error as Error).message);
          invalidArgumentErrors.set(call.id, reason); state.validationErrors.push(reason);
          if (state.repairs + state.toolFormatCorrections < config.max_json_repairs_per_task) {
            state.toolFormatCorrections++;
            toolFormatInstruction = `上一工具参数被拒绝且未执行后台查询：${reason}。按工具Schema纠正；每次最多${config.max_queries_per_search}条查询，仍受原总预算限制。`;
            trace.event('review.argument_correction', { error: reason, corrections_used: state.repairs + state.toolFormatCorrections });
          } else { blocked.add(key); forceFinal = true; }
        }
        }
      }
    }
    if (event.type !== "message_update") trace.event(`pi.${event.type}`, event);
  });
  const deadline = setTimeout(() => { state.stopReason = "deadline_exceeded"; agent.abort(); }, config.task_deadline_seconds * 1000);
  try {
    if (firstQuery) {
      const programTool = async (name: string, id: string, params: JsonObject, onAdmitted?: () => void) => {
        validateArgs(name, params); budget.chargeTool(id, name); onAdmitted?.();
        const result = await execute(name, id, params);
        if (retrievalCircuitOpen) throw new Error("RETRIEVAL_CIRCUIT_OPEN");
        return result;
      };
      await programTool("search_cases", "evaluation:search", firstQuery);
      const schedule = sourceReadSchedule(state.searches.map(q => ({ ...q, candidates: (q.candidates ?? []).slice(0, 3) })), "evaluation");
      trace.event("evaluation.prefetch_plan", { sources: schedule.planned.map(c => c.source_id), selection: "top3_candidates_required_anchor_order_max3_windows", query: firstQuery.queries[0].text });
      let endReason = "read_limit";
      try {
        for (let index = 0; index < 3; index++) {
          if (budget.remainingMs() <= 0) throw new Error("deadline_exceeded");
          const candidate = schedule.next(index);
          if (!candidate) { endReason = "no_unattempted_anchors"; break; }
          const anchor = candidate.evidence?.[0];
          if (!anchor) { state.protocolInvalid = true; endReason = "invalid_anchor"; break; }
          const read = await programTool("read_source", `evaluation:read:${index}`, { source_id: candidate.source_id, anchor_evidence_id: anchor.evidence_id, max_chars: 1000 }, () => schedule.attempt(candidate));
          if (read.terminate) throw new Error(retrievalCircuitOpen ? "RETRIEVAL_CIRCUIT_OPEN" : "deadline_exceeded");
        }
      } catch (error) {
        const reason = String((error as Error).message);
        endReason = reason === "budget_exhausted" ? "tool_budget" : reason === "RETRIEVAL_CIRCUIT_OPEN" ? "retrieval_circuit_open" : reason;
        throw error;
      } finally { schedule.end(endReason); }
    }
    await agent.prompt(JSON.stringify({ input: { source_id: firstQuery ? evaluationInputAlias : task.input.source_id, user_context: task.input.user_context, segments: task.input.segments.map(s => ({ segment_id: s.segment_id, text: s.text })) }, instruction: firstQuery ? "一次提交evaluation_first JSON，覆盖全部输入及候选点并给出终判。" : "先理解所有输入段落，提交精简analysis JSON；不得因结构示例points为空而跳过真实风险。" })); await agent.waitForIdle();
  }
  catch (error) {
    if (!admissionStopped) {
      state.validationErrors.push(String(error));
      const reason = String((error as Error).message);
      if (evaluationEnabled && ["deadline_exceeded", "budget_exhausted"].includes(reason)) state.stopReason = reason;
      else if (!evaluationEnabled || !["deadline_exceeded", "budget_exhausted", "tool_failure"].includes(state.stopReason)) state.stopReason = "model_failure";
    }
  }
  finally { clearTimeout(deadline); await cancelBackend(); }
  const report = evaluationReport ?? await assembleReport(task, state, budget, bridge);
  if (evaluationReport) Object.assign(report.resources, budget.summary());
  if (evaluationEnabled) {
    if (evaluationCitations) report.versions.prompt_sha256 = hash(systemPrompt);
    report.versions.evaluation_protocol_sha256 = evaluationProtocolHash;
    report.versions.evaluation_mode = config.eval_mode ?? "fixed_rag";
    report.versions.evaluation_arm = config.evaluation_arm ?? (config.eval_mode === "one_shot_rag" ? "R1" : "A");
  }
  if (retrievalAccounting) report.resources.retrieval_accounting = retrievalAccounting;
  if (onlyPartialRerankGaps(state, report) && state.stopReason === "evidence_sufficient") {
    state.stopReason = "retrieval_partial_exhausted"; report.stop.reason = state.stopReason;
  }
  validateProgram(report, "final-report");
  trace.save("validation.json", report.validation); trace.save("report.json", report); saveState();
  if (state.retrievalReliability) {
    try {
      const digest = task.task_dir ? hash(readFileSync(`${task.task_dir}/report.json`, "utf8")) : null;
      const assessment = retrievalAssessment(state, report, digest);
      trace.save("retrieval-assessment.json", assessment); trace.event("retrieval.assessment", assessment);
    } catch (error) {
      if (task.task_dir) rmSync(`${task.task_dir}/retrieval-assessment.json`, { force: true });
      report.task_status = "partial"; report.needs_manual_review = true; state.phase = "partial";
      state.stopReason = "diagnostic_write_failed"; report.stop.reason = state.stopReason;
      report.summary = "审查诊断未能完整保存，本轮结果需复核；原有执行与费用记录保留。";
      report.validation.errors.push("RETRIEVAL_DIAGNOSTIC_WRITE_FAILED");
      validateProgram(report, "final-report");
      trace.save("validation.json", report.validation); trace.save("report.json", report); saveState();
      try {
        const digest = task.task_dir ? hash(readFileSync(`${task.task_dir}/report.json`, "utf8")) : null;
        trace.save("retrieval-assessment.json", retrievalAssessment(state, report, digest));
      } catch { if (task.task_dir) rmSync(`${task.task_dir}/retrieval-assessment.json`, { force: true }); }
      const message = String((error as Error).message);
      trace.event("retrieval.assessment_failed", { code: "RETRIEVAL_DIAGNOSTIC_WRITE_FAILED", diagnostic: message.startsWith("INVALID_PROGRAM_CONTRACT:") ? message.slice(0, 1000) : "DIAGNOSTIC_GENERATION_OR_STORAGE_FAILED" });
    }
  }
  emitProgress(report.task_status === "completed" ? "complete" : "incomplete", report);
  trace.event("task.finished", { status: report.task_status, stop: report.stop, resources: report.resources });
  return { report, events: trace.events };
}
