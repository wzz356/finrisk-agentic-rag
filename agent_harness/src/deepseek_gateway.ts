import { readFileSync } from "node:fs";
import { stream as providerStream } from "@earendil-works/pi-ai/api/openai-completions";
import { createAssistantMessageEventStream } from "@earendil-works/pi-ai/utils/event-stream";
import type { AssistantMessage, TranscriptContext, SimpleStreamOptions } from "@earendil-works/pi-ai";
import type { Budget } from "./budget.js";
import type { TaskState } from "./state.js";
import type { Trace } from "./trace.js";
import { countPayload, deepseekModel, finalizePayload, buildProviderPayload, samePrompt, type PromptCount, type PromptCounter } from "./prompt_counter.js";
import { canonical, type JsonObject } from "./contracts.js";
import { activeTrialBudget, type TrialBudget } from "./trial_budget.js";
import { assessRequest } from './request_admission.js';

export function loadDeepSeekKey(): string {
  const readKeyFile = (path: string) => readFileSync(path, "utf8").replace(/^\uFEFF/, "").trim();
  const key = process.env.DEEPSEEK_API_KEY_FILE ? readKeyFile(process.env.DEEPSEEK_API_KEY_FILE)
    : process.env.DEEPSEEK_API_KEY?.trim() || "";
  if (!/^sk-[A-Za-z0-9_-]+$/.test(key)) throw new Error("INVALID_DEEPSEEK_KEY_FILE_FORMAT");
  return key;
}
export { deepseekModel } from "./prompt_counter.js";
export function deepseekGateway(budget: Budget, state: TaskState, trace: Trace, key: string, fetcher: typeof fetch = fetch, counter?: PromptCounter) {
  const streamFn = (_model: unknown, context: TranscriptContext, options?: SimpleStreamOptions) => {
    const output = createAssistantMessageEventStream();
    const makeError = (reason: string): AssistantMessage => ({ role: "assistant", api: "openai-completions", provider: "deepseek", model: "deepseek-flash", content: [], stopReason: "error", errorMessage: reason, timestamp: Date.now(), usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } } });
    let requestId: string | undefined; let contextCount: PromptCount | undefined;
    const timeoutMs = Math.max(1, Math.floor(Math.min(budget.remainingMs(), budget.config.model_request_timeout_seconds * 1000)));
    const controller = new AbortController();
    let finished = false; let started = false; let rawUsage: JsonObject | undefined; let status: number | undefined; let physical = 0; let transportAttempts = 0; let egressFailure: string | undefined;
    let trialGuard: TrialBudget | undefined; let trialReservation: number | undefined;
    let admittedBody: string | undefined;
    const finishError = (reason: string) => {
      if (finished) return; finished = true; clearTimeout(timer); options?.signal?.removeEventListener("abort", abort);
      controller.abort(); if (requestId) budget.settle(requestId, physical ? undefined : 0);
      const message = makeError(reason); if (!started) output.push({ type: "start", partial: message });
      output.push({ type: "error", reason: "error", error: message });
      trace.event("model.failure", { request_id: requestId ?? null, reason, physical_requests: physical, usage_unknown: physical > 0 });
    };
    const abort = () => finishError(budget.remainingMs() <= 0 ? "deadline_exceeded" : "cancelled");
    const timer = setTimeout(() => finishError(budget.remainingMs() <= 0 ? "deadline_exceeded" : "model_request_timeout"), timeoutMs);
    options?.signal?.addEventListener("abort", abort, { once: true });
    const safeFetch: typeof fetch = async (input, init) => {
      if (finished || ++transportAttempts > 1) throw new Error("HIDDEN_PROVIDER_RETRY_BLOCKED");
      if (!requestId || !contextCount) throw new Error("UNCOUNTED_PROVIDER_PAYLOAD");
      const url = new URL(input instanceof Request ? input.url : String(input));
      if (url.origin !== "https://api.deepseek.com" || url.pathname !== "/chat/completions") throw new Error("UNAPPROVED_PROVIDER_ENDPOINT");
      if (!admittedBody || typeof init?.body !== 'string' || canonical(JSON.parse(init.body)) !== canonical(JSON.parse(admittedBody))) throw new Error('PAYLOAD_PROJECTION_MISMATCH');
      const trial = activeTrialBudget(budget.config.request_admission_policy);
      if (trial) {
        try { trialReservation = trial.reserve(admittedBody); trialGuard = trial; }
        catch (error) { egressFailure = String((error as Error).message); trace.event("model.egress_blocked", { request_id: requestId, reason: egressFailure }); throw error; }
      }
      budget.recordPhysicalRequest();
      physical++;
      trace.event("model.request", { request_id: requestId, model: "deepseek-flash", execution_mode: "live", endpoint: "https://api.deepseek.com/chat/completions", physical_request: true, provider_auto_retries: 0, thinking: "disabled" });
      return fetcher(input, { ...init, body: admittedBody, redirect: "error" });
    };
    const source = providerStream(deepseekModel, context, {
      apiKey: key, fetch: safeFetch, signal: controller.signal, maxRetries: 0, timeoutMs,
      maxTokens: budget.config.max_output_tokens_per_request,
      onPayload: async payload => {
        try {
          const actual = finalizePayload(payload as JsonObject, state.phase, state.repairing);
          const prepared = buildProviderPayload(context, state.phase, state.repairing, budget.config.max_output_tokens_per_request);
          if (!samePrompt(actual, prepared) || actual.max_tokens !== prepared.max_tokens || actual.stream !== prepared.stream) throw new Error("PAYLOAD_PROJECTION_MISMATCH");
          admittedBody = JSON.stringify(prepared);
          contextCount = await countPayload(actual, counter, Math.min(timeoutMs, budget.remainingMs()));
          trace.event("context.counted", { phase: state.phase, ...contextCount, before_egress: true });
          if (finished) throw new Error("cancelled");
          if (contextCount.prompt_tokens + budget.config.context_safety_margin_tokens > budget.config.service_max_input_tokens) throw new Error("context_window_exceeded");
          const admission = activeTrialBudget(budget.config.request_admission_policy)?.preview(admittedBody) ?? (budget.config.request_admission_policy ? assessRequest(admittedBody, budget.config.request_admission_policy) : { fits: true, reason: null });
          trace.event('context.admission', { ...admission, before_egress: true });
          if (!admission.fits) throw new Error(admission.reason!);
          requestId = budget.reserveModel(contextCount.prompt_tokens, state.phase === "finalizing");
          return prepared;
        } catch (error) { egressFailure = String((error as Error).message); throw error; }
      },
      onResponse: response => { status = response.status; },
      onProviderStreamEvent: event => { if ((event as JsonObject)?.usage) rawUsage = (event as JsonObject).usage; },
    });
    void (async () => {
      try {
        for await (const event of source) {
          if (finished) { trace.event("model.late_result", { request_id: requestId, ignored: true }); continue; }
          if (event.type === "start") started = true;
          if (event.type === "error") { finishError(egressFailure ?? (status ? `PROVIDER_HTTP_${status}` : "MODEL_TRANSPORT_FAILURE")); break; }
          if (event.type === "done") {
            const reported = rawUsage && [rawUsage.prompt_tokens, rawUsage.completion_tokens, rawUsage.total_tokens].every(n => Number.isInteger(n) && n >= 0)
              && rawUsage.total_tokens === rawUsage.prompt_tokens + rawUsage.completion_tokens
              && event.message.usage.totalTokens === rawUsage.total_tokens;
            budget.settle(requestId!, reported ? event.message.usage.totalTokens : undefined);
            const costSettled = trialReservation === undefined ? false : trialGuard!.settle(trialReservation, reported ? rawUsage : undefined);
            finished = true; clearTimeout(timer); options?.signal?.removeEventListener("abort", abort);
            trace.event("model.settled", { request_id: requestId, raw_usage: rawUsage ?? null, normalized_usage: event.message.usage, usage_known: Boolean(reported), trial_reservation_id: trialReservation ?? null, cost_settled: costSettled, context_estimated_tokens: contextCount!.prompt_tokens, context_counter_kind: contextCount!.counter_kind, provider_prompt_tokens: reported ? rawUsage!.prompt_tokens : null, duration_ms: event.message.durationMs ?? null });
          }
          output.push(event);
        }
        if (!finished) finishError("MODEL_STREAM_NOT_FINALIZED");
      } catch { finishError("MODEL_STREAM_FAILURE"); }
    })();
    return output;
  };
  return { streamFn, model: deepseekModel };
}
