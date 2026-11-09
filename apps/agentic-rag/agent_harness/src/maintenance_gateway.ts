import { createHash } from "node:crypto";
import type { StreamFn } from "@earendil-works/pi-agent-core";
import type { Api, AssistantMessage, Model, SimpleStreamOptions, TranscriptContext } from "@earendil-works/pi-ai";
import { createAssistantMessageEventStream } from "@earendil-works/pi-ai/utils/event-stream";

export type MaintenanceEvent = Record<string, any>;
export type MaintenanceScriptStep = string | Record<string, unknown> | AssistantMessage["content"] | ((context: TranscriptContext) => unknown | Promise<unknown>);
export interface MaintenanceLedger {
  reserve(request: MaintenanceEvent, signal?: AbortSignal): Promise<unknown>;
  settle(request: MaintenanceEvent): Promise<void>;
}
export interface VerifiedMaintenanceAdapter {
  provider: string;
  model: Model<Api>;
  streamFn: StreamFn;
  /** Must serialize the exact single HTTP request body. It contains no credential. */
  payloadForRequest: (model: Model<Api>, context: TranscriptContext, options: SimpleStreamOptions) => unknown;
  verification: { single_physical_request: true; no_internal_retries: true; uses_injected_fetch: true; context_window_verified?: true; evidence: string };
}
/** Registration is an audited application-code boundary, never accepted from task JSON. */
export class MaintenanceAdapterRegistry {
  private readonly adapters = new Map<string, VerifiedMaintenanceAdapter>();
  register(adapter: VerifiedMaintenanceAdapter): void {
    if (!adapter.provider || adapter.model.provider !== adapter.provider || typeof adapter.streamFn !== "function" || typeof adapter.payloadForRequest !== "function" || adapter.verification?.single_physical_request !== true || adapter.verification.no_internal_retries !== true || adapter.verification.uses_injected_fetch !== true || !adapter.verification.evidence?.trim()) throw new Error("ADAPTER_VERIFICATION_REQUIRED");
    const key = `${adapter.provider}:${adapter.model.id}`;
    if (this.adapters.has(key)) throw new Error("ADAPTER_ALREADY_REGISTERED");
    this.adapters.set(key, Object.freeze({ ...adapter, model: structuredClone(adapter.model), verification: Object.freeze({ ...adapter.verification }) }));
  }
  get(provider: string, model: string): VerifiedMaintenanceAdapter | undefined { return this.adapters.get(`${provider}:${model}`); }
}
export const maintenanceFixtureModel: Model<"openai-completions"> = { id: "maintenance-fixture-v1", name: "Maintenance engineering fixture", api: "openai-completions", provider: "maintenance-fixture", baseUrl: "http://127.0.0.1", reasoning: false, input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 32768, maxTokens: 2048 };
export function maintenanceHash(value: string): string { return createHash("sha256").update(value, "utf8").digest("hex"); }
export interface MaintenanceResources {
  model_requests: number; tool_actions: number; physical_provider_requests: number;
  /** Accounting units are conservative request bytes, never a tokenizer claim. */
  estimated_token_units: number; reserved_token_units: number; output_reserved_token_units: number;
  token_counter_kind: "conservative_byte_upper_bound";
  token_reservation_rule: "cumulative_prompt_upper_bound_plus_max_output_tokens_no_refund";
}
export function emptyMaintenanceResources(): MaintenanceResources {
  return { model_requests: 0, tool_actions: 0, physical_provider_requests: 0, estimated_token_units: 0, reserved_token_units: 0, output_reserved_token_units: 0, token_counter_kind: "conservative_byte_upper_bound", token_reservation_rule: "cumulative_prompt_upper_bound_plus_max_output_tokens_no_refund" };
}
export function maintenanceAbortable<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (signal?.aborted) return Promise.reject(new Error("cancelled"));
  return new Promise<T>((resolve, reject) => {
    const abort = () => { cleanup(); reject(new Error("cancelled")); };
    const cleanup = () => signal?.removeEventListener("abort", abort);
    signal?.addEventListener("abort", abort, { once: true });
    promise.then(value => { cleanup(); resolve(value); }, error => { cleanup(); reject(error); });
  });
}
function assistant(model: Model<Api>, content: AssistantMessage["content"], reason: AssistantMessage["stopReason"]): AssistantMessage {
  return { role: "assistant", content, api: model.api, provider: model.provider, model: model.id, stopReason: reason, timestamp: Date.now(), usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } } };
}
export interface MaintenanceGatewayOptions {
  binding: MaintenanceEvent;
  maxRequests: number;
  fixture: boolean;
  script?: MaintenanceScriptStep[];
  adapter?: VerifiedMaintenanceAdapter;
  ledger: MaintenanceLedger;
  emit: (type: string, data: MaintenanceEvent) => void;
  onStop: (reason: string) => void;
  resources: MaintenanceResources;
  maxOutputTokens: number;
  tokenBudget: number;
  maxInputTokens: number;
  requestTimeoutSeconds: number;
}
export function maintenanceGateway(config: MaintenanceGatewayOptions): StreamFn {
  let position = 0;
  return (model, context, options) => {
    const stream = createAssistantMessageEventStream();
    void (async () => {
      let ticket: unknown, requestId = "", settled = false, started = false;
      let request: MaintenanceEvent | undefined;
      const requestController = new AbortController();
      let timedOut = false, requestTimer: ReturnType<typeof setTimeout> | undefined;
      const cancelRequest = () => requestController.abort();
      options?.signal?.addEventListener("abort", cancelRequest, { once: true });
      if (options?.signal?.aborted) requestController.abort();
      const requestSignal = requestController.signal;
      const fail = (reason: string) => {
        config.onStop(reason);
        config.emit("model.failure", { request_id: requestId, code: reason });
        const message = assistant(model, [], requestSignal.aborted ? "aborted" : "error");
        message.errorMessage = reason;
        stream.push({ type: "start", partial: message });
        stream.push({ type: "error", reason: message.stopReason as "error" | "aborted", error: message });
      };
      const settle = async (usage: unknown, unknown: boolean) => {
        if (settled || !started) return;
        settled = true;
        // Settlement remains possible after cancellation. The host must acknowledge it.
        let timer: ReturnType<typeof setTimeout> | undefined;
        try {
          await Promise.race([config.ledger.settle({ ...request, ticket, usage, unknown, token_charge_units: request!.reserved_token_units, token_reservation_rule: config.resources.token_reservation_rule }), new Promise<never>((_, reject) => { timer = setTimeout(() => reject(new Error("ledger_settlement_failed")), 1000); })]);
          config.emit("model.settled", { request_id: requestId, usage, unknown });
        } catch { throw new Error("ledger_settlement_failed"); }
        finally { clearTimeout(timer); }
      };
      try {
        if (options?.signal?.aborted) { fail("cancelled"); return; }
        if (config.resources.model_requests >= config.maxRequests) { fail("model_budget_exhausted"); return; }
        const sequence = config.resources.model_requests + 1;
        requestId = `${String(config.binding.run_id)}:${String(config.binding.role)}:${String(config.binding.revision)}:model:${sequence}`;
        const providerOptions: SimpleStreamOptions = { ...options, signal: requestSignal, timeoutMs: config.requestTimeoutSeconds * 1000, maxTokens: config.maxOutputTokens, maxRetries: 0, maxRetryDelayMs: 0, transport: "sse" };
        const payload = config.fixture ? { model: model.id, messages: context.messages, max_tokens: config.maxOutputTokens } : config.adapter!.payloadForRequest(model, context, providerOptions);
        const body = JSON.stringify(payload);
        const promptUpperBound = Buffer.byteLength(body, "utf8");
        const tokenReservation = promptUpperBound + config.maxOutputTokens;
        if (promptUpperBound > config.maxInputTokens) { fail("input_token_limit_exceeded"); return; }
        if (!Number.isSafeInteger(tokenReservation) || tokenReservation > config.tokenBudget - config.resources.reserved_token_units) { fail("token_budget_exhausted"); return; }
        request = { ...config.binding, request_id: requestId, sequence, payload_sha256: maintenanceHash(body), provider: model.provider, model_id: model.id, physical_request: !config.fixture, execution_mode: config.fixture ? "fixture" : "live", max_tokens: config.maxOutputTokens, prompt_token_upper_bound: promptUpperBound, reserved_token_units: tokenReservation, token_counter_kind: "conservative_byte_upper_bound", ...(config.fixture ? { reserved_micro: 0 } : {}) };
        try {
          ticket = await maintenanceAbortable(config.ledger.reserve(request, options?.signal), options?.signal);
          if (ticket === null || ticket === undefined || ticket === false) throw new Error("ledger_denied");
        } catch { fail(options?.signal?.aborted ? "cancelled" : "ledger_denied"); return; }
        started = true;
        config.resources.model_requests++;
        config.resources.estimated_token_units += promptUpperBound;
        config.resources.reserved_token_units += tokenReservation;
        config.resources.output_reserved_token_units += config.maxOutputTokens;
        config.emit("model.request", request);
        requestTimer = setTimeout(() => { timedOut = true; requestController.abort(); }, config.requestTimeoutSeconds * 1000);
        let message: AssistantMessage;
        if (config.fixture) {
          const step = config.script?.[position++];
          if (step === undefined) throw new Error("fixture_script_exhausted");
          const value = await maintenanceAbortable(Promise.resolve().then(() => typeof step === "function" ? step(context) : step), requestSignal);
          const content: AssistantMessage["content"] = Array.isArray(value) ? structuredClone(value) : [{ type: "text", text: typeof value === "string" ? value : JSON.stringify(value) }];
          for (const block of content) if (block.type === "toolCall") block.id = `${requestId}:${block.id}`;
          message = assistant(model, content, content.some(block => block.type === "toolCall") ? "toolUse" : "stop");
        } else {
          let fetchUsed = false;
          // A verified adapter must honor this fetch. Every physical attempt is single-use,
          // bound to the preflight body hash; SDK retries cannot spend the same ticket twice.
          const guardedFetch: typeof fetch = async (url, init) => {
            if (fetchUsed || typeof init?.body !== "string" || maintenanceHash(init.body) !== request!.payload_sha256 || requestSignal.aborted) throw new Error("provider_payload_not_admitted");
            fetchUsed = true;
            config.resources.physical_provider_requests++;
            return fetch(url, { ...init, signal: requestSignal, redirect: "error" });
          };
          const source = await maintenanceAbortable(Promise.resolve(config.adapter!.streamFn(model, context, { ...providerOptions, fetch: guardedFetch, onPayload: actual => { if (JSON.stringify(actual) !== body) throw new Error("provider_payload_not_admitted"); return undefined; } })), requestSignal);
          message = await maintenanceAbortable(source.result(), requestSignal);
        }
        clearTimeout(requestTimer);
        const usageUnknown = message.stopReason === "error" || message.stopReason === "aborted";
        await settle(usageUnknown ? null : message.usage, usageUnknown);
        if (message.stopReason === "error" || message.stopReason === "aborted") { fail("model_failure"); return; }
        stream.push({ type: "start", partial: message });
        stream.push({ type: "done", reason: message.stopReason as "stop" | "toolUse" | "length", message });
      } catch (error) {
        const code = error instanceof Error ? error.message : "model_failure";
        const safeCode = timedOut ? "model_request_timeout" : ["fixture_script_exhausted", "provider_payload_not_admitted", "ledger_settlement_failed"].includes(code) ? code : options?.signal?.aborted ? "cancelled" : "model_failure";
        try { await settle(null, true); } catch { fail("ledger_settlement_failed"); return; }
        fail(safeCode);
      } finally { clearTimeout(requestTimer); options?.signal?.removeEventListener("abort", cancelRequest); }
    })();
    return stream;
  };
}
