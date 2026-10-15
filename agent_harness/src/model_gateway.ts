import type { StreamFn } from "@earendil-works/pi-agent-core";
import type { AssistantMessage, TranscriptContext, Model, Api } from "@earendil-works/pi-ai";
import { createAssistantMessageEventStream } from "@earendil-works/pi-ai/utils/event-stream";
import type { Budget } from "./budget.js";
import type { TaskState } from "./state.js";
import type { Trace } from "./trace.js";
import { estimateTokens } from "./context_builder.js";
import type { JsonObject } from "./contracts.js";
import { countPayload, payloadForContext, type PromptCounter } from "./prompt_counter.js";

export type ScriptStep = JsonObject | unknown[] | string | ((context: TranscriptContext, state: TaskState) => unknown | Promise<unknown>);
export const mockModel: Model<"openai-completions"> = { id: "finrisk-mock-v1", name: "FinRisk离线协议模拟", api: "openai-completions", provider: "finrisk-mock", baseUrl: "http://127.0.0.1", reasoning: false, input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 32768, maxTokens: 4096 };
function assistant(model: Model<Api>, content: AssistantMessage["content"], stopReason: AssistantMessage["stopReason"]): AssistantMessage {
  return { role: "assistant", content, api: model.api, provider: model.provider, model: model.id, stopReason, timestamp: Date.now(), usage: { input: 16, output: 16, cacheRead: 0, cacheWrite: 0, totalTokens: 32, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } } };
}
export function mockGateway(budget: Budget, state: TaskState, trace: Trace, steps: ScriptStep[], counter?: PromptCounter): StreamFn {
  let position = 0;
  return (model, context, options) => {
    const stream = createAssistantMessageEventStream();
    void (async () => {
    let requestId: string;
    try {
      const input = counter ? (await countPayload(payloadForContext(context, state.phase, state.repairing), counter, budget.remainingMs())).prompt_tokens : estimateTokens(context.messages);
      if (counter && input + budget.config.context_safety_margin_tokens > budget.config.service_max_input_tokens) throw new Error("context_window_exceeded");
      requestId = budget.reserveModel(input, state.phase === "finalizing");
      trace.event("model.request", { request_id: requestId, phase: state.phase, execution_mode: "mock", physical_request: false });
    } catch (error) {
      const failed = assistant(model, [], "error"); failed.errorMessage = String((error as Error).message);
      stream.push({ type: "start", partial: failed }); stream.push({ type: "error", reason: "error", error: failed });
      return stream;
    }
    const deadlineMs = Math.min(budget.remainingMs(), budget.config.model_request_timeout_seconds * 1000);
    const step = steps[position++];
    let settled = false;
    const timer = setTimeout(() => finishError(budget.remainingMs() <= 0 ? "deadline_exceeded" : "model_request_timeout", true), deadlineMs);
    const abort = () => finishError("cancelled", true);
    options?.signal?.addEventListener("abort", abort, { once: true });
    function cleanup(): void { clearTimeout(timer); options?.signal?.removeEventListener("abort", abort); }
    function finishError(reason: string, unknown: boolean): void {
      if (settled) return;
      settled = true; cleanup(); budget.settle(requestId, unknown ? undefined : 0);
      const failed = assistant(model, [], reason === "cancelled" ? "aborted" : "error");
      failed.errorMessage = reason;
      trace.event("model.failure", { request_id: requestId, reason, usage_unknown: unknown });
      stream.push({ type: "start", partial: failed }); stream.push({ type: "error", reason: reason === "cancelled" ? "aborted" : "error", error: failed });
    }
    void Promise.resolve().then(() => typeof step === "function" ? step(context, state) : step).then(value => {
      if (settled) { trace.event("model.late_result", { request_id: requestId, ignored: true }); return; }
      if (value === undefined) { finishError("mock_script_exhausted", false); return; }
      settled = true; cleanup();
      const content: AssistantMessage["content"] = Array.isArray(value) ? value as AssistantMessage["content"] : [{ type: "text", text: typeof value === "string" ? value : JSON.stringify(value) }];
      // 工具ID在任务内必须唯一；同一脚本回合再次出现同ID也不能逃过动作计数。
      content.forEach(c => { if (c.type === "toolCall") c.id = `${requestId}:${c.id}`; });
      const message = assistant(model, content, content.some(c => c.type === "toolCall") ? "toolUse" : "stop");
      budget.settle(requestId, message.usage.totalTokens);
      trace.event("model.settled", { request_id: requestId, usage: message.usage, usage_kind: "simulated" });
      stream.push({ type: "start", partial: message });
      stream.push({ type: "done", reason: message.stopReason as "stop" | "toolUse", message });
    }).catch(error => finishError(`mock_source_error:${String(error)}`, true));
    })();
    return stream;
  };
}
