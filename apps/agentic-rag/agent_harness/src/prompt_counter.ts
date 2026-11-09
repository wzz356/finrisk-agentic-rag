import { convertMessages } from "@earendil-works/pi-ai/api/openai-completions";
import { resolveTranscriptTools } from "@earendil-works/pi-ai/utils/transcript";
import { getJsonSchemaToolParameters } from "@earendil-works/pi-ai/api/constrained-sampling";
import type { Model, TranscriptContext } from "@earendil-works/pi-ai";
import { canonical, type JsonObject, type Phase } from "./contracts.js";
import { estimateTokens } from "./context_builder.js";

export const deepseekModel: Model<"openai-completions"> = {
  id: "deepseek-flash", name: "DeepSeek Flash官方接口", provider: "deepseek", api: "openai-completions",
  baseUrl: "https://api.deepseek.com", reasoning: true, input: ["text"],
  contextWindow: 1000000, maxTokens: 384000, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  compat: { supportsDeveloperRole: false, supportsUsageInStreaming: true, maxTokensField: "max_tokens", thinkingFormat: "deepseek", supportsStrictMode: false, requiresReasoningContentOnAssistantMessages: true },
};
// Fixed pi-ai 1.2.0 DeepSeek compatibility; tested against the actual onPayload conversion.
const compat = {
  supportsStore: false, supportsDeveloperRole: false, supportsReasoningEffort: true,
  supportsUsageInStreaming: true, supportsFinishReason: true, maxTokensField: "max_tokens" as const,
  requiresToolResultName: false, requiresAssistantAfterToolResult: false, requiresThinkingAsText: false,
  requiresReasoningContentOnAssistantMessages: true, thinkingFormat: "deepseek" as const,
  openRouterRouting: {}, vercelGatewayRouting: {}, chatTemplateKwargs: {}, chatTemplateArgs: {},
  zaiToolStream: false, supportsStrictMode: false, supportsOpenAIGrammarTools: false,
  supportsMidConvoSystemMessages: false, supportsMidConvoToolAdditions: false,
  sendSessionAffinityHeaders: false, sessionAffinityFormat: "openai" as const, supportsLongCacheRetention: true,
};
export interface PromptCount extends JsonObject { prompt_tokens: number; counter_kind: string; }
export type PromptCounter = (payload: JsonObject, timeoutMs?: number) => Promise<PromptCount>;
export const assetCommit = "5254448103b6254fc646dacbca1cbbf20e9216c1";
const assetHashes = {
  "tokenizer.json": "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b",
  "encoding.py": "502bdaec8a3fd88ebc24c4721a7038fbe42f2063c664638127056107920035c1",
  "tokenizer_config.json": "6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547",
  "chat_template.jinja": "d959d804d5101b79a49b6ff1bf3c54cd5affa9a5a78c503d925f3090040b31c3",
  "LICENSE": "f2c6c602815669d292889e5be8c802f2ed950653b77999b1584e8e6aed25d040",
};

export function finalizePayload(payload: JsonObject, phase: Phase, repairing = false): JsonObject {
  return { ...payload, thinking: { type: "disabled" }, reasoning_effort: "none", stream_options: { include_usage: true },
    ...(phase === "analyzing" || phase === "finalizing" || repairing ? { response_format: { type: "json_object" }, tools: [], tool_choice: "none" } : {}) };
}
export function promptProjection(payload: JsonObject): JsonObject {
  return { model: payload.model ?? "deepseek-flash", messages: payload.messages, tools: payload.tools ?? [],
    ...(payload.response_format ? { response_format: payload.response_format } : {}) };
}
export function payloadForContext(context: TranscriptContext, phase: Phase, repairing = false): JsonObject {
  const tools = resolveTranscriptTools(context.messages, false).requestTools.map(tool => ({ type: "function", function: {
    name: tool.name, description: tool.description, parameters: getJsonSchemaToolParameters(tool, undefined),
  } }));
  return promptProjection(finalizePayload({ model: deepseekModel.id, messages: convertMessages(deepseekModel, context, compat), tools }, phase, repairing));
}
export function buildProviderPayload(context: TranscriptContext, phase: Phase, repairing = false, maxTokens = 4096): JsonObject {
  return finalizePayload({ ...payloadForContext(context, phase, repairing), stream: true, max_tokens: maxTokens }, phase, repairing);
}
export async function countPayload(payload: JsonObject, counter?: PromptCounter, timeoutMs?: number): Promise<PromptCount> {
  const projected = promptProjection(payload);
  try {
    if (!counter) throw new Error("TOKEN_COUNTER_UNAVAILABLE");
    const count = await counter(projected, timeoutMs);
    if (!Number.isInteger(count.prompt_tokens) || count.prompt_tokens < 1 || count.counter_kind !== "deepseek_v41_official"
      || count.model_version !== "DeepSeek-V4.1-Flash" || count.asset_commit !== assetCommit
      || canonical(count.asset_hashes ?? {}) !== canonical(assetHashes)) throw new Error("TOKEN_COUNTER_METADATA_MISMATCH");
    return count;
  } catch (error) {
    const code = String((error as Error).message).split(":", 1)[0] ?? "TOKEN_COUNTER_FAILED";
    // Byte fallback covers rare Unicode's byte tokens. Reserve extra framing for
    // the fixed V4.1 role/tool/JSON wrappers; this is an upper bound, not a ratio profile.
    const byteUpperBound = Buffer.byteLength(JSON.stringify(projected), "utf8") + 4096;
    return { prompt_tokens: Math.max(estimateTokens(projected), byteUpperBound), counter_kind: "conservative_payload_estimate", fallback_reason: /^[A-Z_0-9]+$/.test(code) ? code : "TOKEN_COUNTER_FAILED", fallback_method: "max_legacy_utf8_payload_plus_4096_framing" };
  }
}
export function samePrompt(left: JsonObject, right: JsonObject): boolean {
  return canonical(promptProjection(left)) === canonical(promptProjection(right));
}
