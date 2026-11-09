import type { JsonObject } from "./contracts.js";

export type ReviewReply =
  | { type: "native_tools"; decision?: JsonObject; ignored_note: boolean }
  | { type: "tool_intent"; name: string }
  | { type: "json"; value: JsonObject }
  | { type: "invalid"; error: string };

// 只分类正文与原生调用，不执行工具、不读取任务状态、不接纳任何事实。
export function classifyReviewReply(text: string, hasNativeTools: boolean): ReviewReply {
  let value: unknown;
  try { value = JSON.parse(text); }
  catch (error) {
    return hasNativeTools ? { type: "native_tools", ignored_note: Boolean(text.trim()) }
      : { type: "invalid", error: String((error as Error).message) };
  }
  const row = value && typeof value === "object" && !Array.isArray(value) ? value as JsonObject : undefined;
  if (hasNativeTools) return { type: "native_tools", ...(row?.kind === "decision" ? { decision: row } : {}), ignored_note: Boolean(text.trim()) && row?.kind !== "decision" };
  if (!row) return { type: "invalid", error: "INVALID_REVIEW_OBJECT" };
  if (["search_cases", "read_source", "get_risk_definition"].includes(row.kind)) return { type: "tool_intent", name: row.kind };
  return { type: "json", value: row };
}
