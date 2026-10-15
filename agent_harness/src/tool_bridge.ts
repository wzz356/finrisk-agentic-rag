import type { JsonObject, ToolEnvelope } from "./contracts.js";
import type { PromptCount } from "./prompt_counter.js";
export interface ToolBridge {
  call(name: string, args: JsonObject, queryLimit: number, timeoutMs: number): Promise<ToolEnvelope>;
  verifyEvidence(ids: string[], timeoutMs: number): Promise<{ valid_ids: string[]; rejected_ids: string[] }>;
  countTokens?(payload: JsonObject, timeoutMs: number): Promise<PromptCount>;
  previewRead?(args: JsonObject, timeoutMs: number): Promise<ToolEnvelope>;
  cancelRetrieval?(timeoutMs?: number): Promise<JsonObject>;
}
export class HttpToolBridge implements ToolBridge {
  constructor(private taskId: string, private baseUrl: string, private token: string) {
    const url = new URL(baseUrl);
    if (url.hostname !== "127.0.0.1" && url.hostname !== "localhost") throw new Error("BRIDGE_MUST_BE_LOOPBACK");
    if (!token) throw new Error("SERVICE_TOKEN_REQUIRED");
  }
  private async post(path: string, body: JsonObject, timeoutMs: number): Promise<any> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      method: "POST", redirect: "error", headers: { "content-type": "application/json", "authorization": `Bearer ${this.token}` },
      body: JSON.stringify({ task_id: this.taskId, ...body }), signal: AbortSignal.timeout(Math.max(1, Math.floor(timeoutMs))),
    });
    if (!response.ok) throw new Error(`TOOL_TRANSPORT_HTTP_${response.status}`);
    const text = await response.text();
    if (text.length > 100000) throw new Error("TOOL_TRANSPORT_BODY_TOO_LARGE");
    return JSON.parse(text);
  }
  call(name: string, args: JsonObject, queryLimit: number, timeoutMs: number): Promise<ToolEnvelope> {
    const paths: Record<string, string> = { search_cases: "/internal/agent-rag/search", read_source: "/internal/agent-rag/source/read", get_risk_definition: "/internal/agent-rag/risk-definitions" };
    const path = paths[name]; if (!path) throw new Error("UNKNOWN_TOOL");
    return this.post(path, { params: args, query_limit: queryLimit, ...(name === 'search_cases' ? { timeout_ms: Math.max(1, Math.floor(timeoutMs)) } : {}) }, timeoutMs);
  }
  cancelRetrieval(timeoutMs = 2000): Promise<JsonObject> {
    return this.post('/internal/agent-rag/retrieval/cancel', {}, timeoutMs);
  }
  verifyEvidence(ids: string[], timeoutMs: number): Promise<{ valid_ids: string[]; rejected_ids: string[] }> {
    return this.post("/internal/agent-rag/verify", { evidence_ids: ids }, timeoutMs);
  }
  countTokens(payload: JsonObject, timeoutMs: number): Promise<PromptCount> {
    return this.post("/internal/agent-rag/token-count", { payload }, timeoutMs);
  }
  previewRead(args: JsonObject, timeoutMs: number): Promise<ToolEnvelope> {
    return this.post("/internal/agent-rag/source/preview", { params: args, query_limit: 0 }, timeoutMs);
  }
}
