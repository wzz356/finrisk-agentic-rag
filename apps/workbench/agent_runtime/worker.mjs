import { createInterface } from "node:readline";
import { AgentTaskError, RUNTIME_NAME, RUNTIME_VERSION, runJsonTask } from "./runtime.mjs";

const active = new Map();
const pendingTools = new Map();
let toolSequence = 0;
const write = value => process.stdout.write(`${JSON.stringify(value)}\n`);
const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
write({ type: "ready", runtime: RUNTIME_NAME, version: RUNTIME_VERSION });

lines.on("line", line => {
  let message;
  try { message = JSON.parse(line); }
  catch { write({ type: "protocol_error", error: "invalid_json" }); return; }
  if (message.type === "tool_result") {
    const key = `${message.id}:${message.tool_call_id}`;
    const pending = pendingTools.get(key);
    if (pending) {
      pendingTools.delete(key);
      if (message.error) pending.reject(new Error(message.error));
      else pending.resolve(message.result);
    }
    return;
  }
  if (message.type === "cancel") { active.get(message.id)?.abort(); return; }
  if (message.type !== "run" || typeof message.id !== "string" || active.has(message.id)) {
    write({ id: message.id, error: { code: "protocol_error", message: "无效的 Pi 请求。", retryable: false } });
    return;
  }
  const controller = new AbortController();
  active.set(message.id, controller);
  runJsonTask({
    request: message.request,
    settings: message.settings,
    apiKey: process.env.FINRISK_PI_API_KEY,
    signal: controller.signal,
    toolExecutor: async (name, args) => {
      const toolCallId = `tool-${++toolSequence}`;
      const key = `${message.id}:${toolCallId}`;
      return new Promise((resolve, reject) => {
        if (controller.signal.aborted) { reject(new Error("审查已取消")); return; }
        const onAbort = () => { pendingTools.delete(key); reject(new Error("审查已取消")); };
        const finish = callback => value => { controller.signal.removeEventListener("abort", onAbort); callback(value); };
        controller.signal.addEventListener("abort", onAbort, { once: true });
        pendingTools.set(key, { resolve: finish(resolve), reject: finish(reject) });
        write({ type: "tool_call", id: message.id, tool_call_id: toolCallId, name, args });
      });
    },
  }).then(result => write({ id: message.id, result })).catch(error => {
    const safe = error instanceof AgentTaskError ? error : new AgentTaskError("runtime_error", "Pi Agent 运行失败，请检查本地依赖和配置。");
    write({ id: message.id, error: { code: safe.code, message: safe.message, retryable: safe.retryable } });
  }).finally(() => active.delete(message.id));
});
lines.on("close", () => {
  for (const controller of active.values()) controller.abort();
});
