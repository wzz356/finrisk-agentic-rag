import { createInterface } from "node:readline";
import { runMaintenanceAgent, type MaintenanceTask } from "./maintenance_runner.js";
import { emptyMaintenanceResources, type MaintenanceEvent, type MaintenanceScriptStep } from "./maintenance_gateway.js";

// One role run per process. No provider is registered here and live is always denied.
const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
const controller = new AbortController();
const pending = new Map<string, { expected: string; resolve: (value: any) => void; reject: (error: Error) => void; cleanup: () => void }>();
let started = false, finished = false, sequence = 0;
const write = (message: MaintenanceEvent) => process.stdout.write(JSON.stringify(message) + "\n");
function rpc(kind: string, expected: string, payload: MaintenanceEvent, signal?: AbortSignal): Promise<any> {
  const requestId = typeof payload.request_id === "string" ? payload.request_id : `bridge:${++sequence}`;
  return new Promise((resolve, reject) => {
    if (signal?.aborted) { reject(new Error("CANCELLED")); return; }
    const abort = () => { pending.delete(requestId); cleanup(); reject(new Error("CANCELLED")); };
    const cleanup = () => signal?.removeEventListener("abort", abort);
    pending.set(requestId, { expected, resolve, reject, cleanup });
    signal?.addEventListener("abort", abort, { once: true });
    write({ ...payload, kind, request_id: requestId });
  });
}
function finish(message: MaintenanceEvent): void {
  if (finished) return;
  finished = true;
  write({ ...message, kind: "final_result" });
  for (const item of pending.values()) { item.cleanup(); item.reject(new Error("RUN_FINISHED")); }
  pending.clear(); lines.close(); process.stdin.pause();
}
lines.on("line", line => {
  if (finished) return;
  let message: MaintenanceEvent;
  try { message = JSON.parse(line); if (!message || typeof message !== "object") throw new Error(); }
  catch { controller.abort(); finish({ output: null, events: [], stop_reason: "invalid_protocol" }); return; }
  if (message.kind === "cancel") { controller.abort(); return; }
  if (!started) {
    if (message.kind !== "start") { finish({ output: null, events: [], stop_reason: "start_required" }); return; }
    started = true;
    const task = message.task as MaintenanceTask;
    void runMaintenanceAgent(task, { async call(name, args, binding, signal) { const reply = await rpc("tool_request", "tool_result", { name, args, ...binding }, signal); return reply.result; } }, {
      signal: controller.signal, script: message.script as MaintenanceScriptStep[] | undefined,
      ledger: {
        async reserve(payload, signal) { const reply = await rpc("model_reserve", "model_ticket", payload, signal); return reply.ticket; },
        async settle(payload) { await rpc("model_settle", "model_settled", payload); },
      },
    }).then(finish, error => {
      const code = error instanceof Error && /^[A-Z_]+$/.test(error.message) ? error.message : "MAINTENANCE_START_FAILED";
      finish({ run_id: task?.run_id, role: task?.role, revision: task?.revision, fixture: task?.execution_mode === "fixture", execution_mode: task?.execution_mode, output: null, events: [], resources: emptyMaintenanceResources(), stop_reason: code });
    });
    return;
  }
  const item = pending.get(String(message.request_id));
  if (!item || item.expected !== message.kind) { controller.abort(); return; }
  pending.delete(String(message.request_id)); item.cleanup();
  if (message.error !== undefined && message.error !== null) item.reject(new Error("HOST_REQUEST_REJECTED")); else item.resolve(message);
});
lines.on("close", () => { if (!finished) controller.abort(); });
process.on("SIGINT", () => controller.abort());
process.on("SIGTERM", () => controller.abort());
