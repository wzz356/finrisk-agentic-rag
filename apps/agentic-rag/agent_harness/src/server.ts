import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { timingSafeEqual } from "node:crypto";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { root, type TaskInput } from "./contracts.js";
import { runTask } from "./runner.js";
import { HttpToolBridge } from "./tool_bridge.js";
import { atomicJson } from "./trace.js";

const token = process.env.FINRISK_AGENT_SERVICE_TOKEN;
if (!token) throw new Error("FINRISK_AGENT_SERVICE_TOKEN_REQUIRED");
const baseUrl = process.env.FINRISK_AGENT_PYTHON_URL ?? "http://127.0.0.1:8765";
if (!["127.0.0.1", "localhost"].includes(new URL(baseUrl).hostname)) throw new Error("PYTHON_MUST_BE_LOOPBACK");
const taskRoot = process.env.FINRISK_AGENT_TASK_ROOT ?? join(root, "runtime/agent_rag/tasks");
const port = Number(process.env.FINRISK_AGENT_NODE_PORT ?? 8766);
let active = false;
if (existsSync(taskRoot)) for (const id of readdirSync(taskRoot)) {
  if (!/^[a-f0-9]{32}$/.test(id)) continue;
  const path = join(taskRoot, id, "state.json");
  if (!existsSync(path)) continue;
  const state = JSON.parse(readFileSync(path, "utf8"));
  if (!["completed", "partial", "failed", "interrupted"].includes(state.phase)) atomicJson(path, { ...state, phase: "interrupted", stop_reason: "interrupted", restart_action: "no_auto_resume" });
}
function respond(response: ServerResponse, status: number, value: unknown): void {
  response.writeHead(status, { "content-type": "application/json; charset=utf-8" }); response.end(JSON.stringify(value));
}
async function body(request: IncomingMessage): Promise<any> {
  let data = ""; for await (const chunk of request) { data += chunk; if (data.length > 4096) throw new Error("BODY_TOO_LARGE"); }
  return JSON.parse(data);
}
const server = createServer(async (request, response) => {
  const auth = Buffer.from(request.headers.authorization ?? ""); const expected = Buffer.from(`Bearer ${token}`);
  if (auth.length !== expected.length || !timingSafeEqual(auth, expected)) { respond(response, 401, { error: "UNAUTHORIZED" }); return; }
  if (request.method === "GET" && request.url === "/internal/harness/health") { respond(response, 200, { status: "ok", active, execution_mode: "mock", pi_agent_core: "1.2.0" }); return; }
  if (request.method !== "POST" || request.url !== "/internal/harness/run") { respond(response, 404, { error: "NOT_FOUND" }); return; }
  if (active) { respond(response, 429, { error: "CAPACITY_EXCEEDED" }); return; }
  active = true;
  try {
    const payload = await body(request);
    if (Object.keys(payload).length !== 1 || !/^[a-f0-9]{32}$/.test(payload.task_id)) throw new Error("INVALID_TASK_ID");
    const directory = join(taskRoot, payload.task_id);
    const reportPath = join(directory, "report.json");
    if (existsSync(reportPath)) { respond(response, 200, { report: JSON.parse(readFileSync(reportPath, "utf8")), reused: true }); return; }
    if (existsSync(join(directory, "state.json"))) throw new Error("EXISTING_RUN_CANNOT_AUTO_RESUME");
    const bundleResponse = await fetch(`${baseUrl}/internal/agent-rag/tasks/${payload.task_id}/bundle`, { headers: { authorization: `Bearer ${token}` }, redirect: "error", signal: AbortSignal.timeout(10000) });
    if (!bundleResponse.ok) throw new Error(`BUNDLE_HTTP_${bundleResponse.status}`);
    const task = await bundleResponse.json() as TaskInput;
    if (task.input.task_id !== payload.task_id) throw new Error("TASK_ID_MISMATCH");
    task.task_dir = directory;
    const result = await runTask(task, new HttpToolBridge(payload.task_id, baseUrl, token));
    respond(response, 200, { report: result.report });
  } catch (error) { respond(response, 422, { error: String((error as Error).message).slice(0, 300) }); }
  finally { active = false; }
});
server.requestTimeout = 10000;
server.listen(port, "127.0.0.1", () => console.log(`FinRisk Pi harness PID ${process.pid}: http://127.0.0.1:${port}; live ${process.env.FINRISK_AGENT_ALLOW_LIVE === "1" ? "enabled" : "disabled"}`));
