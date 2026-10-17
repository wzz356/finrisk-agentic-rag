import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { root } from "../src/contracts.js";
import type { PromptCounter, PromptCount } from "../src/prompt_counter.js";

// A local injected counter for tests; no HTTP, SDK fetch, dotenv, or provider credentials.
export function offlineCounter(): { count: PromptCounter; close: () => void } {
  const python = spawn(`${root}/.agent-venv/Scripts/python.exe`, ["-X", "utf8", "-u", "-c", "import sys,json\nfrom agent_rag.token_count import official_counter\nfor line in sys.stdin:\n try: print(json.dumps(official_counter.count(json.loads(line))),flush=True)\n except Exception as e: print(json.dumps({'error':str(e)}),flush=True)"], { cwd: root, env: { ...process.env, PYTHONPATH: `${root}/src` }, windowsHide: true });
  const pending: Array<{ resolve: (count: PromptCount) => void; reject: (error: Error) => void }> = [];
  let diagnostic = "";
  python.stderr.on("data", chunk => { diagnostic += String(chunk); });
  const reader = createInterface({ input: python.stdout });
  reader.on("line", line => { const item = pending.shift(); if (!item) return; const value = JSON.parse(line); if (value.error) item.reject(new Error(value.error)); else item.resolve(value); });
  python.on("error", error => { pending.splice(0).forEach(item => item.reject(error)); });
  python.on("exit", code => { pending.splice(0).forEach(item => item.reject(new Error(`OFFLINE_COUNTER_EXITED_${code}:${diagnostic}`))); });
  return { count: payload => new Promise((resolve, reject) => { pending.push({ resolve, reject }); python.stdin.write(JSON.stringify(payload) + "\n"); }), close: () => { reader.close(); python.kill(); } };
}
