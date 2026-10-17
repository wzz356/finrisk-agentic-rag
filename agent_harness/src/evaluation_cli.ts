import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { type TaskInput } from "./contracts.js";
import { evaluationConfig, evaluationProtocolHash, evaluationProtocolVersion, evaluationSchema, evaluationSystemPromptHash, type EvaluationArm } from "./evaluation.js";
import { runTask } from "./runner.js";
import { HttpToolBridge } from "./tool_bridge.js";
import type { ScriptStep } from "./model_gateway.js";
import { retrievalReliabilityDescription } from "./retrieval_quality.js";

export async function evaluationCli(argv: string[]): Promise<unknown> {
  if (argv.length === 1 && argv[0] === "--describe") return { schema_version: evaluationProtocolVersion, evaluation_protocol_sha256: evaluationProtocolHash, system_prompt_sha256: evaluationSystemPromptHash, envelope_schema: evaluationSchema, retrieval_reliability: retrievalReliabilityDescription, arms: { R1: "one_shot_rag", R2: "fixed_rag", A: "adaptive_rag" }, budgets: { max_model_requests: 6, max_search_calls: 2, max_tool_actions: 8, task_deadline_seconds: 270 } };
  if (process.env.FINRISK_AGENT_ALLOW_EVALUATION !== "1") throw new Error("EVALUATION_NOT_ADMITTED");
  const args = new Map<string, string>();
  for (let i = 0; i < argv.length; i += 2) {
    if (!["--bundle", "--arm", "--base-url"].includes(argv[i]!) || !argv[i + 1] || args.has(argv[i]!)) throw new Error("INVALID_EVALUATION_ARGUMENTS");
    args.set(argv[i]!, argv[i + 1]!);
  }
  if (args.size !== 3 || !["R1", "R2", "A"].includes(args.get("--arm")!)) throw new Error("INVALID_EVALUATION_ARGUMENTS");
  const token = process.env.FINRISK_AGENT_SERVICE_TOKEN;
  if (!token) throw new Error("FINRISK_AGENT_SERVICE_TOKEN_REQUIRED");
  const task = JSON.parse(readFileSync(args.get("--bundle")!, "utf8")) as TaskInput & { mock_script?: ScriptStep[] };
  if (task.task_dir && ["state.json", "report.json", "events.jsonl"].some(name => existsSync(resolve(task.task_dir!, name)))) throw new Error("EXISTING_RUN_CANNOT_AUTO_RESUME");
  if (task.config?.execution_mode === "live" && task.mock_script !== undefined) throw new Error("LIVE_MOCK_SCRIPT_FORBIDDEN");
  task.config = evaluationConfig(args.get("--arm") as EvaluationArm, task.config);
  return runTask(task, new HttpToolBridge(task.input.task_id, args.get("--base-url")!, token), task.mock_script);
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  evaluationCli(process.argv.slice(2)).then(result => process.stdout.write(JSON.stringify(result) + "\n")).catch(error => {
    // Never print input, headers or tokens in failure output.
    process.stderr.write(JSON.stringify({ error: String((error as Error).message).split(":", 1)[0] }) + "\n"); process.exitCode = 1;
  });
}
