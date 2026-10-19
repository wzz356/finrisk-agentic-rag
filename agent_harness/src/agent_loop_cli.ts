import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { type TaskInput, type JsonObject } from "./contracts.js";
import { loopEvaluationDescription, validateLoopEvaluationBundle, type LoopArm, type LoopStage } from "./agent_loop_evaluation.js";
import { runPlannedTask } from "./agent_loop_runner.js";
import { HttpToolBridge } from "./tool_bridge.js";
import { TrialBudget } from "./trial_budget.js";

export async function agentLoopCli(argv: string[]): Promise<JsonObject> {
  if (argv.length === 1 && argv[0] === "--describe") return loopEvaluationDescription();
  if (argv.length === 1 && argv[0] === "--help") return { usage: "--describe | --audit-model-ledger PATH | --bundle PATH --arm R1|R2|A --stage smoke|development|independent --base-url URL" };
  if (argv.length === 2 && argv[0] === "--audit-model-ledger") {
    if (!existsSync(argv[1]!)) throw new Error("MODEL_LEDGER_MISSING");
    const d = loopEvaluationDescription();
    // Constructor validates an existing ledger; it does not persist or initialize it.
    new TrialBudget(argv[1]!, 1, "v2-v1", { budget_profile_id: "planned-evidence-loop-v1", budget_profile_sha256: d.budget_profile_sha256, model_trial_cap_micro_cny: 1000000 });
    return JSON.parse(readFileSync(argv[1]!, "utf8"));
  }
  const args = new Map<string, string>();
  for (let i = 0; i < argv.length; i += 2) {
    if (!["--bundle", "--arm", "--stage", "--base-url"].includes(argv[i]!) || !argv[i + 1] || args.has(argv[i]!)) throw new Error("INVALID_LOOP_EVALUATION_ARGUMENTS");
    args.set(argv[i]!, argv[i + 1]!);
  }
  if (args.size !== 4) throw new Error("INVALID_LOOP_EVALUATION_ARGUMENTS");
  if (process.env.FINRISK_AGENT_ALLOW_LOOP_EVALUATION !== "1") throw new Error("LOOP_EVALUATION_NOT_ADMITTED");
  const token = process.env.FINRISK_AGENT_SERVICE_TOKEN;
  if (!token) throw new Error("FINRISK_AGENT_SERVICE_TOKEN_REQUIRED");
  const task = JSON.parse(readFileSync(args.get("--bundle")!, "utf8"));
  validateLoopEvaluationBundle(task, args.get("--arm") as LoopArm, args.get("--stage") as LoopStage);
  if (!task.task_dir || ["state.json", "report.json", "events.jsonl"].some(name => existsSync(resolve(task.task_dir, name)))) throw new Error("EXISTING_RUN_CANNOT_AUTO_RESUME");
  if (task.mock_script !== undefined) throw new Error("LIVE_MOCK_SCRIPT_FORBIDDEN");
  const url = new URL(args.get("--base-url")!);
  if (url.protocol !== "http:" || !["127.0.0.1", "localhost"].includes(url.hostname)) throw new Error("LOCAL_TOOL_SERVICE_REQUIRED");
  return runPlannedTask(task as TaskInput, new HttpToolBridge(task.input.task_id, url.href.replace(/\/$/, ""), token));
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  agentLoopCli(process.argv.slice(2)).then(result => process.stdout.write(JSON.stringify(result) + "\n")).catch(error => {
    process.stderr.write(JSON.stringify({ error: String((error as Error).message).split(":", 1)[0] }) + "\n"); process.exitCode = 1;
  });
}
