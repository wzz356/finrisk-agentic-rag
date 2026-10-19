import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { type Config, type JsonObject } from "./contracts.js";
import { loopArmBinding } from "./agent_loop_evaluation.js";
import { plannedExecutionContract } from "./agent_loop_runner.js";

/** Regenerate with: node dist/src/agent_loop_service_contract.js --describe. */
export function plannedServiceContractInventory(): JsonObject {
  const config = loopArmBinding("A", "smoke").base_config as Config;
  const { profile_sha256: _, ...fixed } = plannedExecutionContract(config);
  return { schema_version: "finrisk-planned-service-contract/1.0",
    binding: Object.fromEntries(["profile_id", "retrieval_profile", "request_admission_policy", "budget_profile_id"].map(key => [key, (config as unknown as JsonObject)[key]])),
    execution_contract: fixed };
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  if (process.argv.length !== 3 || process.argv[2] !== "--describe") throw new Error("USAGE: --describe (offline generated service inventory)");
  process.stdout.write(JSON.stringify(plannedServiceContractInventory()) + "\n");
}
