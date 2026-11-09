import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import test from "node:test";
import { root, type Config } from "../src/contracts.js";
import { loopArmBinding } from "../src/agent_loop_evaluation.js";
import { plannedExecutionContract } from "../src/agent_loop_runner.js";
import { plannedServiceContractInventory } from "../src/agent_loop_service_contract.js";

test("Python shared service inventory equals actual Node exports; regenerate on drift", () => {
  const inventory = JSON.parse(readFileSync(`${root}/contracts/agent_rag/planned-execution-contract.json`, "utf8"));
  assert.deepEqual(inventory, plannedServiceContractInventory());
  assert.equal(Object.keys(inventory.execution_contract).length, 8);
  assert.equal("profile_sha256" in inventory.execution_contract, false);
});

test("both standard service engines share fixed contracts and bind their actual profile", () => {
  const inventory = plannedServiceContractInventory();
  for (const arm of ["A", "R2"] as const) {
    const config = loopArmBinding(arm, "smoke").base_config as Config;
    const { profile_sha256, ...fixed } = plannedExecutionContract(config);
    assert.match(profile_sha256!, /^[0-9a-f]{64}$/);
    assert.deepEqual(inventory.execution_contract, fixed);
    assert.equal(config.fast_path_enabled, false);
  }
});
