import { readFileSync } from "node:fs";
import { canonical, defaults, hash, root, type Config, type JsonObject } from "./contracts.js";
import { budgetProfileSha256, loadBudgetProfile } from "./budget_profile.js";
import { loopContractInventory } from "./agent_loop_contracts.js";
import { plannedExecutionContract, plannedProtocolHash, plannedSystemPrompt, plannedToolSchemas } from "./agent_loop_runner.js";
import { describeFastPathRules, fastPathRuleHash } from "./fast_path.js";

export const loopEvaluationVersion = "finrisk-loop-evaluation/1.0";
export type LoopArm = "R1" | "R2" | "A";
export type LoopStage = "smoke" | "development" | "independent";
const arms: LoopArm[] = ["R1", "R2", "A"];
const stages: LoopStage[] = ["smoke", "development", "independent"];

/** Same single-level registered profile expansion as Python admission. No task service is constructed. */
function baseConfig(arm: LoopArm): Config {
  const override = JSON.parse(readFileSync(`${root}/configs/agent_rag/live-deepseek-flash-planned.json`, "utf8"));
  delete override.extends;
  const budget = loadBudgetProfile(override.budget_profile_id);
  const { profile_id: _, ...limits } = budget;
  return { ...defaults, ...override, ...limits, budget_profile_sha256: budgetProfileSha256(budget),
    max_external_retrieval_requests: budget.rerank_max_requests,
    ...({ rerank_cap_cny: budget.rerank_cap_micro_cny / 1000000,
      model_trial_cap_cny: budget.model_trial_cap_micro_cny / 1000000, task_total_cost_cap_cny: budget.task_total_cost_cap_micro_cny / 1000000 }),
    engine: arm === "R2" ? "fixed_rag" : "controlled_agent",
    orchestration_mode: arm === "R2" ? "planned_fixed" : "planned_adaptive" };
}
export function loopArmBinding(arm: LoopArm, stage: LoopStage): JsonObject {
  if (!arms.includes(arm) || !stages.includes(stage)) throw new Error("INVALID_LOOP_EVALUATION_ARGUMENTS");
  const base = baseConfig(arm);
  const config: Config = { ...base, orchestration_mode: arm === "R1" ? "staged_one_shot" : arm === "R2" ? "planned_fixed" : "planned_adaptive",
    fast_path_enabled: arm === "A" && stage !== "smoke" };
  return { experiment_config_id: `${loopEvaluationVersion}:${stage}:${arm}`, service_profile_id: base.profile_id,
    base_hash_kind: "node canonical shape; actual Python admission byte hash is separately frozen by prepare",
    base_config: base, base_config_node_sha256: hash(canonical(base)), config, config_sha256: hash(canonical(config)),
    planned_execution_contract: plannedExecutionContract(config) };
}
const protocol = { version: loopEvaluationVersion, planned_protocol_sha256: plannedProtocolHash,
  arms: { R1: "staged_one_shot", R2: "planned_fixed", A: "planned_adaptive" },
  fast: "A only; smoke=false; development/independent=true; frozen before execution",
  query: "canonical full text/user_context; <=500 Unicode code points; refuse without truncation",
  caches: "hybrid_result=off; rerank_pair=off; provider_prompt_cache=observed_only",
  scheduling: "serial; six balanced arm permutations; smoke A only",
  denominator: "all planned rows including missing/invalid; invalid empty gold never correct",
  business_valid: "all segments processed; current accepted assessments; no hard/global blocker; auditable local_partial allowed",
  semantic_review: "risk point coverage, necessary unknown semantics and case applicability pending human review",
  budget: "known+unknown+2500000 next task; round+stage+total40; no unknown refund; settle components once",
  gain: "candidate only; observation->gap->adaptive_model new query->new required read->later accepted use->reference improvement; human confirmation required",
  statistics: "tag-set exact; microPRF1; no-risk FP; wins/ties/losses; mean/median/max; no p95/significance" };
export const loopEvaluationProtocolHash = hash(canonical(protocol));
export function loopEvaluationDescription(): JsonObject {
  const config = loopArmBinding("A", "smoke").config as Config;
  return { schema_version: loopEvaluationVersion, evaluation_protocol_sha256: loopEvaluationProtocolHash, protocol,
    loop_contract_inventory: loopContractInventory(), planned_protocol_sha256: plannedProtocolHash,
    planned_execution_contract: plannedExecutionContract(config), planned_system_prompt: plannedSystemPrompt,
    planned_tool_schemas: plannedToolSchemas(config), budget_profile: loadBudgetProfile(config.budget_profile_id!),
    budget_profile_sha256: config.budget_profile_sha256, fast_path_rules: describeFastPathRules(), fast_path_rule_sha256: fastPathRuleHash,
    stages: Object.fromEntries(stages.map(stage => [stage, Object.fromEntries(arms.map(arm => [arm, loopArmBinding(arm, stage)]))])),
    model_metadata: { provider_model_alias: "deepseek-flash", officially_reported_release: "DeepSeek-V4.1-Flash-1017",
      release_verified_date: "2026-10-18", release_source: "https://api-docs.deepseek.com/zh-cn/updates/",
      actual_response_model: "unknown_unless_captured", actual_system_fingerprint: "unknown_unless_captured",
      tokenizer_assets_version: "DeepSeek-V4.1-Flash", tokenizer_assets_commit: "5254448103b6254fc646dacbca1cbbf20e9216c1",
      tokenizer_package_version: "0.23.1", provider_alias_is_immutable: false },
    pricing: { currency: "CNY", per_million: { input_cache_hit: 0.04, input_cache_miss: 2, output: 8 },
      checked_date: "2026-10-18", source: "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/", interpretation: "peak usage estimate; supplier actual debit unknown" } };
}
export function validateLoopEvaluationBundle(task: JsonObject, arm: LoopArm, stage: LoopStage): void {
  const row = loopArmBinding(arm, stage), meta = task.loop_evaluation;
  if (!meta || meta.stage !== stage || meta.arm !== arm || meta.protocol_sha256 !== loopEvaluationProtocolHash
    || meta.base_config_node_sha256 !== row.base_config_node_sha256 || canonical(task.config) !== canonical(row.config)
    || canonical(task.execution_contract) !== canonical(row.planned_execution_contract)) throw new Error("LOOP_EVALUATION_BINDING_MISMATCH");
}
