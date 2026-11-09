import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import Ajv2020 from "ajv/dist/2020.js";
import type { Config, JsonObject } from "./contracts.js";

const root = fileURLToPath(new URL("../../..", import.meta.url));
const schema = JSON.parse(readFileSync(`${root}/contracts/agent_rag/budget-profile.schema.json`, "utf8"));
const ajv = new Ajv2020.default({ strict: false, allErrors: true });
const validateSchema = ajv.compile(schema);
export const PLANNED_BUDGET_PROFILE = "planned-evidence-loop-v1";
export const LEGACY_V2_BUDGET_PROFILE = "legacy-live-v2-v1";
const modes = ["staged_one_shot", "planned_fixed", "planned_adaptive"];
export type BudgetProfile = { profile_id: string } & Record<string, any>;

function normalize(value: unknown): any {
  if (value === null || typeof value === "string" || typeof value === "boolean") return value;
  if (typeof value === "number") {
    if (!Number.isSafeInteger(value)) throw new Error("INVALID_BUDGET_PROFILE: unsafe integer");
    return Object.is(value, -0) ? 0 : value;
  }
  if (Array.isArray(value)) return value.map(normalize);
  if (value && typeof value === "object" && Object.getPrototypeOf(value) === Object.prototype) {
    return Object.fromEntries(Object.keys(value).sort().map(key => [key, normalize((value as JsonObject)[key])]));
  }
  throw new Error("INVALID_BUDGET_PROFILE: non-JSON value");
}

export function canonicalBudgetJson(value: unknown): string {
  // Do not delegate object ordering to JSON.stringify: it reorders integer-like keys.
  const item = normalize(value);
  const encode = (row: any): string => {
    if (Array.isArray(row)) return `[${row.map(encode).join(",")}]`;
    if (row && typeof row === "object") return `{${Object.keys(row).sort().map(key => `${JSON.stringify(key)}:${encode(row[key])}`).join(",")}}`;
    return JSON.stringify(row);
  };
  return encode(item);
}

export function budgetProfileSha256(value: unknown): string {
  return createHash("sha256").update(canonicalBudgetJson(value)).digest("hex");
}

export function validateBudgetProfile(value: unknown): BudgetProfile {
  const candidate = normalize(value);
  if (!validateSchema(candidate)) throw new Error(`INVALID_BUDGET_PROFILE: ${ajv.errorsText(validateSchema.errors)}`);
  const row = candidate as BudgetProfile;
  for (const [key, number] of Object.entries(row)) {
    if (key !== "profile_id" && !Number.isSafeInteger(number)) throw new Error("INVALID_BUDGET_PROFILE: numeric fields must be safe integers");
  }
  if ((row.initial_query_limit ?? 0) > row.max_search_query_attempts
    || (row.feedback_query_reserve ?? 0) > row.max_search_query_attempts
    || (row.initial_query_limit ?? 0) + (row.feedback_query_reserve ?? 0) > row.max_search_query_attempts) throw new Error("INVALID_BUDGET_PROFILE: initial queries consume feedback reserve");
  if (row.rerank_cap_micro_cny + row.model_trial_cap_micro_cny !== row.task_total_cost_cap_micro_cny) throw new Error("INVALID_BUDGET_PROFILE: task cost does not equal component costs");
  return row;
}

export function loadBudgetProfile(profileId: string): BudgetProfile {
  const catalog = JSON.parse(readFileSync(`${root}/configs/agent_rag/budget-profiles.json`, "utf8"));
  if (typeof profileId !== "string" || !Object.hasOwn(catalog.profiles, profileId)) throw new Error("UNKNOWN_BUDGET_PROFILE");
  const profile = validateBudgetProfile(catalog.profiles[profileId]);
  if (profile.profile_id !== profileId) throw new Error("INVALID_BUDGET_PROFILE: catalog id mismatch");
  return profile;
}

export function validateBudgetConfig(config: Config | JsonObject): BudgetProfile | undefined {
  const id = config.budget_profile_id;
  const mode = config.orchestration_mode;
  const fast = config.fast_path_enabled;
  if (id === undefined) {
    if (mode !== undefined || fast !== undefined || config.budget_profile_sha256 !== undefined) throw new Error("INVALID_BUDGET_CONFIG: orchestration requires a budget profile");
    return undefined;
  }
  const profile = loadBudgetProfile(id);
  if (id !== PLANNED_BUDGET_PROFILE || !modes.includes(mode)) throw new Error("INVALID_BUDGET_CONFIG: orchestration/profile mismatch");
  if (config.retrieval_profile !== "v2-hybrid-flash-v1") throw new Error("INVALID_BUDGET_CONFIG: planned orchestration requires v2");
  if (fast !== undefined && typeof fast !== "boolean") throw new Error("INVALID_BUDGET_CONFIG: fast_path_enabled must be boolean");
  if (!["controlled_agent", "fixed_rag"].includes(config.engine) || config.engine === "fixed_rag" && mode !== "planned_fixed") throw new Error("INVALID_BUDGET_CONFIG: engine/mode mismatch");
  if (config.budget_profile_sha256 !== budgetProfileSha256(profile)) throw new Error("INVALID_BUDGET_CONFIG: budget profile hash mismatch");
  try {
    const effective = validateBudgetProfile(Object.fromEntries(Object.keys(profile).map(key => [key, key === "profile_id" ? id : (config as JsonObject)[key]])));
    for (const [key, limit] of Object.entries(profile)) {
      if (key !== "profile_id" && effective[key] > limit) throw new Error("resource exceeds registered profile");
    }
    for (const key of ["rerank_cap", "model_trial_cap", "task_total_cost_cap"]) {
      const alias = (config as JsonObject)[`${key}_cny`];
      if (alias !== undefined && (typeof alias !== "number" || alias !== effective[`${key}_micro_cny`] / 1_000_000)) throw new Error("currency alias differs from integer cap");
    }
    const external = config.max_external_retrieval_requests ?? effective.rerank_max_requests;
    if (!Number.isSafeInteger(external) || external < 1 || external > effective.rerank_max_requests) throw new Error("external requests exceed rerank budget");
  } catch (error) { throw new Error(`INVALID_BUDGET_CONFIG: ${String(error)}`); }
  return profile;
}

export function budgetCapacityDiagnostic(profile: BudgetProfile, effectiveRerankMaxRequests?: number): JsonObject {
  const row = validateBudgetProfile(profile);
  const effective = effectiveRerankMaxRequests ?? row.rerank_max_requests;
  if (!Number.isSafeInteger(effective) || effective < 1 || effective > row.rerank_max_requests) throw new Error("INVALID_BUDGET_CONFIG: effective rerank limit");
  const theoretical = row.max_search_query_attempts * row.rerank_candidates_per_query * row.rerank_attempts_per_candidate;
  return {theoretical_rerank_requests:theoretical, effective_rerank_max_requests:effective,
    may_stop_before_all_queries:effective < theoretical};
}
