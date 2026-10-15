import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import Ajv2020 from "ajv/dist/2020.js";

export const root = fileURLToPath(new URL("../../..", import.meta.url));
export const modelSchema = JSON.parse(readFileSync(`${root}/contracts/agent_rag/model-output.schema.json`, "utf8"));
export const toolSchemas = JSON.parse(readFileSync(`${root}/contracts/agent_rag/tool-parameters.schema.json`, "utf8"));
const profileRules = JSON.parse(readFileSync(`${root}/contracts/agent_rag/tool-contract-profiles.json`, "utf8"));
export function toolSchemasForProfile(profile: string): JsonObject {
  const row = profileRules.profiles[profile];
  if (!row) throw new Error('UNKNOWN_TOOL_CONTRACT_PROFILE');
  const schemas = structuredClone(toolSchemas);
  schemas.search_cases.properties.queries.maxItems = row.max_queries_per_search;
  return schemas;
}
export const defaults: Config = JSON.parse(readFileSync(`${root}/configs/agent_rag/mock-baseline.json`, "utf8"));
export const officialTags: string[] = modelSchema.$defs.tag.enum;
export const MODEL_WIRE_VERSION = "finrisk-model-wire/1.2";
const ajv = new Ajv2020.default({ strict: false, allErrors: true });
const validate = ajv.compile(modelSchema);
const phaseSchemas = new Map<string, JsonObject>();
const phaseValidators = new Map<string, ReturnType<typeof ajv.compile>>();
export function contractForKind(kind: string): JsonObject {
  if (!["analysis", "decision", "final"].includes(kind)) throw new Error("INVALID_OUTPUT_KIND");
  if (phaseSchemas.has(kind)) return phaseSchemas.get(kind)!;
  const definitions: JsonObject = {};
  const visit = (value: unknown): void => {
    if (Array.isArray(value)) { value.forEach(visit); return; }
    if (!value || typeof value !== "object") return;
    const row = value as JsonObject;
    if (typeof row.$ref === "string" && row.$ref.startsWith("#/$defs/")) add(row.$ref.slice(8));
    Object.values(row).forEach(visit);
  };
  const add = (name: string): void => {
    if (definitions[name]) return;
    if (!modelSchema.$defs[name]) throw new Error("UNKNOWN_PROTOCOL_DEFINITION");
    definitions[name] = modelSchema.$defs[name]; visit(definitions[name]);
  };
  add(kind);
  const schema = { $schema: modelSchema.$schema, $ref: `#/$defs/${kind}`, $defs: definitions };
  phaseSchemas.set(kind, schema); return schema;
}
const programValidators = new Map<string, ReturnType<typeof ajv.compile>>();
export function validateProgram(value: unknown, name: string): void {
  let validator = programValidators.get(name);
  if (!validator) {
    validator = ajv.compile(JSON.parse(readFileSync(`${root}/contracts/agent_rag/${name}.schema.json`, "utf8")));
    programValidators.set(name, validator);
  }
  if (!validator(value)) throw new Error(`INVALID_PROGRAM_CONTRACT:${name}:${ajv.errorsText(validator.errors)}`);
}
export type JsonObject = Record<string, any>;
export function validateOutput(value: unknown, kind?: string): asserts value is JsonObject {
  if (kind && (!value || typeof value !== "object" || (value as JsonObject).kind !== kind)) throw new Error(`INVALID_OUTPUT_KIND:${kind}:顶层直接输出kind，不要type或analysis/final外层包装`);
  let validator = validate;
  if (kind) {
    if (!phaseValidators.has(kind)) phaseValidators.set(kind, ajv.compile(contractForKind(kind)));
    validator = phaseValidators.get(kind)!;
  }
  if (!validator(value)) throw new Error(`INVALID_SCHEMA:${(validator.errors ?? []).slice(0, 12).map(e => `${e.instancePath}:${e.message}:${JSON.stringify(e.params)}`).join("; ").slice(0, 1200)}`);
}
const argumentValidators = new Map<string, ReturnType<typeof ajv.compile>>();
export function validateToolArguments(name: string, args: unknown, schemas = toolSchemas): void {
  if (!schemas[name]) throw new Error("UNKNOWN_TOOL");
  const key = `${hash(canonical(schemas[name]))}:${name}`;
  let validator = argumentValidators.get(key);
  if (!validator) { validator = ajv.compile(schemas[name]); argumentValidators.set(key, validator); }
  if (!validator(args)) throw new Error(`INVALID_TOOL_ARGUMENTS:${ajv.errorsText(validator.errors).slice(0, 400)}`);
}
export function hash(value: string): string { return createHash("sha256").update(value).digest("hex"); }
export function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") return `{${Object.keys(value).sort().map(k => `${JSON.stringify(k)}:${canonical((value as JsonObject)[k])}`).join(",")}}`;
  return JSON.stringify(value);
}
export interface Config {
  profile_id: string; execution_mode: "mock" | "live"; retrieval_profile: string;
  engine: "controlled_agent" | "fixed_rag"; max_model_requests: number; max_tool_actions: number;
  max_search_calls: number; max_search_query_attempts: number; max_queries_per_search: number;
  max_external_retrieval_requests: number; task_deadline_seconds: number;
  model_request_timeout_seconds: number; tool_request_timeout_seconds: number;
  max_tool_transport_retries: number; provider_auto_retries: number; max_model_transport_retries: number;
  max_json_repairs_per_task: number; no_progress_limit: number; service_max_input_tokens: number;
  material_max_tokens: number; max_output_tokens_per_request: number; task_token_budget: number;
  context_safety_margin_tokens: number; model_context_window: number; search_return_max_chars: number;
  max_candidate_risk_points: number; max_concurrent_tasks: number;
  request_admission_policy?: string;
}
export interface InputRecord extends JsonObject {
  task_id: string; text: string; source_id: string; text_sha256: string; text_version: string;
  input_sha256: string; user_context: Record<string, string | null>;
  segments: { segment_id: string; start: number; end: number; text: string; source_id: string }[];
}
export interface ToolEnvelope extends JsonObject {
  schema_version: string; status: "ok" | "empty" | "partial" | "error"; data: any;
  meta: { backend_query_attempts: number; cache_hits: number; external_retrieval_requests: number; snapshot_id: string; [key: string]: any };
  error: { code: string; message?: string; retryable?: boolean } | null;
}
export type Phase = "analyzing" | "reviewing" | "finalizing" | "completed" | "partial";
export interface TaskInput {
  input: InputRecord; snapshot: JsonObject; definitions: JsonObject[]; config?: Partial<Config>;
  task_dir?: string; mock_scenario?: "generic" | "gift-example";
  execution_contract?: { model_wire_sha256: string; criteria_sha256: string; prompt_sha256: string; tool_contract_sha256?: string; request_admission_policy_sha256?: string };
}
