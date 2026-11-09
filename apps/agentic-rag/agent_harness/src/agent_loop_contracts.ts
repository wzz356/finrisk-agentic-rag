import { readFileSync } from "node:fs";
import Ajv2020, { type ErrorObject } from "ajv/dist/2020.js";
import { canonical, hash, officialTags, root, type JsonObject } from "./contracts.js";
import { criteriaByteHash, criteriaFor, criteriaHash } from "./criteria.js";
import { ModelJsonError, ModelSchemaError, parseModelJson, type SchemaIssue } from "./model_json.js";

export const AGENT_LOOP_VERSION = "finrisk-agent-loop/1.0";
export type LoopOutputKind = "plan" | "assess" | "one_shot";
export interface LoopValidationContext {
  expected_assessment_id?: string;
  expected_state_revision?: number;
  expected_risk_point_id?: string;
  candidate_tags?: readonly string[];
  visible_case_refs?: readonly string[];
}

const wireBytes = readFileSync(`${root}/contracts/agent_rag/model-wire.schema.json`, "utf8");
const wire = JSON.parse(wireBytes);
const loopBytes = readFileSync(`${root}/contracts/agent_rag/agent-loop.schema.json`, "utf8");
const assessmentBytes = readFileSync(`${root}/contracts/agent_rag/agent-assessment.schema.json`, "utf8");
export const loopSchemaHash = hash(loopBytes);
export const agentAssessmentSchemaHash = hash(assessmentBytes);
const kinds: readonly LoopOutputKind[] = ["plan", "assess", "one_shot"];

// Bind external definitions from the existing wire file, without maintaining a
// second tag dictionary or a copy of its reference/fact/condition constraints.
function bindSchema(raw: JsonObject): JsonObject {
  const definitions: JsonObject = {};
  const resolveWire = (pointer: string): JsonObject => {
    let value = wire;
    for (const key of pointer.split("/").slice(1)) value = value[key];
    return value;
  };
  const bind = (value: any): any => {
    if (Array.isArray(value)) return value.map(bind);
    if (!value || typeof value !== "object") return value;
    const prefix = `${wire.$id}#`;
    if (typeof value.$ref === "string" && value.$ref.startsWith(prefix)) {
      const pointer = value.$ref.slice(prefix.length);
      const bound = bind(resolveWire(pointer));
      return pointer === "/$defs/tag" ? { ...bound, enum: [...officialTags] } : bound;
    }
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, bind(item)]));
  };
  const visit = (value: any): void => {
    if (Array.isArray(value)) { value.forEach(visit); return; }
    if (!value || typeof value !== "object") return;
    if (typeof value.$ref === "string" && value.$ref.startsWith("#/$defs/")) add(value.$ref.slice(8));
    Object.values(value).forEach(visit);
  };
  const add = (name: string): void => {
    if (definitions[name]) return;
    definitions[name] = bind(raw.$defs[name] ?? wire.$defs[name]);
    visit(definitions[name]);
  };
  Object.keys(raw.$defs).forEach(add);
  return { ...raw, $defs: definitions };
}
const loopSchema = bindSchema(JSON.parse(loopBytes));
const assessmentSchema = bindSchema(JSON.parse(assessmentBytes));
const phaseSchemas = new Map<LoopOutputKind, JsonObject>();

const categories = {
  LOOP_JSON_SYNTAX: "syntax", LOOP_SCHEMA: "schema", LOOP_INVALID_KIND: "schema",
  LOOP_ASSESSMENT_CONTEXT_REQUIRED: "semantic", LOOP_ASSESSMENT_ID_MISMATCH: "semantic",
  LOOP_STATE_REVISION_MISMATCH: "semantic", LOOP_CANDIDATE_TAGS_MISMATCH: "semantic",
  LOOP_CRITERIA_MISMATCH: "semantic", LOOP_SEMANTIC_ERROR: "semantic",
  LOOP_TOOL_FORMAT: "tool_format", LOOP_INVALID_CASE_REF: "reference",
  LOOP_INVALID_INPUT_REFERENCE: "reference", LOOP_INVALID_CONTEXT_REFERENCE: "reference",
  LOOP_RISK_POINT_MISMATCH: "reference",
} as const;
export type LoopContractErrorCode = keyof typeof categories;
const trustedProperties = new Set<string>();
function collectProperties(value: any): void {
  if (Array.isArray(value)) { value.forEach(collectProperties); return; }
  if (!value || typeof value !== "object") return;
  if (value.properties) Object.keys(value.properties).forEach(name => trustedProperties.add(name));
  Object.values(value).forEach(collectProperties);
}
collectProperties(loopSchema); collectProperties(assessmentSchema);

export class LoopContractError extends Error {
  public readonly code: LoopContractErrorCode;
  public readonly category: typeof categories[LoopContractErrorCode];
  public readonly schemaIssues: readonly Readonly<SchemaIssue>[];
  constructor(code: LoopContractErrorCode, errors?: readonly ErrorObject[] | null, public readonly position: number | null = null) {
    const safeCode = Object.hasOwn(categories, code) ? code : "LOOP_SEMANTIC_ERROR";
    super(safeCode); this.name = "LoopContractError"; this.code = safeCode; this.category = categories[safeCode];
    this.schemaIssues = Object.freeze(new ModelSchemaError(errors).schemaIssues.map(issue => {
      const safe = { ...issue };
      if (safe.instancePath !== "[redacted]" && safe.instancePath.split("/").slice(1).some(part => !/^\d+$/.test(part) && !trustedProperties.has(part))) safe.instancePath = "[redacted]";
      if (safe.additionalProperty && !trustedProperties.has(safe.additionalProperty)) safe.additionalProperty = "[redacted]";
      if (safe.missingProperty && !trustedProperties.has(safe.missingProperty)) safe.missingProperty = "[redacted]";
      return Object.freeze(safe);
    }));
  }
}

export function loopContract(kind: LoopOutputKind): JsonObject {
  if (!kinds.includes(kind)) throw new LoopContractError("LOOP_INVALID_KIND");
  if (!phaseSchemas.has(kind)) {
    const definitions: JsonObject = {};
    const visit = (value: any): void => {
      if (Array.isArray(value)) { value.forEach(visit); return; }
      if (!value || typeof value !== "object") return;
      if (typeof value.$ref === "string" && value.$ref.startsWith("#/$defs/")) add(value.$ref.slice(8));
      Object.values(value).forEach(visit);
    };
    const add = (name: string): void => {
      if (definitions[name]) return;
      definitions[name] = loopSchema.$defs[name]; visit(definitions[name]);
    };
    add(kind);
    phaseSchemas.set(kind, { $schema: loopSchema.$schema, $ref: `#/$defs/${kind}`, $defs: definitions });
  }
  return structuredClone(phaseSchemas.get(kind)!);
}
export function loopContractHash(kind: LoopOutputKind): string { return hash(canonical(loopContract(kind))); }
export function agentAssessmentContract(): JsonObject { return structuredClone(assessmentSchema); }
export function loopContractInventory(): JsonObject {
  return { protocol_version: AGENT_LOOP_VERSION, loop_schema_sha256: loopSchemaHash,
    assessment_schema_sha256: agentAssessmentSchemaHash, wire_schema_sha256: hash(wireBytes),
    criteria_sha256: criteriaByteHash, criteria_content_sha256: criteriaHash,
    official_tags_sha256: hash(canonical(officialTags)),
    stages: Object.fromEntries(kinds.map(kind => [kind, { sha256: loopContractHash(kind),
      fields: Object.keys(loopSchema.$defs[kind].properties),
      point_fields: kind === "assess" ? [] : Object.keys(loopSchema.$defs[kind === "plan" ? "planPoint" : "oneShotPoint"].properties) }])) };
}

const ajv = new Ajv2020.default({ strict: false, allErrors: true });
const validators = new Map<LoopOutputKind, ReturnType<typeof ajv.compile>>();
const validateSidecar = ajv.compile(assessmentSchema);
type AssessmentContext = Required<LoopValidationContext>;
function requireAssessmentContext(context?: LoopValidationContext): asserts context is AssessmentContext {
  if (!context || typeof context.expected_assessment_id !== "string" || !/^AS[0-9]+$/.test(context.expected_assessment_id)
    || !Number.isSafeInteger(context.expected_state_revision) || context.expected_state_revision! < 0
    || typeof context.expected_risk_point_id !== "string" || !/^RP[0-9]+$/.test(context.expected_risk_point_id)
    || !Array.isArray(context.candidate_tags) || context.candidate_tags.length < 1 || context.candidate_tags.length > 3
    || new Set(context.candidate_tags).size !== context.candidate_tags.length || context.candidate_tags.some(tag => !officialTags.includes(tag))
    || !Array.isArray(context.visible_case_refs) || context.visible_case_refs.some(value => typeof value !== "string" || !/^C[0-9a-f]{20}$/.test(value))) {
    throw new LoopContractError("LOOP_ASSESSMENT_CONTEXT_REQUIRED");
  }
}
function validateChecks(checks: JsonObject[], context?: LoopValidationContext, expectedTags?: readonly string[]): void {
  const tags = checks.map(check => check.tag as string);
  if (new Set(tags).size !== tags.length || (expectedTags && (tags.length !== expectedTags.length || expectedTags.some(tag => !tags.includes(tag))))) throw new LoopContractError("LOOP_CANDIDATE_TAGS_MISMATCH");
  for (const check of checks) {
    const expected = criteriaFor(check.tag).map(criterion => criterion.id as string);
    const actual = check.conditions.map((condition: JsonObject) => condition.criterion_id as string);
    if (actual.length !== expected.length || new Set(actual).size !== actual.length || expected.some(id => !actual.includes(id))) throw new LoopContractError("LOOP_CRITERIA_MISMATCH");
    for (const comparison of check.case_assessments) {
      if (!context?.visible_case_refs?.includes(comparison.case_ref)) throw new LoopContractError("LOOP_INVALID_CASE_REF");
    }
  }
}

// Structural validity never proves an input quote exists. Existing normalization
// and TaskState must still verify input/context references before acceptance.
export function validateLoopOutput(value: unknown, kind: LoopOutputKind, context?: LoopValidationContext): asserts value is JsonObject {
  if (!kinds.includes(kind)) throw new LoopContractError("LOOP_INVALID_KIND");
  if (!validators.has(kind)) validators.set(kind, ajv.compile(loopContract(kind)));
  const validate = validators.get(kind)!;
  if (!validate(value)) throw new LoopContractError("LOOP_SCHEMA", validate.errors);
  const output = value as JsonObject;
  if (kind === "assess") {
    requireAssessmentContext(context);
    if (output.assessment_id !== context.expected_assessment_id) throw new LoopContractError("LOOP_ASSESSMENT_ID_MISMATCH");
    if (output.state_revision !== context.expected_state_revision) throw new LoopContractError("LOOP_STATE_REVISION_MISMATCH");
    if (output.risk_point_id !== context.expected_risk_point_id) throw new LoopContractError("LOOP_RISK_POINT_MISMATCH");
    validateChecks(output.tag_checks, context, context.candidate_tags);
  } else if (kind === "one_shot") {
    for (const point of output.points) validateChecks(point.tag_checks, context);
  }
}
export function validateAgentAssessment(value: unknown): asserts value is JsonObject {
  if (!validateSidecar(value)) throw new LoopContractError("LOOP_SCHEMA", validateSidecar.errors);
}
export function parseLoopOutput(text: string, kind: LoopOutputKind, context?: LoopValidationContext): JsonObject {
  let value: JsonObject;
  try { value = parseModelJson(text); }
  catch (error) {
    if (error instanceof ModelJsonError && error.category === "syntax") throw new LoopContractError("LOOP_JSON_SYNTAX", undefined, error.position);
    throw new LoopContractError("LOOP_SCHEMA");
  }
  validateLoopOutput(value, kind, context); return value;
}
/** A normalizer may mark only known model-owned data errors, never evidence/identity/program errors. */
export class LoopModelDataError extends LoopContractError {
  constructor(readonly sourceCode: string) {
    super(sourceCode === "INVALID_INPUT_REFERENCE" || sourceCode === "INVALID_SEGMENT_REFERENCE" ? "LOOP_INVALID_INPUT_REFERENCE"
      : sourceCode === "INVALID_CONTEXT_REFERENCE" ? "LOOP_INVALID_CONTEXT_REFERENCE" : "LOOP_SEMANTIC_ERROR");
    this.name = "LoopModelDataError"; this.message = sourceCode;
  }
}
export function isolatableLoopOutput(error: unknown): boolean {
  return error instanceof LoopModelDataError || error instanceof LoopContractError && ["LOOP_JSON_SYNTAX", "LOOP_SCHEMA",
    "LOOP_CANDIDATE_TAGS_MISMATCH", "LOOP_CRITERIA_MISMATCH", "LOOP_INVALID_CASE_REF"].includes(error.code);
}
export function loopFormatRepairable(error: unknown): boolean {
  return error instanceof LoopContractError && (error.category === "syntax" || error.category === "schema");
}
export function loopOutputDiagnostic(error: unknown, text: string): JsonObject {
  const safe = error instanceof LoopContractError ? error : new LoopContractError("LOOP_SEMANTIC_ERROR");
  return { category: safe.category, code: safe.code, position: safe.position,
    ...(safe.schemaIssues.length ? { schema_issues: safe.schemaIssues } : {}),
    reply_sha256: hash(text), reply_chars: text.length };
}
