import type { ErrorObject } from "ajv";
import { hash, type JsonObject } from "./contracts.js";

export class ModelJsonError extends Error {
  constructor(public code: string, public category: "syntax" | "schema", public position: number | null = null) { super(code); }
}

// Only trusted validator structure may enter a repair system message. Ajv messages,
// schema paths, invalid values and arbitrary Error text are never copied here.
export const schemaIssuePolicy = Object.freeze({ maxIssues: 3, maxPathChars: 160, maxPropertyChars: 64 });
export type SchemaIssue = { instancePath: string; keyword: string; additionalProperty?: string; missingProperty?: string };
const schemaKeywords = new Set(["additionalProperties", "required", "type", "enum", "const", "minItems", "maxItems", "minLength", "maxLength", "pattern", "oneOf", "anyOf", "allOf", "not", "if", "then", "else", "uniqueItems", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minProperties", "maxProperties", "dependentRequired", "contains", "minContains", "maxContains", "unevaluatedProperties", "propertyNames", "format"]);
function safeProperty(value: unknown): string {
  return typeof value === "string" && value.length <= schemaIssuePolicy.maxPropertyChars && /^[A-Za-z_][A-Za-z0-9_]*$/.test(value) ? value : "[redacted]";
}
function safeInstancePath(value: string): string {
  return value.length <= schemaIssuePolicy.maxPathChars && /^(?:\/(?:[A-Za-z_][A-Za-z0-9_]{0,63}|[0-9]{1,10}))*$/.test(value) ? value : "[redacted]";
}
export class ModelSchemaError extends ModelJsonError {
  public readonly schemaIssues: readonly Readonly<SchemaIssue>[];
  constructor(errors: readonly ErrorObject[] | null | undefined) {
    super("INVALID_EVALUATION_ENVELOPE", "schema");
    this.schemaIssues = Object.freeze((errors ?? []).slice(0, schemaIssuePolicy.maxIssues).map(error => {
      const issue: SchemaIssue = { instancePath: safeInstancePath(error.instancePath), keyword: schemaKeywords.has(error.keyword) ? error.keyword : "[redacted]" };
      if (error.keyword === "additionalProperties") issue.additionalProperty = safeProperty(error.params.additionalProperty);
      if (error.keyword === "required") issue.missingProperty = safeProperty(error.params.missingProperty);
      return Object.freeze(issue);
    }));
  }
}

// Keep parser excerpts out of operational errors; the original reply is already
// preserved in the private model transcript, without any automatic rewriting.
export function syntaxPosition(error: unknown, text: string): number | null {
  const message = String((error as Error)?.message ?? "");
  const match = /(?:position|char) (\d+)/.exec(message);
  if (match) return Number(match[1]);
  return /end of JSON/i.test(message) ? text.length : null;
}
export function parseModelJson(text: string): JsonObject {
  let value: unknown;
  try { value = JSON.parse(text); }
  catch (error) { throw new ModelJsonError("MODEL_JSON_SYNTAX", "syntax", syntaxPosition(error, text)); }
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new ModelJsonError("MODEL_JSON_ROOT_NOT_OBJECT", "schema");
  return value as JsonObject;
}
export function outputDiagnostic(error: unknown, text: string): JsonObject {
  const message = String((error as Error)?.message ?? "");
  const category = error instanceof ModelJsonError ? error.category
    : /^(INVALID_SCHEMA|INVALID_OUTPUT_KIND):/.test(message) ? "schema" : "semantic";
  return { category, code: error instanceof ModelJsonError ? error.code : message.split(":", 1)[0],
    position: error instanceof ModelJsonError ? error.position : null, ...(error instanceof ModelSchemaError ? { schema_issues: error.schemaIssues } : {}), reply_sha256: hash(text), reply_chars: text.length };
}
export function evaluationFormatRepairable(error: unknown): boolean {
  return error instanceof ModelJsonError;
}
