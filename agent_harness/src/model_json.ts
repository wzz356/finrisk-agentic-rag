import { hash, type JsonObject } from "./contracts.js";

export class ModelJsonError extends Error {
  constructor(public code: string, public category: "syntax" | "schema", public position: number | null = null) { super(code); }
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
    : /^(INVALID_EVALUATION_ENVELOPE|INVALID_SCHEMA|INVALID_OUTPUT_KIND):/.test(message) ? "schema" : "semantic";
  return { category, code: error instanceof ModelJsonError ? error.code : message.split(":", 1)[0],
    position: error instanceof ModelJsonError ? error.position : null, reply_sha256: hash(text), reply_chars: text.length };
}
export function evaluationFormatRepairable(error: unknown): boolean {
  return error instanceof ModelJsonError || String((error as Error)?.message ?? "").startsWith("INVALID_EVALUATION_ENVELOPE:");
}
