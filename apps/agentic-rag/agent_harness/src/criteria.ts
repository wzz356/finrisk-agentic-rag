import { readFileSync } from "node:fs";
import { canonical, hash, officialTags, root, type JsonObject } from "./contracts.js";

const criteriaBytes = readFileSync(`${root}/contracts/agent_rag/risk-criteria.json`, "utf8");
export const criteriaByteHash = hash(criteriaBytes);
export const criteria = JSON.parse(criteriaBytes);
export const criteriaHash = hash(canonical(criteria));
const byTag = new Map<string, JsonObject>(criteria.tags.map((row: JsonObject) => [row.tag, row]));
export function checkCriteriaBinding(definitions: JsonObject[], dictionaryHash: string): void {
  if (criteria.dictionary_sha256 !== dictionaryHash || byTag.size !== officialTags.length || definitions.length !== officialTags.length
    || officialTags.some(tag => !byTag.has(tag)) || definitions.some(d => byTag.get(d.risk_tag)?.definition !== d.description)) throw new Error("CRITERIA_DICTIONARY_MISMATCH");
}
export function criteriaFor(tag: string): JsonObject[] {
  const row = byTag.get(tag);
  if (!row) throw new Error("INVALID_RISK_TAG");
  return row.criteria;
}
export function criteriaProjection(tags: string[]): JsonObject[] {
  return [...new Set(tags)].map(tag => ({ tag, definition: byTag.get(tag)?.definition, criteria: criteriaFor(tag).map(c => ({ id: c.id, statement: c.statement })) }));
}
