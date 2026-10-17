import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { root, validateOutput, contractForKind } from "../src/contracts.js";

test("phase contract feedback contains only selected phase and actual allowed enum values", () => {
  const value = JSON.parse(readFileSync(`${root}/contracts/agent_rag/fixtures/valid_analysis.json`, "utf8"));
  value.document_context.scene_status = "declared";
  assert.throws(() => validateOutput(value, "analysis"), error => {
    const message = String(error);
    assert.match(message, /scene_status/); assert.match(message, /provided/);
    assert.ok(!message.includes("risk_point_id") && !message.includes("gap_updates"));
    return true;
  });
  const schema = contractForKind("analysis");
  assert.equal(schema.$ref, "#/$defs/analysis");
  assert.equal(schema.$defs.analysis.properties.document_context.properties.scene_status.enum[0], "provided");
  assert.ok(!schema.$defs.final && !schema.$defs.decision);
});

for (const name of readdirSync(`${root}/contracts/agent_rag/fixtures`).filter(n => n.endsWith(".json"))) {
  test(`cross-language schema ${name}`, () => {
    const value = JSON.parse(readFileSync(`${root}/contracts/agent_rag/fixtures/${name}`, "utf8").replace(/^\uFEFF/, ""));
    if (name.startsWith("invalid_")) assert.throws(() => validateOutput(value));
    else validateOutput(value, value.kind);
  });
}
