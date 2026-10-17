import test from "node:test";
import assert from "node:assert/strict";
import { parseModelJson, outputDiagnostic, evaluationFormatRepairable } from "../src/model_json.js";

test("strict JSON receiver accepts complete nested objects without rewriting", () => {
  const text = ' {"points":[{"case_matches":[]}],"continuation":{"gaps":[]}} ';
  assert.deepEqual(parseModelJson(text), JSON.parse(text));
});

test("missing point and array closures expose a safe syntax diagnostic", () => {
  const text = '{"points":[{"case_matches":[],"continuation":{"gaps":[]}}';
  try { parseModelJson(text); assert.fail("must reject malformed JSON"); }
  catch (error) {
    const diagnostic = outputDiagnostic(error, text);
    assert.equal(diagnostic.category, "syntax"); assert.equal(diagnostic.code, "MODEL_JSON_SYNTAX");
    assert.equal(diagnostic.position, text.length);
    assert.equal(evaluationFormatRepairable(error), true);
    assert.ok(!String(error).includes('"case_matches"'));
  }
});

test("arrays, null and scalar JSON are schema failures rather than accepted outputs", () => {
  for (const text of ['[]', 'null', '1', '"value"']) {
    try { parseModelJson(text); assert.fail("root must be an object"); }
    catch (error) { assert.equal(outputDiagnostic(error, text).category, "schema"); assert.equal(evaluationFormatRepairable(error), true); }
  }
});

test("semantic and evidence failures are not eligible for evaluation format repair", () => {
  for (const code of ['EVALUATION_INPUT_GAP_WITHOUT_UNKNOWN', 'EVALUATION_CRITERIA_INCOMPLETE', 'INVALID_CASE_REF', 'INVALID_INPUT_REFERENCE']) {
    const error = new Error(code);
    assert.equal(outputDiagnostic(error, "{}").category, "semantic");
    assert.equal(evaluationFormatRepairable(error), false);
  }
  assert.equal(evaluationFormatRepairable(new Error('INVALID_EVALUATION_ENVELOPE:bad fields')), true);
});
