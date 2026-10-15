import test from "node:test";
import assert from "node:assert/strict";
import { loadDeepSeekKey } from "../src/deepseek_gateway.js";

test("public package requires an explicitly supplied key and has no personal file fallback", () => {
  const names = ["DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY_FILE"];
  const previous = names.map(name => process.env[name]);
  try {
    names.forEach(name => delete process.env[name]);
    assert.throws(() => loadDeepSeekKey(), /INVALID_DEEPSEEK_KEY/);
    process.env.DEEPSEEK_API_KEY = ["sk", "synthetic", "test", "key"].join("-");
    assert.equal(loadDeepSeekKey(), ["sk", "synthetic", "test", "key"].join("-"));
  } finally {
    names.forEach((name, i) => {
      if (previous[i] === undefined) delete process.env[name];
      else process.env[name] = previous[i];
    });
  }
});
