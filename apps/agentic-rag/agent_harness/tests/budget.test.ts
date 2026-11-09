import test from "node:test";
import assert from "node:assert/strict";
import { Budget } from "../src/budget.js";
import { defaults } from "../src/contracts.js";

test("H03/H22: reserve final request, then stop at physical generation limit", () => {
  const budget = new Budget({ ...defaults, max_model_requests: 2 });
  budget.settle(budget.reserveModel(100, false), 80);
  assert.equal(budget.shouldFinalize(100), true);
  budget.settle(budget.reserveModel(100, true), 80);
  assert.throws(() => budget.reserveModel(100, true), /budget_exhausted/);
  assert.equal(budget.summary().model_request_units, 2);
});
test("H18/H23: unknown usage stays reserved, duplicate events do not double settle", () => {
  const budget = new Budget(defaults);
  const id = budget.reserveModel(100, false);
  budget.settle(id);
  assert.equal(budget.summary().unknown_usage_requests, 1);
  assert.equal(budget.summary().reserved_tokens, 100 + defaults.max_output_tokens_per_request);
  budget.settle(id, 100);
  budget.settle(id, 100);
  assert.equal(budget.summary().settled_tokens, 100);
  assert.equal(budget.summary().reserved_tokens, 0);
});
test("H24: tool/cache/retry actions share task budget, repeated event is idempotent", () => {
  const budget = new Budget({ ...defaults, max_tool_actions: 2, max_search_calls: 2 });
  budget.chargeTool("t1", "search_cases"); budget.chargeTool("t1", "search_cases");
  budget.chargeTool("t2", "read_source");
  assert.throws(() => budget.chargeTool("t3", "get_risk_definition"), /budget_exhausted/);
  assert.equal(budget.summary().tool_actions, 2);
});
test("deadline uses monotonic clock and never resets", () => {
  let now = 0; const budget = new Budget({ ...defaults, task_deadline_seconds: 1 }, () => now);
  now = 1001;
  assert.equal(budget.remainingMs(), 0);
  assert.throws(() => budget.reserveModel(10, true), /deadline_exceeded/);
});
test("final input/output reservation forces earlier finalization", () => {
  const budget = new Budget({ ...defaults, task_token_budget: 9000 });
  assert.equal(budget.shouldFinalize(1000), true);
  assert.throws(() => budget.reserveModel(1000, false), /budget_exhausted/);
});
