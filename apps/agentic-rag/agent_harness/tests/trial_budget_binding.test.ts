import test from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { TrialBudget, activeTrialBudget } from "../src/trial_budget.js";
import { root } from "../src/contracts.js";
import { loadBudgetProfile, budgetProfileSha256 } from "../src/budget_profile.js";

const profile = loadBudgetProfile("planned-evidence-loop-v1");
const binding = { budget_profile_id: profile.profile_id, budget_profile_sha256: budgetProfileSha256(profile) };
const directory = join(root, ".test-temp-trial"); mkdirSync(directory, { recursive: true });
const path = () => join(mkdtempSync(join(directory, "bound-")), "ledger.json");
const body = JSON.stringify({ model: "deepseek-flash", max_tokens: 1, thinking: { type: "disabled" }, messages: [{ role: "user", content: "a" }] });
const open = (p: string, b: any = binding, cap = 1) => new (TrialBudget as any)(p, cap, "v2-v1", b);

test("bound ledger writes registered identity and enforces twelve including unknown attempts", () => {
  const p = path(), guard = open(p); guard.initializeEmpty();
  assert.equal(JSON.parse(readFileSync(p, "utf8")).budget_profile_id, binding.budget_profile_id);
  for (let i = 0; i < 12; i++) guard.reserve(body);
  assert.throws(() => guard.reserve(body), /BUDGET_EXHAUSTED/);
  assert.equal(open(p).preview(body).fits, false);
  assert.equal(JSON.parse(readFileSync(p, "utf8")).request_reservations.length, 12);
});
test("same cached bound guard revalidates disk and cannot be downgraded to unbound", () => {
  const p = path(), prior = process.env.FINRISK_AGENT_TRIAL_LEDGER; process.env.FINRISK_AGENT_TRIAL_LEDGER = p;
  try {
    const guard = (activeTrialBudget as any)("v2-v1", binding); guard.initializeEmpty(); guard.reserve(body);
    assert.throws(() => new TrialBudget(p, 1, "v2-v1"), /BINDING/);
    const data = JSON.parse(readFileSync(p, "utf8")); data.budget_profile_sha256 = "0".repeat(64); writeFileSync(p, JSON.stringify(data));
    assert.throws(() => (activeTrialBudget as any)("v2-v1", binding).preview(body), /BINDING/);
  } finally { if (prior === undefined) delete process.env.FINRISK_AGENT_TRIAL_LEDGER; else process.env.FINRISK_AGENT_TRIAL_LEDGER = prior; }
});
test("bound settlement merges disk reservations and unknown usage stays reserved", () => {
  const p = path(), a = open(p), id = a.reserve(body), b = open(p); b.reserve(body);
  assert.equal(a.settle(id, { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 }), true);
  const data = JSON.parse(readFileSync(p, "utf8")); assert.equal(data.request_reservations.length, 2);
  assert.equal(data.request_reservations[1].settlement, undefined); assert.ok(data.reserved_cost_micro_cny > 0);
});
test("bound profile hash and cap are verified before a ledger can be created", () => {
  assert.throws(() => open(path(), { ...binding, budget_profile_sha256: "0".repeat(64) }), /BINDING/);
  assert.throws(() => open(path(), binding, 0.5), /BINDING/);
  const p = path(); new TrialBudget(p, 1, "v2-v1").reserve(body);
  assert.throws(() => open(p), /BINDING/);
});

test("cached bound ledger refuses truncation even when totals are internally valid", () => {
  const p = path(), guard = open(p); guard.reserve(body); guard.reserve(body);
  const data = JSON.parse(readFileSync(p, "utf8")); data.request_reservations.pop();
  data.reserved_cost_micro_cny = data.request_reservations[0].reserved_micro_cny; data.reserved_cost_cny = data.reserved_cost_micro_cny / 1e6;
  writeFileSync(p, JSON.stringify(data)); assert.throws(() => guard.preview(body), /ROLLBACK/);
});
test("cached bound ledger refuses rewriting an already settled usage to a lower cost", () => {
  const p = path(), guard = open(p), id = guard.reserve(body);
  guard.settle(id, { prompt_tokens: 10, completion_tokens: 1, total_tokens: 11 });
  const data = JSON.parse(readFileSync(p, "utf8")), s = data.request_reservations[0].settlement;
  s.usage = { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2, prompt_cache_hit_tokens: 0, prompt_cache_miss_tokens: 1 };
  s.cost_micro_cny = 10; s.cost_cny = 0.00001; data.settled_cost_micro_cny = 10; data.settled_cost_cny = 0.00001;
  writeFileSync(p, JSON.stringify(data)); assert.throws(() => guard.preview(body), /ROLLBACK/);
});

test("future fee admission requires enough for act and reassessment together", () => {
  const p = path(), guard = open(p), large = body.replace('"a"', '"' + "x".repeat(30000) + '"');
  const preview = (guard as any).previewSequence;
  assert.equal(typeof preview, "function");
  // Spend 0.88 CNY with verified usage, leaving enough for one bounded body only.
  const id = guard.reserve(body); guard.settle(id, { prompt_tokens: 440000, completion_tokens: 0, total_tokens: 440000 });
  assert.equal(guard.preview(large).fits, true);
  assert.equal(preview.call(guard, [large, large]).fits, false);
  assert.equal(JSON.parse(readFileSync(p, "utf8")).request_reservations.length, 1);
});

test("an explicitly bound tightened cap persists and reopens without increasing authorization", () => {
  const p = path(), tight = { ...binding, model_trial_cap_micro_cny: 500000 };
  const guard = new (TrialBudget as any)(p, 0.5, "v2-v1", tight); guard.reserve(body);
  assert.equal(JSON.parse(readFileSync(p, "utf8")).cap_cny, 0.5);
  assert.equal(new (TrialBudget as any)(p, 0.5, "v2-v1", tight).preview(body).fits, true);
  assert.throws(() => new (TrialBudget as any)(p, 1, "v2-v1", tight), /BINDING/);
});
