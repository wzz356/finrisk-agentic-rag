import test from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { TrialBudget } from "../src/trial_budget.js";
import { root } from "../src/contracts.js";

const tempRoot = join(root, ".test-temp-trial");
mkdirSync(tempRoot, { recursive: true });

function payload(text = "测试材料", max_tokens = 4096): string {
  return JSON.stringify({ model: "deepseek-flash", thinking: { type: "disabled" }, max_tokens, messages: [{ role: "user", content: text }] });
}
function ledgerPath(): string { return join(mkdtempSync(join(tempRoot, "finrisk-trial-")), "ledger.json"); }
function saved(path: string): any { return JSON.parse(readFileSync(path, "utf8")); }
function settle(guard: TrialBudget, id: unknown, usage?: unknown): boolean {
  return guard.settle(id as number, usage);
}
const usage = { prompt_tokens: 100, completion_tokens: 40, total_tokens: 140, prompt_cache_hit_tokens: 60, prompt_cache_miss_tokens: 40 };
test("paid trial reserves peak CNY bound before egress and persists across restart", () => {
  const path = join(mkdtempSync(join(tempRoot, "finrisk-trial-")), "ledger.json");
  const guard = new TrialBudget(path, 1);
  const body = payload();
  guard.reserve(body);
  const saved = JSON.parse(readFileSync(path, "utf8"));
  assert.equal(saved.request_reservations.length, 1);
  assert.ok(saved.reserved_cost_cny > 0);
  const restarted = new TrialBudget(path, 1);
  restarted.reserve(body);
  assert.equal(JSON.parse(readFileSync(path, "utf8")).request_reservations.length, 2);
  assert.ok(!readFileSync(path, "utf8").includes("测试材料"));
});
test("paid trial blocks next request when whole response reservation exceeds 1 CNY", () => {
  const path = join(mkdtempSync(join(tempRoot, "finrisk-trial-")), "ledger.json");
  const guard = new TrialBudget(path, 1);
  const body = payload("A".repeat(22000));
  let count = 0;
  while (true) {
    try { guard.reserve(body); count++; } catch (error) { assert.match(String(error), /TRIAL_CNY_BUDGET_EXHAUSTED/); break; }
  }
  assert.ok(count < 18);
  const saved = JSON.parse(readFileSync(path, "utf8"));
  assert.equal(saved.request_reservations.length, count);
  assert.ok(saved.reserved_cost_cny <= 1);
});
test("paid trial rejects oversized body and unsupported output/thinking/model", () => {
  const path = join(mkdtempSync(join(tempRoot, "finrisk-trial-")), "ledger.json");
  const guard = new TrialBudget(path, 1);
  assert.throws(() => guard.reserve(payload("A".repeat(25000))), /TRIAL_REQUEST_BODY_TOO_LARGE/);
  assert.throws(() => guard.reserve(payload("a", 4097)), /TRIAL_PAYLOAD_NOT_ADMITTED/);
  assert.throws(() => guard.reserve(JSON.stringify({ model: "other", max_tokens: 4096, thinking: { type: "disabled" } })), /TRIAL_PAYLOAD_NOT_ADMITTED/);
  assert.throws(() => guard.reserve(JSON.stringify({ model: "deepseek-flash", max_tokens: 4096, thinking: { type: "enabled" } })), /TRIAL_PAYLOAD_NOT_ADMITTED/);
});

test("reservations have stable cumulative IDs across restart", () => {
  const path = ledgerPath();
  assert.equal(new TrialBudget(path, 1).reserve(payload()), 1);
  assert.equal(new TrialBudget(path, 1).reserve(payload()), 2);
  assert.deepEqual(saved(path).request_reservations.map((row: any) => row.ordinal), [1, 2]);
});

test("known usage settles frozen peak cache prices and releases excess reservation", () => {
  const path = ledgerPath(); const guard = new TrialBudget(path, 1);
  const id = guard.reserve(payload());
  assert.equal(settle(guard, id, usage), true);
  const ledger = saved(path);
  assert.equal(ledger.schema_version, "finrisk-paid-trial/1.1");
  assert.equal(ledger.peak_cny_per_million.input_cache_hit, 0.04);
  assert.equal(ledger.settled_cost_micro_cny, 403);
  assert.equal(ledger.settled_cost_cny, 0.000403);
  assert.equal(ledger.reserved_cost_micro_cny, 0);
  assert.equal(ledger.request_reservations[0].settlement.cost_micro_cny, 403);
  assert.deepEqual(ledger.request_reservations[0].settlement.usage, usage);
  const after = readFileSync(path, "utf8");
  assert.equal(settle(guard, id, { ...usage, completion_tokens: 0, total_tokens: 100 }), true);
  assert.equal(settle(new TrialBudget(path, 1), id, undefined), true);
  assert.equal(readFileSync(path, "utf8"), after);
});

test("absent cache breakdown uses full prompt miss price and cache fractions round up", () => {
  const path = ledgerPath(); const guard = new TrialBudget(path, 1);
  settle(guard, guard.reserve(payload()), { prompt_tokens: 100, completion_tokens: 40, total_tokens: 140 });
  assert.equal(saved(path).settled_cost_micro_cny, 520);
  settle(guard, guard.reserve(payload()), { prompt_tokens: 1, completion_tokens: 0, total_tokens: 1, prompt_cache_hit_tokens: 1, prompt_cache_miss_tokens: 0 });
  assert.equal(saved(path).settled_cost_micro_cny, 521);
});

test("missing invalid and inconsistent usage retain the full reservation", () => {
  const path = ledgerPath(); const guard = new TrialBudget(path, 1);
  const id = guard.reserve(payload()); const before = readFileSync(path, "utf8");
  const invalid = [undefined, null, [], {}, { ...usage, prompt_tokens: undefined },
    { ...usage, prompt_tokens: -1 }, { ...usage, completion_tokens: 0.5 },
    { ...usage, total_tokens: 141 }, { ...usage, total_tokens: "140" },
    { ...usage, prompt_cache_hit_tokens: -1 }, { ...usage, prompt_cache_hit_tokens: 61 },
    { ...usage, prompt_cache_miss_tokens: undefined }, { ...usage, prompt_cache_hit_tokens: null },
    { prompt_tokens: Number.MAX_SAFE_INTEGER + 1, completion_tokens: 0, total_tokens: Number.MAX_SAFE_INTEGER + 1 }];
  for (const value of invalid) {
    assert.equal(settle(guard, id, value), false);
    assert.equal(readFileSync(path, "utf8"), before);
  }
  assert.equal(settle(guard, 999, usage), false);
  assert.equal(settle(guard, "1", usage), false);
  assert.equal(settle(new TrialBudget(path, 1), id, usage), true);
});

test("CNY admission counts settled spend plus every outstanding reservation", () => {
  const path = ledgerPath(); const guard = new TrialBudget(path, 0.05);
  const id = guard.reserve(payload());
  assert.throws(() => guard.reserve(payload()), /TRIAL_CNY_BUDGET_EXHAUSTED/);
  settle(guard, id, usage);
  guard.reserve(payload());
  const ledger = saved(path);
  assert.equal(ledger.settled_cost_micro_cny, 403);
  assert.ok(ledger.reserved_cost_micro_cny > 0);
  assert.throws(() => new TrialBudget(path, 0.05).reserve(payload()), /TRIAL_CNY_BUDGET_EXHAUSTED/);
});

test("zero usage refunds reservation but never resets the eighteen request authorization", () => {
  const path = ledgerPath(); let guard = new TrialBudget(path, 1);
  for (let i = 1; i <= 18; i++) {
    const id = guard.reserve(payload());
    assert.equal(id, i);
    assert.equal(settle(guard, id, { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }), true);
    if (i === 9) guard = new TrialBudget(path, 1);
  }
  assert.equal(saved(path).reserved_cost_micro_cny, 0);
  assert.equal(saved(path).settled_cost_micro_cny, 0);
  assert.throws(() => new TrialBudget(path, 1).reserve(payload()), /TRIAL_CNY_BUDGET_EXHAUSTED/);
});

test("valid unexpectedly large usage records overspend without negative reservation or clamping", () => {
  const path = ledgerPath(); const guard = new TrialBudget(path, 1);
  const id = guard.reserve(payload());
  assert.equal(settle(guard, id, { prompt_tokens: 100, completion_tokens: 200000, total_tokens: 200100 }), true);
  assert.equal(saved(path).settled_cost_micro_cny, 1600200);
  assert.equal(saved(path).reserved_cost_micro_cny, 0);
  assert.throws(() => new TrialBudget(path, 1).reserve(payload()), /TRIAL_CNY_BUDGET_EXHAUSTED/);
});

test("usage cost arithmetic overflow retains reservation and persistently blocks further requests", () => {
  const path = ledgerPath(); const guard = new TrialBudget(path, 1);
  const id = guard.reserve(payload()); const before = saved(path).reserved_cost_micro_cny;
  assert.equal(settle(guard, id, { prompt_tokens: 0, completion_tokens: Number.MAX_SAFE_INTEGER, total_tokens: Number.MAX_SAFE_INTEGER }), false);
  assert.equal(saved(path).reserved_cost_micro_cny, before);
  assert.equal(saved(path).settled_cost_micro_cny, 0);
  assert.equal(saved(path).budget_blocked_due_to_usage_overflow, true);
  const after = readFileSync(path, "utf8");
  assert.equal(settle(new TrialBudget(path, 1), id, usage), false);
  assert.equal(readFileSync(path, "utf8"), after);
  assert.throws(() => new TrialBudget(path, 1).reserve(payload()), /TRIAL_CNY_BUDGET_EXHAUSTED/);
});

function legacyLedger(count: number): any {
  const bodyBytes = Buffer.byteLength(payload(), "utf8"); const reserved = (bodyBytes + 512) * 2 + 4096 * 8;
  return { schema_version: "finrisk-paid-trial/1.0", model: "deepseek-flash", cap_cny: 1,
    price_source: "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/", price_checked_date: "2026-10-14",
    peak_cny_per_million: { input_cache_miss: 2, output: 8 }, prompt_overhead_token_allowance: 512,
    reserved_cost_micro_cny: reserved * count, reserved_cost_cny: reserved * count / 1_000_000,
    request_reservations: Array.from({ length: count }, (_, i) => ({ ordinal: i + 1,
      reserved_at: "2026-10-14T00:00:00.000Z", body_bytes: bodyBytes, max_output_tokens: 4096,
      reserved_micro_cny: reserved, reserved_cny: reserved / 1_000_000 })),
    reservation_kind: "UTF8_request_body_bytes_plus512_prompt_bound; max_tokens_output; peak_price" };
}

test("v1.0 migration leaves sixteen historical reservations intact and retains authorization count", () => {
  const path = ledgerPath(); const legacy = legacyLedger(16);
  writeFileSync(path, JSON.stringify(legacy), "utf8"); const before = readFileSync(path, "utf8");
  const guard = new TrialBudget(path, 1);
  assert.equal(readFileSync(path, "utf8"), before);
  assert.equal(settle(guard, 1, { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }), false);
  assert.equal(readFileSync(path, "utf8"), before);
  assert.equal(guard.reserve(payload()), 17);
  assert.equal(settle(guard, 17, usage), true);
  const migrated = saved(path);
  assert.deepEqual(migrated.request_reservations.slice(0, 16), legacy.request_reservations);
  assert.equal(migrated.reserved_cost_micro_cny, legacy.reserved_cost_micro_cny);
  assert.equal(migrated.legacy_reservation_count, 16);
  assert.equal(new TrialBudget(path, 1).reserve(payload()), 18);
  assert.throws(() => new TrialBudget(path, 1).reserve(payload()), /TRIAL_CNY_BUDGET_EXHAUSTED/);
});

test("restart rejects corrupt reservation identifiers amounts and settled counters", () => {
  const path = ledgerPath(); const guard = new TrialBudget(path, 1);
  settle(guard, guard.reserve(payload()), usage);
  const valid = saved(path);
  const edits = [
    (ledger: any) => { ledger.request_reservations[0].ordinal = 0; },
    (ledger: any) => { ledger.request_reservations[0].reserved_micro_cny = -1; },
    (ledger: any) => { ledger.request_reservations[0].settlement.cost_micro_cny = 0; },
    (ledger: any) => { ledger.request_reservations[0].settlement.usage.total_tokens++; },
    (ledger: any) => { ledger.settled_cost_micro_cny = 0; },
    (ledger: any) => { ledger.reserved_cost_cny = -1; },
    (ledger: any) => { ledger.peak_cny_per_million.input_cache_hit = 0; },
    (ledger: any) => { ledger.legacy_reservation_count = 1; },
    (ledger: any) => { ledger.budget_blocked_due_to_usage_overflow = true; },
  ];
  for (const edit of edits) {
    const ledger = structuredClone(valid); edit(ledger);
    writeFileSync(path, JSON.stringify(ledger), "utf8");
    assert.throws(() => new TrialBudget(path, 1), /INVALID_EXISTING_TRIAL_LEDGER/);
  }
  writeFileSync(path, JSON.stringify(valid), "utf8");
  assert.doesNotThrow(() => new TrialBudget(path, 1));
});

test("restart validates legacy rows rather than accepting a balanced negative aggregate", () => {
  const path = ledgerPath(); const legacy = legacyLedger(1);
  legacy.request_reservations[0].reserved_micro_cny = -1;
  legacy.reserved_cost_micro_cny = -1; legacy.reserved_cost_cny = -0.000001;
  writeFileSync(path, JSON.stringify(legacy), "utf8");
  assert.throws(() => new TrialBudget(path, 1), /INVALID_EXISTING_TRIAL_LEDGER/);
});


test("explicit initialization persists a validated zero ledger without consuming authorization", () => {
  for (const policy of ['legacy-v1', 'v2-v1']) {
    const path = ledgerPath(); const guard = new TrialBudget(path, 1, policy);
    guard.initializeEmpty();
    const zero = saved(path);
    assert.deepEqual(zero.request_reservations, []);
    assert.equal(zero.reserved_cost_micro_cny, 0);
    assert.equal(zero.settled_cost_micro_cny, 0);
    const before = readFileSync(path, 'utf8');
    guard.initializeEmpty();
    assert.equal(readFileSync(path, 'utf8'), before);
    assert.equal(guard.reserve(payload()), 1);
  }
});

test("initialization preserves historical and outstanding reservations byte for byte", () => {
  const historical = ledgerPath(); writeFileSync(historical, JSON.stringify(legacyLedger(16)));
  const before = readFileSync(historical, 'utf8');
  const legacy = new TrialBudget(historical, 1); legacy.initializeEmpty();
  assert.equal(readFileSync(historical, 'utf8'), before);
  assert.equal(legacy.reserve(payload()), 17);
  const path = ledgerPath(); const guard = new TrialBudget(path, 1);
  guard.reserve(payload()); const outstanding = readFileSync(path, 'utf8');
  new TrialBudget(path, 1).initializeEmpty();
  assert.equal(readFileSync(path, 'utf8'), outstanding);
  assert.equal(new TrialBudget(path, 1).reserve(payload()), 2);
});

test("initialization revalidates an existing file and never repairs corrupt data", () => {
  const path = ledgerPath(); const guard = new TrialBudget(path, 1);
  writeFileSync(path, '{}');
  assert.throws(() => guard.initializeEmpty(), /TRIAL_LEDGER|INVALID_EXISTING/);
  assert.equal(readFileSync(path, 'utf8'), '{}');
});

test("initialization write failure cannot authorize a request", () => {
  const parent = join(mkdtempSync(join(tempRoot, 'finrisk-init-fail-')), 'file');
  writeFileSync(parent, 'block parent directory');
  const guard = new TrialBudget(join(parent, 'ledger.json'), 1);
  assert.throws(() => guard.initializeEmpty(), /EEXIST|ENOTDIR/);
});
