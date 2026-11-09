import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { canonical } from "./contracts.js";
import { admissionPolicy, admissionPolicyHash, assessRequest, type Admission } from './request_admission.js';

import { loadBudgetProfile, budgetProfileSha256, PLANNED_BUDGET_PROFILE } from "./budget_profile.js";

export interface TrialBudgetBinding { budget_profile_id: string; budget_profile_sha256: string; model_trial_cap_micro_cny?: number }
function boundLimit(binding: TrialBudgetBinding | undefined, capCny: number, policyId: string): number {
  if (!binding) return 18;
  const profile = loadBudgetProfile(binding.budget_profile_id);
  const effectiveCap = binding.model_trial_cap_micro_cny ?? profile.model_trial_cap_micro_cny;
  if (!Number.isSafeInteger(effectiveCap) || effectiveCap < 1 || effectiveCap > profile.model_trial_cap_micro_cny
    || binding.budget_profile_id !== PLANNED_BUDGET_PROFILE || budgetProfileSha256(profile) !== binding.budget_profile_sha256
    || capCny !== effectiveCap / 1_000_000 || policyId !== "v2-v1") throw new Error("TRIAL_LEDGER_BINDING_MISMATCH");
  return profile.max_model_requests;
}
interface TrialUsage {
  prompt_tokens: number; completion_tokens: number; total_tokens: number;
  prompt_cache_hit_tokens: number; prompt_cache_miss_tokens: number;
}
function nonnegativeInteger(value: unknown): value is number { return Number.isSafeInteger(value) && (value as number) >= 0; }
function checkedUsage(raw: unknown): TrialUsage | undefined {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return undefined;
  const row = raw as Record<string, unknown>;
  const prompt = row.prompt_tokens; const completion = row.completion_tokens; const total = row.total_tokens;
  if (!nonnegativeInteger(prompt) || !nonnegativeInteger(completion) || !nonnegativeInteger(total) || prompt + completion !== total) return undefined;
  let hit = 0; let miss = prompt;
  if (Object.hasOwn(row, "prompt_cache_hit_tokens") || Object.hasOwn(row, "prompt_cache_miss_tokens")) {
    if (!nonnegativeInteger(row.prompt_cache_hit_tokens) || !nonnegativeInteger(row.prompt_cache_miss_tokens)
      || row.prompt_cache_hit_tokens + row.prompt_cache_miss_tokens !== prompt) return undefined;
    hit = row.prompt_cache_hit_tokens; miss = row.prompt_cache_miss_tokens;
  }
  return { prompt_tokens: prompt, completion_tokens: completion, total_tokens: total,
    prompt_cache_hit_tokens: hit, prompt_cache_miss_tokens: miss };
}
// 冻结已核高峰价：命中0.04、未命中2、输出8元/百万令牌；整数微元向上取整。
function usageCost(usage: TrialUsage): bigint {
  return (BigInt(usage.prompt_cache_hit_tokens) + 24n) / 25n
    + BigInt(usage.prompt_cache_miss_tokens) * 2n + BigInt(usage.completion_tokens) * 8n;
}
function validateLedger(ledger: any, capCny: number, policyId = 'legacy-v1', binding?: TrialBudgetBinding): void {
  const maxRequests = boundLimit(binding, capCny, policyId);
  if (binding ? ledger?.budget_profile_id !== binding.budget_profile_id || ledger?.budget_profile_sha256 !== binding.budget_profile_sha256
    : ledger?.budget_profile_id !== undefined || ledger?.budget_profile_sha256 !== undefined) throw new Error("TRIAL_LEDGER_BINDING_MISMATCH");
  const invalid = () => { throw new Error("INVALID_EXISTING_TRIAL_LEDGER"); };
  const versioned = ledger?.schema_version === 'finrisk-paid-trial/1.2';
  if ((versioned && (ledger.policy_id !== policyId || ledger.policy_sha256 !== admissionPolicyHash(policyId)
    || ledger.body_max_bytes !== admissionPolicy(policyId).body_max_bytes)) || (!versioned && policyId !== 'legacy-v1')) throw new Error('TRIAL_LEDGER_POLICY_MISMATCH');
  const bodyLimit = versioned ? admissionPolicy(policyId).body_max_bytes : 24000;
  if (!ledger || typeof ledger !== "object" || !["finrisk-paid-trial/1.0", "finrisk-paid-trial/1.1", 'finrisk-paid-trial/1.2'].includes(ledger.schema_version)
    || ledger.cap_cny !== capCny || ledger.model !== "deepseek-flash"
    || ledger.peak_cny_per_million?.input_cache_miss !== 2 || ledger.peak_cny_per_million?.output !== 8
    || ledger.prompt_overhead_token_allowance !== 512 || !Array.isArray(ledger.request_reservations)
    || ledger.request_reservations.length > maxRequests) invalid();
  const legacy = ledger.schema_version === "finrisk-paid-trial/1.0";
  const legacyCount = legacy ? ledger.request_reservations.length : ledger.legacy_reservation_count;
  if (!nonnegativeInteger(legacyCount) || legacyCount > ledger.request_reservations.length
    || (!legacy && ledger.peak_cny_per_million.input_cache_hit !== 0.04)) invalid();
  let reserved = 0n; let settled = 0n; const overflowCosts: bigint[] = [];
  for (const [index, row] of ledger.request_reservations.entries()) {
    if (!row || row.ordinal !== index + 1 || !nonnegativeInteger(row.body_bytes) || row.body_bytes > bodyLimit
      || !nonnegativeInteger(row.max_output_tokens) || row.max_output_tokens < 1 || row.max_output_tokens > 4096
      || typeof row.reserved_at !== "string" || !Number.isFinite(Date.parse(row.reserved_at))
      || row.reserved_micro_cny !== (row.body_bytes + 512) * 2 + row.max_output_tokens * 8
      || row.reserved_cny !== row.reserved_micro_cny / 1_000_000) invalid();
    if (!Object.hasOwn(row, "settlement")) { reserved += BigInt(row.reserved_micro_cny); continue; }
    if (index < legacyCount || !row.settlement || typeof row.settlement !== "object") invalid();
    const settlement = row.settlement; const usage = checkedUsage(settlement.usage);
    if (!usage || typeof settlement.settled_at !== "string" || !Number.isFinite(Date.parse(settlement.settled_at))) invalid();
    const cost = usageCost(usage!);
    if (settlement.status === "settled") {
      if (!nonnegativeInteger(settlement.cost_micro_cny) || cost !== BigInt(settlement.cost_micro_cny)
        || settlement.cost_cny !== settlement.cost_micro_cny / 1_000_000) invalid();
      settled += cost;
    } else if (settlement.status === "cost_overflow") {
      reserved += BigInt(row.reserved_micro_cny); overflowCosts.push(cost);
    } else invalid();
  }
  if (!nonnegativeInteger(ledger.reserved_cost_micro_cny) || reserved !== BigInt(ledger.reserved_cost_micro_cny)
    || ledger.reserved_cost_cny !== ledger.reserved_cost_micro_cny / 1_000_000) invalid();
  if (!legacy && (!nonnegativeInteger(ledger.settled_cost_micro_cny) || settled !== BigInt(ledger.settled_cost_micro_cny)
    || ledger.settled_cost_cny !== ledger.settled_cost_micro_cny / 1_000_000
    || ledger.budget_blocked_due_to_usage_overflow !== (overflowCosts.length > 0)
    || overflowCosts.some(cost => cost + settled <= BigInt(Number.MAX_SAFE_INTEGER)))) invalid();
}

// 本轮显式授权的3条小样专用；按已核官方人民币高峰价预留，未知用量不退预留。
export class TrialBudget {
  private ledger: any;
  constructor(private path: string, private capCny: number, private policyId = 'legacy-v1', private binding?: TrialBudgetBinding) {
    if (!(capCny > 0 && capCny <= 1)) throw new Error("INVALID_TRIAL_CNY_CAP");
    boundLimit(binding, capCny, policyId);
    const policy = admissionPolicy(policyId);
    this.ledger = existsSync(path) ? JSON.parse(readFileSync(path, "utf8")) : {
      ...(binding ?? {}), schema_version: policyId === 'legacy-v1' ? "finrisk-paid-trial/1.1" : 'finrisk-paid-trial/1.2', model: "deepseek-flash", cap_cny: capCny,
      ...(policyId === 'legacy-v1' ? {} : { policy_id: policyId, policy_sha256: admissionPolicyHash(policyId), body_max_bytes: policy.body_max_bytes }),
      price_source: "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/", price_checked_date: "2026-10-14",
      peak_cny_per_million: { input_cache_miss: 2, input_cache_hit: 0.04, output: 8 }, prompt_overhead_token_allowance: 512,
      reserved_cost_micro_cny: 0, reserved_cost_cny: 0, request_reservations: [],
      settled_cost_micro_cny: 0, settled_cost_cny: 0, legacy_reservation_count: 0, budget_blocked_due_to_usage_overflow: false,
      reservation_kind: "UTF8_request_body_bytes_plus512_prompt_bound; max_tokens_output; peak_price",
    };
    validateLedger(this.ledger, capCny, policyId, binding);
    // 只在内存升级，直到新请求才写入；旧行不可结算，不追认历史用量或授权。
    if (this.ledger.schema_version === "finrisk-paid-trial/1.0") this.ledger = { ...this.ledger,
      schema_version: "finrisk-paid-trial/1.1", peak_cny_per_million: { ...this.ledger.peak_cny_per_million, input_cache_hit: 0.04 },
      legacy_reservation_count: this.ledger.request_reservations.length, settled_cost_micro_cny: 0,
      settled_cost_cny: 0, budget_blocked_due_to_usage_overflow: false };
  }
  initializeEmpty(): void {
    if (existsSync(this.path)) {
      // Revalidate and reuse persisted reservations without rewriting migrations.
      this.reload();
      return;
    }
    if (this.ledger.request_reservations.length || this.ledger.reserved_cost_micro_cny || this.ledger.settled_cost_micro_cny)
      throw new Error("TRIAL_LEDGER_INITIALIZATION_REQUIRES_EMPTY");
    this.persist(structuredClone(this.ledger));
  }
  /** Refresh before every decision; a shared cached instance is never authorization. */
  private reload(): void {
    boundLimit(this.binding, this.capCny, this.policyId);
    if (!existsSync(this.path)) {
      if (this.binding && this.ledger.request_reservations.length) throw new Error("TRIAL_LEDGER_ROLLBACK");
      return;
    }
    const next = new TrialBudget(this.path, this.capCny, this.policyId, this.binding).ledger;
    if (this.binding) {
      for (const [index, prior] of this.ledger.request_reservations.entries()) {
        const current = next.request_reservations[index];
        const { settlement: oldSettlement, ...oldIdentity } = prior;
        const { settlement: newSettlement, ...newIdentity } = current ?? {};
        if (!current || canonical(oldIdentity) !== canonical(newIdentity)
          || oldSettlement && canonical(oldSettlement) !== canonical(newSettlement)) throw new Error("TRIAL_LEDGER_ROLLBACK");
      }
    }
    this.ledger = next;
  }
  preview(body: string): Admission {
    this.reload();
    const assessment = assessRequest(body, this.policyId);
    if (!assessment.fits) return assessment;
    const reserved = assessment.metrics!.reserved_micro_cny;
    if (this.ledger.request_reservations.length >= boundLimit(this.binding, this.capCny, this.policyId) || this.ledger.budget_blocked_due_to_usage_overflow
      || BigInt(this.ledger.settled_cost_micro_cny) + BigInt(this.ledger.reserved_cost_micro_cny) + BigInt(reserved)
        > BigInt(Math.floor(this.capCny * 1_000_000))) return { ...assessment, fits: false, reason: 'TRIAL_CNY_BUDGET_EXHAUSTED' };
    return assessment;
  }
  /** Read-only admission for one action request and its required future assessments. */
  previewSequence(bodies: readonly string[]): Admission {
    this.reload();
    if (!bodies.length) throw new Error("INVALID_TRIAL_FUTURE_RESERVE");
    const assessments = bodies.map(body => assessRequest(body, this.policyId));
    const failed = assessments.find(row => !row.fits);
    if (failed) return failed;
    const total = assessments.reduce((n, row) => n + BigInt(row.metrics!.reserved_micro_cny), 0n);
    if (this.ledger.request_reservations.length + bodies.length > boundLimit(this.binding, this.capCny, this.policyId)
      || this.ledger.budget_blocked_due_to_usage_overflow
      || BigInt(this.ledger.settled_cost_micro_cny) + BigInt(this.ledger.reserved_cost_micro_cny) + total > BigInt(Math.floor(this.capCny * 1_000_000)))
      return { ...assessments[0]!, fits: false, reason: "TRIAL_CNY_BUDGET_EXHAUSTED" };
    return assessments[0]!;
  }
  reserve(body: string): number {
    const preview = this.preview(body);
    if (!preview.fits) throw new Error(preview.reason!);
    const { body_bytes: bodyBytes, max_output_tokens: maxTokens, reserved_micro_cny: reserved } = preview.metrics!;
    const id = this.ledger.request_reservations.length + 1;
    const next = structuredClone(this.ledger);
    next.request_reservations.push({ ordinal: id,
      reserved_at: new Date().toISOString(), body_bytes: bodyBytes, max_output_tokens: maxTokens,
      reserved_micro_cny: reserved, reserved_cny: reserved / 1_000_000 });
    next.reserved_cost_micro_cny += reserved;
    next.reserved_cost_cny = next.reserved_cost_micro_cny / 1_000_000;
    this.persist(next);
    return id;
  }
  settle(id: number, rawUsage?: unknown): boolean {
    this.reload();
    if (!Number.isSafeInteger(id) || id <= this.ledger.legacy_reservation_count) return false;
    const row = this.ledger.request_reservations.find((reservation: any) => reservation.ordinal === id);
    if (!row) return false;
    if (row.settlement) return row.settlement.status === "settled";
    const usage = checkedUsage(rawUsage);
    if (!usage) return false;
    const cost = usageCost(usage);
    const next = structuredClone(this.ledger); const reservation = next.request_reservations[id - 1];
    const settledAt = new Date().toISOString();
    if (cost + BigInt(next.settled_cost_micro_cny) > BigInt(Number.MAX_SAFE_INTEGER)) {
      reservation.settlement = { status: "cost_overflow", settled_at: settledAt, usage };
      next.budget_blocked_due_to_usage_overflow = true;
      this.persist(next); return false;
    }
    reservation.settlement = { status: "settled", settled_at: settledAt, usage,
      cost_micro_cny: Number(cost), cost_cny: Number(cost) / 1_000_000 };
    next.reserved_cost_micro_cny -= reservation.reserved_micro_cny;
    next.reserved_cost_cny = next.reserved_cost_micro_cny / 1_000_000;
    next.settled_cost_micro_cny += Number(cost);
    next.settled_cost_cny = next.settled_cost_micro_cny / 1_000_000;
    this.persist(next); return true;
  }
  private persist(next: any): void {
    validateLedger(next, this.capCny, this.policyId, this.binding);
    mkdirSync(dirname(this.path), { recursive: true });
    const temporary = `${this.path}.tmp`;
    writeFileSync(temporary, JSON.stringify(next, null, 2), "utf8");
    renameSync(temporary, this.path);
    this.ledger = next;
  }
}
let sharedTrial: TrialBudget | undefined;
let sharedPath: string | undefined;
export function activeTrialBudget(policyId = 'legacy-v1', binding?: TrialBudgetBinding): TrialBudget | undefined {
  const path = process.env.FINRISK_AGENT_TRIAL_LEDGER;
  if (!path) return undefined;
  const key = `${path}:${policyId}:${JSON.stringify(binding ?? null)}`;
  if (key !== sharedPath) { sharedTrial = new TrialBudget(path, (binding?.model_trial_cap_micro_cny ?? 1_000_000) / 1_000_000, policyId, binding); sharedPath = key; }
  return sharedTrial;
}
