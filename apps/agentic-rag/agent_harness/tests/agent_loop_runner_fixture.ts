import assert from "node:assert/strict";
import { after } from "node:test";
import { offlineCounter } from "./offline_counter.js";
import { fixture, comparison } from "./agent_loop_fixture.js";
import { hash, type JsonObject } from "../src/contracts.js";
const counter = offlineCounter(); after(() => counter.close());
export function stage(context: any): any {
  const message = context.messages.findLast((m: any) => m.role === "system" && typeof m.content === "string" && m.content.includes("<loop_stage>"));
  assert.ok(message, "complete typed/native stage must reach actual gateway");
  return JSON.parse(message.content.split("<loop_stage>")[1].split("</loop_stage>")[0]);
}
export function ledger(context: any): any {
  const message = context.messages.find((m: any) => m.role === "system" && String(m.content).startsWith("<task_state>"));
  return JSON.parse(message.content.slice(12, -13));
}
export function setup(mode = "planned_adaptive", overrides: JsonObject = {}) {
  const f = fixture(); Object.assign(f.task.config!, { orchestration_mode: mode, fast_path_enabled: false, ...overrides });
  const calls: {name: string; args: JsonObject}[] = [], previews: JsonObject[] = [];
  const source = (name: string, id: string) => {
    const quote = name === "GIFT" ? "处罚原文：购买保险送购物卡。" : "处罚原文：宣传保险保证收益。";
    return { evidence_id: id, source_id: name, source_kind: "penalty", text_version: "1", text_sha256: hash(quote),
      start: 0, end: Array.from(quote).length, quote, source_verified: true, context_has_more: false };
  };
  const bridge: any = {
    calls, previews, countTokens: counter.count,
    async call(name: string, args: JsonObject) {
      calls.push({ name, args: structuredClone(args) });
      const meta = { snapshot_id: "SNAPSHOT", backend_query_attempts: name === "search_cases" ? 1 : 0, cache_hits: 0, external_retrieval_requests: 0 };
      let data: any;
      if (name === "search_cases") data = { queries: args.queries.map((q: JsonObject) => {
        const src = q.text.includes("赠礼") ? "GIFT" : "GAIN", anchor = source(src, src + ":E");
        return { ...q, query_id: "Q" + calls.length, status: "ok", complete: true,
          stage_status: { bm25: "completed", bge: "completed", rrf: "completed", rerank: "completed" },
          ranking_coverage: { schema_version: "finrisk-ranking-coverage/1.0", expected_units: 1, scored_units: 1, failed_units: 0,
            ranking_complete: true, topk_scope: "full_pool", attempts_exhausted: false, failure_counts: {}, failed_unit_ids: [],
            expected_set_sha256: "a".repeat(64), scored_set_sha256: "a".repeat(64), failed_set_sha256: "b".repeat(64), rerank_protocol: "span-ref-v2" },
          candidates: [{ source_id: src, unit_id: src + ":U", evidence: [anchor], required_evidence_ids: [anchor.evidence_id] }] };
      }) };
      else if (name === "read_source") data = { evidence: source(args.source_id, args.source_id + ":W") };
      else data = { definitions: structuredClone(f.task.definitions.filter(d => args.tags.includes(d.risk_tag))) };
      return { schema_version: "finrisk-tools/1.0", status: "ok", data, meta, error: null };
    },
    async previewRead(args: JsonObject) {
      previews.push(structuredClone(args));
      return { schema_version: "finrisk-tools/1.0", status: "ok", data: { evidence: source(args.source_id, args.source_id + ":W") },
        meta: { snapshot_id: "SNAPSHOT", backend_query_attempts: 0, cache_hits: 0, external_retrieval_requests: 0 }, error: null };
    },
    async verifyEvidence(ids: string[]) { return { valid_ids: ids, rejected_ids: [] }; },
  };
  return { f, bridge, calls };
}
export function assess(f: ReturnType<typeof fixture>, context: any, match = "reliable_match") {
  const s = stage(context), options = ledger(context).citation_options;
  return f.assess(s.request, [options.length ? [comparison(options.at(-1).case_ref, match)] : []]);
}
