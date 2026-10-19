import { canonical, type JsonObject } from "./contracts.js";
import { candidateRead } from "./model_protocol.js";
import { sourceReadPlan } from "./retrieval_plan.js";
import type { TaskState } from "./state.js";
import type { Trace } from "./trace.js";

/** Attempts belong to this phase; V2 coverage is recalculated from actual read windows. */
export function createSourceReadSchedule({ state, trace, attempted: sharedAttempts, maxLogicalReads = 3 }: { state: TaskState; trace: Trace; attempted?: Set<string>; maxLogicalReads?: number }, queries: JsonObject[], phase: string) {
  const dynamic = queries.some(q => (q.candidates ?? []).some((c: JsonObject) => c.required_evidence_ids !== undefined));
  // V1 保留一次性多样性排序；V2 每次按真实窗口覆盖重算下一个锚点。
  const planned = sourceReadPlan(queries, dynamic ? 1 : 3, e => candidateRead(e, state));
  const attempted = sharedAttempts ?? new Set<string>(); const skipped = new Set<string>();
  const key = (e: JsonObject) => canonical([e.source_id, e.evidence_id]);
  let logicalReads = 0;
  return {
    planned,
    next: (index: number) => dynamic ? sourceReadPlan(queries, 1, e => {
      const covered = candidateRead(e, state);
      if (covered && !skipped.has(key(e))) {
        skipped.add(key(e));
        trace.event("retrieval.read_skipped", { phase, source_id: e.source_id, anchor_evidence_id: e.evidence_id, reason: "already_covered" });
      }
      return covered || attempted.has(key(e));
    })[0] : planned[index],
    attempt: (candidate: JsonObject) => {
      attempted.add(key(candidate.evidence[0])); logicalReads++;
      trace.event("retrieval.read_selected", { phase, source_id: candidate.source_id, anchor_evidence_id: candidate.evidence[0].evidence_id, logical_read: logicalReads, max_logical_reads: maxLogicalReads });
    },
    end: (reason: string) => trace.event("retrieval.read_phase_ended", { phase, logical_reads: logicalReads, reason }),
  };
}
