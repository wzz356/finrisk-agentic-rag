import type { AgentMessage } from "@earendil-works/pi-agent-core";
import { type JsonObject } from "./contracts.js";
import { criteriaProjection } from "./criteria.js";
import { AGENT_LOOP_VERSION, loopContract, type LoopOutputKind } from "./agent_loop_contracts.js";
import { derivePointGaps, type AllowedAction } from "./agent_gap_policy.js";
import type { LoopRegistry, AssessmentRequest } from "./agent_loop_state.js";

/** Materials remain user data; permissions and output contracts live in the program stage. */
export function loopInputMessages(registry: LoopRegistry, systemPrompt: string, pointId?: string): AgentMessage[] {
 const { input } = registry.task, point = pointId ? registry.points.get(pointId) : undefined;
 return [{ role: "system", content: systemPrompt, timestamp: 0 }, { role: "user", content: JSON.stringify({ input: {
  source_id: input.source_id, user_context: input.user_context,
  ...(point ? { input_refs: point.input_refs.map(ref => ({ ...ref, ...registry.state.resolve(ref) })) }
    : { segments: input.segments.map(s => ({ segment_id: s.segment_id, text: s.text })) }),
 } }), timestamp: 0 }];
}
export function loopStageData(registry: LoopRegistry, kind: LoopOutputKind | "act", req?: AssessmentRequest, allowed?: AllowedAction[], repair?: JsonObject): JsonObject {
 const pointId = req?.risk_point_id || allowed?.[0]?.point_id, point = pointId ? registry.points.get(pointId) : undefined;
 const tags = point?.candidate_tags ?? registry.task.definitions.map(d => d.risk_tag);
 const selected = new Set(allowed?.flatMap(a => a.gap_ids) ?? []);
 return { protocol_version: AGENT_LOOP_VERSION, kind,
  ...(req ? { request: { assessment_id: req.assessment_id, risk_point_id: req.risk_point_id, state_revision: req.state_revision,
    candidate_tags: [...req.candidate_tags], candidate_version: req.candidate_version } } : {}),
  ...(point ? { point: { risk_point_id: point.risk_point_id, plan_revision: point.plan_revision, state_revision: point.state_revision,
    candidate_version: point.candidate_version, assessment_revision: point.assessment_revision, input_refs: point.input_refs, facts: point.facts },
    current_checks: (registry.currentAccepted(point) ?? point.draft)?.result.tag_checks ?? [] } : {}),
  ...(kind === "act" ? { allowed_actions: allowed, actionable_gaps: derivePointGaps(registry).filter(g => selected.has(g.gap_id)),
    instruction: "仅执行许可点和目标的原生工具；每轮最多一次search且单query；也可只输出{kind:finalize}停止，保留缺口。" }
   : { output_schema: loopContract(kind), instruction: kind === "plan" ? "仅从当前输入和明确背景制定计划；程序分配RP，主query先覆盖不同点再使用备选。"
    : "覆盖全部候选标签和每个criterion；不新增事实，不把案例当当前事实，只从当前citation_options选case_ref。" }),
  dictionary: registry.task.definitions.filter(d => kind === "plan" || kind === "one_shot" || tags.includes(d.risk_tag)).map(d => ({ tag: d.risk_tag, definition: d.description })),
  ...(kind === "plan" ? {} : { criteria: criteriaProjection(tags) }),
  ...(repair ? { format_correction: repair } : {}),
 };
}
