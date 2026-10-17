import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { TaskState } from "./state.js";
import { canonical, type Config, type JsonObject } from "./contracts.js";
import { candidateRead } from "./model_protocol.js";
import { buildCitationOptions } from "./citation_bundles.js";
import { projectRankingCoverage } from "./retrieval_quality.js";

export function estimateTokens(value: unknown): number {
  if (Array.isArray(value) && value.every(item => item && typeof item.role === "string")) {
    // Pi的details/usage/时间戳是本机事件元数据，不进入模型消息正文，不能重复计引文。
    value = value.map(message => message.role === "system" ? {
      role: message.role, content: message.content, sections: message.sections,
      toolsAdded: message.toolsAdded, toolsRemoved: message.toolsRemoved,
    } : message.role === "toolResult" ? {
      role: message.role, content: message.content, toolCallId: message.toolCallId, toolName: message.toolName,
    } : { role: message.role, content: message.content });
  }
  // ASCII按每字符1、非ASCII按每code point 2估计；明确是估计，不是模型tokenizer。
  return Array.from(JSON.stringify(value)).reduce((n, char) => n + (char.codePointAt(0)! < 128 ? 1 : 2), 64);
}
interface ProjectionOptions {
  pruneByCharacters?: boolean;
  omittedEvidenceIds?: Set<string>;
  omittedCandidateKeys?: Set<string>;
}

function protectedEvidence(e: JsonObject | undefined): boolean {
  return Boolean(e && (e.was_read || e.accepted_as_relevant || e.presented_in_fixed_context));
}
function bundleCandidate(candidate: JsonObject): boolean {
  return Array.isArray(candidate.required_evidence_ids);
}
function candidateKey(candidate: JsonObject): string {
  return candidate.unit_id ? `unit:${candidate.unit_id}`
    : canonical([candidate.source_id, candidate.required_evidence_ids]);
}
function candidateEvidenceIds(candidate: JsonObject): string[] {
  return [...new Set<string>([...candidate.required_evidence_ids,
    ...(candidate.evidence ?? []).map((e: JsonObject) => e.evidence_id)])];
}
function omitNextWireBundle(state: TaskState, omittedEvidenceIds: Set<string>, omittedCandidateKeys: Set<string>): boolean {
  const candidates = state.searches.flatMap(q => q.candidates ?? []).filter(bundleCandidate);
  const retained = candidates.filter(c => !omittedCandidateKeys.has(candidateKey(c)));
  // 必要锚点是一包；已读/接受/固定引文保护整个候选，其余候选按原顺序从尾部移除。
  for (let i = retained.length - 1; i >= 0; i--) {
    const candidate = retained[i]!;
    const ids = candidateEvidenceIds(candidate);
    if (ids.some(id => protectedEvidence(state.evidence.get(id)) || (state.evidence.has(id) && candidateRead(state.evidence.get(id)!, state)))) continue;
    const key = candidateKey(candidate);
    omittedCandidateKeys.add(key);
    const shared = new Set(retained.filter(c => candidateKey(c) !== key).flatMap(candidateEvidenceIds));
    for (const id of ids) if (!shared.has(id) && state.evidence.has(id)) omittedEvidenceIds.add(id);
    return true;
  }
  const bound = new Set(retained.flatMap(candidateEvidenceIds));
  // 兼容没有复合候选的旧证据；不能通过单条裁减破坏仍保留的V2包。
  const unread = [...state.evidence.values()].findLast(e => !omittedEvidenceIds.has(e.evidence_id)
    && !bound.has(e.evidence_id) && !protectedEvidence(e) && state.phase !== "finalizing");
  if (!unread) return false;
  omittedEvidenceIds.add(unread.evidence_id);
  return true;
}

export function buildContext(messages: AgentMessage[], state: TaskState, config: Config, trailingMessages: AgentMessage[] = [], projection: ProjectionOptions = {}): AgentMessage[] {
  if (state.protocolVersion === "finrisk-model-wire/1.2") return buildWireContext(messages, state, config, trailingMessages, projection);
  const system = messages.filter(m => m.role === "system" && !(typeof m.content === "string" && (m.content.startsWith("<task_state>") || m.content.startsWith("<runtime_stage>"))));
  const firstUser = messages.find(m => m.role === "user");
  // 只删除完整旧轮；保留最后assistant及其全部toolResult，不能留下孤立结果。
  const groups: AgentMessage[][] = [];
  for (const message of messages) {
    if (message.role === "system" || message === firstUser) continue;
    if (message.role === "assistant" || !groups.length) groups.push([]);
    groups[groups.length - 1]!.push(message);
  }
  if (state.phase === "finalizing" && state.analysis) for (let i = groups.length - 1; i >= 0; i--) {
    const group = groups[i]!;
    if (group.length !== 1 || group[0]!.role !== "assistant") continue;
    const message = group[0]!;
    if (message.role !== "assistant" || message.content.some(c => c.type === "toolCall")) continue;
    try {
      const value = JSON.parse(message.content.filter(c => c.type === "text").map(c => c.type === "text" ? c.text : "").join(""));
      if (canonical(value) === canonical(state.analysis)) groups.splice(i, 1);
    } catch { /* 只有已接受且完整保存在账本中的analysis可以去重。 */ }
  }
  const nativeIds = new Set<string>();
  for (let i = groups.length - 1; i >= 0; i--) {
    const group = groups[i]!;
    const ids = group.flatMap(m => m.role === "assistant" ? m.content.filter(c => c.type === "toolCall").map(c => c.type === "toolCall" ? c.id : "") : []);
    if (new Set(ids).size !== ids.length) groups.splice(i, 1);
  }
  for (let i = 0; i < groups.length; i++) {
    const ids = groups[i]!.flatMap(m => m.role === "assistant" ? m.content.filter(c => c.type === "toolCall").map(c => c.type === "toolCall" ? c.id : "") : []);
    if (ids.some(id => nativeIds.has(id))) { groups.splice(i--, 1); continue; }
    ids.forEach(id => nativeIds.add(id));
  }
  const ledger: AgentMessage = { role: "system", content: "<task_state>" + JSON.stringify({
    phase: state.phase, budget: state.runtimeBudget, risk_points: state.points, gaps: [...state.gaps.values()],
    analysis_scope: state.analysis ? { document_context: state.analysis.document_context, segment_assessments: state.analysis.segment_assessments, unprocessed_segment_ids: state.analysis.unprocessed_segment_ids } : null,
    evidence_locator: [...state.evidence.values()].filter(e => !e.accepted_as_relevant && !e.presented_in_fixed_context).map(e => ({ evidence_id: e.evidence_id, source_id: e.source_id, start: e.start, end: e.end, accepted_as_relevant: false })),
    accepted_evidence: [...state.evidence.values()].filter(e => e.accepted_as_relevant).map(e => ({ evidence_id: e.evidence_id, source_id: e.source_id, text_version: e.text_version, text_sha256: e.text_sha256, start: e.start, end: e.end, quote: e.quote, context_has_more: e.context_has_more })),
    evidence_for_assessment: [...state.evidence.values()].filter(e => e.presented_in_fixed_context && !e.accepted_as_relevant).map(e => ({ evidence_id: e.evidence_id, source_id: e.source_id, text_version: e.text_version, text_sha256: e.text_sha256, start: e.start, end: e.end, quote: e.quote, context_has_more: e.context_has_more, semantic_acceptance: "not_accepted" })),
    searches: state.searches.map(q => ({ focus_refs: q.focus_refs, risk_point_ids: q.risk_point_ids, status: q.status, complete: q.complete })),
  }) + "</task_state>", timestamp: Date.now() };
  const ceiling = config.service_max_input_tokens - 800; // 实际阶段/工具声明也参与估计，另留序列化余量。
  const assemble = () => [...system, ledger, ...(firstUser ? [firstUser] : []), ...groups.flat(), ...trailingMessages];
  while (groups.length > 1 && estimateTokens(assemble()) > ceiling) groups.shift();
  if (estimateTokens(assemble()) > ceiling) for (const group of groups) for (let i = 0; i < group.length; i++) {
    const message = group[i]!;
    if (message.role !== "toolResult" || message.toolName !== "search_cases") continue;
    const content = message.content.find(c => c.type === "text");
    if (!content || content.type !== "text") continue;
    try {
      const response = JSON.parse(content.text);
      if (!response.data?.queries) continue;
      const shortened = structuredClone(response);
      while (estimateTokens(assemble()) > ceiling && shortened.data.queries.some((q: any) => q.candidates.length)) {
        const candidates = shortened.data.queries.filter((q: any) => q.candidates.length);
        const largest = candidates.sort((a: any, b: any) => b.candidates.length - a.candidates.length)[0];
        largest.candidates.pop(); largest.returned_candidates_truncated = true;
        shortened.meta.context_projection = "whole_candidates_only; full original in task trace";
        group[i] = { ...message, content: [{ type: "text", text: JSON.stringify(shortened) }] };
      }
    } catch { /* 非本工具JSON保留，交给网关拒绝超限。 */ }
  }
  if (estimateTokens(assemble()) > ceiling) for (const group of groups) for (let i = 0; i < group.length; i++) {
    const message = group[i]!;
    if (message.role !== "toolResult" || message.toolName !== "read_source") continue;
    const content = message.content.find(c => c.type === "text");
    if (!content || content.type !== "text") continue;
    try {
      const response = JSON.parse(content.text); const evidence = response.data?.evidence;
      if (!evidence) continue;
      const projected = structuredClone(response);
      projected.data.evidence = { evidence_id: evidence.evidence_id, source_id: evidence.source_id, start: evidence.start, end: evidence.end, quote_omitted_from_this_tool_message: true };
      projected.meta.context_projection = state.evidence.get(evidence.evidence_id)?.accepted_as_relevant ? "完整原话保留在task_state.accepted_evidence" : "原话未放入本轮上下文；不能用此定位代替事实，可用read_source读取更小窗口";
      group[i] = { ...message, content: [{ type: "text", text: JSON.stringify(projected) }] };
    } catch { /* 未识别内容保持原样，不剪原话。 */ }
  }
  // 超限仍保留原文和最后一组，网关拒绝而不是静默裁断条件/输入。
  return assemble();
}

function buildWireContext(messages: AgentMessage[], state: TaskState, config: Config, trailing: AgentMessage[], projection: ProjectionOptions): AgentMessage[] {
  const system = messages.filter(m => m.role === "system" && !(typeof m.content === "string" && (m.content.startsWith("<task_state>") || m.content.startsWith("<runtime_stage>"))));
  const user = messages.find(m => m.role === "user");
  // 程序账本是已接受状态的唯一投影；终稿不携带旧工具对话，完整trace另存。
  const omittedEvidenceIds = projection.omittedEvidenceIds ?? new Set<string>();
  const omittedCandidateKeys = projection.omittedCandidateKeys ?? new Set<string>();
  const evidence = () => [...state.evidence.values()].filter(e => !omittedEvidenceIds.has(e.evidence_id) && (state.phase !== "finalizing" || protectedEvidence(e)));
  let usableEvidence = evidence();
  let visibleIds = new Set(usableEvidence.map(e => e.evidence_id));
  const readCoverage = (id: string): string[] => {
    const anchor = state.evidence.get(id);
    if (!anchor || !Number.isInteger(anchor.start) || !Number.isInteger(anchor.end)) return [];
    const windows = usableEvidence.filter(e => e.was_read && e.source_id === anchor.source_id
      && Number.isInteger(e.start) && Number.isInteger(e.end) && e.end > anchor.start && e.start < anchor.end).sort((a,b) => a.start-b.start);
    let covered = anchor.start;
    const ids: string[] = [];
    for (const window of windows) {
      if (window.end <= covered) continue;
      if (window.start > covered) break;
      covered = window.end; ids.push(window.evidence_id);
      if (covered >= anchor.end) return ids;
    }
    return [];
  };
  const visibleCandidate = (c: JsonObject): boolean => !bundleCandidate(c)
    || (!omittedCandidateKeys.has(candidateKey(c)) && c.required_evidence_ids.length > 0
      && c.required_evidence_ids.every((id: string) => visibleIds.has(id) || readCoverage(id).length));
  const candidateProjection = (c: JsonObject): JsonObject => ({
    source_id: c.source_id, rank: c.rank, duplicate_hint: c.duplicate_hint ?? null,
    evidence_ids: (c.evidence ?? []).map((e: JsonObject) => e.evidence_id),
    ...(c.required_evidence_ids ? { file_id: c.file_id, unit_id: c.unit_id,
      required_evidence_ids: c.required_evidence_ids, associations: c.associations,
      read_evidence_ids: [...new Set(c.required_evidence_ids.flatMap(readCoverage))],
      required_evidence_locators: c.required_evidence_ids.filter((id: string) => !visibleIds.has(id)).map((id: string) => {
        const e = state.evidence.get(id)!;
        return { evidence_id: id, source_id: e.source_id, start: e.start, end: e.end, covered_by: readCoverage(id) };
      }),
    } : {}),
  });
  const inputLocations = (v: any): any => {
    if (Array.isArray(v)) return v.map(inputLocations);
    if (!v || typeof v !== "object") return v;
    if (v.segment_id && v.quote) { const span = state.resolve(v); return { segment_id: span.segment_id, start: span.start, end: span.end }; }
    return Object.fromEntries(Object.entries(v).map(([key, value]) => [key, inputLocations(value)]));
  };
  const refreshCitationOptions = () => {
    const candidates = state.searches.flatMap(q => q.candidates ?? []).filter(c => !omittedCandidateKeys.has(candidateKey(c)));
    const options = buildCitationOptions(state.citationSnapshotId ?? "", candidates, usableEvidence);
    state.citationOptions = new Map(options.map(option => [option.case_ref, option]));
    // 模型只选择原子参照；被隐藏的检索锚点ID不作为可引用的输出。
    return options.map(({ required_evidence_ids, ...option }) => option);
  };
  const ledgerContent = () => "<task_state>" + JSON.stringify({
    ...(state.evaluationCitations ? { citation_options: refreshCitationOptions() } : {}),
    phase: state.phase,
    budget: state.runtimeBudget,
    scope: state.analysis ? { document_context: state.analysis.document_context, segment_assessments: state.analysis.segment_assessments, unprocessed_segment_ids: state.analysis.unprocessed_segment_ids } : null,
    input_reference_projection: "输入原话只在input.segments；账本中的start/end是Unicode字符半开区间。生成引用时从对应输入段复制原话。",
    risk_points: state.points.map(p => inputLocations({ risk_point_id: p.risk_point_id, input_refs: p.input_refs, candidate_tags: p.candidate_tags, facts: p.observations })),
    gaps: [...state.gaps.values()],
    evidence: usableEvidence.map(e => ({ evidence_id: e.evidence_id, source_id: e.source_id, quote: e.quote, context_has_more: e.context_has_more, anchor_not_fully_covered: e.anchor_not_fully_covered ?? false, was_read: Boolean(e.was_read), semantic_acceptance: e.accepted_as_relevant ? "model_assessed" : "not_accepted" })),
    unread_candidates_omitted: [...state.evidence.keys()].filter(id => !usableEvidence.some(e => e.evidence_id === id)),
    searches: state.searches.map(q => ({ focus_refs: inputLocations(q.focus_refs), risk_point_ids: q.risk_point_ids, goal: q.goal, status: q.status, complete: q.complete,
      ...(state.retrievalReliability && q.ranking_coverage ? { ranking_coverage: projectRankingCoverage(q) } : {}),
      candidates: (q.candidates ?? []).filter(visibleCandidate).map(candidateProjection) })),
  }) + "</task_state>";
  const ledger: AgentMessage = { role: "system", content: ledgerContent(), timestamp: Date.now() };
  const recent: AgentMessage[] = [];
  if (state.phase !== "finalizing") {
    const last = messages.findLastIndex(m => m.role === "assistant");
    const group = last < 0 ? [] : messages.slice(last).filter(m => m.role !== "system");
    const assistant = group[0];
    if (assistant?.role === "assistant" && assistant.content.some(c => c.type === "toolCall")) {
      const ids = assistant.content.filter(c => c.type === "toolCall").map(c => c.type === "toolCall" ? c.id : "");
      const results = group.filter(m => m.role === "toolResult");
      if (new Set(ids).size === ids.length && ids.every(id => results.some(m => m.role === "toolResult" && m.toolCallId === id))) {
        recent.push(assistant, ...results);
      }
    }
  }
  const projectRecent = () => recent.map(m => {
          if (m.role !== "toolResult") return m;
          const block = m.content.find(c => c.type === "text");
          try {
            const response = JSON.parse(block?.type === "text" ? block.text : "");
            const strip = (v: any): any => {
              if (Array.isArray(v)) return v.map(strip);
              if (!v || typeof v !== "object") return v;
              const projected: JsonObject = Object.fromEntries(Object.entries(v).map(([key, value]) => [key,
                key === "quote" && v.evidence_id && state.evidence.has(v.evidence_id)
                  ? "若本轮提供原话，见task_state.evidence；否则须回源，不能仅凭定位引用。"
                  : key === "candidates" && Array.isArray(value) ? value.filter(visibleCandidate).map(strip) : strip(value)]));
              if (Array.isArray(v.candidates) && projected.candidates.length < v.candidates.length) projected.returned_candidates_truncated = true;
              return projected;
            };
            const projected = strip(response);
            if (state.retrievalReliability) {
              projected.meta = { snapshot_id: response.meta?.snapshot_id };
              for (const q of projected.data?.queries ?? []) {
                if (q.ranking_coverage) q.ranking_coverage = projectRankingCoverage(q);
              }
            }
            return { ...m, content: [{ type: "text" as const, text: JSON.stringify(projected) }] };
          } catch { return m; }
  });
  const assemble = () => [...system, ...(user ? [user] : []), ledger, ...projectRecent(), ...trailing];
  while (projection.pruneByCharacters !== false && estimateTokens(assemble()) > config.service_max_input_tokens - 800) {
    if (!omitNextWireBundle(state, omittedEvidenceIds, omittedCandidateKeys)) break;
    usableEvidence = evidence(); visibleIds = new Set(usableEvidence.map(e => e.evidence_id));
    ledger.content = ledgerContent();
  }
  state.contextEvidenceIds = new Set(usableEvidence.map(e => e.evidence_id));
  // 不裁原文/必要条件/已读引文；超过服务上限仍由网关明确拒绝。
  return assemble();
}

export async function buildCountedContext<T extends { prompt_tokens: number; admission_fits?: boolean }>(
  messages: AgentMessage[], state: TaskState, config: Config, trailing: AgentMessage[],
  count: (messages: AgentMessage[]) => Promise<T>,
): Promise<{ messages: AgentMessage[]; count: T }> {
  const omittedEvidenceIds = new Set<string>();
  const omittedCandidateKeys = new Set<string>();
  let projected = buildContext(messages, state, config, trailing, { pruneByCharacters: false, omittedEvidenceIds, omittedCandidateKeys });
  let counted = await count(projected);
  while (state.protocolVersion === "finrisk-model-wire/1.2" && (counted.prompt_tokens > config.service_max_input_tokens - config.context_safety_margin_tokens || counted.admission_fits === false)) {
    if (!omitNextWireBundle(state, omittedEvidenceIds, omittedCandidateKeys)) break;
    projected = buildContext(messages, state, config, trailing, { pruneByCharacters: false, omittedEvidenceIds, omittedCandidateKeys });
    counted = await count(projected);
  }
  return { messages: projected, count: counted };
}
