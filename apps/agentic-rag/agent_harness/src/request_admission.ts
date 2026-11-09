import { readFileSync } from 'node:fs';
import { canonical, hash, root } from './contracts.js';
const rules = JSON.parse(readFileSync(`${root}/contracts/agent_rag/request-admission-policy.json`, 'utf8'));
export interface RequestMetrics { body_bytes: number; max_output_tokens: number; reserved_micro_cny: number; }
export interface Admission { fits: boolean; reason: string | null; metrics?: RequestMetrics; }
export function admissionPolicy(id = 'legacy-v1'): { body_max_bytes: number; max_output_tokens: number } {
  if (!Object.hasOwn(rules.policies, id)) throw new Error('UNKNOWN_REQUEST_ADMISSION_POLICY');
  return { ...rules.policies[id] };
}
export function admissionPolicyHash(id = 'legacy-v1'): string { return hash(canonical(admissionPolicy(id))); }
export function requestMetrics(body: string): RequestMetrics {
  let payload;
  try { payload = JSON.parse(body); } catch { throw new Error('TRIAL_PAYLOAD_NOT_ADMITTED'); }
  if (payload?.model !== 'deepseek-flash' || payload.thinking?.type !== 'disabled'
    || !Number.isInteger(payload.max_tokens) || payload.max_tokens < 1 || payload.max_tokens > 4096) throw new Error('TRIAL_PAYLOAD_NOT_ADMITTED');
  const body_bytes = Buffer.byteLength(body, 'utf8');
  return { body_bytes, max_output_tokens: payload.max_tokens, reserved_micro_cny: (body_bytes + 512) * 2 + payload.max_tokens * 8 };
}
export function assessRequest(body: string, id = 'legacy-v1'): Admission {
  const policy = admissionPolicy(id);
  try {
    const metrics = requestMetrics(body);
    return { fits: metrics.body_bytes <= policy.body_max_bytes, reason: metrics.body_bytes > policy.body_max_bytes ? 'TRIAL_REQUEST_BODY_TOO_LARGE' : null, metrics };
  } catch { return { fits: false, reason: 'TRIAL_PAYLOAD_NOT_ADMITTED' }; }
}
