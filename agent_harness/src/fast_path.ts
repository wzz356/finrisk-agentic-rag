import { canonical, hash, type InputRecord } from "./contracts.js";

function freeze<T>(v: T): T { if (v && typeof v === "object") { Object.values(v).forEach(freeze); Object.freeze(v); } return v; }
const rules = freeze({
 version: "finrisk-fast-path/2", max_unicode: 300, max_query_unicode: 500,
 vocabulary: { version: "fast-vocabulary/1",
  insurance: ["保险"], dividend: ["分红"], transaction: ["购买", "投保", "买保险"],
  gift: ["赠送", "赠", "送"], benefit: ["购物卡", "礼品", "红包", "现金", "实物", "礼券", "优惠券"],
  beneficiary: ["投保人", "客户", "消费者", "您", "你"],
  contract: ["合同约定之外", "合同约定之内", "合同约定内", "合同之外", "合同外", "合同内"],
  contract_outside: ["合同约定之外", "合同之外", "合同外"], contract_inside: ["合同约定之内", "合同约定内", "合同内"],
  guarantee: ["保证", "保底", "最低回报"], return: ["收益", "回报", "分红"],
  family_tags: { transaction_gift: ["赠送利益", "合同外利益"], guaranteed_return: ["保证收益", "承诺收益"] } },
 purpose_mapping: { version: "fast-purpose/1", field: "purpose_hint", direct_promotion: ["当前保险营销宣传", "保险营销宣传", "保险销售宣传", "保险销售话术"] },
 exclusions: { version: "fast-exclusions/1",
  terms: ["非保险", "prohibition_reminder", "risk_education", "fact_description", "employee_instruction", "unknown", "禁止", "不", "不得", "不应", "不要", "不可", "并非", "不是", "不保证", "不承诺", "未", "无", "否", "风险提示", "提醒", "培训", "反例", "引用", "转述", "报道", "研究", "招聘", "教育", "历史", "示例", "假设", "未知", "不详", "不明", "待核", "不确定"],
  multiple: ["同时", "另外", "此外", "以及", "还将", "又", "第二", "并且", "；", ";", "且"],
  delimiters: ["\r", "\n", "\u2028", "\u2029", "“", "”", "‘", "’", "「", "」", "《", "》", "\""] },
 policy: { single_segment: true, single_sentence: true, single_family: true, purpose_exact: true,
  gift_contract_explicit: true,
  benefit_contract_forms: ["{contract}的{benefit}", "{contract}{benefit}", "{benefit}属于{contract}", "{benefit}在{contract}", "{benefit}为{contract}"],
  product_hint_pattern: "^(?:本产品|该产品|当前产品)?(?:是|为|属于)?保险(?:产品)?[。.]?$",
  extra_word_proves_contract_outside: false, decision: "route_only_no_risk_judgment" },
});
export type FastPathFamily = keyof typeof rules.vocabulary.family_tags;
export type FastPathSelection = { selected: false; reason: string } | { selected: true; family: FastPathFamily; reason: string };
export function describeFastPathRules() { return rules; }
export const fastPathRuleHash = hash(canonical(rules));
/** Only current input and explicit background are inspected; this never determines a risk label. */
export function selectFastPath(input: InputRecord): FastPathSelection {
 const reject = (reason: string): FastPathSelection => ({ selected: false, reason });
 if (input.segments.length !== 1 || !input.text.trim()) return reject("segment_count");
 if (Array.from(input.text).length > rules.max_unicode) return reject("input_length");
 if (Array.from(canonical({ text: input.text, user_context: input.user_context })).length > rules.max_query_unicode) return reject("query_length");
 const background = Object.values(input.user_context).filter((v): v is string => typeof v === "string").join(" "), all = input.text + " " + background;
 const has = (text: string, terms: readonly string[]) => terms.some(t => text.includes(t));
 if (!rules.purpose_mapping.direct_promotion.includes(String(input.user_context.purpose_hint ?? ""))) return reject("purpose_unknown");
 if (has(all, rules.exclusions.terms) || has(input.text, rules.exclusions.multiple) || has(input.text, rules.exclusions.delimiters)) return reject("excluded_structure");
 if (input.text.replace(/[。！？!?]$/, "").match(/[。！？!?]/)) return reject("multiple_conduct");
 const v = rules.vocabulary;
 const productHint = input.user_context.product_hint?.trim();
 if (productHint && !new RegExp(rules.policy.product_hint_pattern).test(productHint)) return reject("background_conflict");
 const giftSignal = has(input.text, v.gift) && has(input.text, v.benefit);
 const returnSignal = has(input.text, v.guarantee) && has(input.text, v.return);
 if (has(all, v.contract_outside) && has(all, v.contract_inside)) return reject("background_conflict");
 if (has(all, v.gift) && has(all, v.benefit) && has(all, v.guarantee) && has(all, v.return)) return reject("mixed_families");
 const benefits = v.benefit.filter(benefit => input.text.includes(benefit));
 // Contract wording must describe the same benefit that is being offered, within one
 // explicit material field. Unrelated coverage/insured-amount ownership is insufficient.
 const benefitContract = benefits.length > 0 && benefits.every(benefit => [input.text, ...Object.values(input.user_context)].some(text =>
  typeof text === "string" && v.contract.some(contract => rules.policy.benefit_contract_forms.some(form =>
   text.includes(form.replace("{contract}", contract).replace("{benefit}", benefit))))));
 if (giftSignal && has(input.text, v.transaction) && has(input.text, v.insurance) && has(input.text, v.beneficiary) && benefitContract)
  return { selected: true, family: "transaction_gift", reason: "explicit_single_mechanism" };
 if (!giftSignal && returnSignal && (has(all, v.insurance) || has(all, v.dividend)))
  return { selected: true, family: "guaranteed_return", reason: "explicit_single_mechanism" };
 return reject("mechanism_or_fact_unknown");
}
