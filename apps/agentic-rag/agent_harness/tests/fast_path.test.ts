import test from "node:test";
import assert from "node:assert/strict";
import { fixture } from "./agent_loop_fixture.js";
import { canonical, hash } from "../src/contracts.js";
const modulePath = "../src/fast_path.js";
async function api() { const a = await import(modulePath).catch(() => ({})); assert.equal(typeof a.selectFastPath, "function"); return a; }
const gift = "购买本保险即向投保人赠送合同约定之外的购物卡。";
function input(text = gift, context: Record<string, string | null> = { purpose_hint: "当前保险营销宣传" }) {
 const f = fixture(); f.task.input.text = text; f.task.input.user_context = context;
 f.task.input.segments = [{ ...f.task.input.segments[0]!, text, end: Array.from(text).length }]; return f.task.input;
}
test("narrow current conduct selects only the two frozen families with explicit purpose and gift contract ownership", async () => {
 const a = await api();
 assert.equal(a.selectFastPath(input()).family, "transaction_gift");
 assert.equal(a.selectFastPath(input("本保险向客户保证最低回报。" )).family, "guaranteed_return");
 assert.equal(a.selectFastPath(input("购买本保险即向投保人赠送购物卡。", { purpose_hint: "当前保险营销宣传", contract_hint: "购物卡属于合同约定之外" })).selected, true);
 assert.equal(a.selectFastPath(input("购买本保险即向投保人额外赠送购物卡。")).selected, false);
});
test("rules reject mixed negated quoted multiple conduct roles conflicting background and unknown facts", async () => {
 const a = await api();
 for (const text of [gift + "同时保证保险收益。", "禁止" + gift, "不得保证保险收益，但我们保证保底回报。", "培训反例：" + gift, "转述客户：" + gift, gift + "购买第二份保险也送购物卡。", "招聘销售人员可获赠购物卡。", "购买本保险即向投保人赠送购物卡，合同归属未知。"])
  assert.equal(a.selectFastPath(input(text)).selected, false, text);
 for (const context of [{}, { purpose_hint: "保险风险教育" }, { purpose_hint: "当前保险营销宣传", actor_hint: "招聘人员" }, { purpose_hint: "当前保险营销宣传", product_hint: "不是保险" }, { purpose_hint: "当前保险营销宣传", contract_hint: "合同归属待核" }])
  assert.equal(a.selectFastPath(input(gift, context as unknown as Record<string, string | null>)).selected, false, JSON.stringify(context));
});
test("Unicode limit counts code points and rejects paragraphs and multiple input segments", async () => {
 const a = await api(), text = gift.slice(0, -1) + "😀".repeat(300 - Array.from(gift).length) + "。";
 assert.equal(a.selectFastPath(input(text)).selected, true); assert.equal(a.selectFastPath(input(text + "😀")).selected, false);
 assert.equal(a.selectFastPath(input(gift + "\n补充说明")).selected, false);
 const i = input(); i.segments.push({ ...i.segments[0]!, segment_id: "S0002" }); assert.equal(a.selectFastPath(i).selected, false);
});
test("rule inventory independently versions vocabulary purpose exclusions and canonical hash", async () => {
 const a = await api(), d = a.describeFastPathRules();
 assert.match(d.version, /^finrisk-fast-path\//); assert.ok(d.vocabulary.version && d.purpose_mapping.version && d.exclusions.version);
 assert.equal(a.fastPathRuleHash, hash(canonical(d))); assert.ok(Object.isFrozen(d));
});

test("explicit background cannot contradict the purpose contract ownership or single family", async () => {
 const a = await api();
 for (const context of [
  { purpose_hint: "当前保险营销宣传", material_function: "risk_education" },
  { purpose_hint: "当前保险营销宣传", contract_hint: "利益属于合同内" },
  { purpose_hint: "当前保险营销宣传", product_hint: "非保险产品" },
  { purpose_hint: "当前保险营销宣传", benefit_hint: "同时保证保险收益" },
 ]) assert.equal(a.selectFastPath(input(gift, context as unknown as Record<string, string | null>)).selected, false, JSON.stringify(context));
});

test("gift contract belongs to the actual benefit and product background must agree", async () => {
 const a = await api();
 const cases: [string, Record<string, string | null>][] = [
  ["购买本保险即向投保人赠送购物卡，保额属于合同内。", { purpose_hint: "当前保险营销宣传" }],
  ["购买本保险即向投保人赠送购物卡。", { purpose_hint: "当前保险营销宣传", contract_hint: "保额属于合同内" }],
  [gift, { purpose_hint: "当前保险营销宣传", product_hint: "该产品实际为银行理财产品" }],
 ];
 for (const [text, context] of cases) assert.equal(a.selectFastPath(input(text, context)).selected, false, JSON.stringify({text, context}));
});
