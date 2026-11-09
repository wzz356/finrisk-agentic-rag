import assert from "node:assert/strict";
import { createServer } from "node:http";
import { once } from "node:events";
import { test } from "node:test";
import { runJsonTask } from "../runtime.mjs";

const baseRequest = {
  messages: [{ role: "system", content: "返回JSON对象。" }, { role: "user", content: "测试材料" }],
  task: "json_review", review_context: {}, temperature: 0, max_tokens: 300, extra_body: {},
};
const baseSettings = {
  model: "deepseek-flash", thinking: "disabled", read_timeout_seconds: 3, agent_max_turns: 3,
};

test("an unrelated negative mechanism needs no invented quote but positive mechanisms do", async () => {
  const request = {...baseRequest,task:"risk_review",review_context:{review_policy:"direct",
    target_ids:["s1"],allowed_labels:["虚列费用套取资金"],sentences:[{sentence_id:"s1",text:"虚开发票套取资金。"}]}};
  await withProvider((_,n)=>({tool:"submit_result",args:{items:[{
    sentence_id:"s1",risk_score:.8,risk_labels:["虚列费用套取资金"],evidence:["虚开发票套取资金"],
    reasoning:"费用套取成立，本句不涉及回访。",decision_state:"risk",mechanism_assessments:[
      {mechanism_id:"M01",decision_state:"no_clear_risk",risk_labels:[],evidence:[],reasoning:"不涉及回访"},
      {mechanism_id:"M12",decision_state:"risk",risk_labels:["虚列费用套取资金"],evidence:n===1?[]:["虚开发票套取资金"]},
    ],
  }]}}),async settings=>{
    const result=await runJsonTask({request,settings,apiKey:"dummy"});
    assert.equal(result.agent_trace.turns,2);
    assert.equal(result.data.items[0].mechanism_assessments[0].evidence.length,0);
    assert.equal(result.data.items[0].mechanism_assessments[1].evidence.length,1);
  });
});

test("native evidence schema restricts references to registered IDs before validation", async () => {
  const request = { ...baseRequest, task: "risk_review", review_context: {
    review_policy: "direct", evidence_mode: "source_ids", target_ids: ["target"], allowed_labels: ["承诺收益"],
    compact_payload: { source_sentences: [{ id: "s1", text: "保证收益8%。" }],
      targets: [{ sentence_id: "target", source_ids: ["s1"] }] },
    sentences: [{sentence_id:"target",text:"保证收益8%。",context:{source_sentences:[{text:"保证收益8%。"}]}}],
  } };
  await withProvider(() => ({tool:"submit_result",args:{items:[{
    sentence_id:"target",risk_score:.8,risk_labels:["承诺收益"],evidence:["s1"],
    reasoning:"直接收益承诺。",decision_state:"risk",
  }]}}), async (settings, requests) => {
    const result = await runJsonTask({request,settings,apiKey:"dummy"});
    const schema = requests[0].tools[0].function.parameters.properties.items.items.properties.evidence.items;
    assert(JSON.stringify(schema).includes('"const":"s1"') || JSON.stringify(schema).includes('"enum":["s1"]'));
    assert.deepEqual(result.data.items[0].evidence,["保证收益8%。"]);
  });
});

test("listwise rerank validates the complete permutation and repairs once with native schema", async () => {
  const request = { ...baseRequest, task: "case_rerank", review_context: {
    review_policy: "direct", document_count: 3,
  } };
  await withProvider((_, n) => ({ tool: "submit_result", args: {
    ranking: n === 1 ? [0, 0, 2] : [2, 0, 1],
  } }), async (settings, requests) => {
    const result = await runJsonTask({ request, settings, apiKey: "dummy" });
    assert.deepEqual(result.data.ranking, [2, 0, 1]);
    assert.equal(result.agent_trace.turns, 2);
    assert.equal(result.agent_trace.tool_failures, 1);
    assert.equal(result.agent_trace.validation_failures.length, 1);
    assert.deepEqual(requests[0].tools.map(tool => tool.function.name), ["submit_result"]);
    const schema = requests[0].tools[0].function.parameters;
    assert(schema.properties.ranking);
    assert(!schema.properties.result_json);
  });
});

test("adaptive review submits native fields and preserves an independent risk beside unknown facts", async () => {
  const request = { ...baseRequest, task: "risk_review", review_context: {
    review_policy: "direct", target_ids: ["s1"], allowed_labels: ["承诺收益", "虚假宣传"],
    sentences: [{ sentence_id: "s1", text: "保证收益8%，停售公告待核。" }],
    required_predicates: { M05: ["M05.promise"] },
  } };
  await withProvider((_, n) => ({ tool: "submit_result", args: { items: [{
    sentence_id: "s1", risk_score: .8, risk_labels: ["承诺收益"], evidence: [n === 1 ? "错误引文" : "保证收益8%"],
    reasoning: "收益承诺成立，公告另核。", decision_state: "risk", missing_facts: ["停售公告"],
    mechanism_assessments: [
      { mechanism_id: "M05", decision_state: "risk", risk_labels: ["承诺收益"], evidence: ["保证收益8%"],
        condition_assessments: [{ condition_id: "M05.promise", status: "met", evidence: ["保证收益8%"] }] },
      { mechanism_id: "M09", decision_state: "insufficient_information", missing_facts: ["停售公告"] },
    ],
  }] } }), async (settings, requests) => {
    const result = await runJsonTask({ request, settings: { ...settings, agent_max_turns: 4 }, apiKey: "dummy" });
    assert.equal(result.agent_trace.turns, 2);
    assert.equal(result.agent_trace.tool_failures, 1);
    assert.equal(result.agent_trace.validation_failures.length, 1);
    assert.deepEqual(result.data.items[0].risk_labels, ["承诺收益"]);
    const schema = requests[0].tools.find(t => t.function.name === "submit_result").function.parameters;
    assert(schema.properties.items);
    assert(!schema.properties.result_json);
    assert.deepEqual(requests[0].tools.map(t => t.function.name), ["submit_result"]);
  });
});

async function withProvider(handler, execute) {
  const requests = [];
  const server = createServer(async (req, res) => {
    let data = "";
    for await (const chunk of req) data += chunk;
    const body = JSON.parse(data);
    requests.push(body);
    const answer = handler(body, requests.length);
    if (answer.hold) return;
    if (answer.status) {
      res.writeHead(answer.status, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: { message: "provider error", type: "test_error" } }));
      return;
    }
    res.writeHead(200, { "Content-Type": "text/event-stream" });
    const delta = answer.tool
      ? { tool_calls: [{ index: 0, id: "call_" + requests.length, type: "function",
          function: { name: answer.tool, arguments: JSON.stringify(answer.args) } }] }
      : { content: answer.text || "尚未提交" };
    const chunk = {
      id: "test", object: "chat.completion.chunk", created: 1, model: "deepseek-flash",
      choices: [{ index: 0, delta, finish_reason: null }],
    };
    const done = { ...chunk, choices: [{ index: 0, delta: {}, finish_reason: answer.tool ? "tool_calls" : "stop" }],
      usage: { prompt_tokens: 20, completion_tokens: 10, total_tokens: 30 } };
    res.end("data: " + JSON.stringify(chunk) + "\n\ndata: " + JSON.stringify(done) + "\n\ndata: [DONE]\n\n");
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const settings = { ...baseSettings, base_url: "http://127.0.0.1:" + server.address().port + "/v1" };
  try { await execute(settings, requests); }
  finally { server.closeAllConnections(); server.close(); await once(server, "close"); }
}

test("long-text screening requires a complete disjoint partition and repairs missing indexes", async () => {
  const request = { ...baseRequest, task: "risk_screen", review_context: { expected_indexes: [0, 1, 2, 3] } };
  await withProvider((_, n) => ({ tool: "submit_result", args: {
    complete: true, review_indexes: [2], normal_ranges: n === 1 ? [[0, 1]] : [[0, 1], [3, 3]],
  } }), async (settings, requests) => {
    const result = await runJsonTask({ request, settings, apiKey: "dummy" });
    assert.equal(result.agent_trace.turns, 2);
    assert.equal(result.agent_trace.tool_failures, 1);
    assert.deepEqual(result.data.review_indexes, [2]);
    assert.deepEqual(requests[0].tools.map(t => t.function.name), ["submit_result"]);
    assert(requests[0].tools[0].function.parameters.properties.normal_ranges);
  });
});

test("real Pi loop reads context, submits once, and exposes no raw text in trace", async () => {
  await withProvider((_, n) => n === 1
    ? { tool: "read_review_context", args: {} }
    : { tool: "submit_result", args: { result_json: '{"ok":true}' } }, async (settings, requests) => {
    const result = await runJsonTask({ request: baseRequest, settings, apiKey: "dummy" });
    assert.deepEqual(result.data, { ok: true });
    assert.equal(result.agent_trace.turns, 2);
    assert.equal(result.agent_trace.tool_calls, 2);
    assert.equal(requests.length, 2);
    assert.deepEqual(requests[0].thinking, { type: "disabled" });
    assert.deepEqual(requests[0].tools.map(v => v.function.name), ["read_review_context", "submit_result"]);
    assert(!JSON.stringify(result.agent_trace).includes("测试材料"));
    assert(!JSON.stringify(result.agent_trace).includes("dummy"));
  });
});

test("malformed submission becomes a tool error and is repaired within the same Agent", async () => {
  await withProvider((_, n) => ({ tool: "submit_result", args: {
    result_json: n === 1 ? "{invalid" : '{"ok":true}',
  } }), async (settings, requests) => {
    const result = await runJsonTask({ request: baseRequest, settings, apiKey: "dummy" });
    assert.equal(result.agent_trace.tool_failures, 1);
    assert.equal(result.agent_trace.turns, 2);
    assert(requests[1].messages.some(m => m.role === "tool" && m.content.includes("有效")));
  });
});

test("risk review refuses invented quotes and repairs using provided source", async () => {
  const request = { ...baseRequest, task: "risk_review", review_context: {
    target_ids: ["s1"], allowed_labels: ["承诺收益"],
    sentences: [{ sentence_id: "s1", text: "保证收益8%。", context: {
      source_sentences: [{ text: "保证收益8%。" }],
    } }],
  } };
  await withProvider((_, n) => ({ tool: "submit_result", args: {
    result_json: JSON.stringify({ items: [{ sentence_id: "s1", risk_score: .9,
      risk_labels: ["承诺收益"], reasoning: "原文承诺收益。", evidence: [n === 1 ? "从不存在的原文" : "保证收益8%"] }] }),
  } }), async settings => {
    const result = await runJsonTask({ request, settings, apiKey: "dummy" });
    assert.equal(result.agent_trace.tool_failures, 1);
    assert.deepEqual(result.data.items[0].evidence, ["保证收益8%"]);
  });
});

test("unknown tool cannot execute and the Agent can correct its next call", async () => {
  await withProvider((_, n) => n === 1
    ? { tool: "bash", args: { command: "should never run" } }
    : { tool: "submit_result", args: { result_json: '{"safe":true}' } }, async settings => {
    const result = await runJsonTask({ request: baseRequest, settings, apiKey: "dummy" });
    assert.deepEqual(result.data, { safe: true });
    assert.equal(result.agent_trace.tool_failures, 1);
  });
});

test("missing structured submission stops at the turn budget", async () => {
  await withProvider(() => ({ text: "继续分析，但不提交" }), async (settings, requests) => {
    await assert.rejects(
      runJsonTask({ request: baseRequest, settings, apiKey: "dummy" }),
      error => error.code === "turn_budget",
    );
    assert.equal(requests.length, 3);
  });
});

test("missing current business facts cannot be submitted as an established risk", async () => {
  const request = { ...baseRequest, task: "risk_review", review_context: {
    target_ids: ["s1"], allowed_labels: ["虚假宣传"],
    sentences: [{ sentence_id: "s1", text: "产品下周停售。" }],
  } };
  await withProvider((_, n) => ({ tool: "submit_result", args: {
    result_json: JSON.stringify({ items: [{ sentence_id: "s1", risk_score: .6,
      risk_labels: [], reasoning: "缺少当前公司公告。", evidence: ["产品下周停售"],
      decision_state: n === 1 ? "risk" : "insufficient_information",
      candidate_labels: ["虚假宣传"], missing_facts: ["公司公告及产品日期"],
      condition_assessments: [{ condition_id: "notice", status: "unknown", evidence: [] }],
    }] }),
  } }), async settings => {
    const result = await runJsonTask({ request, settings, apiKey: "dummy" });
    assert.equal(result.agent_trace.turns, 2);
    assert.equal(result.agent_trace.tool_failures, 1);
    assert.equal(result.data.items[0].decision_state, "insufficient_information");
  });
});

test("provider authentication failure has sanitized, nonretryable error", async () => {
  await withProvider(() => ({ status: 401 }), async settings => {
    await assert.rejects(
      runJsonTask({ request: baseRequest, settings, apiKey: "dummy" }),
      error => error.code === "provider_error" && !error.message.includes("dummy") && !error.retryable,
    );
  });
});

test("already cancelled task does not contact the provider", async () => {
  await withProvider(() => ({ text: "unused" }), async (settings, requests) => {
    const controller = new AbortController();
    controller.abort();
    await assert.rejects(
      runJsonTask({ request: baseRequest, settings, apiKey: "dummy", signal: controller.signal }),
      error => error.code === "cancelled",
    );
    assert.equal(requests.length, 0);
  });
});

test("cancelling an active provider request aborts the Pi run", async () => {
  let notifyStarted;
  const started = new Promise(resolve => { notifyStarted = resolve; });
  await withProvider(() => { notifyStarted(); return { hold: true }; }, async settings => {
    const controller = new AbortController();
    const pending = runJsonTask({ request: baseRequest, settings, apiKey: "dummy", signal: controller.signal });
    const check = assert.rejects(pending, error => error.code === "cancelled");
    await started;
    controller.abort();
    await check;
  });
});

test("provider request timeout produces a bounded, retryable failure", async () => {
  await withProvider(() => ({ hold: true }), async settings => {
    await assert.rejects(
      runJsonTask({ request: baseRequest, settings: { ...settings, read_timeout_seconds: .1 }, apiKey: "dummy" }),
      error => error.code === "timeout" && error.retryable,
    );
  });
});
