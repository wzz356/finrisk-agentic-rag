import test from "node:test";
import assert from "node:assert/strict";
import { stream } from "@earendil-works/pi-ai/api/openai-responses";
import { deepseekModel, buildProviderPayload } from "../src/prompt_counter.js";
import { assessRequest } from "../src/request_admission.js";

// This is an offline SDK compatibility probe, not a new production gateway.
// Every request is intercepted in memory; no real key or network is used.
test("pinned Pi can carry Responses JSON Schema but current Chat admission refuses its body", async () => {
  const schema = { type: "object", additionalProperties: false, required: ["kind"], properties: { kind: { type: "string", enum: ["probe"] } } };
  const text = '{"kind":"probe"}';
  const item = { type: "message", id: "msg-probe", role: "assistant", status: "completed", content: [{ type: "output_text", text, annotations: [] }] };
  const events = [
    { type: "response.created", response: { id: "probe" } },
    { type: "response.output_item.added", output_index: 0, item: { ...item, status: "in_progress", content: [] } },
    { type: "response.output_text.delta", output_index: 0, content_index: 0, delta: text },
    { type: "response.output_item.done", output_index: 0, item },
    { type: "response.completed", response: { id: "probe", status: "completed", output: [item], usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15, input_tokens_details: { cached_tokens: 0 } } } },
  ];
  let calls = 0; let body: any;
  const context: any = { messages: [{ role: "user", content: "Return JSON", timestamp: 0 }] };
  const model = { ...deepseekModel, api: "openai-responses" as const };
  const result = await stream(model, context, { apiKey: "offline-fake-key", maxTokens: 4096, maxRetries: 0,
    onPayload(payload) { return { ...payload as any, text: { format: { type: "json_schema", name: "finrisk_probe", schema } }, reasoning: { effort: "none" } }; },
    fetch: async (url, init) => {
      calls++; assert.equal(String(url), "https://api.deepseek.com/responses");
      body = JSON.parse(String(init?.body));
      return new Response(events.map(event => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`).join(""), { headers: { "content-type": "text/event-stream" } });
    },
  }).result();
  assert.equal(calls, 1); assert.equal(result.stopReason, "stop");
  assert.equal(result.content.filter(c => c.type === "text").map(c => c.type === "text" ? c.text : "").join(""), text);
  assert.deepEqual(body.text.format.schema, schema); assert.equal(body.max_output_tokens, 4096);
  assert.equal(body.reasoning.effort, "none"); assert.ok(!Object.hasOwn(body, "messages"));
  assert.equal(assessRequest(JSON.stringify(body), "v2-v1").fits, false, "must not bill Responses as a Chat payload");
  const current = buildProviderPayload(context, "analyzing");
  assert.deepEqual(current.response_format, { type: "json_object" }); assert.ok(!current.text);
});
