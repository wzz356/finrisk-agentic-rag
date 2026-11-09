import test from "node:test";
import assert from "node:assert/strict";
import { ResponseObserver } from "../src/response_observer.js";

const content = '{"说明":"中文🙂","points":[]}';
const frame = (delta: unknown, finish: unknown = null) => `data: ${JSON.stringify({ choices: [{ index: 0, delta, finish_reason: finish }] })}\r\n\r\n`;
const tail = frame({}, "stop") + 'data: {"choices":[],"usage":{"total_tokens":3}}\r\n\r\ndata: [DONE]\r\n\r\n';
function feed(observer: ResponseObserver, text: string) { observer.feed(new TextEncoder().encode(text)); observer.flush(); }

test("observer preserves bytes and Chinese split across every UTF8 byte", async () => {
  const observer = new ResponseObserver();
  const wire = ': heartbeat\r\n\r\n' + frame({ role: "assistant" }) + frame({ content: content.slice(0, 9) }) + frame({ content: content.slice(9) }) + tail;
  const bytes = new TextEncoder().encode(wire);
  const original = new Response(new ReadableStream({ start(controller) { for (const byte of bytes) controller.enqueue(Uint8Array.of(byte)); controller.close(); } }), { headers: { "content-type": "text/event-stream" } });
  const observed = observer.observeResponse(original);
  assert.deepEqual(new Uint8Array(await observed.arrayBuffer()), bytes);
  const result = observer.complete(content);
  assert.equal(result.capture_complete, true); assert.equal(result.raw_json_valid, true); assert.equal(result.text_matches, true);
  assert.equal(result.fragment_count, 2); assert.equal(result.fragments.join(""), content);
});

test("observer reports an assembly mismatch only with a complete independent capture", () => {
  const observer = new ResponseObserver(); feed(observer, frame({ content }) + tail);
  const result = observer.complete(content + " ");
  assert.equal(result.capture_complete, true); assert.equal(result.text_matches, false);
  assert.notEqual(result.raw_text_sha256, result.assembled_text_sha256);
});

test("missing DONE cannot be misclassified as a provider or assembly error", () => {
  const observer = new ResponseObserver(); feed(observer, frame({ content }) + frame({}, "stop"));
  const result = observer.complete("different");
  assert.equal(result.capture_complete, false); assert.equal(result.text_matches, null); assert.equal(result.raw_json_valid, null);
});

test("missing finish reason cannot create complete capture", () => {
  const observer = new ResponseObserver(); feed(observer, frame({ content }) + 'data: [DONE]\n\n');
  assert.equal(observer.complete(content).capture_complete, false);
});

test("unknown event names are incomplete even when data resembles a chat chunk", () => {
  const observer = new ResponseObserver(); feed(observer, 'event: thread.unrecognized\n' + frame({ content }) + tail);
  const result = observer.complete(content);
  assert.equal(result.capture_complete, false); assert.equal(result.text_matches, null);
  assert.equal(result.incomplete_reason, "unsupported_event");
});

test("bounded capture cannot silently truncate and blame provider", async () => {
  const observer = new ResponseObserver({ max_chars: 3, max_frame_chars: 1024, max_fragments: 5 });
  const wire = frame({ content: "{}" }) + frame({ content: "overflow" }) + tail;
  const response = observer.observeResponse(new Response(wire, { headers: { "content-type": "text/event-stream" } }));
  assert.equal(await response.text(), wire);
  const result = observer.complete("anything");
  assert.equal(result.incomplete_reason, "text_limit"); assert.equal(result.text_matches, null);
  assert.ok(result.captured_chars <= 3);
});

test("fragment and frame limits bound response diagnostics", () => {
  const fragments = new ResponseObserver({ max_chars: 100, max_frame_chars: 1024, max_fragments: 1 });
  feed(fragments, frame({ content: "{" }) + frame({ content: "}" }) + tail);
  assert.equal(fragments.complete("{}").fragment_count, 1);
  assert.equal(fragments.complete("{}").incomplete_reason, "text_limit");
  const large = new ResponseObserver({ max_chars: 100, max_frame_chars: 10, max_fragments: 2 });
  feed(large, "data: " + "x".repeat(30));
  assert.equal(large.complete("").incomplete_reason, "frame_limit");
});

test("observer never records reasoning, refusal or unknown error bodies", () => {
  const secret = "sk-DO-NOT-LOG Authorization: Bearer marker";
  const observer = new ResponseObserver();
  feed(observer, frame({ content, reasoning_content: secret, refusal: secret }) + tail);
  assert.equal(observer.complete(content).capture_complete, true);
  assert.ok(!JSON.stringify(observer.complete(content)).includes(secret));
  const bad = new ResponseObserver(); feed(bad, `data: ${JSON.stringify({ error: { message: secret } })}\n\n`);
  assert.equal(bad.complete("").capture_complete, false);
  assert.ok(!JSON.stringify(bad.complete("")).includes(secret));
});

test("native tool argument text is not stored as assistant JSON", () => {
  const observer = new ResponseObserver();
  feed(observer, frame({ tool_calls: [{ function: { name: "search_cases", arguments: '{"q":"private"}' } }] }) + frame({}, "tool_calls") + 'data: [DONE]\n\n');
  const result = observer.complete("");
  assert.equal(result.has_tool_calls, true); assert.equal(result.raw_json_valid, null);
  assert.deepEqual(result.fragments, []);
});

test("invalid UTF8, malformed frames and nonzero choices remain incomplete", () => {
  const utf = new ResponseObserver(); utf.feed(Uint8Array.of(255)); utf.flush();
  assert.equal(utf.complete("").incomplete_reason, "invalid_utf8");
  for (const value of ['data: not-json\n\n', 'data: {"choices":[{"index":1,"delta":{"content":"wrong"}}]}\n\n']) {
    const observer = new ResponseObserver(); feed(observer, value + tail);
    assert.equal(observer.complete("").capture_complete, false); assert.equal(observer.complete("").text_matches, null);
  }
});

test("response cancellation reaches the original source without a second reader", async () => {
  let cancelled = false;
  const original = new Response(new ReadableStream({ pull(controller) { controller.enqueue(new TextEncoder().encode(': heartbeat\n\n')); }, cancel() { cancelled = true; } }), { headers: { "content-type": "text/event-stream" } });
  const reader = new ResponseObserver().observeResponse(original).body!.getReader();
  await reader.read(); await reader.cancel();
  await new Promise(resolve => setTimeout(resolve, 0));
  assert.equal(cancelled, true);
});
