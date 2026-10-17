import { hash, type JsonObject } from "./contracts.js";
import { syntaxPosition } from "./model_json.js";

// Independent of pi-ai's SSE parser. Only assistant text of choice zero is
// retained: no headers, request data, error bodies, reasoning or refusal text.
export class ResponseObserver {
  private decoder = new TextDecoder("utf-8", { fatal: true });
  private buffer = "";
  private fragments: string[] = [];
  private chars = 0;
  private failure: string | null = null;
  private finished = false;
  private done = false;
  private hasTools = false;
  private finishReason: string | null = null;
  constructor(private limits = { max_chars: 262144, max_frame_chars: 65536, max_fragments: 8192 }) {}
  observeResponse(response: Response): Response {
    if (!response.ok || !response.body || !response.headers.get("content-type")?.includes("text/event-stream")) {
      this.failure = "unsupported_response"; return response;
    }
    const observer = this;
    const body = response.body.pipeThrough(new TransformStream<Uint8Array, Uint8Array>({
      transform(bytes, controller) { observer.feed(bytes); controller.enqueue(bytes); },
      flush() { observer.flush(); },
    }));
    return new Response(body, { status: response.status, statusText: response.statusText, headers: response.headers });
  }
  feed(bytes: Uint8Array): void {
    if (this.failure) return;
    try { this.buffer += this.decoder.decode(bytes, { stream: true }); }
    catch { this.failure = "invalid_utf8"; this.buffer = ""; return; }
    this.consume();
  }
  private consume(): void {
    let boundary: RegExpExecArray | null;
    while ((boundary = /\r\n\r\n|\n\n|\r\r/.exec(this.buffer))) {
      const frame = this.buffer.slice(0, boundary.index);
      this.buffer = this.buffer.slice(boundary.index + boundary[0].length);
      if (frame.length > this.limits.max_frame_chars) { this.failure = "frame_limit"; break; }
      this.frame(frame);
      if (this.failure) break;
    }
    if (this.buffer.length > this.limits.max_frame_chars) this.failure = "frame_limit";
    if (this.failure) this.buffer = "";
  }
  private frame(frame: string): void {
    const lines = frame.replace(/^\uFEFF/, "").split(/\r\n|\r|\n/);
    // Named events can be wrapped/ignored differently by the SDK. This observer
    // only understands ordinary Chat SSE; do not guess their text semantics.
    if (lines.some(line => line.startsWith("event:"))) { this.failure = "unsupported_event"; return; }
    const data = lines.filter(line => line.startsWith("data:")).map(line => line.slice(5).replace(/^ /, "")).join("\n");
    if (!data) return;
    if (this.done) { this.failure = "frame_after_done"; return; }
    if (data === "[DONE]") { this.done = true; return; }
    let value: JsonObject;
    try { value = JSON.parse(data); } catch { this.failure = "invalid_frame"; return; }
    if (!value || !Array.isArray(value.choices)) { this.failure = "unsupported_frame"; return; }
    if (!value.choices.length) return; // usage-only event
    if (value.choices.length !== 1 || value.choices[0]?.index !== 0) { this.failure = "unsupported_choice"; return; }
    const choice = value.choices[0];
    const content = choice.delta?.content;
    if (content !== undefined && content !== null) {
      if (typeof content !== "string" || this.finished) { this.failure = "unsupported_content"; return; }
      if (content) {
        if (this.chars + content.length > this.limits.max_chars || this.fragments.length >= this.limits.max_fragments) { this.failure = "text_limit"; return; }
        this.fragments.push(content); this.chars += content.length;
      }
    }
    if (choice.delta?.tool_calls?.length) this.hasTools = true;
    if (choice.finish_reason !== undefined && choice.finish_reason !== null) {
      if (this.finished || !["stop", "length", "tool_calls", "content_filter"].includes(choice.finish_reason)) { this.failure = "unsupported_finish"; return; }
      this.finished = true; this.finishReason = choice.finish_reason;
    }
  }
  flush(): void {
    if (this.failure) return;
    try { this.buffer += this.decoder.decode(); } catch { this.failure = "invalid_utf8"; }
    this.consume();
    if (this.buffer.trim()) this.failure ??= "unterminated_frame";
  }
  complete(assembled: string): JsonObject {
    const raw = this.fragments.join("");
    const complete = !this.failure && this.done && this.finished;
    let valid: boolean | null = null; let position: number | null = null;
    if (complete && !this.hasTools && raw.trim()) {
      try { JSON.parse(raw); valid = true; } catch (error) { valid = false; position = syntaxPosition(error, raw); }
    }
    return { schema_version: "finrisk-response-observation/1.0", capture_complete: complete,
      incomplete_reason: this.failure ?? (complete ? null : "missing_termination"), finish_reason: this.finishReason,
      has_tool_calls: this.hasTools, fragment_count: this.fragments.length, captured_chars: this.chars,
      raw_text_sha256: hash(raw), assembled_text_sha256: hash(assembled), text_matches: complete ? raw === assembled : null,
      raw_json_valid: valid, raw_json_error_position: position, fragments: [...this.fragments] };
  }
}
