import { appendFileSync, mkdirSync, writeFileSync, renameSync } from "node:fs";
import { join } from "node:path";
import type { JsonObject } from "./contracts.js";
export function atomicJson(path: string, value: unknown): void {
  const temporary = `${path}.${process.pid}.tmp`;
  writeFileSync(temporary, JSON.stringify(value, null, 2), "utf8"); renameSync(temporary, path);
}
export class Trace {
  events: JsonObject[] = [];
  onEvent?: (event: JsonObject) => void;
  constructor(public directory?: string, private eventTaskId?: string) { if (directory) mkdirSync(directory, { recursive: true }); }
  event(type: string, data: unknown): void {
    const row = { sequence: this.events.length + 1, type, at: new Date().toISOString(), data, ...(this.eventTaskId ? { event_id: this.eventTaskId + ":" + (this.events.length + 1) } : {}) };
    this.events.push(row);
    if (this.directory) appendFileSync(join(this.directory, "events.jsonl"), `${JSON.stringify(row)}\n`, "utf8");
    if (this.directory && type === "progress.snapshot") appendFileSync(join(this.directory, "progress_events.jsonl"), `${JSON.stringify(row)}\n`, "utf8");
    this.onEvent?.(row);
  }
  message(value: unknown): void {
    if (this.directory) appendFileSync(join(this.directory, "model_messages.jsonl"), `${JSON.stringify(value)}\n`, "utf8");
  }
  save(name: string, value: unknown): void { if (this.directory) atomicJson(join(this.directory, name), value); }
}
