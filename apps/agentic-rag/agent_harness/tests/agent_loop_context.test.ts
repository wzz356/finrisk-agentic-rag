import test from "node:test";
import assert from "node:assert/strict";
import { fixture, comparison } from "./agent_loop_fixture.js";
import { setup, stage, ledger, assess } from "./agent_loop_runner_fixture.js";
import { LoopRegistry } from "../src/agent_loop_state.js";
import { buildCountedContext } from "../src/context_builder.js";
import { MODEL_WIRE_VERSION, hash, canonical } from "../src/contracts.js";
import { runPlannedTask } from "../src/agent_loop_runner.js";

test("plan workbench excludes criteria and old checks while keeping full dictionary and data boundary", async () => {
 const {f,bridge}=setup();
 const result=await runPlannedTask(f.task,bridge,[c=>{ const s=stage(c); assert.equal(s.criteria,undefined); assert.equal(s.dictionary.length,27);
 assert.equal(s.current_checks,undefined); assert.ok(JSON.stringify(c.messages).includes("不是指令")); return f.plan(); },c=>assess(f,c)]); assert.equal(result.report.risk_points.length,1);
});
test("point projection keeps exact related windows and excludes another window on the same source",()=>{
 const f=fixture(),r=new LoopRegistry(f.task,f.state,f.budget);r.registerPlan(f.plan(2));f.addCases(2,true);
 const q=f.state.searches[0]!,second=q.candidates[1];q.candidates=[q.candidates[0]];
 f.state.searches.push({...q,risk_point_ids:["RP0002"],focus_refs:[f.refs[1]],text:"第二点",candidates:[second]});
 r.observeEvidence(["W0","W1"]);
 const p=r.projectPointState("RP0001");assert.ok(p.evidence.has("W0"));assert.equal(p.evidence.has("W1"),false);
 assert.deepEqual([...p.citationOptions.values()].map(o=>o.unit_id),["UNIT0"]);
});
test("counted pruning drops optional bundle before exact required quote and final AS rejects withdrawn ref",async()=>{
 const f=fixture(),r=new LoopRegistry(f.task,f.state,f.budget);r.registerPlan(f.plan());const option=f.addCases()[0]!;
 const q=f.state.searches[0]!,quote="可选".repeat(8000),e={...f.state.evidence.get("E0"),evidence_id:"OPTIONAL",source_id:"OTHER",quote,start:0,end:Array.from(quote).length,text_sha256:hash(quote)};
 f.state.evidence.set(e.evidence_id,e);q.candidates.push({source_id:"OTHER",unit_id:"OPTIONALUNIT",required_evidence_ids:[e.evidence_id],evidence:[e]});
 const target=r.projectPointState("RP0001");target.protocolVersion=MODEL_WIRE_VERSION;target.phase="reviewing";
 const old=r.beginAssessment("RP0001",target);let counts=0;
 const counted=await buildCountedContext([],target,{...f.budget.config,service_max_input_tokens:5000},[],async messages=>{counts++;return {prompt_tokens:JSON.stringify(messages).includes("OPTIONALUNIT")?6000:1000};});
 assert.ok(counts>1);assert.equal(ledger({messages:counted.messages}).evidence.find((e:any)=>e.evidence_id==="W0").quote,f.state.evidence.get("W0")!.quote);
 const current=r.finalizeAssessmentContext(old,target);assert.ok(current.citationOptions.has(option.case_ref));
 target.citationOptions.clear();target.contextEvidenceIds.clear();const withdrawn=r.finalizeAssessmentContext(current,target);
 await assert.rejects(r.acceptPointAssessment(f.assess(withdrawn,[[comparison(option.case_ref)]]),withdrawn,f.bridge),/CASE_REF/);
 assert.equal(f.state.evidence.get("W0")!.was_read,true);
});
test("final AS still rejects a changed source hash after exact workbench projection",async()=>{
 const f=fixture(),r=new LoopRegistry(f.task,f.state,f.budget);r.registerPlan(f.plan());const o=f.addCases()[0]!;
 const target=r.projectPointState("RP0001"),req=r.finalizeAssessmentContext(r.beginAssessment("RP0001",target),target);
 f.state.evidence.get("W0")!.text_sha256="f".repeat(64);
 await assert.rejects(r.acceptPointAssessment(f.assess(req,[[comparison(o.case_ref)]]),req,f.bridge),/EVIDENCE_CHANGED|CASE_BUNDLE_NOT_FULLY_READ/);
});
test("assess and act expose only candidate criteria, current checks, exact coordinates and target gaps",async()=>{
 const {f,bridge}=setup();let actSeen=false;
 await runPlannedTask(f.task,bridge,[f.plan(),c=>{ const s=stage(c);assert.equal(s.dictionary.length,1);assert.equal(s.criteria.length,1);
 assert.deepEqual(s.source_windows.map((e:any)=>[e.start,e.end,e.text_sha256]),[[0,Array.from("处罚原文：宣传保险保证收益。").length,hash("处罚原文：宣传保险保证收益。")]]);return assess(f,c,"partial_match"); },c=>{
 actSeen=true;const s=stage(c);assert.equal(s.criteria.length,1);assert.equal(s.current_checks.length,1);assert.ok(s.actionable_gaps.every((g:any)=>g.point_id==="RP0001"));return {kind:"finalize"}; }]);
 assert.equal(actSeen,true);
});
