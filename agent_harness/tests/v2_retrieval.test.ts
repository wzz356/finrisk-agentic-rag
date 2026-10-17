import test from 'node:test';
import assert from 'node:assert/strict';
import * as protocol from '../src/model_protocol.js';
import { TaskState } from '../src/state.js';
import { sourceReadPlan } from '../src/retrieval_plan.js';
import { hash, canonical, root, type InputRecord } from '../src/contracts.js';
import { readFileSync } from 'node:fs';
import { runTask } from '../src/runner.js';
import { HttpToolBridge } from '../src/tool_bridge.js';
import { progressSnapshot } from '../src/progress.js';
import { Budget } from '../src/budget.js';
import { defaults } from '../src/contracts.js';

function input(): InputRecord {
  const text='购买保险送购物卡。';
  return { task_id:'v2-test',text,source_id:'INPUT:v2-test',text_sha256:hash(text),text_version:'1',
    input_sha256:hash(canonical({text,user_context:{}})),user_context:{},
    segments:[{segment_id:'S0001',text,start:0,end:Array.from(text).length,source_id:'INPUT:v2-test'}] };
}
const e1={evidence_id:'E1',source_id:'SRC',start:0,end:5,quote:'主体与行为',source_verified:true};
const e2={evidence_id:'E2',source_id:'SRC',start:5000,end:5005,quote:'合同外条件',source_verified:true};
const candidate={source_id:'SRC',rank:1,evidence:[e1,e2],required_evidence_ids:['E1','E2']};

test('V2 multi-span bundle requires every necessary anchor read and presented',()=>{
  const fn=(protocol as any).candidateBundleRead;
  assert.equal(typeof fn,'function','缺少复合证据核读判断');
  const state=new TaskState(input());
  state.evidence.set('E1',{...e1,was_read:true}); state.evidence.set('E2',e2);
  state.contextEvidenceIds.add('E1'); state.contextEvidenceIds.add('E2');
  assert.equal(fn(candidate,state),false);
  state.evidence.set('E2',{...e2,was_read:true});
  assert.equal(fn(candidate,state),true);
  state.contextEvidenceIds.delete('E2');
  assert.equal(fn(candidate,state,true),false);
});

test('V2 read plan keeps distinct necessary anchors from the same source',()=>{
  const plan=(sourceReadPlan as any)([{candidates:[candidate]}],3,()=>false);
  assert.deepEqual(plan.map((c:any)=>c.evidence[0].evidence_id),['E1','E2']);
  const remaining=(sourceReadPlan as any)([{candidates:[candidate]}],3,(e:any)=>e.evidence_id==='E1');
  assert.deepEqual(remaining.map((c:any)=>c.evidence[0].evidence_id),['E2']);
});

test('V2 progress cannot report a composite case read when a condition anchor is unread',()=>{
  const state=new TaskState(input());
  state.evidence.set('E1',{...e1,was_read:true});
  state.evidence.set('E2',e2);
  state.searches.push({text:'查询',risk_point_ids:[],focus_refs:[{segment_id:'S0001',quote:input().text,occurrence:1}],goal:'mechanism_search',complete:true,status:'ok',candidates:[candidate]});
  const progress=progressSnapshot(state,new Budget(defaults),'condition_comparison',1);
  assert.equal(progress.counts.pending_case_queries,1);
});

test('V2 final binds selected unit to the full read bundle rather than another case in the file',()=>{
  const definitions=JSON.parse(readFileSync(`${root}/contracts/agent_rag/risk-definitions.json`,'utf8'));
  const task:any={input:input(),snapshot:{snapshot_id:'v2',dictionary_sha256:definitions[0].dictionary_sha256},definitions,config:{retrieval_profile:'v2-hybrid-flash-v1'}};
  const state=new TaskState(task.input);
  const ref={segment_id:'S0001',quote:task.input.text,occurrence:1};
  const segments=[{segment_id:'S0001',status:'reviewed',reason:'测试'}];
  state.acceptAnalysis(protocol.normalizeModelOutput({kind:'analysis',segments,points:[{input_refs:[ref],candidate_tags:['赠送利益'],facts:[]}]},'analysis',state,task));
  state.searches.push({text:'查询',risk_point_ids:['RP0001'],focus_refs:[ref],goal:'mechanism_search',complete:true,status:'ok',candidates:[{...candidate,unit_id:'U1'}]});
  state.evidence.set('E1',{...e1,was_read:true});state.evidence.set('E2',{...e2,was_read:true});
  state.contextEvidenceIds.add('E1');state.contextEvidenceIds.add('E2');
  const match={unit_id:'U1',source_id:'SRC',evidence_ids:['E1','E2'],same_points:['相同机制'],differences:['当前条件未知'],match_status:'partial_match',supports_tags:['赠送利益']};
  const value={kind:'final',segments,points:[{risk_point_id:'RP0001',tag_checks:[],case_matches:[match]}]};
  assert.doesNotThrow(()=>protocol.normalizeModelOutput(value,'final',state,task));
  const other={evidence_id:'E3',source_id:'SRC',start:9000,end:9005,quote:'另一项行为',source_verified:true};
  state.evidence.set('E3',other);state.contextEvidenceIds.add('E3');
  const withCitations=(ids:string[])=>({...value,points:[{...value.points[0],case_matches:[{...match,evidence_ids:ids}]}]});
  assert.throws(()=>protocol.normalizeModelOutput(withCitations(['E3']),'final',state,task),/CASE_CITATION_NOT_READ/);
  state.evidence.set('E3',{...other,was_read:true});
  assert.throws(()=>protocol.normalizeModelOutput(withCitations(['E3']),'final',state,task),/CASE_EVIDENCE_UNIT_BINDING_MISMATCH/);
  assert.throws(()=>protocol.normalizeModelOutput(withCitations(['E1']),'final',state,task),/CASE_CITATION_BUNDLE_INCOMPLETE/);
  assert.throws(()=>protocol.normalizeModelOutput(withCitations(['E1','E2','E3']),'final',state,task),/CASE_EVIDENCE_UNIT_BINDING_MISMATCH/);
  const window={evidence_id:'EW',source_id:'SRC',start:0,end:5010,quote:'真实回源窗口',source_verified:true,was_read:true};
  state.evidence.set('EW',window);state.contextEvidenceIds.add('EW');
  assert.doesNotThrow(()=>protocol.normalizeModelOutput(withCitations(['EW']),'final',state,task));
  state.evidence.delete('EW');state.contextEvidenceIds.delete('EW');
  state.evidence.set('E2',e2);
  assert.throws(()=>protocol.normalizeModelOutput(value,'final',state,task),/CASE_BUNDLE_NOT_FULLY_READ/);
  state.evidence.set('E2',{...e2,was_read:true});
  assert.throws(()=>protocol.normalizeModelOutput({...value,points:[{...value.points[0],case_matches:[{...match,unit_id:'other'}]}]},'final',state,task),/CASE_UNIT_BINDING_MISMATCH/);
});

test('V2 retrieval profile runs through existing harness in mock without physical requests',async()=>{
  const definitions=JSON.parse(readFileSync(`${root}/contracts/agent_rag/risk-definitions.json`,'utf8'));
  const task:any={input:input(),snapshot:{snapshot_id:'v2-test',dictionary_sha256:definitions[0].dictionary_sha256,retrieval_profile:'v2-hybrid-flash-v1'},definitions,
    config:{retrieval_profile:'v2-hybrid-flash-v1',profile_id:'mock-v2-flash-replay-v1',max_queries_per_search:1,max_search_calls:2,max_external_retrieval_requests:240}};
  let cancellations=0;
  const bridge:any={async cancelRetrieval(){cancellations++;return {external_retrieval_requests:0};},async verifyEvidence(){return {valid_ids:[],rejected_ids:[]};},async call(){throw new Error('unexpected tool');}};
  const segments=[{segment_id:'S0001',status:'reviewed',reason:'测试输入审阅'}];
  const result=await runTask(task,bridge,[{kind:'analysis',segments,points:[]},{kind:'final',segments,points:[]}]);
  assert.equal(result.report.task_status,'completed');
  assert.equal(result.report.resources.physical_provider_requests,0);
  assert.equal(cancellations,1);
});

test('HTTP search propagates timeout and supports authenticated backend cancellation',async()=>{
  const prior=globalThis.fetch;
  const calls:any[]=[];
  globalThis.fetch=(async (url:any,init:any)=>{
    calls.push({url:String(url),body:JSON.parse(init.body),headers:init.headers});
    return new Response(JSON.stringify({cancelled:true,external_retrieval_requests:0}),{status:200});
  }) as any;
  try {
    const bridge=new HttpToolBridge('task','http://127.0.0.1:9999','secret');
    await bridge.call('search_cases',{queries:[]},2,1234);
    assert.equal(calls[0].body.timeout_ms,1234);
    assert.equal(typeof (bridge as any).cancelRetrieval,'function');
    await (bridge as any).cancelRetrieval();
    assert.ok(calls[1].url.endsWith('/internal/agent-rag/retrieval/cancel'));
    assert.equal(calls[1].body.task_id,'task');
  } finally {globalThis.fetch=prior;}
});

test('V2 live agent cannot bypass the model fee ledger',async()=>{
  const allow=process.env.FINRISK_AGENT_ALLOW_LIVE;
  const ledger=process.env.FINRISK_AGENT_TRIAL_LEDGER;
  process.env.FINRISK_AGENT_ALLOW_LIVE='1';
  delete process.env.FINRISK_AGENT_TRIAL_LEDGER;
  const definitions=JSON.parse(readFileSync(`${root}/contracts/agent_rag/risk-definitions.json`,'utf8'));
  try {
    await assert.rejects(()=>runTask({input:input(),snapshot:{snapshot_id:'v2',dictionary_sha256:definitions[0].dictionary_sha256},definitions,config:{execution_mode:'live',profile_id:'live-deepseek-flash-v2-v1',retrieval_profile:'v2-hybrid-flash-v1'}},{} as any),/MODEL_TRIAL_LEDGER_REQUIRED/);
  } finally {
    if(allow===undefined)delete process.env.FINRISK_AGENT_ALLOW_LIVE;else process.env.FINRISK_AGENT_ALLOW_LIVE=allow;
    if(ledger===undefined)delete process.env.FINRISK_AGENT_TRIAL_LEDGER;else process.env.FINRISK_AGENT_TRIAL_LEDGER=ledger;
  }
});
