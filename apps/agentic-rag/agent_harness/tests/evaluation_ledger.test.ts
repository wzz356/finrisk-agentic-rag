import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { canonical, hash, root, toolSchemasForProfile, type TaskInput } from '../src/contracts.js';
import { criteriaByteHash } from '../src/criteria.js';
import { wireSchemaHash } from '../src/model_protocol.js';
import { admissionPolicyHash } from '../src/request_admission.js';
import { evaluationConfig, evaluationProtocolHash } from '../src/evaluation.js';
import { runTask } from '../src/runner.js';

function task(): TaskInput {
  const definitions = JSON.parse(readFileSync(`${root}/contracts/agent_rag/risk-definitions.json`, 'utf8'));
  const text = '投保赠送购物卡。', user_context = {};
  const tools = toolSchemasForProfile('v2-hybrid-flash-v1');
  tools.search_cases.properties.queries.maxItems = 1;
  const promptPath = `${root}/configs/agent_rag/evaluation-system-prompt.txt`;
  const prompt = readFileSync(existsSync(promptPath) ? promptPath : `${root}/configs/agent_rag/system-prompt.txt`, 'utf8');
  return {input: {task_id:'zero-ledger', source_id:'INPUT:zero-ledger', text, user_context,
    text_sha256:hash(text), input_sha256:hash(canonical({text,user_context})), text_version:'offline',
    segments:[{segment_id:'S0001',source_id:'INPUT:zero-ledger',text,start:0,end:Array.from(text).length}]},
    definitions, snapshot:{snapshot_id:'zero-ledger-test',dictionary_sha256:definitions[0].dictionary_sha256},
    config:evaluationConfig('R1',{execution_mode:'live',profile_id:'live-deepseek-flash-v2-v1',
      retrieval_profile:'v2-hybrid-flash-v1',retrieval_reliability:true,max_queries_per_search:1,request_admission_policy:'v2-v1'}),
    execution_contract:{model_wire_sha256:wireSchemaHash,criteria_sha256:criteriaByteHash,
      prompt_sha256:hash(prompt),tool_contract_sha256:hash(canonical(tools)),
      request_admission_policy_sha256:admissionPolicyHash('v2-v1'),evaluation_protocol_sha256:evaluationProtocolHash}} as TaskInput;
}

async function isolated(path: string, action: () => Promise<void>) {
  const keys = ['FINRISK_AGENT_ALLOW_EVALUATION','FINRISK_AGENT_ALLOW_LIVE','FINRISK_AGENT_TRIAL_LEDGER','DEEPSEEK_API_KEY','DEEPSEEK_API_KEY_FILE'];
  const before = new Map(keys.map(k => [k,process.env[k]]));
  const priorFetch = globalThis.fetch; let egress = 0;
  Object.assign(process.env,{FINRISK_AGENT_ALLOW_EVALUATION:'1',FINRISK_AGENT_ALLOW_LIVE:'1',
    FINRISK_AGENT_TRIAL_LEDGER:path,DEEPSEEK_API_KEY:'sk-offline-fake-not-a-secret'});
  delete process.env.DEEPSEEK_API_KEY_FILE;
  globalThis.fetch = async () => { egress++; throw new Error('FORBIDDEN_OFFLINE_EGRESS'); };
  try { await action(); assert.equal(egress,0); }
  finally { globalThis.fetch = priorFetch; for(const key of keys) { const value=before.get(key); if(value===undefined) delete process.env[key]; else process.env[key]=value; } }
}

test('live V2 prefetch circuit persists a valid zero model ledger before the first tool', async () => {
  const path=join(mkdtempSync(join(tmpdir(),'finrisk-zero-ledger-')),'model.json'); let calls=0; let existedBeforeTool=false;
  await isolated(path, async () => {
    const result=await runTask(task(), {async call(_name:any,args:any) {
      calls++;
      existedBeforeTool=existsSync(path);
      return {schema_version:'finrisk-tools/1.1',status:'partial',data:{queries:args.queries.map((q:any)=>({...q,query_id:'Q1',status:'error',complete:false,candidates:[],error:{code:'RETRIEVAL_CIRCUIT_OPEN'}}))},
        meta:{snapshot_id:'zero-ledger-test',backend_query_attempts:1,cache_hits:0,external_retrieval_requests:0,
          transport_circuit_open:true,transport_failure_streak:3},error:null};
    },async verifyEvidence(ids:string[]) {return {valid_ids:ids,rejected_ids:[]};}} as any);
    assert.ok(existedBeforeTool,'zero ledger must exist before reranking can start');
    assert.equal(result.report.resources.physical_provider_requests,0);
    assert.equal(result.report.stop.reason,'tool_failure');
    const ledger=JSON.parse(readFileSync(path,'utf8'));
    assert.deepEqual(ledger.request_reservations,[]);
    assert.equal(ledger.settled_cost_micro_cny,0); assert.equal(ledger.reserved_cost_micro_cny,0);
  });
  assert.equal(calls,1);
});

test('zero ledger write failure blocks every tool and provider egress', async () => {
  const directory=mkdtempSync(join(tmpdir(),'finrisk-zero-fail-')), parent=join(directory,'file');
  writeFileSync(parent,'not a directory'); let calls=0;
  await isolated(join(parent,'ledger.json'),async () => {
    await assert.rejects(()=>runTask(task(),{async call(){calls++;throw new Error('TOOL_MUST_NOT_RUN');}} as any),/EEXIST|ENOTDIR/);
  });
  assert.equal(calls,0);
});

test('invalid input never creates an initialized ledger', async () => {
  const path=join(mkdtempSync(join(tmpdir(),'finrisk-zero-invalid-')),'model.json');
  await isolated(path,async()=>{
    const input=task();input.input.text_sha256='invalid';
    await assert.rejects(()=>runTask(input,{} as any),/INPUT_HASH_MISMATCH/);
    assert.equal(existsSync(path),false);
  });
});

test('live V2 evaluation cannot silently omit the new retrieval reliability gate', async () => {
  const path=join(mkdtempSync(join(tmpdir(),'finrisk-reliability-admission-')),'model.json');
  await isolated(path,async()=>{
    const input=task();delete input.config!.retrieval_reliability;
    await assert.rejects(()=>runTask(input,{async call(){throw new Error('TOOL_MUST_NOT_RUN');}} as any),/RETRIEVAL_RELIABILITY_BINDING_REQUIRED/);
    assert.equal(existsSync(path),false);
  });
});
