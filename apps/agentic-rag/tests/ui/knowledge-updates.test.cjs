const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const source = path.resolve(__dirname, '../../src/web/static/knowledge-updates.js');
const ui = fs.existsSync(source) ? require(source) : {};
const status = (extra = {}) => ({csrf_token:'token', settings:{crawl_enabled:false,usage_enabled:false,daily_hour:8}, configuration:{ready:true,generation_allowed:true}, selection:{}, batches:[], versions:[], jobs:[], ...extra});
function harness(initial = status()) {
  const calls = [], timers = [];
  let current = initial, visible = true, jobNumber = 0;
  let maintenance = {csrf_token:'token',live_ready:false,fixture_ready:false,publication_ready:false,events:[],jobs:[],runs:[],stages:{candidate:0,model_review:0,engineering_replay:0,build:0}};
  const request = async (url, options = {}) => {
    calls.push({url,...options});
    if (url === '/status') return current;
    if (url === '/maintenance/status') return maintenance;
    if (url === '/maintenance/import-batch') return {count:1,event_ids:['event-1'],publication_allowed:false};
    if (url === '/maintenance/runs') return {run_id:'run-1',status:'queued',publication_allowed:false};
    if (url.startsWith('/maintenance/changesets/')) return {changeset:{operations:[{before:{text:'旧条目'},after:{text:'<script>新条目</script>'}}]},review_runs:[{payload:{decision:'request_changes',feedback:'补证'}}]};
    if (url.startsWith('/maintenance/runs/')) return {run_id:'run-1',status:'failed',budget:{requests:2,settled_micro:0},publication_allowed:false};
    if (url.startsWith('/documents?')) return {items:[{digest:'abc',title:'原件'}],total:23};
    if (url.startsWith('/documents/')) return {text:'<script>原文</script>',has_more:true,next_cursor:10};
    if (options.method === 'PATCH') return {settings:options.body};
    return {job_id:'job-' + (++jobNumber),status:'queued'};
  };
  const controller = ui.createController({request, isVisible:()=>visible, setTimer:(fn,delay)=>{timers.push({fn,delay});return timers.length;},clearTimer:()=>{}});
  return {controller,calls,timers,setStatus:v=>current=v,setMaintenance:v=>maintenance={...maintenance,...v},setVisible:v=>visible=v};
}
test('初始化与刷新只读取状态，不写入关闭的设置', async()=>{
  assert.equal(typeof ui.createController,'function');
  const h=harness(); await h.controller.start();
  assert.deepEqual(h.calls.map(c=>[c.url,c.method||'GET']),[['/status','GET'],['/maintenance/status','GET']]);
  assert.equal(h.timers[0].delay,3000);
  h.controller.stop();
});
test('页面不可见时停止轮询，恢复后只读取状态', async()=>{
  const h=harness(); await h.controller.start(); h.setVisible(false);
  await h.timers[0].fn(); assert.equal(h.calls.length,2);
  h.setVisible(true); await h.controller.resume(); assert.equal(h.calls.length,4);
  assert.equal(h.calls[2].url,'/status');
});
test('采集和扩展使用开关分别写入，不相互开启', async()=>{
  const h=harness(status({selection:{version:'v1'}})); await h.controller.start();
  await h.controller.setCrawlEnabled(true);
  assert.deepEqual(h.calls.find(c=>c.method==='PATCH').body,{crawl_enabled:true});
  await h.controller.setUsageEnabled(false);
  assert.deepEqual(h.calls.find(c=>c.url==='/usage').body,{enabled:false});
  assert.equal(h.calls.filter(c=>c.url==='/activate').length,0);
});
test('异步受理不伪报完成，等待作业期间不重复提交', async()=>{
  const h=harness(); await h.controller.start(); await h.controller.importHistory();
  assert.match(h.controller.state.notice,/已受理/);
  assert.doesNotMatch(h.controller.state.notice,/已完成|导入成功/);
  await assert.rejects(h.controller.importHistory(),/已有任务/);
  assert.equal(h.calls.filter(c=>c.url==='/import-history').length,1);
});
test('采集或处理作业完成不自动激活版本或开启使用', async()=>{
  const h=harness(); await h.controller.start();
  await h.controller.processBatch({batch_id:'b1',cap_cny:1,max_documents:2,max_requests:5});
  h.setStatus(status({jobs:[{id:'job-1',kind:'process',status:'completed'}],versions:[{version:'v1',status:'ready'}]}));
  await h.controller.refresh();
  assert.equal(h.calls.filter(c=>['/activate','/usage'].includes(c.url)).length,0);
  assert.equal(h.controller.isLocked(),false);
});
test('仅配置就绪且允许生成时可处理，并要求显式正数预算', async()=>{
  const h=harness(status({configuration:{ready:false,generation_allowed:false,reason:'缺少基础快照'}})); await h.controller.start();
  await assert.rejects(h.controller.processBatch({batch_id:'b',cap_cny:1,max_documents:1,max_requests:1}),/缺少基础快照/);
  assert.equal(h.calls.length,2);
  for (const invalid of [{cap_cny:0,max_documents:1,max_requests:1},{cap_cny:1,max_documents:0,max_requests:1},{cap_cny:1,max_documents:1,max_requests:1.5}]) {
    assert.throws(()=>ui.validateProcess({batch_id:'b',...invalid}),/上限/);
  }
});
test('日期补采校验真实日期、先后顺序和1至100页上限', ()=>{
  assert.deepEqual(ui.validateCrawl({since:'2026-01-01',until:'2026-01-31',max_pages:'10'}),{since:'2026-01-01',until:'2026-01-31',max_pages:10});
  for (const invalid of [{since:'2026-02-30',until:'2026-03-01',max_pages:1},{since:'2026-03-02',until:'2026-03-01',max_pages:1},{since:'2026-03-01',until:'2026-03-01',max_pages:101}]) assert.throws(()=>ui.validateCrawl(invalid));
});
test('文书分页固定20条，原文游标读取保持文本内容',async()=>{
  const h=harness(); await h.controller.start();
  await h.controller.loadDocuments('b/1',20);
  assert.equal(h.calls.at(-1).url,'/documents?batch_id=b%2F1&offset=20&limit=20');
  assert.equal(h.controller.state.documents.total,23);
  const text=await h.controller.loadText('a/b',0);
  assert.equal(h.calls.at(-1).url,'/documents/a%2Fb?cursor=0');
  assert.equal(text.text,'<script>原文</script>');
});
test('写请求携带状态令牌并保留服务端错误',async()=>{
  const seen=[];
  const request=ui.createRequest(async(url,options)=>{seen.push({url,options});return {ok:false,status:409,json:async()=>({detail:'版本父节点已变化，请刷新'})};});
  await assert.rejects(request('/activate',{method:'POST',body:{version:'v2'},csrfToken:'abc'}),/版本父节点已变化/);
  assert.equal(seen[0].options.headers['X-Knowledge-CSRF'],'abc');
  assert.equal(seen[0].options.credentials,'same-origin');
});
test('未选择版本时禁止开启扩展，手动激活只提交指定版本',async()=>{
  const h=harness(status({versions:[{version:'v2',status:'ready'}]})); await h.controller.start();
  await assert.rejects(h.controller.setUsageEnabled(true),/先选择/);
  await h.controller.activateVersion('v2');
  assert.deepEqual(h.calls.find(c=>c.url==='/activate').body,{version:'v2'});
  assert.equal(h.calls.filter(c=>c.url==='/usage').length,0);
});
test('状态摘要区分用户选择和服务实际版本，并解释空闲换版',()=>{
  const message=ui.selectionText(status({settings:{usage_enabled:true},selection:{version:'v2',actual_version:'v1',pending_switch:true}}));
  assert.match(message,/v2/); assert.match(message,/v1/); assert.match(message,/空闲/);
});
test('页面动态内容采用文本节点，README保留独立管理入口',()=>{
  const js=fs.readFileSync(source,'utf8');
  assert.doesNotMatch(js,/\.innerHTML\s*=|insertAdjacentHTML/);
  const readme=fs.readFileSync(path.resolve(__dirname,'../../README.md'),'utf8');
  assert.match(readme,/`\/agent-rag\/knowledge`/);
  const html=fs.readFileSync(path.resolve(__dirname,'../../src/web/static/knowledge-updates.html'),'utf8');
  assert.match(html,/src="\/agent-rag\/knowledge-updates.js"/);
});
test('未完成补采从批次保存的相同日期区间继续，沿用每轮页数限制',()=>{
  const batch={kind:'crawl',status:'incomplete',metadata:{since:'2026-01-01',until:'2026-01-31',max_pages:2},details:{next_pages:{'1':3}}};
  assert.deepEqual(ui.resumeCrawlInput(batch),{since:'2026-01-01',until:'2026-01-31',max_pages:2});
  assert.equal(ui.resumeCrawlInput({...batch,status:'collected'}),null);
  assert.equal(ui.resumeCrawlInput({...batch,kind:'history'}),null);
});
test('历史导入和官网补采在构建未配置时仍可提交，未完整补采结束后允许继续',async()=>{
  const unconfigured=status({configuration:{ready:false,generation_allowed:false,reason:'基础快照未绑定'}});
  const h=harness(unconfigured); await h.controller.start();
  await h.controller.importHistory();
  h.setStatus({...unconfigured,jobs:[{id:'job-1',kind:'import_history',status:'completed'}]}); await h.controller.refresh();
  await h.controller.crawl({since:'2026-01-01',until:'2026-01-02',max_pages:2});
  h.setStatus({...unconfigured,jobs:[{id:'job-2',kind:'crawl',status:'completed',result:{complete:false,next_pages:{'1':3}}}]}); await h.controller.refresh();
  assert.match(h.controller.state.notice,/未完整完成/);
  assert.match(h.controller.state.notice,/水位未推进/);
  assert.equal(h.controller.isLocked(),false);
  await h.controller.crawl({since:'2026-01-01',until:'2026-01-02',max_pages:2});
  assert.equal(h.calls.filter(c=>c.url==='/crawl').length,2);
});
test('选择版本作业的状态不暗示已经切换或开启扩展使用',()=>{
  assert.match(ui.jobText({kind:'activate',status:'completed'}),/版本选定/);
});
test('处理未完整完成显示待处理数和计费结果，不误报采集水位',()=>{
  const message=ui.jobText({kind:'process',status:'completed',result:{complete:false,remaining_documents:3,generation_budget:{settled_cost_cny:0.12,reserved_cost_cny:0.03,cap_cny:1,requests:2,max_requests:5},review_queue:[{code:'DOCUMENT_NEEDS_REVIEW'}]}});
  assert.match(message,/批次尚有3份待处理\/待核，请查看结果/);
  assert.doesNotMatch(message,/水位|采集/);
  assert.match(message,/预算结果/);
  assert.match(message,/已结算0\.12元/);
  assert.match(message,/已预留0\.03元/);
  assert.match(message,/费用上限1元/);
  assert.match(message,/请求2\/5次/);
});
test('每日采集调度错误明确解释缺少补采起点，其余保留安全错误码',()=>{
  assert.equal(ui.schedulerText('BACKFILL_DATE_NOT_CONFIGURED'),'未配置有证据的补采起始日期，请通过本机配置。');
  assert.equal(ui.schedulerText('CRAWL_WORKER_FAILED'),'每日采集调度失败：CRAWL_WORKER_FAILED');
  assert.equal(ui.schedulerText(null),'');
});
test('绑定后需重启时提示指定本管理知识库，且不展示本地路径',()=>{
  const message=ui.selectionText(status({selection:{restart_required:true}}));
  assert.match(message,/绑定后.*重启审查服务/);
  assert.match(message,/指定本管理知识库注册表/);
  assert.doesNotMatch(message,/[A-Z]:[\\/]/);
});
test('页面不提供历史回退建议，关闭扩展使用回到固定基线',()=>{
  const html=fs.readFileSync(path.resolve(__dirname,'../../src/web/static/knowledge-updates.html'),'utf8');
  assert.doesNotMatch(html,/可选回之前人工选择过的版本/);
  assert.match(html,/拒绝历史回退/);
  assert.match(html,/关闭扩展使用可回到固定基线/);
});
test('真实页面绑定在未配置时保留采集导入按钮，并按配置控制处理按钮',async()=>{
  class Element {
    constructor(){this.children=[];this.value='';this.disabled=false;this.hidden=false;this._text='';}
    set textContent(value){this._text=String(value);this.children=[];}
    get textContent(){return this._text;}
    get firstChild(){return this.children[0];}
    append(...items){this.children.push(...items);}
    replaceChildren(...items){this.children=items;this.value='';}
  }
  const html=fs.readFileSync(path.resolve(__dirname,'../../src/web/static/knowledge-updates.html'),'utf8');
  const elements=new Map([...html.matchAll(/id="([^"]+)"/g)].map(match=>[match[1],new Element()]));
  const doc={hidden:true,getElementById:id=>elements.get(id),createElement:()=>new Element(),addEventListener:()=>{}};
  const calls=[];
  let value=status({scheduler_error:'BACKFILL_DATE_NOT_CONFIGURED',selection:{restart_required:true},configuration:{ready:false,generation_allowed:false,reason:'基础快照未绑定'},batches:[{id:'b1',kind:'crawl',status:'incomplete',metadata:{since:'2026-01-01',until:'2026-01-02',max_pages:2}}]});
  const oldFetch=globalThis.fetch,oldListener=globalThis.addEventListener;
  globalThis.fetch=async(url,options)=>{calls.push({url,options});return {ok:true,json:async()=>url.endsWith('/maintenance/status') ? {csrf_token:'token',live_ready:false,fixture_ready:false,publication_ready:false,live_reason:'缺少经验证的第二个独立模型适配器。',events:[{id:'e1',event_type:'new_source',status:'pending',reason:'<script>候选</script>'}],jobs:[],runs:[],stages:{candidate:1,model_review:0,engineering_replay:0,build:0}} : value};};
  globalThis.addEventListener=()=>{};
  let controller;
  try {
    controller=ui.mount(doc); await controller.refresh();
    assert.equal(elements.get('importHistory').disabled,false);
    assert.equal(elements.get('crawlSubmit').disabled,false);
    assert.equal(elements.get('crawlEnabled').disabled,false);
    assert.equal(elements.get('processSubmit').disabled,true);
    assert.match(elements.get('configuration').textContent,/基础快照未绑定/);
    assert.equal(elements.get('schedulerError').hidden,false);
    assert.match(elements.get('schedulerError').textContent,/未配置有证据的补采起始日期/);
    assert.match(elements.get('selection').textContent,/重启审查服务/);
    assert.equal(elements.get('crawlEnabled').checked,false);
    assert.equal(elements.get('usageEnabled').checked,false);
    assert.equal(elements.get('maintenanceRun').disabled,true);
    assert.match(elements.get('maintenanceConfiguration').textContent,/第二.*适配器/);
    assert.equal(elements.get('maintenanceEvents').children[0].children[3].textContent,'<script>候选</script>');
    elements.get('maintenanceBatch').value='b1'; elements.get('maintenanceBatch').onchange();
    assert.equal(elements.get('maintenanceImport').disabled,false);
    assert.equal(calls.every(call=>call.options.method==='GET'),true);
    elements.get('processBatch').value='b1'; elements.get('processBatch').onchange();
    value={...value,configuration:{ready:true,generation_allowed:false}}; await controller.refresh();
    assert.equal(elements.get('processSubmit').disabled,true);
    value={...value,scheduler_error:null,configuration:{ready:true,generation_allowed:true},batches:[...value.batches,{id:'b2',kind:'process',status:'partially_processed'},{id:'b3',kind:'process',status:'no_new_documents'}]}; await controller.refresh();
    assert.equal(elements.get('processSubmit').disabled,false);
    assert.equal(elements.get('schedulerError').hidden,true);
    assert.equal(elements.get('batches').children[1].children[2].textContent,'部分处理，仍有待处理或待核文书');
    assert.equal(elements.get('batches').children[2].children[2].textContent,'无新增待处理文书');
    const actions=elements.get('batches').children[0].children.at(-1).children;
    assert.equal(actions.some(item=>item.textContent==='继续本区间补采'),true);
  } finally {
    controller?.stop(); globalThis.fetch=oldFetch;
    if(oldListener===undefined) delete globalThis.addEventListener; else globalThis.addEventListener=oldListener;
  }
});

test('维护批次导入只登记待审事件，不调用旧处理、选版或使用接口',async()=>{
  const h=harness(); await h.controller.start();
  await h.controller.importMaintenanceBatch('batch-1');
  const call=h.calls.find(item=>item.url==='/maintenance/import-batch');
  assert.deepEqual(call.body,{batch_id:'batch-1'});
  assert.equal(call.csrfToken,'token');
  assert.match(h.controller.state.notice,/1.*待审/);
  assert.doesNotMatch(h.controller.state.notice,/审核通过|构建完成/);
  assert.equal(h.calls.some(item=>['/process','/activate','/usage'].includes(item.url)),false);
});

test('维护真实互审未配置时禁止提交，并说明缺少第二适配器',async()=>{
  const h=harness(); await h.controller.start();
  await assert.rejects(h.controller.runMaintenanceEvents(['event-1']),/第二.*适配器/);
  assert.equal(h.calls.some(item=>item.url==='/maintenance/runs'),false);
});

test('维护详情保留差异、模型意见和费用，工程回放不标为人工审核',async()=>{
  const h=harness(); await h.controller.start();
  await h.controller.loadChangeset('a/b');
  assert.equal(h.calls.at(-1).url,'/maintenance/changesets/a%2Fb');
  assert.equal(h.controller.state.maintenanceDetail.changeset.operations[0].after.text,'<script>新条目</script>');
  await h.controller.loadMaintenanceRun('r/1');
  assert.equal(h.calls.at(-1).url,'/maintenance/runs/r%2F1');
  assert.equal(h.controller.state.maintenanceDetail.budget.requests,2);
  const text=ui.maintenanceText({live_ready:false,stages:{candidate:2,model_review:0,engineering_replay:1,build:0}},status({selection:{version:'selected',actual_version:'actual'}}));
  for(const phrase of ['候选','模型审核','工程回放','构建','selected','actual']) assert.match(text,new RegExp(phrase));
  assert.doesNotMatch(text,/人工审核/);
});

test('处理登记待审的结果明确不等于审核通过或生成可选版本',()=>{
  const text=ui.jobText({kind:'process',status:'completed',result:{maintenance_queued:true,event_ids:['e1']}});
  assert.match(text,/待审/);
  assert.match(text,/尚未.*审核|未.*版本/);
});

test('维护页面有独立入口及未配置说明，显示详情先隐藏路径和凭据',()=>{
  const html=fs.readFileSync(path.resolve(__dirname,'../../src/web/static/knowledge-updates.html'),'utf8');
  for(const id of ['maintenanceImport','maintenanceRun','maintenanceEvents','maintenanceRuns','maintenanceDetail']) assert.match(html,new RegExp('id="'+id+'"'));
  const value=ui.safeDisplay({api_key:'secret',trace_path:'C:/private/file',reason:'原文 C:\\private\\file',operations:[{after:{text:'正文'}}]});
  assert.doesNotMatch(JSON.stringify(value),/secret|C:\/private|C:\\\\private/);
  assert.equal(value.operations[0].after.text,'正文');
});

test('维护状态单独失败仍允许采集，不沿用旧维护状态启动互审',async()=>{
  const calls=[];
  const controller=ui.createController({request:async(url,options={})=>{calls.push({url,...options});if(url==='/maintenance/status') throw new Error('维护读取失败');if(url==='/status') return status();return {job_id:'job1'};},setTimer:()=>1,clearTimer:()=>{}});
  await controller.start();
  assert.equal(controller.state.error,'');
  assert.match(controller.state.maintenanceError,/维护读取失败/);
  await controller.importHistory();
  assert.equal(calls.some(call=>call.url==='/import-history'),true);
  await assert.rejects(controller.importMaintenanceBatch('b1'),/先成功读取维护状态/);
  controller.stop();
});

test('范围建议不能被展示为已审核，旧关键词字段不替代分流结果',()=>{
  assert.match(ui.scopeText({contains_insurance:true}),/重新分流/);
  assert.match(ui.scopeText({scope_assessment:{category:'insurance_entity'}}),/任职线索.*尚未审核/);
  assert.match(ui.scopeText({scope_assessment:{category:'non_insurance_candidate'}}),/待核.*尚未审核/);
});
