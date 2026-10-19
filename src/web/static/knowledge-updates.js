(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.FinRiskKnowledge = api;
  if (typeof document === 'object' && document.getElementById('knowledgeManager')) api.mount(document);
})(typeof globalThis === 'object' ? globalThis : this, function () {
  'use strict';
  const active = new Set(['queued', 'pending', 'running', 'processing']);
  const terminal = new Set(['completed', 'failed', 'partial', 'incomplete', 'cancelled', 'interrupted', 'rejected']);
  const ready = new Set(['ready']);
  const names = {queued:'排队中',pending:'等待中',running:'执行中',processing:'处理中',collecting:'采集中',completed:'完成',failed:'失败',partial:'未完整完成',incomplete:'未完整完成',cancelled:'已取消',interrupted:'已中断',rejected:'已拒绝',ready:'待选定',validated:'已校验',built:'已构建',collected:'已采集',imported:'已导入',processed:'已处理',partially_processed:'部分处理，仍有待处理或待核文书',no_new_documents:'无新增待处理文书',needs_review:'需要人工复核',baseline_duplicate:'基线已有正文',crawl:'官网补采',official_crawl:'官网补采',scheduled_crawl:'每日采集',legacy_import:'历史导入',import_history:'历史导入',history:'历史导入',process:'原件处理',activate:'版本选定',usage:'扩展使用'};
  const label = value => names[value] || String(value || '未记录');
  function safeDisplay(value) {
    if (Array.isArray(value)) return value.map(safeDisplay);
    if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).filter(([key])=>!/key|secret|credential|password|authorization|token|trusted_runs|approval_context|(^|_)path$|(^|_)root$/i.test(key)).map(([key,item])=>[key,safeDisplay(item)]));
    if (typeof value === 'string') return value.replace(/[A-Za-z]:[\\/][^\s"<>]*|\\\\[^\s"<>]+|(?:^|\s)\/(?:Users|home|tmp|var|etc)\/[^\s"<>]*/g,'[本地路径已隐藏]');
    return value;
  }
  function maintenanceText(value, collection) {
    const stages = value?.stages || {};
    return `候选事件：${stages.candidate ?? 0}；模型审核通过：${stages.model_review ?? 0}；工程回放通过：${stages.engineering_replay ?? 0}；已构建：${stages.build ?? 0}。已选版本：${selectedVersion(collection) || '未选择'}；服务实际版本：${collection?.selection?.actual_version || '未报告'}。工程回放不能用于生产发布。`;
  }
  function errorText(value) {
    if (typeof value === 'string') return value;
    if (value && typeof value === 'object') return value.message || value.reason || value.code || JSON.stringify(value);
    return '未提供错误详情';
  }
  function createRequest(fetcher, prefix = '/api/agent-rag/knowledge') {
    return async (path, options = {}) => {
      const method = options.method || 'GET';
      const headers = {'Accept':'application/json'};
      const init = {method, headers, credentials:'same-origin', cache:'no-store'};
      if (method !== 'GET') {
        if (!options.csrfToken) throw new Error('缺少管理令牌，请刷新状态后重试。');
        headers['Content-Type'] = 'application/json';
        headers['X-Knowledge-CSRF'] = options.csrfToken;
        init.body = JSON.stringify(options.body || {});
      }
      let response;
      try { response = await fetcher(prefix + path, init); }
      catch (_) { throw new Error('无法连接管理服务，请检查服务是否运行。'); }
      let value;
      try { value = await response.json(); }
      catch (_) { throw new Error(`管理服务返回了无效响应（${response.status}）。`); }
      if (!response.ok) throw new Error(errorText(value.detail || value.error || value));
      return value;
    };
  }
  function validDate(value) {
    if (!/^\d{4}-\d{2}-\d{2}$/.test(value || '')) return false;
    const date = new Date(value + 'T00:00:00Z');
    return Number.isFinite(date.getTime()) && date.toISOString().slice(0,10) === value;
  }
  function validateCrawl(input) {
    const since = input.since, until = input.until, pages = Number(input.max_pages);
    if (!validDate(since) || !validDate(until)) throw new Error('请选择有效的起止日期。');
    if (since > until) throw new Error('开始日期不能晚于结束日期。');
    if (!Number.isInteger(pages) || pages < 1 || pages > 100) throw new Error('采集页数上限须为1至100的整数。');
    return {since, until, max_pages:pages};
  }
  function validateProcess(input) {
    const cap = Number(input.cap_cny), documents = Number(input.max_documents), requests = Number(input.max_requests);
    if (!input.batch_id) throw new Error('请先选择原件批次。');
    if (!Number.isFinite(cap) || cap <= 0) throw new Error('请填写大于0的费用上限。');
    if (!Number.isSafeInteger(documents) || documents <= 0) throw new Error('文书数上限须为大于0的整数。');
    if (!Number.isSafeInteger(requests) || requests <= 0) throw new Error('模型请求上限须为大于0的整数。');
    return {batch_id:String(input.batch_id),cap_cny:cap,max_documents:documents,max_requests:requests};
  }
  function resumeCrawlInput(batch) {
    if (!['crawl','official_crawl','scheduled_crawl'].includes(batch.kind) || !['incomplete','partial','failed','interrupted'].includes(batch.status)) return null;
    const input = {...(batch.details || {}),...(batch.metadata || {})};
    try { return validateCrawl({since:input.since,until:input.until,max_pages:input.max_pages ?? 10}); }
    catch (_) { return null; }
  }
  function selectedVersion(status) { return status?.selection?.version || status?.selection?.selected_version || ''; }
  function schedulerText(code) {
    if (!code) return '';
    if (code === 'BACKFILL_DATE_NOT_CONFIGURED') return '未配置有证据的补采起始日期，请通过本机配置。';
    return '每日采集调度失败：' + String(code);
  }
  function selectionText(status) {
    const selection = status?.selection || {}, selected = selectedVersion(status);
    const actual = selection.actual_version || selection.current_version || selection.active_version;
    const enabled = status?.settings?.usage_enabled === true;
    const pieces = [`扩展使用${enabled ? '已开启' : '已关闭'}`, `已选累计版本：${selected || '尚未选择'}`, `服务实际版本：${actual || '服务尚未报告'}`];
    if (!enabled) pieces.push('目标为固定基线，已选版本仍保留');
    if (selection.pending_switch || selection.pending || (enabled && selected && actual && actual !== selected)) pieces.push('版本切换可能等待当前任务结束、服务空闲后生效');
    if (selection.restart_required) pieces.push('绑定后请重启审查服务，并指定本管理知识库注册表');
    pieces.push('正在进行的审查保持原快照，下一次审查使用实际生效版本');
    return pieces.join('；') + '。';
  }
  function jobText(job) {
    let text = `${label(job.kind)} · ${label(job.status)}`;
    if (job.error) return text + '：' + errorText(job.error);
    const result = job.result || {};
    if (result.maintenance_queued) text += `：已登记${result.event_ids?.length ?? result.count ?? '若干'}个待审事件；尚未通过模型审核，未生成可选版本。`;
    if (result.complete === false || result.incomplete === true || result.status === 'incomplete' || result.status === 'partial') {
      if (job.kind === 'process') text += `：批次尚有${result.remaining_documents ?? result.review_queue?.length ?? '若干'}份待处理/待核，请查看结果`;
      else if (['crawl','official_crawl','scheduled_crawl'].includes(job.kind)) text += '：本轮未完整完成，成功水位未推进，请查看结果并续跑';
      else text += '：本轮未完整完成，请查看结果';
    }
    if (job.kind === 'process' && result.generation_budget) {
      const budget = result.generation_budget, pieces = [];
      if (budget.settled_cost_cny != null) pieces.push(`已结算${budget.settled_cost_cny}元`);
      if (budget.reserved_cost_cny != null) pieces.push(`已预留${budget.reserved_cost_cny}元`);
      if (budget.cap_cny != null) pieces.push(`费用上限${budget.cap_cny}元`);
      if (budget.requests != null) pieces.push(`请求${budget.requests}/${budget.max_requests ?? '未报告'}次`);
      if (budget.usage_unknown_requests) pieces.push(`用量待核${budget.usage_unknown_requests}次`);
      text += '；预算结果：' + (pieces.join('，') || '请查看完整结果');
    }
    if (result.reason) text += '：' + errorText(result.reason);
    return text;
  }
  function createController(deps) {
    const request = deps.request;
    const emit = () => { if (deps.onChange) deps.onChange(state); };
    const setTimer = deps.setTimer || setTimeout, clearTimer = deps.clearTimer || clearTimeout;
    const visible = deps.isVisible || (() => true);
    const state = {status:null,error:'',notice:'',busy:false,maintenance:null,maintenanceError:'',maintenanceBusy:false,maintenanceDetail:null,documents:{items:[],total:0,offset:0,batch_id:''}};
    const pending = new Set();
    let timer = null, running = false, refreshPromise = null, documentRequest = 0;
    function isLocked() { return state.busy || pending.size > 0 || (state.status?.jobs || []).some(job => active.has(job.status)); }
    function schedule() {
      if (timer !== null) clearTimer(timer);
      timer = null;
      if (running && visible()) timer = setTimer(async () => { timer = null; if (visible()) await refresh().catch(() => {}); }, 3000);
    }
    async function refresh() {
      if (refreshPromise) return refreshPromise;
      refreshPromise = (async () => {
        try {
          const value = await request('/status');
          state.status = value;
          state.error = '';
          try { state.maintenance = await request('/maintenance/status'); state.maintenanceError = ''; }
          catch (error) { state.maintenanceError = error.message; }
          for (const job of value.jobs || []) {
            if (pending.has(job.id) && terminal.has(job.status)) {
              pending.delete(job.id);
              state.notice = jobText(job) + (job.kind === 'activate' ? '；已选版本请查看当前状态，实际切换可能等待空闲。' : job.kind === 'usage' ? '；请以服务实际版本为准。' : '；未自动选定版本或开启扩展使用。');
            }
          }
          emit();
          return value;
        } catch (error) { state.error = error.message; emit(); throw error; }
      })();
      try { return await refreshPromise; }
      finally { refreshPromise = null; schedule(); }
    }
    async function write(path, body, method = 'POST', allowDuringJob = false) {
      if (!state.status || state.error) throw new Error('请先成功读取管理状态后再操作。');
      if (state.busy || (!allowDuringJob && isLocked())) throw new Error('已有任务正在执行或等待结果，请勿重复提交。');
      state.busy = true; state.notice = '正在提交……'; emit();
      try {
        const result = await request(path,{method,body,csrfToken:state.status.csrf_token});
        if (method === 'POST') {
          if (!result.job_id) throw new Error('服务未返回作业编号，请刷新核实，勿重复提交。');
          pending.add(result.job_id);
          state.notice = `任务已受理（${result.job_id}），等待后台结果；尚未确认完成。`;
        } else state.notice = '采集开关已更新，请以最新状态为准。';
        await refresh().catch(() => {});
        return result;
      } catch (error) { state.notice = '操作未确认完成：' + error.message; throw error; }
      finally { state.busy = false; emit(); }
    }
    async function maintenanceWrite(path, body, notice) {
      if (!state.maintenance || state.maintenanceError) throw new Error('请先成功读取维护状态后再操作。');
      if (state.maintenanceBusy) throw new Error('维护操作正在提交，请勿重复操作。');
      state.maintenanceBusy = true; emit();
      try {
        const result = await request('/maintenance'+path,{method:'POST',body,csrfToken:state.maintenance.csrf_token || state.status?.csrf_token});
        state.notice = notice(result);
        await refresh();
        return result;
      } finally { state.maintenanceBusy = false; emit(); }
    }
    async function maintenanceDetail(path) {
      const value = await request('/maintenance'+path);
      state.maintenanceDetail = value; emit(); return value;
    }
    return {
      state,isLocked,refresh,
      async start() { running = true; return refresh(); },
      stop() { running = false; if (timer !== null) clearTimer(timer); timer = null; },
      async resume() { if (running && visible()) return refresh(); schedule(); },
      setCrawlEnabled(enabled) { return write('/settings',{crawl_enabled:enabled === true},'PATCH',true); },
      async setUsageEnabled(enabled) {
        if (enabled && !selectedVersion(state.status)) throw new Error('请先选择并手动选定一个累计版本。');
        return write('/usage',{enabled:enabled === true});
      },
      importHistory() { return write('/import-history',{}); },
      async importMaintenanceBatch(batch_id) {
        if (!batch_id) throw new Error('请选择需要登记的原件批次。');
        return maintenanceWrite('/import-batch',{batch_id},result=>`已登记${result.count ?? result.event_ids?.length ?? 0}个待审事件；尚未通过模型审核，未构建或选定版本。`);
      },
      async runMaintenanceEvents(event_ids, execution_mode = 'live') {
        if (execution_mode === 'live' && state.maintenance?.live_ready !== true) throw new Error(state.maintenance?.live_reason || '缺少经验证的第二个独立模型适配器，真实互审暂不可用。');
        if (execution_mode !== 'live' && state.maintenance?.fixture_ready !== true) throw new Error('工程回放脚本尚未配置，请使用离线测试入口。');
        if (!Array.isArray(event_ids) || !event_ids.length) throw new Error('请选择待审事件。');
        return maintenanceWrite('/runs',{event_ids,execution_mode},result=>`维护作业已受理（${result.run_id}），请查看后台记录；尚未确认审核结果。`);
      },
      loadChangeset(event_id) { return maintenanceDetail(`/changesets/${encodeURIComponent(event_id)}`); },
      loadMaintenanceRun(run_id) { return maintenanceDetail(`/runs/${encodeURIComponent(run_id)}`); },
      cancelMaintenanceRun(run_id) { return maintenanceWrite(`/runs/${encodeURIComponent(run_id)}/cancel`,{},()=> '已请求取消维护作业，请刷新查看最终状态；已用预算保留。'); },
      async crawl(input) { return write('/crawl',validateCrawl(input)); },
      async processBatch(input) {
        const config = state.status?.configuration;
        if (!config?.ready || !config?.generation_allowed) throw new Error(config?.reason || '处理配置未就绪，或尚未允许调用生成模型。');
        return write('/process',validateProcess(input));
      },
      async activateVersion(version) {
        const candidate = state.status?.versions?.find(item => item.version === version);
        if (!candidate || !ready.has(candidate.status)) throw new Error('请选择已校验、可以选定的累计版本。');
        return write('/activate',{version});
      },
      async loadDocuments(batch_id, offset = 0) {
        const id = ++documentRequest;
        const value = await request(`/documents?batch_id=${encodeURIComponent(batch_id || '')}&offset=${Math.max(0,offset)}&limit=20`);
        if (id === documentRequest) { state.documents = {items:value.items || [],total:value.total || 0,offset:Math.max(0,offset),batch_id:batch_id || ''}; emit(); }
        return value;
      },
      loadText(digest, cursor = 0) { return request(`/documents/${encodeURIComponent(digest)}?cursor=${encodeURIComponent(cursor)}`); }
    };
  }
  function mount(doc) {
    const el = id => doc.getElementById(id);
    function node(tag,text,className) { const item = doc.createElement(tag); if (text !== undefined) item.textContent = text; if (className) item.className = className; return item; }
    function button(text, action, disabled) { const item = node('button',text); item.type = 'button'; item.disabled = !!disabled; item.onclick = () => perform(action); return item; }
    function safeLink(text, url) {
      try { const parsed = new URL(url); if (['https:','http:'].includes(parsed.protocol)) { const a = node('a',text); a.href = parsed.href; a.target = '_blank'; a.rel = 'noopener noreferrer'; return a; } } catch (_) {}
      return node('span',text);
    }
    let documentBusy = false, textBusy = false, textDigest = '', textCursor = null;
    const controller = createController({request:createRequest(globalThis.fetch.bind(globalThis)),isVisible:()=>!doc.hidden,onChange:render});
    async function perform(action) {
      el('actionError').textContent = '';
      try { await action(); }
      catch (error) { el('actionError').textContent = error.message; }
      finally { render(controller.state); }
    }
    function updateOptions(select, entries, emptyText) {
      const previous = select.value;
      select.replaceChildren(node('option',emptyText)); select.firstChild.value = '';
      for (const [value,text] of entries) { const option = node('option',text); option.value = value; select.append(option); }
      if (entries.some(([value]) => value === previous)) select.value = previous;
    }
    function render(state) {
      const status = state.status, unavailable = !status || !!state.error;
      el('connection').textContent = state.error ? '状态读取失败：' + state.error + '；未确认任何后台操作成功。' : status ? '管理服务已连接；页面可见时每3秒刷新状态。' : '正在读取管理状态……';
      el('connection').className = state.error ? 'error' : 'small';
      el('actionStatus').textContent = state.notice;
      const locked = unavailable || controller.isLocked();
      el('crawlEnabled').checked = status?.settings?.crawl_enabled === true;
      el('usageEnabled').checked = status?.settings?.usage_enabled === true;
      el('crawlEnabled').disabled = unavailable || state.busy;
      el('usageEnabled').disabled = locked || (!selectedVersion(status) && !status?.settings?.usage_enabled);
      el('schedule').textContent = `每天北京时间${String(status?.settings?.daily_hour ?? 8).padStart(2,'0')}:00采集，仅在后台服务运行时执行；关闭开关阻止新轮次。自动采集不会处理原件或切换版本。`;
      el('schedulerError').textContent = schedulerText(status?.scheduler_error);
      el('schedulerError').hidden = !status?.scheduler_error;
      el('selection').textContent = selectionText(status);
      const config = status?.configuration || {};
      el('configuration').textContent = config.ready && config.generation_allowed ? '兼容批次入口只登记待审事件；真实互审与生产发布以维护区状态为准。' : '兼容批次入口暂不可用：' + (config.reason || '配置未就绪。') + '；仍可在维护区登记待审事件。';
      el('configuration').className = config.ready && config.generation_allowed ? 'small' : 'notice';
      el('watermark').textContent = '最近完整成功采集：' + (status?.last_successful_crawl ? (typeof status.last_successful_crawl === 'string' ? status.last_successful_crawl : JSON.stringify(status.last_successful_crawl)) : '暂无记录；失败或限页截断不视为完整成功。');
      const counts = status?.counts || {};
      const countNames = {documents:'原件',batches:'批次',versions:'版本',sources:'来源记录',raw_versions:'原始字节版本',pending_documents:'待处理原件',new_documents:'新增原件',duplicates:'重复原件',baseline_duplicates:'基线重复'};
      el('counts').replaceChildren(...Object.entries(counts).map(([key,value])=>node('span',`${countNames[key] || key}：${value}`,'tag')));
      el('importHistory').disabled = locked; el('crawlSubmit').disabled = locked;
      const batches = status?.batches || [];
      updateOptions(el('processBatch'),batches.map(batch=>[batch.id,`${batch.id} · ${label(batch.kind)} · ${label(batch.status)}`]),'选择待处理批次');
      el('processSubmit').disabled = locked || !config.ready || !config.generation_allowed || !el('processBatch').value;
      el('batches').replaceChildren();
      for (const batch of batches) {
        const tr = node('tr');
        tr.append(node('td',batch.id),node('td',label(batch.kind)),node('td',label(batch.status)),node('td',batch.created_at || '未记录'),node('td',`${batch.documents ?? 0} / ${batch.new_documents ?? 0} / ${batch.duplicates ?? 0} / ${batch.baseline_duplicates ?? 0}`));
        const action = node('td'); action.append(button('查看原件',()=>loadDocuments(batch.id,0),documentBusy));
        const resume = resumeCrawlInput(batch);
        if (resume) {
          action.append(node('p',`${resume.since} 至 ${resume.until} · 每栏目最多${resume.max_pages}页`,'small'));
          action.append(button('继续本区间补采',()=>controller.crawl(resume),locked));
        }
        const details = node('details'); details.append(node('summary','批次详情'),node('pre',JSON.stringify(safeDisplay({metadata:batch.metadata || {},result:batch.details || {}}),null,2))); action.append(details); tr.append(action); el('batches').append(tr);
      }
      el('batchEmpty').hidden = batches.length > 0;
      const versions = status?.versions || [];
      updateOptions(el('versionSelect'),versions.filter(v=>ready.has(v.status) && v.version !== selectedVersion(status)).map(v=>[v.version,`${v.version} · ${v.source_count ?? 0}份来源 · 新增${v.new_units ?? 0}单元`]),'选择可用版本');
      el('activateSubmit').disabled = locked || !el('versionSelect').value;
      el('versions').replaceChildren();
      for (const version of versions) {
        const tr = node('tr'); tr.append(node('td',version.version),node('td',version.version === selectedVersion(status) ? '已选择' : label(version.status)),node('td',version.source_count ?? '未记录'),node('td',version.new_units ?? '未记录'),node('td',version.parent_snapshot_id || '基线 / 未报告')); el('versions').append(tr);
      }
      el('versionEmpty').hidden = versions.length > 0;
      el('jobs').replaceChildren();
      for (const job of status?.jobs || []) {
        const item = node('li'); item.append(node('p',`${jobText(job)} · ${job.created_at || ''}`,['failed','partial','incomplete'].includes(job.status) ? 'error' : ''));
        const details = node('details'); details.append(node('summary',`查看结果 · ${job.id}`),node('pre',JSON.stringify(safeDisplay(job.result || job.error || {}),null,2))); item.append(details); el('jobs').append(item);
      }
      el('jobEmpty').hidden = (status?.jobs || []).length > 0;
      renderMaintenance(state, batches);
      renderDocuments(state.documents);
    }
    function renderMaintenance(state, batches) {
      const value = state.maintenance, blocked = !value || !!state.maintenanceError || state.maintenanceBusy;
      el('maintenanceSummary').textContent = maintenanceText(value,state.status);
      el('maintenanceConfiguration').textContent = state.maintenanceError ? '维护状态读取失败：'+state.maintenanceError : (value?.live_ready ? '真实互审适配器已就绪。' : value?.live_reason || '缺少经验证的第二个独立模型适配器，真实互审暂不可用。');
      el('maintenancePublication').textContent = value?.publication_reason || '生产发布与版本选择适配尚未配置。';
      updateOptions(el('maintenanceBatch'),batches.map(batch=>[batch.id,`${batch.id} · ${label(batch.status)}`]),'选择原件批次');
      el('maintenanceImport').disabled = blocked || !el('maintenanceBatch').value;
      const events = value?.events || [], jobs = value?.jobs || [];
      updateOptions(el('maintenanceEvent'),events.map(item=>[item.id,`${item.id} · ${label(item.status)}`]),'选择待审事件');
      el('maintenanceRun').disabled = blocked || value?.live_ready !== true || !el('maintenanceEvent').value;
      el('maintenanceEvents').replaceChildren();
      for (const item of events) {
        const job = jobs.filter(job=>job.event_id === item.id).at(-1);
        const tr = node('tr'), action = node('td');
        const maintenanceNames = {pending:'候选待审',approved:'模型审核通过',engineering_complete:'工程回放通过',failed:'失败',proposing:'提议中',needs_evidence:'需要补充证据',rejected:'模型拒绝',needs_review:'模型审核待定'};
        tr.append(node('td',item.id),node('td',item.event_type),node('td',maintenanceNames[job?.status || item.status] || label(job?.status || item.status)),node('td',safeDisplay(item.reason)));
        action.append(button('查看条目、差异与模型意见',()=>controller.loadChangeset(item.id),!job)); tr.append(action); el('maintenanceEvents').append(tr);
      }
      el('maintenanceEventEmpty').hidden = events.length > 0;
      el('maintenanceRuns').replaceChildren();
      for (const run of value?.runs || []) {
        const item = node('li'), id = run.run_id || run.id;
        item.append(node('p',`${id} · ${run.execution_mode === 'fixture' ? '工程回放' : '模型互审'} · ${label(run.status)}`),button('查看意见与费用',()=>controller.loadMaintenanceRun(id),!id));
        if (active.has(run.status)) item.append(button('请求取消',()=>controller.cancelMaintenanceRun(id),blocked));
        el('maintenanceRuns').append(item);
      }
      el('maintenanceRunEmpty').hidden = (value?.runs || []).length > 0;
      el('maintenanceDetail').textContent = state.maintenanceDetail ? JSON.stringify(safeDisplay(state.maintenanceDetail),null,2) : '选择已有维护记录，查看条目变更、模型意见、请求次数及结算/预留费用。';
    }
    function renderDocuments(value) {
      el('documentPage').textContent = `批次：${value.batch_id || '未选择'} · 共${value.total}份 · ${value.total ? value.offset + 1 : 0}–${Math.min(value.offset + 20,value.total)}`;
      el('previousPage').disabled = documentBusy || value.offset === 0;
      el('nextPage').disabled = documentBusy || value.offset + 20 >= value.total;
      el('documents').replaceChildren();
      for (const item of value.items) {
        const meta = item.metadata || {}, digest = item.digest || item.text_sha256 || item.sha256;
        const tr = node('tr'), source = node('td');
        source.append(safeLink(item.source || meta.source || '官方原件',item.source_url || item.url || meta.source_url || meta.url));
        tr.append(node('td',item.title || meta.title || item.source_id || digest || '未命名原件'),source,node('td',item.published_at || item.publish_date || meta.publish_date || '未记录'),node('td',label(item.status || item.processing_status || 'collected')));
        const action = node('td'); action.append(button('查看原文',()=>showText(digest,0),textBusy || !digest)); tr.append(action); el('documents').append(tr);
      }
    }
    async function loadDocuments(batch,offset) {
      if (documentBusy) return;
      documentBusy = true; render(controller.state);
      try { await controller.loadDocuments(batch,offset); }
      finally { documentBusy = false; render(controller.state); }
    }
    async function showText(digest,cursor) {
      if (textBusy) return;
      textBusy = true; el('moreText').disabled = true; renderDocuments(controller.state.documents);
      try {
        const value = await controller.loadText(digest,cursor);
        if (cursor === 0 || textDigest !== digest) { textDigest = digest; el('sourceText').textContent = value.text || ''; }
        else el('sourceText').textContent += value.text || '';
        textCursor = value.has_more && value.next_cursor != null ? value.next_cursor : null;
        el('sourceMeta').textContent = `${value.title || '库存原件'} · 正文标识 ${digest} · ${textCursor !== null ? '原文尚有后续' : '已显示可读取全文'}`;
        el('original').hidden = false; el('moreText').hidden = textCursor === null;
      } finally { textBusy = false; el('moreText').disabled = false; renderDocuments(controller.state.documents); }
    }
    el('refresh').onclick = () => perform(()=>controller.refresh());
    el('crawlEnabled').onchange = event => { const enabled = event.target.checked; perform(()=>controller.setCrawlEnabled(enabled)); };
    el('usageEnabled').onchange = event => { const enabled = event.target.checked; perform(()=>controller.setUsageEnabled(enabled)); };
    el('importHistory').onclick = () => perform(()=>controller.importHistory());
    el('crawlForm').onsubmit = event => { event.preventDefault(); perform(()=>controller.crawl({since:el('since').value,until:el('until').value,max_pages:el('maxPages').value})); };
    el('processForm').onsubmit = event => { event.preventDefault(); perform(()=>controller.processBatch({batch_id:el('processBatch').value,cap_cny:el('capCny').value,max_documents:el('maxDocuments').value,max_requests:el('maxRequests').value})); };
    el('activateForm').onsubmit = event => { event.preventDefault(); perform(()=>controller.activateVersion(el('versionSelect').value)); };
    el('processBatch').onchange = () => render(controller.state); el('versionSelect').onchange = () => render(controller.state);
    el('maintenanceBatch').onchange = () => render(controller.state);
    el('maintenanceEvent').onchange = () => render(controller.state);
    el('maintenanceImport').onclick = () => perform(()=>controller.importMaintenanceBatch(el('maintenanceBatch').value));
    el('maintenanceRun').onclick = () => perform(()=>controller.runMaintenanceEvents([el('maintenanceEvent').value]));
    el('previousPage').onclick = () => perform(()=>loadDocuments(controller.state.documents.batch_id,controller.state.documents.offset - 20));
    el('nextPage').onclick = () => perform(()=>loadDocuments(controller.state.documents.batch_id,controller.state.documents.offset + 20));
    el('moreText').onclick = () => { if (textCursor !== null) perform(()=>showText(textDigest,textCursor)); };
    doc.addEventListener('visibilitychange',()=>controller.resume().catch(()=>{}));
    globalThis.addEventListener('pagehide',()=>controller.stop(),{once:true});
    render(controller.state); controller.start().catch(()=>{});
    return controller;
  }
  return {createRequest,createController,validateCrawl,validateProcess,resumeCrawlInput,schedulerText,selectionText,jobText,maintenanceText,safeDisplay,mount};
});
