# 保险相关性统一分流：修改说明

日期：2026-10-24。规则版本：`insurance-scope/1`。开发来源提交：`0d7ca21`。

## 修改目的

此前部分采集和维护入口使用全文是否出现“保险”作为筛选条件，容易漏掉“财险、人寿”等简称，也可能把监管机关或法律名称误当作业务证据。另有银行主体过滤，不能区分银行代理保险与纯信贷业务。

本次改为共享的、可回查原文的范围分流。规则负责提出审核方向，最终审核仍由既有维护流程完成。

## 新旧行为

| 情形 | 本次处理 |
|---|---|
| 财险、人寿等明确机构主体 | 保险机构候选审核 |
| 个人明确在保险机构任职 | 保险任职候选审核，记录主体和原文关系证据 |
| 银行违规销售保险产品 | 保险业务候选审核，不因银行主体直接过滤 |
| 银行贷款管理等明确非保险事实 | 非保险建议，保留待核，不直接批准排除 |
| 只有监管机关、法律书名提到保险 | 不据此判定保险业务相关 |
| 只是保险公司客户、第三方公司被提及 | 不据此认定处罚对象是保险机构人员 |
| 主体关系不足、多主体归属不明 | 保留范围待核 |
| 保险机构换证、许可公告 | 核对文书类型，不直接当处罚案生成 |

这是可解释规则，并非机构官方名录或完整语义分类器。复杂关系、机构别名和无结构文本仍可能需要复核。

## 公开仓库包含的修改

| 文件 | 作用 |
|---|---|
| `src/offline/insurance_relevance.py` | 新增共享评估函数，按主体、任职、业务事实返回分类、路由、证据和未知条件 |
| `src/agent_rag/collection_store.py` | 读取库存时计算当前范围建议，保留旧关键词字段及数据库结构 |
| `src/agent_rag/maintenance_service.py` | 不再静默丢弃无“保险”字样的原件，登记待审事件并记录范围理由 |
| `src/agent_rag/collection_service.py` | 兼容处理路径保留无关键词材料，既有重复文书控制保留 |
| `src/agent_rag/knowledge_updates.py` | 取消全文关键词及银行硬过滤；范围待核写入候选处置记录，不为这些材料自动发起生成 |
| `src/offline/penalty_crawler/filter.py` | 旧采集过滤器共用分流结果，保留待核原件 |
| `src/web/static/knowledge-updates.js` | 页面展示范围建议和“尚未审核”，旧记录缺分类时提示重新分流 |
| `scripts/audit_insurance_scope.py` | 只读校验原件哈希、证据坐标，生成离线JSON/HTML对照 |
| `tests/offline/test_insurance_scope.py`、`tests/ui/knowledge-updates.test.cjs` | 简称、银行业务、法律污染、客户身份、客户经理、字段边界、许可及入口回归 |

主项目另外接入了普通案例入库和旧版Web更新命令：非保险规则建议由自动rejected改为pending，旧命令持久保存scope_review_queue。该普通应用层及其依赖不属于本公开最小交付包，因此未为这次发布引入整个应用栈；公开包内的采集、维护、增量候选入口已接入同一规则。

## 接口与兼容

新增`scope_assessment`返回：

- `category`：insurance_entity / insurance_business / non_insurance_candidate / uncertain。
- `relevance`：兼容insurance / non_insurance / uncertain。
- `route`：candidate_review或scope_review。
- `evidence`：原文字段、引用、start/end半开区间、判断理由。
- `unknowns`和`rule_version`：待核事项与使用的规则版本。
- `review_status=pending`、`publication_allowed=false`：范围规则不授予批准。

旧`contains_insurance`继续表示字面出现，不再决定是否可进入维护审核。无需迁移库存数据库；读取旧原件时使用当前规则重新计算。既有已登记事件仍遵循幂等约束，旧事件不会自动重写。

旧`include_bank_insurance`和`include_all`参数仍可传入，但不再用于恢复银行或无关键词硬排除。范围待核不等于永远忽略，后续通过维护事件复核；本次不自动启动模型审核或版本构建。

## 冻结856份库存离线结果

| 项目 | 数量 |
|---|---:|
| 原件及哈希校验 | 856 |
| 旧全文关键词允许进入 | 405 |
| 恢复可审阅，不再因缺关键词跳过 | 451 |
| 保险机构/明确任职线索 | 343 |
| 保险业务线索 | 37 |
| 非保险建议 | 228 |
| 范围不确定 | 248 |
| 候选审核路由 | 379 |
| 范围或文书类型复核路由 | 477 |

此前预检发现的12份简称潜在漏筛材料均恢复保险关联线索，其中1份为换证公告，仍需文书类型复核。380份保险线索因此对应379份候选审核。

**这些数量不是准确率，也不是正式入库数量。** 本轮模型/API请求为0，未改变正式V2知识库、原件及历史实验。公开[统计摘要](experiments/insurance-scope-20261024/summary.json)保留冻结清单哈希；856份原件和逐件审查信息未随本次代码更新上传，完整库存回放须自备同一冻结材料。原V2 Release保持不变。

## 验证与复现

开发工作区已完成207项Python回归、27项页面测试。公开发布副本独立验证为111项Python测试、27项页面测试通过。首次Python运行有1项Windows临时文件读取异常；未改代码或断言，换独立目录完整复跑通过，记录保留，不宣称已修复该偶发问题。详细结果见[验证记录](experiments/insurance-scope-20261024/verification.json)，两套环境的测试数量分别报告，不将主项目测试冒充公开包测试。

公开包安装`requirements-test.txt`并完成Node构建后，可运行：

```powershell
& '.\.agent-venv\Scripts\python.exe' -m pytest tests/offline/test_insurance_scope.py tests/agent_rag/test_collection_store.py tests/agent_rag/test_collection_service.py tests/agent_rag/test_collection_api.py tests/agent_rag/test_collection_selection.py tests/agent_rag/test_maintenance_service.py tests/agent_rag/test_knowledge_updates.py tests/agent_rag/test_knowledge_maintenance_merge.py -q
node --test tests/ui/knowledge-updates.test.cjs
```

只读库存对照脚本需要`--inventory`冻结JSONL清单、`--blob-root`原件目录和新的`--output`目录。清单要求blob_path、digest、raw_sha256、template_fields（party与finding）、contains_insurance_keyword、scope_triage等既有预检字段。blob_path须指向blob-root内的原件；换机需重新定位原件路径，原始清单和哈希应另行保留。缺少原件或哈希不符会报错，已有输出目录不覆盖。

## 更新后操作

拉取代码后，正常重启正在运行的服务以加载新规则。新分类不代表已审核、已构建、已切换版本或已启用扩展。先抽查范围分流，再按既有Proposer–Reviewer审核、构建和启用流程操作。
