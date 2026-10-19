# 知识维护零收费工程回放

> 公开包说明：本文涉及的真实V2规模和既有库存属于研发环境。公开仓库只包含代码及合成夹具，默认未绑定真实V2；使用者需提供自己的兼容资产。真实维护适配器与完整生产构建器仍未配置。

回放用完全合成的材料，执行Python MaintenanceService → JSONL子进程桥 → Node/Pi角色循环 → 主机工具 → 账本结算 → 候选签据。预定的approve/request_changes/needs_evidence用于验证工程协议，不代表真实模型效果、人工金标或业务准确率。来源副本的历史验收报告、预算账本、签名密钥和轨迹不随本轮代码迁入。

## 入口与夹具

- `scripts/evaluate_knowledge_maintenance.py`：回放与磁盘证据汇总。
- `tests/fixtures/knowledge_maintenance/events.json`：20条合成事件及3条smoke事件ID。
- `tests/agent_rag/test_maintenance_replay.py`：检查实际服务输出、原文读取、独立搜索、候选修订、签据、SQLite账本和生产拒绝。

所有事件标记`example_only=true`，来源使用`fixture:maintenance:*`。法规项使用合成条例，`legal_status=unknown`。抽取纠错固定原始字节，同源更正改变字节；权限项验证公开工具拒绝其他scope的私有材料，且公开service不接纳私有证据。

提案提供operations、claims、source_coverage，审核提供decision、claim_checks、omission_check_completed、issues。字段有精确pointer主张，引用逐字核验；基线、绑定摘要、运行身份和签名由主机生成。Reviewer通过`inspect_changeset`读取主机实际候选；完整阅读不能由搜索预览代替。故意丢失否定词的候选必须经过一次修复，最多修复一轮。

## 可重复执行

在当前主仓库根执行，依赖须已安装。先构建Node维护CLI；输出目录必须为空，每次使用新目录。以下只运行fixture，不读取真实模型密钥，不接受live参数。

```powershell
npm --prefix 'agent_harness' run build
$env:PYTHONUTF8 = '1'
& '.agent-venv/Scripts/python.exe' 'scripts/evaluate_knowledge_maintenance.py' --suite smoke --output 'runtime/maintenance-smoke-new'
& '.agent-venv/Scripts/python.exe' 'scripts/evaluate_knowledge_maintenance.py' --suite full --output 'runtime/maintenance-full-new'
& '.agent-venv/Scripts/python.exe' -m pytest 'tests/agent_rag/test_maintenance_replay.py' -q
```

缺Node或构建产物时返回`MAINTENANCE_RUNNER_BUILD_REQUIRED`，不会生成替代trace。策略限制、工具、修复轮数和账本沿用维护服务；实际基线commit由service读取，不伪造夹具常量。

输出的`summary.json`列出事件终态、requests、physical_provider_requests、settled_micro、remaining_requests、trace_count、expectations_met及实际证据路径；`ledger-requests.json`列出逐请求结算。签名本地密钥只留在独立回放目录，不能提交。fixture模型轮次是脚本协议请求，不能冒称真实provider调用或token用量。

## 判定与边界

须核验所有脚本预期满足、provider请求与收费为零、全部fixture签据被生产`require_approved`以`FIXTURE_FORBIDDEN`拒绝，以及canonical记录前后相同。停用只出现在候选中，回放不执行知识commit或发布。测试与实际报告数字以本轮日志为准，本文不把来源副本旧结果作为当前整合证据。

回放覆盖跨进程工具协议、完整阅读、独立查询轨迹、字段主张/引用一致性、有限修复、持久化和资源结算。真实双模型语义审核、人工金标准确率、实际法规更正/撤回、收费模型预算、生产构建/发布和固定检索集质量不在该回放结论中；调用查询工具不代表完成检索效果验收。
