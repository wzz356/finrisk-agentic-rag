# 官网处罚增量更新

> 公开包说明：本文涉及的真实V2规模和既有库存属于研发环境。公开仓库只包含代码及合成夹具，默认未绑定真实V2；使用者需提供自己的兼容资产。真实维护适配器与完整生产构建器仍未配置。

入口`scripts/update_agent_knowledge.py`，NFRA采集器位于`src/offline/penalty_crawler/`。默认不调用模型、不发布版本。2026-10-19起，正式发布统一受[Proposer–Reviewer维护门禁](KNOWLEDGE_MAINTENANCE.md)控制；旧CLI的`--publish`明确拒绝，`--generate`只生成待审候选。

## 前提

先自备兼容的live V2快照、BGE向量/模型环境，以及原知识库八表规范化脚本、结构化投影脚本和八表合同。本公开包不提供这些私有运行资产。原文与资产需有合法使用权限；安装公共依赖只足够运行离线演示和测试，不代表已配置GPU构建环境。

运行`python scripts/update_agent_knowledge.py init --help`查看绑定参数。init要求显式指定配置、基线快照、normalizer、projector、contract、registry和crawl-dir；配置保存绝对路径及SHA，换机后重新绑定。动态构建脚本会执行，应只绑定你信任的本地资产。

`update --config <配置路径>`只做预检。需要采集时显式增加`--crawl`；兼容候选生成增加`--generate --cap-cny <费用上限> --max-requests <请求上限>`及显式密钥配置。它不会构建正式向量版本或自动成为可选版本。每文书最多两次生成，总预算和未知用量预留持续记账；不可删账本重发未知请求。管理页面的加工入口改为登记维护事件，不调用该收费生成路线。

## 质量与版本

保留完整原文、正文SHA去重、[基于主体与事实的保险范围分流](INSURANCE_SCOPE_20261024.md)；候选检查官方27标签、逐字引文、主体和条件。合计处罚、主体归属不足、格式错误或超长材料进入待审区，不强行补全。自动通过仍标记pending/human_approved=false，机械通过不能替代语义人工审核。

历史增量构建代码保留用于受控资产适配。正式新版本必须来自已批准知识提交、完整资产清单和主机准入凭据；旧状态文件或机械校验不能替代审核。Agent使用`start-agent-rag.ps1 -KnowledgeRegistry <registry路径>`，当前任务仍固定版本；新维护版本的真实构建器尚未全部接入。

`status --config <配置路径>`查看版本与候选；`rollback --config <配置路径> --version <版本>`切回保留版本，不重新调用模型。版本及其原始资产不能删除，否则会破坏旧任务追溯。

## 已知限制

来源副本的历史官网探针曾遇到HTTPS握手失败；本轮整合未访问官网，未完成实时采集成功验收；采集器代码存在不等于任意网络均可访问。历史官网导出的入库/检索/回滚验证不能代替新抓取成功。跨文书同案、官网正文修订和多主体责任仍需审核。本包没有启用自动定时任务。

## 主项目启动选择

先完成库存迁移和可信基线绑定，确认managed注册器的固定基线仍为9,640份来源，采集与扩展使用均关闭。需要让管理页后续的合法选版影响Agent时，显式启动：

```powershell
.\start-agent-rag.ps1 -KnowledgeRegistry 'runtime/knowledge_updates/registry'
```

启动脚本原参数和密钥默认值保留，默认启动会显式设置SNAPSHOT，因此仍按原快照路径运行；不能把代码整合说成已自动切换管理注册器。显式KB_REGISTRY优先，其次显式SNAPSHOT；都没有才回退到已绑定managed注册器。继续显式使用旧`runtime/agent_rag/knowledge-registry`时，管理页会提示`restart_required`，需按上例重启才能跟随新管理选择。没有usage文件的旧注册器保持current指针兼容。
