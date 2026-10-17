# 官网处罚增量更新

入口`scripts/update_agent_knowledge.py`，NFRA采集器位于`src/offline/penalty_crawler/`。默认不调用模型、不发布版本；`--crawl`、`--generate`、`--publish`分别控制采集、收费生成和切换current。

## 前提

先自备兼容的live V2快照、BGE向量/模型环境，以及原知识库八表规范化脚本、结构化投影脚本和八表合同。本公开包不提供这些私有运行资产。原文与资产需有合法使用权限；安装公共依赖只足够运行离线演示和测试，不代表已配置GPU构建环境。

运行`python scripts/update_agent_knowledge.py init --help`查看绑定参数。init要求显式指定配置、基线快照、normalizer、projector、contract、registry和crawl-dir；配置保存绝对路径及SHA，换机后重新绑定。动态构建脚本会执行，应只绑定你信任的本地资产。

`update --config <配置路径>`只做预检。需要采集时显式增加`--crawl`；实际生成增加`--generate --cap-cny <费用上限> --max-requests <请求上限>`及显式密钥配置；需要发布才增加`--publish`。每文书最多两次生成，总预算和未知用量预留持续记账；不可删账本重发未知请求。

## 质量与版本

保留完整原文、正文SHA去重、保险范围粗筛；候选检查官方27标签、逐字引文、主体和条件。合计处罚、主体归属不足、格式错误或超长材料进入待审区，不强行补全。自动通过仍标记pending/human_approved=false，机械通过不能替代语义人工审核。

八表与投影完成后只编码新增BGE片段，旧向量前缀保持，生成不可变版本并原子切换current。Agent使用`start-agent-rag.ps1 -KnowledgeRegistry <registry路径>`，在空闲时换版，当前任务仍使用固定版本；调用live检索或模型须另行显式启用并配置费用控制。

`status --config <配置路径>`查看版本与候选；`rollback --config <配置路径> --version <版本>`切回保留版本，不重新调用模型。版本及其原始资产不能删除，否则会破坏旧任务追溯。

## 已知限制

当前网络的官网HTTPS握手曾失败，未完成实时采集成功验收；采集器代码存在不等于任意网络均可访问。历史官网导出的入库/检索/回滚验证不能代替新抓取成功。跨文书同案、官网正文修订和多主体责任仍需审核。本包没有启用自动定时任务。
