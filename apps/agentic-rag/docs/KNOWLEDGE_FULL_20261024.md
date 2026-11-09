# 完整知识库交付（2026-10-24）

[下载本版Release](https://github.com/wzz356/finrisk-agentic-rag/releases/tag/knowledge-full-20261024) · [附件清单](../knowledge/release-manifest.json) · [SHA256](../knowledge/SHA256SUMS.txt) · [本轮交付核验](../knowledge/verification.json)

本版交付已构建的知识库数据、检索资产和来源追溯。**不改Agent实现、不替换应用当前版本、不自动开启采集或收费调用。** 大文件放在Release附件，普通clone保持轻量，无需Git LFS。

## 数据范围

| 内容 | 赛事来源分区 | 官网来源分区 |
|---|---:|---:|
| 完整来源记录 | 10,000 | 850 |
| 参与处罚材料检索的来源 | 10,000 | 849 |
| 受罚主体案件记录 | 16,173 | 1,895 |
| 行为／责任记录 | 25,924 | 2,448 |
| 检索单元／向量行数 | 67,594 | 2,889 |

两个分区合计10,850份规范化来源，其中10,849份参与处罚材料检索，共70,483条检索单元。计数反映候选记录与物理覆盖，不能解释为人工核定的违规数量。

- 赛事分区：9,639份冻结底库＋360份原预留文书＋1份从缺失TXT的PDF恢复的文书。
- 官网分区：857份原件全部保留；7份与赛事来源内容重复，仅保留来源别名；其余850份规范化。其中1份行政许可仅归档，849份处罚材料进入索引，含8份拟处罚公告，未改为已处罚。
- 官网850份中413份判为保险相关（含许可）、400份明确无保险关联、37份关联不明。保留原文、范围判断和未知项，纯银行事项不强塞保险标签；接入保险审查前应显式决定检索范围。
- 法规参考包：67条既有法规注册、上下文依赖、来源及范围状态。它是单独的参考资产，不改变正式准入或时效核定状态，也不自动为Agent增加法规检索工具。

旧版Release包含的9,640份来源已逐份按全文SHA核对，全部在新版两分区中保留。旧版单库中已接入的长治官网增量`WEB-226fafe8de3eb6c8deb08b5680d37228`现在按来源放入官网分区；**只加载赛事分区时不包含该条**。7份重复官网来源的规范记录在赛事分区，官网单独检索不会重复返回这些案件。

## 下载哪些附件

| 附件 | 内容 |
|---|---|
| `finrisk-original_dataset-runtime-20261024.zip` | 完整赛事全文、结构化索引、27标签字典、冻结重排规则、BGE文档向量及可迁移配置模板 |
| `finrisk-original_dataset-normalized-20261024.zip` | 已合并的赛事8张核心表、6张追溯扩展表、3份阅读视图及字段合同 |
| `finrisk-official_crawl-runtime-20261024.zip` | 独立官网全文、索引、向量及配置模板 |
| `finrisk-official_crawl-normalized-20261024.zip` | 完整官网规范化表、候选追溯及阅读视图 |
| `finrisk-source-audit-20261024.zip` | 本轮360份赛事TXT、857份官网原件、恢复PDF及其TXT、7份去重依据、处理台账和模型复核记录 |
| `finrisk-regulations-reference-20261024.zip` | 67条法规参考注册、来源快照、依赖和原有准入标记 |

`release-manifest.json`和`SHA256SUMS.txt`固定附件大小和指纹；每个包还有逐文件`bundle-manifest.json`。没有上传模型权重、密钥、个人账本、私人测试答案或API对话日志。旧9,639份完整来源正文已在runtime和normalized中，不另重复上传全部原始PDF。

只运行案例检索可下载所需分区runtime包；只阅读知识卡可下载normalized包。完整交付下载全部六包。法规包保持参考用途，不能代替法规有效性核查。

```powershell
gh release download knowledge-full-20261024 --repo wzz356/finrisk-agentic-rag --dir "downloads-20261024" --pattern "finrisk-*.zip" --pattern "release-manifest.json" --pattern "SHA256SUMS.txt" --pattern "verification.json"
Get-FileHash "downloads-20261024\finrisk-original_dataset-runtime-20261024.zip" -Algorithm SHA256
```

与仓库`knowledge/release-manifest.json`固定指纹比较，不要只依据包内自报指纹判断下载来源。

## 解压、核验与安装

以下在仓库根目录执行。解压目录应为新目录，避免覆盖历史版本。

```powershell
Expand-Archive "downloads-20261024\finrisk-original_dataset-runtime-20261024.zip" "runtime\knowledge-20261024\original-assets"
Expand-Archive "downloads-20261024\finrisk-original_dataset-normalized-20261024.zip" "runtime\knowledge-20261024\original-normalized"
Expand-Archive "downloads-20261024\finrisk-official_crawl-runtime-20261024.zip" "runtime\knowledge-20261024\web-assets"
Expand-Archive "downloads-20261024\finrisk-official_crawl-normalized-20261024.zip" "runtime\knowledge-20261024\web-normalized"

python "scripts\install_knowledge_bundle.py" --bundle "runtime\knowledge-20261024\original-assets" --verify-only
python "scripts\install_knowledge_bundle.py" --bundle "runtime\knowledge-20261024\web-assets" --verify-only
python "scripts\install_knowledge_bundle.py" --bundle "runtime\knowledge-20261024\original-normalized" --verify-only
python "scripts\install_knowledge_bundle.py" --bundle "runtime\knowledge-20261024\web-normalized" --verify-only
```

所有包均可用既有安装器的`--verify-only`核对逐文件SHA。阅读知识卡优先打开`reading/case_reading.jsonl`或`reading/behavior_reading.jsonl`；完整文书在`data/documents.jsonl`。

GPU环境、指定revision模型准备方法沿用[旧版说明的“配置完整混合检索”](FULL_KNOWLEDGE_BASE.md#配置完整混合检索)：`BAAI/bge-small-zh-v1.5`，revision `7999e1d3359715c523056ef9478215996d62a620`，Torch `2.6.0+cu124`、Transformers `4.57.1`、CUDA、float16、512维。模型文件必须匹配包内模型指纹清单，不能任意替换。

准备好模型目录和GPU Python后，分别生成机器本地快照：

```powershell
python "scripts\install_knowledge_bundle.py" --bundle "runtime\knowledge-20261024\original-assets" --snapshot-output "runtime\knowledge-20261024\original-snapshot" --model-path "models\7999e1d3359715c523056ef9478215996d62a620" --gpu-python ".gpu-venv\Scripts\python.exe"

python "scripts\install_knowledge_bundle.py" --bundle "runtime\knowledge-20261024\web-assets" --snapshot-output "runtime\knowledge-20261024\web-snapshot" --model-path "models\7999e1d3359715c523056ef9478215996d62a620" --gpu-python ".gpu-venv\Scripts\python.exe"
```

安装器只校验资产和重定位路径，不重算向量、不联网、不调用LLM、不修改current或usage。安装输出目录拒绝覆盖；安装后保留assets目录，移动目录后应重新安装到新snapshot目录。派生快照ID改变，原始ID在`distribution_origin_snapshot_id`保留，知识内容不变。规范化表中的历史路径仅保留来源沿革，不要求复原原机器目录。

## 接入与检索边界

两个runtime分别兼容仓库现有`V2KnowledgeSnapshot`。BM25使用结构化文本在内存构建；BGE使用包内已生成文档向量，输入在线编码；双路按1:1 RRF融合，后续按应用配置执行DeepSeek重排。**当前应用不会自动同时检索两个分区。** 需要显式选择单个快照；若要两库共同使用，应由应用另行配置路由、范围过滤及来源别名处理。

启动真实Agent和真实重排仍按[运行说明](RUNNING.md)显式配置密钥、许可和费用限额。例如已满足这些条件后，选择赛事快照：

```powershell
.\start-agent-rag-v2.ps1 -SnapshotPath "runtime\knowledge-20261024\original-snapshot" -AllowPaidRetrieval -RerankCapCny 0.5 -EnableLiveAgent
```

该命令会启用真实模型调用并可能产生费用；下载、校验与安装不会。要保持当前应用版本，只完成安装，不执行此启动命令。

## 审核、缺失PDF与效果口径

本轮1,217份材料完成结构化候选与引用校验，另有161份独立模型复核；主体名称切片错误及部分责任归属、标签表述问题已修订，保留版本绑定记录。新增候选仍为`pending/candidate`，**不是金融专家金标或人工全案核定**。原文缺失的日期、文号、个人逐项责任和费用去向不补猜。

恢复的《行政处罚信息公示表_1.pdf》涉及松原两家保险机构及金学思、史锋，形成4个案件、7条行为／责任记录和10条检索单元。使用本地扩展编号`F010000`，没有冒认尚未证实的官方缺号`F003959`；PDF、提取文本、SHA及映射记录随审计包保留。

原始构建已完成全文、规范化关系、证据坐标、索引覆盖、来源隔离及基线向量不变核验。本次发布另验证各附件逐文件SHA/CRC、两个分区搬迁后安装、全部运行检索单元的引用回读、规范化证据坐标、ID和计数，以及实际本地BGE查询。结果见[verification.json](../knowledge/verification.json)。

本轮发布未调用DeepSeek、未重测Recall/MRR/NDCG或风险识别准确率。历史实验保持旧版本与旧口径，不能直接作为本版效果。更大覆盖不等于已经证明识别效果提升；法规67条也不表示完整覆盖27标签、现行法规或全部历史时效。

法规参考包保留`production_active`、`research_reference_allowed`、`admission_status`等原标记及上下文依赖；不得把“有条文”解释为“对当前主体和行为必然适用”。本包没有自动接入法规工具或改变Agent判断流程。

## 历史版本

[2026-10-19 Release](https://github.com/wzz356/finrisk-agentic-rag/releases/tag/knowledge-v2-20261019)及其[历史清单](../knowledge/2026-10-19/release-manifest.json)完整保留。知识库数据版本与Agent工程检查点分别管理，本次没有修改`src/`、前端、运行注册器或模型调用逻辑。
