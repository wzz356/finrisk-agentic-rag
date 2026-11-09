# FinRisk 保险营销合规工作台

当前版本使用 Pi Agent、DeepSeek 风险复核及 Qwen 全量文档/query 向量。
2026-10-24 知识库包含 **10,000 份赛事文书、866 份官网文书、18,092 个主体案例、27 类标签**。
共 10,866 份文书；列表按文书计数，主体案例单独统计。官网分区包含仅归档和范围待核材料。
原始 10,000 份 PDF 位于 `raw_pdfs/`，均完成全文与指纹绑定，原件不改写。
本次发布的数据、查重、范围分流及检索口径见[知识库更新记录](docs/knowledge_update_20261024.md)。
GitHub分发源码、新评测集和冻结结果；密钥、原始PDF和全量索引保存在本地。
已部署到 yuyunCloud：[打开工作台](http://156.239.2.239/)。保留 80 端口转发至本机 8000，
部署、验证及回滚方式见[服务器部署记录](docs/deployment_yuyuncloud_20261022.md)。

## 启动与检查

Windows 首次安装运行 `install.bat`，启动运行 `start.bat`，完整检查运行 `verify.bat`。
Linux 对应 `install.sh`、`start.sh`、`verify.sh`。运行需要 Python 3.11+、Node.js 22.19+。
`.env` 存放本地 API 密钥，配置项见 `.env.example`；不要覆盖已有 `.env`。

```powershell
# 默认端口 8000；本次本地验证使用 18088。
.venv\Scripts\python.exe scripts/run_web.py --port 18088
# 数据和入口完整性检查，不调用外部模型。
.venv\Scripts\python.exe scripts/verify_delivery.py
```

本地地址：<http://127.0.0.1:18088>，健康检查：<http://127.0.0.1:18088/api/health>。
服务加载全量索引约需 40–45 秒；query 不会触发全库文档编码。
缺少同指纹的预构建 Qwen 索引时会显示降级，应先离线构建：

```powershell
.venv\Scripts\python.exe scripts/build_competition_index.py --qwen
```

首次构建使用真实 Qwen API；同指纹缓存和向量检查点会复用。全量向量、原件、知识库及
`.env` 都保留在本地，不是源码的一部分。已有运行数据可直接恢复；从源码重建时按顺序运行：

```powershell
.venv\Scripts\python.exe scripts/fetch_competition_assets.py
.venv\Scripts\python.exe scripts/import_competition_kb.py
.venv\Scripts\python.exe scripts/fetch_competition_assets.py --p0
.venv\Scripts\python.exe scripts/merge_kb_p0.py
.venv\Scripts\python.exe scripts/refresh_case_metadata.py
.venv\Scripts\python.exe scripts/build_competition_index.py --qwen
.venv\Scripts\python.exe scripts/verify_delivery.py
```

以上是旧版资产的恢复链路。本版使用上游 `knowledge-full-20261024` Release，按
`configs/knowledge_full_20261024.json` 校验六个附件，再运行 `scripts/sync_full_knowledge.py`
生成新版目录，具体命令见更新记录。原始 PDF 另放入 `raw_pdfs/`。

## 检索配置

| 配置项 | 可选值 | 用途 |
|---|---|---|
| `FINRISK_RERANK_PROVIDER` | `none` / `cloud` / `deepseek` | 保留融合排序 / Qwen 重排 / Pi-DeepSeek 列表重排 |
| `FINRISK_RETRIEVAL_TIMING` | `post` / `pre` / `adaptive` | 判定后检索 / 判定前注入 / 自适应前后融合 |
| `FINRISK_EMBEDDING_PROVIDER` | `cloud` / `bge` / `local` | 同模型 Qwen 向量 / 可选本地 BGE / 本地 TF-IDF |
| `FINRISK_CASE_KB_ROOT` | 本地目录 | 当前 `data/processed/knowledge/finrisk_full_20261024` |

当前默认 `deepseek` 重排、`adaptive` 检索时机，由新开发集综合评分冻结。
三种重排和Pre、Post、自适应全部可在Web按次切换；本地候选模型及示例使用400条开发Query。
长文本默认启用全文语义初筛与详细核查协同，`FINRISK_LONGTEXT_SCREENING=false`可停用。
可选BGE后端保留，同一次检索的文档和Query必须使用同模型空间。

## 处理流程

文件/文本解析并保留原文位置 → ST1–ST6 场景和作用域 → 本地规则与机制候选 → 必要时 pre
检索 → Pi/DeepSeek 分批复核 → 引证与覆盖检查及有预算的定向补查 → 复用 post 案例池 →
风险、信息不足、无明确风险三态及证据。BM25、Qwen 与中性行为查询先融合，再统一重排。
相似案例不能证明当前业务事实，也不独立触发风险结论。
结构化提交按目标句扩大输出预算，证据编号由协议约束；失败补查只重送未完成目标。
27类标签分别核对成立条件，相关标签作为候选，强规则同时命中多个标签仍须语义核准。

## 目录和维护入口

| 路径 | 内容 |
|---|---|
| `src/`、`agent_runtime/` | 应用、WebUI、Pi worker 与原生工具提交 |
| `tests/` | 后端、Pi 协议、前端原文定位回归测试 |
| `configs/` | 配置说明、重建知识库所需的 27 类风险域映射 |
| `data/processed/knowledge/finrisk_full_20261024/` | 当前原始库、静态官网分区、证据注册库、版本锁及原件绑定 |
| `data/cache/dense/` | 活跃全量 Qwen 索引与断点重建检查点 |
| `data/eval/competition500_v1/` | 500案例、400开发Query、100开发核对子集、200源文书隔离测试Query |
| `data/runtime_knowledge/` | 随源码分发的P0法规、机制及原文快照，不含旧代码和旧评测 |
| `data/risk/calibration/` | 400开发Query训练的三态及27标签轻量模型 |
| `raw_pdfs/` | 原始 PDF，清理和查询均不改写原件 |
| `scripts/run_web.py`、`scripts/verify_delivery.py` | 正式启动和本地完整性检查 |
| `scripts/build_competition_index.py`、`scripts/prepare_bge.py` | 离线向量构建及可选 BGE 准备 |
| `scripts/fetch_competition_assets.py`、`scripts/import_competition_kb.py`、`scripts/merge_kb_p0.py` | 固定版本下载、导入、P0 资产与全量投影构建 |
| `scripts/benchmark_competition500.py` | 新500候选池风险、检索及三种重排对照 |
| `scripts/update_cases_from_web.py`、`src/offline/` | 监管处罚增量采集及入库 |
| `reports/competition500/` | 当前复核参考下的真实API指标、逐条预测、逐标签FP/FN及数据指纹 |
| `outputs/` | 本地API过程文件、构建收据和运行日志，不进入Git |
| `docs/` | 当前架构、契约、数据口径和效果报告 |
| `deploy/`、`scripts/package_submission.ps1` | 部署配置和本地交付打包；雨云使用 `finrisk-yuyuncloud.service` 和 `nginx-finrisk-yuyuncloud.conf` |

当前评测入口（调用真实API，支持断点续跑）：

```powershell
$env:FINRISK_BENCH_RUN='reproduction'
.venv\Scripts\python.exe scripts/benchmark_competition500.py --help
.venv\Scripts\python.exe scripts/audit_competition_dataset.py
.venv\Scripts\python.exe scripts/benchmark_competition500.py test --mode deepseek_adaptive
.venv\Scripts\python.exe scripts/benchmark_competition500.py retrieval-test --rerank-providers deepseek
.venv\Scripts\python.exe scripts/publish_self_review_results.py --run reproduction
```

本轮使用500份文书的500案例、400开发Query及200源文书隔离测试Query。Codex自己逐条复核600条Query及500个候选案例，未调用其他模型审核答案。新参考为Codex自复核结果；相关性由逐条核查的行为目录确定性展开，不冒充人类专家金标或30万对独立人工评分。

修改前后、证据和理由随数据集保存。真实API只执行项目推理与效果测试；训练和few-shot仅使用修正后的400条开发Query。参考版本已改变，新指标不与旧银标分数直接比较。最新结果见[自复核与真实API报告](docs/dataset_review_20261019.md)。

[任务映射](docs/competition_delivery.md)、[数据说明](docs/dataset_card.md)、[Query流程](docs/query_pipeline.md)。
Web效果页、下载结果及CSV/JSONL统一读取`reports/competition500/`。

## 每日官网增量

已接入国家金融监督管理总局 4113/4114/4115 三个处罚栏目。Web 服务内置每日调度，
默认北京时间 04:30 执行。首次启动以当天为永久起始日期，随后更新到前一天，
保留重叠日期补查及待办队列；不会先回补全站历史案例。
请求串行并设随机间隔、HTTP 缓存和每轮预算，遇限流或验证页暂停。

原始知识库、官网采集、网页录入分开存放，查询时统一检索；页面可按三种来源筛选。
新增案例继续使用 Pi/DeepSeek、同一 27 类标签及原文证据链，Qwen 复用向量检查点更新索引。
配置、数据位置与运行命令见[采集说明](src/offline/penalty_crawler/README.md)，
本轮验证见[接入与测试记录](docs/official_collection_20261019.md)。
