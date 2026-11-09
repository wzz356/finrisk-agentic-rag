# 官网处罚案例每日增量

入口为 `scripts/update_cases_from_web.py`；Web 服务内置相同的每日调度，二者共用进度和文件锁。
当前接入国家金融监督管理总局的总局机关 4113、省监管局 4114、监管分局 4115 三个栏目。
参考上游 `wzz356/finrisk-agentic-rag` 的日期窗口和原件留存设计，按本项目的入库、证据及检索接口实现。

默认每天北京时间 04:30 更新到昨天。`FINRISK_COLLECTION_START_DATE` 为空时，首次启用将当天固定到
`data/raw/cases/official_crawl/job_state.json`。部署一个全新的运行目录时，日期从部署当天开始；
重启保留原日期。后续只访问此日期以后的详情，旧文书只在列表上判断日期，不下载正文。

```powershell
.venv\Scripts\python.exe scripts/update_cases_from_web.py --dry-run
.venv\Scripts\python.exe scripts/update_cases_from_web.py --status
.venv\Scripts\python.exe scripts/update_cases_from_web.py --activate
.venv\Scripts\python.exe scripts/update_cases_from_web.py
```

`--dry-run` 不访问官网或模型。手动检查今天可以加 `--through-today`；同一天重复执行由任务台账跳过。
`--collect-only` 只留存原件和待办，不调用模型。初始日期可在首次激活时用 `--start-date YYYY-MM-DD` 指定；
已有任务状态不被这一参数重置。已移除原来不限页数的全历史抓取入口。

每轮请求串行执行，间隔 3 秒加 0–2 秒随机量；默认每栏目最多 3 页、共 12 份详情、共 32 次官网请求。
robots、列表复核及 CDN 缺失后的动态接口请求均计入预算。检查 robots.txt，复用 ETag/Last-Modified，
403/429、验证码页及带 Retry-After 的 503 立即暂停；不换身份、代理或地址绕过网站限制。
静态 JSON 仅在 404/410 时尝试对应动态 API。

更新时回看上次成功水位附近 3 天，并以官网文书 ID、规范正文 SHA256、案例 ID、文号与当事人去重。
列表首页采集前后校验；日期不明、页数不足或处理失败时不把相应栏目标记为全部完成。
已经发现、尚未下载的详情保存在 `pending_details.json`，下次优先续采；失败的模型处理重用本地原件。

原始知识库保持原目录；新增数据分区存放：

| 来源 | 结构化记录、证据与分区清单 | 原件 |
|---|---|---|
| 原始知识库 | `data/processed/knowledge/finrisk_v2_1_20261018/` | `raw_pdfs/` |
| 官网采集 | `data/processed/case_sources/official_crawl/` | `data/raw/cases/official_crawl/` |
| 网页录入 | `data/processed/case_sources/web_ingestion/` | `data/raw/cases/uploads/` |

两个新增分区各有 `cases.jsonl`、`case_chunks.jsonl`、`case_expansions.jsonl`、`evidence_registry.sqlite3`
和 `manifest.json`。分区入库有跨进程写锁、失败回滚和中断恢复日志，原始库不被追加或改写。
查询统一读取三个分区，证据服务联合读取注册库，网页显示来源计数并可按来源筛选。

汇总表先拆分、筛选保险主体，再交给同一 Pi/DeepSeek 入库链：逐主体提取、27 类标签、逐字事实引文。
发布日期单独保留，不代替未知的处罚决定日期。入库后后台更新 BM25 与 Qwen；Qwen 只请求检查点中
缺少的文本向量，在本地生成合并后的索引。向量失败时保留已入库案例和待刷新状态，下一轮继续。
部署时须保留 `data/raw/cases/official_crawl/`、`data/processed/case_sources/` 及整个 `data/cache/dense/`
（包括向量 SQLite 检查点）。服务运行时执行定时任务；无需额外重复设置系统定时器。

状态接口：`GET /api/collection/status`。返回起始日期、下次时间、栏目水位、待办数量、限额和索引状态。
真实小流量及模型链测试结果见 `reports/collection/latest_smoke.json`，回归测试位于
`tests/offline/test_incremental_collection.py`、`tests/storage/test_case_repository.py`。
