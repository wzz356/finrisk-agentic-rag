# 数据契约

跨模块数据模型只定义在 `src/domain`。JSON 字段使用英文 snake_case，日期使用 ISO 8601，
分数范围为 0–1，未知值使用 `null`。业务数据不得保存开发者本机绝对路径。

## 1. 文件提取

`FileExtractionResult` 是案例和 Query 的共享输入，关键字段：

| 字段 | 含义 |
| --- | --- |
| `document_id` | 由文件内容确定的稳定 ID |
| `file_purpose` | `case` 或 `query` |
| `source_filename / source_uri` | 原始文件名和逻辑来源 |
| `checksum_sha256 / file_size_bytes` | 文件完整性信息 |
| `full_text` | 按阅读顺序合并的全文 |
| `pages / tables` | 页、块、词和表格 |
| `extraction_method` | pdf_text、rapidocr、hybrid、plain_text、docx |
| `quality_score / warnings` | 质量与异常说明 |

`TextBlock` 和 `WordContent` 保存页内/全文字符偏移与归一化 polygon。原生 PDF 文本优先，
低质量或无文本层页面才使用 RapidOCR。原始文件不能被提取过程覆盖。

## 2. 案例知识库物理表

主库 schema 沿用队员构建包，避免重建 15,620 个主体数据。
原始库、`case_sources/official_crawl`、`case_sources/web_ingestion` 分开存储，
由同一个 `CaseRepository` 联合读取；两个新增分区独立保存三份 JSONL、证据 SQLite 与 manifest。

### `cases.jsonl`

一行一个案例。运行时重点字段：

| 字段 | 含义 |
| --- | --- |
| `case_id` | 主键；新增为 `U` + 文件哈希前缀，多主体增加独立后缀 |
| `file_id / source_sha256` | 来源文件身份 |
| `case_origin` | 构建批次来源、`official_crawl` 或 `web_ingestion` |
| `source_group` | 对外统一为 `original`、`official_crawl`、`web_ingestion` |
| `case_title / party_name / penalty_doc_no` | 案例身份字段 |
| `violation_behavior / penalty_content` | 违法事实和处罚内容 |
| `regulator / penalty_date` | 处罚机关和日期 |
| `risk_tags / risk_domains` | 案例标签和风险归口 |
| `risk_tag_basis` | 标签的原文证据 |
| `retrieval_status` | `active` 或 `inactive` |
| `evidence_status` | `verified` 或 `limited` |
| `source_file / source_url` | 可追溯来源 |
| `publish_date / collected_at / official_doc_id` | 官网发布日期、采集时间和文书标识 |
| `quality_flags / review_status` | 自动化质量与复核状态 |
| `data_version` | 数据版本 |

历史值 `official_gold` 是旧流程内部质量层，不代表官方权威。Web 新增记录若使用规则降级，
必须标记 `manual_review_required / limited / rule_fallback`。

### `case_chunks.jsonl`

| 字段 | 含义 |
| --- | --- |
| `chunk_id / case_id / file_id` | 切片主键和外键 |
| `chunk_type` | fact、evidence、penalty、legal_basis、party |
| `chunk_text` | 可展示或检索的文本 |
| `supports_risk_tags` | 本切片支持的案例标签 |
| `is_verbatim / source_span` | 是否逐字原文及位置 |
| `retrieval_enabled / retrieval_weight` | 是否入索引及权重 |

当前 BM25 只索引启用案例的 fact/evidence 切片。

### `case_expansions.jsonl`

扩展是检索召回辅助，不是案例事实证据。`case_links` 关联一个或多个案例，
`retrieval_enabled` 控制是否入索引。生成扩展不能在 UI 中伪装成原文。

### 增量一致性

新案例必须同时产生一个 cases 记录、至少一个 chunk、零到多个 expansion。仓库在同一锁内
在目标分区追加三表并刷新 manifest；失败按原字节长度回滚。提交标记与写前日志支持进程中断恢复。
三种来源使用同一 schema、构建器和检索接口，新增不改写原始分区。原始案例启停使用独立状态覆盖文件。

## 3. 案例 Web/API 模型

- `CaseSummary`：列表和检索共用摘要；
- `CaseDetail`：摘要 + 来源、相关方、法律依据、质量标记和证据切片；
- `CaseListPage`：`items / total / page / page_size / total_pages`；
- `CaseStatistics`：总数、原始数、官网采集数、网页录入数、启用数、标签数和数据版本；
- `CaseIngestionResult`：新案例详情、提取方式、是否使用 LLM、警告；
- `MatchedCase`：案例 ID、违法事实、标签、0–1 展示分、命中文本和命中单元。

`is_user_managed` 表示官网采集或网页录入的新案例可在页面启停；`source_group` 表示存储来源。
`GET /api/collection/status` 返回每日增量任务的日期边界、下次时间、待办、限额和索引状态。

## 4. Query 契约

`QueryProcessingResult` 2.0：

- 文件身份和全文；
- `classification`：六类场景、置信度、方法、证据、版本；
- `sentences[]`：Risk 唯一四字段输入；
- `source_mappings[]`：与句子一一对应的原文偏移和页面区域；
- `extraction`：媒体类型、SHA-256、页数、OCR页数、方法、质量和块/字词统计；
- `metadata`：渠道和提取质量等附加信息。

`RiskQuerySentence` 严格只有：

```json
{
  "query_id": "query_...",
  "sentence_id": "query_..._s0001",
  "scene": "ST1_poster_sms_social",
  "text": "待判断原句"
}
```

Risk 不修改这四个字段，使用 `sentence_id` 回写结果。

## 5. 风险输出

`QueryAnalysisResult` 4.0 继承 Query 2.0 并增加：

### `risk_summary`

| 字段 | 含义 |
| --- | --- |
| `overall_score / overall_level` | 最高句为主、多风险句有限增益后的全文风险 |
| `overall_level_name / action` | 中文显示和建议动作 |
| `risk_labels / risk_findings` | 阈值以上标签及其最高分、句数和句子ID |
| `top_matched_cases` | 跨句去重后的Top标签对齐案例 |
| `risk_display_threshold / retrieval_threshold` | 风险展示与案例检索门槛 |
| `sentence_count / risky_sentence_count` | 总句数和分数不低于 0.20 的句数 |
| `llm_used / llm_model` | 本次是否真正使用 LLM |
| `llm_reviewed_sentence_count / local_direct_sentence_count` | 复核与本地直出数量 |
| `warnings` | 降级或供应商异常 |

### `risk_assessments[]`

每项与一个 `sentence_id` 对应，包含 `text / risk_score / risk_level / risk_labels /
reasoning / method / local_score / local_confidence / llm_reviewed / retrieval_performed /
evidence / matched_cases`。`MatchedCase` 同时给出 BM25、向量、RRF 与最终重排分。

## 6. LLM 协议

所有调用方只依赖 `LLMClient.complete_json(JsonChatRequest) -> JsonChatResponse`。
客户端负责鉴权、超时、HTTP 状态、JSON 解码和供应商响应校验；业务模块负责各自的输出
Pydantic schema。API Key 不属于任何业务模型，也不能写入日志或返回值。

## 7. 版本与变更

- 案例物理表沿用`case_kb_v1`契约，当前数据版本`finrisk-v2.1-local-20261018`；
- Query：2.0；
- Query + Risk 聚合输出：4.0；
- 风险阈值和词库变更必须新增回归测试；
- 删除/重命名字段前确认现有 JSONL 是否需要迁移，不静默重建主库。
