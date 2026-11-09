# Competition500 自复核数据集

版本 `competition500-codex-reviewed-v2-20261019`。案例来自500份不同公开处罚文书，Query原由DeepSeek改写。本次由Codex自己逐条阅读600条Query与500个案例，对33个摘录不完整条目回查全文，没有调用其他模型复核或改答案。新参考是Codex直接复核结果，不标为人类专家金标。真实API仅运行项目推理和测试。

| 文件 | 数量 | 用途 |
|---|---:|---|
| cases.jsonl | 500 | 固定候选案例池与来源SHA |
| development.jsonl | 400 | 本地候选模型拟合、示例与开发调试 |
| validation.jsonl | 100 | 开发核对输入，是development前100条的显式子集 |
| test.jsonl | 200 | 与开发源文书分离的冻结效果测试输入 |
| reference/labels.jsonl | 600 | 六场景、三态、27标签参考；评分器读取 |
| reference/qrels_development.jsonl | 400×500 | 开发相关性参考与原文证据 |
| reference/qrels_test.jsonl | 200×500 | 测试相关性参考与原文证据 |
| reference/generation_plan.jsonl | 600 | 种子案例及制作审计 |
| reference/query_reviews.jsonl | 600 | 逐条复核结论、机制、证据及理由 |
| reference/case_reviews.jsonl | 500 | 案例事实核查、机制及上下文补证 |
| reference/review_changes.json | — | 修改前后与理由 |
| reference/relevance_protocol.json | — | 相关性协议及机制别名 |
| manifest.json | — | 来源、版本和SHA256 |

100条validation用于开发核对，包含在400条轻量模型拟合数据中；独立效果以200条test为准。开发Query来自300份文书，测试来自其余200份，源文书无交集。model_inputs.jsonl是开发与测试合并的制作记录。

本轮修正357条Query的标签、三态或场景，校正72个案例的标签或事实摘录，输入文本与案例ID保持稳定。修改前后的答案与理由均已留档，未依据系统预测修改答案。旧参考备份和API过程保留本地；本次未上传GitHub或服务器。

相关性使用`direct-source-review-mechanisms-v3`：0无共同动作，1仅宽泛类目或载体，2同一具体动作，3至少两个相同具体动作。完整矩阵从逐条复核的行为目录确定性展开，不宣称30万对都是独立人工逐对评分。强相关阈值为2并提供双方原文与来源SHA；无强相关Query保留并单列，不强行将生成种子作为正例。索引及风险模型不读取Query参考答案、机制目录或相关性等级。

验证数据完整性：`python scripts/audit_competition_dataset.py`。重跑效果：设置FINRISK_BENCH_RUN后运行`python scripts/benchmark_competition500.py test --mode deepseek_adaptive`。

独立检索测试：`python scripts/benchmark_competition500.py retrieval-test --rerank-providers deepseek`。新参考与旧银标指标不可直接比较。
