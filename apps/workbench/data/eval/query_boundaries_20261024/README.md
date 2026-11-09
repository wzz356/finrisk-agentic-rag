# Query 边界训练集（2026-10-24）

本集由 Codex 逐条编写并复核，用于现有字符 n-gram 候选模型训练、提示示例和工程回归。没有调用外部模型标注，没有微调 DeepSeek 权重，也没有使用原 200 条测试答案编写样本。标注不是人工专家金标。

训练集有 24 个场景组、72 条；验证集有另 8 个场景组、24 条。每组各有一条明确风险、一条合规或否定示例、一条缺必要事实的待核样本，按整组划分。训练覆盖现有 27 个标签；验证侧重否决方案与独立事实、跨产品归因、银行与保险业务区分、引用与实际采纳、问答关联、客户与财务数据区分，以及长文中的独立事实。

`risk_labels` 只记录当前输入已支持的标签。待核条件写入 `missing_facts`，候选标签不当作已成立风险；一个机制待核不会清除另一个明确成立的机制。每条保留文本 SHA-256、Unicode 码点区间和原文证据。三条验证长文超过 1,600 字符，用于检查前文合规提示能否覆盖后文独立事实。

原开发集 400 条与本训练集 72 条共同训练本地候选模型；新增 24 条验证和原 200 条测试均不进入训练。原开发集前 100 条属于训练内回归，不能作为独立泛化结果。新增验证集可用于选择候选版本，原 200 条只在版本冻结后做一次对照验收。该 200 条此前已经用于项目评估，因此是重复使用的工程回归集，也不能声称为全新独立测试。

运行方式：

```powershell
.venv/Scripts/python.exe scripts/build_query_boundary_set.py
.venv/Scripts/python.exe scripts/audit_query_boundary_set.py
.venv/Scripts/python.exe scripts/train_local_calibration.py --boundary-training data/eval/query_boundaries_20261024/train.jsonl
.venv/Scripts/python.exe scripts/evaluate_query_development.py --run candidate_development --dev-limit 100
```

`manifest.json` 固定样本数量和内容哈希。修改样本需建立新版本，不能根据测试结果改写参考答案。审计脚本检查分组、状态、标签、字面证据、内容哈希和与旧数据的文本重叠；相似度只是字符三元组 Jaccard 检查，不能证明语义完全独立。
