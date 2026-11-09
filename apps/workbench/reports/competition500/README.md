# Competition500 自复核参考下的测试结果

`evaluation.json`供Web效果页和下载使用，展示本轮200条Query的真实API结果。默认Pi/DeepSeek风险判断、DeepSeek重排和自适应Pre/Post。`selection_development.json`保留开发试验及冻结策略；没有按测试结果调整答案或参数。

`predictions_test200.jsonl`保留逐条风险预测，`retrieval_test200.jsonl`保留独立检索排序。`submission.jsonl`是比赛输出格式；`200Query复核答案与预测.csv`可逐条查看参考与预测，`27标签复核后测试指标.csv`列出支持数、TP、FP、FN和F1。`dataset_hashes.json`、`dataset_audit.json`、`verification.json`绑定输入、参考、来源与验收。

600条Query和500个候选案例由Codex自己逐条复核，未调用其他模型审核答案。新参考为Codex自复核结果，不是人类专家金标；相关性从核对过的机制目录展开完整矩阵。100条validation是开发子集，测试种子文书与开发隔离。真实API只执行项目推理和检索。

本轮风险准确率91.0%，标签Micro-F1为65.0%，Hit@5为92.5%，Recall@10为23.6%。13条无强相关Query单列。旧参考已修正，不能将旧银标分数直接与本轮比较；JSON中的previous_predictions_rescored仅将旧冻结预测用新参考重评分，并非重新调用旧模型。完整说明见docs/dataset_review_20261019.md。发布内容包含代码、复核数据和汇总结果；API密钥、全库运行数据和过程日志保留本地，本轮未部署服务器。
