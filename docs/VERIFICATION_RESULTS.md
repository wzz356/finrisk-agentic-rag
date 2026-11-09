# 2026-11-09 整合验证记录

环境：Windows、Python 3.12.14；Node.js 24.15.0。两个应用分别使用自己的依赖版本，业务源码按导入快照保留。

## 已完成的验证

| 范围 | 实际结果 |
|---|---|
| 根目录新增入口 | 5项通过，覆盖目录定位、端口、退出码、缺环境与关闭继承的live开关。 |
| Agent TypeScript | 通过原tsconfig构建。 |
| Agent离线端到端 | 3个合成任务完成，进度及原文校验通过，live请求被拒绝，模型API调用0。 |
| 工作台Pi协议及Web原文定位 | 37项Node测试通过。 |
| 原始模块与历史 | 两个导入目录的Git内容树与来源提交一致；两个来源提交均为整合分支祖先。 |
| 文档审查 | 本地链接核对；工作台安装器改变当前目录的问题已用Push/Pop-Location修正并复核。 |

## 公开回归的实际限制

### Agent Python

首轮全量：1290通过、13失败、1跳过。首轮部分失败受构建尚未完成和Windows过长临时目录影响。

构建完成后改用短临时目录复测首轮失败项，其中12项经复测或单项前序复核通过；仍有1项范围分流测试失败：

`tests/agent_rag/test_collection_runtime.py::test_bank_insurance_is_opt_in_while_noninsurance_is_still_skipped`

相同用例在未修改的原Agent仓库快照中同样失败，当前实现返回待处理材料，而测试期望空列表。当前保留原业务实现与测试，差异作为后续修复项记录。

### Agent Node

原`npm test`完成TypeScript构建后，部分测试未能自然退出，本轮已终止该测试树。使用30秒单测试超时的受限诊断仍出现快路径断言失败及超时，随后终止，不能报告全量通过。

原快照使用同一构建产物和依赖单项复核：

`agent_harness/tests/agent_loop_fast_path.test.ts`中的`fast success uses one fixed query and one determination with old report and final bound sidecars`

同样出现`0 !== 1`断言失败。该项作为原有问题保留，不归因于新增启动入口；其他超时未在本次整合中完成根因修复。

### 工作台 Python

全量：318通过、8失败、1跳过。

其中7项涉及未随源码提供的知识资产，缺少旧版本`cases.jsonl`或标签字典；另1项发布快照新鲜度用例返回`stale`，在未修改的原工作台副本中也可复现。本次没有下载完整知识库、原始PDF及向量重建全套运行环境。

## 结论范围

已验证新增入口和可公开体验的离线演示，保留业务源文件及历史。完整回归尚未全部通过，真实模型质量与全库索引也未在本次重新评测。历史质量指标按[原始实验口径](EVALUATION.md)引用。
