# FinRisk · 保险营销合规审查与知识库维护系统

面向保险营销话术、宣传材料与业务文档的风险初筛项目，结合监管案例知识库、混合检索与 Agent 证据核查，输出可回读的风险依据及待复核条件。

本仓库整合比赛项目的工作台、独立 Agentic RAG 和知识维护研究成果，保留两个来源仓库的完整提交历史与作者记录。

**技术栈：** Python · FastAPI · TypeScript / Node.js · Pi Agent · DeepSeek · BM25 · RRF · Qwen / BGE Embedding

## 项目能力

- **知识工程：** 按来源文书、主体案例、行为与证据组织万级监管材料，保留稳定 ID、文本指纹、来源区间和版本清单。
- **混合检索：** BM25 与向量双路召回，经 RRF 融合与模型重排，为风险核查提供相似案例。
- **Agent 审查：** 围绕风险点规划检索、回读来源并核对条件，约束工具调用、上下文和运行预算，保留信息不足及未完成事项。
- **工作台：** 文件解析、六类场景处理、27 类风险标签、逐句结果、原文定位、案例管理与效果展示。
- **知识维护研究：** 采集隔离、Proposer–Reviewer 提议与复核、人工待审候选及版本发布准入。

## 目录

```text
finrisk/
├── finrisk.py                 # 统一启动入口
├── apps/
│   ├── workbench/             # 应用工作台、解析、审查、Web与评测
│   └── agentic-rag/           # 独立Agent、工具服务、检索与维护工程
├── docs/                      # 统一架构、安装、评测及来源说明
├── tests/                     # 整合入口的行为测试
├── NOTICE.md                  # 项目及第三方许可说明
└── SOURCE_REPOSITORIES.json   # 来源提交与模块映射
```

两个模块共享版本化知识来源，分别维护运行环境和索引：工作台默认使用 Qwen，Agent V2 使用 BGE。统一入口提供模块选择，两个审查运行层仍分别运行。详见[架构说明](docs/ARCHITECTURE.md)。

## 快速体验：离线 Agent 演示

Windows PowerShell，推荐 Python 3.12、Node.js 22.19+。以下命令从仓库根目录执行；首次安装依赖需要网络。

```powershell
python -m venv "apps\agentic-rag\.agent-venv"
& "apps\agentic-rag\.agent-venv\Scripts\python.exe" -m pip install -r "apps\agentic-rag\requirements-agent-rag.txt"
npm --prefix "apps\agentic-rag\agent_harness" ci --ignore-scripts --no-audit --no-fund
npm --prefix "apps\agentic-rag\agent_harness" run build
& "apps\agentic-rag\.agent-venv\Scripts\python.exe" "apps\agentic-rag\scripts\build_demo.py"
python "finrisk.py" demo
```

打开 <http://127.0.0.1:8765/agent-rag>，填入演示材料。默认使用三份合成案例和 mock 模型；根目录 `demo` 入口强制关闭真实模型与真实重排开关，无需 API Key。按 Ctrl+C 停止。

快照已存在时跳过 `build_demo.py`，直接启动即可。自定义端口示例：`python finrisk.py demo --port 18765 --node-port 18766`。

工作台安装、知识资产恢复、真实模型配置及 Linux 离线核验见[安装与运行](docs/SETUP.md)。

## 结果与评测

| 结果 | 指标 | 口径 |
|---|---|---|
| 工作台当前发布验收 | 二分类准确率 **91.50%**，风险召回率 **97.27%**，二分类 F1 **92.64%** | 200 条工程回归，参考经 Codex 复核，非专家金标或官方赛事成绩。 |
| 独立 Agent 初筛实测 | 首轮 **94/100**；失败请求人工重试后 **98/100** | 80 条官方风险话术 + 20 条补充对照，含开发材料，非独立盲测。 |
| 知识维护开发实验 | 20 份材料：14 建议新增 / 5 建议排除 / 1 待核 | 影子候选与人工待审包，未更新正式库；部分排除依据仍待核。 |

完整数据、逐条结果与限制见[评测说明](docs/EVALUATION.md)。各批结果分别统计。

## 验证

根目录入口测试只依赖 Python 标准库：

```powershell
python -m unittest discover -s "tests" -v
```

Agent 的完整公开测试与端到端离线演示：

```powershell
& "apps\agentic-rag\.agent-venv\Scripts\python.exe" -m pip install -r "apps\agentic-rag\requirements-test.txt"
npm --prefix "apps\agentic-rag\agent_harness" test
Push-Location "apps\agentic-rag"
try {
    & ".agent-venv\Scripts\python.exe" -m pytest tests -q
    & ".agent-venv\Scripts\python.exe" "scripts\smoke_demo.py"
} finally { Pop-Location }
```

完整核验入口见[验证说明](docs/VERIFICATION.md)。

本次整合的实际通过项、资产依赖和上游已知失败见[验证记录](docs/VERIFICATION_RESULTS.md)。

## 知识资产

完整知识库、向量、原始 PDF 和本机运行数据不随源码提交。版本资产继续从原[知识库 Release](https://github.com/wzz356/finrisk-agentic-rag/releases/tag/knowledge-full-20261024)恢复；重建时核对版本与哈希，分别生成各模块索引。

## 来源与协作

团队共同研发。整合保留了以下工程来源，具体映射见[来源记录](docs/PROVENANCE.md)：

- [qby1210/finrisk-framework](https://github.com/qby1210/finrisk-framework)：工作台及应用工程。
- [wzz356/finrisk-agentic-rag](https://github.com/wzz356/finrisk-agentic-rag)：Agentic RAG、知识交付与维护研究。

风险意见用于辅助筛查与复核。历史案例不能补齐当前材料中缺失的事实。项目许可及第三方资源见 [NOTICE](NOTICE.md)。
