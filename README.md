# FinRisk Agentic RAG

保险业务材料风险审查的最小工程交付包。输入一句或一段材料，智能体按风险点检索案例、回读证据、核对标签条件，输出疑似风险与待复核事项。

**默认是无费用的离线模拟演示，使用3份完全合成的案例。** 模拟意见由固定脚本提供，用于检查运行流程，不能展示真实模型的识别准确率。本仓库不提供赛事处罚原文、完整知识库、向量索引、模型权重或真实审查轨迹。

## 包含什么

- Pi核心库驱动的TypeScript运行层，负责状态、工具调用、上下文、停止条件和结构化终稿。
- Python FastAPI来源服务，提供冻结快照、检索、原文定位、证据核验、任务和进度接口。
- 三个工具：`search_cases`、`read_source`、`get_risk_definition`。
- 保留既有27类标签及审查条件配置；条件是内部解释草稿，不是新增监管规则。
- V2检索实现：BM25 + BGE双路召回 → RRF融合 → DeepSeek Flash相关性重排 → 文书聚合。需要另行提供兼容的外部资产；默认演示只运行BM25。
- 独立演示页面、合成案例、构建脚本和离线测试。

Python与Node之间使用本机认证HTTP桥接；本版本未实现MCP传输层。

## 快速开始（Windows PowerShell）

要求：Python 3.12、Node.js >=22.19，以及可安装依赖的网络。下面命令在仓库根目录执行。

```powershell
python -m venv ".agent-venv"
& ".\.agent-venv\Scripts\python.exe" -m pip install -r "requirements-agent-rag.txt"
npm --prefix "agent_harness" ci
npm --prefix "agent_harness" run build
& ".\.agent-venv\Scripts\python.exe" "scripts\build_demo.py"
.\start-agent-rag.ps1
```

打开 <http://127.0.0.1:8765/agent-rag>，点击“填入赠礼演示样例”，再开始演示。按Ctrl+C结束本轮服务。已有快照时构建脚本会拒绝覆盖；可通过`--output`指定一个新目录，再向启动脚本传入`-SnapshotPath`。

页面可查看风险意见、输入引文、待核条件、参考来源及进度。默认检索的合成资料在正文、快照和页面中均有标注；协议中的`source_kind=penalty`等字段沿用原结构，不表示这些示例是真实处罚。

## 运行验证

```powershell
npm --prefix "agent_harness" test
& ".\.agent-venv\Scripts\python.exe" -m unittest discover -s "tests" -v
& ".\.agent-venv\Scripts\python.exe" "scripts\smoke_demo.py"
```

最后一条命令会自行启动两个临时本机服务，创建合成快照，检查端到端任务、进度、原文与防误启收费开关，完成后关闭。不会调用模型API，不修改你的默认快照。

## 架构

```text
输入材料与已知背景
        ↓
Python：冻结输入、快照、任务与执行契约
        ↓
Node / Pi：分析风险点 → 规划工具 → 核读证据 → 检查条件 → 终稿
        ↕ 本机认证HTTP
Python工具：search_cases / read_source / get_risk_definition
        ↓
结构化报告 + 可核对引文 + 待复核条件 + 运行进度与资源记录
```

标签条件必须绑定当前输入或用户提供的背景。历史案例不能补齐当前合同、说话者、用途等事实。案例相似分只用于排序，不是违规概率；原文哈希通过只代表文本对应，不代表人工批准或语义正确。

| 目录 | 内容 |
|---|---|
| `agent_harness/` | Pi运行层、模型网关、上下文、预算与测试 |
| `src/agent_rag/` | API、任务、来源、检索、重排和证据核验 |
| `contracts/agent_rag/` | JSON Schema、标签字典、内部条件草稿 |
| `configs/agent_rag/` | mock/live配置与系统提示词 |
| `examples/` | 合成案例与请求示例 |
| `resources/` | 重排提示词、带许可证的token计数资产 |
| `scripts/` | 离线演示和兼容外部资产准备工具 |
| `docs/` | 运行、设计边界与交付验证说明 |

## 真实模型与完整检索

见[运行说明](docs/RUNNING.md)。默认不读取个人密钥文件，也不自动开启真实模型或真实重排。真实运行会将输入及召回片段发送给模型服务并产生费用，需要自行设置密钥、显式开启并配置账本。

V2适配器保留了原实验的资产、版本和哈希约束，并不是适用于任意知识库的一键导入器。外部资产缺失时会拒绝启动V2，不会静默替换成其他算法。

## 当前边界

- 当前是Baseline工程，不是完整的法规审查系统，也不构成法律认定。
- 默认mock验证工程协议；识别效果需要独立标注集与真实模型评测。
- V2的BGE在线单条编码与原批次编码仍有细微数值和排序差异，严格等价尚未通过。
- 网络失败可能造成部分重排缺失；未知usage保留费用预留，不会伪装成零成本。
- 单实例、单活跃任务，服务只绑定本机；不包含公网认证、用户隔离与生产部署能力。
- DeepSeek模型ID、tokenizer和价格规则是本交付版本的固定适配。接入前须确认当前官方接口兼容性，勿直接将旧价格规则当成当前账单保证。

第三方资源和许可范围见[NOTICE](NOTICE.md)。项目自有代码尚未指定开源许可证；公开可见不等同于已授予任意再分发许可。
