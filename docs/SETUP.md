# 安装与运行

## 环境

- 推荐Python 3.12；工作台原要求Python 3.11+，Agent交付按Python 3.12配置。
- Node.js 22.19+及npm。
- Windows的交互演示使用PowerShell。根目录入口不自动安装依赖、读取密钥或下载知识资产。

两个模块使用独立虚拟环境。它们的锁定版本不同，不要将两份requirements装入同一个环境。

## 离线Agent演示：Windows

从仓库根目录执行：

```powershell
python -m venv "apps\agentic-rag\.agent-venv"
& "apps\agentic-rag\.agent-venv\Scripts\python.exe" -m pip install -r "apps\agentic-rag\requirements-agent-rag.txt"
npm --prefix "apps\agentic-rag\agent_harness" ci --ignore-scripts --no-audit --no-fund
npm --prefix "apps\agentic-rag\agent_harness" run build
& "apps\agentic-rag\.agent-venv\Scripts\python.exe" "apps\agentic-rag\scripts\build_demo.py"
python "finrisk.py" demo
```

页面：<http://127.0.0.1:8765/agent-rag>。演示使用三份合成案例和mock模型，启动入口覆盖继承的live开关为关闭。

`build_demo.py`拒绝覆盖已有快照；首次构建后直接执行最后一行启动。需要其他快照或真实模型时，进入模块使用原运行说明和显式配置；根目录`demo`只用于离线演示。

可以覆盖解释器和端口：

```powershell
python "finrisk.py" demo --python "apps\agentic-rag\.agent-venv\Scripts\python.exe" --port 18765 --node-port 18766
```

Python和Node端口必须不同。按Ctrl+C停止本轮进程。

## 工作台

安装依赖：

```powershell
Push-Location "."
try { & "apps\workbench\install.ps1" }
finally { Pop-Location }
```

安装器创建模块内`.venv`和`.env`。在`apps/workbench/.env`配置本地模型密钥及数据路径；完整工作台还需要兼容知识库、原始资产和预构建索引。

数据恢复先阅读[工作台README](../apps/workbench/README.md)和[知识库更新说明](../apps/workbench/docs/knowledge_update_20261024.md)。完整资产来自[knowledge-full-20261024](https://github.com/wzz356/finrisk-agentic-rag/releases/tag/knowledge-full-20261024)，文件校验清单在[模块配置](../apps/workbench/configs/knowledge_full_20261024.json)。

完成配置与资产恢复后：

```powershell
python "finrisk.py" workbench --port 8000
```

页面：<http://127.0.0.1:8000>。自定义端口、原始模块启动器及本地环境均可继续使用。

工作台配置可启用真实模型、云向量、重排和官网增量采集。它与默认离线Agent演示的运行条件不同。

## Linux离线构建与核验

独立Agent的自动化smoke脚本可以直接启动临时Python/Node服务并在结束后关闭；根目录交互`demo`启动器使用Windows原启动脚本。

```sh
python3.12 -m venv apps/agentic-rag/.agent-venv
apps/agentic-rag/.agent-venv/bin/python -m pip install -r apps/agentic-rag/requirements-test.txt
npm --prefix apps/agentic-rag/agent_harness ci --ignore-scripts --no-audit --no-fund
npm --prefix apps/agentic-rag/agent_harness run build
cd apps/agentic-rag
.agent-venv/bin/python scripts/smoke_demo.py
```

工作台Linux安装与部署入口保留在模块内。部署时以`apps/workbench`为应用根目录，相应调整原service配置中的路径。

## 真实模型与完整知识库

真实模型和V2检索需分别安装兼容资产、配置密钥、显式live开关及预算；参见[Agent运行说明](../apps/agentic-rag/docs/RUNNING.md)与[完整知识库说明](../apps/agentic-rag/docs/KNOWLEDGE_FULL_20261024.md)。

原始PDF、全库向量、密钥、任务状态与费用账本保留在本机，不加入Git。下载资产不等于完成范围复核或人工批准。
