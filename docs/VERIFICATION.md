# 验证入口

实际执行范围与已知失败见[本次整合记录](VERIFICATION_RESULTS.md)。全量公开回归仍有原快照可复现的问题，部分工作台用例依赖另行恢复的知识资产。

## 整合入口

在仓库根目录执行，无应用依赖：

```powershell
python -m unittest discover -s "tests" -v
```

覆盖跨工作目录与空格路径启动、端口传递、子进程失败退出码、缺少环境提示、离线demo覆盖继承的live开关。demo启动行为测试使用临时脚本，不调用模型。

## Agent公开回归与离线端到端

先完成[安装](SETUP.md)，安装测试依赖并进入模块：

```powershell
& "apps\agentic-rag\.agent-venv\Scripts\python.exe" -m pip install -r "apps\agentic-rag\requirements-test.txt"
npm --prefix "apps\agentic-rag\agent_harness" test
Push-Location "apps\agentic-rag"
try {
    & ".agent-venv\Scripts\python.exe" -m pytest tests -q
    & ".agent-venv\Scripts\python.exe" "scripts\smoke_demo.py"
    & ".agent-venv\Scripts\python.exe" "docs\experiments\incremental-maintenance-20261024\verify_results.py"
} finally { Pop-Location }
```

smoke会创建临时合成快照，启动Python与Node服务，验证任务、进度、原文及live拒绝路径，完成后关闭本轮服务。公开回归不包含研发私有资产测试，不能把通过数量解释为全部研发结果。

## 工作台

完整数据验收需要本地知识资产。已恢复资产时，在`apps/workbench`中使用原`verify.ps1`；无全量知识资产时可先运行源码回归与宽松的交付检查：

```powershell
Push-Location "apps\workbench"
try {
    & ".venv\Scripts\python.exe" "scripts\verify_delivery.py" --allow-missing-knowledge
    & ".venv\Scripts\python.exe" -m pytest -q
    node --test "tests/web/*.test.mjs"
    node --test "agent_runtime/test/*.test.mjs"
} finally { Pop-Location }
```

这些检查不替代真实模型质量评测或全库索引验收。

## Windows临时目录

维护回放会生成较长的带哈希文件名。若仓库路径很长，测试临时目录也放在其下可能超过Windows路径限制。可通过pytest的`--basetemp`指定一个较短且专用于本轮测试的目录；不要复用存放业务数据的目录。

Node全量测试若不能自然退出，先保留失败和超时记录，再停止本轮测试进程；不要把中止的运行记作通过。
