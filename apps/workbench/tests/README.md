# 验证入口

本地功能回归不请求外部模型：

```powershell
.venv\Scripts\python.exe -X utf8 -m pytest -q
node --test agent_runtime/test/*.test.mjs tests/web/*.test.mjs
.venv\Scripts\python.exe -X utf8 -m ruff check src scripts tests
.venv\Scripts\python.exe -X utf8 -m mypy src
```

覆盖原文定位、跨句语境、否定范围、多主体拆分、非保险排除、入库回滚、证据编号、三种重排与三种检索时机。

真实API效果验证只使用新的500案例协议：

```powershell
.venv\Scripts\python.exe -X utf8 scripts/audit_competition_dataset.py
.venv\Scripts\python.exe -X utf8 scripts/benchmark_competition500.py --help
```

数据划分、标注来源和指标定义见 `docs/dataset_card.md`。旧效果集和对应脚本已退出当前评测。
