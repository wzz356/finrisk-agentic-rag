# 独立采集、维护登记与手动启用

> 公开包说明：本文涉及的真实V2规模和既有库存属于研发环境。公开仓库只包含代码及合成夹具，默认未绑定真实V2；使用者需提供自己的兼容资产。真实维护适配器与完整生产构建器仍未配置。

采集模块已整合到当前主项目，默认管理根为 `runtime/knowledge_updates`。原件库存、维护候选和Agent检索分别管理；首次启动每日采集、扩展使用均关闭，真实维护模型执行不可用。主项目现用9,640份来源、65,478条结构单元的知识基线保持原样。

## 配置与库存

在当前仓库根运行以下PowerShell命令。尖括号内容须替换为已确认的本机路径或状态标识。

```powershell
$python = '.\.agent-venv\Scripts\python.exe'
& $python '.\scripts\manage_knowledge.py' status
& $python '.\scripts\manage_knowledge.py' bind --config '<可信旧update配置的绝对路径>'
& $python '.\scripts\manage_knowledge.py' configure-history --source '<含index.jsonl的历史目录绝对路径>' --since '2026-08-28'
& $python '.\scripts\manage_knowledge.py' import-history
```

绑定要求基础快照、normalizer、projector和contract齐全，并校验绝对路径与SHA；写目录重定向到本项目。公开代码不包含完整私有运行资产。`configure-history`可重复传`--source`；浏览器接口不接受本机路径。导入保留原始字节和来源，全文SHA去重；库存材料不会自动成为检索知识。

顶层`--root 'runtime/knowledge_updates'`可更换管理根，必须放在子命令前；解析后的路径必须在本项目`runtime`内。不能指向源项目、历史原件或外部链接。

## 有界采集与调度

```powershell
& $python '.\scripts\manage_knowledge.py' crawl --since '2026-08-28' --until '2026-10-18' --max-pages 2
& $python '.\scripts\manage_knowledge.py' set-collection --enabled
& $python '.\scripts\manage_knowledge.py' worker --once
& $python '.\scripts\manage_knowledge.py' worker
& $python '.\scripts\manage_knowledge.py' set-collection --disabled
```

日期为发布日期闭区间，每栏目页数上限1–100；覆盖NFRA的4113、4114、4115栏目。限页、详情失败或列表失败保留游标，重复同一区间可续跑；只有日期连续覆盖起点且列表确认完整才推进水位。两遍有界列表确认也计入页预算，不等于官网提供原子历史快照。补采起点必须结合各栏目覆盖证据确定，示例日期不能证明更早没有缺口。

常驻服务或worker每30秒检查，开启后按北京时间每天08:00采集原件。退出期间不能采集；当日已经开始但中断的轮次需核对后手动续跑，不会在重启时重复自动执行。关闭开关阻止后续轮次，当前作业继续完成。未安装Windows计划任务；仅打开开关不保证系统后台常驻。worker不会调用模型、登记维护事件、选版或开启扩展使用。

## 维护登记与版本使用

`status.configuration.ready`表示基础资产就绪；`generation_allowed`仅表示旧环境权限。默认`process`只把指定批次登记为待审维护事件，返回`maintenance_queued`、`model_requests_sent=0`、`published=false`；不会收费、构建向量、生成可选版本或启用扩展。旧的`FINRISK_AGENT_ALLOW_KB_GENERATION=1`权限门和三个有界参数仍保留，参数不代表实际消费。

```powershell
& $python '.\scripts\manage_knowledge.py' process --batch-id '<批次ID>' --cap-cny 1 --max-documents 2 --max-requests 4
```

未设置旧权限时该兼容命令拒绝执行。管理页“登记维护事件”使用专用维护入口。真实双模型适配器与完整生产构建器尚未接入，当前不能从待审事件生成可供选择的新生产版本。旧`update_agent_knowledge.py --generate`仍是单独的显式收费候选路线，`--publish`始终拒绝，不能绕过维护审批。

只有已完成、获批且主机准入的完整版本才能选择；选择与使用开关分开。首次选择不自动开启，关闭使用回到固定基线，保留选择和原件。新候选或后续发布不能自动替换使用中的选择；陈旧父版本须重建。当前任务固定其开始时的快照。

```powershell
& $python '.\scripts\manage_knowledge.py' select --version '<已准入完整版本>'
& $python '.\scripts\manage_knowledge.py' set-usage --enabled
& $python '.\scripts\manage_knowledge.py' set-usage --disabled
```

## 本机管理与限制

管理地址为`/agent-rag/knowledge`。状态接口不初始化GPU或模型；写入受loopback、同源和CSRF检查保护。显式`FINRISK_AGENT_KB_REGISTRY`优先，其次显式`FINRISK_AGENT_SNAPSHOT`；均未指定时才使用已绑定的managed注册器。没有usage文件的历史注册器保留current指针兼容。

```powershell
& $python -X utf8 -B -m uvicorn 'agent_rag.app:app' --app-dir 'src' --host '127.0.0.1' --port 8789 --workers 1
# 本机访问 http://127.0.0.1:8789/agent-rag/knowledge
```

本轮代码整合不访问NFRA或模型服务；官网可达、真实历史导入数量、各栏目覆盖与检索质量必须分别验收。库存迁入后也保持待审，不能把来源副本历史报告当作当前项目验收。详见[维护门禁](KNOWLEDGE_MAINTENANCE.md)与[工程回放](KNOWLEDGE_MAINTENANCE_REPLAY.md)。
