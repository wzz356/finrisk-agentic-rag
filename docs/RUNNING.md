# 运行与外部资产接入

## 默认离线演示

按README创建Python环境、安装Node依赖并编译，运行`build_demo.py`和`start-agent-rag.ps1`。默认执行`mock-bm25-baseline-v1`，没有模型出网和GPU依赖。

示例请求（服务启动后，在另一个PowerShell窗口执行）：

```powershell
$body = @{ text = '购买这份保险就送购物卡。'; mock_scenario = 'gift-example' } | ConvertTo-Json
$task = Invoke-RestMethod 'http://127.0.0.1:8765/api/agent-rag/tasks' -Method Post -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body))
Invoke-RestMethod ("http://127.0.0.1:8765" + $task.poll_url)
```

任务返回后需要轮询至`completed/partial/failed/interrupted`。`mock_scenario=gift-example`只接受指定合成输入；其他输入使用`generic`，不会假装完成真实语义判断。`examples/requests.json`含三条请求。

任务轨迹写入`runtime/`，已被Git忽略。来源原文与输入都可能进入任务文件；不要把运行目录提交公开仓库。

## 使用自己的原文做BM25快照

仅使用你有权处理的UTF-8 TXT。以下操作仅构建索引，不调用模型：

```powershell
& ".\.agent-venv\Scripts\python.exe" "scripts\build_agent_snapshot.py" --source-root "D:\your-data\texts" --dictionary "contracts\agent_rag\risk_type_dictionary.csv" --output "runtime\agent_rag\snapshot-own"
.\start-agent-rag.ps1 -SnapshotPath "runtime\agent_rag\snapshot-own"
```

构建器只扫描该目录直接包含的`*.txt`。保留原文哈希、字符区间、来源回读校验。输入格式沿用原工程官方TXT适配器；字段中的“official”并不替你保证外部文件真实性。默认仍为mock。

## 可选：真实DeepSeek Agent + BM25

这是收费操作，需要自己显式执行。不会在安装、编译或离线测试时触发。

1. 通过`DEEPSEEK_API_KEY`或`DEEPSEEK_API_KEY_FILE`提供密钥；文件须放仓库外。`-KeyFile`也支持显式文件路径。不要将密钥写进命令历史或Git。
2. 设置独立账本路径`FINRISK_AGENT_TRIAL_LEDGER`（例如当前仓库`runtime/model-trial.json`）。现有试跑保护固定为账本累计1元/18次、单任务6次；达到上限即停。不要靠删除账本重置已有费用。
3. 使用`start-agent-rag.ps1 -EnableLive`启动；API请求显式指定`experiment_profile_id=live-deepseek-flash-bm25-v1`。网页仅是mock演示入口，不自动切换live。
4. API适配和费用规则按交付版本冻结；真实运行前核实接口、模型和价格，必要时更新适配并重新验收。

环境变量例子只给空值见`.env.example`；不会自动加载`.env`。代码只读取显式环境变量或显式密钥路径。

## 可选：V2混合检索

完整知识库和文档向量已作为同仓库Release附件交付，优先按[完整知识库安装说明](FULL_KNOWLEDGE_BASE.md)操作。模型权重与GPU环境另行准备。以下为旧实验兼容准备器的说明，需要原实验兼容的：

- 规范化`documents.jsonl`：`source_id/official_file_id/full_text/text_sha256/text_version`等字段。
- 结构化单元`structured.jsonl`：`unit_id/file_id/text`、行为/主体/条件定位及关联信息。
- 与这些单元完全对应的冻结BGE文档向量、元数据和模型指纹；旧配置适配原实验65475个单元；增量版本必须显式声明incremental-v1及完整document_count，不能只改数量绕过资产校验。
- 本地GPU Python环境、BGE指定模型revision、原Torch/Transformers版本及CUDA半精度配置。
- 回放模式另需冻结查询向量、真实评分缓存和候选映射；只支持已有缓存对应的查询。

`scripts/prepare-agent-rag-v2.py`是原实验目录结构的兼容准备器，使用`--experiment-root`、`--destination`、`--scope`、`--mode`、`--gpu-python`。先运行`--help`，检查代码中的相对文件结构；它不下载资产、不训练、不重算文档向量、不读取密钥。

准备好的快照含资产绝对路径及SHA，适用于本机；搬迁机器后需要用对应本机路径重新准备，不直接复用旧manifest。

```powershell
& ".\.agent-venv\Scripts\python.exe" "scripts\prepare-agent-rag-v2.py" --experiment-root "D:\your-experiment" --destination "runtime\agent_rag\snapshot-v2" --scope full9639 --mode live --gpu-python "D:\gpu-env\Scripts\python.exe"
```

上面只准备本地清单。真实运行需要显式指定`-AllowPaidRetrieval -RerankCapCny`，真实Agent再加`-EnableLiveAgent`；参考`start-agent-rag-v2.ps1`的参数。模型和重排共用`DEEPSEEK_API_KEY`环境变量也可分别指定文件。**mock Agent搭配live检索仍会产生重排费用。**

V2每次搜索限制1条查询，单任务最多2次搜索；双路各100个单元、RRF(k=60)后取100个单元评分；全服务重排并发8。真实重排按任务单独受费用、请求数和截止时间保护，Agent另有持久账本。这里的金额为本地保守估算，非供应商账户账单。


### 可选的片段编号重排协议

兼容准备器默认沿用quote-copy-v1。启用span-ref-v2时，另存一份新的快照manifest，在backend中显式设置rerank_protocol为span-ref-v2，并将backend.assets.rules指向resources/agent_rag/deepseek-rerank-span-rules.md及该文件实际SHA256；其他知识资产绑定保持。不要覆盖已有执行使用的清单。

span-ref-v2只用于live重排；旧评分回放继续使用quote-copy-v1。协议、规则或请求模板变化会改变缓存/检查点绑定，必须重新准备运行，不能借旧缓存掩盖格式差异。片段引用仍必须回填同一单元连续原文。

## 服务边界

Python对外演示API和Node内部服务仅绑定`127.0.0.1`，启动脚本为本轮服务生成随机认证token。不要直接反向代理成公网服务。任务重启后只读已有结果，不自动恢复付费请求。

状态接口不加载重资产；V2需要warmup，readiness未就绪返回503。启动脚本跟踪本轮进程树，结束时关闭自己的进程；不会按端口杀别的窗口。Windows需允许本机进程查询。

其他系统可按同一环境变量手动启动Python与Node；本次只验收Windows启动流程。

## 增量知识更新与实验

[知识更新说明](KNOWLEDGE_UPDATES.md)介绍候选隔离、不可变版本、current切换及回滚；[实验说明](EXPERIMENTS.md)介绍三组比较与首轮证据、耗时和费用的记录边界。两者都需另外提供兼容资产，不随默认合成演示自动运行。
