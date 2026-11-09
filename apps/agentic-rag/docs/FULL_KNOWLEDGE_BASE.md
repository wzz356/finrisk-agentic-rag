# 完整V2知识库交付（2026-10-19）

本页保留旧版本说明；最新交付见[2026-10-24完整知识库](KNOWLEDGE_FULL_20261024.md)。

代码与知识库统一从本仓库交付：[下载本版Release](https://github.com/wzz356/finrisk-agentic-rag/releases/tag/knowledge-v2-20261019)。大数据放Release附件，普通clone仍保持轻量。**无需Git LFS。**

## 版本与内容

本版对应Agent现用来源快照`v2-7b870a28eb8c26b42b9cfef3`，由9,639份V2底库和此前已接入的1份增量组成。本轮暂停的856份采集材料与新候选均未纳入，也未接入独立法规库。

| 内容 | 数量 |
|---|---:|
| 来源全文 | 9,640 |
| 案件卡 | 15,621 |
| 行为卡 | 25,187 |
| 证据片段 | 114,968 |
| 字段标注 | 182,364 |
| 材料记录 | 12,081 |
| 检索单元 | 65,478 |
| 官方风险标签 | 27 |

- `finrisk-v2-runtime-20261019.zip`：完整来源全文、结构化检索单元、官方字典、重排规则、BGE文档向量与对应元数据、可迁移配置模板。BM25使用这些结构化文本在运行时构建，不另有一个缺失的磁盘索引。
- `finrisk-v2-normalized-20261019.zip`：合并后的8张核心表、6张追溯扩展表、3份阅读视图及字段合同。`reading/behavior_reading.jsonl`和`reading/case_reading.jsonl`适合队友先看知识卡，`data/documents.jsonl`可回读完整来源文本。
- `release-manifest.json`与`SHA256SUMS.txt`：附件大小、SHA256及逐表计数；各压缩包内还有逐文件SHA256清单。

完整指当前已接入版本的数据与检索资产齐全，不代表最初约一万份材料全部被收录，也不代表全部经过独立人工审核。保留`human_approved=false`、`semantic_coverage_verified=false`及各行原有`pending/candidate/local_internal`等历史状态。公开交付授权不会升级事实、标签或语义质量的审核状态。

结构化表按原字段保留来源、引用、创建者未知值及历史路径，用于核查沿革；历史绝对路径仅是元数据，不要求队友重建原机器目录。运行配置通过安装器生成本机路径，重新计算配置哈希和派生快照ID。未打包私人测试材料、金标参照、模型对话轨迹、费用账本、密钥、模型权重或原始PDF二进制；来源全文已包含在两个包内。

## 下载与检查

在仓库根目录执行（也可在Release页面手动下载对应附件）：

```powershell
gh release download knowledge-v2-20261019 --repo wzz356/finrisk-agentic-rag --dir "downloads" --pattern "finrisk-v2-*.zip" --pattern "SHA256SUMS.txt" --pattern "release-manifest.json"
Get-FileHash "downloads\finrisk-v2-runtime-20261019.zip" -Algorithm SHA256
Get-FileHash "downloads\finrisk-v2-normalized-20261019.zip" -Algorithm SHA256
```

将结果与本仓库`knowledge/2026-10-19/release-manifest.json`中固定SHA256比较。只有要阅读知识卡时，下载结构化包即可；只有运行检索时，下载运行包即可。需要完整交付则下载两份。

```powershell
Expand-Archive "downloads\finrisk-v2-runtime-20261019.zip" "runtime\knowledge-v2\assets"
Expand-Archive "downloads\finrisk-v2-normalized-20261019.zip" "runtime\knowledge-v2\normalized"
python "scripts\install_knowledge_bundle.py" --bundle "runtime\knowledge-v2\assets" --verify-only
python "scripts\install_knowledge_bundle.py" --bundle "runtime\knowledge-v2\normalized" --verify-only
```

校验和阅读无需GPU或模型调用。不要在已解压目录中编辑资产，也不要覆盖已有运行版本。

## 配置完整混合检索

沿用已冻结的BGE编码条件：`BAAI/bge-small-zh-v1.5`，revision `7999e1d3359715c523056ef9478215996d62a620`，Torch `2.6.0+cu124`、Transformers `4.57.1`、CUDA、float16。当前实现要求该环境，CPU运行不是本版已验证能力。先按README安装Agent依赖，并另行准备GPU Python环境，例如：

```powershell
py -3.12 -m venv ".gpu-venv"
& ".\.gpu-venv\Scripts\python.exe" -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
& ".\.gpu-venv\Scripts\python.exe" -m pip install transformers==4.57.1 numpy huggingface_hub
```

模型文件需与运行包`dense/vectors.json`的`binding.model.files`清单完全一致，包含配置和tokenizer，不只权重。以下下载指定revision的清单文件，并复制成独立普通文件，避免缓存符号链接影响既有指纹检查：

```powershell
@'
import json, shutil
from pathlib import Path
from huggingface_hub import snapshot_download
meta = json.loads(Path("runtime/knowledge-v2/assets/dense/vectors.json").read_text("utf-8"))
model = meta["binding"]["model"]
files = [entry["path"] for entry in model["files"]]
cache = Path(snapshot_download("BAAI/bge-small-zh-v1.5", revision=model["revision"], allow_patterns=files))
dest = Path("models") / model["revision"]
dest.mkdir(parents=True, exist_ok=False)
for relative in files:
    target = dest / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cache / relative, target)
print(dest.resolve())
'@ | & ".\.gpu-venv\Scripts\python.exe" -
```

生成本机快照（安装器只校验与配置，不重算向量、不出网、不调用LLM、不切换现有current）：

```powershell
python "scripts\install_knowledge_bundle.py" --bundle "runtime\knowledge-v2\assets" --snapshot-output "runtime\knowledge-v2\snapshot" --model-path "models\7999e1d3359715c523056ef9478215996d62a620" --gpu-python ".gpu-venv\Scripts\python.exe"
```

安装后保留assets目录，snapshot通过绝对路径引用它。移动目录后请向新的`--snapshot-output`重新安装；既有安装目录拒绝覆盖。派生快照ID与原ID不同，知识正文和向量不变，原ID在`distribution_origin_snapshot_id`中保留。

## 启动

完整混合检索仍包含真实DeepSeek重排，必须自行配置密钥和费用限额；下载、校验、安装不会自动启用收费。按[运行说明](RUNNING.md)配置`DEEPSEEK_API_KEY`或仓库外密钥文件后，可显式启动：

```powershell
.\start-agent-rag-v2.ps1 -SnapshotPath "runtime\knowledge-v2\snapshot" -AllowPaidRetrieval -RerankCapCny 0.5 -EnableLiveAgent
```

此命令会启用真实Agent和真实重排，任务可能发送输入及案例片段至DeepSeek并产生费用。原有单任务/账本限制继续生效；不要删除账本绕过费用记录。启动不等于真实效果已验证，也不自动启用采集或知识维护发布。

## 本轮验证边界

验证报告见同Release的`verification.json`。本轮检查归档完整性、逐文件SHA、完整来源/单元加载、全文引文、核心ID及已检查引用关系、向量顺序与模型文件指纹、独立目录安装。未新增付费审查实验，未宣称提高识别准确率。原Agent与V2检索的既有质量限制继续适用。

数据来自赛事处罚材料与此前已接入的官方采集摘要；官方文书与第三方模型分别沿用其原有权利归属，本项目代码的许可状态见`NOTICE.md`。
