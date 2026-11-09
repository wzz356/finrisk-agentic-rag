FinRisk 本地交付说明（2026-10-24）

入口为README.md。Windows使用install/start/verify.bat，Linux使用对应.sh。
运行需要Python3.11+、Node.js22.19+；Pi依赖由agent_runtime/package-lock.json固定。
原始库10,000份文书，官网866份，合计18,092个主体案例、27标签；Query场景六类。
默认DeepSeek重排和自适应pre/post；三种排序及三种时机均可切换。

运行数据需保留data/processed/knowledge/finrisk_full_20261024、data/processed/case_sources、data/raw/cases、raw_pdfs、有效Qwen索引及其模型缓存、data/risk/calibration及.env。旧知识库和不再使用的向量文件可在新版本验收后清理。
打包默认排除密钥、node_modules、工具缓存和运行日志。恢复后先install、verify，再启动Web。
新增语料需要运行scripts/build_competition_index.py --qwen刷新离线索引。

当前效果集为data/eval/competition500_v1，500案例、400开发Query、100开发核对子集、200源文书隔离测试Query，旧集和对应入口已清理。
真实API的银标测试结果和PPT可用CSV位于reports/competition500；完整说明见docs/competition_upgrade_20261019.md。
源码、新评测集与冻结结果随GitHub分发；密钥、原始PDF、全量知识库/索引及调用日志保存在本地与授权服务器。新版知识说明见docs/knowledge_update_20261024.md。
