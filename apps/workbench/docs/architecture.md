FinRisk 当前架构

应用采用模块化单体：FastAPI 负责接口装配，application 编排解析、风险、检索和复核；domain
保存稳定契约，storage 读取同一份知识库。浏览器通过 API 展示原文、证据、案例和审核记录。
Python 的大模型任务统一交给 agent_runtime/ 的 Pi Agent；密钥只在本地环境中使用。

输入先由 extraction 解析并保留原文偏移，再由 query 完成六场景识别、切句、说话人与作用域。
risk 使用本地规则、400开发Query训练的轻量候选模型及Pi/DeepSeek复核，逐机制保存风险和必要事实未知状态。
1600字且20句以上的长文先做完整分区语义初筛，再详细核查候选；规则标签和上下文依赖强制保留，失败回到完整复核。adaptive 根据条件
选择 pre 材料，按作用域/机制共享 post 案例池，并做一次有预算的定向补查；post 和 pre 仍独立可选。
相似案例分数不进入风险概率计算。自适应检索已统一通过 _retrieve_by_mechanism，旧不可达分支已删。

retrieval 使用 BM25 与同一模型/同一维度的 Qwen 文档及 query 向量，中性行为查询融合后再重排。
重排 provider 为 none、cloud（Qwen）和 deepseek（Pi 列表排序）。DeepSeek 的下标排列经两端
完整性校验，失败回退。可选 BGE 与 TF-IDF 后端仍保留；不同模型缓存不混用。

当前默认案例库为 data/processed/knowledge/finrisk_full_20261024。原始分区10,000份文书，
官网分区联合既有增量后866份，主体案例18,092个；服务器另保留原有网页录入。
列表按文书计数，区分处罚、拟处罚、许可与保险范围待核。原始 PDF 位于 raw_pdfs，均已绑定。
证据与法条通过 SHA、ID 和来源区间回核；机制指南不是法律原文，候选状态和人工待审状态继续保留。

上游Release按固定SHA重建，P0法规和机制资产独立冻结，指纹及原文区间持续校验。评测参考隔离在 data/eval 中，
不进入索引或模型输入。运行过程在本地outputs，公开评测摘要在reports/competition500。正式构建、验收及评测入口见 README。
