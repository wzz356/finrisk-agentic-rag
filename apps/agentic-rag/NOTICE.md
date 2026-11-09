# 来源与许可说明

本仓库为FinRisk Agentic RAG的独立工程交付，采用新Git历史。经用户授权，公开处罚语料、知识库、索引及向量通过同仓库Release单独提供，来源和审核范围见相应交付说明；代码历史不存储这些大型载荷。不包含模型权重、私人测试答案、真实API对话轨迹、费用账本或密钥。

- `examples/cases/`为本项目专门编写的虚构工程样例，不是真实处罚决定，也不是准确率评测集。
- `contracts/agent_rag/risk_type_dictionary.csv`保留项目使用的赛事27类标签定义，用于协议兼容；`risk-definitions.json`是对应配置。未对赛事材料或该字典另行授予第三方权利。
- `risk-criteria.json`、系统提示词与重排提示词包含项目内部解释，保持草稿标记，不代表赛事方或监管机关认可。
- `resources/deepseek-v41/`为固定版本的DeepSeek token计数资源，按其目录内的MIT许可证保留来源、许可及哈希。它们不是模型权重；文件来源见该目录的`manifest.json`。
- Pi、FastAPI等依赖通过各自包管理器安装，其许可证分别适用。仓库不包含`node_modules`或虚拟环境。

未为项目自有代码选定统一开源许可证。需要对外授予MIT、Apache等许可时，由权利人单独决定；不得将其自动扩展到赛事语料或第三方资源。
