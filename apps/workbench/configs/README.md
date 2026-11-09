配置说明

运行配置由src/settings.py及本地.env读取，模板为.env.example。
FINRISK_RERANK_PROVIDER：none、cloud、deepseek；FINRISK_RETRIEVAL_TIMING：post、pre、adaptive。
当前默认deepseek/adaptive，由新开发集综合选择冻结。三种排序和三种时机全部保留。
FINRISK_LOCAL_CALIBRATION=true启用400开发Query训练的本地候选模型与示例。
FINRISK_CASE_KB_ROOT指向新库finrisk_v2_1_20261018，不回退旧案例库。

risk_domain_mapping.json供知识库重建，import_competition_kb.py复制到当前库；运行时读当前库字典。
真实API密钥只放.env或环境变量，不写入输入、配置说明和报告。
