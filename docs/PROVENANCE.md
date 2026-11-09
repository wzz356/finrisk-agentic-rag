# 来源与历史保留

本仓库为比赛项目的后续工程整合，由团队共同研发成果组成。模块映射固定记录在[SOURCE_REPOSITORIES.json](../SOURCE_REPOSITORIES.json)。

| 目录 | 来源仓库 | 导入提交 |
|---|---|---|
| `apps/workbench` | [qby1210/finrisk-framework](https://github.com/qby1210/finrisk-framework) | `f68cf35b5dd2c358e3977da0f029ad984d60149c` |
| `apps/agentic-rag` | [wzz356/finrisk-agentic-rag](https://github.com/wzz356/finrisk-agentic-rag) | `c0cf642bc2ee1aad6ea8759b8fe6effecc72da5c` |

使用`git subtree add`且不squash，原始提交、作者及时间保留在同一Git历史中；导入时两个模块的跟踪文件分别为309和642个，业务源码按原快照保留。

整合新增根目录项目说明、启动器、入口测试和统一文档。原始README、版本说明、评测、失败记录、来源与许可说明仍在各模块内。

完整知识资产继续引用原Release，不将源码搬迁视为重新审核或重新发布语料。原实验中的本机路径属于历史运行定位记录，不是新仓库的安装路径。

可用以下命令检查历史和来源：

```powershell
git log --all --graph --oneline
git shortlog -sn HEAD
git merge-base --is-ancestor f68cf35b5dd2c358e3977da0f029ad984d60149c HEAD
git merge-base --is-ancestor c0cf642bc2ee1aad6ea8759b8fe6effecc72da5c HEAD
```

根目录及整合内容不替代原作者署名，不将团队成果归为个人独立完成。许可范围见[NOTICE.md](../NOTICE.md)。
