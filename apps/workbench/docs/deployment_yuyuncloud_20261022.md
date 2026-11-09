# yuyunCloud 部署记录（2026-10-22）

公网入口：<http://156.239.2.239/>。健康检查：<http://156.239.2.239/api/health>。

保留现有 Nginx 的 80 端口入口，转发到 `127.0.0.1:8000`。本次更新的是
`finrisk.service`，没有改动 Nginx 转发配置。18081 端口的 `bid-review-demo.service`
保持原进程、启动时间和零重启次数，检查返回 HTTP 200。

## 版本与运行数据

- 源码基线：`9cd62d40511d9aba0d8ca948ec9c2c425dba01dd`，另包含本次原件读取的跨平台修复。
- 新版目录：`/opt/finrisk/releases/20261022-9cd62d4/app`。
- 当前版本链接：`/opt/finrisk/current`，指向本次 release。
- 独立 Python 环境：`/opt/finrisk/current/venv`；Python 3.12。
- Node.js：`/opt/finrisk/runtime/node-v22.20.0-linux-x64/bin/node`；Pi Agent 1.2.0。
- 私有配置：`/etc/finrisk/finrisk.env`，文件权限为 0600，不通过静态文件入口提供。
- 数据快照：基础库 9,639 份文书，官网增量 366 份文书，共 10,005 份文书、15,986 个主体案例。
- 迁移了 10,000 份原始 PDF、证据数据库、增量来源文件、轻量校准模型与预构建 Qwen 索引。
- Qwen 索引包含 117,158 个检索单元，启动复用预构建索引；未重新调用 API 编码全库。

完整主数据包 SHA-256：
`e7f69b446699d6fafd61496cff891b9d1b7a00fe2fb3d8be22d0c088797c5a03`。
部署逐文件校验使用 release 中的 `DEPLOY_MANIFEST.json`、`DEPLOY_SOURCE_MANIFEST.json`
及 `DEPLOY_PORTABILITY_MANIFEST.json`；后者记录本次代码覆盖与原件短名映射。

## 跨平台修复与验证

历史原件路径中包含 Windows 反斜杠；读取原件时统一为跨平台路径。一份原始 PDF 的文件名
包含 321 个 UTF-8 字节，超过 Linux 单文件名限制，部署时使用原文件名 SHA-256 生成短名，
保留原始内容及原名映射。原件 API 支持该短名，同时继续检查允许访问的来源目录。

- Windows 和 Linux 均通过 236 项 Python 测试，包括普通路径、Windows 路径、长名映射及目录边界检查。
- 31 项 Node 测试通过；Ruff 和 83 个源文件的 Mypy 检查通过。
- 知识库完整性检查、Competition500 数据审计与 Linux 依赖一致性检查通过。
- 正式账号下 OCR 模型初始化和图片文字识别通过。
- 临时端口验证后切换；健康接口确认 Pi worker 运行、Qwen `dense_cloud` 通道和索引就绪，启动告警为空。
- 真实 Pi 文本审查约 37 秒返回 HTTP 200，工具提交成功，无降级告警；TXT 上传分析、基础库 PDF 下载、官网增量来源下载及效果评测接口通过。

## 运维与回滚

服务配置保存在仓库的 `deploy/finrisk-yuyuncloud.service`；本次服务器实际使用的 Nginx 配置
保存在 `deploy/nginx-finrisk-yuyuncloud.conf`，提供直接打开 IP 的 80 端口入口。
单进程运行，并设置 2 核 CPU 配额、
3 GiB 内存软限制和 4 GiB 内存硬限制。服务已设置开机启动与失败重启。

```bash
ssh yuyunCloud 'systemctl status finrisk.service --no-pager'
ssh yuyunCloud 'journalctl -u finrisk.service -n 80 --no-pager'
```

旧版程序与旧虚拟环境分别保留在 `/opt/finrisk/app` 和 `/opt/finrisk/venv`；旧服务配置、
私有配置及回滚脚本位于 `/opt/finrisk/backups/20261022-9cd62d4/`。需要恢复旧版时执行：

```bash
ssh yuyunCloud 'bash /opt/finrisk/backups/20261022-9cd62d4/rollback.sh'
```

回滚脚本只恢复 FinRisk 的配置和进程。临时验证端口及临时传输服务在完成部署后关闭。
官网增量沿用应用配置，每天北京时间 04:30 执行，状态可从 `/api/collection/status` 查看。
