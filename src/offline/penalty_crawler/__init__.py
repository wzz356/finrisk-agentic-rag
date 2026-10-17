"""处罚文书采集器：金监总局 + 人民银行处罚公示 → 纯文本 txt，供 add_penalty_case.py 入库。

数据源（均实测验证）：
- 国家金融监督管理总局 行政处罚栏目（itemId 4113 总局机关 / 4114 省监管局 / 4115 监管分局）
  列表与详情均为 CDN 预生成静态 JSON，pageSize 固定 18。
- 中国人民银行 政务公开 → 行政执法信息 → 行政处罚公示（银罚决字批次）

产出布局（staging，不直接写入 kb_increments，人工确认后再入库）：
  outputs/penalty_crawl/
    nfra/<itemId>/<docId>.txt
    pbc/<url段>.txt
    index.jsonl      # 每条文书一行元数据（含 source_url / publish_date / sha1）
    state.json       # 已见文书键，用于增量断点
    errors.jsonl     # 抓取失败记录，不中断整体
"""
