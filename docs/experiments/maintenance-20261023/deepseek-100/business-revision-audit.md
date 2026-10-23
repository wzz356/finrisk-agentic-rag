业务修订只读复核（最终14案）

最终已读取100个terminal，全部14个business_revision_used=true均已纳入：B006、B015、B018、B020、B031、B044、B059、B061、B062、B074、B080、B085、B090、B096。B093只有格式修正、没有业务修订，不纳入本报告。此前88个terminal的11案快照已由本文替代。

本报告由Codex按冻结原文、P初稿、Reviewer问题、PR修订、必要的完整字段及字典进行自动回溯判断。不是人工金标，不给出全量准确率，也不把模型approve等同语义全对。

11案存在有原文依据的具体标签删除/降级，其中9案取得实验协议approve；B059责任未闭合，B090的PR未完成审核。B015主要完成处罚对象姓名澄清，仍有缺目的要件的标签。B020只增强已有“其他”的说明；B085将“其他”改为空标签虽允许，但不能证明初稿错误，两案不计作已证标签纠错。

14案模型终态：实验协议approve 9、format_repair_exhausted 2、held 1、reviewer_requested_changes 1、budget_stopped 1。此分布是运行状态，不是语义正确率。

| ID | 模型终态 | 修改与复核结论 |
|---|---|---|
| B006 | experiment_protocol_approved_not_applied | 删除/降级不满足全部要件的标签，有原文依据；三项行为及机构责任均保留。 |
| B015 | format_repair_exhausted | 刘晓妍是明确受处罚个人，case主体澄清有依据；行为主体仍不等于亲自实施者，机构归属和标签过度归类尚未解决。 |
| B018 | experiment_protocol_approved_not_applied | 把“委托无资质机构销售”改为空标签正确，保留原行为。 |
| B020 | held | 初稿本已仅贴“其他”；第二稿只加强uncertainty与reason说明，不能算作删除错误标签的成功纠错。 |
| B031 | experiment_protocol_approved_not_applied | 删除培训材料和客户信息两个越界标签有依据；四项独立行为仍被合并为一条，粒度要求未落实。 |
| B044 | experiment_protocol_approved_not_applied | “虚列费用套取资金”降为“其他”有依据，原文只支持虚假财务业务资料。 |
| B059 | reviewer_requested_changes | 删错标签和简化个人姓名有依据；机构actor改个人后再被要求改回机构/未知，责任归属未闭合。 |
| B061 | experiment_protocol_approved_not_applied | 把财务业务数据不真实的客户信息标签降级有依据；个人责任边界和另一标签仍值得复核。 |
| B062 | experiment_protocol_approved_not_applied | 删除“诱导投保”正确：被诱导对象是保险代理人，原文未写诱导消费者投保。 |
| B074 | experiment_protocol_approved_not_applied | 删除“虚列费用套取资金”正确，原文没有套取资金目的或结果。 |
| B080 | experiment_protocol_approved_not_applied | 医护人员不等于无资质机构，删去该标签改其他有依据；三主体处罚分别保留。 |
| B085 | format_repair_exhausted | 把“其他”改空标签符合允许0标签，但不是由原文要件迫使的纠错；Reviewer对兜底标签尺度及处罚对象识别不稳。 |
| B090 | budget_stopped | 两项标签降级有依据，但P已检出同决定仍误称不同案，Reviewer分页漏读，新增判断漏改；PR未通过投影上界及后续事件预算，未形成复核终稿。 |
| B096 | experiment_protocol_approved_not_applied | 不把第一项“套取资金”的目的扩给第二项“编制虚假财务资料”，标签分离有依据。 |

需要单独保留的失败/问题证据：

- B015：被处罚个人刘晓妍有直接文字依据；不等于已证其亲自实施三项行为。修订的actor uncertainty已承认实施主体未知。原文只写“虚列费用”，两稿却都肯定贴“虚列费用套取资金”，Reviewer漏检目的要件。机构归属段被改为needs_clarification；最终Reviewer以只有non_blocking问题仍request_changes，触发合同一致性失败，终态format_repair_exhausted。
- B020：P初稿risk_tags已经只有“其他”，uncertainty也说字典无直接对应。Reviewer把不存在的“虚列费用套取资金”当需删除对象，PR仅扩写不确定性，没有业务值实质变化。第二轮Reviewer模型失败，held必须保留，不算纠错成功。
- B059：第一轮Reviewer要求机构actor改张娟及responsible_person；第二轮又因未证亲自实施要求改机构或未知，形成可观测的往返矛盾。PR虽加了个人实施未知，仍不能解决actor含义。删去“委托无资质机构销售”是有依据的独立纠正。旧2025年37号机构案与本2026年27号个人案是否双罚关联未核验；相同案由不证明同一事实，不得据此停用或覆盖旧案。Reviewer还误称derived_summary支持时须用supported_as_hypothesis，与当前协议不符。
- B031：两个越界标签已删除；四项案由全文仍在，但合成一条behavior，未落实独立行为分拆。不能把此问题表述为原文字面事实已被删除。
- B061/B062/B074：本次标签修正有依据，但个人actor是责任人员关联还是亲自实施者事实，Reviewer尺度与B059不同。B061已保留个人实施未知；B062/B074的actor/role为fact且没有该限定。这里记录字段边界风险，不把所有个人处罚自动判错。B061保留的“虚列费用套取资金”还需说明“虚构中介业务套取手续费”如何完整满足费用列支要件。

逐案字段、前后变化、Reviewer意见、依据原句、未解决问题与输入SHA详见同目录business-revision-audit.json。源文引文已逐条检查为全文子串，所有读取输入在落盘前复算SHA未变；只新增本JSON及本文，不修改冻结、终态或角色结果。


最终补核三案：

- B085：字典“其他”定义就是其他类型违规行为，“提供有虚假陈述的报告”有原文依据，空标签是允许选项，不是唯一正确修复。Reviewer第二轮否定subject中明确受罚的祥云县支公司，但处罚段直接点名该公司，不能只允许首行证据。重复案由是排版重复还是两项事实确实未知，应留待核；最终仍因只有non_blocking问题却request_changes触发合同失败。
- B090：已独立读取旧来源FRV2-SOURCE-F006485-1f05a8186df09e487a899c36全文718码点，正文SHA c2395e8339f3599a207292be785a2e7419520e92b0f83702c8b956d1526c7951；总局完整文号、主体、日期、六项事实、法律依据和机构及25名个人处罚均为同一决定重现。P实际“文号+主体”查询返回4条，第4条就是该来源，raw_evidence无丢弃；P未读旧全文却称命中均不同案。R同查询只取默认前3条、next_offset=3，未补页。因此分别是P命中后误判与R分页遗漏，不能笼统写成全库未命中。只搜文号的候选排名22不是实际P查询排名。两项标签降为“其他”有依据，但P/PR的add和不同案理由都未纠正；随后PR投影越界，格式修正因单事件预算预留不足停发，未形成终稿。
- B096：原句先写“直销业务虚挂代理人套取资金”，后写“编制虚假财务资料”。只给后者删除套取资金标签、保留前者与两项行为区分，有原文依据；未把机构行为逐一归给两名责任人员。

B090同案重现需重复核对；这不授权改正式库、停用旧案或宣布旧知识卡错误。完整旧源、实际查询返回ID和分页信息已加入JSON。所有读取输入再次复算SHA未变；本任务只更新business-revision-audit.json/.md。
