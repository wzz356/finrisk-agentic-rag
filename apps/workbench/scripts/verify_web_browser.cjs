const { chromium } = require(process.env.FINRISK_PLAYWRIGHT_MODULE || "playwright");
const fs = require("node:fs");
const path = require("node:path");
const assert = require("node:assert/strict");

async function main() {
  const root = path.resolve(__dirname, "..");
  const out = path.join(root, "outputs/competition_upgrade_20261019/browser");
  fs.mkdirSync(out, { recursive: true });
  const launch = { headless: true };
  if (process.env.FINRISK_BROWSER_EXECUTABLE) launch.executablePath = process.env.FINRISK_BROWSER_EXECUTABLE;
  else if (process.platform === "win32") launch.channel = "msedge";
  const browser = await chromium.launch(launch);
  const baseURL = process.env.FINRISK_WEB_BASE_URL || "http://127.0.0.1:18088";
  const context = await browser.newContext({ baseURL, viewport: { width: 1365, height: 950 }, acceptDownloads: true });
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", e => errors.push(e.message));
  try {
    await page.goto(baseURL, { waitUntil: "networkidle" });
    assert.equal(await page.locator("#query-rerank option").count(), 4);
    assert.equal(await page.locator("#query-timing option").count(), 4);
    assert.equal(await page.locator("#query-scene option").count(), 7);
    await page.screenshot({ path: path.join(out, "01_workbench.png"), fullPage: true });
    const result = {};
    if (!process.argv.includes("--skip-analyze")) {
      await page.locator("#query-text").fill("产品介绍：本产品的分红收益保证每年达到8%，客户不需要阅读免责条款。现在投保就送合同约定以外的加油卡。");
      await page.locator("#query-rerank").selectOption("");
      await page.locator("#query-timing").selectOption("");
      const responsePromise = page.waitForResponse(r => r.url().endsWith("/api/query/analyze/text"), { timeout: 180000 });
      await page.locator("#analyze-button").click();
      const response = await responsePromise;
      assert.equal(response.status(), 200);
      const data = await response.json();
      await page.locator("#risk-content").waitFor({ state: "visible" });
      result.analysis = { sentences: data.risk_assessments.length, labels: data.risk_summary.risk_labels,
        state: data.risk_summary.decision_state, warnings: data.risk_summary.warnings,
        agent_runs: data.risk_summary.agent_runs, strategy: data.risk_summary.retrieval_strategy };
      assert.ok(data.risk_assessments.some(a => a.remediation_suggestions.length));
      await page.screenshot({ path: path.join(out, "02_analysis.png"), fullPage: true });
      const downloadPromise = page.waitForEvent("download");
      await page.locator("#export-submission").click();
      const download = await downloadPromise;
      const destination = path.join(out, "submission_browser.jsonl");
      await download.saveAs(destination);
      const submission = JSON.parse(fs.readFileSync(destination, "utf8").trim());
      assert.ok(submission.question_id && Array.isArray(submission.retrieved_cases));
      assert.equal(typeof submission.suggestion, "string");
      result.submission_download = true;
      const reportPromise = page.waitForEvent("download");
      await page.locator("#export-risk-report").click();
      await (await reportPromise).saveAs(path.join(out, "审查报告_browser.html"));
    }
    await page.getByRole("button", { name: "案例管理", exact: true }).click();
    await page.locator("#case-list [data-open-case]").first().waitFor();
    assert.equal(await page.locator("#case-origin option").count(), 4);
    assert.match(await page.locator("#case-source-counts").innerText(), /原始知识库.*官网采集.*网页录入/);
    const collection = await (await page.request.get("/api/collection/status")).json();
    assert.equal(collection.daily_time, "04:30");
    result.collection = { enabled: collection.enabled, start_date: collection.start_date, next_run_at: collection.next_run_at };
    const statistics = await (await page.request.get("/api/cases/statistics")).json();
    for (const [origin, count] of [["original", statistics.base_cases], ["official_crawl", statistics.crawled_cases], ["web_ingestion", statistics.user_cases]]) {
      const filtered = await (await page.request.get(`/api/cases?origin=${origin}&page_size=1`)).json();
      assert.equal(filtered.total, count);
      if (filtered.items.length) assert.equal(filtered.items[0].source_group, origin);
    }
    await page.screenshot({ path: path.join(out, "03a_collection_sources.png"), fullPage: true });
    if (statistics.crawled_cases > 0) {
      await page.locator("#case-origin").selectOption("official_crawl");
      const response = page.waitForResponse(r => r.url().includes("/api/cases?") && r.url().includes("origin=official_crawl"));
      await page.getByRole("button", { name: "查询", exact: true }).click();
      await response;
      await page.locator("#case-list .origin-chip").filter({ hasText: "官网采集" }).first().waitFor();
      await page.locator("#case-list [data-open-case]").first().click();
      await page.locator("#case-detail").getByText("官网采集", { exact: true }).waitFor();
      const original = await page.locator("#case-detail").getByRole("link", { name: "查看来源原件 ↗" }).getAttribute("href");
      const source = await page.request.get(original);
      assert.equal(source.status(), 200);
      assert.match(await source.text(), /保险/);
      result.crawled_source_download = true;
      await page.screenshot({ path: path.join(out, "03b_official_case.png"), fullPage: true });
      await page.locator("#drawer-close").click();
      await page.locator("#case-origin").selectOption("");
    }
    const recordPath = path.join(root, "data/processed/knowledge/finrisk_v2_1_20261018/data/cases.jsonl");
    const raw = fs.readFileSync(recordPath, "utf8").split("\n").find(line => line.includes('"raw_source_file"'));
    const record = JSON.parse(raw);
    await page.locator("#case-search").fill(record.case_id);
    await page.getByRole("button", { name: "查询", exact: true }).click();
    await page.locator(`#case-list [data-open-case="${record.case_id}"]`).first().click();
    await page.locator("#case-detail").getByText("机构类型", { exact: true }).waitFor();
    assert.ok(await page.locator("#case-detail").getByText("主体关联", { exact: true }).count());
    assert.ok(await page.locator("#case-detail").getByRole("link", { name: "打开原文 PDF ↗" }).count());
    const pdf = await page.request.get(`/api/cases/${encodeURIComponent(record.case_id)}/source`);
    assert.equal(pdf.status(), 200);
    assert.equal((await pdf.body()).subarray(0, 4).toString(), "%PDF");
    result.original_pdf = { case_id: record.case_id, bytes: (await pdf.body()).length };
    await page.screenshot({ path: path.join(out, "03_case_source.png"), fullPage: true });
    await page.locator("#drawer-close").click();
    const filtered = await page.request.get("/api/cases?institution_type=" + encodeURIComponent("人身险公司") + "&page_size=1");
    const filteredData = await filtered.json();
    assert.ok(filteredData.total > 0);
    assert.equal(filteredData.items[0].institution_type, "人身险公司");
    result.institution_filter_total = filteredData.total;
    await page.getByRole("button", { name: "效果评测", exact: true }).click();
    await page.waitForTimeout(500);
    await page.screenshot({ path: path.join(out, "04_evaluation.png"), fullPage: true });
    result.evaluation = await page.locator("#evaluation-content").innerText();
    result.horizontal_overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
    assert.equal(result.horizontal_overflow, false);
    assert.deepEqual(errors, []);
    result.browser_errors = errors;
    fs.writeFileSync(path.join(out, "verification.json"), JSON.stringify(result, null, 2));
    console.log(JSON.stringify({ success: true, errors, pdf: result.original_pdf, filter_total: result.institution_filter_total }));
  } finally { await browser.close(); }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
