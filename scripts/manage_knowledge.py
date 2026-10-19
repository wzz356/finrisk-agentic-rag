"""管理采集库存与登记待审维护事件；默认关闭采集和扩展使用。"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agent_rag.collection_service import CollectionService  # noqa: E402


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, help="管理目录，仅允许位于本副本runtime内；相对路径从仓库根解析"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="查看配置、开关、批次、作业及待启用版本")
    bind = sub.add_parser("bind", help="只读绑定可信旧update配置，将写目录重定向到本副本")
    bind.add_argument("--config", type=Path, required=True)
    history = sub.add_parser(
        "configure-history", help="在服务端配置只读历史源和有证据的补采起始日期"
    )
    history.add_argument("--source", type=Path, action="append", required=True)
    history.add_argument(
        "--since", required=True, help="YYYY-MM-DD；根据各栏目证据确定，不能以最后运行日代替"
    )
    sub.add_parser("import-history", help="导入已在服务端配置的历史原件，不执行旧脚本")
    crawl = sub.add_parser("crawl", help="按发布日期闭区间有界采集，不调用生成模型")
    crawl.add_argument("--since", required=True)
    crawl.add_argument("--until", required=True)
    crawl.add_argument("--max-pages", type=int, required=True, help="每栏目本轮页数，1至100")
    worker = sub.add_parser("worker", help="只执行已开启的每日采集；常驻时每30秒检查")
    worker.add_argument("--once", action="store_true", help="检查一次，并等待该次采集作业结束")
    for name, help_text in (
        ("set-collection", "独立设置每日采集开关"),
        ("set-usage", "独立设置选定扩展版本的使用开关"),
    ):
        command = sub.add_parser(name, help=help_text)
        switches = command.add_mutually_exclusive_group(required=True)
        switches.add_argument("--enabled", dest="enabled", action="store_true")
        switches.add_argument("--disabled", dest="enabled", action="store_false")
    select = sub.add_parser("select", help="手动选择已完成且父版本一致的累计版本")
    select.add_argument("--version", required=True)
    process = sub.add_parser("process", help="登记待审维护事件，沿用旧权限门；不调用模型或构建版本")
    process.add_argument("--batch-id", required=True)
    process.add_argument("--cap-cny", type=float, required=True)
    process.add_argument("--max-documents", type=int, required=True)
    process.add_argument("--max-requests", type=int, required=True)
    return parser


def _managed_root(value, project_root):
    project = Path(project_root).resolve()
    runtime = (project / "runtime").resolve()
    candidate = value if value is not None else Path("runtime/knowledge_updates")
    root = (candidate if candidate.is_absolute() else project / candidate).resolve()
    if not runtime.is_relative_to(project) or not root.is_relative_to(runtime):
        raise ValueError("ROOT_OUTSIDE_PROJECT_RUNTIME")
    return root


def _emit(value, *, error=False):
    print(
        json.dumps(value, ensure_ascii=False, allow_nan=False),
        file=sys.stderr if error else sys.stdout,
        flush=True,
    )


def _finish_job(service, thread):
    thread.join()
    job = next((item for item in service.jobs() if item["id"] == thread.job_id), None)
    if job is None:
        raise ValueError("JOB_RESULT_NOT_FOUND")
    _emit(job)
    return 0 if job["status"] == "completed" else 1


def _worker(service, once):
    while True:
        # This is the same persisted collection range used by the backend scheduler.
        current = service.store.get_value("daily_range")
        if current and service.store.get_value("crawl_success_until", "") >= current["until"]:
            service.store.set_value("daily_range", None)
        thread = service.tick()
        if thread is not None:
            result = _finish_job(service, thread)
            if result:
                return result
        elif once:
            _emit({"status": "idle", "model_requests_sent": 0})
        if once:
            return 0
        time.sleep(30)


def main(argv=None, *, project_root=None):
    args = _parser().parse_args(argv)
    service = None
    try:
        root = _managed_root(args.root, ROOT if project_root is None else project_root)
        if args.command == "process" and os.environ.get("FINRISK_AGENT_ALLOW_KB_GENERATION") != "1":
            raise ValueError("GENERATION_DISABLED")
        service = CollectionService(root)
        if args.command == "status":
            _emit(service.status())
        elif args.command == "bind":
            _emit(service.bind(args.config))
        elif args.command == "configure-history":
            service.configure_history(args.source, backfill_since=args.since)
            _emit(
                {
                    "status": "configured",
                    "source_count": len(args.source),
                    "backfill_since": args.since,
                }
            )
        elif args.command == "set-collection":
            _emit(service.set_crawl_enabled(args.enabled))
        elif args.command == "worker":
            return _worker(service, args.once)
        else:
            if args.command == "import-history":
                kind, values = "import_history", {}
            elif args.command == "crawl":
                kind = "crawl"
                values = {"since": args.since, "until": args.until, "max_pages": args.max_pages}
            elif args.command == "select":
                kind, values = "activate", {"version": args.version}
            elif args.command == "set-usage":
                kind, values = "usage", {"enabled": args.enabled}
            else:
                kind = "process"
                values = {
                    name: getattr(args, name)
                    for name in ("batch_id", "cap_cny", "max_documents", "max_requests")
                }
            return _finish_job(service, service.submit(kind, values))
        return 0
    except KeyboardInterrupt:
        _emit({"error": "INTERRUPTED"}, error=True)
        return 130
    except Exception as error:
        code = str(error).split(":", 1)[0]
        _emit(
            {
                "error": code
                if re.fullmatch(r"[A-Z][A-Z0-9_]{0,120}", code)
                else type(error).__name__
            },
            error=True,
        )
        return 1
    finally:
        if service is not None:
            service.close()


if __name__ == "__main__":
    raise SystemExit(main())
