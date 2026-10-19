"""官网文书候选预检/生成；正式构建与发布须使用受控维护审批入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agent_rag.knowledge_updates import run_update, update_lock, write_json  # noqa: E402
from agent_rag.v2_snapshot import V2KnowledgeSnapshot, file_hash  # noqa: E402


def load_config(path, *, verify_build=True):
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "finrisk-web-update/1.0":
        raise ValueError("INVALID_UPDATE_CONFIG")
    if not isinstance(config.get("registry"), str) or not Path(config["registry"]).is_absolute():
        raise ValueError("INVALID_REGISTRY_CONFIG")
    if not verify_build:
        return config
    for asset in config["build_assets"].values():
        if (
            not Path(asset["path"]).is_absolute()
            or file_hash(Path(asset["path"])) != asset["sha256"]
        ):
            raise ValueError("BUILD_ASSET_HASH_MISMATCH")
    if file_hash(Path(config["base_snapshot"]) / "manifest.json") != config["base_manifest_sha256"]:
        raise ValueError("BASE_MANIFEST_HASH_MISMATCH")
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="绑定本机既有V2资产，不调用模型")
    for name in (
        "config",
        "base-snapshot",
        "normalizer",
        "projector",
        "contract",
        "registry",
        "crawl-dir",
    ):
        init.add_argument("--" + name, type=Path, required=True)
    update = sub.add_parser("update", help="默认只读预检；生成、官网采集、发布需分别显式开启")
    update.add_argument("--config", type=Path, required=True)
    update.add_argument("--crawl", action="store_true")
    update.add_argument(
        "--columns", type=int, nargs="+", choices=[4113, 4114, 4115], default=[4113, 4114, 4115]
    )
    update.add_argument("--max-pages", type=int, default=1)
    update.add_argument("--delay", type=float, default=1.5)
    update.add_argument("--max-documents", type=int, default=3)
    update.add_argument("--generate", action="store_true", help="向DeepSeek发送原文并实际扣费")
    update.add_argument(
        "--publish", action="store_true", help="旧参数已禁用；正式发布须经过维护审批"
    )
    update.add_argument(
        "--include-review",
        action="store_true",
        help="显式重新检查待审候选，默认跳过以免占用后续名额",
    )
    update.add_argument("--cap-cny", type=float, default=1.0)
    update.add_argument("--max-requests", type=int, default=6)
    update.add_argument("--key-file", type=Path)
    status = sub.add_parser("status")
    status.add_argument("--config", type=Path, required=True)
    rollback = sub.add_parser("rollback")
    rollback.add_argument("--config", type=Path, required=True)
    rollback.add_argument("--version", required=True)
    args = parser.parse_args(argv)
    if args.command == "update" and args.publish:
        raise ValueError("MAINTENANCE_APPROVAL_REQUIRED")
    if args.command == "init":
        if args.config.exists():
            raise ValueError("CONFIG_ALREADY_EXISTS")
        snapshot = V2KnowledgeSnapshot(args.base_snapshot)
        if snapshot.manifest.get("backend", {}).get("mode") != "live":
            raise ValueError("BASE_MUST_BE_LIVE_SNAPSHOT")
        config = {
            "schema_version": "finrisk-web-update/1.0",
            "base_snapshot": str(args.base_snapshot.resolve()),
            "base_manifest_sha256": file_hash(args.base_snapshot / "manifest.json"),
            "registry": str(args.registry.resolve()),
            "crawl_dir": str(args.crawl_dir.resolve()),
            "build_assets": {
                name: {
                    "path": str(getattr(args, name).resolve()),
                    "sha256": file_hash(getattr(args, name)),
                }
                for name in ("normalizer", "projector", "contract")
            },
        }
        write_json(args.config, config)
        result = {
            "status": "configured",
            "config": str(args.config.resolve()),
            "model_requests_sent": 0,
            "base_snapshot_id": snapshot.manifest["snapshot_id"],
        }
    else:
        config = load_config(args.config, verify_build=args.command == "update")
        if args.command == "update":
            if args.crawl:
                if not 1 <= args.max_pages <= 10 or not 1.5 <= args.delay <= 60:
                    raise ValueError("INVALID_CRAWL_LIMITS")
                from offline.penalty_crawler.nfra import crawl_all

                with update_lock(Path(config["registry"])):
                    stats = crawl_all(
                        Path(config["crawl_dir"]),
                        max_pages=args.max_pages,
                        delay=args.delay,
                        columns=args.columns,
                    )
                write_json(Path(config["crawl_dir"]) / "last-crawl.json", {"columns": stats})
                print(json.dumps({"crawl": stats}, ensure_ascii=False), flush=True)
                if sum(s["pages"] for s in stats) == 0 and any(s["errors"] for s in stats):
                    raise ValueError("OFFICIAL_CRAWL_FAILED_SEE_ERRORS_JSONL")
            result = run_update(
                config,
                max_documents=args.max_documents,
                generate=args.generate,
                publish=args.publish,
                cap_cny=args.cap_cny,
                max_requests=args.max_requests,
                key_file=args.key_file,
                include_review=args.include_review,
            )
        elif args.command == "rollback":
            from agent_rag.knowledge_registry import rollback_snapshot

            with update_lock(Path(config["registry"])):
                result = rollback_snapshot(Path(config["registry"]), args.version)
        else:
            registry = Path(config["registry"])
            result = {
                "current": json.loads((registry / "current.json").read_text(encoding="utf-8"))
                if (registry / "current.json").exists()
                else None,
                "versions": sorted(p.name for p in (registry / "versions").glob("*") if p.is_dir()),
                "candidate_count": sum(1 for _ in (registry / "candidates").glob("*/state.json")),
                "model_requests_sent": 0,
            }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # 不输出transport错误体、环境变量、认证头或密钥路径内容。
        print(
            json.dumps(
                {"status": "failed", "code": str(error).split(":", 1)[0][:140]}, ensure_ascii=False
            ),
            file=sys.stderr,
        )
        sys.exit(1)
