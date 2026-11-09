import copy
import importlib
import importlib.util
import json
from pathlib import Path

import pytest
from test_maintenance_contracts import setup_case

from agent_rag import maintenance_contracts as contracts
from agent_rag.maintenance_tools import MaintenanceCorpus


def api():
    assert importlib.util.find_spec("agent_rag.maintenance_publish"), "缺少受控知识发布器"
    return importlib.import_module("agent_rag.maintenance_publish")


def publisher(tmp_path, **kwargs):
    return api().KnowledgePublisher(
        tmp_path / "managed", MaintenanceCorpus(tmp_path / "corpus"), **kwargs
    )


def approved(base, *, event="event1", kind="add", before=None, version=None, text=None):
    ch, review, runs, context, evidence = setup_case()
    ch["base_knowledge_commit"] = base
    ch["event"]["event_id"] = event
    ch["event"]["idempotency_key"] = contracts.canonical_sha256(event)
    op = ch["operations"][0]
    op.update(kind=kind, before_sha256=before, expected_record_version=version)
    if text is not None:
        op["after"]["summary"] = text
        ch["claims"][0]["statement"] = text
    if kind == "deactivate":
        op["after"]["active"] = False
    for run in runs.values():
        run["base_knowledge_commit"] = base
    ch["proposer_attestation_sha256"] = contracts.attestation_sha256(runs["P1"])
    ch["changeset_sha256"] = contracts.changeset_sha256(ch)
    runs["P1"]["output_sha256"] = ch["changeset_sha256"]
    context["expected_base"] = base
    receipt = contracts.issue_receipt(
        ch, review, reviewer_run_id="R1", key_id="host-test", key=b"x" * 32, **context
    )
    runs["R1"].update(
        receipt_id=receipt["receipt_id"], receipt_sha256=contracts.canonical_sha256(receipt)
    )
    context["key_resolver"] = {"host-test": b"x" * 32}
    return ch, receipt, context


def resign(ch, receipt, context):
    runs = context["trusted_runs"]
    ch["proposer_attestation_sha256"] = contracts.attestation_sha256(runs["P1"])
    ch["changeset_sha256"] = contracts.changeset_sha256(ch)
    runs["P1"]["output_sha256"] = ch["changeset_sha256"]
    review = {
        key: receipt[key]
        for key in ("decision", "claim_checks", "omission_check_completed", "issues")
    }
    ctx = {key: value for key, value in context.items() if key != "key_resolver"}
    receipt = contracts.issue_receipt(
        ch, review, reviewer_run_id="R1", key_id="host-test", key=b"x" * 32, **ctx
    )
    runs["R1"].update(
        receipt_id=receipt["receipt_id"], receipt_sha256=contracts.canonical_sha256(receipt)
    )
    return receipt


def test_actual_git_commit_is_approved_idempotent_and_recovers_corpus(tmp_path, monkeypatch):
    p = publisher(tmp_path)
    initial = p.head()
    assert len(initial) == 40
    ch, receipt, context = approved(initial)
    original = p._sync_corpus
    monkeypatch.setattr(
        p, "_sync_corpus", lambda commit: (_ for _ in ()).throw(OSError("interrupted after CAS"))
    )
    with pytest.raises(OSError):
        p.commit_changeset(ch, receipt, context)
    monkeypatch.setattr(p, "_sync_corpus", original)
    result = p.commit_changeset(ch, receipt, context)
    assert result["knowledge_commit"] != initial
    assert result["idempotent"] is True
    assert p.corpus.records(result["knowledge_commit"])[0]["body"] == ch["operations"][0]["after"]
    assert (
        p.commit_changeset(ch, receipt, context)["knowledge_commit"] == result["knowledge_commit"]
    )
    assert p.head() == result["knowledge_commit"]
    assert (p.root / "knowledge_git" / ".git").is_dir()


def test_missing_trust_stale_base_and_event_revision_conflict_cannot_commit(tmp_path):
    p = publisher(tmp_path)
    initial = p.head()
    ch, receipt, context = approved(initial)
    with pytest.raises(ValueError, match="APPROVAL_CONTEXT"):
        p.commit_changeset(ch, receipt)
    assert p.head() == initial
    p.commit_changeset(ch, receipt, context)
    stale, stale_r, stale_ctx = approved(initial, event="event2")
    with pytest.raises(ValueError, match="BASE_MISMATCH"):
        p.commit_changeset(stale, stale_r, stale_ctx)
    other, other_r, other_ctx = approved(p.head())
    with pytest.raises(ValueError, match="IDEMPOTENCY_CONFLICT"):
        p.commit_changeset(other, other_r, other_ctx)


def test_before_hash_version_and_scope_are_checked_with_old_versions_retained(tmp_path):
    p = publisher(tmp_path)
    ch, r, ctx = approved(p.head())
    first = p.commit_changeset(ch, r, ctx)["knowledge_commit"]
    record = p.corpus.records(first)[0]
    bad, r, ctx = approved(first, event="bad", kind="revise", before="f" * 64, version=1)
    with pytest.raises(ValueError, match="BEFORE_HASH"):
        p.commit_changeset(bad, r, ctx)
    bad, r, ctx = approved(
        first, event="bad-version", kind="revise", before=record["sha256"], version=2
    )
    with pytest.raises(ValueError, match="RECORD_VERSION"):
        p.commit_changeset(bad, r, ctx)
    revised, r, ctx = approved(
        first, event="revised", kind="revise", before=record["sha256"], version=1
    )
    second = p.commit_changeset(revised, r, ctx)["knowledge_commit"]
    inactive, r, ctx = approved(
        second, event="inactive", kind="deactivate", before=record["sha256"], version=2
    )
    third = p.commit_changeset(inactive, r, ctx)["knowledge_commit"]
    assert p.corpus.records(first)[0]["record_version"] == 1
    assert p.corpus.records(third)[0]["body"]["active"] is False
    assert p.corpus.records(third)[0]["record_version"] == 3


def builder(request, staging):
    assets = {}
    for name in api().REQUIRED_ASSETS:
        path = staging / (name + ".json")
        path.write_text(
            json.dumps({"asset": name, "records": request["records"]}, ensure_ascii=False),
            encoding="utf-8",
        )
        assets[name] = {"path": str(path), "sha256": contracts.canonical_sha256(None)}
        # Hash the real file; a declared digest never replaces byte verification.
        import hashlib

        assets[name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"assets": assets}


def validators():
    return {
        name: lambda path: bool(json.loads(path.read_text(encoding="utf-8"))["asset"])
        for name in api().REQUIRED_ASSETS
    }


def baseline(p, tmp_path):
    staging = tmp_path / "base-assets"
    staging.mkdir()
    manifest = {
        "schema": api().RELEASE_SCHEMA,
        "release_id": "baseline",
        "knowledge_commit": p.head(),
        "parent_release": None,
        "assets": builder({"records": []}, staging)["assets"],
        "asset_bindings": {name: "a" * 64 for name in api().REQUIRED_ASSETS},
        "approval": {"kind": "pinned_baseline"},
    }
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    import hashlib

    return p.bind_baseline(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_no_builder_or_incomplete_assets_stays_blocked_and_not_selectable(tmp_path):
    p = publisher(tmp_path)
    ch, r, ctx = approved(p.head())
    commit = p.commit_changeset(ch, r, ctx)["knowledge_commit"]
    result = p.build_release(commit)
    assert result["status"] == "blocked" and result["selectable"] is False
    with pytest.raises(ValueError, match="RELEASE_NOT_REGISTERED"):
        p.select_release(result["release_id"])
    p = publisher(
        tmp_path,
        approval_context=ctx,
        builders={"default": lambda request, path: {"assets": {}}},
        asset_validators=validators(),
    )
    assert p.build_release(commit)["status"] == "blocked"


def test_complete_release_atomic_selection_usage_parent_and_registered_rollback(tmp_path):
    p = publisher(tmp_path, builders={"default": builder}, asset_validators=validators())
    baseline(p, tmp_path)
    ch, r, ctx = approved(p.head())
    first = p.build_release(
        p.commit_changeset(ch, r, ctx)["knowledge_commit"], parent_release="baseline"
    )
    assert first["status"] == "built" and first["selectable"] is True
    p.select_release(first["release_id"])
    assert p.state()["enabled"] is False
    assert p.state()["effective_release"] == "baseline"
    p.set_usage(True)
    assert p.state()["effective_release"] == first["release_id"]
    second_ch, r, ctx = approved(
        p.head(),
        event="second",
        kind="revise",
        before=ch["operations"][0]["after"]
        and contracts.canonical_sha256(ch["operations"][0]["after"]),
        version=1,
    )
    second = p.build_release(
        p.commit_changeset(second_ch, r, ctx)["knowledge_commit"],
        parent_release=first["release_id"],
    )
    p.select_release(second["release_id"])
    with pytest.raises(ValueError, match="STALE_PARENT"):
        p.select_release(first["release_id"])
    p.rollback(first["release_id"])
    assert p.state()["effective_release"] == first["release_id"]
    p.set_usage(False)
    assert p.state()["effective_release"] == "baseline"
    assert (p.root / "releases" / second["release_id"] / "manifest.json").is_file()
    with pytest.raises(ValueError, match="RELEASE_NOT_REGISTERED"):
        p.rollback("arbitrary")


def test_tampered_release_asset_and_unapproved_git_commit_fail_closed(tmp_path):
    p = publisher(tmp_path, builders={"default": builder}, asset_validators=validators())
    with pytest.raises(ValueError, match="COMMIT_NOT_APPROVED"):
        p.build_release(p.head())
    baseline(p, tmp_path)
    ch, r, ctx = approved(p.head())
    built = p.build_release(
        p.commit_changeset(ch, r, ctx)["knowledge_commit"], parent_release="baseline"
    )
    manifest = json.loads(Path(built["manifest_path"]).read_text(encoding="utf-8"))
    Path(manifest["assets"]["dense"]["path"]).write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="ASSET_HASH"):
        p.select_release(built["release_id"])
    assert p.state()["selected_release"] is None


def test_old_generate_publish_rejected_before_models_or_builders(tmp_path):
    from agent_rag.knowledge_updates import run_update

    with pytest.raises(ValueError, match="APPROVAL_REQUIRED"):
        run_update(
            {"registry": str(tmp_path / "registry"), "crawl_dir": str(tmp_path / "crawl")},
            generate=True,
            publish=True,
            model_factory=lambda *a: pytest.fail("不得调用模型"),
        )


def test_managed_baseline_requires_host_sha_pin_and_extension_requires_admission(tmp_path):
    from test_knowledge_registry import version_snapshot

    from agent_rag import knowledge_registry as kr

    base = version_snapshot(tmp_path / "original", "base")
    registry = tmp_path / "registry"
    with pytest.raises(ValueError, match="BASELINE_HOST_PIN_REQUIRED"):
        kr.initialize_usage(registry, base)
    from agent_rag.v2_snapshot import file_hash

    kr.initialize_usage(registry, base, expected_manifest_sha256=file_hash(base / "manifest.json"))
    forged = version_snapshot(registry, "forged")
    with pytest.raises(ValueError, match="HOST_ADMISSION_REQUIRED"):
        kr.publish_snapshot(registry, forged)
    from test_collection_selection import completed

    completed(registry, "complete-but-unreviewed", "snapshot-base")
    with pytest.raises(ValueError, match="HOST_ADMISSION_REQUIRED"):
        kr.select_version(registry, "complete-but-unreviewed")


def test_allowlisted_builder_rejects_wrong_hash_empty_asset_and_outside_paths(tmp_path):
    def damaged(request, staging):
        output = builder(request, staging)
        asset = output["assets"]["dense"]
        Path(asset["path"]).write_text("", encoding="utf-8")
        return output

    p = publisher(tmp_path, builders={"default": damaged}, asset_validators=validators())
    ch, r, ctx = approved(p.head())
    commit = p.commit_changeset(ch, r, ctx)["knowledge_commit"]
    assert p.build_release(commit)["status"] == "blocked"
    assert not p.state()["registered"]
    outside = tmp_path / "outside.json"
    outside.write_text('{"asset":"dense"}', encoding="utf-8")

    def outside_builder(request, staging):
        output = builder(request, staging)
        from agent_rag.v2_snapshot import file_hash

        output["assets"]["dense"] = {"path": str(outside), "sha256": file_hash(outside)}
        return output

    p.builders["default"] = outside_builder
    assert p.build_release(commit)["code"] == "RELEASE_ASSET_PATH_OR_STALE_REUSE"


def test_selection_failure_keeps_usage_and_selected_in_one_atomic_pointer(tmp_path, monkeypatch):
    p = publisher(tmp_path, builders={"default": builder}, asset_validators=validators())
    baseline(p, tmp_path)
    ch, r, ctx = approved(p.head())
    built = p.build_release(
        p.commit_changeset(ch, r, ctx)["knowledge_commit"], parent_release="baseline"
    )
    before = (p.root / "release-state.json").read_bytes()
    original = api().write_json

    def interrupted(path, value):
        if path.name == "release-state.json":
            raise OSError("interrupted")
        return original(path, value)

    monkeypatch.setattr(api(), "write_json", interrupted)
    with pytest.raises(OSError):
        p.select_release(built["release_id"])
    assert (p.root / "release-state.json").read_bytes() == before


def test_cli_cannot_publish_even_before_config_or_crawl(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/update_agent_knowledge.py"
    spec = importlib.util.spec_from_file_location("blocked_publish_cli", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    with pytest.raises(ValueError, match="MAINTENANCE_APPROVAL_REQUIRED"):
        cli.main(
            [
                "update",
                "--config",
                str(tmp_path / "missing.json"),
                "--generate",
                "--publish",
                "--crawl",
            ]
        )


def test_managed_enable_cannot_bypass_admission_with_edited_usage_state(tmp_path):
    from test_knowledge_registry import version_snapshot, write_json

    from agent_rag import knowledge_registry as kr
    from agent_rag.v2_snapshot import file_hash

    base = version_snapshot(tmp_path / "original", "base")
    registry = tmp_path / "registry"
    kr.initialize_usage(registry, base, expected_manifest_sha256=file_hash(base / "manifest.json"))
    forged = version_snapshot(registry, "forged")
    state = kr.usage_state(registry)
    state.update(
        selected_version="forged", selected_manifest_sha256=file_hash(forged / "manifest.json")
    )
    write_json(registry / "usage.json", state)
    with pytest.raises(ValueError, match="HOST_ADMISSION_REQUIRED"):
        kr.set_usage(registry, True)


def test_host_admission_binds_complete_approved_release_to_actual_v2_snapshot(tmp_path):
    import shutil

    from test_knowledge_registry import version_snapshot, write_json

    from agent_rag import knowledge_registry as kr
    from agent_rag.v2_snapshot import file_hash

    def v2_builder(request, staging):
        output = builder(request, staging)
        path = version_snapshot(staging / "v2", "candidate")
        body = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        body.update(
            knowledge_commit=request["knowledge_commit"],
            release_id=request["release_id"],
            parent_snapshot_id="snapshot-base",
        )
        write_json(path / "manifest.json", body)
        output["v2_snapshot"] = {
            "path": str(path),
            "manifest_sha256": file_hash(path / "manifest.json"),
        }
        return output

    p = publisher(tmp_path, builders={"default": v2_builder}, asset_validators=validators())
    baseline(p, tmp_path)
    ch, r, ctx = approved(p.head())
    built = p.build_release(
        p.commit_changeset(ch, r, ctx)["knowledge_commit"], parent_release="baseline"
    )
    manifest = json.loads(Path(built["manifest_path"]).read_text(encoding="utf-8"))
    registry = tmp_path / "legacy-registry"
    base = version_snapshot(tmp_path / "original-v2", "base")
    kr.initialize_usage(registry, base, expected_manifest_sha256=file_hash(base / "manifest.json"))
    target = registry / "versions/approved/snapshot"
    target.mkdir(parents=True)
    shutil.copyfile(
        Path(manifest["v2_snapshot"]["path"]) / "manifest.json", target / "manifest.json"
    )
    write_json(
        target.parent / "update-complete.json",
        {
            "status": "built",
            "version": "approved",
            "snapshot_id": "snapshot-candidate",
            "snapshot_path": str(target),
        },
    )
    authority = api().HostAdmissionAuthority(p)
    admission = authority.issue(registry, target, built["release_id"])
    assert admission["authentication"]["algorithm"] == "HMAC-SHA256"
    forged = copy.deepcopy(admission)
    forged["manifest_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="HOST_ADMISSION_REQUIRED"):
        kr.select_version(registry, "approved", authority=authority, admission=forged)
    kr.select_version(registry, "approved", authority=authority, admission=admission)
    assert kr.usage_state(registry)["enabled"] is False
    kr.set_usage(registry, True, authority=authority, admission=admission)
    assert kr.retrieval_snapshot(registry) == target


def test_approved_original_bytes_and_metadata_are_preserved_by_hash(tmp_path):
    p = publisher(tmp_path)
    ch, receipt, ctx = approved(p.head())
    source = ctx["evidence_resolver"]("E1")
    result = p.commit_changeset(ch, receipt, ctx)
    assert (p.root / "originals" / source["raw_sha256"]).read_bytes() == source["raw_bytes"]
    assert (p.root / "texts" / source["text_sha256"]).read_text(encoding="utf-8") == source["text"]
    assert (
        p._read(result["knowledge_commit"], "evidence.json")["E1"]["extractor_version"]
        == source["extractor_version"]
    )


def test_relation_cannot_target_unknown_or_cross_scope_records(tmp_path):
    p = publisher(tmp_path)
    ch, receipt, ctx = approved(p.head())
    p.commit_changeset(ch, receipt, ctx)
    relation, receipt, ctx = approved(p.head(), event="relation", kind="relate")
    relation["operations"][0].update(record_kind="relation", target_id="R1")
    relation["operations"][0]["after"].update(
        relation_type="supports", source_id="B1", target_id="missing"
    )
    relation["affected_record_ids"] = ["R1"]
    receipt = resign(relation, receipt, ctx)
    with pytest.raises(ValueError, match="RELATION_TARGET"):
        p.commit_changeset(relation, receipt, ctx)
    relation["operations"][0]["after"]["target_id"] = "B1"
    receipt = resign(relation, receipt, ctx)
    result = p.commit_changeset(relation, receipt, ctx)
    assert any(row["kind"] == "relation" for row in p.corpus.records(result["knowledge_commit"]))


def test_release_cannot_forward_select_older_knowledge_by_declaring_new_parent(tmp_path):
    p = publisher(tmp_path, builders={"default": builder}, asset_validators=validators())
    baseline(p, tmp_path)
    ch, r, ctx = approved(p.head())
    commit1 = p.commit_changeset(ch, r, ctx)["knowledge_commit"]
    first = p.build_release(commit1, parent_release="baseline")
    p.select_release(first["release_id"])
    ch, r, ctx = approved(
        p.head(),
        event="revise",
        kind="revise",
        before=contracts.canonical_sha256(ch["operations"][0]["after"]),
        version=1,
    )
    commit2 = p.commit_changeset(ch, r, ctx)["knowledge_commit"]
    second = p.build_release(commit2, parent_release=first["release_id"])
    p.select_release(second["release_id"])
    with pytest.raises(ValueError, match="STALE_KNOWLEDGE_PARENT"):
        p.build_release(commit1, parent_release=second["release_id"])


def test_fixture_approval_cannot_commit_through_publisher(tmp_path):
    p = publisher(tmp_path)
    initial = p.head()
    ch, r, ctx = approved(initial)
    ctx["trusted_runs"]["R1"]["execution_mode"] = "fixture"
    with pytest.raises(ValueError, match="FIXTURE_FORBIDDEN"):
        p.commit_changeset(ch, r, ctx)
    assert p.head() == initial and p.corpus.records(initial) == []


def test_unchanged_assets_reuse_verified_parent_and_changed_dense_cannot_reuse(tmp_path):
    def incremental(request, staging):
        output = builder(request, staging)
        if request["parent"]:
            output["assets"] = {
                name: asset
                for name, asset in output["assets"].items()
                if request["asset_bindings"][name] != request["parent"]["asset_bindings"][name]
            }
        return output

    p = publisher(tmp_path, builders={"default": incremental}, asset_validators=validators())
    baseline(p, tmp_path)
    ch, r, ctx = approved(p.head())
    first = p.build_release(
        p.commit_changeset(ch, r, ctx)["knowledge_commit"], parent_release="baseline"
    )
    one = json.loads(Path(first["manifest_path"]).read_text(encoding="utf-8"))
    ch, r, ctx = approved(
        p.head(),
        event="revise",
        kind="revise",
        before=contracts.canonical_sha256(ch["operations"][0]["after"]),
        version=1,
    )
    second = p.build_release(
        p.commit_changeset(ch, r, ctx)["knowledge_commit"], parent_release=first["release_id"]
    )
    two = json.loads(Path(second["manifest_path"]).read_text(encoding="utf-8"))
    assert two["assets"]["model"] == one["assets"]["model"]
    assert two["assets"]["law_scope"] == one["assets"]["law_scope"]
    assert two["assets"]["dense"] != one["assets"]["dense"]

    def stale(request, staging):
        output = builder(request, staging)
        output["assets"]["dense"] = request["parent"]["assets"]["dense"]
        return output

    p.builders["default"] = stale
    ch, r, ctx = approved(
        p.head(),
        event="revise2",
        kind="revise",
        before=contracts.canonical_sha256(ch["operations"][0]["after"]),
        version=2,
    )
    result = p.build_release(
        p.commit_changeset(ch, r, ctx)["knowledge_commit"], parent_release=second["release_id"]
    )
    assert result["status"] == "blocked" and result["code"] == "RELEASE_ASSET_PATH_OR_STALE_REUSE"


def test_seed_pinned_legacy_records_once_without_promoting_candidate_to_approved(tmp_path):
    from test_knowledge_registry import version_snapshot

    from agent_rag.v2_snapshot import file_hash

    p = publisher(tmp_path)
    assert hasattr(p, "seed_legacy_records"), "旧库尚未注册到受控知识分支"
    baseline_path = version_snapshot(tmp_path / "original", "base") / "manifest.json"
    evidence = p.corpus.register_evidence("旧库证据".encode(), origin_id="legacy-baseline")
    record = {
        "record_id": "OLD-B1",
        "kind": "behavior",
        "body": {"summary": "旧库候选"},
        "record_version": 1,
        "scope_id": "public",
        "classification": "public",
        "evidence_ids": [evidence["evidence_id"]],
    }
    with pytest.raises(ValueError, match="BASELINE_HOST_PIN"):
        p.seed_legacy_records([record], base_manifest_path=baseline_path, expected_sha256="f" * 64)
    seeded = p.seed_legacy_records(
        [record], base_manifest_path=baseline_path, expected_sha256=file_hash(baseline_path)
    )
    assert p.head() == seeded["knowledge_commit"]
    assert p.records()[0]["admission_status"] == "legacy_candidate"
    assert p.records()[0]["baseline_manifest_sha256"] == file_hash(baseline_path)
    assert p.corpus.records(p.head())[0]["body"] == record["body"]
    assert evidence["evidence_id"] in p._read(p.head(), "evidence.json")
    assert p.seed_legacy_records(
        [record], base_manifest_path=baseline_path, expected_sha256=file_hash(baseline_path)
    )["idempotent"]
    with pytest.raises(ValueError, match="COMMIT_NOT_APPROVED"):
        p.build_release(p.head())
    ch, r, ctx = approved(p.head())
    p.commit_changeset(ch, r, ctx)
    with pytest.raises(ValueError, match="LEGACY_SEED_ONLY_EMPTY_HEAD"):
        p.seed_legacy_records(
            [record], base_manifest_path=baseline_path, expected_sha256=file_hash(baseline_path)
        )


def test_seed_rejects_replacement_baseline_and_changed_legacy_record(tmp_path):
    from test_knowledge_registry import version_snapshot

    from agent_rag.v2_snapshot import file_hash

    p = publisher(tmp_path)
    assert hasattr(p, "seed_legacy_records"), "旧库尚未注册到受控知识分支"
    path = version_snapshot(tmp_path / "original", "base") / "manifest.json"
    record = {"record_id": "OLD-B1", "kind": "behavior", "body": {"summary": "旧库候选"}}
    p.seed_legacy_records([record], base_manifest_path=path, expected_sha256=file_hash(path))
    record["body"]["summary"] = "替换"
    with pytest.raises(ValueError, match="LEGACY_SEED_IMMUTABLE"):
        p.seed_legacy_records([record], base_manifest_path=path, expected_sha256=file_hash(path))


def test_restart_registers_complete_staging_once_after_pointer_write_failure(tmp_path, monkeypatch):
    calls = []

    def counted(request, staging):
        calls.append(request["release_id"])
        return builder(request, staging)

    p = publisher(tmp_path, builders={"default": counted}, asset_validators=validators())
    baseline(p, tmp_path)
    ch, r, ctx = approved(p.head())
    commit = p.commit_changeset(ch, r, ctx)["knowledge_commit"]
    original = api().write_json

    def interrupted(path, value):
        if path.name == "release-state.json":
            raise OSError("interrupted after complete staging")
        return original(path, value)

    monkeypatch.setattr(api(), "write_json", interrupted)
    assert p.build_release(commit, parent_release="baseline")["status"] == "blocked"
    monkeypatch.setattr(api(), "write_json", original)
    p = publisher(
        tmp_path, approval_context=ctx, builders={"default": counted}, asset_validators=validators()
    )
    result = p.build_release(commit, parent_release="baseline")
    assert result["status"] == "built" and len(calls) == 1
    assert p.head() == commit


@pytest.mark.parametrize("reparent", [True, False])
def test_copied_approval_cannot_admit_an_unrelated_git_tree(tmp_path, reparent):
    p = publisher(tmp_path, builders={"default": builder}, asset_validators=validators())
    ch, r, ctx = approved(p.head())
    commit = p.commit_changeset(ch, r, ctx)["knowledge_commit"]
    records = p._read(commit, "records.json")
    records["B1"]["body"]["summary"] = "未批准的树内容"
    events = p._read(commit, "events.json")
    parent = commit if reparent else ch["base_knowledge_commit"]
    next(iter(events.values()))["base_commit"] = parent
    tree = p._tree(
        {
            "records.json": records,
            "events.json": events,
            "approvals.json": p._read(commit, "approvals.json"),
            "evidence.json": p._read(commit, "evidence.json"),
            "baseline.json": p._read(commit, "baseline.json"),
        }
    )
    unrelated = p._git("commit-tree", tree, "-p", parent, data=b"Unrelated local tree\n")
    p._git("update-ref", "refs/heads/knowledge", unrelated, commit)
    p._contexts[unrelated] = ctx
    with pytest.raises(ValueError, match="COMMIT_APPROVAL_BINDING"):
        p.build_release(unrelated)


def test_law_revision_invalidates_derived_guides_and_relations(tmp_path):
    p = publisher(tmp_path)
    ch, _, _ = approved(p.head())
    records = {
        "LAW1": {"kind": "law_article", "body": {"text": "旧法规"}},
        "GUIDE1": {"kind": "guide", "body": {"text": "指引", "depends_on": ["LAW1"]}},
    }
    for record in records.values():
        record.update(scope_id="public", classification="public")
    before = p._asset_bindings(records, ch["bindings"], {})
    records["LAW1"]["body"]["text"] = "新法规"
    after = p._asset_bindings(records, ch["bindings"], {})
    assert before["guides_relations"] != after["guides_relations"]


def test_old_enabled_unreviewed_managed_pointer_cannot_bypass_new_read_gate(tmp_path):
    from test_knowledge_registry import version_snapshot, write_json

    from agent_rag import knowledge_registry as kr
    from agent_rag.v2_snapshot import file_hash

    registry = tmp_path / "registry"
    base = version_snapshot(tmp_path / "original", "base")
    kr.initialize_usage(registry, base, expected_manifest_sha256=file_hash(base / "manifest.json"))
    forged = version_snapshot(registry, "old-unreviewed")
    state = kr.usage_state(registry)
    state.update(
        enabled=True,
        selected_version="old-unreviewed",
        selected_manifest_sha256=file_hash(forged / "manifest.json"),
    )
    write_json(registry / "usage.json", state)
    with pytest.raises(ValueError, match="HOST_ADMISSION_REQUIRED"):
        kr.retrieval_snapshot(registry)
    rebound = kr.initialize_usage(
        registry, base, expected_manifest_sha256=file_hash(base / "manifest.json")
    )
    assert rebound["enabled"] is False and rebound["selected_version"] is None
    assert kr.retrieval_snapshot(registry).parent.name == rebound["base_version"]
    assert forged.is_dir()


def test_managed_current_pointer_without_verified_registration_fails_closed(tmp_path):
    from test_knowledge_registry import version_snapshot, write_json

    from agent_rag import knowledge_registry as kr

    registry = tmp_path / "registry"
    forged = version_snapshot(registry, "unreviewed")
    kr.publish_snapshot(registry, forged)  # Historical mechanical registry.
    write_json(
        registry / "host-baseline.json", {"version": "baseline", "manifest_sha256": "f" * 64}
    )
    with pytest.raises(ValueError, match="HOST_ADMISSION_REQUIRED"):
        kr.current_snapshot(registry)


def test_disabling_usage_recovers_baseline_when_selected_manifest_is_corrupt(tmp_path):
    from test_knowledge_registry import version_snapshot, write_json

    from agent_rag import knowledge_registry as kr
    from agent_rag.v2_snapshot import file_hash

    registry = tmp_path / "registry"
    base = version_snapshot(tmp_path / "original", "base")
    kr.initialize_usage(registry, base, expected_manifest_sha256=file_hash(base / "manifest.json"))
    selected = version_snapshot(registry, "corrupt")
    state = kr.usage_state(registry)
    state.update(
        enabled=True,
        selected_version="corrupt",
        selected_manifest_sha256=file_hash(selected / "manifest.json"),
    )
    write_json(registry / "usage.json", state)
    (selected / "manifest.json").write_text("corrupt", encoding="utf-8")
    kr.set_usage(registry, False)
    assert kr.retrieval_snapshot(registry).parent.name == state["base_version"]


def test_preserved_original_uses_short_same_directory_temporary_name(tmp_path, monkeypatch):
    import hashlib

    deep = tmp_path / ("r" * max(1, 165 - len(str(tmp_path)) - 1))
    p = publisher(deep)
    original = api().os.replace
    observed = []

    def replace(source, destination):
        observed.append(Path(source))
        assert len(Path(source).name) <= 40
        assert Path(source).parent == Path(destination).parent
        original(source, destination)

    monkeypatch.setattr(api().os, "replace", replace)
    digest = hashlib.sha256(b"original").hexdigest()
    p._preserve_blob("originals", digest, b"original")
    assert observed and (p.root / "originals" / digest).read_bytes() == b"original"


def unapproved_intermediate(p, parent):
    files = {
        name: p._read(parent, name)
        for name in (
            "records.json",
            "events.json",
            "approvals.json",
            "evidence.json",
            "baseline.json",
        )
    }
    files["records.json"]["UNREVIEWED"] = copy.deepcopy(files["records.json"]["B1"])
    files["records.json"]["UNREVIEWED"]["body"]["summary"] = "not covered by any approved operation"
    forged = p._git(
        "commit-tree", p._tree(files), "-p", parent, data=b"Unapproved intermediate tree\n"
    )
    p._git("update-ref", "refs/heads/knowledge", forged, parent)
    return forged, files


def test_unapproved_ancestor_blocks_following_approved_commit(tmp_path):
    p = publisher(tmp_path)
    ch, r, ctx = approved(p.head())
    c1 = p.commit_changeset(ch, r, ctx)["knowledge_commit"]
    forged, _ = unapproved_intermediate(p, c1)
    ch2, r2, ctx2 = approved(
        forged,
        event="next",
        kind="revise",
        before=contracts.canonical_sha256(ch["operations"][0]["after"]),
        version=1,
    )
    with pytest.raises(ValueError, match="COMMIT_NOT_APPROVED|UNAPPROVED_KNOWLEDGE_ANCESTOR"):
        p.commit_changeset(ch2, r2, ctx2)
    assert p.head() == forged


def test_unapproved_ancestor_blocks_preexisting_locally_valid_approved_tip(tmp_path):
    p = publisher(tmp_path, builders={"default": builder}, asset_validators=validators())
    ch, r, ctx = approved(p.head())
    c1 = p.commit_changeset(ch, r, ctx)["knowledge_commit"]
    forged, files = unapproved_intermediate(p, c1)
    ch2, r2, ctx2 = approved(
        forged,
        event="next",
        kind="revise",
        before=contracts.canonical_sha256(ch["operations"][0]["after"]),
        version=1,
    )
    record = files["records.json"]["B1"]
    record.update(record_version=2, body=ch2["operations"][0]["after"])
    key = contracts.canonical_sha256([ch2["event"]["event_id"], ch2["revision"]])
    files["events.json"][key] = {
        "event_id": ch2["event"]["event_id"],
        "revision": ch2["revision"],
        "changeset_sha256": ch2["changeset_sha256"],
        "receipt_sha256": contracts.canonical_sha256(r2),
        "base_commit": forged,
    }
    files["approvals.json"][key] = {"changeset": ch2, "receipt": r2}
    c2 = p._git("commit-tree", p._tree(files), "-p", forged, data=b"Locally valid approved tip\n")
    p._git("update-ref", "refs/heads/knowledge", c2, forged)
    p._contexts[c2] = ctx2
    with pytest.raises(ValueError, match="COMMIT_NOT_APPROVED|UNAPPROVED_KNOWLEDGE_ANCESTOR"):
        p.build_release(c2)


def test_existing_seed_requires_explicit_host_pin_migration_before_new_approval(tmp_path):
    from test_knowledge_registry import version_snapshot

    from agent_rag.v2_snapshot import file_hash

    p = publisher(tmp_path)
    path = version_snapshot(tmp_path / "original", "base") / "manifest.json"
    records = [{"record_id": "OLD-B1", "kind": "behavior", "body": {"summary": "old candidate"}}]
    seeded = p.seed_legacy_records(
        records, base_manifest_path=path, expected_sha256=file_hash(path)
    )
    assert hasattr(p, "pin_legacy_seed"), "缺少显式host seed迁移"
    (p.root / "knowledge-anchors.json").unlink()
    p = publisher(tmp_path)
    ch, r, ctx = approved(p.head())
    with pytest.raises(ValueError, match="COMMIT_NOT_APPROVED|UNAPPROVED_KNOWLEDGE_ANCESTOR"):
        p.commit_changeset(ch, r, ctx)
    with pytest.raises(ValueError, match="BASELINE_HOST_PIN"):
        p.pin_legacy_seed(
            seeded["knowledge_commit"], base_manifest_path=path, expected_sha256="f" * 64
        )
    p.pin_legacy_seed(
        seeded["knowledge_commit"], base_manifest_path=path, expected_sha256=file_hash(path)
    )
    result = p.commit_changeset(ch, r, ctx)
    assert {row["record_id"] for row in p.corpus.records(result["knowledge_commit"])} == {
        "OLD-B1",
        "B1",
    }


def test_lineage_structure_cache_still_rechecks_mutable_host_approval(tmp_path, monkeypatch):
    p = publisher(tmp_path)
    ch, r, ctx = approved(p.head())
    commit = p.commit_changeset(ch, r, ctx)["knowledge_commit"]
    calls = []
    original = p._git

    def counted(*args, **kwargs):
        if args[0] == "show":
            calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(p, "_git", counted)
    assert p.build_release(commit)["status"] == "blocked"
    first = len(calls)
    assert first > 0
    assert p.build_release(commit)["status"] == "blocked"
    assert len(calls) == first
    ctx["trusted_runs"]["R1"]["execution_mode"] = "fixture"
    with pytest.raises(ValueError, match="FIXTURE_FORBIDDEN"):
        p.build_release(commit)


def test_seed_replay_cannot_anchor_noninitial_history_with_matching_payload(tmp_path):
    from test_knowledge_registry import version_snapshot

    from agent_rag.v2_snapshot import file_hash

    p = publisher(tmp_path)
    path = version_snapshot(tmp_path / "original", "base") / "manifest.json"
    records = [{"record_id": "OLD-B1", "kind": "behavior", "body": {"summary": "old candidate"}}]
    seeded = p.seed_legacy_records(
        records, base_manifest_path=path, expected_sha256=file_hash(path)
    )
    seed = seeded["knowledge_commit"]
    files = {
        name: p._read(seed, name)
        for name in (
            "records.json",
            "events.json",
            "approvals.json",
            "evidence.json",
            "baseline.json",
        )
    }
    unrelated = p._git(
        "commit-tree", p._tree(files), "-p", seed, data=b"Unapproved repeated seed payload\n"
    )
    p._git("update-ref", "refs/heads/knowledge", unrelated, seed)
    with pytest.raises(ValueError, match="LEGACY_SEED_ANCHOR_INVALID"):
        p.seed_legacy_records(records, base_manifest_path=path, expected_sha256=file_hash(path))
    assert unrelated not in p._anchors()["legacy_seeds"]
