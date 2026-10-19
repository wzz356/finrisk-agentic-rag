import importlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest


def module():
    try:
        return importlib.import_module("agent_rag.maintenance_store")
    except ModuleNotFoundError:
        pytest.fail("MaintenanceStore has not been implemented")


def make_store(root):
    return module().MaintenanceStore(root)


def event(store, key="event-1"):
    return store.create_event("evidence_added", ["e1"], ["r1"], "scope-1", key, "新证据")


def assert_code(code, call):
    with pytest.raises(ValueError) as caught:
        call()
    assert caught.value.code == code


def test_event_idempotency_survives_restart_and_rejects_different_content(tmp_path):
    store = make_store(tmp_path)
    first = event(store)
    assert event(make_store(tmp_path)) == first
    assert len(store.list_events()) == 1
    assert store.get_event(first["id"])["evidence_ids"] == ["e1"]
    assert_code(
        "IDEMPOTENCY_CONFLICT",
        lambda: store.create_event(
            "evidence_added", ["other"], ["r1"], "scope-1", "event-1", "新证据"
        ),
    )
    assert_code("EVENT_NOT_FOUND", lambda: store.get_event("missing"))


def test_concurrent_event_creation_yields_exactly_one_event(tmp_path):
    make_store(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(lambda _: event(make_store(tmp_path)), range(24)))
    assert len({value["id"] for value in values}) == 1
    assert len(make_store(tmp_path).list_events()) == 1


def test_changesets_reviews_receipts_policies_and_artifacts_are_persistent(tmp_path):
    store = make_store(tmp_path)
    item = event(store)
    first = store.save_changeset(item["id"], {"operations": [{"type": "add"}]})
    assert first["revision"] == 1
    assert store.save_changeset(item["id"], first["payload"], revision=1) == first
    second = store.save_changeset(item["id"], {"operations": []})
    assert second["revision"] == 2
    assert_code("IMMUTABLE_REVISION", lambda: store.save_changeset(item["id"], {}, revision=1))
    run = store.save_review_run(item["id"], 1, {"verdict": "reject"}, run_id="review-1")
    assert store.save_review_run(item["id"], 1, {"verdict": "reject"}, run_id="review-1") == run
    assert_code(
        "IMMUTABLE_REVIEW", lambda: store.save_review_run(item["id"], 1, {}, run_id="review-1")
    )
    receipt = store.save_receipt(item["id"], 2, {"applied": True, "record_ids": ["r1"]})
    assert store.save_receipt(item["id"], 2, receipt["payload"]) == receipt
    assert_code("IMMUTABLE_RECEIPT", lambda: store.save_receipt(item["id"], 2, {}))
    store.save_policy("p1", {"limit": 2})
    store.save_artifact("a1", {"sha256": "a" * 64})
    reopened = make_store(tmp_path)
    assert reopened.get_changeset(item["id"]) == second
    assert reopened.get_changeset(item["id"], 1) == first
    assert reopened.list_review_runs(item["id"]) == [run]
    assert reopened.get_receipt(item["id"], 2) == receipt
    assert reopened.get_policy("p1")["payload"] == {"limit": 2}
    assert reopened.get_artifact("a1")["payload"]["sha256"] == "a" * 64
    assert_code("IMMUTABLE_POLICY", lambda: store.save_policy("p1", {}))
    assert_code("IMMUTABLE_ARTIFACT", lambda: store.save_artifact("a1", {}))
    with sqlite3.connect(store.db_path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="IMMUTABLE_REVISION"):
            connection.execute(
                "UPDATE revisions SET payload = ? WHERE event_id = ?", ("{}", item["id"])
            )


def test_job_compare_and_swap_is_atomic_between_workers(tmp_path):
    store = make_store(tmp_path)
    item = event(store)
    job = store.create_job(item["id"], idempotency_key="job-key")
    assert store.create_job(item["id"], idempotency_key="job-key") == job
    lease = store.acquire_lease(job["id"], "worker", 60)

    def change(_):
        try:
            return make_store(tmp_path).transition_job(
                job["id"], "pending", "running", "worker", lease["fencing_token"]
            )
        except ValueError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(change, range(12)))
    assert sum(isinstance(value, dict) for value in values) == 1
    assert values.count("JOB_STATUS_CONFLICT") == 11
    assert make_store(tmp_path).get_job(job["id"])["status"] == "running"


def test_lease_takeover_fences_old_worker_and_all_event_writes(tmp_path):
    store = make_store(tmp_path)
    item = event(store)
    job = store.create_job(item["id"])
    lease = store.acquire_lease(job["id"], "old", 5, now=10)
    assert_code("LEASE_HELD", lambda: store.acquire_lease(job["id"], "new", 5, now=11))
    assert_code("LEASE_REQUIRED", lambda: store.save_changeset(item["id"], {}))
    new = make_store(tmp_path).acquire_lease(job["id"], "new", 5, now=16)
    assert new["fencing_token"] == lease["fencing_token"] + 1
    old_args = {
        "job_id": job["id"],
        "owner": "old",
        "fencing_token": lease["fencing_token"],
        "now": 17,
    }
    assert_code("STALE_LEASE", lambda: store.save_changeset(item["id"], {}, **old_args))
    assert_code(
        "STALE_LEASE",
        lambda: store.transition_job(
            job["id"], "pending", "running", "old", lease["fencing_token"], now=17
        ),
    )
    args = {"job_id": job["id"], "owner": "new", "fencing_token": new["fencing_token"], "now": 17}
    change = store.save_changeset(item["id"], {"approved": True}, **args)
    store.save_review_run(item["id"], change["revision"], {"verdict": "accept"}, **args)
    receipt = store.save_receipt(item["id"], change["revision"], {"applied": True}, **args)
    store.set_event_status(item["id"], "pending", "completed", **args)
    store.renew_lease(job["id"], "new", new["fencing_token"], 5, now=18)
    store.release_lease(job["id"], "new", new["fencing_token"], now=19)
    assert_code(
        "LEASE_EXPIRED",
        lambda: store.save_receipt(item["id"], 1, receipt["payload"], **{**args, "now": 20}),
    )
    assert make_store(tmp_path).get_receipt(item["id"], 1) == receipt
    assert make_store(tmp_path).list_jobs()[0]["id"] == job["id"]


def test_lease_token_increments_after_release_even_for_same_owner(tmp_path):
    store = make_store(tmp_path)
    job = store.create_job(event(store)["id"])
    lease = store.acquire_lease(job["id"], "worker", 10, now=0)
    store.release_lease(job["id"], "worker", lease["fencing_token"], now=1)
    assert store.acquire_lease(job["id"], "worker", 10, now=2)["fencing_token"] == 2


def test_cross_event_lease_cannot_write_another_event(tmp_path):
    store = make_store(tmp_path)
    one, two = event(store, "one"), event(store, "two")
    job = store.create_job(one["id"])
    lease = store.acquire_lease(job["id"], "worker", 10)
    assert_code(
        "LEASE_EVENT_MISMATCH",
        lambda: store.save_changeset(
            two["id"], {}, job_id=job["id"], owner="worker", fencing_token=lease["fencing_token"]
        ),
    )


def test_process_restart_retains_publication_receipt_and_lease_fence(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    code = """import json,os,sys
sys.path.insert(0,sys.argv[2])
from agent_rag.maintenance_store import MaintenanceStore
store=MaintenanceStore(sys.argv[1])
event=store.create_event('evidence_added',['e1'],['r1'],'scope-1','event-1','新证据')
job=store.create_job(event['id'])
lease=store.acquire_lease(job['id'],'dead-worker',5,now=10)
args={'job_id':job['id'],'owner':'dead-worker','fencing_token':lease['fencing_token'],'now':11}
store.save_changeset(event['id'],{'operations':[]},**args)
store.save_receipt(event['id'],1,{'applied':True},**args)
os._exit(0)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(tmp_path),
            str(Path(__file__).resolve().parents[2] / "src"),
        ],
        capture_output=True,
        text=True,
        check=False,
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr
    store = make_store(tmp_path)
    item = event(store)
    job = store.list_jobs()[0]
    receipt = store.get_receipt(item["id"], 1)
    lease = store.acquire_lease(job["id"], "new-worker", 10, now=20)
    assert lease["fencing_token"] == 2
    assert (
        store.save_receipt(
            item["id"],
            1,
            {"applied": True},
            job_id=job["id"],
            owner="new-worker",
            fencing_token=lease["fencing_token"],
            now=21,
        )
        == receipt
    )
    assert len(store.list_events()) == 1


def test_boolean_fencing_token_and_nonfinite_expiry_are_rejected(tmp_path):
    store = make_store(tmp_path)
    job = store.create_job(event(store)["id"])
    store.acquire_lease(job["id"], "worker", 10, now=1)
    assert_code(
        "INVALID_FENCING_TOKEN", lambda: store.assert_lease(job["id"], "worker", True, now=2)
    )
    assert_code(
        "INVALID_LEASE_TTL", lambda: store.acquire_lease(job["id"], "worker", 1e308, now=1e308)
    )
