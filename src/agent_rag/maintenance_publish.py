"""Host-only approved knowledge Git and complete, atomically selected releases.

Builders and content validators are trusted Python callables supplied by the host.
No model supplied command, executable, path, or approval declaration is executed.
Without those adapters the publisher records a blocked build, never a serving
release. The business source checkout is never used as the knowledge repository.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
import re
import secrets
import subprocess
from pathlib import Path
from uuid import uuid4

from agent_rag.knowledge_updates import update_lock, write_json
from agent_rag.maintenance_contracts import canonical_bytes, canonical_sha256, require_approved
from agent_rag.v2_snapshot import file_hash

RELEASE_SCHEMA = "finrisk-unified-release/1.0"
REQUIRED_ASSETS = frozenset(
    {
        "cases",
        "law_scope",
        "guides_relations",
        "bm25",
        "dense",
        "model",
        "rowmap",
        "active",
        "reader",
        "dictionary",
        "query",
        "acl",
        "raw_catalog",
    }
)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def _need(ok, code):
    if not ok:
        raise ValueError(code)


def validate_release_manifest(value, *, validators=None, require_validators=False):
    """Rehash every asset; content admission additionally needs host validators."""
    _need(isinstance(value, dict) and value.get("schema") == RELEASE_SCHEMA, "RELEASE_SCHEMA")
    _need(
        isinstance(value.get("release_id"), str) and _ID.fullmatch(value["release_id"]),
        "RELEASE_ID",
    )
    _need(
        isinstance(value.get("knowledge_commit"), str)
        and _COMMIT.fullmatch(value["knowledge_commit"]),
        "RELEASE_COMMIT",
    )
    assets = value.get("assets")
    _need(
        isinstance(assets, dict) and REQUIRED_ASSETS <= assets.keys(), "INCOMPLETE_RELEASE_ASSETS"
    )
    bindings = value.get("asset_bindings")
    _need(
        isinstance(bindings, dict) and REQUIRED_ASSETS <= bindings.keys(),
        "INCOMPLETE_RELEASE_BINDINGS",
    )
    for name, asset in assets.items():
        _need(
            isinstance(asset, dict) and isinstance(asset.get("path"), str), "INVALID_RELEASE_ASSET"
        )
        path = Path(asset["path"])
        _need(
            path.is_absolute() and path.is_file() and path.stat().st_size > 0,
            "RELEASE_ASSET_MISSING_OR_EMPTY",
        )
        _need(file_hash(path) == asset.get("sha256"), "RELEASE_ASSET_HASH_MISMATCH")
        check = (validators or {}).get(name)
        _need(not require_validators or callable(check), "ASSET_VALIDATOR_REQUIRED")
        if callable(check):
            _need(check(path) is True, "RELEASE_ASSET_CONTENT_INVALID")
    return copy.deepcopy(value)


class KnowledgePublisher:
    def __init__(
        self, root, corpus, *, approval_context=None, builders=None, asset_validators=None
    ):
        self.root, self.corpus = Path(root).resolve(), corpus
        self.root.mkdir(parents=True, exist_ok=True)
        self.repo = self.root / "knowledge_git"
        self._context = approval_context
        self._contexts = {}
        self._synced = set()
        # Git commit IDs bind their complete tree and parents. Cache only that
        # immutable structural proof; host approval and source bytes are rechecked.
        self._structures = {}
        self._empty_roots = set()
        self.builders, self.validators = dict(builders or {}), dict(asset_validators or {})
        _need(
            all(
                isinstance(k, str) and _ID.fullmatch(k) and callable(v)
                for k, v in self.builders.items()
            ),
            "INVALID_HOST_BUILDERS",
        )
        with update_lock(self.root / "publisher-lock"):
            self._initialize()
            self._sync_corpus(self._head())

    def _git(self, *args, data=None):
        # Clear inherited repository/config injection. No network command is used.
        env = {
            key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")
        }
        env.update(
            GIT_CONFIG_NOSYSTEM="1",
            GIT_CONFIG_GLOBAL=os.devnull,
            GIT_AUTHOR_NAME="FinRisk Knowledge Host",
            GIT_AUTHOR_EMAIL="knowledge@localhost",
            GIT_COMMITTER_NAME="FinRisk Knowledge Host",
            GIT_COMMITTER_EMAIL="knowledge@localhost",
        )
        result = subprocess.run(
            [
                "git",
                "-c",
                "core.hooksPath=" + str(self.root / "empty-hooks"),
                "-c",
                "core.fsmonitor=false",
                "-c",
                "commit.gpgsign=false",
                "-C",
                str(self.repo),
                *args,
            ],
            input=data,
            capture_output=True,
            env=env,
            check=False,
        )
        _need(result.returncode == 0, "KNOWLEDGE_GIT_FAILED")
        return result.stdout.decode("utf-8").strip()

    def _initialize(self):
        owner = self.root / "knowledge-git-owner.json"
        expected = {"schema": "finrisk-knowledge-git/1.0", "corpus": str(self.corpus.root)}
        _need(
            self.repo.resolve().parent == self.root and not self.repo.is_symlink(),
            "KNOWLEDGE_GIT_PATH",
        )
        if self.repo.exists() and any(self.repo.iterdir()):
            _need(
                owner.is_file()
                and json.loads(owner.read_text(encoding="utf-8")) == expected
                and (self.repo / ".git").is_dir()
                and not (self.repo / ".git").is_symlink(),
                "FOREIGN_KNOWLEDGE_REPOSITORY",
            )
        else:
            self.repo.mkdir(exist_ok=True)
            write_json(owner, expected)
            self._git("init", "--quiet")
            self._git("symbolic-ref", "HEAD", "refs/heads/knowledge")
        refs = self._git("for-each-ref", "--format=%(objectname)", "refs/heads/knowledge")
        if not refs:
            tree = self._tree(
                {
                    "records.json": {},
                    "events.json": {},
                    "approvals.json": {},
                    "evidence.json": {},
                    "baseline.json": None,
                }
            )
            commit = self._git("commit-tree", tree, data=b"Initialize controlled knowledge\n")
            self._git("update-ref", "refs/heads/knowledge", commit, "0" * 40)

    def _head(self):
        commit = self._git("rev-parse", "refs/heads/knowledge")
        _need(_COMMIT.fullmatch(commit), "INVALID_KNOWLEDGE_HEAD")
        return commit

    def head(self):
        with update_lock(self.root / "publisher-lock"):
            commit = self._head()
            self._sync_corpus(commit)
            return commit

    def _read(self, commit, name):
        _need(isinstance(commit, str) and _COMMIT.fullmatch(commit), "INVALID_KNOWLEDGE_COMMIT")
        return json.loads(self._git("show", commit + ":" + name))

    def _tree(self, files):
        entries = []
        for name, value in sorted(files.items()):
            blob = self._git("hash-object", "-w", "--stdin", data=canonical_bytes(value))
            entries.append("100644 blob " + blob + "\t" + name + "\n")
        return self._git("mktree", data="".join(entries).encode())

    def _anchors(self):
        path = self.root / "knowledge-anchors.json"
        value = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else {"schema": "finrisk-knowledge-anchors/1.0", "legacy_seeds": {}}
        )
        _need(
            value.get("schema") == "finrisk-knowledge-anchors/1.0"
            and isinstance(value.get("legacy_seeds"), dict),
            "INVALID_KNOWLEDGE_ANCHORS",
        )
        return value

    def _record_seed_anchor(self, commit, metadata, path, expected_sha256):
        anchors = self._anchors()
        entry = {
            "seed_commit": commit,
            "base_manifest_path": str(path),
            "manifest_sha256": expected_sha256,
            "seed_metadata": copy.deepcopy(metadata),
        }
        previous = anchors["legacy_seeds"].get(commit)
        _need(previous is None or previous == entry, "LEGACY_SEED_IMMUTABLE")
        if previous is None:
            anchors["legacy_seeds"][commit] = entry
            write_json(self.root / "knowledge-anchors.json", anchors)

    def _empty_initial(self, commit):
        if commit in self._empty_roots:
            return True
        expected = {
            "records.json": {},
            "events.json": {},
            "approvals.json": {},
            "evidence.json": {},
            "baseline.json": None,
        }
        parents = self._git("rev-list", "--parents", "-n", "1", commit).split()[1:]
        valid = (
            not parents
            and set(self._git("ls-tree", "--name-only", commit).splitlines()) == set(expected)
            and all(self._read(commit, name) == value for name, value in expected.items())
        )
        if valid:
            self._empty_roots.add(commit)
        return valid

    def pin_legacy_seed(self, seed_commit, *, base_manifest_path, expected_sha256):
        """Explicit host migration using the previously recorded seed SHA.

        The host must supply its trusted baseline-bound commit, never the current
        head inferred from this repository. No model tool exposes this method.
        """
        from agent_rag.knowledge_registry import _validate_snapshot

        path = Path(base_manifest_path).resolve()
        _need(
            isinstance(seed_commit, str) and _COMMIT.fullmatch(seed_commit),
            "INVALID_KNOWLEDGE_COMMIT",
        )
        _need(
            path.name == "manifest.json"
            and isinstance(expected_sha256, str)
            and re.fullmatch("[0-9a-f]{64}", expected_sha256)
            and file_hash(path) == expected_sha256,
            "BASELINE_HOST_PIN_MISMATCH",
        )
        with update_lock(self.root / "publisher-lock"):
            previous = self._anchors()["legacy_seeds"].get(seed_commit)
            if previous is not None:
                _need(
                    previous.get("seed_commit") == seed_commit
                    and previous.get("manifest_sha256") == expected_sha256
                    and previous.get("base_manifest_path") == str(path),
                    "LEGACY_SEED_IMMUTABLE",
                )
                return {"knowledge_commit": seed_commit, "idempotent": True}
            _need(
                seed_commit in self._git("rev-list", self._head()).splitlines(),
                "LEGACY_SEED_NOT_RETAINED",
            )
            metadata = self._read(seed_commit, "baseline.json")
            records, evidence = (
                self._read(seed_commit, "records.json"),
                self._read(seed_commit, "evidence.json"),
            )
            parents = self._git("rev-list", "--parents", "-n", "1", seed_commit).split()[1:]
            _need(
                len(parents) == 1
                and self._empty_initial(parents[0])
                and isinstance(metadata, dict)
                and metadata.get("manifest_sha256") == expected_sha256
                and metadata.get("admission_status") == "legacy_candidate"
                and metadata.get("records_sha256") == canonical_sha256(records)
                and metadata.get("evidence_manifest_sha256") == canonical_sha256(evidence)
                and records
                and not self._read(seed_commit, "events.json")
                and not self._read(seed_commit, "approvals.json")
                and set(self._git("ls-tree", "--name-only", seed_commit).splitlines())
                == {
                    "records.json",
                    "events.json",
                    "approvals.json",
                    "evidence.json",
                    "baseline.json",
                }
                and all(
                    record.get("admission_status") == "legacy_candidate"
                    and record.get("baseline_manifest_sha256") == expected_sha256
                    for record in records.values()
                ),
                "LEGACY_SEED_ANCHOR_INVALID",
            )
            _validate_snapshot(path.parent)
            _need(file_hash(path) == expected_sha256, "BASELINE_HOST_PIN_MISMATCH")
            self._check_originals(evidence)
            self._record_seed_anchor(seed_commit, metadata, path, expected_sha256)
            return {"knowledge_commit": seed_commit, "idempotent": False}

    def _validate_lineage(self, commit):
        anchors = self._anchors()["legacy_seeds"]
        pending = []
        anchored = False
        for line in self._git("rev-list", "--parents", commit).splitlines():
            node, *parents = line.split()
            entry = anchors.get(node)
            if entry is not None:
                _need(
                    entry.get("seed_commit") == node
                    and entry.get("seed_metadata") == self._read(node, "baseline.json")
                    and file_hash(Path(entry["base_manifest_path"]))
                    == entry.get("manifest_sha256"),
                    "LEGACY_SEED_ANCHOR_INVALID",
                )
                anchored = True
                break
            if not parents:
                _need(self._empty_initial(node), "UNAPPROVED_KNOWLEDGE_ANCESTOR")
                anchored = True
                break
            _need(len(parents) == 1, "UNAPPROVED_KNOWLEDGE_ANCESTOR")
            pending.append(node)
        _need(anchored, "UNAPPROVED_KNOWLEDGE_ANCESTOR")
        proof = None
        for node in reversed(pending):
            try:
                proof = self._approved_step(node)
            except ValueError as error:
                if node != commit:
                    raise ValueError("UNAPPROVED_KNOWLEDGE_ANCESTOR") from error
                raise
        return proof

    def _sync_corpus(self, commit):
        if commit in self._synced:
            return
        records = self._read(commit, "records.json")
        bulk = getattr(self.corpus, "register_records", None)
        if callable(bulk):
            bulk(
                [{"record_id": identity, **record} for identity, record in records.items()],
                commit=commit,
            )
            self._synced.add(commit)
            return
        for identity, record in records.items():
            self.corpus.register_record(
                identity,
                record["kind"],
                record["body"],
                commit=commit,
                record_version=record["record_version"],
                evidence_ids=record["evidence_ids"],
                scope_id=record["scope_id"],
                classification=record["classification"],
            )
        self._synced.add(commit)

    def records(self, commit=None):
        commit = commit or self._head()
        return [
            {
                "record_id": identity,
                "knowledge_commit": commit,
                "sha256": canonical_sha256(record["body"]),
                **record,
            }
            for identity, record in self._read(commit, "records.json").items()
        ]

    def seed_legacy_records(self, records, *, base_manifest_path, expected_sha256):
        """One host-pinned historical import; candidate provenance is not approval.

        The host derives these stable-ID records from its verified V2 baseline.
        This API is deliberately absent from model tool declarations.
        """
        from agent_rag.knowledge_registry import _validate_snapshot

        path = Path(base_manifest_path).resolve()
        _need(
            path.name == "manifest.json"
            and isinstance(expected_sha256, str)
            and re.fullmatch("[0-9a-f]{64}", expected_sha256)
            and file_hash(path) == expected_sha256,
            "BASELINE_HOST_PIN_MISMATCH",
        )
        _validate_snapshot(path.parent)
        _need(file_hash(path) == expected_sha256, "BASELINE_HOST_PIN_MISMATCH")
        _need(isinstance(records, list) and records, "LEGACY_RECORDS_REQUIRED")
        normalized = {}
        for item in records:
            _need(isinstance(item, dict), "INVALID_LEGACY_RECORD")
            identity = item.get("record_id")
            _need(
                isinstance(identity, str) and identity and identity not in normalized,
                "INVALID_LEGACY_RECORD_ID",
            )
            _need(
                item.get("kind") in {"case", "behavior", "law_article", "guide", "relation"}
                and isinstance(item.get("body"), dict)
                and item["body"],
                "INVALID_LEGACY_RECORD",
            )
            version = item.get("record_version", 1)
            _need(type(version) is int and version >= 1, "INVALID_LEGACY_RECORD_VERSION")
            scope_id, classification = (
                item.get("scope_id", "public"),
                item.get("classification", "public"),
            )
            _need(
                isinstance(scope_id, str)
                and scope_id
                and classification in {"public", "team_private", "business_sensitive"},
                "INVALID_LEGACY_RECORD_SCOPE",
            )
            evidence = item.get("evidence_ids", [])
            _need(
                isinstance(evidence, list)
                and all(isinstance(eid, str) and eid for eid in evidence),
                "INVALID_LEGACY_RECORD_EVIDENCE",
            )
            normalized[identity] = {
                "kind": item["kind"],
                "body": copy.deepcopy(item["body"]),
                "record_version": version,
                "evidence_ids": evidence,
                "scope_id": scope_id,
                "classification": classification,
                "admission_status": "legacy_candidate",
                "baseline_manifest_sha256": expected_sha256,
            }
        metadata = {
            "manifest_sha256": expected_sha256,
            "records_sha256": canonical_sha256(normalized),
            "admission_status": "legacy_candidate",
        }
        with update_lock(self.root / "publisher-lock"):
            head = self._head()
            events = self._read(head, "events.json")
            _need(not events, "LEGACY_SEED_ONLY_EMPTY_HEAD")
            source_ids = sorted(
                {eid for record in normalized.values() for eid in record["evidence_ids"]}
            )
            sources = {eid: self.corpus.resolve_evidence(eid) for eid in source_ids}
            refs = [
                {key: source[key] for key in ("evidence_id", "raw_sha256", "text_sha256")}
                for source in sources.values()
            ]
            evidence = self._preserve_evidence(
                {"evidence_refs": refs}, {"evidence_resolver": sources.get}, {}
            )
            metadata["evidence_manifest_sha256"] = canonical_sha256(evidence)
            previous = self._read(head, "baseline.json")
            if previous is not None:
                _need(
                    previous == metadata
                    and self._read(head, "records.json") == normalized
                    and self._read(head, "evidence.json") == evidence,
                    "LEGACY_SEED_IMMUTABLE",
                )
                parents = self._git("rev-list", "--parents", "-n", "1", head).split()[1:]
                _need(
                    len(parents) == 1
                    and self._empty_initial(parents[0])
                    and set(self._git("ls-tree", "--name-only", head).splitlines())
                    == {
                        "records.json",
                        "events.json",
                        "approvals.json",
                        "evidence.json",
                        "baseline.json",
                    },
                    "LEGACY_SEED_ANCHOR_INVALID",
                )
                self._record_seed_anchor(head, metadata, path, expected_sha256)
                self._sync_corpus(head)
                return {"knowledge_commit": head, "idempotent": True, **metadata}
            _need(self._empty_initial(head), "LEGACY_SEED_ONLY_EMPTY_HEAD")
            tree = self._tree(
                {
                    "records.json": normalized,
                    "events.json": {},
                    "approvals.json": {},
                    "evidence.json": evidence,
                    "baseline.json": metadata,
                }
            )
            commit = self._git(
                "commit-tree", tree, "-p", head, data=b"Pin historical candidate baseline\n"
            )
            self._git("update-ref", "refs/heads/knowledge", commit, head)
            self._record_seed_anchor(commit, metadata, path, expected_sha256)
            self._sync_corpus(commit)
            return {"knowledge_commit": commit, "idempotent": False, **metadata}

    def _approval(self, changeset, receipt, context=None, *, base=None):
        ctx = context if context is not None else self._context
        if callable(ctx):
            ctx = ctx(copy.deepcopy(changeset))
        _need(isinstance(ctx, dict), "APPROVAL_CONTEXT_REQUIRED")
        ctx = dict(ctx)
        ctx["expected_base"] = base or changeset["base_knowledge_commit"]
        require_approved(changeset, receipt, **ctx)
        return ctx

    def _preserve_blob(self, directory, digest, data):
        _need(hashlib.sha256(data).hexdigest() == digest, "ORIGINAL_HASH_MISMATCH")
        folder = self.root / directory
        folder.mkdir(exist_ok=True)
        _need(folder.resolve().parent == self.root and not folder.is_symlink(), "ORIGINAL_PATH")
        destination = folder / digest
        if destination.exists():
            _need(file_hash(destination) == digest, "ORIGINAL_HASH_MISMATCH")
            return
        temporary = folder / (".tmp-" + uuid4().hex)
        try:
            with temporary.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def _preserve_evidence(self, changeset, context, previous):
        evidence = copy.deepcopy(previous)
        for ref in changeset["evidence_refs"]:
            source = context["evidence_resolver"](ref["evidence_id"])
            self._preserve_blob(
                "originals", ref["raw_sha256"], source.get("raw_bytes", source.get("raw"))
            )
            self._preserve_blob("texts", ref["text_sha256"], source["text"].encode("utf-8"))
            metadata = self._evidence_metadata(ref, source)
            old = evidence.get(ref["evidence_id"])
            _need(old is None or old == metadata, "EVIDENCE_VERSION_IMMUTABLE")
            evidence[ref["evidence_id"]] = metadata
        return evidence

    @staticmethod
    def _evidence_metadata(ref, source):
        return {
            **ref,
            "extractor_version": source["extractor_version"],
            "scope_id": source["scope_id"],
            "classification": source["classification"],
            "origin_id": source.get("origin_id"),
            "text_length_codepoints": len(source["text"]),
        }

    def commit_changeset(self, changeset, receipt, approval_context=None):
        """Host authorization then one Git update-ref CAS; replay repairs mirror only."""
        with update_lock(self.root / "publisher-lock"):
            head = self._head()
            self._validate_lineage(head)
            key = canonical_sha256([changeset["event"]["event_id"], changeset["revision"]])
            events = self._read(head, "events.json")
            if key in events:
                entry = events[key]
                _need(
                    entry["changeset_sha256"] == changeset["changeset_sha256"]
                    and entry["receipt_sha256"] == canonical_sha256(receipt),
                    "IDEMPOTENCY_CONFLICT",
                )
                ctx = self._approval(changeset, receipt, approval_context)
                # The first ancestor containing this key is the atomic publication.
                commit = next(
                    c
                    for c in self._git("rev-list", "--first-parent", "--reverse", head).splitlines()
                    if key in self._read(c, "events.json")
                )
                self._contexts[commit] = ctx
                self._sync_corpus(commit)
                return {"knowledge_commit": commit, "idempotent": True, **entry}
            ctx = self._approval(changeset, receipt, approval_context, base=head)
            records = self._read(head, "records.json")
            scope = changeset["scope"]
            for op in changeset["operations"]:
                identity, old = op["target_id"], records.get(op["target_id"])
                if op["kind"] in {"add", "relate"}:
                    _need(old is None, "RECORD_ALREADY_EXISTS")
                    version = 1
                else:
                    _need(old is not None, "UNKNOWN_RECORD")
                    _need(
                        old["record_version"] == op["expected_record_version"],
                        "RECORD_VERSION_MISMATCH",
                    )
                    _need(
                        canonical_sha256(old["body"]) == op["before_sha256"], "BEFORE_HASH_MISMATCH"
                    )
                    _need(old["kind"] == op["record_kind"], "RECORD_KIND_MISMATCH")
                    _need(
                        old["scope_id"] == scope["scope_id"]
                        and old["classification"] == scope["classification"],
                        "RECORD_SCOPE_MISMATCH",
                    )
                    version = old["record_version"] + 1
                if op["kind"] == "relate":
                    _need(op["record_kind"] == "relation", "RELATION_KIND_REQUIRED")
                if op["kind"] == "deactivate":
                    _need(op["after"].get("active") is False, "DEACTIVATION_REQUIRES_INACTIVE")
                records[identity] = {
                    "kind": op["record_kind"],
                    "record_version": version,
                    "body": copy.deepcopy(op["after"]),
                    "evidence_ids": [r["evidence_id"] for r in changeset["evidence_refs"]],
                    "admission_status": "model_review_approved",
                    "baseline_manifest_sha256": old.get("baseline_manifest_sha256")
                    if old
                    else None,
                    **scope,
                }
            from agent_rag.maintenance_tools import authorized

            for record in records.values():
                dependencies = record["body"].get("depends_on", [])
                if record["kind"] == "relation":
                    dependencies = [
                        *dependencies,
                        record["body"].get("source_id"),
                        record["body"].get("target_id"),
                    ]
                for target in dependencies:
                    _need(isinstance(target, str) and target in records, "RELATION_TARGET_UNKNOWN")
                    _need(authorized(records[target], record), "RELATION_SCOPE_MISMATCH")
            evidence = self._preserve_evidence(changeset, ctx, self._read(head, "evidence.json"))
            approvals = self._read(head, "approvals.json")
            approvals[key] = {"changeset": changeset, "receipt": receipt}
            entry = {
                "event_id": changeset["event"]["event_id"],
                "revision": changeset["revision"],
                "changeset_sha256": changeset["changeset_sha256"],
                "receipt_sha256": canonical_sha256(receipt),
                "base_commit": head,
            }
            events[key] = entry
            tree = self._tree(
                {
                    "records.json": records,
                    "events.json": events,
                    "approvals.json": approvals,
                    "evidence.json": evidence,
                    "baseline.json": self._read(head, "baseline.json"),
                }
            )
            commit = self._git(
                "commit-tree", tree, "-p", head, data=("Approve knowledge " + key + "\n").encode()
            )
            self._git("update-ref", "refs/heads/knowledge", commit, head)
            self._contexts[commit] = ctx
            self._sync_corpus(commit)
            return {"knowledge_commit": commit, "idempotent": False, **entry}

    def _proof(self, commit):
        # Must be an ancestor of this owned repository's canonical branch.
        _need(commit in self._git("rev-list", self._head()).splitlines(), "COMMIT_NOT_APPROVED")
        proof = self._validate_lineage(commit)
        _need(proof is not None, "COMMIT_NOT_APPROVED")
        return proof

    def _check_originals(self, evidence):
        for source in evidence.values():
            _need(
                file_hash(self.root / "originals" / source["raw_sha256"]) == source["raw_sha256"]
                and file_hash(self.root / "texts" / source["text_sha256"]) == source["text_sha256"],
                "ORIGINAL_HASH_MISMATCH",
            )

    def _approved_step(self, commit):
        cached = self._structures.get(commit)
        if cached is not None:
            proof, evidence = cached["proof"], cached["evidence"]
            ch = proof["changeset"]
            ctx = self._approval(ch, proof["receipt"], self._contexts.get(commit))
            for ref in ch["evidence_refs"]:
                _need(
                    evidence[ref["evidence_id"]]
                    == self._evidence_metadata(ref, ctx["evidence_resolver"](ref["evidence_id"])),
                    "COMMIT_APPROVAL_BINDING",
                )
            self._check_originals(evidence)
            return copy.deepcopy(proof)
        events = self._read(commit, "events.json")
        parents = self._git("rev-list", "--parents", "-n", "1", commit).split()[1:]
        _need(len(parents) == 1, "COMMIT_NOT_APPROVED")
        parent = parents[0]
        matches = [key for key, event in events.items() if event["base_commit"] == parent]
        _need(len(matches) == 1, "COMMIT_NOT_APPROVED")
        proof = self._read(commit, "approvals.json")[matches[0]]
        ch = proof["changeset"]
        _need(
            ch["base_knowledge_commit"] == parent
            and matches[0] == canonical_sha256([ch["event"]["event_id"], ch["revision"]])
            and events[matches[0]]["changeset_sha256"] == ch["changeset_sha256"]
            and events[matches[0]]["receipt_sha256"] == canonical_sha256(proof["receipt"]),
            "COMMIT_APPROVAL_BINDING",
        )
        ctx = self._contexts.get(commit)
        ctx = self._approval(ch, proof["receipt"], ctx)
        expected = self._read(parent, "records.json")
        for op in ch["operations"]:
            old = expected.get(op["target_id"])
            if op["kind"] in {"add", "relate"}:
                _need(old is None, "COMMIT_APPROVAL_BINDING")
            else:
                _need(
                    old is not None
                    and old["record_version"] == op["expected_record_version"]
                    and canonical_sha256(old["body"]) == op["before_sha256"],
                    "COMMIT_APPROVAL_BINDING",
                )
            expected[op["target_id"]] = {
                "kind": op["record_kind"],
                "record_version": old["record_version"] + 1 if old else 1,
                "body": op["after"],
                "evidence_ids": [ref["evidence_id"] for ref in ch["evidence_refs"]],
                "admission_status": "model_review_approved",
                "baseline_manifest_sha256": old.get("baseline_manifest_sha256") if old else None,
                **ch["scope"],
            }
        _need(
            expected == self._read(commit, "records.json")
            and self._read(commit, "baseline.json") == self._read(parent, "baseline.json")
            and events == {**self._read(parent, "events.json"), matches[0]: events[matches[0]]}
            and self._read(commit, "approvals.json")
            == {**self._read(parent, "approvals.json"), matches[0]: proof},
            "COMMIT_APPROVAL_BINDING",
        )
        evidence = self._read(parent, "evidence.json")
        for ref in ch["evidence_refs"]:
            evidence[ref["evidence_id"]] = self._evidence_metadata(
                ref, ctx["evidence_resolver"](ref["evidence_id"])
            )
        _need(evidence == self._read(commit, "evidence.json"), "COMMIT_APPROVAL_BINDING")
        self._check_originals(evidence)
        self._structures[commit] = copy.deepcopy({"proof": proof, "evidence": evidence})
        return proof

    def _state(self):
        path = self.root / "release-state.json"
        if not path.exists():
            return {
                "schema": "finrisk-release-selection/1.0",
                "baseline_release": None,
                "selected_release": None,
                "enabled": False,
                "history": [],
                "registered": {},
            }
        value = json.loads(path.read_text(encoding="utf-8"))
        _need(
            value.get("schema") == "finrisk-release-selection/1.0"
            and type(value.get("enabled")) is bool
            and isinstance(value.get("registered"), dict)
            and isinstance(value.get("history"), list),
            "INVALID_RELEASE_STATE",
        )
        return value

    def state(self):
        value = self._state()
        value["effective_release"] = (
            value["selected_release"] if value["enabled"] else value["baseline_release"]
        )
        return value

    def _registered(self, release_id, state=None):
        state = state or self._state()
        _need(
            isinstance(release_id, str)
            and _ID.fullmatch(release_id)
            and release_id in state["registered"],
            "RELEASE_NOT_REGISTERED",
        )
        entry = state["registered"][release_id]
        path = self.root / "releases" / release_id / "manifest.json"
        _need(
            path.resolve().is_relative_to((self.root / "releases").resolve())
            and file_hash(path) == entry["manifest_sha256"],
            "RELEASE_MANIFEST_HASH_MISMATCH",
        )
        value = json.loads(path.read_text(encoding="utf-8"))
        pinned = release_id == state["baseline_release"]
        validate_release_manifest(value, validators=self.validators, require_validators=not pinned)
        _need(value["release_id"] == release_id, "RELEASE_ID_MISMATCH")
        if not pinned:
            proof = self._proof(value["knowledge_commit"])
            _need(
                value["approval"]
                == {
                    "changeset_sha256": proof["changeset"]["changeset_sha256"],
                    "receipt_sha256": canonical_sha256(proof["receipt"]),
                },
                "RELEASE_APPROVAL_BINDING",
            )
        return value

    def bind_baseline(self, manifest_path, expected_sha256):
        """Administrative host-only bind: fixed bytes, never a model legacy flag."""
        with update_lock(self.root / "publisher-lock"):
            path = Path(manifest_path)
            _need(
                isinstance(expected_sha256, str)
                and re.fullmatch("[0-9a-f]{64}", expected_sha256)
                and file_hash(path) == expected_sha256,
                "BASELINE_HOST_PIN_MISMATCH",
            )
            manifest = validate_release_manifest(json.loads(path.read_text(encoding="utf-8")))
            state = self._state()
            if state["baseline_release"] is not None:
                _need(
                    state["baseline_release"] == manifest["release_id"]
                    and state["registered"][manifest["release_id"]]["source_manifest_sha256"]
                    == expected_sha256,
                    "BASELINE_ALREADY_BOUND",
                )
                self._registered(manifest["release_id"], state)
                return self.state()
            rid = manifest["release_id"]
            _need(rid not in state["registered"], "RELEASE_IMMUTABLE")
            destination = self.root / "releases" / rid / "manifest.json"
            write_json(destination, manifest)
            state["registered"][rid] = {
                "manifest_sha256": file_hash(destination),
                "source_manifest_sha256": expected_sha256,
            }
            state["baseline_release"] = rid
            write_json(self.root / "release-state.json", state)
            return self.state()

    def _asset_bindings(self, records, bindings, evidence):
        def by_kind(kinds):
            return {key: val for key, val in records.items() if val["kind"] in kinds}

        cases, laws, guides = (
            by_kind({"case", "behavior"}),
            by_kind({"law_article"}),
            by_kind({"guide", "relation"}),
        )
        values = {
            "cases": [cases, bindings["dictionary_sha256"]],
            "law_scope": laws,
            "guides_relations": [guides, laws, cases, bindings["dictionary_sha256"]],
            "model": bindings.get("model_sha256"),
            "dictionary": bindings["dictionary_sha256"],
            "reader": bindings["business_schema_sha256"],
            "query": bindings["tool_contract_sha256"],
            "acl": {key: [v["scope_id"], v["classification"]] for key, v in records.items()},
            "raw_catalog": evidence,
        }
        values.update({name: [records, bindings] for name in ("bm25", "dense", "rowmap", "active")})
        return {name: canonical_sha256(value) for name, value in values.items()}

    def _recover_release(self, destination, expected, parent, state):
        """Revalidate a complete orphan after registration was interrupted."""
        manifest = json.loads(destination.read_text(encoding="utf-8"))
        _need(
            all(manifest.get(key) == value for key, value in expected.items()), "RELEASE_IMMUTABLE"
        )
        validate_release_manifest(manifest, validators=self.validators, require_validators=True)
        staging_root = (self.root / "staging").resolve()
        for name, asset in manifest["assets"].items():
            path = Path(asset["path"]).resolve()
            owned = path.is_relative_to(staging_root) and path.relative_to(staging_root).parts[
                0
            ].startswith(expected["release_id"] + "-")
            inherited = (
                parent is not None
                and asset == parent["assets"].get(name)
                and parent["asset_bindings"][name] == expected["asset_bindings"][name]
            )
            _need(owned or inherited, "RELEASE_ASSET_PATH_OR_STALE_REUSE")
        snapshot = manifest.get("v2_snapshot")
        if snapshot is not None:
            path = Path(snapshot["path"]).resolve()
            _need(
                path.is_relative_to(staging_root)
                and file_hash(path / "manifest.json") == snapshot["manifest_sha256"],
                "SNAPSHOT_BUILD_BINDING",
            )
            body = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            _need(
                body.get("knowledge_commit") == expected["knowledge_commit"]
                and body.get("release_id") == expected["release_id"],
                "SNAPSHOT_BUILD_BINDING",
            )
        state["registered"][expected["release_id"]] = {"manifest_sha256": file_hash(destination)}
        write_json(self.root / "release-state.json", state)

    def build_release(self, commit, *, parent_release=None, builder_id="default"):
        """Build only approved commits through a host allowlist; no placeholder assets."""
        with update_lock(self.root / "publisher-lock"):
            proof = self._proof(commit)
            rid = "release-" + canonical_sha256([commit, parent_release, builder_id])[:32]
            state = self._state()
            if rid in state["registered"]:
                self._registered(rid, state)
                return {
                    "status": "built",
                    "selectable": True,
                    "release_id": rid,
                    "manifest_path": str(self.root / "releases" / rid / "manifest.json"),
                }
            parent = self._registered(parent_release, state) if parent_release else None
            if parent:
                _need(
                    parent["knowledge_commit"] in self._git("rev-list", commit).splitlines(),
                    "STALE_KNOWLEDGE_PARENT",
                )
            builder = self.builders.get(builder_id)
            result = {
                "status": "blocked",
                "selectable": False,
                "release_id": rid,
                "knowledge_commit": commit,
            }
            if not callable(builder):
                return {**result, "code": "RELEASE_BUILDER_NOT_CONFIGURED"}
            if not REQUIRED_ASSETS <= self.validators.keys():
                return {**result, "code": "ASSET_VALIDATOR_REQUIRED"}
            records = self._read(commit, "records.json")
            evidence = self._read(commit, "evidence.json")
            fingerprints = self._asset_bindings(records, proof["changeset"]["bindings"], evidence)
            approval = {
                "changeset_sha256": proof["changeset"]["changeset_sha256"],
                "receipt_sha256": canonical_sha256(proof["receipt"]),
            }
            destination = self.root / "releases" / rid / "manifest.json"
            if destination.exists():
                try:
                    self._recover_release(
                        destination,
                        {
                            "release_id": rid,
                            "knowledge_commit": commit,
                            "parent_release": parent_release,
                            "builder_id": builder_id,
                            "asset_bindings": fingerprints,
                            "approval": approval,
                        },
                        parent,
                        state,
                    )
                    return {
                        **result,
                        "status": "built",
                        "selectable": True,
                        "manifest_path": str(destination),
                        "recovered": True,
                    }
                except (ValueError, OSError, KeyError, TypeError):
                    return {**result, "code": "RELEASE_RECOVERY_BLOCKED"}
            staging = self.root / "staging" / (rid + "-" + uuid4().hex[:8])
            staging.mkdir(parents=True)
            try:
                request = {
                    "knowledge_commit": commit,
                    "release_id": rid,
                    "records": copy.deepcopy(records),
                    "parent": copy.deepcopy(parent),
                    "asset_bindings": fingerprints,
                    "bindings": proof["changeset"]["bindings"],
                    "raw_catalog": {
                        identity: {
                            **source,
                            "raw_path": str(self.root / "originals" / source["raw_sha256"]),
                            "text_path": str(self.root / "texts" / source["text_sha256"]),
                        }
                        for identity, source in evidence.items()
                    },
                }
                output = builder(request, staging)
                _need(
                    isinstance(output, dict) and isinstance(output.get("assets"), dict),
                    "INVALID_BUILD_OUTPUT",
                )
                assets = copy.deepcopy(output["assets"])
                for name in REQUIRED_ASSETS:
                    if (
                        name not in assets
                        and parent
                        and parent["asset_bindings"][name] == fingerprints[name]
                    ):
                        assets[name] = copy.deepcopy(parent["assets"][name])
                for name, asset in assets.items():
                    path = Path(asset["path"]).resolve()
                    if path.is_relative_to(staging.resolve()):
                        _need(not Path(asset["path"]).is_symlink(), "RELEASE_ASSET_PATH")
                    else:
                        _need(
                            parent is not None
                            and name in parent["assets"]
                            and asset == parent["assets"][name]
                            and parent["asset_bindings"][name] == fingerprints[name],
                            "RELEASE_ASSET_PATH_OR_STALE_REUSE",
                        )
                manifest = {
                    "schema": RELEASE_SCHEMA,
                    "release_id": rid,
                    "knowledge_commit": commit,
                    "parent_release": parent_release,
                    "assets": assets,
                    "asset_bindings": fingerprints,
                    "builder_id": builder_id,
                    "approval": approval,
                }
                if "v2_snapshot" in output:
                    snapshot = output["v2_snapshot"]
                    snapshot_path = Path(snapshot["path"]).resolve()
                    _need(
                        snapshot_path.is_relative_to(staging.resolve())
                        and file_hash(snapshot_path / "manifest.json")
                        == snapshot["manifest_sha256"],
                        "SNAPSHOT_BUILD_BINDING",
                    )
                    body = json.loads((snapshot_path / "manifest.json").read_text(encoding="utf-8"))
                    _need(
                        body.get("knowledge_commit") == commit and body.get("release_id") == rid,
                        "SNAPSHOT_BUILD_BINDING",
                    )
                    manifest["v2_snapshot"] = snapshot
                validate_release_manifest(
                    manifest, validators=self.validators, require_validators=True
                )
                # Validate staging in full before a final manifest/registration is visible.
                write_json(staging / "manifest.json", manifest)
                destination = self.root / "releases" / rid / "manifest.json"
                if destination.exists():
                    _need(
                        json.loads(destination.read_text(encoding="utf-8")) == manifest,
                        "RELEASE_IMMUTABLE",
                    )
                else:
                    write_json(destination, manifest)
                state["registered"][rid] = {"manifest_sha256": file_hash(destination)}
                write_json(self.root / "release-state.json", state)
                return {
                    **result,
                    "status": "built",
                    "selectable": True,
                    "manifest_path": str(destination),
                }
            except (ValueError, OSError, KeyError, TypeError) as error:
                code = (
                    str(error)
                    if re.fullmatch("[A-Z][A-Z0-9_]{0,100}", str(error))
                    else "RELEASE_BUILD_FAILED"
                )
                write_json(staging / "build-blocked.json", {**result, "code": code})
                return {**result, "code": code}

    def select_release(self, release_id):
        with update_lock(self.root / "publisher-lock"):
            state = self._state()
            release = self._registered(release_id, state)
            previous = state["selected_release"] or state["baseline_release"]
            _need(previous is not None, "BASELINE_NOT_BOUND")
            if release_id != previous:
                _need(
                    release.get("parent_release") == previous,
                    "STALE_PARENT_REBUILD_ON_SELECTED_VERSION",
                )
                state["history"] = list(dict.fromkeys([*state["history"], previous]))
            state["selected_release"] = release_id
            write_json(self.root / "release-state.json", state)
            return self.state()

    def rollback(self, release_id):
        with update_lock(self.root / "publisher-lock"):
            state = self._state()
            self._registered(release_id, state)
            _need(
                release_id in state["history"] or release_id == state["selected_release"],
                "RELEASE_NOT_RETAINED",
            )
            previous = state["selected_release"]
            if previous:
                state["history"] = list(dict.fromkeys([*state["history"], previous]))
            state["selected_release"] = release_id
            write_json(self.root / "release-state.json", state)
            return self.state()

    def set_usage(self, enabled):
        _need(type(enabled) is bool, "INVALID_USAGE_ENABLED")
        with update_lock(self.root / "publisher-lock"):
            state = self._state()
            _need(state["baseline_release"] is not None, "BASELINE_NOT_BOUND")
            self._registered(state["baseline_release"], state)
            if enabled:
                _need(state["selected_release"] is not None, "USAGE_NO_SELECTED_VERSION")
                self._registered(state["selected_release"], state)
            state["enabled"] = enabled
            write_json(self.root / "release-state.json", state)
            return self.state()


class HostAdmissionAuthority:
    """Opaque host capability: only a publisher's complete approved release admits V2.

    This object is an injected service dependency, never deserialized from input.
    A boolean, a marker file, or a caller supplied receipt cannot substitute for it.
    """

    def __init__(self, publisher, *, key=None):
        _need(isinstance(publisher, KnowledgePublisher), "INVALID_ADMISSION_HOST")
        self.publisher = publisher
        self._key = key if key is not None else secrets.token_bytes(32)
        _need(type(self._key) is bytes and len(self._key) >= 32, "INVALID_ADMISSION_KEY")

    def issue(self, registry, snapshot_path, release_id):
        release = self.publisher._registered(release_id)
        _need(
            release_id != self.publisher._state()["baseline_release"], "EXTENSION_APPROVAL_REQUIRED"
        )
        snapshot = release.get("v2_snapshot")
        path = Path(snapshot_path).resolve()
        _need(
            isinstance(snapshot, dict)
            and file_hash(Path(snapshot["path"]) / "manifest.json") == snapshot["manifest_sha256"]
            and file_hash(path / "manifest.json") == snapshot["manifest_sha256"],
            "SNAPSHOT_BUILD_BINDING",
        )
        value = {
            "registry": str(Path(registry).resolve()),
            "snapshot_path": str(path),
            "manifest_sha256": snapshot["manifest_sha256"],
            "release_id": release_id,
        }
        value["authentication"] = {
            "algorithm": "HMAC-SHA256",
            "signature": hmac.new(self._key, canonical_bytes(value), hashlib.sha256).hexdigest(),
        }
        return value

    def verify(self, registry, snapshot_path, admission):
        _need(type(admission) is dict, "HOST_ADMISSION_REQUIRED")
        expected = self.issue(registry, snapshot_path, admission.get("release_id"))
        _need(
            hmac.compare_digest(canonical_bytes(admission), canonical_bytes(expected)),
            "HOST_ADMISSION_REQUIRED",
        )
        return True
