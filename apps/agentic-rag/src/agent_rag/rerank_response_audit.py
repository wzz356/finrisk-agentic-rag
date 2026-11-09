"""Bounded model-message capture; never stores headers, keys or reasoning fields."""

import hashlib
import json
import os
import tempfile
from pathlib import Path


def record_response(control, packet: dict, response: dict, protocol: str) -> None:
    choices = response.get("choices") if isinstance(response, dict) else None
    message = choices[0].get("message") if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    output = content if isinstance(content, str) else None
    path = control.ledger_path.with_name(control.ledger_path.name + ".responses.json")
    with control._lock:
        ledger = control._read()
        matching = [row for row in ledger["request_reservations"] if row["pair_id"] == packet["pair_id"]]
        ordinal = matching[-1]["ordinal"]
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if (data.get("schema_version") != "finrisk-rerank-responses/1.0"
                    or data.get("ledger_ref") != str(control.ledger_path)
                    or not isinstance(data.get("records"), list)
                    or any(not isinstance(row, dict) or type(row.get("ordinal")) is not int for row in data["records"])
                    or len({row["ordinal"] for row in data["records"]}) != len(data["records"])):
                raise ValueError("INVALID_RERANK_RESPONSE_AUDIT")
        else:
            data = {"schema_version": "finrisk-rerank-responses/1.0", "ledger_ref": str(control.ledger_path), "records": []}
        if any(row["ordinal"] == ordinal for row in data["records"]):
            raise ValueError("DUPLICATE_RERANK_RESPONSE_AUDIT")
        data["records"].append({
            "ordinal": ordinal, "pair_id": packet["pair_id"], "rerank_protocol": protocol,
            "query_text_sha256": hashlib.sha256(packet["query_text"].encode("utf-8")).hexdigest(),
            "unit_text_sha256": hashlib.sha256(packet["unit_text"].encode("utf-8")).hexdigest(),
            "capture_kind": "selected_model_message_content_not_raw_HTTP",
            "model_output": output[:16384] if output is not None else None,
            "model_output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest() if output is not None else None,
            "capture_complete": output is not None and len(output) <= 16384,
        })
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=2)
                stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
