"""Synthetic shared fixtures; never load a key or real corpus."""
import importlib
import importlib.util

def backend():
    assert importlib.util.find_spec("agent_rag.rerank_backend") is not None, "重排后端未实现"
    return importlib.import_module("agent_rag.rerank_backend")

def unit(number=1, text="允许\n部分领取，需符合合同条件。", file_id=None):
    return {"unit_id": f"U{number}", "file_id": file_id or f"F{number}", "text": text}

def control(tmp_path, **kwargs):
    return backend().RerankTaskControl(
        tmp_path / "ledger.json", **{"cap_cny": 1, "max_requests": 100, **kwargs}
    )
