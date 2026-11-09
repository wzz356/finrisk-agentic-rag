import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from build_demo import build
from agent_rag.input_registry import InputRegistry
from agent_rag.knowledge_snapshot import KnowledgeSnapshot, make_evidence
from agent_rag.token_count import OfficialPromptCounter


class PublicDemoTests(unittest.TestCase):
    def test_synthetic_source_metadata_and_hash_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = build(Path(temporary) / "snapshot")
            self.assertTrue(snapshot.manifest["synthetic"])
            self.assertEqual(snapshot.manifest["source_count"], 3)
            self.assertEqual(len(snapshot.definitions), 27)
            source_id = next(iter(snapshot.sources))
            source, text = snapshot.read(source_id)
            self.assertIn("合成", text)
            self.assertTrue(source["synthetic"])
            evidence = make_evidence(source, 0, len(text), text)
            self.assertEqual(evidence["text_quality"], "synthetic_demo_not_real_penalty")
            Path(source["path"]).write_text(text + "篡改", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SOURCE_HASH_MISMATCH"):
                snapshot.read(source_id)

    def test_unicode_positions_are_exact(self):
        registry = InputRegistry("A😀同句。同句。", task_id="demo")
        span = registry.resolve({"segment_id": "S0001", "quote": "同句。", "occurrence": 2})
        self.assertEqual(registry.text[span["start"]:span["end"]], "同句。")
        self.assertEqual(span["start"], 5)

    def test_v2_backend_imports_without_loading_gpu_or_network(self):
        import agent_rag.v2_runtime
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)

    def test_pinned_counter_assets_verify_and_count_locally(self):
        counter = OfficialPromptCounter()
        result = counter.count({"model": "deepseek-flash", "messages": [
            {"role": "system", "content": "离线测试"},
            {"role": "user", "content": "完全合成的输入。"}]})
        self.assertGreater(result["prompt_tokens"], 0)
        self.assertEqual(result["counter_kind"], "deepseek_v41_official")


if __name__ == "__main__":
    unittest.main()
