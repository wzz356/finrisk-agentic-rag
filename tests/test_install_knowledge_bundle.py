import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/install_knowledge_bundle.py'
spec = importlib.util.spec_from_file_location('install_bundle', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'data.txt').write_text('fixed evidence', encoding='utf-8')
        self.manifest = {'files':{'data.txt':{'bytes':14,'sha256':hashlib.sha256(b'fixed evidence').hexdigest()}}}
        (self.root / 'bundle-manifest.json').write_text(json.dumps(self.manifest))

    def test_verified_files(self):
        self.assertEqual(module.verify(self.root)['files'], self.manifest['files'])

    def test_corruption_rejected(self):
        (self.root/'data.txt').write_text('changed')
        with self.assertRaises(ValueError): module.verify(self.root)

    def test_traversal_rejected(self):
        self.manifest['files']['../outside'] = self.manifest['files'].pop('data.txt')
        (self.root/'bundle-manifest.json').write_text(json.dumps(self.manifest))
        with self.assertRaises(ValueError): module.verify(self.root)

    def test_cache_key_binds_relocated_model(self):
        meta = {'ordered_unit_text_sha256':'abc','binding':{'model':{'path':'old'}}}
        result = module.relocate_metadata(meta, self.root)
        self.assertEqual(result['binding']['model']['path'], str(self.root.resolve()))
        expected = hashlib.sha256(json.dumps({'text_sha':'abc','config':result['binding']},sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        self.assertEqual(result['key'], expected)
        self.assertEqual(meta['binding']['model']['path'], 'old')


if __name__ == '__main__': unittest.main()
