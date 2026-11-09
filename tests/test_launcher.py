"""Behavior checks for the repository entrypoint; no application dependencies."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT / "finrisk.py").is_file(), "Unified launcher is missing")
        self.temporary = tempfile.TemporaryDirectory(prefix="finrisk launcher ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        shutil.copy2(ROOT / "finrisk.py", self.root / "finrisk.py")
        self.app = self.root / "apps/workbench"
        (self.app / "scripts").mkdir(parents=True)
        (self.app / "scripts/run_web.py").write_text(
            "import json, os, sys\n"
            "print(json.dumps({'cwd': os.getcwd(), 'args': sys.argv[1:]}))\n"
            "raise SystemExit(int(os.environ.get('FINRISK_TEST_EXIT', '0')))\n",
            encoding="utf-8",
        )

    def run_launcher(self, *args, env=None):
        return subprocess.run(
            [sys.executable, str(self.root / "finrisk.py"), *args],
            cwd=ROOT.parent,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

    def test_workbench_uses_its_own_directory_and_forwards_port(self):
        result = self.run_launcher("workbench", "--python", sys.executable, "--port", "18088")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(Path(payload["cwd"]), self.app)
        self.assertEqual(payload["args"], ["--port", "18088"])

    def test_child_failure_is_returned_to_caller(self):
        env = dict(os.environ, FINRISK_TEST_EXIT="7")
        result = self.run_launcher("workbench", "--python", sys.executable, env=env)
        self.assertEqual(result.returncode, 7, result.stderr)

    def test_missing_environment_has_actionable_error(self):
        result = self.run_launcher("workbench")
        self.assertEqual(result.returncode, 2)
        self.assertIn("docs/SETUP.md", result.stderr)

    def test_invalid_port_is_rejected(self):
        result = self.run_launcher("workbench", "--python", sys.executable, "--port", "0")
        self.assertEqual(result.returncode, 2)

    @unittest.skipUnless(os.name == "nt", "Existing demo launcher uses Windows PowerShell")
    def test_demo_disables_inherited_live_switches(self):
        app = self.root / "apps/agentic-rag"
        app.mkdir(parents=True)
        (app / "start-agent-rag.ps1").write_text(
            "param([string]$PythonPath, [int]$PythonPort, [int]$NodePort)\n"
            "@{live=$env:FINRISK_AGENT_ALLOW_LIVE; "
            "retrieval=$env:FINRISK_AGENT_ALLOW_RETRIEVAL_LIVE; "
            "port=$PythonPort; node=$NodePort; cwd=(Get-Location).Path} "
            "| ConvertTo-Json -Compress\n",
            encoding="utf-8",
        )
        env = dict(os.environ, FINRISK_AGENT_ALLOW_LIVE="1", FINRISK_AGENT_ALLOW_RETRIEVAL_LIVE="1")
        result = self.run_launcher("demo", "--python", sys.executable, "--port", "18765", env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual((payload["live"], payload["retrieval"]), ("0", "0"))
        self.assertEqual(payload["port"], 18765)
        self.assertEqual(Path(payload["cwd"]), app)


if __name__ == "__main__":
    unittest.main()
