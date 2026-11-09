"""Launch either FinRisk application from the unified repository."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def port_number(value):
    number = int(value)
    if not 1 <= number <= 65535:
        raise argparse.ArgumentTypeError("Port must be between 1 and 65535")
    return number


def main():
    parser = argparse.ArgumentParser(description="FinRisk workbench and offline Agent demo")
    parser.add_argument("app", choices=("demo", "workbench"))
    parser.add_argument("--python", type=Path, help="Application virtual-environment Python")
    parser.add_argument("--port", type=port_number, help="HTTP port (demo: 8765; workbench: 8000)")
    parser.add_argument("--node-port", type=port_number, help="Demo harness port (default: 8766)")
    args = parser.parse_args()
    if args.app == "workbench" and args.node_port is not None:
        parser.error("--node-port is only supported by demo")
    app = ROOT / "apps" / ("agentic-rag" if args.app == "demo" else "workbench")
    environment = ".agent-venv" if args.app == "demo" else ".venv"
    executable = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    python = (args.python or app / environment / executable).resolve()
    if not python.is_file():
        parser.error(f"Python environment is missing: {python}. Follow docs/SETUP.md")
    env = os.environ.copy()
    if args.app == "demo":
        if os.name != "nt":
            parser.error("The demo launcher requires Windows; see docs/SETUP.md for offline checks")
        shell = shutil.which("powershell.exe") or shutil.which("pwsh")
        if shell is None:
            parser.error("PowerShell is missing; see docs/SETUP.md")
        port = args.port or 8765
        node_port = args.node_port or 8766
        if port == node_port:
            parser.error("Demo HTTP and Node ports must be different")
        env.update(FINRISK_AGENT_ALLOW_LIVE="0", FINRISK_AGENT_ALLOW_RETRIEVAL_LIVE="0")
        command = [
            shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
            str(app / "start-agent-rag.ps1"), "-PythonPath", str(python),
            "-PythonPort", str(port), "-NodePort", str(node_port),
        ]
    else:
        command = [str(python), str(app / "scripts/run_web.py"), "--port", str(args.port or 8000)]
    try:
        return subprocess.run(command, cwd=app, env=env, check=False).returncode
    except OSError as error:
        print(f"Cannot launch {args.app}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
