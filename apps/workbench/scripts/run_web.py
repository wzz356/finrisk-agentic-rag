"""加载项目环境变量并启动 FinRisk Web 工作台。"""

import argparse
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """只从项目根目录加载 .env，避免依赖调用者的当前目录。"""

    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description="启动本地 FinRisk 工作台")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(
        "api.app:app",
        host="127.0.0.1",
        port=args.port,
        reload=False,
        app_dir=str(PROJECT_ROOT / "src"),
    )


if __name__ == "__main__":
    main()
