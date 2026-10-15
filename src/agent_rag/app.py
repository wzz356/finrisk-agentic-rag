"""轻量独立入口；也可通过api.app使用同一router。"""

from fastapi import FastAPI

from agent_rag.router import router

app = FastAPI(title="FinRisk智能体化RAG Baseline", version="0.1.0")
app.include_router(router)
