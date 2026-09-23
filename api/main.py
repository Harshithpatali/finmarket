from fastapi import FastAPI
from api.routes.analysis import router as analysis_router

app = FastAPI(
    title="FinMarket AI",
    version="0.1.0",
    description="Global stock modeling + financial RAG research API",
)

app.include_router(analysis_router, prefix="/api")

@app.get("/health")
def health():
    return {"status": "ok"}
