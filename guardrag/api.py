from fastapi import FastAPI

from guardrag.db import connect

app = FastAPI(title="GuardRAG")


@app.get("/health")
def health() -> dict[str, object]:
    with connect() as conn:
        passages = conn.execute("SELECT count(*) FROM passages").fetchone()[0]
    return {"status": "ok", "passages": passages}
