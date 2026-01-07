from fastapi import FastAPI

app = FastAPI(title="AI Customer Support Bot")


@app.get("/health")
def health():
    return {"ok": True}
