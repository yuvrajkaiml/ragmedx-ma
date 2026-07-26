from fastapi import FastAPI

from app.api.router import router

app = FastAPI(
    title="RAGMedX-MA",
    version="0.1.0",
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "message": "RAGMedX-MA Backend Running"
    }