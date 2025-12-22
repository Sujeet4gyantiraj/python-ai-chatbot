from fastapi import FastAPI

app = FastAPI(title="RAG API", version="1.0", root_path="/rag")


@app.get("/")
def hello():
    return {"message": "Hello from server"}
