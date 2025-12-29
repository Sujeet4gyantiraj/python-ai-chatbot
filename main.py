import os
import uuid
import aiofiles
import aiohttp
from dotenv import load_dotenv
import io
import pdfplumber
import docx
import csv
from fastapi import UploadFile, HTTPException
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Depends,
    HTTPException,
    APIRouter,
)
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import select

import cuid
import re
from fastapi import status
from pinecone_client import index
from utils.chunking import extract_chunks, extract_chunks_from_text

from genai_service import (
    generate_and_stream_ai_response,
    GenerateAIRequest,
)

load_dotenv()

# -------------------------
# App & Router
# -------------------------
app = FastAPI(title="RAG API", version="1.0")
router = APIRouter()

MAX_TOTAL_SIZE = 10 * 1024 * 1024
GENAI_API_BASE_URL = os.getenv("GENAI_API_BASE_URL")




# read document of different types
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".csv", ".txt"}
def read_upload_file(filename: str, content: bytes) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}"
        )
    if ext == ".pdf":
        text = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        return "\n".join(text)
    elif ext in {".docx", ".doc"}:
        doc = docx.Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    elif ext == ".csv":
        decoded = content.decode("utf-8", errors="ignore")
        reader = csv.reader(decoded.splitlines())
        rows = [" ".join(row) for row in reader]
        return "\n".join(rows)
    elif ext == ".txt":
        return content.decode("utf-8", errors="ignore")
    raise HTTPException(status_code=400, detail="File reading failed")





def clean_chunk(text: str) -> str:
    text = re.sub(r'%PDF.*', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()



# =========================
# EMBEDDING (ASYNC OK)
# =========================
async def embed_content_batch(chunks: list[str]) -> list[list[float]]:
    embeddings = []

    async with aiohttp.ClientSession() as session:
        for chunk in chunks:
            async with session.post(
                f"{GENAI_API_BASE_URL}/embed/",
                json={"text": chunk, "task_type": "search_document"},
            ) as resp:
                if resp.status != 200:
                    raise Exception(await resp.text())
                data = await resp.json()
                embeddings.append(data["embedding"])

    return embeddings


# =========================
# UPLOAD KNOWLEDGE
# =========================
@app.post("/bots/{bot_id}/knowledge/upload")
async def upload_knowledge(
    knowledge_source_id: str,
    bot_id: str,
    files: list[UploadFile] = File(...),
    
):

    total_size = 0
    results = []
    try:
        for file in files:
            content = await file.read()
            total_size += len(content)
            if total_size > MAX_TOTAL_SIZE:
                raise HTTPException(status_code=400, detail="Upload exceeds 10MB")

            temp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
            storage_path = f"knowledge/{bot_id}/{uuid.uuid4()}_{file.filename}"

            async with aiofiles.open(temp_path, "wb") as f:
                await f.write(content)

            # Read file as text according to type
            text = read_upload_file(file.filename, content)

            # Chunk the extracted text, not the file
            chunks = extract_chunks_from_text(text)
            print("Extracted chunks:", len(chunks))

            # Embedding and upsert (optional, keep if you want Pinecone)
            chunks = [clean_chunk(c) for c in chunks]
            chunks = [c.strip() for c in chunks if c.strip()]
            chunks = [c[:1000] for c in chunks if len(c) > 50]
            embeddings = await embed_content_batch(chunks)
            for chunk, emb in zip(chunks, embeddings):
                print("Chunk:", chunk[:60], "Embedding length:", len(emb))
            print("Embeddings received:", len(embeddings))
            vectors = [
                {
                    "id": str(uuid.uuid4()),
                    "values": emb,
                    "metadata": {
                        "sourceId": knowledge_source_id,
                        "botId": bot_id,
                        "fileName": file.filename,
                        "storagePath": storage_path,
                        "fileType": file.content_type,
                        "content": chunk,
                    },
                }
                for chunk, emb in zip(chunks, embeddings)
            ]

            index.upsert(
                vectors=vectors,
                namespace=bot_id
            )

            os.remove(temp_path)

            results.append({
                "fileName": file.filename,
                "storagePath": storage_path,
                "fileType": file.content_type,
                "botId": bot_id
            })

        return {"uploaded": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





# =========================
# GENAI RAG ENDPOINT
# =========================
@router.post("/generate-ai-response")
async def generate_ai_response_endpoint(req: GenerateAIRequest):
    result = await generate_and_stream_ai_response(
        bot_id=req.bot_id,
        session_id=req.session_id,
        last_user_message=req.last_user_message,
        ai_node_data=req.ai_node_data,
    )
    return JSONResponse(content=result)


# REGISTER ROUTER
app.include_router(router)


# =========================
# DELETE KNOWLEDGE FROM PINECONE BY SOURCE ID
# =========================


@app.delete("/knowledge/pinecone/{source_id}", status_code=status.HTTP_200_OK)
def delete_knowledge_from_pinecone(source_id: str, bot_id: str):
    """
    Delete all vectors in Pinecone with metadata sourceId == source_id for the given bot_id namespace.
    """
    try:
        # Pinecone delete by metadata filter
        index.delete(
            filter={"sourceId": source_id},
            namespace=bot_id
        )
        return {"deleted": True, "source_id": source_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone delete failed: {str(e)}")

