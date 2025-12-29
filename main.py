# import os
# import uuid
# import aiofiles
# import aiohttp
# from dotenv import load_dotenv
# from fastapi import (
#     FastAPI,
#     UploadFile,
#     File,
#     Depends,
#     HTTPException,
#     APIRouter,
# )
# from fastapi.responses import JSONResponse
# from sqlalchemy.ext.asyncio import AsyncSession
# from sqlalchemy import select

# import cuid

# from db import get_db
# from models import Bot, KnowledgeSource
# from pinecone_client import index
# from utils.chunking import extract_chunks

# # ✅ IMPORT FROM genai_service
# from genai_service import (
#     generate_and_stream_ai_response,
#     GenerateAIRequest,
# )

# load_dotenv()

# # -------------------------
# # App & Router
# # -------------------------
# app = FastAPI(title="RAG API", version="1.0")
# router = APIRouter()

# MAX_TOTAL_SIZE = 10 * 1024 * 1024
# GENAI_API_BASE_URL = os.getenv("GENAI_API_BASE_URL")


# # =========================
# # EMBEDDING
# # =========================
# async def embed_content_batch(chunks: list[str]) -> list[list[float]]:
#     embeddings = []

#     async with aiohttp.ClientSession() as session:
#         for chunk in chunks:
#             async with session.post(
#                 f"{GENAI_API_BASE_URL}/embed/",
#                 json={"text": chunk, "task_type": "search_document"},
#             ) as resp:
#                 if resp.status != 200:
#                     raise Exception(await resp.text())
#                 data = await resp.json()
#                 embeddings.append(data["embedding"])

#     return embeddings


# # =========================
# # UPLOAD KNOWLEDGE
# # =========================
# @app.post("/bots/{bot_id}/knowledge/upload")
# async def upload_knowledge(
#     bot_id: str,
#     files: list[UploadFile] = File(...),
#     db: AsyncSession = Depends(get_db),
# ):
#     # ✅ Async-safe query
#     result = await db.execute(select(Bot).where(Bot.id == bot_id))
#     bot = result.scalar_one_or_none()

#     if not bot:
#         raise HTTPException(status_code=404, detail="Bot not found")

#     total_size = 0

#     try:
#         for file in files:
#             content = await file.read()
#             total_size += len(content)

#             if total_size > MAX_TOTAL_SIZE:
#                 raise HTTPException(status_code=400, detail="Upload exceeds 10MB")

#             temp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
#             storage_path = f"knowledge/{bot_id}/{uuid.uuid4()}_{file.filename}"

#             async with aiofiles.open(temp_path, "wb") as f:
#                 await f.write(content)

#             chunks = await extract_chunks(temp_path)

#             source = KnowledgeSource(
#                 id=cuid.cuid(),
#                 fileName=file.filename,
#                 storagePath=storage_path,
#                 fileType=file.content_type,
#                 botId=bot_id,
#             )

#             db.add(source)
#             await db.flush()  # ✅ REQUIRED to get source.id

#             embeddings = await embed_content_batch(chunks)

#             vectors = [
#                 {
#                     "id": str(uuid.uuid4()),
#                     "values": emb,
#                     "metadata": {
#                         "botId": bot_id,
#                         "sourceId": source.id,
#                         "content": chunk,
#                     },
#                 }
#                 for chunk, emb in zip(chunks, embeddings)
#                 if chunk.strip()
#             ]

#             # index.upsert(vectors=vectors)
#             index.upsert(
# 					vectors=vectors,
# 					namespace=bot_id
# 				)
#             os.remove(temp_path)

#         await db.commit()
#         return {"status": "success"}

#     except Exception as e:
#         await db.rollback()
#         raise HTTPException(status_code=500, detail=str(e))


# # =========================
# # LIST KNOWLEDGE
# # =========================
# @app.get("/bots/{bot_id}/knowledge")
# async def list_sources(
#     bot_id: str,
#     db: AsyncSession = Depends(get_db),
# ):
#     result = await db.execute(
#         select(KnowledgeSource)
#         .where(KnowledgeSource.botId == bot_id)
#         .order_by(KnowledgeSource.createdAt.desc())
#     )
#     return result.scalars().all()


# # =========================
# # DELETE KNOWLEDGE
# # =========================
# # @app.delete("/knowledge/{source_id}")
# # async def delete_source(
# #     source_id: str,
# #     db: AsyncSession = Depends(get_db),
# # ):
# #     result = await db.execute(
# #         select(KnowledgeSource).where(KnowledgeSource.id == source_id)
# #     )
# #     source = result.scalar_one_or_none()

# #     if not source:
# #         raise HTTPException(status_code=404, detail="Not found")

# #     index.delete(filter={"sourceId": source.id})
# #     await db.delete(source)
# #     await db.commit()

# #     return {"deleted": True}


# @app.delete("/knowledge/{source_id}")
# async def delete_source(
#     source_id: str,
#     db: AsyncSession = Depends(get_db),
# ):
#     result = await db.execute(
#         select(KnowledgeSource).where(KnowledgeSource.id == source_id)
#     )
#     source = result.scalar_one_or_none()

#     if not source:
#         raise HTTPException(status_code=404, detail="Knowledge source not found")

#     try:
#         # Pinecone delete (sync, but OK)
#         index.delete(filter={"sourceId": source.id})

#         # DB delete
#         await db.delete(source)
#         await db.commit()

#     except Exception as e:
#         await db.rollback()
#         raise HTTPException(
#             status_code=500,
#             detail=f"Delete failed: {str(e)}"
#         )

#     return {"deleted": True, "source_id": source.id}

# # =========================
# # GENAI RAG ENDPOINT
# # =========================
# @router.post("/generate-ai-response")
# async def generate_ai_response_endpoint(req: GenerateAIRequest):
#     result = await generate_and_stream_ai_response(
#         session_id=req.session_id,
#         last_user_message=req.last_user_message,
#         ai_node_data=req.ai_node_data,
#     )
#     return JSONResponse(content=result)


# # ✅ REGISTER ROUTER
# app.include_router(router)



import os
import uuid
import aiofiles
import aiohttp
from dotenv import load_dotenv
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

from db import get_db
from models import Bot, KnowledgeSource
from pinecone_client import index
from utils.chunking import extract_chunks

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

import io
import pdfplumber
import docx
import csv
from fastapi import UploadFile, HTTPException

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".csv", ".txt"}
async def read_upload_file(file: UploadFile) -> str:
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}"
        )

    content = await file.read()  # 🔥 READ ONCE, AT START

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
    bot_id: str,
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    bot = db.execute(
        select(Bot).where(Bot.id == bot_id)
    ).scalar_one_or_none()

    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")

    total_size = 0

    try:
        for file in files:
            content = await file.read()
            total_size += len(content)
            breakpoint()
            if total_size > MAX_TOTAL_SIZE:
                raise HTTPException(status_code=400, detail="Upload exceeds 10MB")

            temp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
            storage_path = f"knowledge/{bot_id}/{uuid.uuid4()}_{file.filename}"

            async with aiofiles.open(temp_path, "wb") as f:
                await f.write(content)
            text = await read_upload_file(file)
            chunks = await extract_chunks(temp_path)
            print("Extracted chunksssssssssssssssssssssssssssssssssssssssssssssssssssssssss:", len(chunks))
            source = KnowledgeSource(
                id=cuid.cuid(),
                fileName=file.filename,
                storagePath=storage_path,
                fileType=file.content_type,
                botId=bot_id,
            )

            db.add(source)
            db.flush()  # sync flush
            chunks = [clean_chunk(c) for c in chunks]
            chunks = [c.strip() for c in chunks if c.strip()]
            chunks = [c[:1000] for c in chunks if len(c) > 50]
            embeddings = await embed_content_batch(chunks)
            print("Embeddings receiveddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd:", len(embeddings))
            vectors = [
                {
                    "id": str(uuid.uuid4()),
                    "values": emb,
                    "metadata": {
                        "botId": bot_id,
                        "sourceId": source.id,
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

        db.commit()
        return {"status": "success"}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# LIST KNOWLEDGE
# =========================
@app.get("/bots/{bot_id}/knowledge")
def list_sources(
    bot_id: str,
    db: Session = Depends(get_db),
):
    return db.execute(
        select(KnowledgeSource)
        .where(KnowledgeSource.botId == bot_id)
        .order_by(KnowledgeSource.createdAt.desc())
    ).scalars().all()


# =========================
# DELETE KNOWLEDGE
# =========================
@app.delete("/knowledge/{source_id}")
def delete_source(
    source_id: str,
    db: Session = Depends(get_db),
):
    source = db.execute(
        select(KnowledgeSource).where(KnowledgeSource.id == source_id)
    ).scalar_one_or_none()

    if not source:
        raise HTTPException(status_code=404, detail="Knowledge source not found")

    try:
        index.delete(
            filter={"sourceId": source.id},
            namespace=source.botId
        )

        db.delete(source)
        db.commit()

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Delete failed: {str(e)}"
        )

    return {"deleted": True, "source_id": source.id}


# =========================
# GENAI RAG ENDPOINT
# =========================
@router.post("/generate-ai-response")
async def generate_ai_response_endpoint(req: GenerateAIRequest):
    result = await generate_and_stream_ai_response(
        session_id=req.session_id,
        last_user_message=req.last_user_message,
        ai_node_data=req.ai_node_data,
    )
    return JSONResponse(content=result)


# REGISTER ROUTER
app.include_router(router)

