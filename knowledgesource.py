import os
import uuid

import string

import asyncio
import uuid
import aiofiles
import aiohttp
import io
import re
import string
import fitz  # PyMuPDF
import docx
import pandas as pd
from fastapi import HTTPException, UploadFile, File
from pinecone_client import index
from utils.chunking import extract_chunks_from_text
from embedding import embed_content_batch


MAX_TOTAL_SIZE = 10 * 1024 * 1024


# read document of different types


import asyncio
from typing import Callable, Any, List

async def run_in_thread(func: Callable[..., Any], *args, **kwargs) -> Any:
    return await asyncio.to_thread(func, *args, **kwargs)



ALLOWED_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xlsx", ".xls", ".csv", ".txt"
}


import asyncio


def read_pdf(content: bytes) -> str:
    text = []
    with fitz.open(stream=content, filetype="pdf") as pdf:
        for page in pdf:
            t = page.get_text("text")
            if t:
                text.append(t)
    return "\n".join(text)

def read_docx(content: bytes) -> str:
    doc = docx.Document(io.BytesIO(content))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def read_excel(content: bytes) -> str:
    excel = pd.read_excel(io.BytesIO(content), sheet_name=None)
    text = []
    for sheet, df in excel.items():
        text.append(f"Sheet: {sheet}")
        text.append(df.astype(str).fillna("").to_csv(index=False))
    return "\n".join(text)

def read_csv(content: bytes) -> str:
    df = pd.read_csv(io.BytesIO(content))
    return df.astype(str).fillna("").to_csv(index=False)



async def read_upload_file(filename: str, content: bytes) -> str:
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    if ext == ".pdf":
        return await run_in_thread(read_pdf, content)

    if ext in {".doc", ".docx"}:
        return await run_in_thread(read_docx, content)

    if ext in {".xls", ".xlsx"}:
        return await run_in_thread(read_excel, content)

    if ext == ".csv":
        return await run_in_thread(read_csv, content)

    if ext == ".txt":
        return content.decode("utf-8", errors="ignore")

    raise HTTPException(status_code=400, detail="File reading failed")




def clean_chunk(text: str) -> str:
    try:
        if not text:
            return ""

        # Remove PDF binary junk
        text = re.sub(r'%PDF-.*', '', text)
        text = re.sub(r'\x00|\x01|\x02|\x03|\x04|\x05|\x06|\x07', '', text)

        # Remove HTML/XML tags
        text = re.sub(r'<[^>]+>', ' ', text)

        # Remove repeated symbols (----, _____, ****)
        text = re.sub(r'[_\-*=]{3,}', ' ', text)

        # Remove non-printable characters
        text = ''.join(ch for ch in text if ch in string.printable)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chunk clean error: {str(e)}")



import uuid
import asyncio
import logging
from typing import List

logger = logging.getLogger(__name__)

PINECONE_UPSERT_BATCH_SIZE = 100


async def background_embedding_job(
    cleaned_chunks: List[str],
    knowledge_source_id: str,
    bot_id: str,
    file_name: str,
):
    embeddings = await embed_content_batch(cleaned_chunks)

    vectors = []
    for idx, (chunk, emb) in enumerate(zip(cleaned_chunks, embeddings)):
        if emb is None:
            continue

        vectors.append(
            {
                "id": str(uuid.uuid4()),
                "values": emb,
                "metadata": {
                    "sourceId": knowledge_source_id,
                    "botId": bot_id,
                    "fileName": file_name,
                    "chunkIndex": idx,
                    # truncate to avoid metadata limits
                    "content": chunk[:1000],
                },
            }
        )

    failed = len(cleaned_chunks) - len(vectors)

    if not vectors:
        logger.warning(
            "No vectors to upsert | file=%s failed=%d",
            file_name,
            failed,
        )
        return

    try:
        for i in range(0, len(vectors), PINECONE_UPSERT_BATCH_SIZE):
            batch = vectors[i : i + PINECONE_UPSERT_BATCH_SIZE]
            await asyncio.to_thread(
                index.upsert,
                vectors=batch,
                namespace=bot_id,
            )
    except Exception as e:
        logger.error(
            "Pinecone upsert failed | file=%s error=%s",
            file_name,
            str(e),
        )
        raise

    logger.info(
        "Embedding job completed | file=%s success=%d failed=%d",
        file_name,
        len(vectors),
        failed,
    )


async def upload_knowledge(
    knowledge_source_id: str,
    bot_id: str,
    files: list[UploadFile] = File(...),
):
    total_size = 0
    results = []
    tasks: list[asyncio.Future] = []

    for file in files:
        content = await file.read()
        total_size += len(content)

        if total_size > MAX_TOTAL_SIZE:
            raise HTTPException(status_code=400, detail="Upload exceeds 10MB")

        text = await read_upload_file(file.filename, content)
        chunks = extract_chunks_from_text(text)

        cleaned_chunks = await asyncio.gather(
            *[asyncio.to_thread(clean_chunk, c) for c in chunks]
        )
        cleaned_chunks = [
            c.strip() for c in cleaned_chunks if c.strip()
        ]

        # Queue embedding + Pinecone upsert; all tasks awaited after loop
        tasks.append(
            background_embedding_job(
                cleaned_chunks,
                knowledge_source_id,
                bot_id,
                file.filename,
            )
        )

        results.append({
            "fileName": file.filename,
            "status": "completed",
            "chunks": len(cleaned_chunks)
        })

    # Run all embedding jobs concurrently and wait for completion
    if tasks:
        await asyncio.gather(*tasks)

    return {
        "message": "Files uploaded and embedded successfully.",
        "files": results
    }

