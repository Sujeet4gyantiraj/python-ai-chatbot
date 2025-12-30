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



ALLOWED_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xlsx", ".xls", ".csv", ".txt"
}

import aiofiles.tempfile
import asyncio

async def read_upload_file(filename: str, content: bytes) -> str:
    """
    Reads different document types and returns extracted text. Now async.
    """
    try:
        ext = os.path.splitext(filename)[1].lower()

        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}"
            )

        # For file types that require a file path, use aiofiles.tempfile
        if ext == ".pdf":
            # PyMuPDF requires a file-like object or bytes, so we can use bytes directly
            text = []
            with fitz.open(stream=content, filetype="pdf") as pdf:
                for page in pdf:
                    page_text = page.get_text("text")
                    if page_text:
                        text.append(page_text)
            return "\n".join(text)

        elif ext in {".doc", ".docx"}:
            # docx.Document can use BytesIO
            doc = docx.Document(io.BytesIO(content))
            return "\n".join(
                p.text for p in doc.paragraphs if p.text.strip()
            )

        elif ext in {".xls", ".xlsx"}:
            # pandas can use BytesIO
            excel = pd.read_excel(io.BytesIO(content), sheet_name=None)
            text = []
            for sheet_name, df in excel.items():
                text.append(f"Sheet: {sheet_name}")
                text.append(
                    df.astype(str).fillna("").to_csv(
                        index=False, header=True
                    )
                )
            return "\n".join(text)

        elif ext == ".csv":
            df = pd.read_csv(io.BytesIO(content))
            return df.astype(str).fillna("").to_csv(index=False)

        elif ext == ".txt":
            # For large files, use aiofiles with a temp file
            async with aiofiles.tempfile.NamedTemporaryFile("wb+", delete=True) as tmp:
                await tmp.write(content)
                await tmp.seek(0)
                # Read as text
                result = await tmp.read()
                return result.decode("utf-8", errors="ignore")

        raise HTTPException(status_code=400, detail="File reading failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File read error: {str(e)}")




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

            # Read file as text according to type (async)
            text = await read_upload_file(file.filename, content)

            # Chunk the extracted text, not the file
            chunks = extract_chunks_from_text(text)
            print("Extracted chunks:", len(chunks))

            # Embedding and upsert (optional, keep if you want Pinecone)
            # Use asyncio.to_thread for parallelism if needed
            loop = asyncio.get_event_loop()
            cleaned_chunks = await asyncio.gather(*[
                loop.run_in_executor(None, clean_chunk, c) for c in chunks
            ])
            cleaned_chunks = [c.strip() for c in cleaned_chunks if c.strip()]
            cleaned_chunks = [c[:1000] for c in cleaned_chunks if len(c) > 50]
            embeddings = await embed_content_batch(cleaned_chunks)
            for chunk, emb in zip(cleaned_chunks, embeddings):
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
                        "content": chunk,
                    },
                }
                for chunk, emb in zip(cleaned_chunks, embeddings)
            ]

            index.upsert(
                vectors=vectors,
                namespace=bot_id
            )

            os.remove(temp_path)

            results.append({
                "fileName": file.filename,
                "botId": bot_id
            })

        return {"uploaded": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
