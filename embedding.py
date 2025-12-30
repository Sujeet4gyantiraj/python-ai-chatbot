import asyncio
import aiohttp
import os
from typing import List
from fastapi import HTTPException

GENAI_API_BASE_URL = os.getenv("GENAI_API_BASE_URL")

async def embed_content_batch(chunks: List[str]) -> List[List[float]]:
    async def fetch_embedding(session, chunk):
        try:
            async with session.post(
                f"{GENAI_API_BASE_URL}/embed/",
                json={"text": chunk, "task_type": "search_document"},
            ) as resp:
                if resp.status != 200:
                    raise HTTPException(status_code=500, detail=await resp.text())
                data = await resp.json()
                return data["embedding"]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")

    try:
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_embedding(session, chunk) for chunk in chunks]
            embeddings = await asyncio.gather(*tasks)
        return embeddings
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding batch error: {str(e)}")
