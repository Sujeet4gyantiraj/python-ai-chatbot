import asyncio
import aiohttp
import os
import logging
from typing import List
from fastapi import HTTPException


# -------------------- LOGGER SETUP --------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
)

logger = logging.getLogger(__name__)
# ----------------------------------------------------


GENAI_API_BASE_URL = os.getenv("GENAI_API_BASE_URL")


async def embed_content_batch(chunks: List[str]) -> List[List[float]]:
    timeout = aiohttp.ClientTimeout(total=60)

    logger.info("Embedding batch started | chunks=%d", len(chunks))

    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def fetch_embedding(chunk: str):
            logger.debug("Embedding chunk | length=%d", len(chunk))

            async with session.post(
                f"{GENAI_API_BASE_URL}/embed/",
                json={"text": chunk, "task_type": "search_document"},
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(
                        "Embedding API failed | status=%s | response=%s",
                        resp.status,
                        error_text,
                    )
                    raise RuntimeError(error_text)

                data = await resp.json()
                logger.debug("Embedding received | vector_size=%d", len(data.get("embedding", [])))
                return data["embedding"]

        try:
            result = await asyncio.gather(
                *[fetch_embedding(chunk) for chunk in chunks],
                return_exceptions=False
            )
            logger.info("Embedding batch completed successfully | chunks=%d", len(chunks))
            return result

        except Exception as e:
            logger.exception("Embedding batch failed")
            raise

