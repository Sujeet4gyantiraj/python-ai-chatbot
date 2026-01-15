import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import List, Optional

from genai_core import EmbeddingService
logger = logging.getLogger(__name__)

_embedding_service = EmbeddingService()

_embedding_executor = ThreadPoolExecutor(max_workers=1)

EMBED_BATCH_SIZE = 128  # increase gradually


async def embed_content_batch(chunks: List[str]) -> List[Optional[List[float]]]:
    results = [None] * len(chunks)
    loop = asyncio.get_running_loop()

    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch_indices = []
        batch_texts = []

        for i in range(start, min(start + EMBED_BATCH_SIZE, len(chunks))):
            text = chunks[i].strip()
            if text:
                batch_indices.append(i)
                batch_texts.append(text)

        if not batch_texts:
            continue

        embeddings = await loop.run_in_executor(
            _embedding_executor,
            _embedding_service.generate_batch,
            batch_texts,
            "search_document",
        )

        for idx, emb in zip(batch_indices, embeddings):
            results[idx] = emb

    return results