import asyncio
import os
import logging
from typing import List

from genai_core import EmbeddingService


# -------------------- LOGGER SETUP --------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
)

logger = logging.getLogger(__name__)
# ----------------------------------------------------


_embedding_service = EmbeddingService()


async def embed_content_batch(chunks: List[str]) -> List[List[float]]:
    logger.info("Embedding batch started | chunks=%d", len(chunks))

    if not chunks:
        return []

    loop = asyncio.get_running_loop()
    BATCH_SIZE = 16

    def _run_batch(batch_chunks: List[str]) -> List[List[float]]:
        emb = _embedding_service.generate(text=batch_chunks, task_type="search_document")

        # EmbeddingService returns list[list[float]] for batched input
        if not isinstance(emb, list):
            raise RuntimeError("EmbeddingService returned invalid format")

        if emb and not isinstance(emb[0], list):
            # Single embedding returned for some reason
            emb = [emb]

        return [[float(x) for x in row] for row in emb]

    try:
        tasks = []
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            tasks.append(loop.run_in_executor(None, _run_batch, batch))

        all_results: List[List[float]] = []
        if tasks:
            batch_results = await asyncio.gather(*tasks, return_exceptions=False)
            for br in batch_results:
                all_results.extend(br)

        logger.info("Embedding batch completed successfully | chunks=%d", len(chunks))
        return all_results

    except Exception:
        logger.exception("Embedding batch failed")
        raise

