from typing import List
import threading
import torch
import logging
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)



class EmbeddingService:
    def __init__(self, model_id: str = "nomic-ai/nomic-embed-text-v1.5"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = HuggingFaceEmbeddings(
            model_name=model_id,
            model_kwargs={
                "trust_remote_code": True,
                "device": self.device,
            },
        )

        self.prompts = {
            "search_query": "search_query: ",
            "search_document": "search_document: ",
        }

        self._lock = threading.Lock()

    def generate_batch(self, texts: List[str], task_type: str) -> List[List[float]]:
        prefix = self.prompts[task_type]
        prefixed = [prefix + t.strip() for t in texts]

        with self._lock:
            embeddings = self.model.embed_documents(prefixed)

        return embeddings

    def generate(self, text: str, task_type: str = "search_query") -> List[float]:
        """Generate an embedding for a single text.

        Convenience wrapper around generate_batch used by query-time code.
        """

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string")

        batch = self.generate_batch([text], task_type)
        if not batch or not isinstance(batch[0], list):
            raise RuntimeError("EmbeddingService.generate_batch returned invalid format")

        # Ensure plain list[float]
        return [float(x) for x in batch[0]]
