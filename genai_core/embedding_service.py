from . import hf_config  # noqa: F401
import torch
from typing import Union, List
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(self, model_id: str = "nomic-ai/nomic-embed-text-v1.5"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SentenceTransformer on {self.device}...")
        self.model = SentenceTransformer(
            model_id,
            trust_remote_code=True,
            device=self.device,
        )
        self.prompts = {
            "search_query": "search_query: ",
            "search_document": "search_document: ",
        }

    def generate(
        self,
        text: Union[str, List[str]],
        task_type: str = "search_query",
        batch_size: int = 32,
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for single text or batch of texts."""
        if not text or (isinstance(text, list) and len(text) == 0):
            raise ValueError("Input text cannot be empty")

        prefix = self.prompts.get(task_type, self.prompts["search_query"])

        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        texts_with_prefix = [prefix + t for t in texts]

        embeddings = self.model.encode(
            texts_with_prefix,
            batch_size=batch_size,
            convert_to_tensor=False,
            normalize_embeddings=True,
            show_progress_bar=len(texts_with_prefix) > 100,
        )

        result = embeddings.tolist()

        return result[0] if is_single else result
