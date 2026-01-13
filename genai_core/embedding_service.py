
from . import hf_config  # noqa: F401
from typing import Union, List
from langchain_community.embeddings import HuggingFaceEmbeddings



class EmbeddingService:
    def __init__(self, model_id: str = "nomic-ai/nomic-embed-text-v1.5"):
        print(f"Loading LangChain HuggingFaceEmbeddings: {model_id} ...")
        self.model_id = model_id
        self.model = HuggingFaceEmbeddings(model_name=model_id, model_kwargs={"trust_remote_code": True})
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
        """Generate embeddings for single text or batch of texts using LangChain."""
        if not text or (isinstance(text, list) and len(text) == 0):
            raise ValueError("Input text cannot be empty")

        prefix = self.prompts.get(task_type, self.prompts["search_query"])
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        texts_with_prefix = [prefix + t for t in texts]

        # LangChain's embed_documents returns List[List[float]]
        embeddings = self.model.embed_documents(texts_with_prefix)

        return embeddings[0] if is_single else embeddings
