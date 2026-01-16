
# genai_service.py
# FastAPI GenAI service using Pinecone for RAG (logic unchanged)

import logging
from typing import List, Optional, Dict, Any
import re
import asyncio
import os

from dotenv import load_dotenv
from pinecone import Pinecone

from genai_core import EmbeddingService, LLMService
from utils.prompt_builder import (
    build_augmented_system_instruction,
    format_prompt_for_llama3,
    # get_memory
)
from utils.redis_client import load_chat_history, save_chat_history

# ------------------------------------------------------------------
# LOGGING SETUP
# ------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
)

logger = logging.getLogger("genai_service")

# ------------------------------------------------------------------

load_dotenv()

# Pinecone Setup
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX")
pinecone_index = pc.Index(PINECONE_INDEX_NAME)

EMBEDDING_DIM = 768

# Local model services (loaded once per process)
_embedding_service = EmbeddingService()
_llm_service = LLMService()


def get_models_health() -> Dict[str, Any]:
    """Return a lightweight health summary for local LLM and embedding models."""
    status: Dict[str, Any] = {
        "llm": {
            "loaded": _llm_service is not None,
            "class": type(_llm_service).__name__ if _llm_service is not None else None,
        },
        "embedding": {
            "loaded": _embedding_service is not None,
            "class": type(_embedding_service).__name__ if _embedding_service is not None else None,
            "device": getattr(_embedding_service, "device", None) if _embedding_service is not None else None,
        },
    }

    return status


# ------------------------------------------------------------------
# Embedding API
# ------------------------------------------------------------------
async def generate_embedding(
    text: str,
    task_type: str = "search_query"
) -> List[float]:
    if not text or len(text.strip()) < 3:
        logger.warning("Embedding skipped due to short/empty text")
        return [0.0] * EMBEDDING_DIM
    loop = asyncio.get_running_loop()

    def _run_embedding() -> List[float]:
        emb = _embedding_service.generate(text=text, task_type=task_type)

        if isinstance(emb, list) and emb and isinstance(emb[0], list):
            emb_flat = emb[0]
        else:
            emb_flat = emb

        if not isinstance(emb_flat, list):
            raise TypeError("EmbeddingService returned invalid format")

        return [float(x) for x in emb_flat]

    try:
        emb = await loop.run_in_executor(None, _run_embedding)
        logger.debug("Embedding generated successfully | dim=%d", len(emb))
        return emb
    except Exception:
        logger.exception("Local embedding generation failed")
        raise


embed_query = generate_embedding



def clean_llm_output(text: str) -> str:
    # Remove any [ACTION:...] blocks
    text = re.sub(r"\[ACTION:.*?\]", "", text, flags=re.DOTALL)

    # Stop at markers like #, END RESPONSE, END CONVERSATION, etc.
    text = re.split(
        r"#|END CONVERSATION|END RESPONSE|END SESSION",
        text,
        maxsplit=1
    )[0]

    return text.strip()


# ------------------------------------------------------------------
# Chat Generation API
# ------------------------------------------------------------------
async def generate_chat_response(
    messages: list[dict],
    max_tokens: int = 300,
    temperature: float = 0.1,
    top_p: float = 0.9,
) -> str:
    loop = asyncio.get_running_loop()

    def _run_llm() -> str:
        try:
            result = _llm_service.generate_without_tools(
                messages_data=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            # result = _llm_service.generate_with_tools(
            #     messages_data=messages,
            #     max_tokens=max_tokens,
            #     temperature=temperature,
            #     top_p=top_p,
            # )
        except Exception as e:
            logger.error("Local LLM error: %s", e)
            raise

        text = (result or {}).get("generated_text", "")
        return text.strip()

    try:
        response_text = await loop.run_in_executor(None, _run_llm)
        if not response_text:
            return (
                "I'm sorry, I couldn't generate a response. "
                "Please request human assistance."
            )

        logger.debug("LLM response generated successfully")
        return response_text

    except asyncio.CancelledError:
        logger.warning("Request cancelled")
        raise
    except Exception:
        logger.exception("Unexpected local LLM error")
        return (
            "An unexpected error occurred. "
            "Please request human assistance."
        )


# ------------------------------------------------------------------
def create_chat_session(
    system_instruction: Dict[str, str],
    history: List[Dict[str, Any]]
) -> List[Dict[str, str]]:

    messages = [system_instruction]

    for h in history:
        messages.append({
            "role": h["role"],
            "content": "\n".join(p["text"] for p in h["parts"])
        })

    return messages


def extract_before_hash(text: str) -> str:
    match = re.match(r"^(.*?)\s*#", text)
    return match.group(1).strip() if match else text.strip()


# ------------------------------------------------------------------
# MAIN RAG FUNCTION
# ------------------------------------------------------------------
async def generate_and_stream_ai_response(
    bot_id: str,
    session_id: str,
    user_query: str,
    ai_node_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[str]]:

    logger.info(
        "New chat request | bot_id=%s | session_id=%s",
        bot_id,
        session_id,
    )

    try:
        full_text = ""
        clean_text = ""
        action = None

        try:
            # ---------------- RAG ----------------
            knowledge_base = ""

            if not ai_node_data or not ai_node_data.get("disableKnowledgeBase"):
                query_embedding = await embed_query(user_query)
                logger.debug("Pinecone query embedding generated")
                res = pinecone_index.query(
                    vector=query_embedding,
                    top_k=5,
                    include_metadata=True,
                    namespace=bot_id,
                )

                logger.debug("Pinecone raw response: %s", res)

                SIMILARITY_THRESHOLD = 0.55
                matches: list = []
                try:
                    if isinstance(res, dict):
                        matches = res.get("matches", []) or []
                    elif hasattr(res, "matches"):
                        matches = list(getattr(res, "matches") or [])
                    elif hasattr(res, "to_dict"):
                        d = res.to_dict()
                        if isinstance(d, dict):
                            matches = d.get("matches", []) or []
                except Exception:
                    matches = []

                if matches:
                    filtered_contents = []
                    for m in matches:
                        if isinstance(m, dict):
                            md = m.get("metadata", {}) or {}
                            content = md.get("content") if isinstance(md, dict) else None
                            score = m.get("score", 0.0) or 0.0
                        else:
                            md = getattr(m, "metadata", {}) or {}
                            content = md.get("content") if isinstance(md, dict) else None
                            score = getattr(m, "score", 0.0) or 0.0
                        if content and score >= SIMILARITY_THRESHOLD:
                            filtered_contents.append(content)

                    if filtered_contents:
                        knowledge_base = "\n\n---\n\n".join(filtered_contents)

                logger.debug("Knowledge base size=%d chars", len(knowledge_base))

            # ---------------- PROMPT ----------------
           
            history = await load_chat_history(bot_id, session_id, k=10)
            prompt_dict = build_augmented_system_instruction(
                user_message=user_query,
                knowledge_base=knowledge_base,
                custom_instruction="",
            )

            max_tokens = prompt_dict.get("max_tokens", 300)
            action = prompt_dict.get("detected_intent")

            logger.debug("Detected intent: %s", action)
            logger.debug("Final system prompt prepared")

            messages = [
                prompt_dict["system_message"],
                *history,
                {"role": "user", "content": user_query},
            ]

            # ---------------- LLM ----------------
            full_text = await generate_chat_response(
                messages,
                max_tokens=max_tokens,
                temperature=0.2,
                top_p=0.9,
            )

            # ---------------- POST PROCESS ----------------
            clean_text = clean_llm_output(full_text)

            if action == "greeting":
                clean_text = extract_before_hash(clean_text)

            # ---------------- SAVE HISTORY ----------------
            history.append({"role": "user", "content": user_query})
            history.append({"role": "assistant", "content": clean_text})
            await save_chat_history(bot_id, session_id, history, k=10)

            logger.info("Chat response generated successfully")

            return {
                "fullText": full_text,
                "cleanText": clean_text,
                "action": action,
            }

        except Exception:
            logger.exception("GENAI inner processing error")
            return {"fullText": "", "cleanText": "", "action": None}

    except Exception:
        logger.exception("GENAI outer error")
        return {"fullText": "", "cleanText": "", "action": None}






