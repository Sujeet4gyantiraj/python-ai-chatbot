
# genai_service.py
# FastAPI GenAI service using Pinecone for RAG (logic unchanged)

import logging
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import re
import httpx
import asyncio
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from llm import generate_chat_response_qwen
from utils.prompt_builder import (
    build_augmented_system_instruction,
    format_prompt_for_llama3,
    get_memory
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

GENAI_API_BASE_URL = os.getenv('GENAI_API_BASE_URL')
EMBEDDING_DIM = 768


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

    async with httpx.AsyncClient(timeout=30) as client:
        res = await client.post(
            f"{GENAI_API_BASE_URL}/embed/",
            json={"text": text, "task_type": task_type}
        )

        if res.status_code != 200:
            logger.error(
                "Embedding API failed | status=%s | response=%s",
                res.status_code,
                res.text
            )
            raise Exception("Embedding API failed")

        data = res.json()
        logger.debug("Embedding generated successfully")
        return data["embedding"]


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
    prompt: str,
    max_tokens: int = 300,
    temperature: float = 0.1,
    top_p: float = 0.9,
) -> str:

    timeout = httpx.Timeout(
        connect=10.0,
        read=90.0,
        write=10.0,
        pool=10.0
    )

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            res = await client.post(
                f"{GENAI_API_BASE_URL}/generate/",
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    # "repetition_penalty":1.05,
                    "stop": [
                        "END",
                        "BEGIN KNOWLEDGE BASE",
                        "END KNOWLEDGE BASE",
                        "====================",
                        "QUESTION:",
                        "TASK:"
                       
                    ]
                },
            )

            if res.status_code != 200:
                logger.error(
                    "GENAI HTTP error | status=%s | body=%s",
                    res.status_code,
                    res.text
                )
                return (
                    "I'm sorry, I'm having trouble generating a response right now. "
                    "Please try again."
                )

            data = res.json()

            if data.get("error"):
                if data.get("safety") and "unsafe" in data["safety"]:
                    logger.warning("Safety violation detected by model")
                    return "I cannot respond to that request as it violates safety policies."

                logger.error("GENAI model error: %s", data)
                return (
                    "I'm sorry, I couldn't generate a response. "
                    "Please request human assistance."
                )

            logger.debug("LLM response generated successfully")
            return data.get("generated_text", "").strip()

        except httpx.ReadTimeout:
            logger.error("GENAI Read timeout from LLM")
            return (
                "The response is taking longer than expected. "
                "Please try again in a moment."
            )

        except asyncio.CancelledError:
            logger.warning("Request cancelled")
            raise

        except Exception as e:
            logger.exception("Unexpected GENAI error")
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
    ai_node_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Optional[str]]:

    logger.info(
        "New chat request | bot_id=%s | session_id=%s",
        bot_id,
        session_id
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

                pinecone_response = pinecone_index.query(
                    vector=query_embedding,
                    top_k=5,
                    include_metadata=True,
                    namespace=bot_id
                )

                logger.debug("Pinecone raw response: %s", pinecone_response)

                SIMILARITY_THRESHOLD = 0.55
                # breakpoint()
                if pinecone_response and pinecone_response.matches:
                    knowledge_base = "\n\n---\n\n".join(
                        match["metadata"]["content"]
                        for match in pinecone_response["matches"]
                        if (
                            match.get("metadata")
                            and match["metadata"].get("content")
                            and match.get("score", 0) >= SIMILARITY_THRESHOLD
                        )
                    )

                logger.debug("Knowledge base size=%d chars", len(knowledge_base))

            # ---------------- PROMPT ----------------
            history = await load_chat_history(bot_id, session_id, k=10)
            prompt_dict = build_augmented_system_instruction(
                user_query,
                knowledge_base,
                custom_instruction=""
            )

            system_prompt = prompt_dict["system_message"]["content"]
            max_tokens = prompt_dict.get("max_tokens", 300)
            action = prompt_dict.get("detected_intent")

            logger.debug("Detected intent: %s", action)
            logger.debug("Final system prompt prepared")

            messages = [
                prompt_dict["system_message"],
                *history,
                {"role": "user", "content": user_query}
            ]

            prompt = format_prompt_for_llama3(messages)

            # ---------------- LLM ----------------
            full_text = await generate_chat_response(
                prompt,
                max_tokens=max_tokens,
                temperature=0.2,
                top_p=0.9
            )

            # ---------------- POST PROCESS ----------------
            clean_text = clean_llm_output(full_text)
            # breakpoint()

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
                "action": action
            }

        except Exception:
            logger.exception("GENAI inner processing error")
            return {"fullText": "", "cleanText": "", "action": None}

    except Exception:
        logger.exception("GENAI outer error")
        return {"fullText": "", "cleanText": "", "action": None}
    



