
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
        """Post-process raw LLM output.

        - Strip any internal action markers (e.g. ``[ACTION:REQUEST_AGENT]``
            or ``ACTION:\nSHOW_SCHEDULER``) so they are not shown to the user.
        - Stop at explicit end markers like ``END RESPONSE`` but **do not**
            treat a plain ``#`` as an end marker.
        """

        # If an ACTION marker appears, keep only the text before it.
        # This covers both "[ACTION:XYZ]" and multi-line forms like
        # "ACTION:\nXYZ".
        text = re.split(r"\[?ACTION:", text, maxsplit=1)[0]

        # Also strip any leftover [ACTION:...] blocks just in case
        text = re.sub(r"\[ACTION:.*?\]", "", text, flags=re.DOTALL)

        # Stop only at explicit textual end markers, not on every '#'
        text = re.split(
                r"END CONVERSATION|END RESPONSE|END SESSION",
                text,
                maxsplit=1,
        )[0]

        return text.strip()


# ------------------------------------------------------------------
# Chat Generation API
# ------------------------------------------------------------------
async def generate_chat_response(
    messages: list[dict],
    max_tokens: int = 2000,
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
    tenant_name: Optional[str] = None,
    tenant_description: Optional[str] = None,
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
                    top_k=8,
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
            # breakpoint()

            # Build an optional tenant-specific instruction so the LLM
            # understands which tenant (brand/customer) it is answering for.
            tenant_custom_instruction: Optional[str] = None
            if tenant_name or tenant_description:
                lines: list[str] = [
                    "You are a professional customer support assistant known for clear, accurate, and helpful responses.",
                ]

                lines.append("")
                lines.append("TENANT CONTEXT:")
                if tenant_name:
                    lines.append(f"- Tenant name: {tenant_name}")
                if tenant_description:
                    lines.append(f"- Tenant description: {tenant_description}")

                tenant_custom_instruction = "\n".join(lines)

            # Count fallback responses in history
            fallback_count = 0
            fallback_phrases = [
                "don't have enough information",
                "unable to answer",
                "please share your contact details",
                "provide your contact information",
                "not able to find any information",
                "i'm sorry i couldn't answer",
                "i'm sorry, i don't have enough information",
                "i'm having trouble understanding your question",
                "i'm not sure i can find any information",
                "i'm going to need a bit more information",
                "i need a bit more information",
                "could you clarify",
                "could you provide more context",
                "can you please provide more context",
                "i'd be happy to help, but i need a bit more information",
                "to assist you better, could you let me know more",
                "i want to make sure i understand correctly",
                "i'd be happy to help! could you clarify",
                "i'd be happy to help, but i need a bit more context",
                "i'm still not sure who or what",
                "i'm not familiar with the name",
                "i'm having trouble finding any relevant information",
                "i'm still unclear about what you're looking for",
                   "don't have enough information",
                        "unable to answer",
                        "please share your contact",
                        "provide your contact",
                        "not able to find any information",
                        "sorry i couldn't answer",
                        "sorry, i don't have enough information",
                        "having trouble understanding",
                        "not sure i can find any information",
                        "need a bit more information",
                        "could you clarify",
                        "could you provide more context",
                        "can you please provide more context",
                        "i'd be happy to help, but i need",
                        "to assist you better, could you let me know more",
                        "make sure i understand",
                        "could you rephrase",
                        "not sure i understand",
                        "still unclear",
                        "not familiar with the name",
                        "having trouble finding any relevant information",
                        "better assist you",
                        "could you please rephrase",
                        "i'm not sure i understand",
                        "i'm not sure what you're looking for",
                        "i'm still not sure",
                        "i'm still unclear",
                        "i'm not familiar with",
                        "i'm having trouble finding",
                        "i'm having trouble understanding",
                        "i'm not sure i can help",
                        "i'm not sure how to help",
                        "i'm not sure i can answer",
                        "i'm not sure i have enough information",
                        "i'm not sure i understand what you're looking for",
                    
            ]
       
            for m in history or []:
                if (
                    isinstance(m, dict)
                    and m.get("role") == "assistant"
                ):
                    content = m.get("content", "").lower()
                    # Use substring matching for more robust fallback detection
                    if any(phrase in content for phrase in fallback_phrases):
                        fallback_count += 1
            
            prompt_dict = build_augmented_system_instruction(
                history=history,
                user_message=user_query,
                knowledge_base=knowledge_base,
                custom_instruction=tenant_custom_instruction,
                fallback_count=fallback_count,
            )

           
           
            max_tokens = prompt_dict.get("max_tokens", 800)
            action = prompt_dict.get("detected_intent")

            logger.debug("Detected intent: %s", action)
            logger.debug("Final system prompt prepared")
             # If fallback_count >= 4, set action to 'fallback_max' for escalation
            if fallback_count >= 3:
                # breakpoint()
                action = "fallback_max"

            # Build full message list for the LLM: system -> prior turns -> current user
            messages: list[dict] = [prompt_dict["system_message"]]

            for h in history:
                if not isinstance(h, dict):
                    continue
                role = h.get("role")
                content = h.get("content")
                if not role or not isinstance(content, str) or not content.strip():
                    continue
                messages.append({"role": role, "content": content})

            messages.append({"role": "user", "content": user_query})
             
            # ---------------- LLM ----------------
            full_text = await generate_chat_response(
                messages,
                max_tokens=max_tokens,
                temperature=0.2,
                top_p=0.9,
            )

            # ---------------- POST PROCESS ----------------
            clean_text = clean_llm_output(full_text)
            print("Clean LLM outputttttttttttttttttttttttttttttt:", clean_text)
            print("Actionkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk:", action , fallback_count)
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






