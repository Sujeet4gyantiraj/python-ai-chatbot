import os
import json
import redis.asyncio as redis
import logging

# -------------------- LOGGER SETUP --------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
)
logger = logging.getLogger(__name__)
# ----------------------------------------------------


_redis_client = None


async def get_redis_client():
    global _redis_client
    if _redis_client is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        logger.info("Initializing Redis client | url=%s", redis_url)
        _redis_client = redis.from_url(redis_url, decode_responses=True)
    return _redis_client


def _sanitize_messages(messages: list[dict]) -> list[dict]:
    clean = []
    for m in messages:
        if (
            isinstance(m, dict)
            and "role" in m
            and "content" in m
            and isinstance(m["content"], str)
        ):
            clean.append({
                "role": m["role"],
                "content": m["content"]
            })
    logger.debug("Sanitized messages | original=%d | sanitized=%d", len(messages), len(clean))
    return clean


async def load_chat_history(bot_id: str, session_id: str, k: int = 10) -> list[dict]:
    r = await get_redis_client()
    key = f"chat_history:{bot_id}:{session_id}"

    logger.info("Loading chat history | bot_id=%s | session_id=%s | last_k=%d", bot_id, session_id, k)
    data = await r.get(key)
    if not data:
        logger.info("No chat history found for key=%s", key)
        return []

    try:
        messages = json.loads(data)
        last_k = messages[-k:]
        logger.debug("Loaded chat history | messages=%d | returning=%d", len(messages), len(last_k))
        return last_k
    except Exception as e:
        logger.exception("Failed to load chat history | key=%s", key)
        return []


async def save_contact_details(bot_id: str, session_id: str, details: dict):
    """Persist contact details (phone, email, etc.) for a session."""
    r = await get_redis_client()
    key = f"contact_details:{bot_id}:{session_id}"
    logger.info("Saving contact details | bot_id=%s | session_id=%s", bot_id, session_id)
    try:
        await r.set(key, json.dumps(details), ex=3600)
        logger.debug("Contact details saved | key=%s", key)
    except Exception:
        logger.exception("Failed to save contact details | key=%s", key)


async def load_contact_details(bot_id: str, session_id: str) -> dict | None:
    """Load previously saved contact details for a session."""
    r = await get_redis_client()
    key = f"contact_details:{bot_id}:{session_id}"
    logger.info("Loading contact details | bot_id=%s | session_id=%s", bot_id, session_id)
    data = await r.get(key)
    if not data:
        return None
    try:
        return json.loads(data)
    except Exception:
        logger.exception("Failed to load contact details | key=%s", key)
        return None


async def save_chat_history(bot_id: str, session_id: str, messages: list[dict], k: int = 10):
    r = await get_redis_client()
    key = f"chat_history:{bot_id}:{session_id}"

    clean_messages = _sanitize_messages(messages)
    trimmed = clean_messages[-k:]

    logger.info("Saving chat history | bot_id=%s | session_id=%s | messages_to_save=%d", bot_id, session_id, len(trimmed))
    try:
        await r.set(key, json.dumps(trimmed), ex=3600)
        logger.debug("Chat history saved successfully | key=%s", key)
    except Exception as e:
        logger.exception("Failed to save chat history | key=%s", key)
