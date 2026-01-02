import os
import json
import redis.asyncio as redis

_redis_client = None

async def get_redis_client():
    global _redis_client
    if _redis_client is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        _redis_client = redis.from_url(redis_url, decode_responses=True)
    return _redis_client


# async def save_chat_history(bot_id: str, session_id: str, messages: list[dict], k: int = 5):
#     r = await get_redis_client()
#     key = f"chat_history:{bot_id}:{session_id}"

#     trimmed = messages[-k:]

#     await r.set(
#         key,
#         json.dumps(trimmed),
#         ex=3600  # ⏱ auto-expire after 1 hour
#     )


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
    return clean


async def load_chat_history(bot_id: str, session_id: str, k: int = 10) -> list[dict]:
    r = await get_redis_client()
    key = f"chat_history:{bot_id}:{session_id}"

    data = await r.get(key)
    if not data:
        return []

    try:
        messages = json.loads(data)
        return messages[-k:]
    except Exception:
        return []


async def save_chat_history(bot_id: str, session_id: str, messages: list[dict], k: int = 10):
    r = await get_redis_client()
    key = f"chat_history:{bot_id}:{session_id}"

    clean_messages = _sanitize_messages(messages)
    trimmed = clean_messages[-k:]

    await r.set(key, json.dumps(trimmed), ex=3600)




