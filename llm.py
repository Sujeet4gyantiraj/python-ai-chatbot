import os
import httpx
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()

GROK_API_KEY = os.getenv("GROK_API_KEY")
GROK_API_URL = "https://api.x.ai/v1/chat/completions"


async def generate_grok_response(
    messages: List[Dict[str, str]],
    model: str = "grok-2-latest",
    max_tokens: int = 300,
    temperature: float = 0.3,
    top_p: float = 0.9,
) -> str:
    """
    Generate a response using xAI Grok model.
    messages = [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."}
    ]
    """

    if not GROK_API_KEY:
        raise RuntimeError("GROK_API_KEY not set")

    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    timeout = httpx.Timeout(60.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            res = await client.post(
                GROK_API_URL,
                headers=headers,
                json=payload,
            )

            if res.status_code != 200:
                raise RuntimeError(
                    f"Grok API error {res.status_code}: {res.text}"
                )

            data = res.json()
            return data["choices"][0]["message"]["content"].strip()

        except httpx.ReadTimeout:
            return "The request timed out. Please try again."

        except Exception as e:
            print("[GROK ERROR]", e)
            return "An error occurred while generating a response."
