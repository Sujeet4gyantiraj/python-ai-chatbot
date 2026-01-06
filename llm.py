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
        

# from ollama import Client
# import asyncio
# import logging

# logger = logging.getLogger(__name__)

# ollama_client = Client(host="http://localhost:11434")
# # If on cloud VM:
# # Client(host="http://YOUR_VM_IP:11434")

# async def generate_chat_response_qwen(
#     prompt: str,
#     max_tokens: int = 300,
#     temperature: float = 0.2,
#     top_p: float = 0.9,
#     model: str = "qwen3-coder:480b-cloud"
# ) -> str:
#     try:
#         # breakpoint()
#         response = ollama_client.generate(
#             model=model,
#             prompt=prompt,
#             options={
#                 "temperature": temperature,
#                 "top_p": top_p,
#                 "num_predict": max_tokens,
#                 # IMPORTANT: stop tokens
#                 "stop": [
#                     "END",
#                     "BEGIN KNOWLEDGE BASE",
#                     "END KNOWLEDGE BASE",
#                     "====================",
#                     "QUESTION:",
#                     "TASK:"
#                 ],
#             },
#         )

#         return response.get("response", "").strip()

#     except Exception as e:
#         logger.exception("Ollama generation error")
#         return (
#             "I'm sorry, I couldn't generate a response. "
#             "Please request human assistance."
#         )
    



# import aiohttp
# import asyncio

# async def ollama_async(prompts, model="llama3.1"):
#     async with aiohttp.ClientSession() as session:
#         tasks = []
#         for prompt in prompts:
#             print("Sending prompt:", prompt)
#             tasks.append(
#                 session.post(
#                     'http://localhost:11434/api/generate',
#                     json={'model': model, 'prompt': prompt, 'stream': False}
#                 )
#             )

#         print("Awaiting responses...")
#         responses = await asyncio.gather(*tasks)
#         print("Responses received.")

#         results = []
#         for response in responses:
#             data = await response.json()
#             results.append(data['response'])

#         return results

# # Run
# if __name__ == "__main__":
#     outputs = asyncio.run(
#         ollama_async(["Hello", "How are you?"])
#     )

#     print("\nModel Outputs:")
#     for o in outputs:
#         print("-", o)

