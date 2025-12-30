# genai_service.py
# FastAPI GenAI service using Pinecone for RAG (logic unchanged)
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import re
import httpx
import asyncio
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from utils.prompt_builder import build_augmented_system_instruction, format_prompt_for_llama3, get_memory, load_chat_history, save_chat_history


load_dotenv()


# Pinecone Setup
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX")
pinecone_index = pc.Index(PINECONE_INDEX_NAME)



GENAI_API_BASE_URL = "https://aibot14.studyineurope.xyz/genaiapi"
EMBEDDING_DIM = 768


# Embedding API (single and batch)
async def generate_embedding(
    text: str,
    task_type: str = "search_query"
) -> List[float]:
    if not text or len(text.strip()) < 3:
        return [0.0] * EMBEDDING_DIM
    async with httpx.AsyncClient(timeout=30) as client:
        res = await client.post(
            f"{GENAI_API_BASE_URL}/embed/",
            json={"text": text, "task_type": task_type}
        )
        if res.status_code != 200:
            raise Exception("Embedding API failed")
        data = res.json()
        return data["embedding"]



embed_query = generate_embedding


# Chat Generation API
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
                },
            )

            if res.status_code != 200:
                print("[GENAI ERROR] HTTP", res.status_code, res.text)
                return (
                    "I'm sorry, I'm having trouble generating a response right now. "
                    "Please try again."
                )
            breakpoint()
            data = res.json()

            if data.get("error"):
                if data.get("safety") and "unsafe" in data["safety"]:
                    return "I cannot respond to that request as it violates safety policies."

                print("[GENAI ERROR] Model error:", data)
                return (
                    "I'm sorry, I couldn't generate a response. "
                    "Please request human assistance."
                )

            return data.get("generated_text", "").strip()

        except httpx.ReadTimeout:
            print("[GENAI ERROR] Read timeout from LLM")
            return (
                "The response is taking longer than expected. "
                "Please try again in a moment."
            )

        except asyncio.CancelledError:
            raise

        except Exception as e:
            print("[GENAI ERROR] Unexpected:", e)
            return (
                "An unexpected error occurred. "
                "Please request human assistance."
            )



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





# MAIN RAG FUNCTION
async def generate_and_stream_ai_response(
    bot_id: str,
    session_id: str,
    user_query: str,
    ai_node_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Optional[str]]:
    try:
       
        full_text = ""
        clean_text = ""
        action = None
        try:

            # Load previous chat history from Redis (last 5 messages)
            prev_msgs = await load_chat_history(bot_id, session_id, k=5)
            # Add the new user message
            prev_msgs.append({"role": "user", "content": user_query})
            # Only keep last 5 messages
            prev_msgs = prev_msgs[-5:]
            await save_chat_history(bot_id, session_id, prev_msgs, k=5)

            # RAG (PINECONE)
            knowledge_base = ""
            if not ai_node_data or not ai_node_data.get("disableKnowledgeBase"):
                query_embedding = await embed_query(user_query)
                print("[RAG DEBUG] Pinecone query embedding:", query_embedding[:10], "... (truncated)")
                pinecone_response = pinecone_index.query(
                    vector=query_embedding,
                    top_k=3,
                    include_metadata=True,
                    namespace=bot_id
                )
                print("[RAG DEBUG] Pinecone response:", pinecone_response)
                SIMILARITY_THRESHOLD = 0.5
              
                if pinecone_response and pinecone_response.get("matches"):
                    knowledge_base = "\n\n---\n\n".join(
                        match["metadata"]["content"]
                        for match in pinecone_response["matches"]
                        if (
                            match.get("metadata")
                            and match["metadata"].get("content")
                            and match.get("score", 0) >= SIMILARITY_THRESHOLD
                        )
                    )
                print("[RAG DEBUG] Knowledge base for prompt:", knowledge_base)
            breakpoint()
            # Build improved prompt for LLM (only current user message)
            custom_instruction = ''
            prompt_dict = build_augmented_system_instruction(user_query,knowledge_base,custom_instruction=custom_instruction)
            breakpoint()
            if prompt_dict and isinstance(prompt_dict['system_message'], dict) and 'content' in prompt_dict['system_message']:
                prompt = prompt_dict['system_message']['content']
            else:
                prompt = ""
            print("[RAG DEBUG] Final prompt sent to LLM:\n", prompt)
            full_text = await generate_chat_response(prompt)
            if not full_text:
                full_text = ""

            # Add assistant response to history and save back to Redis
            prev_msgs.append({"role": "assistant", "content": full_text})
            await save_chat_history(bot_id, session_id, prev_msgs, k=5)

            # Extract ACTION
            match = re.search(r"\[ACTION:(.*?)\]", full_text)
            if match:
                action = match.group(1)
                clean_text = re.sub(r"\[ACTION:.*?\]", "", full_text).strip()
            else:
                clean_text = full_text.strip()

            return {
                "fullText": full_text,
                "cleanText": clean_text,
                "action": action
            }
        except Exception as e:
            print("[GENAI ERROR]", e)
            return {
                "fullText": "",
                "cleanText": "",
                "action": None,
            }

    except Exception as e:
        print("[GENAI ERROR] Outer:", e)
        return {
            "fullText": "",
            "cleanText": "",
            "action": None,
        }

