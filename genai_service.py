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
from utils.prompt_builder import build_augmented_system_instruction, format_prompt_for_llama3, get_memory
from utils.redis_client import load_chat_history, save_chat_history


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
                    "stop": [
                        "END",
                        "BEGIN KNOWLEDGE BASE",
                        "END KNOWLEDGE BASE",
                        "====================",
                        "QUESTION:",
                        "TASK:",
                        "\n\n"
                    ]
                },
            )

            if res.status_code != 200:
                print("[GENAI ERROR] HTTP", res.status_code, res.text)
                return (
                    "I'm sorry, I'm having trouble generating a response right now. "
                    "Please try again."
                )
           
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




def extract_before_hash(text: str) -> str:
    """
    Extract text from the beginning until the first '#' character.
    """
    match = re.match(r"^(.*?)\s*#", text)
    if match:
        return match.group(1).strip()
    return text.strip()





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

         
            # RAG (PINECONE)
            knowledge_base = ""
            # breakpoint()
            if not ai_node_data or not ai_node_data.get("disableKnowledgeBase"):
                query_embedding = await embed_query(user_query)
                print("[RAG DEBUG] Pinecone query embedding:", query_embedding[:10], "... (truncated)")
                pinecone_response = pinecone_index.query(
                    vector=query_embedding,
                    top_k=5,
                    include_metadata=True,
                    namespace=bot_id
                )
                print("[RAG DEBUG] Pinecone response:", pinecone_response)
                SIMILARITY_THRESHOLD = 0.55
              
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
                print("[RAG DEBUG] Knowledge base for prompt:", knowledge_base)
          
            # Build improved prompt for LLM (only current user message)
            custom_instruction = ''
          
            # 1. Load previous history
       
            history = await load_chat_history(bot_id, session_id, k=10)
            prompt_dict = build_augmented_system_instruction(user_query,knowledge_base,custom_instruction=custom_instruction)
           
            if prompt_dict and isinstance(prompt_dict['system_message'], dict) and 'content' in prompt_dict['system_message']:
                system_prompt = prompt_dict['system_message']['content']
                max_tokens = prompt_dict.get('max_tokens',300)
            else:
                prompt = ""
                max_tokens = 300
         
            print("[RAG DEBUG] Final prompt sent to LLM:\n", system_prompt)
            action = prompt_dict.get("detected_intent", None)

            # 3. Build full message list
            messages = [
                prompt_dict['system_message'],
                *history,
                {"role": "user", "content": user_query}
            ]

            # 4. Format prompt for Llama-3
            prompt = format_prompt_for_llama3(messages)

          
            full_text = await generate_chat_response(prompt,max_tokens=max_tokens,temperature=0.2,top_p=0.9)
            if not full_text:
                full_text = ""

            
          

            
            
            # Extract ACTION
            match = re.search(r"\[ACTION:(.*?)\]", full_text)
            if match:
                # action = match.group(1)
                clean_text = re.sub(r"\[ACTION:.*?\]", "", full_text).strip()
            else:
                clean_text = full_text.strip()
            if prompt_dict.get("detected_intent") == "greeting":
                clean_text = extract_before_hash(clean_text)


            # 6. Save back to Redis
            history.append({"role": "user", "content": user_query})
            history.append({"role": "assistant", "content": clean_text})

            await save_chat_history(bot_id, session_id, history, k=10)

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

