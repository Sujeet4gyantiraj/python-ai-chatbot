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

from utils.prompt_builder import build_augmented_system_instruction


load_dotenv()

from sqlalchemy.orm import Session

from db import SessionLocal
from models import (
    Session as ChatSession,
    Message,
)

# =========================
# Pinecone Setup
# =========================
from pinecone import Pinecone

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX")
pinecone_index = pc.Index(PINECONE_INDEX_NAME)



GENAI_API_BASE_URL = "https://aibot14.studyineurope.xyz/genaiapi"
EMBEDDING_DIM = 768


# =========================
# Request Models
# =========================
class GenerateAIRequest(BaseModel):
    session_id: str
    last_user_message: str
    ai_node_data: Optional[Dict[str, Any]] = None


# =========================
# Embedding API
# =========================
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


# =========================
# Chat Generation API
# =========================
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
                    "prompt": str(prompt),
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


# =========================
# Prompt Builders
# =========================
# def build_augmented_system_instruction(
#     knowledge_base: str,
#     custom_instruction: Optional[str]
# ) -> Dict[str, str]:

#     personality = (
#         custom_instruction
#         if custom_instruction
#         else "You are a helpful and professional customer support assistant."
#     )

#     return {
#         "role": "system",
#         "content": f"""
# You are a customer support assistant.

# Rules:
# - Answer ONLY using CONTEXT.
# - If answer not found, ask for contact details.
# - Never invent information.

# PERSONALITY:
# {personality}

# CONTEXT:
# {knowledge_base or "No relevant information found in the knowledge base."}
# """
#     }


# def build_augmented_system_instruction(
#     knowledge_base: str,
#     custom_instruction: Optional[str],
# ) -> Dict[str, str]:
#     # --- Non-negotiable rules (EXACT parity with Node.js) ---
#     non_negotiable_rule = """
# You are a customer support assistant. Your primary directive is to answer user questions based *only* on the provided CONTEXT. Follow these rules strictly:
# 1. First, check if the user's intent matches one of the ACTION TRIGGERS below. If it does, your ONLY response MUST be the corresponding action tag (e.g., '[ACTION:REQUEST_AGENT]'). Do NOT add any other text.
# 2. If no action is triggered, analyze the user's question. You MUST answer it using ONLY the information from the CONTEXT section. Do not use any external knowledge.
# 3. If the CONTEXT does not contain the information needed to answer the question, you MUST respond with the following exact phrase and nothing else:
#    "We’d love to assist you further! Kindly share your contact details and one of our customer care representatives will contact you shortly."
# 4. If the user provides a simple greeting, engages in small talk, or expresses gratitude (e.g., 'hi', 'how are you', 'thanks'), respond naturally and courteously. Do not use the CONTEXT for these interactions.
# 5. Never invent answers. If you are not 100% sure the answer is in the CONTEXT, use the fallback phrase from rule #3.
# """

#     # --- Personality logic (same truth table as Node.js) ---
#     default_personality = "You are a helpful and professional customer support assistant."
#     personality = (
#         custom_instruction
#         if custom_instruction is not None
#         else default_personality
#     )

#     # --- Action triggers (EXACT copy) ---
#     action_instruction = """
# // --- ACTION TRIGGER ---
# If the user's intent clearly matches one of the following, respond ONLY with the tag.

# 1. **[ACTION:REQUEST_AGENT]**: The user is expressing frustration, is asking for help that you determine is not available in the CONTEXT, or is explicitly asking to speak to a human, a person, an agent, or wants live support.
#    - User says: "I need to talk to a real person." -> Your response: [ACTION:REQUEST_AGENT]
#    - User says: "This is frustrating, connect me to an agent." -> Your response: [ACTION:REQUEST_AGENT]

# 2. **[ACTION:SHOW_SCHEDULER]**: The user has a clear intent to book a meeting, schedule a demo, set up a call, or ask for a callback. Do NOT trigger this for general pricing or info questions.
#    - User says: "This sounds great, can I book a demo?" -> Your response: [ACTION:SHOW_SCHEDULER]
#    - User says: "Can you have someone call me back?" -> Your response: [ACTION:SHOW_SCHEDULER]

# If no action is needed, proceed to Rule #2.
# // --- END ACTION TRIGGER ---
# """

#     # --- Final system message (structure preserved) ---
#     return {
#         "role": "system",
#         "content": f"""
# {non_negotiable_rule}
# {action_instruction}
# // --- BOT PERSONALITY ---
# {personality}
# // --- CONTEXT ---
# CONTEXT:
# {knowledge_base or "No relevant information found in the knowledge base."}
# --- END CONTEXT ---
# """
#     }

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


def format_prompt_for_llama3(messages: List[Dict[str, str]]) -> str:
    prompt = "<|begin_of_text|>"
    for msg in messages:
        prompt += (
            f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n"
            f"{msg['content']}<|eot_id|>"
        )
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt


# =========================
# MAIN RAG FUNCTION
# =========================
async def generate_and_stream_ai_response(
    session_id: str,
    last_user_message: str,
    ai_node_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Optional[str]]:

    db: Session = SessionLocal()

    full_text = ""
    clean_text = ""
    action = None

    try:
        # -----------------------
        # Fetch Session
        # -----------------------
        session = (
            db.query(ChatSession)
            .filter(ChatSession.sessionId == session_id)
            .first()
        )

        if not session:
            raise Exception("Session not found")

        # -----------------------
        # Fetch History
        # -----------------------
        history = (
            db.query(Message)
            .filter(Message.sessionId == session_id)
            .order_by(Message.createdAt.asc())
            .limit(10)
            .all()
        )

        history_for_ai = [
            {
                "role": "user" if m.role == "user" else "assistant",
                "parts": [{"text": m.text}]
            }
            for m in history
        ]

        if history_for_ai and history_for_ai[0]["role"] == "assistant":
            history_for_ai = history_for_ai[1:]

        # -----------------------
        # RAG (PINECONE)
        # -----------------------
        knowledge_base = ""
       

        if not ai_node_data or not ai_node_data.get("disableKnowledgeBase"):
            query_embedding = await embed_query(last_user_message)
            print("[RAG DEBUG] Pinecone query embedding:", query_embedding[:10], "... (truncated)")
            pinecone_response = pinecone_index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True,
                namespace=session.botId
            )
            print("[RAG DEBUG] Pinecone response:", pinecone_response)
            breakpoint()
            SIMILARITY_THRESHOLD = 0.65

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

        # -----------------------
        # Prompt + Chat
        # -----------------------
        breakpoint()
        system_instruction = build_augmented_system_instruction(
            knowledge_base,
            ai_node_data.get("customPrompt") if ai_node_data else None
        )

        messages = create_chat_session(system_instruction, history_for_ai)
        prompt = format_prompt_for_llama3(messages)
        
        full_text = await generate_chat_response(prompt)

        # -----------------------
        # Extract ACTION
        # -----------------------
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

    finally:
        db.close()



