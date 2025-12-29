
import os
import json


import redis.asyncio as redis

# Async Redis connection (singleton)
_redis_client = None
async def get_redis_client():
    global _redis_client
    if _redis_client is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        _redis_client = redis.from_url(redis_url, decode_responses=True)
    return _redis_client

# Async store and load last k messages for a session
async def save_chat_history(session_id: str, messages: list[dict], k: int = 5):
    r = await get_redis_client()
    trimmed = messages[-k:]
    await r.set(f"chat_history:{session_id}", json.dumps(trimmed))

async def load_chat_history(session_id: str, k: int = 5) -> list[dict]:
    r = await get_redis_client()
    data = await r.get(f"chat_history:{session_id}")
    if not data:
        return []
    try:
        messages = json.loads(data)
        return messages[-k:]
    except Exception:
        return []


from langchain_classic.memory import ConversationBufferWindowMemory

def get_memory() -> ConversationBufferWindowMemory:
    """
    Returns a ConversationBufferWindowMemory instance for storing the last 5 chat messages.
    Note: If you want to persist or share memory across requests, you must store/load messages by session_id.
    """
    return ConversationBufferWindowMemory(k=5, return_messages=True)



def format_prompt_for_llama3(messages: list[dict]) -> str:
    """
    Format messages for Llama 3 model prompt.
    """
    prompt = "<|begin_of_text|>"
    for msg in messages:
        prompt += (
            f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n"
            f"{msg['content']}<|eot_id|>"
        )
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt
# New recommended way
from langchain_core.prompts import PromptTemplate

from typing import Optional, Dict

# def build_augmented_system_instruction(knowledge_base: str, custom_instruction: Optional[str]) -> Dict[str, str]:
#     """
#     Build a system prompt using LangChain PromptTemplate for RAG context.
#     """
#     # Removed breakpoint for production use
#     default_personality = "You are a helpful and professional customer support assistant."
#     personality = custom_instruction if custom_instruction is not None else default_personality
#     template = (
#         """
# You are a customer support assistant. Your primary directive is to answer user questions based *only* on the provided CONTEXT and QUESTION below. Follow these rules strictly.\n\n
# 1. First, check if the user's intent matches one of the ACTION TRIGGERS below. If it does, your ONLY response MUST be the corresponding action tag (e.g., '[ACTION:REQUEST_AGENT]'). Do NOT add any other text.\n
# 2. If no action is triggered, analyze the user's question. You MUST answer it using ONLY the information from the CONTEXT section. Do not use any external knowledge.\n
# 3. If the CONTEXT does not contain the information needed to answer the question, you MUST respond with the following exact phrase and nothing else:\n   'We’d love to assist you further! Kindly share your contact details and one of our customer care representatives will contact you shortly.'\n
# 4. If the user provides a simple greeting, engages in small talk, or expresses gratitude (e.g., 'hi', 'how are you', 'thanks'), respond naturally and courteously. Do not use the CONTEXT for these interactions.\n
# 5. Never invent answers. If you are not 100% sure the answer is in the CONTEXT, use the fallback phrase from rule #3.\n\n
# // --- ACTION TRIGGER ---\n
# If the user's intent clearly matches one of the following, respond ONLY with the tag.\n
# 1. [ACTION:REQUEST_AGENT]: The user is expressing frustration, is asking for help that you determine is not available in the CONTEXT, or is explicitly asking to speak to a human, a person, an agent, or wants live support.\n
# 2. [ACTION:SHOW_SCHEDULER]: The user has a clear intent to book a meeting, schedule a demo, set up a call, or ask for a callback. Do NOT trigger this for general pricing or info questions.\n
# If no action is needed, proceed to Rule #2.\n
# // --- END ACTION TRIGGER ---\n\n
# // --- BOT PERSONALITY ---\n{personality}\n\n
# // --- CONTEXT ---\nCONTEXT:\n{knowledge_base}\n--- END CONTEXT ---\n\n+// --- QUESTION ---\nQUESTION: {question}\n\n+"""
#     )
#     # Add 'question' as an input variable
#     prompt = PromptTemplate(
#         input_variables=["knowledge_base", "personality", "question"],
#         template=template
#     )
#     # The function now expects 'question' as an argument, so update the return accordingly
#     def build(knowledge_base, personality, question):
#         return {
#             "role": "system",
#             "content": prompt.format(
#                 knowledge_base=knowledge_base or "No relevant information found in the knowledge base.",
#                 personality=personality,
#                 question=question
#             )
#         }
#     # Return a function that can be called with the question
#     return build
#     # prompt = PromptTemplate(
#     #     input_variables=["knowledge_base", "personality"],
#     #     template=template
#     # )
#     # return {
#     #     "role": "system",
#     #     "content": prompt.format(knowledge_base=knowledge_base or "No relevant information found in the knowledge base.", personality=personality)
#     # }



def build_augmented_system_instruction(knowledge_base: str, question: str, custom_instruction: Optional[str]) -> Dict[str, str]:
    """
    Build an optimized system prompt for a customer support RAG chatbot.
    Focuses on clarity, consistency, and better response control.
    """
    default_personality = "You are a friendly, professional customer support assistant dedicated to helping customers quickly and accurately."
    personality = custom_instruction if custom_instruction else default_personality
    
    template = """You are a customer support assistant. Your role is to help users by answering their questions accurately and professionally.

    # CORE RULES (Follow in order)

    ## Rule 1: Handle Action Triggers First
    Before answering any question, check if the user's message matches an action trigger below. If it does, respond ONLY with the action tag—nothing else.

    **ACTION TRIGGERS:**
    - [ACTION:REQUEST_AGENT] - User is frustrated, explicitly asks for a human/agent/person/live support, or needs help beyond what the context provides
    - [ACTION:SHOW_SCHEDULER] - User wants to book a meeting, schedule a demo, set up a call, or request a callback (not for general pricing inquiries)

    ## Rule 2: Respond to Greetings & Small Talk Naturally
    If the user says hello, asks how you are, thanks you, or engages in casual conversation, respond warmly and naturally. No need to reference context for these.

    ## Rule 3: Answer Questions Using ONLY the Context
    For all other questions:
    - Search the CONTEXT below carefully for relevant information
    - Answer using ONLY what's in the CONTEXT
    - Be conversational but accurate
    - Do not add information from outside the CONTEXT
    - Do not make assumptions or fill in gaps

    ## Rule 4: When Context Doesn't Have the Answer
    If the CONTEXT lacks the information needed to answer the question, respond with exactly this:

    "I'd be happy to help you with that! To give you the most accurate information, please share your contact details and one of our customer care representatives will reach out to you shortly."

    ---

    # YOUR PERSONALITY
    {personality}

    ---

    # KNOWLEDGE BASE CONTEXT
    {knowledge_base}

    ---

    # USER QUESTION
    {question}

    ---

    Remember: Accuracy over speculation. If you're not certain the answer is in the context, use the fallback response."""

    prompt = PromptTemplate(
        input_variables=["knowledge_base", "personality", "question"],
        template=template
    )
    
  
    return {
        "role": "system",
        "content": prompt.format(
            knowledge_base=knowledge_base or "No relevant information available in the knowledge base.",
            personality=personality,
            question=question
        )
    }

   

    