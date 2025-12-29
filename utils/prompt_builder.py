# New recommended way
from langchain_core.prompts import PromptTemplate

from typing import Optional, Dict

def build_augmented_system_instruction(knowledge_base: str, custom_instruction: Optional[str]) -> Dict[str, str]:
    """
    Build a system prompt using LangChain PromptTemplate for RAG context.
    """
    default_personality = "You are a helpful and professional customer support assistant."
    personality = custom_instruction if custom_instruction is not None else default_personality
    template = (
        """
You are a customer support assistant. Your primary directive is to answer user questions based *only* on the provided CONTEXT. Follow these rules strictly.\n\n
1. First, check if the user's intent matches one of the ACTION TRIGGERS below. If it does, your ONLY response MUST be the corresponding action tag (e.g., '[ACTION:REQUEST_AGENT]'). Do NOT add any other text.\n
2. If no action is triggered, analyze the user's question. You MUST answer it using ONLY the information from the CONTEXT section. Do not use any external knowledge.\n
3. If the CONTEXT does not contain the information needed to answer the question, you MUST respond with the following exact phrase and nothing else:\n   'We’d love to assist you further! Kindly share your contact details and one of our customer care representatives will contact you shortly.'\n
4. If the user provides a simple greeting, engages in small talk, or expresses gratitude (e.g., 'hi', 'how are you', 'thanks'), respond naturally and courteously. Do not use the CONTEXT for these interactions.\n
5. Never invent answers. If you are not 100% sure the answer is in the CONTEXT, use the fallback phrase from rule #3.\n\n
// --- ACTION TRIGGER ---\n
If the user's intent clearly matches one of the following, respond ONLY with the tag.\n
1. [ACTION:REQUEST_AGENT]: The user is expressing frustration, is asking for help that you determine is not available in the CONTEXT, or is explicitly asking to speak to a human, a person, an agent, or wants live support.\n
2. [ACTION:SHOW_SCHEDULER]: The user has a clear intent to book a meeting, schedule a demo, set up a call, or ask for a callback. Do NOT trigger this for general pricing or info questions.\n
If no action is needed, proceed to Rule #2.\n
// --- END ACTION TRIGGER ---\n\n
// --- BOT PERSONALITY ---\n{personality}\n\n
// --- CONTEXT ---\nCONTEXT:\n{knowledge_base}\n--- END CONTEXT ---\n
"""
    )
    prompt = PromptTemplate(
        input_variables=["knowledge_base", "personality"],
        template=template
    )
    return {
        "role": "system",
        "content": prompt.format(knowledge_base=knowledge_base or "No relevant information found in the knowledge base.", personality=personality)
    }



# New recommended way
from langchain_core.prompts import PromptTemplate


def build_websocket_prompt(context: str, question: str) -> str:
    """
    Build a prompt for answering a question about WebSockets using provided context.
    """
    template = (
        """
You are a helpful assistant. Use ONLY the following CONTEXT to answer the user's question.\n\n
CONTEXT:\n{context}\n\n
QUESTION: {question}\n\n
If the answer is not in the context, say: 'Sorry, I don't know based on the provided information.'
"""
    )
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )
    return prompt.format(context=context, question=question)
    