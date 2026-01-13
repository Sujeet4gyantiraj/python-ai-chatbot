import logging
from typing import Optional, Dict

from intent_classification import SemanticRouteClassifier
from langchain_classic.memory import ConversationBufferWindowMemory

from intent_classification import get_hybrid_classifier



# -------------------- LOGGER SETUP --------------------
import os
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
)
logger = logging.getLogger(__name__)
# ----------------------------------------------------


def get_memory() -> ConversationBufferWindowMemory:
    """
    Returns a ConversationBufferWindowMemory instance for storing the last 5 chat messages.
    Note: If you want to persist or share memory across requests, you must store/load messages by session_id.
    """
    return ConversationBufferWindowMemory(k=5, return_messages=True)


def format_prompt_for_llama3(messages: list[dict], max_chars: int = 8000) -> str:
    """
    Format messages for Llama 3 model prompt.
    Optionally truncates prompt if it exceeds max_chars.
    """
    prompt = "<|begin_of_text|>"
    for msg in messages:
        content = msg.get("content", "")
        prompt += (
            f"<|start_header_id|>{msg.get('role','user')}<|end_header_id|>\n\n"
            f"{content}<|eot_id|>"
        )
        if len(prompt) > max_chars:
            logger.warning("Prompt truncated to max_chars=%d", max_chars)
            break
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt



# -------------------- PROMPT TEMPLATES --------------------
class ChatbotPrompts:
    """
    Modular prompt templates for different chatbot intents.
    Optimized for Llama 3 understanding.
    """

    @staticmethod
    def build_qa_prompt(
        knowledge_base: str,
        question: str,
        custom_instruction: Optional[str] = None
    ) -> Dict[str, str]:
        personality = custom_instruction or (
            "You are a friendly, professional customer support assistant."
        )

        template = """TASK: QUESTION ANSWERING

You are a customer support assistant answering a question using the knowledge base below.

Rules:
- Use ONLY the information from the knowledge base
- You MAY rephrase or summarize the information
- Do NOT copy sentences verbatim
- Do NOT add external knowledge
    - Output ONLY the final answer
- If the knowledge base does NOT describe the topic at all, respond with the fallback sentence EXACTLY

Fallback sentence:
"I'd be happy to help you with that! To give you the most accurate information, please share your contact details and one of our customer care representatives will reach out to you shortly."

BEGIN KNOWLEDGE BASE
====================
{knowledge_base}
====================
END KNOWLEDGE BASE

QUESTION:
{question}

FINAL ANSWER:
- Explain clearly in 3-6 sentences.
- If the query is short, still give a detailed explanation, not just one short sentence.
- Use plain text only.
"""
        return {
            "role": "system",
            "content": template.format(
                personality=personality,
                knowledge_base=knowledge_base or "",
                question=question
            )
        }

    @staticmethod
    def build_greeting_prompt(
        custom_instruction: Optional[str] = None,
        user_message: Optional[str] = None
    ) -> Dict[str, str]:
        default_personality = "You are a friendly, professional customer support assistant."
        personality = custom_instruction or default_personality

        template = """You are a customer support assistant.

# YOUR PERSONALITY
{personality}

# USER MESSAGE
{user_message}

# INSTRUCTIONS
The user is greeting you or making small talk.

Respond with ONE short, friendly reply (1–2 sentences).
Do NOT repeat greetings.
Do NOT ask multiple questions.
Do NOT continue the conversation.
Do NOT explain your reasoning.
Stop immediately after your reply.

Your response must contain ONLY what you would say to the user.
"""
        return {
            "role": "system",
            "content": template.format(
                personality=personality,
                user_message=user_message or ""
            )
        }

    @staticmethod
    def build_agent_request_prompt(custom_instruction: Optional[str] = None) -> Dict[str, str]:
        default_personality = "You are a friendly, professional customer support assistant."
        personality = custom_instruction or default_personality

        template = """You are a customer support assistant helping transfer a user to a human agent.

# YOUR PERSONALITY
{personality}

# INSTRUCTIONS
The user wants to speak with a human agent, is frustrated, or needs help beyond what you can provide.

Acknowledge their request warmly and let them know you're connecting them with a team member.

After your brief acknowledgment, respond with ONLY this action tag on a new line:
[ACTION:REQUEST_AGENT]

Respond directly to the user now:"""

        return {"role": "system", "content": template.format(personality=personality)}

    @staticmethod
    def build_scheduler_prompt(custom_instruction: Optional[str] = None) -> Dict[str, str]:
        default_personality = "You are a friendly, professional customer support assistant."
        personality = custom_instruction or default_personality

        template = """You are a customer support assistant helping a user schedule a meeting.

# YOUR PERSONALITY
{personality}

# INSTRUCTIONS
The user wants to book a meeting, schedule a demo, set up a call, or request a callback.

Acknowledge their request warmly and let them know you're opening the scheduler.

After your brief acknowledgment, respond with ONLY this action tag on a new line:
[ACTION:SHOW_SCHEDULER]

Respond directly to the user now:"""

        return {"role": "system", "content": template.format(personality=personality)}


# -------------------- ROUTER FUNCTION --------------------
def build_augmented_system_instruction(
    user_message: str,
    knowledge_base: Optional[str] = None,
    custom_instruction: Optional[str] = None,
    intent: Optional[str] = None
) -> Dict[str, any]:
    """
    Automatically route user message to appropriate prompt.

    Returns:
        Dict with 'system_message', 'detected_intent', 'confidence', 'max_tokens'
    """
    prompts = ChatbotPrompts()
    router = get_hybrid_classifier()  # Use hybrid classifier

    try:
        if intent is None:
            detected_intent, confidence_data, _ = router.classify(
                user_message, return_scores=True
            )
        else:
            detected_intent = intent
            confidence_data = {detected_intent: 1.0}
        if detected_intent != "normal_qa" and confidence_data < 0.5:
            detected_intent = "normal_qa"

       
        logger.info("Detected intent | user_message=%s | intent=%s | confidence=%s",
                    user_message, detected_intent, confidence_data)

        # Build prompt
        if detected_intent == "normal_qa":
            system_msg = prompts.build_qa_prompt(
                knowledge_base or "", user_message, custom_instruction
            )
            max_tokens = 600
        elif detected_intent == "greeting":
            system_msg = prompts.build_greeting_prompt(custom_instruction, user_message)
            max_tokens = 50
        elif detected_intent == "agent_request":
            system_msg = prompts.build_agent_request_prompt(custom_instruction)
            max_tokens = 50
        elif detected_intent == "scheduler":
            system_msg = prompts.build_scheduler_prompt(custom_instruction)
            max_tokens = 50
        else:
            logger.error("Unknown intent detected: %s", detected_intent)
            raise ValueError(f"Unknown intent: {detected_intent}")

        return {
            "system_message": system_msg,
            "detected_intent": detected_intent,
            "confidence": confidence_data,
            "max_tokens": max_tokens
        }

    except Exception as e:
        logger.exception("Failed to build system instruction")
        raise e
