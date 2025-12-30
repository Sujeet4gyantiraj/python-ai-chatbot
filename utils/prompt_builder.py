
import os
import json


import redis.asyncio as redis

from intent_classification import IntentRouter

# Async Redis connection (singleton)
_redis_client = None
async def get_redis_client():
    global _redis_client
    if _redis_client is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        _redis_client = redis.from_url(redis_url, decode_responses=True)
    return _redis_client

# Async store and load last k messages for a session and bot
async def save_chat_history(bot_id: str, session_id: str, messages: list[dict], k: int = 5):
    r = await get_redis_client()
    trimmed = messages[-k:]
    await r.set(f"chat_history:{bot_id}:{session_id}", json.dumps(trimmed))

async def load_chat_history(bot_id: str, session_id: str, k: int = 5) -> list[dict]:
    r = await get_redis_client()
    data = await r.get(f"chat_history:{bot_id}:{session_id}")
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

# def build_augmented_system_instruction(knowledge_base: str, question: str, custom_instruction: Optional[str]) -> Dict[str, str]:
#     """
#     Build an optimized system prompt for a customer support RAG chatbot.
#     Focuses on clarity, consistency, and better response control.
#     """
#     default_personality = "You are a friendly, professional customer support assistant dedicated to helping customers quickly and accurately."
#     personality = custom_instruction if custom_instruction else default_personality
    
#     template = """You are a customer support assistant. Your role is to help users by answering their questions accurately and professionally.

#     # CORE RULES (Follow in order)

#     ## Rule 1: If the CONTEXT contains any related or partial information about the user's question, always summarize or paraphrase what the context says as helpfully as possible, even if there is no direct answer.

#     ## Rule 2: Handle Action Triggers
#     Before answering any question, check if the user's message matches an action trigger below. If it does, respond ONLY with the action tag—nothing else.

#     **ACTION TRIGGERS:**
#     - [ACTION:REQUEST_AGENT] - User is frustrated, explicitly asks for a human/agent/person/live support, or needs help beyond what the context provides
#     - [ACTION:SHOW_SCHEDULER] - User wants to book a meeting, schedule a demo, set up a call, or request a callback (not for general pricing inquiries)

#     ## Rule 3: Respond to Greetings & Small Talk Naturally
#     If the user says hello, asks how you are, thanks you, or engages in casual conversation, respond warmly and naturally. No need to reference context for these.

#     ## Rule 4: Answer Questions Using ONLY the Context
#     For all other questions:
#     - Search the CONTEXT below carefully for relevant information
#     - Answer using ONLY what's in the CONTEXT
#     - Be conversational but accurate
#     - Do not add information from outside the CONTEXT
#     - Do not make assumptions or fill in gaps

#     ## Rule 5: If the CONTEXT truly lacks any relevant information, respond with exactly this:

#     "I'd be happy to help you with that! To give you the most accurate information, please share your contact details and one of our customer care representatives will reach out to you shortly."

#     ---

#     # YOUR PERSONALITY
#     {personality}
#      ==== PERSONALITY END ====
#     ---

#     ==== CONTEXT START ====
#     {knowledge_base}
#     ==== CONTEXT END ====

#     ---

#     # USER QUESTION
#     {question}
#      ==== USER QUESTION END ====
#     ---

#     Remember: If there is any related or partial information in the context, always summarize it for the user. Only use the fallback response if there is truly nothing relevant.

#     ---
#     # INSTRUCTIONS TO ASSISTANT
#     Respond ONLY with your reply to the user. Do NOT include any reasoning, rules, or explanations in your answer.
    
    
#     # FINAL INSTRUCTION
#     IMPORTANT: Respond ONLY with what the assistant would say to the user. Do NOT include any reasoning, rules, or explanations. Do NOT repeat this instruction.
    
    
#     """

#     prompt = PromptTemplate(
#         input_variables=["knowledge_base", "personality", "question"],
#         template=template
#     )
    
#     breakpoint()
#     return {
#         "role": "system",
#         "content": prompt.format(
#             knowledge_base=knowledge_base or "No relevant information available in the knowledge base.",
#             personality=personality,
#             question=question
#         )
#     }

    




from typing import Optional, Dict

class ChatbotPrompts:
    """
    Modular prompt templates for different chatbot intents.
    Optimized for Llama 3 understanding.
    """
    
    @staticmethod
    def build_qa_prompt(knowledge_base: str, question: str, custom_instruction: Optional[str] = None) -> Dict[str, str]:
        """
        Question-answering prompt for RAG-based responses.
        """
        default_personality = "You are a friendly, professional customer support assistant."
        personality = custom_instruction if custom_instruction else default_personality
        
        template = """You are a customer support assistant helping a user with their question.

# YOUR PERSONALITY
{personality}

# KNOWLEDGE BASE
{knowledge_base}

# USER QUESTION
{question}

# INSTRUCTIONS
1. If the knowledge base contains ANY related information, summarize it clearly and helpfully for the user
2. Answer using ONLY information from the knowledge base above
3. Be conversational and friendly
4. Do NOT invent information or add external knowledge
5. If the knowledge base has NO relevant information at all, respond with exactly:
   "I'd be happy to help you with that! To give you the most accurate information, please share your contact details and one of our customer care representatives will reach out to you shortly."

Respond directly to the user now:"""

        return {
            "role": "system",
            "content": template.format(
                personality=personality,
                knowledge_base=knowledge_base or "No information available.",
                question=question
            )
        }
    
    @staticmethod
    def build_greeting_prompt(custom_instruction: Optional[str] = None) -> Dict[str, str]:
        """
        Simple prompt for handling greetings and small talk.
        """
        default_personality = "You are a friendly, professional customer support assistant."
        personality = custom_instruction if custom_instruction else default_personality
        
        template = """You are a customer support assistant having a friendly conversation.

# YOUR PERSONALITY
{personality}

# INSTRUCTIONS
The user is greeting you or making small talk (saying hello, asking how you are, thanking you, etc.).

Respond warmly and naturally. Keep it brief and friendly. Then offer to help them with anything they need.

Respond directly to the user now:"""

        return {
            "role": "system",
            "content": template.format(personality=personality)
        }
    
    @staticmethod
    def build_agent_request_prompt(custom_instruction: Optional[str] = None) -> Dict[str, str]:
        """
        Prompt for when user wants to speak with a human agent.
        """
        default_personality = "You are a friendly, professional customer support assistant."
        personality = custom_instruction if custom_instruction else default_personality
        
        template = """You are a customer support assistant helping transfer a user to a human agent.

# YOUR PERSONALITY
{personality}

# INSTRUCTIONS
The user wants to speak with a human agent, is frustrated, or needs help beyond what you can provide.

Acknowledge their request warmly and let them know you're connecting them with a team member.

After your brief acknowledgment, respond with ONLY this action tag on a new line:
[ACTION:REQUEST_AGENT]

Respond directly to the user now:"""

        return {
            "role": "system",
            "content": template.format(personality=personality)
        }
    
    @staticmethod
    def build_scheduler_prompt(custom_instruction: Optional[str] = None) -> Dict[str, str]:
        """
        Prompt for when user wants to schedule a meeting/demo/call.
        """
        default_personality = "You are a friendly, professional customer support assistant."
        personality = custom_instruction if custom_instruction else default_personality
        
        template = """You are a customer support assistant helping a user schedule a meeting.

# YOUR PERSONALITY
{personality}

# INSTRUCTIONS
The user wants to book a meeting, schedule a demo, set up a call, or request a callback.

Acknowledge their request warmly and let them know you're opening the scheduler.

After your brief acknowledgment, respond with ONLY this action tag on a new line:
[ACTION:SHOW_SCHEDULER]

Respond directly to the user now:"""

        return {
            "role": "system",
            "content": template.format(personality=personality)
        }





# Main router function
def build_augmented_system_instruction(
    user_message: str,
    knowledge_base: Optional[str] = None,
    custom_instruction: Optional[str] = None,
    intent: Optional[str] = None
) -> Dict[str, any]:
    """
    Automatically route user message to appropriate prompt.
    
    Args:
        user_message: The user's input text
        knowledge_base: Context for QA (optional)
        custom_instruction: Optional custom personality
        intent: Optional manual intent override (one of 'qa', 'greeting', 'agent_request', 'scheduler')
    
    Returns:
        Dict with 'system_message', 'detected_intent', and 'confidence'
    """
    prompts = ChatbotPrompts()
    router = IntentRouter()
    
    # Detect intent if not provided
    if intent is None:
        detected_intent = router.detect_intent(user_message)
    else:
        detected_intent = intent
    
    confidence = router.get_confidence_score(user_message, detected_intent)
    
    # Build appropriate prompt
    if detected_intent == "qa":
        system_msg = prompts.build_qa_prompt(
            knowledge_base or "",
            user_message,
            custom_instruction
        )
    elif detected_intent == "greeting":
        system_msg = prompts.build_greeting_prompt(custom_instruction)
    elif detected_intent == "agent_request":
        system_msg = prompts.build_agent_request_prompt(custom_instruction)
    elif detected_intent == "scheduler":
        system_msg = prompts.build_scheduler_prompt(custom_instruction)
    else:
        raise ValueError(f"Unknown intent: {detected_intent}")
    
    return {
        "system_message": system_msg,
        "detected_intent": detected_intent,
        "confidence": confidence
    }


# Example usage:
"""
# Test cases showing improved accuracy:

# 1. Greeting (short message bonus)
result = build_augmented_system_instruction("Hi there!")
# Intent: greeting, Confidence: 0.9

# 2. Frustrated user (frustration signal boost)
result = build_augmented_system_instruction(
    "This is ridiculous, I need to speak to a human agent now"
)
# Intent: agent_request, Confidence: 0.95+

# 3. Scheduler with context
result = build_augmented_system_instruction(
    "Can we schedule a demo call for next Tuesday?"
)
# Intent: scheduler, Confidence: 0.8+

# 4. False positive avoided (pricing question, not booking)
result = build_augmented_system_instruction(
    "How much does a demo cost? What are your meeting room prices?"
)
# Intent: qa (scheduler heavily penalized), Confidence: 0.75

# 5. Question detection
result = build_augmented_system_instruction(
    "What are your business hours?"
)
# Intent: qa, Confidence: 0.75

# 6. Ambiguous agent request (weak signal)
result = build_augmented_system_instruction(
    "Can I talk to someone about this?"
)
# Intent: agent_request, Confidence: 0.5-0.6

# 7. Strong scheduler signal
result = build_augmented_system_instruction(
    "I'd like to book an appointment to discuss pricing"
)
# Intent: scheduler, Confidence: 0.8+ (strong phrase overrides "pricing")

# Low confidence handling
result = build_augmented_system_instruction("xyz123")
if result['confidence'] < 0.4:
    # Fall back to QA or ask clarifying question
    print("Unclear intent, using QA default")

# With knowledge base for QA
result = build_augmented_system_instruction(
    user_message="How does your product work?",
    knowledge_base="Our product uses AI to automate customer support...",
    custom_instruction="You are AcmeCorp's support bot"
)
"""