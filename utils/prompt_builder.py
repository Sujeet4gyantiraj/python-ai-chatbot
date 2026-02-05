import logging
from typing import Optional, Dict

# -------------------- LOGGER SETUP --------------------
import os
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
)
logger = logging.getLogger(__name__)
# ----------------------------------------------------


class ChatbotPrompts:
    """
    Professional-grade prompt templates for Llama 3.1 chatbot.
    Optimized for clarity, consistency, and controlled outputs.
    """

    @staticmethod
    def build_qa_prompt(
        knowledge_base: str,
        question: str,
        custom_instruction: Optional[str] = None,
        history: Optional[list[dict]] = None,
    ) -> Dict[str, str]:
        """
        Question-Answering prompt with strict grounding to knowledge base.
        Optimized for Llama 3.1's instruction-following capabilities.
        """
        personality = custom_instruction or (
            "You are a polite, professional, and friendly customer support "
            "assistant. You greet users in a calm, businesslike way and "
            "focus on how you can help them. You avoid technical jargon "
            "whenever possible, and if you must use a technical term you "
            "briefly explain it in simple words. You do not talk about "
            "your own feelings or how you are doing."
        )

        # Serialize recent conversation history into a compact text block
        history_lines: list[str] = []
        if history:
            for m in history:
                if not isinstance(m, dict):
                    continue
                role = m.get("role", "user")
                content = str(m.get("content", "")).strip()
                if not content:
                    continue
                if role == "assistant":
                    prefix = "Assistant"
                elif role == "system":
                    prefix = "System"
                else:
                    prefix = "User"
                history_lines.append(f"{prefix}: {content}")

        history_block = "\n".join(history_lines) if history_lines else "No prior conversation history."

        template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

# ROLE AND PERSONALITY
{personality}

# CORE OBJECTIVE
Provide accurate answers to customer questions using ONLY the knowledge base provided below.

# CONVERSATION HISTORY (MOST RECENT LAST)

{history_block}

# KNOWLEDGE BASE

{knowledge_base}


# RESPONSE RULES
0. TENANT CONTEXT: If your role description above includes a tenant name or tenant description, assume all questions refer specifically to that tenant's products, services, and policies. Keep tone, wording, and examples aligned with that tenant.
1. GROUNDING: Use ONLY information from the knowledge base above.
2. ACCURACY: Never invent or assume information not present in the knowledge base.
3. CLARITY: Use clear sentences that are easy to understand.
4. TONE: Warm, friendly, conversational, and empathetic.
5. FORMAT: Use plain text. You may use a numbered list (1., 2., 3.) when describing steps.
6. REPHRASING: You may rephrase information for clarity, but never add new facts.
7. COMPLETENESS: Include every distinct step, option, warning, and important condition from the knowledge base that relates to the user's question, even if the answer becomes long. Do not skip or compress steps that appear in the knowledge base.
8. BREVITY: Start with a short direct answer (1–2 sentences), then add only the most important supporting details.
9. AVOID REDUNDANCY: Do not repeat the same point in different words or sections.

# RESPONSE FORMAT (CONVERSATIONAL BUT STRUCTURED)
Format your answer so it feels like a natural support conversation while still being easy to follow:
- Start by directly answering the user's question in plain language.
- If the user is asking "how to" or "how do I" or is clearly asking for a procedure or setup (for example: install, connect, configure, steps, process, procedure), provide the rest of the answer as a detailed numbered list of steps (1., 2., 3., ...), one step per line. The list MUST cover all relevant steps from the knowledge base, including steps that appear later in the text or after separators like "---".
- If the knowledge base text already contains step-by-step or bullet-point instructions that are relevant to the question, preserve that structure and do not shorten, merge, or omit distinct steps or sub-steps.
- For non-procedural questions (definitions, descriptions, general information), you may use multiple short paragraphs so that all relevant points from the knowledge base are covered.
- If the user asks multiple questions in one message, answer each question briefly in a separate paragraph or numbered point.
- If the question is unclear but the topic exists in the knowledge base, ask ONE clear clarifying question instead of using the fallback message.

# FALLBACK PROTOCOL
If the knowledge base contains NO relevant information to answer the question:
- Respond with EXACTLY this message (word-for-word):
"I'm sorry, I don't have enough information to answer that right now. Please provide your contact details and our team will connect with you shortly."
If there is any relevant information in the knowledge base, you MUST answer using that information and MUST NOT use the fallback message.

# FORBIDDEN ACTIONS
- Do NOT use external knowledge beyond the knowledge base
- Do NOT copy sentences verbatim from the knowledge base
- Do NOT say "based on the knowledge base" or reference the source
- Do NOT apologize for limitations
- Do NOT offer to connect them to support (unless using fallback)

<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return {
            "role": "system",
            "content": template.format(
                personality=personality,
                knowledge_base=knowledge_base or "No knowledge base provided.",
                history_block=history_block,
                question=question,
            )
        }

    @staticmethod
    def build_greeting_prompt(
        custom_instruction: Optional[str] = None,
        user_message: Optional[str] = None
    ) -> Dict[str, str]:
        """Greeting prompt for starting or continuing a friendly conversation in a warm, helpful, and customer-focused way."""

        personality = custom_instruction or (
            "You are a warm, friendly, and professional customer support assistant. "
            "You greet users in a positive, welcoming way and immediately focus on how you can help them. "
            "If a user says 'how are you' or similar, you may briefly say you are doing well (for example, 'I'm doing great, thanks for asking') "
            "and then immediately offer help (for example, 'Ready to help you with anything you need.'). "
            "For greetings that do NOT include 'how are you' or a similar question, do NOT talk about how you are doing; just welcome the user and offer help."
        )

        template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

# ROLE AND PERSONALITY
{personality}

# SITUATION
The user is greeting you or sending a very short, friendly message (for example "hi", "hello", "how are you").

User message: "{user_message}"

# YOUR TASK
- Respond with ONE warm, friendly greeting that makes the user feel welcome.
- Briefly offer your help in a natural way (for example, asking how you can assist).
- If the user says "how are you" or similar, you may briefly say you are doing well and then immediately offer your help (see examples).
 - If the user says "good morning", "good afternoon", or "good evening", start your reply with the same phrase (for example, "Good morning! How can I help you today?").

# RESPONSE RULES
1. LENGTH: 1–2 short sentences (maximum 25 words).
2. TONE: Warm, friendly, and professional (no slang, no jokes).
3. CONSTRAINTS:
   - Do NOT repeat their exact greeting word-for-word.
    - Do NOT mention the time of day (no "good morning", "good afternoon", or "good evening") unless the user explicitly uses it first.
    - Unless the user explicitly asks how you are (for example, "how are you" or "how are you doing"), do NOT talk about your own feelings, mood, or health.
   - Do NOT say "you’re welcome" or give closing/thank‑you style responses.



# OUTPUT FORMAT
Provide ONLY your direct response to the user. No preamble, no explanation.

<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return {
            "role": "system",
            "content": template.format(
                personality=personality,
                user_message=user_message or ""
            )
        }

    @staticmethod
    def build_agent_request_prompt(
        custom_instruction: Optional[str] = None,
        user_message: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Agent handoff prompt with empathetic acknowledgment.
        Includes action tag for system integration.
        """
        personality = custom_instruction or (
            "You are an empathetic customer support assistant who prioritizes "
            "customer satisfaction and knows when to escalate issues."
        )

        template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

# ROLE AND PERSONALITY
{personality}

# SITUATION
The user has requested human assistance or is expressing frustration that requires escalation.

User message: "{user_message}"

# YOUR TASK
1. Acknowledge their request with empathy
2. Reassure them that help is coming
3. Trigger the agent handoff

# RESPONSE RULES
1. TONE: Empathetic, professional, reassuring
2. LENGTH: 1-2 sentences of acknowledgment
3. ACTION: Must end with the action tag on a new line
4. CONSTRAINTS:
   - Do NOT try to solve their problem first
   - Do NOT ask clarifying questions
   - Do NOT apologize excessively
   - Do NOT promise specific timeframes

# RESPONSE STRUCTURE
[Your brief, empathetic acknowledgment]

[ACTION:REQUEST_AGENT]

# EXAMPLES OF GOOD ACKNOWLEDGMENTS
- "I understand you'd like to speak with someone from our team. Let me connect you right away."
- "I'll get you to the right person who can help with this immediately."
- "Absolutely, I'm connecting you with a team member now."

# OUTPUT FORMAT
Provide your acknowledgment, then add the action tag on a new line.

<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return {
            "role": "system",
            "content": template.format(
                personality=personality,
                user_message=user_message or "I need to speak with someone."
            )
        }

    @staticmethod
    def build_scheduler_prompt(
        custom_instruction: Optional[str] = None,
        user_message: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Meeting scheduler prompt with action tag integration.
        Handles various scheduling-related requests.
        """
        personality = custom_instruction or (
            "You are a helpful customer support assistant who efficiently "
            "coordinates meetings and calls."
        )

        template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

# ROLE AND PERSONALITY
{personality}

# SITUATION
The user wants to schedule a meeting, demo, call, or callback.

User message: "{user_message}"

# YOUR TASK
1. Acknowledge their scheduling request positively
2. Indicate you're opening the scheduler
3. Trigger the scheduler interface

# RESPONSE RULES
1. TONE: Efficient, positive, helpful
2. LENGTH: 1-2 sentences of acknowledgment
3. ACTION: Must end with the action tag on a new line
4. CONSTRAINTS:
   - Do NOT ask for their availability
   - Do NOT suggest specific times
   - Do NOT ask clarifying questions about the meeting purpose
   - Do NOT explain how the scheduler works

# RESPONSE STRUCTURE
[Your brief acknowledgment]

[ACTION:SHOW_SCHEDULER]

# EXAMPLES OF GOOD ACKNOWLEDGMENTS
- "I'd be happy to help you schedule that. Let me open the scheduler for you."
- "Perfect! I'm pulling up the scheduling tool now."
- "Great, let's get that meeting set up for you."

# OUTPUT FORMAT
Provide your acknowledgment, then add the action tag on a new line.

<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return {
            "role": "system",
            "content": template.format(
                personality=personality,
                user_message=user_message or "I'd like to schedule a meeting."
            )
        }

    @staticmethod
    def build_fallback_prompt(
        custom_instruction: Optional[str] = None,
        user_message: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Fallback prompt for unclear or out-of-scope requests.
        Gracefully handles edge cases.
        """
        personality = custom_instruction or (
            "You are a customer support assistant who handles unclear "
            "requests with patience and clarity."
        )

        # If fallback_count >= 2, escalate to contact details message
        import inspect
        fallback_count = 0
        # Try to get fallback_count from caller if passed
        frame = inspect.currentframe()
        if frame is not None:
            outer = frame.f_back
            if outer and 'fallback_count' in outer.f_locals:
                fallback_count = outer.f_locals['fallback_count']

        if fallback_count >= 3:
            template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

# ROLE AND PERSONALITY
{personality}

# SITUATION
The user's request is unclear, ambiguous, or outside your scope.

User message: "{user_message}"

# YOUR TASK
Let the user know you are unable to answer after several attempts and politely ask them to provide their contact details so a human team member can follow up.

# RESPONSE RULES
1. TONE: Patient, helpful, non-judgmental
2. LENGTH: 2-3 sentences maximum
3. APPROACH:
   - Acknowledge their message
   - Politely explain you couldn't answer after several tries
   - Ask for contact details for follow-up
4. CONSTRAINTS:
   - Do NOT guess at their intent
   - Do NOT provide generic lists of what you can do
   - Do NOT apologize excessively

# EXAMPLES OF GOOD RESPONSES
- "I'm sorry I couldn't answer your question after several tries. Please provide your contact details and our team will reach out to you."
- "It looks like I wasn't able to help with your request. If you'd like, please share your contact information and a team member will follow up."

# OUTPUT FORMAT
Provide ONLY your direct response to the user.

<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        else:
            template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

# ROLE AND PERSONALITY
{personality}

# SITUATION
The user's request is unclear, ambiguous, or outside your scope.

User message: "{user_message}"

# YOUR TASK
Politely acknowledge and offer to clarify or redirect.

# RESPONSE RULES
1. TONE: Patient, helpful, non-judgmental
2. LENGTH: 2-3 sentences maximum
3. APPROACH:
   - Acknowledge their message
   - Politely ask for clarification OR
   - Offer to connect them with support
4. CONSTRAINTS:
   - Do NOT guess at their intent
   - Do NOT provide generic lists of what you can do
   - Do NOT apologize excessively

# EXAMPLES OF GOOD RESPONSES
- "I want to make sure I understand correctly. Could you provide a bit more detail about what you're looking for?"
- "I'd be happy to help! Could you clarify what specific information you need?"
- "To assist you better, could you let me know more about your question?"

# OUTPUT FORMAT
Provide ONLY your direct response to the user.

<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return {
            "role": "system",
            "content": template.format(
                personality=personality,
                user_message=user_message or ""
            )
        }


def build_augmented_system_instruction(
    history: list[dict],
    user_message: str,
    knowledge_base: Optional[str] = None,
    custom_instruction: Optional[str] = None,
    intent: Optional[str] = None,
    fallback_count: int = 0
) -> Dict[str, any]:
    """
    Automatically route user message to appropriate prompt template.
    
    Args:
        user_message: The user's input message
        knowledge_base: Optional knowledge base for QA tasks
        custom_instruction: Optional custom personality/instructions
        intent: Optional pre-classified intent (if None, will auto-detect)
    
    Returns:
        Dict with 'system_message', 'detected_intent', 'confidence', 'max_tokens'
    """
    from intent_classification import get_hybrid_classifier
    
    prompts = ChatbotPrompts()
    router = get_hybrid_classifier()
    # breakpoint()
    try:
        # Special-case: user explicitly says they have no questions
        normalized = (user_message or "").strip().lower()
        no_question_phrases = [
            "i dont have any questions",
            "i don't have any questions",
            "i do not have any questions",
            "i dont have questions",
            "i don't have questions",
            "no questions",
        ]

        if intent is None and any(p in normalized for p in no_question_phrases):
            detected_intent = "greeting"
            confidence_data = {"greeting": 1.0}
        elif intent is None:
            # Normal intent classification
            detected_intent, confidence_data, _ = router.classify(
                user_message, return_scores=True
            )
        else:
            detected_intent = intent
            confidence_data = {detected_intent: 1.0}
        
        # Confidence threshold check (handle both float and dict)
        if isinstance(confidence_data, dict):
            confidence_score = confidence_data.get(detected_intent, 0.0)
        else:
            confidence_score = float(confidence_data)

        # Rely mainly on the classifier, but still guard against
        # extremely low-confidence non-"normal_qa" routes.
        # For scheduler/agent_request, allow slightly lower scores
        # so requests like "arrange call from sales" don't fall
        # back to normal_qa too aggressively.
        min_confidence = 0.30 if detected_intent in {"scheduler", "agent_request"} else 0.40

        if detected_intent != "normal_qa" and confidence_score < min_confidence:
            logger.warning(
                "Low confidence for intent=%s (%.2f), defaulting to normal_qa",
                detected_intent, confidence_score
            )
            detected_intent = "normal_qa"
        
        logger.info(
            "Intent routing | message='%s' | intent=%s | confidence=%.2f",
            user_message[:50], detected_intent, confidence_score
        )

        # Route to appropriate prompt
        if detected_intent == "normal_qa":
            # If fallback_count >= 2, use fallback prompt directly
            if fallback_count >= 2:
                system_msg = prompts.build_fallback_prompt(
                    custom_instruction, user_message
                )
                max_tokens = 100
            else:
                system_msg = prompts.build_qa_prompt(
                    knowledge_base or "",
                    user_message,
                    custom_instruction,
                    history,
                )
                max_tokens = 600
            
        elif detected_intent == "greeting":
            system_msg = prompts.build_greeting_prompt(
                custom_instruction, user_message
            )
            max_tokens = 80
            
        elif detected_intent == "agent_request":
            system_msg = prompts.build_agent_request_prompt(
                custom_instruction, user_message
            )
            max_tokens = 100
            
        elif detected_intent == "scheduler":
            system_msg = prompts.build_scheduler_prompt(
                custom_instruction, user_message
            )
            max_tokens = 100
            
        else:
            logger.warning("Unknown intent: %s, using fallback", detected_intent)
            system_msg = prompts.build_fallback_prompt(
                custom_instruction, user_message
            )
            max_tokens = 150
            detected_intent = "fallback"

        return {
            "system_message": system_msg,
            "detected_intent": detected_intent,
            "confidence": confidence_data,
            "max_tokens": max_tokens
        }

    except Exception as e:
        logger.exception("Failed to build system instruction for message: %s", user_message)
        raise e


def format_prompt_for_llama3(messages: list[dict], max_chars: int = 8000) -> str:
    """
    Format conversation messages for Llama 3.1 with proper chat template.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        max_chars: Maximum character limit for the prompt
    
    Returns:
        Formatted prompt string ready for Llama 3.1
    """
    prompt = "<|begin_of_text|>"
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        prompt += (
            f"<|start_header_id|>{role}<|end_header_id|>\n\n"
            f"{content}<|eot_id|>"
        )
        
        # Truncation check
        if len(prompt) > max_chars:
            logger.warning(
                "Prompt truncated | original_length=%d | max_chars=%d",
                len(prompt), max_chars
            )
            prompt = prompt[:max_chars]
            break
    
    # Add assistant header for generation
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    
    return prompt