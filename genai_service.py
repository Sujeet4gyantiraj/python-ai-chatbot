# genai_service.py
# FastAPI GenAI service using Pinecone for RAG (logic unchanged)

import logging
from typing import List, Optional, Dict, Any
import re
import asyncio
import os

from dotenv import load_dotenv
from pinecone import Pinecone

from genai_core import EmbeddingService, LLMService
from utils.prompt_builder import (
    build_augmented_system_instruction,
    format_prompt_for_llama3,
    # get_memory
)
from utils.redis_client import (
    load_chat_history,
    save_chat_history,
    save_contact_details,
    load_contact_details,
)

# ------------------------------------------------------------------
# LOGGING SETUP
# ------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
)

logger = logging.getLogger("genai_service")

# ------------------------------------------------------------------

load_dotenv()

# Pinecone Setup
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX")
pinecone_index = pc.Index(PINECONE_INDEX_NAME)

EMBEDDING_DIM = 768

# Local model services (loaded once per process)
_embedding_service = EmbeddingService()
_llm_service = LLMService()


def get_models_health() -> Dict[str, Any]:
    """Return a lightweight health summary for local LLM and embedding models."""
    status: Dict[str, Any] = {
        "llm": {
            "loaded": _llm_service is not None,
            "class": type(_llm_service).__name__ if _llm_service is not None else None,
        },
        "embedding": {
            "loaded": _embedding_service is not None,
            "class": type(_embedding_service).__name__ if _embedding_service is not None else None,
            "device": getattr(_embedding_service, "device", None) if _embedding_service is not None else None,
        },
    }

    return status


# ------------------------------------------------------------------
# Embedding API
# ------------------------------------------------------------------
async def generate_embedding(
    text: str,
    task_type: str = "search_query"
) -> List[float]:
    if not text or len(text.strip()) < 3:
        logger.warning("Embedding skipped due to short/empty text")
        return [0.0] * EMBEDDING_DIM
    loop = asyncio.get_running_loop()

    def _run_embedding() -> List[float]:
        emb = _embedding_service.generate(text=text, task_type=task_type)

        if isinstance(emb, list) and emb and isinstance(emb[0], list):
            emb_flat = emb[0]
        else:
            emb_flat = emb

        if not isinstance(emb_flat, list):
            raise TypeError("EmbeddingService returned invalid format")

        return [float(x) for x in emb_flat]

    try:
        emb = await loop.run_in_executor(None, _run_embedding)
        logger.debug("Embedding generated successfully | dim=%d", len(emb))
        return emb
    except Exception:
        logger.exception("Local embedding generation failed")
        raise


embed_query = generate_embedding


def clean_llm_output(text: str) -> str:
        """Post-process raw LLM output.

        - Strip any internal action markers (e.g. ``[ACTION:REQUEST_AGENT]``
            or ``ACTION:\nSHOW_SCHEDULER``) so they are not shown to the user.
        - Stop at explicit end markers like ``END RESPONSE`` but **do not**
            treat a plain ``#`` as an end marker.
        """

        # If an ACTION marker appears, keep only the text before it.
        # This covers both "[ACTION:XYZ]" and multi-line forms like
        # "ACTION:\nXYZ".
        text = re.split(r"\[?ACTION:", text, maxsplit=1)[0]

        # Also strip any leftover [ACTION:...] blocks just in case
        text = re.sub(r"\[ACTION:.*?\]", "", text, flags=re.DOTALL)

        # Stop only at explicit textual end markers, not on every '#'
        text = re.split(
                r"END CONVERSATION|END RESPONSE|END SESSION",
                text,
                maxsplit=1,
        )[0]

        return text.strip()


# ------------------------------------------------------------------
# Chat Generation API
# ------------------------------------------------------------------
async def generate_chat_response(
    messages: list[dict],
    max_tokens: int = 2000,
    temperature: float = 0.3,
    top_p: float = 0.4,
) -> str:
    loop = asyncio.get_running_loop()

    def _run_llm() -> str:
        try:
            result = _llm_service.generate_without_tools(
                messages_data=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            # result = _llm_service.generate_with_tools(
            #     messages_data=messages,
            #     max_tokens=max_tokens,
            #     temperature=temperature,
            #     top_p=top_p,
            # )
        except Exception as e:
            logger.error("Local LLM error: %s", e)
            raise

        text = (result or {}).get("generated_text", "")
        return text.strip()

    try:
        response_text = await loop.run_in_executor(None, _run_llm)
        if not response_text:
            return (
                "I'm sorry, I couldn't generate a response. "
                "Please request human assistance."
            )

        logger.debug("LLM response generated successfully")
        return response_text

    except asyncio.CancelledError:
        logger.warning("Request cancelled")
        raise
    except Exception:
        logger.exception("Unexpected local LLM error")
        return (
            "An unexpected error occurred. "
            "Please request human assistance."
        )


# ------------------------------------------------------------------
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
    match = re.match(r"^(.*?)\s*#", text)
    return match.group(1).strip() if match else text.strip()


# ------------------------------------------------------------------
# MAIN RAG FUNCTION
# ------------------------------------------------------------------

# ---- helpers ----
def _norm(text: str) -> str:
    """Lowercase, collapse whitespace."""
    return " ".join(text.lower().split())


def _phrase_match(phrase: str, text: str) -> bool:
    """Check if *phrase* appears in *text* as whole words (word-boundary safe).

    Avoids false positives like 'no' matching inside 'know'.
    """
    return bool(re.search(r'\b' + re.escape(phrase) + r'\b', text))


def _extract_phone(text: str) -> Optional[str]:
    """Extract and return a valid Indian mobile number (10 digits, starts with 6-9).

    Accepts formats like: 9876543210, +919876543210, 91-9876543210,
    98765 43210, +91 98765-43210, etc.
    Returns the clean 10-digit number or None if invalid.
    """
    # Strip everything except digits
    digits = re.sub(r"\D", "", text)

    # Remove leading country code 91 if present (resulting in 12 digits)
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    # Also handle 0-prefix trunk code (e.g. 09876543210)
    if len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]

    # Must be exactly 10 digits starting with 6, 7, 8, or 9
    if len(digits) == 10 and digits[0] in "6789":
        return digits
    return None


def _has_digit_intent(text: str) -> bool:
    """Return True if the user message looks like it's trying to provide a phone number
    (contains ≥6 consecutive-ish digits), even if the number is invalid."""
    digits = re.findall(r"\d", text)
    return len(digits) >= 6


def _extract_email(text: str) -> Optional[str]:
    m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return m.group(0) if m else None


def _phone_from_dict(d: Optional[Dict[str, Any]]) -> Optional[str]:
    """Try common key names for a phone number in a dict and validate as Indian mobile."""
    if not d or not isinstance(d, dict):
        return None
    raw = (
        d.get("phone")
        or d.get("phone_number")
        or d.get("contact_number")
        or d.get("mobile")
    )
    if not raw:
        return None
    # Validate through Indian mobile number check
    return _extract_phone(str(raw))


async def generate_and_stream_ai_response(
    bot_id: str,
    session_id: str,
    user_query: str,
    ai_node_data: Optional[Dict[str, Any]] = None,
    tenant_name: Optional[str] = None,
    tenant_description: Optional[str] = None,
    user_details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[str]]:

    logger.info(
        "New chat request | bot_id=%s | session_id=%s",
        bot_id,
        session_id,
    )

    # If user details are provided in the payload, persist them
    # Only save when the payload actually contains non-null values
    if user_details and any(v for v in user_details.values() if v):
        try:
            # Merge with existing stored contact so we don't overwrite
            # phone collected during chat with an empty payload
            existing = None
            try:
                existing = await load_contact_details(bot_id, session_id)
            except Exception:
                pass
            merged = dict(existing) if existing else {}
            for k, v in user_details.items():
                if v:  # only overwrite with non-null values
                    merged[k] = v
            await save_contact_details(bot_id, session_id, merged)
            logger.info("User details saved from payload | bot_id=%s | session_id=%s", bot_id, session_id)
        except Exception:
            logger.exception("Failed to save user details from payload")

    # Entertainment/joke/story detection phrases (used in intercept and fallback)
    _entertainment_phrases = [
        "tell me a joke", "joke", "make me laugh", "funny story",
        "tell me a story", "story", "entertain me", "make me smile",
        "say something funny", "say a joke", "give me a joke",
        "share a joke", "share a story", "tell joke", "tell story",
        "can you joke", "can you tell a joke", "can you tell me a joke",
        "can you tell a story", "can you tell me a story",
        "do you know a joke", "do you know any jokes", "do you know a story",
        "do you know any stories",
    ]

    try:
        full_text = ""
        clean_text = ""
        action = None

        try:
            # ============================================================
            # 1. LOAD HISTORY + STORED CONTACT DETAILS
            # ============================================================
            history = await load_chat_history(bot_id, session_id, k=10)

            try:
                stored_contact = await load_contact_details(bot_id, session_id)
            except Exception:
                stored_contact = None

            # Determine if we already know the user's phone
            known_phone = (
                _phone_from_dict(stored_contact)
                or _phone_from_dict(user_details)
            )

            # Determine visitor name from payload or stored contact
            visitor_name: Optional[str] = None
            if user_details and isinstance(user_details, dict):
                visitor_name = user_details.get("name") or user_details.get("Name")
            if not visitor_name and stored_contact and isinstance(stored_contact, dict):
                visitor_name = stored_contact.get("name") or stored_contact.get("Name")
            if visitor_name:
                visitor_name = visitor_name.strip()

            # FALLBACK: If no phone in Redis or payload, scan chat history
            # for a phone the user previously provided in conversation
            if not known_phone and history:
                # Strategy 1: Look for bot messages that echo a confirmed number
                for h in reversed(history):
                    if not isinstance(h, dict) or h.get("role") != "assistant":
                        continue
                    nc = _norm(str(h.get("content", "")))
                    num_match = re.search(
                        r"(?:contact number (?:as|to)|your number (?:as|to))\s+(\d{8,})",
                        nc,
                    )
                    if num_match:
                        known_phone = num_match.group(1)
                        break

                # Strategy 2: Look for any valid Indian phone sent by the user
                # (most recent one wins, regardless of what bot said before it)
                if not known_phone:
                    for h in reversed(history):
                        if not isinstance(h, dict) or h.get("role") != "user":
                            continue
                        phone_in_msg = _extract_phone(str(h.get("content", "")))
                        if phone_in_msg:
                            known_phone = phone_in_msg
                            break

                # Persist recovered phone to Redis so future calls find it directly
                if known_phone:
                    try:
                        save_payload = dict(stored_contact) if stored_contact else {}
                        save_payload["phone"] = known_phone
                        await save_contact_details(bot_id, session_id, save_payload)
                    except Exception:
                        pass

            # Check if contact was CONFIRMED in this session
            # (user said "yes" to confirmation → bot replied with "thank you for confirming")
            # Only then do we stop showing the number for re-confirmation
            _contact_confirmed_sigs = [
                "thank you for confirming your number",
                "we've received your details",
                "we have received your details",
                "a representative will be in touch",
            ]
            contact_already_collected = False
            # Also check the Redis confirmed flag
            if stored_contact and isinstance(stored_contact, dict) and stored_contact.get("confirmed"):
                contact_already_collected = True
            else:
                for h in (history or []):
                    if (
                        isinstance(h, dict)
                        and h.get("role") == "assistant"
                        and any(sig in _norm(str(h.get("content", ""))) for sig in _contact_confirmed_sigs)
                    ):
                        contact_already_collected = True
                        break

            norm_query = _norm(user_query) if user_query else ""

            # Helper: find last assistant message
            last_assistant_msg: Optional[str] = None
            for h in reversed(history or []):
                if isinstance(h, dict) and h.get("role") == "assistant":
                    c = str(h.get("content", "")).strip()
                    if c:
                        last_assistant_msg = c
                        break
            norm_last = _norm(last_assistant_msg) if last_assistant_msg else ""

            # ============================================================
            # 2. INTERCEPT: Confirmation of stored number
            #    (bot asked "Is <number> correct?")
            # ============================================================
            confirm_markers = [
                "we already have your contact number as",
                "we've updated your contact number to",
                "is this correct",
            ]
            if any(m in norm_last for m in confirm_markers):
                # --- YES / CONFIRM ---
                yes_phrases = [
                    "yes", "yeah", "yep", "correct", "right",
                    "its correct", "it's correct", "that is correct",
                    "that's correct", "ok", "okay", "confirmed", "confirm",
                ]
                if any(y in norm_query for y in yes_phrases):
                    # Mark the stored contact as confirmed
                    try:
                        existing = await load_contact_details(bot_id, session_id)
                        if existing and isinstance(existing, dict):
                            existing["confirmed"] = True
                            await save_contact_details(bot_id, session_id, existing)
                    except Exception:
                        logger.exception("Failed to mark contact as confirmed")

                    reply = "Thank you for confirming your number. Our team will connect with you shortly."
                    history.append({"role": "user", "content": user_query})
                    history.append({"role": "assistant", "content": reply})
                    await save_chat_history(bot_id, session_id, history, k=10)
                    return {"fullText": reply, "cleanText": reply, "action": "contact_confirmed"}

                # --- NO / DENY ---
                no_phrases = [
                    "no", "nope", "not correct", "incorrect", "wrong",
                    "not my number", "wrong number", "this is not my number",
                ]
                if any(n in norm_query for n in no_phrases):
                    reply = (
                        "No problem. Please share your correct contact number "
                        "so we can update it and reach you on the right number."
                    )
                    history.append({"role": "user", "content": user_query})
                    history.append({"role": "assistant", "content": reply})
                    await save_chat_history(bot_id, session_id, history, k=10)
                    return {"fullText": reply, "cleanText": reply, "action": "contact_update_requested"}

            # ============================================================
            # 3. INTERCEPT: User providing a phone/email after we asked
            #    (bot asked "please share your contact details" or
            #     "please share your correct contact number")
            # ============================================================
            contact_ask_markers = [
                "please share your contact details",
                "please provide your contact details",
                "please share your correct contact number",
                "if you'd like, please share your contact details",
                "please enter a valid 10-digit indian mobile number",
            ]
            if any(m in norm_last for m in contact_ask_markers):
                phone = _extract_phone(user_query)
                email = _extract_email(user_query)

                # If user tried to type a number but it's invalid, tell them
                if not phone and not email and _has_digit_intent(user_query):
                    reply = (
                        "The mobile number you entered doesn't appear to be valid. "
                        "Please enter a valid 10-digit Indian mobile number "
                        "starting with 6, 7, 8, or 9."
                    )
                    history.append({"role": "user", "content": user_query})
                    history.append({"role": "assistant", "content": reply})
                    await save_chat_history(bot_id, session_id, history, k=10)
                    return {"fullText": reply, "cleanText": reply, "action": "invalid_phone"}

                if phone or email:
                    # Merge new details into existing stored contact
                    contact_payload: Dict[str, Any] = {}
                    try:
                        existing = await load_contact_details(bot_id, session_id)
                        if existing and isinstance(existing, dict):
                            contact_payload = dict(existing)
                    except Exception:
                        pass
                    contact_payload["raw"] = user_query
                    if phone:
                        contact_payload["phone"] = phone
                    if email:
                        contact_payload["email"] = email
                    # Clear the confirmed flag so user must re-confirm
                    contact_payload.pop("confirmed", None)

                    try:
                        await save_contact_details(bot_id, session_id, contact_payload)
                    except Exception:
                        logger.exception("Failed to save contact details")

                    # Show the new number back for confirmation
                    # (cycle continues until user says "yes")
                    saved_num = phone or email
                    reply = (
                        f"Thank you. We've updated your contact number to {saved_num}. "
                        "Is this correct? If not, please share the correct number."
                    )
                    history.append({"role": "user", "content": user_query})
                    history.append({"role": "assistant", "content": reply})
                    await save_chat_history(bot_id, session_id, history, k=10)
                    return {"fullText": reply, "cleanText": reply, "action": "contact_updated"}

            # ============================================================
            # 4. INTERCEPT: Short acknowledgements ("ok", "okay", "thanks")
            #    after contact was saved or confirmed
            # ============================================================
            ack_markers = [
                "ok", "okay", "ok thanks", "ok thank you", "okay thanks",
                "okay thank you", "fine", "ok fine", "okay fine",
                "ok its fine", "ok it's fine", "its fine", "it is fine",
                "thats fine", "that's fine", "fine thanks", "fine thank you",
                "thanks", "thank you", "great", "alright",
            ]
            thank_markers = [
                "our team will contact you",
                "our team will connect with you",
                "our team will reach out to you",
                "we've received your details",
                "we have received your details",
                "a representative will be in touch",
            ]
            if any(m in norm_last for m in thank_markers) and any(_phrase_match(a, norm_query) for a in ack_markers):
                reply = "Great, I'm glad that's all set. If you need anything else, just let me know."
                history.append({"role": "user", "content": user_query})
                history.append({"role": "assistant", "content": reply})
                await save_chat_history(bot_id, session_id, history, k=10)
                return {"fullText": reply, "cleanText": reply, "action": "acknowledgement"}

            # ============================================================
            # 4b. INTERCEPT: Conversation closers after bot's "all set"
            #     or fallback messages — "no", "nothing", "bye", etc.
            # ============================================================
            _all_set_markers = [
                "i'm glad that's all set",
                "im glad thats all set",
                "if you need anything else",
                "feel free to reach out",
                "don't hesitate to come back",
                "i'm here whenever you need",
                "im here whenever you need",
                "i'm always here if you",
                "im always here if you",
            ]
            _close_phrases = [
                "no", "nope", "nah", "bye", "goodbye", "done",
                "nothing", "nothing else", "no need", "not needed",
                "im fine", "i am fine", "im good", "i am good",
                "im done", "i am done", "all done", "thats all",
                "that is all", "thats it", "that is it",
                "no thanks", "no thank you", "never mind",
                "nevermind", "leave it", "forget it",
                "not interested", "not right now", "maybe later",
                "i dont need any answer", "i do not need any answer",
                "i dont need anything", "i do not need anything",
                "i dont want anything", "i do not want anything",
                "no more questions", "no i am ok", "no its ok",
            ]
            if any(m in norm_last for m in _all_set_markers) and any(_phrase_match(c, norm_query) for c in _close_phrases):
                reply = "Take care! I'm here whenever you need help. Have a great day!"
                history.append({"role": "user", "content": user_query})
                history.append({"role": "assistant", "content": reply})
                await save_chat_history(bot_id, session_id, history, k=10)
                return {"fullText": reply, "cleanText": reply, "action": "conversation_close"}

            # ============================================================
            # 4c. INTERCEPT: User wants to change/update contact number
            #     "i want to change", "i want to change my number", etc.
            #     *** MUST run BEFORE "my number" lookup (step 4d) ***
            # ============================================================
            _change_contact_phrases = [
                "i want to change", "i want to update",
                "i want change", "i want update",
                "change my number", "update my number",
                "change my contact", "update my contact",
                "i want to change my number", "i want to update my number",
                "i want to change number", "i want to update number",
                "i want change my number", "i want change number",
                "change number", "update number",
                "want to change", "want to update",
                "want change", "want update",
                "need to change", "need to update",
                "can i change", "can i update",
            ]
            # Check if the context is about contact/number
            _number_context_in_last = any(
                m in norm_last
                for m in [
                    "contact number", "phone number", "your number",
                    "on file for you", "is this correct",
                    "we already have your contact",
                    "is there anything else",
                ]
            )
            if any(p in norm_query for p in _change_contact_phrases) and (
                _number_context_in_last or "number" in norm_query or "contact" in norm_query
                or norm_query in [
                    "i want to change", "i want to update",
                    "i want change", "i want update",
                    "want to change", "want to update",
                    "want change", "want update",
                ]
            ):
                reply = (
                    "No problem. Please share your correct contact number "
                    "so we can update it and reach you on the right number."
                )
                history.append({"role": "user", "content": user_query})
                history.append({"role": "assistant", "content": reply})
                await save_chat_history(bot_id, session_id, history, k=10)
                return {"fullText": reply, "cleanText": reply, "action": "contact_update_requested"}

            # ============================================================
            # 4d. INTERCEPT: User asking for their own info
            #     "tell my name", "what is my name", "tell my number", etc.
            # ============================================================
            _my_name_phrases = [
                "tell my name", "tell me my name", "what is my name",
                "what's my name", "whats my name", "say my name",
                "do you know my name", "my name", "you know my name",
                "remember my name", "who am i",
            ]
            _my_number_phrases = [
                "tell my number", "tell me my number", "what is my number",
                "what's my number", "whats my number", "my number",
                "my phone number", "my contact number", "my phone",
                "do you know my number", "you know my number",
                "what number do you have", "which number do you have",
                "show my number", "show my contact",
            ]

            if any(p in norm_query for p in _my_name_phrases):
                if visitor_name:
                    reply = f"Your name is {visitor_name}. How can I help you today?"
                else:
                    reply = "I don't have your name on file. Could you please share your name?"
                history.append({"role": "user", "content": user_query})
                history.append({"role": "assistant", "content": reply})
                await save_chat_history(bot_id, session_id, history, k=10)
                return {"fullText": reply, "cleanText": reply, "action": "user_info"}

            # Ensure _self_intro_match is defined for the following check
            _self_intro_pattern = re.compile(
                r'^i\s+(?:am|work\s+as|work\s+in|work\s+at|work\s+for)'
                r'\s+(?:a\s+|an\s+)?(.+)$',
                re.IGNORECASE,
            )
            _self_intro_match = _self_intro_pattern.match(user_query.strip())
            _is_pure_statement = _self_intro_match and '?' not in user_query

            # ============================================================
            # 4i. INTERCEPT: Joke/Story/Entertainment requests
            #     "tell me a joke", "tell me a story", "make me laugh", etc.
            #     Respond with support-only message.
            # ============================================================
            _entertainment_patterns = [
                r"\bjoke\b", r"\bstory\b", r"\bfunny\b", r"\bentertain\b", r"\blaugh\b", r"\bsmile\b",
                r"tell me a joke", r"tell me a story", r"make me laugh", r"make me smile",
                r"say something funny", r"say a joke", r"give me a joke", r"share a joke", r"share a story",
                r"tell joke", r"tell story", r"can you joke", r"can you tell a joke", r"can you tell me a joke",
                r"can you tell a story", r"can you tell me a story", r"do you know a joke", r"do you know any jokes",
                r"do you know a story", r"do you know any stories",
            ]
            _entertainment_match = any(re.search(p, user_query.lower()) for p in _entertainment_patterns)
            if _entertainment_match:
                reply = (
                    "I'm here to help with support-related questions. How can I assist you?"
                )
                history.append({"role": "user", "content": user_query})
                history.append({"role": "assistant", "content": reply})
                await save_chat_history(bot_id, session_id, history, k=10)
                return {"fullText": reply, "cleanText": reply, "action": "support_only"}

            if _is_pure_statement:
                role_info = _self_intro_match.group(1).strip()
                name_prefix = f"That's great, {visitor_name}! " if visitor_name else "That's great! "
                reply = (
                    f"{name_prefix}Thanks for sharing that you're a {role_info}. "
                    "How can I assist you today?"
                )
                history.append({"role": "user", "content": user_query})
                history.append({"role": "assistant", "content": reply})
                await save_chat_history(bot_id, session_id, history, k=10)
                return {"fullText": reply, "cleanText": reply, "action": "user_info"}

            # ============================================================
            # 4e. INTERCEPT: User asking about themselves from history
            #     "what is my designation", "what do i do", "what i do"
            # ============================================================
            _my_role_phrases = [
                "what is my designation", "what's my designation",
                "whats my designation", "my designation",
                "what is my role", "what's my role", "whats my role",
                "my role", "what is my job", "what's my job",
                "what do i do", "what i do", "what is my profession",
                "what's my profession", "what is my position",
                "what am i", "tell me my role", "tell me my designation",
                "tell me what i do",
            ]
            if any(p in norm_query for p in _my_role_phrases):
                # Scan history for user's self-introduction
                _intro_re = _re_mod.compile(
                    r'i\s+(?:am|work\s+as|work\s+in|work\s+at|work\s+for)'
                    r'\s+(?:a\s+|an\s+)?(.+)',
                    _re_mod.IGNORECASE,
                )
                found_role: Optional[str] = None
                for h in reversed(history or []):
                    if isinstance(h, dict) and h.get("role") == "user":
                        m = _intro_re.search(str(h.get("content", "")))
                        if m:
                            found_role = m.group(1).strip().rstrip('.')
                            break
                if found_role:
                    reply = f"Based on what you shared earlier, you're a {found_role}. Is there anything else I can help with?"
                else:
                    reply = "I don't have that information yet. Could you tell me about your role or designation?"
                history.append({"role": "user", "content": user_query})
                history.append({"role": "assistant", "content": reply})
                await save_chat_history(bot_id, session_id, history, k=10)
                return {"fullText": reply, "cleanText": reply, "action": "user_info"}

            # ============================================================
            # 4g. INTERCEPT: User proactively providing a phone number
            #     "my new number is 9807079807", "9807079807", "new number 98..."
            #     This catches phone numbers even when the bot didn't ask.
            # ============================================================
            _proactive_phone = _extract_phone(user_query)
            if _proactive_phone:
                # Check if the user is clearly providing/updating a number
                _providing_number_cues = [
                    "new number", "my number is", "my new number",
                    "number is", "my contact is", "my phone is",
                    "update to", "change to", "updated number",
                    "correct number", "right number",
                ]
                # Also catch: bare phone number or number after bot asked
                # for contact update
                _update_context = any(
                    m in norm_last
                    for m in [
                        "please share your correct contact number",
                        "please share your contact details",
                        "please provide your contact details",
                        "please enter a valid",
                        "share the correct number",
                        "update it and reach you",
                    ]
                )
                _has_providing_cue = any(c in norm_query for c in _providing_number_cues)
                # Pure number (just digits, maybe with +91 etc.)
                _is_bare_number = len(re.sub(r'[\s\-\+]', '', user_query).strip()) <= 15 and _has_digit_intent(user_query)

                if _update_context or _has_providing_cue or _is_bare_number:
                    # Save the new number to Redis
                    contact_payload: Dict[str, Any] = {}
                    try:
                        existing = await load_contact_details(bot_id, session_id)
                        if existing and isinstance(existing, dict):
                            contact_payload = dict(existing)
                    except Exception:
                        pass
                    contact_payload["raw"] = user_query
                    contact_payload["phone"] = _proactive_phone
                    contact_payload.pop("confirmed", None)

                    try:
                        await save_contact_details(bot_id, session_id, contact_payload)
                    except Exception:
                        logger.exception("Failed to save proactive phone number")

                    reply = (
                        f"Thank you. We've updated your contact number to {_proactive_phone}. "
                        "Is this correct? If not, please share the correct number."
                    )
                    history.append({"role": "user", "content": user_query})
                    history.append({"role": "assistant", "content": reply})
                    await save_chat_history(bot_id, session_id, history, k=10)
                    return {"fullText": reply, "cleanText": reply, "action": "contact_updated"}

            # ============================================================
            # 4h. INTERCEPT: Gibberish / unintelligible input
            #     "asdkjasd123", "xyzpqr", "hfjksdhf", etc.
            #     Detect and ask user to rephrase instead of
            #     sending garbage to the LLM.
            # ============================================================
            _COMMON_WORDS = {
                'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she',
                'it', 'they', 'them', 'a', 'an', 'the', 'is', 'am', 'are',
                'was', 'were', 'be', 'been', 'do', 'does', 'did', 'have',
                'has', 'had', 'will', 'would', 'can', 'could', 'shall',
                'should', 'may', 'might', 'must', 'to', 'of', 'in', 'on',
                'at', 'for', 'by', 'with', 'from', 'up', 'out', 'about',
                'into', 'over', 'after', 'and', 'but', 'or', 'not', 'no',
                'yes', 'so', 'if', 'then', 'than', 'that', 'this', 'what',
                'which', 'who', 'how', 'when', 'where', 'why', 'all',
                'each', 'every', 'some', 'any', 'few', 'more', 'most',
                'much', 'many', 'very', 'just', 'also', 'too', 'only',
                'own', 'same', 'new', 'old', 'good', 'bad', 'big',
                'small', 'long', 'first', 'last', 'next', 'now', 'here',
                'there', 'back', 'want', 'need', 'know', 'think', 'make',
                'go', 'get', 'come', 'take', 'see', 'look', 'find',
                'give', 'tell', 'say', 'ask', 'use', 'try', 'work',
                'help', 'call', 'talk', 'show', 'let', 'keep', 'put',
                'set', 'run', 'pay', 'buy', 'sell', 'open', 'close',
                'start', 'stop', 'turn', 'move', 'live', 'change',
                'play', 'like', 'love', 'hate', 'feel', 'leave',
                'hi', 'hello', 'hey', 'bye', 'ok', 'okay', 'thanks',
                'thank', 'please', 'sorry', 'sure', 'right', 'well',
                'name', 'number', 'phone', 'email', 'contact', 'service',
                'services', 'order', 'place', 'price', 'cost', 'product',
                'account', 'booking', 'book', 'schedule', 'meeting',
                'appointment', 'time', 'day', 'today', 'tomorrow',
                'week', 'month', 'year', 'date', 'morning', 'evening',
                'night', 'afternoon', 'joke', 'story', 'question',
                'answer', 'information', 'info', 'detail', 'details',
                'update', 'delete', 'create', 'add', 'remove', 'plan',
                'offer', 'available', 'support', 'team', 'agent',
                'human', 'person', 'manager', 'developer', 'engineer',
                'data', 'system', 'website', 'app', 'company', 'business',
                'animal', 'national', 'country', 'city', 'world',
            }

            # Extract words (letters only, min 2 chars to skip "a"/"I")
            _input_words = re.findall(r'[a-zA-Z]{2,}', user_query.lower())
            _is_gibberish = False

            if _input_words:
                _recognized = sum(1 for w in _input_words if w in _COMMON_WORDS)
                _ratio = _recognized / len(_input_words)
                # If less than 30% of words are recognized AND the query
                # is short-ish, treat as gibberish
                if _ratio < 0.30 and len(_input_words) <= 10:
                    _is_gibberish = True
            elif user_query.strip():
                # No alphabetic words at all (pure numbers/symbols beyond
                # a phone number — phone was already caught in step 4g)
                if not _extract_phone(user_query):
                    _is_gibberish = True

            if _is_gibberish:
                reply = "Sorry, I didn't understand that. Could you please rephrase your question?"
                history.append({"role": "user", "content": user_query})
                history.append({"role": "assistant", "content": reply})
                await save_chat_history(bot_id, session_id, history, k=10)
                return {"fullText": reply, "cleanText": reply, "action": "gibberish"}

                # ============================================================
                # 4i. INTERCEPT: Joke/Story/Entertainment requests
                #     "tell me a joke", "tell me a story", "make me laugh", etc.
                #     Respond with support-only message.
                # ============================================================
                _entertainment_phrases = [
                    "tell me a joke", "joke", "make me laugh", "funny story",
                    "tell me a story", "story", "entertain me", "make me smile",
                    "say something funny", "say a joke", "give me a joke",
                    "share a joke", "share a story", "tell joke", "tell story",
                    "can you joke", "can you tell a joke", "can you tell me a joke",
                    "can you tell a story", "can you tell me a story",
                    "do you know a joke", "do you know any jokes", "do you know a story",
                    "do you know any stories",
                ]
                if any(_phrase_match(p, norm_query) for p in _entertainment_phrases):
                    reply = (
                        "I'm here to help with support-related questions. How can I assist you?"
                    )
                    history.append({"role": "user", "content": user_query})
                    history.append({"role": "assistant", "content": reply})
                    await save_chat_history(bot_id, session_id, history, k=10)
                    return {"fullText": reply, "cleanText": reply, "action": "support_only"}
            # ============================================================
            # 5. RAG RETRIEVAL
            # ============================================================
            knowledge_base = ""

            if not ai_node_data or not ai_node_data.get("disableKnowledgeBase"):
                query_embedding = await embed_query(user_query)
                logger.debug("Pinecone query embedding generated")
                res = pinecone_index.query(
                    vector=query_embedding,
                    top_k=8,
                    include_metadata=True,
                    namespace=bot_id,
                )

                logger.debug("Pinecone raw response: %s", res)

                SIMILARITY_THRESHOLD = 0.55
                matches: list = []
                try:
                    if isinstance(res, dict):
                        matches = res.get("matches", []) or []
                    elif hasattr(res, "matches"):
                        matches = list(getattr(res, "matches") or [])
                    elif hasattr(res, "to_dict"):
                        d = res.to_dict()
                        if isinstance(d, dict):
                            matches = d.get("matches", []) or []
                except Exception:
                    matches = []

                if matches:
                    filtered_contents = []
                    for m in matches:
                        if isinstance(m, dict):
                            md = m.get("metadata", {}) or {}
                            content = md.get("content") if isinstance(md, dict) else None
                            score = m.get("score", 0.0) or 0.0
                        else:
                            md = getattr(m, "metadata", {}) or {}
                            content = md.get("content") if isinstance(md, dict) else None
                            score = getattr(m, "score", 0.0) or 0.0
                        if content and score >= SIMILARITY_THRESHOLD:
                            filtered_contents.append(content)

                    if filtered_contents:
                        knowledge_base = "\n\n---\n\n".join(filtered_contents)

                logger.debug("Knowledge base size=%d chars", len(knowledge_base))

            has_kb = bool(knowledge_base.strip())

            # ============================================================
            # 6. INTENT CLASSIFICATION + PROMPT BUILDING
            # ============================================================
            tenant_custom_instruction: Optional[str] = None
            if tenant_name or tenant_description or visitor_name:
                lines: list[str] = [
                    "You are a professional customer support assistant known for clear, accurate, and helpful responses.",
                ]
                lines.append("")
                lines.append("TENANT CONTEXT:")
                if tenant_name:
                    lines.append(f"- Tenant name: {tenant_name}")
                if tenant_description:
                    lines.append(f"- Tenant description: {tenant_description}")
                if visitor_name:
                    lines.append("")
                    lines.append("VISITOR CONTEXT:")
                    lines.append(f"- The visitor's name is: {visitor_name}")
                    lines.append("- Address the visitor by their name naturally in your responses (e.g., 'Hi Sujeet, ...' or 'Sure Sujeet, ...').")
                    lines.append("- Do NOT overuse the name — use it once at the start of your response, not in every sentence.")
                tenant_custom_instruction = "\n".join(lines)

            # Count how many consecutive recent assistant messages were fallbacks
            fallback_count = 0
            _fallback_sigs = [
                "i don't have that information available",
                "i don't have enough information to answer",
                "please provide your contact details and our team will connect",
                "please share your contact details and our team will connect",
            ]
            for h in reversed(history or []):
                if not isinstance(h, dict):
                    continue
                if h.get("role") == "assistant":
                    ac = _norm(str(h.get("content", "")))
                    if any(sig in ac for sig in _fallback_sigs):
                        fallback_count += 1
                    else:
                        break
                elif h.get("role") == "user":
                    continue

            prompt_dict = build_augmented_system_instruction(
                history=history,
                user_message=user_query,
                knowledge_base=knowledge_base,
                custom_instruction=tenant_custom_instruction,
                fallback_count=fallback_count,
            )

            max_tokens = prompt_dict.get("max_tokens", 800)
            action = prompt_dict.get("detected_intent")

            # ── 6b. BOT-IDENTITY QUESTIONS ──────────────────────────
            # "What is your name", "who are you", etc. are about the bot
            # itself, not the knowledge base.  Reclassify as greeting so
            # the LLM answers using tenant_name / custom_instruction.
            _bot_identity_phrases = [
                "what is your name", "what's your name", "whats your name",
                "who are you", "what are you", "tell me your name",
                "what should i call you", "may i know your name",
                "what do you call yourself", "your name please",
                "your name", "ur name", "bot name",
                "i want to your name", "i want your name",
            ]
            if action == "normal_qa" and any(p in norm_query for p in _bot_identity_phrases):
                action = "greeting"
                prompt_dict = build_augmented_system_instruction(
                    history=history,
                    user_message=user_query,
                    knowledge_base=knowledge_base,
                    custom_instruction=tenant_custom_instruction,
                    fallback_count=fallback_count,
                    intent="greeting",
                )

            logger.debug("Detected intent: %s", action)
            print("knowledge_base:ddddddddddddddddddddddddddddddddddddddddddddddddddd", knowledge_base)

            # ============================================================
            # 6c. CONVERSATION CLOSE → respond naturally, no KB/fallback
            # ============================================================
            if action == "conversation_close":
                # Use LLM to generate a short, natural closing response
                messages: list[dict] = [prompt_dict["system_message"]]
                for h in history:
                    if not isinstance(h, dict):
                        continue
                    role = h.get("role")
                    content = h.get("content")
                    if not role or not isinstance(content, str) or not content.strip():
                        continue
                    messages.append({"role": role, "content": content})
                messages.append({"role": "user", "content": user_query})

                close_text = await generate_chat_response(
                    messages,
                    max_tokens=prompt_dict.get("max_tokens", 60),
                    temperature=0.3,
                    top_p=0.9,
                )
                close_text = clean_llm_output(close_text)
                if not close_text:
                    close_text = "No worries! Feel free to reach out anytime you need help."

                history.append({"role": "user", "content": user_query})
                history.append({"role": "assistant", "content": close_text})
                await save_chat_history(bot_id, session_id, history, k=10)
                return {
                    "fullText": close_text,
                    "cleanText": close_text,
                    "action": "conversation_close",
                }

            # ============================================================
            # 7. KB EMPTY + NORMAL_QA → let LLM respond, append contact
            #    details after. (No hardcoded fallback — LLM uses the
            #    FALLBACK PROTOCOL in its system prompt to craft a
            #    natural "I don't have that info" message.)
            # ============================================================
            # Flag used later in Step 9 to append contact-details suffix
            _kb_empty_fallback = (not has_kb and action == "normal_qa")

            # ============================================================
            # 8. LLM GENERATION (KB has content or non-QA intent)
            # ============================================================
            messages: list[dict] = [prompt_dict["system_message"]]

            # Filter history: strip trailing consecutive close/goodbye
            # turns so the LLM doesn't see a pattern of closings and
            # keep generating "Take care! Bye!" for new questions.
            _close_sigs = [
                "take care", "have a great day", "i'm here whenever",
                "im here whenever", "feel free to reach out",
                "don't hesitate", "no worries", "glad that's all set",
                "glad thats all set", "alright, have a great day",
            ]
            _filtered_history = list(history or [])
            while _filtered_history:
                last = _filtered_history[-1]
                if not isinstance(last, dict):
                    _filtered_history.pop()
                    continue
                content = _norm(str(last.get("content", "")))
                role = last.get("role", "")
                # Remove trailing close from assistant OR the user msg
                # that triggered it (e.g., "bye", "ok")
                if role == "assistant" and any(s in content for s in _close_sigs):
                    _filtered_history.pop()
                    # Also pop the user message that preceded it
                    if _filtered_history and isinstance(_filtered_history[-1], dict) and _filtered_history[-1].get("role") == "user":
                        _filtered_history.pop()
                else:
                    break

            for h in _filtered_history:
                if not isinstance(h, dict):
                    continue
                role = h.get("role")
                content = h.get("content")
                if not role or not isinstance(content, str) or not content.strip():
                    continue
                messages.append({"role": role, "content": content})

            messages.append({"role": "user", "content": user_query})

            full_text = await generate_chat_response(
                messages,
                max_tokens=max_tokens,
                temperature=0.2,
                top_p=0.9,
            )

            # ============================================================
            # 9. POST-PROCESSING
            # ============================================================
            clean_text = clean_llm_output(full_text)

            # ── 9a. KB was empty → append contact details to LLM response
            if _kb_empty_fallback and clean_text:
                # Entertainment override: always respond with support-only message for joke/story
                if any(_phrase_match(p, norm_query) for p in _entertainment_phrases):
                    clean_text = "I'm here to help with support-related questions. How can I assist you?"
                    full_text = clean_text
                    action = "support_only"
                else:
                    action = "fallback_msg"
                    # Strip any contact-details sentence the LLM may have
                    # generated so we don't double up.  We work on the
                    # *original* clean_text (not normalized) using a regex
                    # that removes the sentence containing the contact-ask.
                    _contact_strip_re = re.compile(
                        r'[^.]*?'
                        r'(?:share|provide|leave)\s+(?:your\s+)?'
                        r'(?:correct\s+)?(?:contact\s+)?'
                        r'(?:details|information|info|number)'
                        r'[^.]*\.?\s*',
                        re.IGNORECASE,
                    )
                    clean_text = _contact_strip_re.sub('', clean_text).strip()

                    # Also strip stray trailing phrases like
                    # "and our team will connect with you shortly."
                    _trailing_team_re = re.compile(
                        r'\s*(?:and\s+)?(?:our|a|the)\s+team\s+'
                        r'(?:will|can|shall)\s+[^.]*\.?\s*$',
                        re.IGNORECASE,
                    )
                    clean_text = _trailing_team_re.sub('', clean_text).strip()

                    if not clean_text:
                        clean_text = "I'm sorry, I don't have that information available right now."

                    # Append the appropriate contact-details suffix
                    if contact_already_collected:
                        clean_text = (
                            f"{clean_text.rstrip()} "
                            "We already have your contact details on file and our team will reach out to you shortly."
                        )
                    elif known_phone:
                        clean_text = (
                            f"{clean_text.rstrip()} "
                            f"We already have your contact number as {known_phone}. "
                            "Is this correct? If not, please share the correct number so our team can reach out to you."
                        )
                    else:
                        clean_text = (
                            f"{clean_text.rstrip()} "
                            "If you'd like, please share your contact details and our team will connect with you shortly."
                        )
                    full_text = clean_text

            # ── 9b. KB had content but LLM still produced a fallback
            elif clean_text:
                nc = _norm(clean_text)
                fallback_phrases = [
                    "share your contact details and our team will connect with you shortly",
                    "provide your contact details and our team will connect with you shortly",
                    "i don't have enough information to answer that right now",
                    "i don't have that information available in our system right now",
                ]
                if any(fp in nc for fp in fallback_phrases):
                    action = "fallback_msg"

                    # Strip LLM-generated contact sentence first
                    _contact_strip_re2 = re.compile(
                        r'[^.]*?'
                        r'(?:share|provide|leave)\s+(?:your\s+)?'
                        r'(?:correct\s+)?(?:contact\s+)?'
                        r'(?:details|information|info|number)'
                        r'[^.]*\.?\s*',
                        re.IGNORECASE,
                    )
                    stripped = _contact_strip_re2.sub('', clean_text).strip()
                    _trailing_team_re2 = re.compile(
                        r'\s*(?:and\s+)?(?:our|a|the)\s+team\s+'
                        r'(?:will|can|shall)\s+[^.]*\.?\s*$',
                        re.IGNORECASE,
                    )
                    stripped = _trailing_team_re2.sub('', stripped).strip()
                    if stripped:
                        clean_text = stripped

                    if contact_already_collected:
                        clean_text = (
                            f"{clean_text.rstrip()} "
                            "We already have your contact details on file and our team will reach out to you shortly."
                        )
                        full_text = clean_text
                    elif known_phone:
                        clean_text = (
                            f"{clean_text.rstrip()} "
                            f"We already have your contact number as {known_phone}. "
                            "Is this correct? If not, please share the correct number so our team can reach out to you."
                        )
                        full_text = clean_text

            # ============================================================
            # 10. UNSATISFIED USER HANDLING
            # ============================================================
            unsatisfied_phrases = [
                "no i want other services", "no i want more services",
                "no i want new services", "no this is not what i asked",
                "no this is not what i want", "this is not what i asked",
                "this is not what i want", "this does not answer my question",
                "you didn't answer my question", "you did not answer my question",
                "this is not helpful", "not helpful", "not satisfied",
            ]

            is_unsatisfied = any(p in norm_query for p in unsatisfied_phrases)

            if not is_unsatisfied:
                last_user_texts = [
                    str(h.get("content", "")).strip()
                    for h in (history or [])
                    if isinstance(h, dict) and h.get("role") == "user"
                ][-3:]
                for txt in last_user_texts:
                    if any(p in _norm(txt) for p in unsatisfied_phrases):
                        is_unsatisfied = True
                        break

            if not is_unsatisfied and norm_query:
                wants_more = any(
                    p in norm_query
                    for p in ["i want new services", "i want other services", "i want more services"]
                )
                if wants_more:
                    for h in history or []:
                        if isinstance(h, dict) and h.get("role") == "assistant":
                            if "services" in _norm(str(h.get("content", ""))):
                                is_unsatisfied = True
                                break

            if is_unsatisfied and clean_text:
                nc = _norm(clean_text)
                if "please provide your contact details" not in nc and "please share your contact details" not in nc:
                    if contact_already_collected:
                        clean_text = (
                            f"{clean_text.rstrip()} "
                            "We already have your contact details on file and our team will reach out to you shortly."
                        )
                    elif known_phone:
                        clean_text = (
                            f"{clean_text.rstrip()} "
                            f"We already have your contact number as {known_phone}. "
                            "Is this correct? If not, please share the correct number so our team can reach out to you."
                        )
                    else:
                        clean_text = (
                            f"{clean_text.rstrip()} "
                            "Please provide your contact details and our team will connect with you shortly."
                        )
                action = "fallback_msg"

            # ============================================================
            # 11. GREETING CLEANUP
            # ============================================================
            if action == "greeting":
                clean_text = extract_before_hash(clean_text)

            # ============================================================
            # 12. SAVE HISTORY & RETURN
            # ============================================================
            history.append({"role": "user", "content": user_query})
            history.append({"role": "assistant", "content": clean_text})
            await save_chat_history(bot_id, session_id, history, k=10)

            logger.info("Chat response generated successfully")
            return {
                "fullText": full_text,
                "cleanText": clean_text,
                "action": action,
            }

        except Exception:
            logger.exception("GENAI inner processing error")
            return {"fullText": "", "cleanText": "", "action": None}

    except Exception:
        logger.exception("GENAI outer error")
        return {"fullText": "", "cleanText": "", "action": None}






