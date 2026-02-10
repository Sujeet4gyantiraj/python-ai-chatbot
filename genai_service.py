
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
            if any(m in norm_last for m in thank_markers) and any(a in norm_query for a in ack_markers):
                reply = "Great, I'm glad that's all set. If you need anything else, just let me know."
                history.append({"role": "user", "content": user_query})
                history.append({"role": "assistant", "content": reply})
                await save_chat_history(bot_id, session_id, history, k=10)
                return {"fullText": reply, "cleanText": reply, "action": "acknowledgement"}

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

            # ============================================================
            # 7. KB EMPTY + NORMAL_QA → FALLBACK (no LLM call needed)
            # ============================================================
            if not has_kb and action == "normal_qa":
                # ── Rotating fallback pools so the user never sees the
                #    exact same message twice in a row.
                name_hi = f"{visitor_name}, " if visitor_name else ""
                name_hi_lower = name_hi.lower()  # for mid-sentence use

                _no_info_variants = [
                    f"I'm sorry {name_hi_lower}I don't have that information available in our system right now.",
                    f"Unfortunately {name_hi_lower}that isn't something I'm able to answer at the moment.",
                    f"I appreciate your question{', ' + visitor_name if visitor_name else ''}! Unfortunately, that's outside the information I currently have access to.",
                    f"Thanks for asking{', ' + visitor_name if visitor_name else ''}. I'm not able to find an answer to that in our system right now.",
                    f"That's a great question{', ' + visitor_name if visitor_name else ''}. However, I don't have the details to answer it at this time.",
                ]

                _ask_contact_variants = [
                    "If you'd like, please share your contact details and our team will connect with you shortly.",
                    "Would you like to share your contact information so our team can get back to you with an answer?",
                    "Please feel free to leave your contact details and a team member will follow up with you.",
                    "You're welcome to provide your contact info and we'll have the right person reach out to you.",
                    "If you share your contact details, our team will be happy to assist you further.",
                ]

                _already_have_contact_variants = [
                    "We already have your contact details on file and our team will reach out to you shortly.",
                    "Don't worry — we have your contact information and someone from our team will get back to you soon.",
                    "Our team already has your details and will be in touch with you shortly.",
                    "We've noted your contact information and a team member will follow up soon.",
                    "Rest assured, we have your details on record and our team will connect with you.",
                ]

                idx = fallback_count % len(_no_info_variants)

                if known_phone:
                    fallback_reply = (
                        f"{_no_info_variants[idx]} "
                        f"We already have your contact number as {known_phone}. "
                        "Is this correct? If not, please share the correct number so our team can reach out to you."
                    )
                elif contact_already_collected:
                    fallback_reply = (
                        f"{_no_info_variants[idx]} "
                        f"{_already_have_contact_variants[idx]}"
                    )
                else:
                    fallback_reply = (
                        f"{_no_info_variants[idx]} "
                        f"{_ask_contact_variants[idx]}"
                    )

                history.append({"role": "user", "content": user_query})
                history.append({"role": "assistant", "content": fallback_reply})
                await save_chat_history(bot_id, session_id, history, k=10)

                return {
                    "fullText": fallback_reply,
                    "cleanText": fallback_reply,
                    "action": "fallback_msg",
                }

            # ============================================================
            # 8. LLM GENERATION (KB has content or non-QA intent)
            # ============================================================
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

            # Detect if LLM itself produced a fallback-style message
            if clean_text:
                nc = _norm(clean_text)
                fallback_phrases = [
                    "share your contact details and our team will connect with you shortly",
                    "provide your contact details and our team will connect with you shortly",
                    "i don't have enough information to answer that right now",
                    "i don't have that information available in our system right now",
                ]
                if any(fp in nc for fp in fallback_phrases):
                    action = "fallback_msg"

                    # Use the same rotating pool as Step 7
                    name_hi = f"{visitor_name}, " if visitor_name else ""
                    name_hi_lower = name_hi.lower()
                    _no_info_v = [
                        f"I'm sorry {name_hi_lower}I don't have that information available in our system right now.",
                        f"Unfortunately {name_hi_lower}that isn't something I'm able to answer at the moment.",
                        f"I appreciate your question{', ' + visitor_name if visitor_name else ''}! Unfortunately, that's outside the information I currently have access to.",
                        f"Thanks for asking{', ' + visitor_name if visitor_name else ''}. I'm not able to find an answer to that in our system right now.",
                        f"That's a great question{', ' + visitor_name if visitor_name else ''}. However, I don't have the details to answer it at this time.",
                    ]
                    _contact_v = [
                        "We already have your contact details on file and our team will reach out to you shortly.",
                        "Don't worry — we have your contact information and someone from our team will get back to you soon.",
                        "Our team already has your details and will be in touch with you shortly.",
                    ]
                    ix = fallback_count % len(_no_info_v)

                    if known_phone:
                        clean_text = (
                            f"{_no_info_v[ix]} "
                            f"We already have your contact number as {known_phone}. "
                            "Is this correct? If not, please share the correct number so our team can reach out to you."
                        )
                        full_text = clean_text
                    elif contact_already_collected:
                        clean_text = (
                            f"{_no_info_v[ix]} "
                            f"{_contact_v[ix % len(_contact_v)]}"
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
                    if known_phone:
                        clean_text = (
                            f"{clean_text.rstrip()} "
                            f"We already have your contact number as {known_phone}. "
                            "Is this correct? If not, please share the correct number so our team can reach out to you."
                        )
                    elif contact_already_collected:
                        clean_text = (
                            f"{clean_text.rstrip()} "
                            "We already have your contact details on file and our team will reach out to you shortly."
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






