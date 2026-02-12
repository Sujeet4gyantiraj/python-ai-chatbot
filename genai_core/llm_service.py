import os

from . import hf_config  # noqa: F401
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.agents import create_agent
from .tools import get_user_ticket, create_ticket, web_search


class LLMService:
    """LangChain-based LLM service using HuggingFaceEndpoint.

    This replaces the previous vLLM-backed implementation and relies on
    Hugging Face Inference (or compatible endpoint) configured via
    HF_TOKEN and repo_id.
    """

    def __init__(self):
        # Allow overriding the model via env, fall back to previous default.
        model_id = os.getenv("GENAI_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")

        # Default generation settings; can be overridden per-call.
        # Note: we intentionally do NOT pass max_new_tokens to the
        # Hugging Face chat API, because some providers (e.g. novita)
        # don't support that argument on chat_completion.
        self.default_max_new_tokens = int(os.getenv("GENAI_MAX_TOKENS", "1024"))
        self.default_temperature = float(os.getenv("GENAI_TEMPERATURE", "0.3"))
        self.default_top_p = float(os.getenv("GENAI_TOP_P", "0.9"))

        print(f"Loading LangChain HuggingFaceEndpoint chat model: {model_id} ...")

        # Use conversational task; novita only supports this for this model.
        base_llm = HuggingFaceEndpoint(
            repo_id=model_id,
            temperature=self.default_temperature,
            top_p=self.default_top_p,
            task="text-generation",
        )

        self.chat_model = ChatHuggingFace(llm=base_llm)
        # Tools available for agent-style workflows
        self.tools = [web_search, get_user_ticket, create_ticket]
        self.tool_system_prompt = (
            "You are an assistant that can answer questions and use tools. "
            "You have access to web_search for browsing the internet and "
            "ticket tools for support workflows. Decide when a tool is "
            "actually needed; otherwise answer directly."
        )

    def _convert_messages(self, messages_data):
        """Convert list[dict] into LangChain message objects for chat model."""
        messages = []

        if isinstance(messages_data, list):
            for m in messages_data:
                if not isinstance(m, dict):
                    continue
                role = m.get("role", "user")
                content = m.get("content", "")

                if role == "system":
                    messages.append(SystemMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
                else:
                    messages.append(HumanMessage(content=content))
        else:
            messages.append(HumanMessage(content=str(messages_data)))

        return messages

    def generate_without_tools(self, messages_data, max_tokens: int, temperature: float, top_p: float):
        """Generate text for a chat conversation (system + history + user)."""
        chat = self.chat_model.bind(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        
        messages = self._convert_messages(messages_data)
        # breakpoint()
        response = chat.invoke(messages)
         
        text_out = getattr(response, "content", str(response)).strip()
        print("Converted messages for LLMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMm:", self.chat_model)
        return {
            "generated_text": text_out,
            "finish_reason": "stop",
        }

    def generate_with_tools(self, messages_data, max_tokens: int, temperature: float, top_p: float):
        """Use an agent with tools (web_search, tickets) when no KB context.

        Expects the same messages_data format as `generate`; it will
        extract the latest user message and run the tool-enabled agent
        only on that input.
        """

        # Extract last user message content
        user_message = ""
       
        if isinstance(messages_data, list):
            for m in reversed(messages_data):
                if isinstance(m, dict) and m.get("role") == "user":
                    user_message = str(m.get("content", ""))
                    break

        if not user_message:
            # Fallback: stringify entire messages_data
            user_message = str(messages_data)

        llm = self.chat_model.bind(
            temperature=temperature,
            top_p=top_p,
        )
     
        agent = create_agent(model=llm,tools=self.tools,system_prompt=self.tool_system_prompt)

        result = agent.invoke({"input": user_message})

        # Normalize agent result to plain text
        if hasattr(result, "content"):
            text_out = result.content
        else:
            text_out = str(result)

        text_out = text_out.strip()

        return {
            "generated_text": text_out,
            "finish_reason": "stop",
        }
