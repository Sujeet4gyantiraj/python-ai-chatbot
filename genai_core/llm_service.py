import os

from . import hf_config  # noqa: F401
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage


class LLMService:
    """LangChain-based LLM service using HuggingFaceEndpoint.

    This replaces the previous vLLM-backed implementation and relies on
    Hugging Face Inference (or compatible endpoint) configured via
    HUGGINGFACEHUB_API_TOKEN and repo_id.
    """

    def __init__(self):
        # Allow overriding the model via env, fall back to previous default.
        model_id = os.getenv("GENAI_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")

        # Default generation settings; can be overridden per-call.
        # Note: we intentionally do NOT pass max_new_tokens to the
        # Hugging Face chat API, because some providers (e.g. novita)
        # don't support that argument on chat_completion.
        self.default_max_new_tokens = int(os.getenv("GENAI_MAX_TOKENS", "512"))
        self.default_temperature = float(os.getenv("GENAI_TEMPERATURE", "0.6"))
        self.default_top_p = float(os.getenv("GENAI_TOP_P", "0.9"))

        print(f"Loading LangChain HuggingFaceEndpoint chat model: {model_id} ...")

        # Use the conversational task since meta-llama/3.1 instruct models on
        # Hugging Face Inference are exposed as chat/conversational endpoints.
        base_llm = HuggingFaceEndpoint(
            repo_id=model_id,
            temperature=self.default_temperature,
            top_p=self.default_top_p,
            task="conversational",
        )

        self.chat_model = ChatHuggingFace(llm=base_llm)

    def generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float):
        """Generate text for a single prompt.

        Returns a dict aligned with the previous vLLM-based contract.
        """
        # Build a temporary chat model with per-request params while
        # reusing the underlying repo_id and provider configuration.
        #
        # We avoid passing max_new_tokens here because some providers
        # don't accept it on chat_completion; instead, rely on the
        # model's own default max tokens.
        chat = self.chat_model.bind(
            temperature=temperature,
            top_p=top_p,
        )

        # Our upstream code already formats a full prompt string; treat it as
        # a single user message for the conversational endpoint.
        message = HumanMessage(content=prompt)
        response = chat.invoke([message])

        text_out = getattr(response, "content", str(response)).strip()

        return {
            "generated_text": text_out,
            "finish_reason": "stop",
        }
