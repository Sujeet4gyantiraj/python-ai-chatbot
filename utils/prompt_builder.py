from langchain.prompts import PromptTemplate

def build_websocket_prompt(context: str, question: str) -> str:
    """
    Build a prompt for answering a question about WebSockets using provided context.
    """
    template = (
        """
        You are a helpful assistant. Use ONLY the following CONTEXT to answer the user's question.\n\n"
        "CONTEXT:\n{context}\n\n"
        "QUESTION: {question}\n\n"
        "If the answer is not in the context, say: 'Sorry, I don't know based on the provided information.'"
        """
    )
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )
    return prompt.format(context=context, question=question)
