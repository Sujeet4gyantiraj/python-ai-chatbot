
from langchain_pinecone import PineconeVectorStore
from langchain_core.vectorstores import VectorStoreRetriever

def get_pinecone_retriever(index, namespace: str = "", top_k: int = 5, **kwargs):
    """
    Returns a LangChain retriever for Pinecone using the given index, namespace, and extra keyword arguments (e.g., embedding_function).
    """
    vectorstore = PineconeVectorStore(index=index, namespace=namespace, **kwargs)
    retriever = VectorStoreRetriever(vectorstore=vectorstore, search_kwargs={"k": top_k})
    return retriever



# Example usage:

def retrieve_relevant_docs(index, query: str, namespace: str = "", top_k: int = 5, embedding_function=None):
    """
    Retrieve relevant documents from Pinecone using LangChain retriever and a custom embedding function.
    """
    retriever = get_pinecone_retriever(index, namespace, top_k, embedding_function=embedding_function)
    docs = retriever.get_relevant_documents(query)
    return docs