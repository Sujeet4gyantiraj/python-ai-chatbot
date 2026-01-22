
def extract_chunks_from_text(text: str) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    return [chunk for chunk in splitter.split_text(text) if len(chunk.strip()) >= 20]