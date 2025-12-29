import aiofiles
from langchain_text_splitters import RecursiveCharacterTextSplitter

async def extract_chunks(file_path: str) -> list[str]:
    async with aiofiles.open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = await f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50
    )

    return [
        chunk for chunk in splitter.split_text(text)
        if len(chunk.strip()) >= 20
    ]
