import os
from fastapi import HTTPException
from dotenv import load_dotenv
from fastapi import UploadFile, HTTPException
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Depends,
    HTTPException,
    APIRouter,
)
from fastapi.responses import JSONResponse
from knowledgesource import upload_knowledge
from fastapi import status
from pinecone_client import index
from genai_service import generate_and_stream_ai_response
from models import GenerateAIRequest

load_dotenv()


# App & Router Setup
app = FastAPI(title="RAG API", version="1.0",root_path="/ragai")
router = APIRouter()


GENAI_API_BASE_URL = os.getenv("GENAI_API_BASE_URL")



# UPLOAD KNOWLEDGE ENDPOINT
@app.post("/bots/{bot_id}/knowledge/upload")
async def upload_knowledge_endpoint(
    knowledge_source_id: str,
    bot_id: str,
    files: list[UploadFile] = File(...),
):
    return await upload_knowledge(
        knowledge_source_id=knowledge_source_id,
        bot_id=bot_id,
        files=files,
    )




# GENAI RAG ENDPOINT
@router.post("/generate-ai-response")
async def generate_ai_response_endpoint(req: GenerateAIRequest):
    result = await generate_and_stream_ai_response(
        bot_id=req.bot_id,
        session_id=req.session_id,
        user_query=req.user_query,
        ai_node_data=req.ai_node_data,
    )
    return JSONResponse(content=result)


app.include_router(router)


# DELETE KNOWLEDGE FROM PINECONE BY SOURCE ID
@app.delete("/knowledge/pinecone/{source_id}", status_code=status.HTTP_200_OK)
def delete_knowledge_from_pinecone(source_id: str, bot_id: str):
    """
    Delete all vectors in Pinecone with metadata sourceId == source_id for the given bot_id namespace.
    """
    try:
        # Pinecone delete by metadata filter
        index.delete(
            filter={"sourceId": source_id},
            namespace=bot_id
        )
        return {"deleted": True, "source_id": source_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone delete failed: {str(e)}")

