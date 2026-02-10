import os
import logging
from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    APIRouter,
    status,
)
from fastapi.responses import JSONResponse

from knowledgesource import upload_knowledge
from pinecone_client import index
from genai_service import generate_and_stream_ai_response, get_models_health
from models import GenerateAIRequest


# -------------------- LOGGER SETUP --------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
)

logger = logging.getLogger(__name__)
# ----------------------------------------------------


load_dotenv()


# App & Router Setup
app = FastAPI(title="RAG API", version="1.0", root_path="/ragai")
router = APIRouter()


@app.get("/health/genai")
def genai_health_check():
    """Health check for local LLM and embedding models."""
    try:
        status_payload = get_models_health()
        return {
            "status": "ok",
            "models": status_payload,
        }
    except Exception as e:
        logger.exception("GenAI health check failed")
        raise HTTPException(status_code=500, detail=str(e))


# -------------------- UPLOAD KNOWLEDGE ENDPOINT --------------------
@app.post("/bots/{bot_id}/knowledge/upload")
async def upload_knowledge_endpoint(
    knowledge_source_id: str,
    bot_id: str,
    files: list[UploadFile] = File(...),
):
    logger.info(
        "Knowledge upload requested | bot_id=%s | knowledge_source_id=%s | files=%d",
        bot_id,
        knowledge_source_id,
        len(files),
    )

    return await upload_knowledge(
        knowledge_source_id=knowledge_source_id,
        bot_id=bot_id,
        files=files,
    )


# -------------------- GENAI RAG ENDPOINT --------------------
@router.post("/generate-ai-response")
async def generate_ai_response_endpoint(req: GenerateAIRequest):
    logger.info(
        "Generate AI response requested | bot_id=%s | session_id=%s",
        req.bot_id,
        req.session_id,
    )

    result = await generate_and_stream_ai_response(
        bot_id=req.bot_id,
        session_id=req.session_id,
        user_query=req.user_query,
        ai_node_data=req.ai_node_data,
        tenant_name=req.tenant_name,
        tenant_description=req.tenant_description,
        user_details=(req.user_details.model_dump() if req.user_details else None),
    )

    logger.info(
        "Generate AI response completed | bot_id=%s | session_id=%s",
        req.bot_id,
        req.session_id,
    )

    return JSONResponse(content=result)


app.include_router(router)


# -------------------- DELETE KNOWLEDGE FROM PINECONE --------------------
@app.delete("/knowledge/pinecone/{source_id}", status_code=status.HTTP_200_OK)
def delete_knowledge_from_pinecone(source_id: str, bot_id: str):
    """
    Delete all vectors in Pinecone with metadata sourceId == source_id
    for the given bot_id namespace.
    """
    logger.info(
        "Pinecone delete requested | source_id=%s | bot_id=%s",
        source_id,
        bot_id,
    )

    try:
        index.delete(
            filter={"sourceId": source_id},
            namespace=bot_id
        )

        logger.info(
            "Pinecone delete successful | source_id=%s | bot_id=%s",
            source_id,
            bot_id,
        )

        return {"deleted": True, "source_id": source_id}

    except Exception as e:
        logger.exception(
            "Pinecone delete failed | source_id=%s | bot_id=%s",
            source_id,
            bot_id,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Pinecone delete failed: {str(e)}"
        )