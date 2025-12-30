from pydantic import BaseModel
from typing import Optional, Dict, Any, List

class GenerateAIRequest(BaseModel):
    bot_id: str
    session_id: str
    user_query: str
    ai_node_data: Optional[Dict[str, Any]] = None

# Add more models and schemas here as needed for requests, responses, or database tables

# Example response schema
class AIResponse(BaseModel):
    fullText: Optional[str]
    cleanText: Optional[str]
    action: Optional[str]

# Example for file upload metadata
class KnowledgeUploadResult(BaseModel):
    fileName: str
    botId: str
    # Add more fields as needed

# Example for Pinecone delete response
class DeleteKnowledgeResponse(BaseModel):
    deleted: bool
    source_id: str
