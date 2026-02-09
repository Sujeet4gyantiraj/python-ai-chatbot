from pydantic import BaseModel
from typing import Optional, Dict, Any, List


class UserDetails(BaseModel):
    """Optional end‑user contact information attached to a chat request."""

    name: Optional[str] = None
    email: Optional[str] = None
    phone_number: Optional[str] = None


class GenerateAIRequest(BaseModel):
    bot_id: str
    session_id: str
    user_query: str
    # Optional multi-tenant context so the LLM can tailor responses
    tenant_name: Optional[str] = None
    tenant_description: Optional[str] = None
    ai_node_data: Optional[Dict[str, Any]] = None
    # Optional end-user details; if not provided, this will be null/empty
    user_details: Optional[UserDetails] = None


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
