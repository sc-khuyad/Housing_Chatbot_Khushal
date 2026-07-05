from pydantic import BaseModel, Field
from typing import List, Optional


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique session identifier for conversation state")
    query: str = Field(..., description="User query text")
    top_k: Optional[int] = Field(5, description="Number of final hits to return after reranking")
    system_prompt: Optional[str] = Field(None, description="Optional custom system prompt to prepend to the default RAG instructions")
    debug_prompt: bool = Field(True, description="Whether to print the final composed prompt")


class ClearMemoryRequest(BaseModel):
    session_id: str = Field(..., description="Unique session identifier whose saved conversation memory should be cleared")


class SourceItem(BaseModel):
    chunk_id: str
    title: Optional[str]
    url: Optional[str]
    score: float
    snippet: Optional[str]


class ChatResponse(BaseModel):
    session_id: str
    answer: Optional[str]
    sources: List[SourceItem]
    raw_llm: Optional[str]
