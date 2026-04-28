from pydantic import BaseModel
from typing import List, Dict, Any


class ChatRequest(BaseModel):
    question: str


class SourceChunk(BaseModel):
    id: str | None = None
    content: str
    metadata: Dict[str, Any] = {}
    score: float | None = None


class ChatResponse(BaseModel):
    answer: str
    analysis: Dict[str, Any] = {}
    sources: List[SourceChunk] = []
