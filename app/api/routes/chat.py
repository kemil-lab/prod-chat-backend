from fastapi import APIRouter

from app.rag.pipeline import run_rag_pipeline_llamaIndex
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    print(1)
    # result = run_rag_pipeline(request.question)
    result = run_rag_pipeline_llamaIndex(request.question)
    # print(result)
    return ChatResponse(
        answer=result["answer"],
        analysis = result.get("analysis", {}),
        sources=result["sources"],
    )
