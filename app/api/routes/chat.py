from fastapi import APIRouter

from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        from app.rag.pipeline import run_rag_pipeline_llamaIndex

        result = run_rag_pipeline_llamaIndex(request.question)

        return ChatResponse(
            answer=result["answer"],
            analysis=result.get("analysis", {}),
            sources=result["sources"],
        )

    except Exception as e:
        print("❌ Error in chat:", e)
        return ChatResponse(
            answer="Something went wrong",
            analysis={"error": str(e)},
            sources=[]
        )