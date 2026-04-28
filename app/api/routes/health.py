from fastapi import APIRouter
router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check():
    print(1)
    return {"status": "ok"}

