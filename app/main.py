from fastapi import FastAPI
from app.api.routes import  chat,health
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core.postprocessor import SentenceTransformerRerank
from app.core.config import settings as config
from app.services.model_store import reranker as global_reranker
app = FastAPI(title="Chatbot Backend API")
print(1)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.on_event("startup")
def load_models():
    global global_reranker
    print("⚡ Loading reranker at startup...")

    from app.services import model_store
    model_store.reranker = SentenceTransformerRerank(
        model=config.RERANK_MODEL,
        top_n=3,
    )

    print("✅ Reranker loaded")
app.include_router(chat.router, prefix="/api/v1")
app.include_router(health.router, prefix="/api/v1")

@app.get("/")
def root():
    return {"message": "Chatbot API is running"}