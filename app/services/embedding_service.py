from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
   
    return SentenceTransformer(settings.EMBEDDING_MODEL,device="cuda")

def embedding_model():
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True, "device":"cuda"},
        # device="cuda"
    )

def embed_query(text: str) -> List[float]:
  
    model = get_embedding_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def embed_documents(texts: List[str]) -> List[List[float]]:
 
    if not texts:
        return []

    model = get_embedding_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return vectors.tolist()