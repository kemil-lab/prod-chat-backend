from functools import lru_cache
from typing import List, Any
import uuid

import chromadb
import numpy as np
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from app.core.config import settings


@lru_cache(maxsize=1)
def get_chroma_client() -> ClientAPI:
    return chromadb.CloudClient(
        api_key=settings.CHROMA_API_KEY,
        tenant=settings.CHROMA_TENANT,
        database=settings.CHROMA_DATABASE,
    )


@lru_cache(maxsize=1)
def get_collection() -> Collection:
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION_NAME,
        metadata={"description": "Main knowledge base collection for chatbot"},
    )


def reset_collection() -> None:
    client = get_chroma_client()

    try:
        client.delete_collection(name=settings.CHROMA_COLLECTION_NAME)
    except Exception:
        pass

    get_collection.cache_clear()
    get_collection()  


def add_documents(documents: List[Any], embeddings: np.ndarray) -> None:
    if len(documents) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(documents)} documents vs {len(embeddings)} embeddings"
        )

    collection = get_collection()

    ids = []
    metadatas = []
    documents_text = []
    embeddings_list = []

    for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
        doc_id = f"doc_{uuid.uuid4()}"
        ids.append(doc_id)

        metadata = dict(getattr(doc, "metadata", {}) or {})
        metadata["doc_index"] = i
        metadata["content_length"] = len(doc.page_content)
        metadatas.append(metadata)

        documents_text.append(doc.page_content)
        embeddings_list.append(embedding.tolist() if hasattr(embedding, "tolist") else embedding)

    collection.add(
        ids=ids,
        embeddings=embeddings_list,
        metadatas=metadatas,
        documents=documents_text,
    )