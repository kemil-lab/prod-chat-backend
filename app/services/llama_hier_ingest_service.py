from typing import Any, Dict, List
import re
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from app.contexts.mongo import docstore
import hashlib
from app.core.config import settings as config
from app.db.Chroma_clientV2 import get_collection
from pathlib import Path
BATCH_SIZE = 300 
def clean_metadata(raw_meta: dict) -> dict:
    source_path = raw_meta.get("source") or raw_meta.get("file_path") or ""
    file_name = Path(source_path).name if source_path else raw_meta.get("file_name", "")

    return {
        "file_name": file_name,
        "page": raw_meta.get("page", ""),
        "total_pages": raw_meta.get("total_pages", ""),
        "author": raw_meta.get("author", ""),
    }
def clean_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    text = re.sub(r"\b(obj|endobj|stream|endstream|xref)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove TOC-like lines:
        # "Question .... 21"
        if re.search(r"\.{2,}\s*\d+\s*$", line):
            continue

        # Remove mostly punctuation noise
        if re.fullmatch(r"[\W_]+", line):
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()
def generate_content_hash(text: str) -> str:
    
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
def ingest_pharma_data_hybrid(input_dir: str = "data/raw") -> Dict[str, Any]:

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=config.EMBEDDING_MODEL,
        # device=device,
    
    )
    loader = DirectoryLoader(
        input_dir,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=True,
    )
    raw_docs = loader.load()

    if not raw_docs:
        return {"status": "error", "message": "No PDF documents loaded"}
    documents: List[Document] = []
    for doc in raw_docs:
        cleaned = clean_text(doc.page_content)
        unique_id = generate_content_hash(cleaned)
        if not cleaned or len(cleaned.split()) < 20:
            continue

        if "obj" in cleaned or "stream" in cleaned:
            continue
        metadata = clean_metadata(doc.metadata or {})

        documents.append(
            Document(
                text=cleaned,
                metadata=metadata,
                id_=unique_id
            )
        )

    if not documents:
        return {"status": "error", "message": "No clean text found in documents"}

    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[2048, 1024, 512],
        chunk_overlap=200
    )
    all_nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(all_nodes)

    chroma_collection = get_collection()
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

   
    docstore.add_documents(all_nodes)

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=docstore,
    )

    index = VectorStoreIndex(
    [],
    storage_context=storage_context,
    )
    for i in range(0, len(leaf_nodes), BATCH_SIZE):
        batch = leaf_nodes[i:i + BATCH_SIZE]
        index.insert_nodes(batch)

    # storage_context.persist(persist_dir=config.CHROMA_PERSIST_DIR)

    return {
        "status": "success",
        "documents_loaded": len(documents),
        "all_nodes": len(all_nodes),
        "leaf_nodes": len(leaf_nodes),
    }
