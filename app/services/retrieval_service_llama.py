from functools import lru_cache

from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever, AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from app.contexts.mongo import docstore
from llama_index.core.postprocessor import SentenceTransformerRerank

from app.core.config import settings as config
from functools import lru_cache
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from app.db.Chroma_clientV2 import get_chroma_client
@lru_cache(maxsize=1)
def reRanker():
    print("Initializing Reranker...")
    return SentenceTransformerRerank(
            model=config.RERANK_MODEL,
            top_n=3,
        )
def setup_hybrid_query_engine():
    db = get_chroma_client()
    chroma_collection = db.get_or_create_collection(config.CHROMA_COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(
        docstore=docstore,
        vector_store=vector_store,
        # persist_dir=config.CHROMA_PERSIST_DIR,
    )
    
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=config.EMBEDDING_MODEL,
        # device=device,
    
    )
    Settings.llm = None
    index = VectorStoreIndex.from_vector_store(
        vector_store, 
        storage_context=storage_context
    )
    vector_retriever = index.as_retriever(similarity_top_k=8)

    # all_nodes = list(storage_context.docstore.docs.values())
    # leaf_nodes = get_leaf_nodes(all_nodes) 
    
    bm25_retriever = BM25Retriever.from_defaults(
        docstore=storage_context.docstore, 
        similarity_top_k=8
    )

    hybrid_retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        num_queries=2,        
        similarity_top_k=5,      
        mode="reciprocal_rerank", 
        use_async=True,
    )

    final_retriever = AutoMergingRetriever(
        hybrid_retriever, 
        storage_context, 
        verbose=True
    )

    query_engine = RetrieverQueryEngine.from_args(
        final_retriever,
        #  node_postprocessors=[reranker]
        # response_mode=""
    )
    
    return query_engine

engine = setup_hybrid_query_engine()