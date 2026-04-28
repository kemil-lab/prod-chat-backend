from llama_index.core import StorageContext
from app.contexts.neo import graph_store
from app.core.config import settings as config
from app.contexts.mongo import docstore
storage_context = StorageContext.from_defaults(
    persist_dir=config.CHROMA_PERSIST_DIR,
    docstore=docstore,       # 
    graph_store=graph_store, # Neo4j
)