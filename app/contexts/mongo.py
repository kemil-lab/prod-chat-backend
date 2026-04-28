from llama_index.storage.docstore.mongodb import MongoDocumentStore

from app.core.config import settings

docstore = MongoDocumentStore.from_uri(
    uri=settings.URI,
    db_name=settings.DB_NAME,
    namespace=settings.NAME_SPACE
)