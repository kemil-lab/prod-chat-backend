from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    GEMINI_API_KEY: str
    CHROMA_PERSIST_DIR: str 
    CHROMA_COLLECTION_NAME: str 
    EMBEDDING_MODEL: str 
    RERANK_MODEL: str 
    # EMBEDDING_MODEL: str 
    GEMINI_MODEL: str 
    CHROMA_API_KEY: str 
    CHROMA_TENANT: str 
    CHROMA_DATABASE: str 
    URI: str 
    DB_NAME: str 
    NAME_SPACE: str 
    HUGGINGFACE_API_KEY: str 
    OPENAI_API_KEY: str
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()