from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    pinecone_api_key: str
    pinecone_index_name: str = "audience-matching-v2"
    pinecone_environment: str = "us-east-1"

    tmdb_api_key: str = ""

    service_api_key: str

    embedding_model: str = "all-mpnet-base-v2"
    hdbscan_min_cluster_size: int = 15


settings = Settings()
