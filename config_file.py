import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from dotenv import load_dotenv

load_dotenv()

@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4"
    temperature: float = 0.1
    max_tokens: int = 2000
    api_key: Optional[str] = None
    
    def __post_init__(self):
        if not self.api_key:
            if self.provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.provider == "gemini":
                self.api_key = os.getenv("GOOGLE_API_KEY")

@dataclass
class DataConfig:
    data_path: str = "./data"
    database_type: str = "duckdb"
    cache_enabled: bool = True
    cache_ttl: int = 300
    use_partitioning: bool = False
    partition_columns: List[str] = field(default_factory=lambda: ["year", "quarter"])
    cloud_storage: Optional[str] = None
    bucket_name: Optional[str] = None
    warehouse: Optional[str] = None
    warehouse_project: Optional[str] = None
    warehouse_dataset: Optional[str] = None

@dataclass
class VectorStoreConfig:
    enabled: bool = False
    provider: str = "faiss"
    embedding_model: str = "text-embedding-ada-002"
    dimension: int = 1536
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    pinecone_index: Optional[str] = None

@dataclass
class ScalabilityConfig:
    use_distributed_processing: bool = False
    processing_framework: str = "dask"
    worker_nodes: int = 4
    memory_per_worker: str = "4GB"
    enable_query_cache: bool = True
    enable_result_pagination: bool = True
    max_results_per_query: int = 10000
    enable_monitoring: bool = True
    monitoring_backend: str = "prometheus"

@dataclass
class UIConfig:
    title: str = "Retail Insights Assistant"
    theme: str = "light"
    port: int = 8501
    enable_file_upload: bool = True
    max_file_size_mb: int = 200

@dataclass
class SystemConfig:
    environment: str = "development"
    llm: LLMConfig = field(default_factory=LLMConfig)
    data: DataConfig = field(default_factory=DataConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    scalability: ScalabilityConfig = field(default_factory=ScalabilityConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    log_level: str = "INFO"
    log_file: Optional[str] = None
    enable_auth: bool = False
    auth_type: str = "basic"
    
    @classmethod
    def from_env(cls) -> 'SystemConfig':
        return cls(
            environment=os.getenv("ENVIRONMENT", "development"),
            llm=LLMConfig(
                provider=os.getenv("LLM_PROVIDER", "openai"),
                model=os.getenv("LLM_MODEL", "gpt-4"),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            ),
            data=DataConfig(
                data_path=os.getenv("DATA_PATH", "./data"),
                cloud_storage=os.getenv("CLOUD_STORAGE"),
                bucket_name=os.getenv("BUCKET_NAME"),
            ),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
    
    def to_dict(self) -> Dict:
        return {
            'environment': self.environment,
            'llm': {
                'provider': self.llm.provider,
                'model': self.llm.model,
                'temperature': self.llm.temperature,
            },
            'data': {
                'database_type': self.data.database_type,
                'cache_enabled': self.data.cache_enabled,
            },
            'scalability': {
                'use_distributed_processing': self.scalability.use_distributed_processing,
                'enable_query_cache': self.scalability.enable_query_cache,
            }
        }

SMALL_SCALE_CONFIG = SystemConfig(
    environment="development",
    llm=LLMConfig(model="gpt-3.5-turbo"),
    data=DataConfig(cache_enabled=True),
    scalability=ScalabilityConfig(
        use_distributed_processing=False,
        max_results_per_query=1000
    )
)

MEDIUM_SCALE_CONFIG = SystemConfig(
    environment="staging",
    llm=LLMConfig(model="gpt-4"),
    data=DataConfig(
        cache_enabled=True,
        use_partitioning=True
    ),
    scalability=ScalabilityConfig(
        use_distributed_processing=False,
        enable_query_cache=True,
        max_results_per_query=10000
    )
)

LARGE_SCALE_CONFIG = SystemConfig(
    environment="production",
    llm=LLMConfig(model="gpt-4"),
    data=DataConfig(
        cache_enabled=True,
        use_partitioning=True,
        cloud_storage="s3",
        warehouse="bigquery"
    ),
    vector_store=VectorStoreConfig(
        enabled=True,
        provider="pinecone"
    ),
    scalability=ScalabilityConfig(
        use_distributed_processing=True,
        processing_framework="pyspark",
        worker_nodes=10,
        enable_query_cache=True,
        enable_monitoring=True
    )
)

def get_config(scale: str = "small") -> SystemConfig:
    configs = {
        "small": SMALL_SCALE_CONFIG,
        "medium": MEDIUM_SCALE_CONFIG,
        "large": LARGE_SCALE_CONFIG
    }
    return configs.get(scale, SMALL_SCALE_CONFIG)

if __name__ == "__main__":
    config = SystemConfig.from_env()
    print("Configuration loaded:")
    print(config.to_dict())
