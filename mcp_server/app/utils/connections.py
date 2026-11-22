"""
Connection Utilities for PanKB MCP Server
"""
import gzip
import json
import logging
from typing import Optional

import requests
from app.config import Config
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

logger = logging.getLogger(__name__)


class MongoDBClient:
    """MongoDB Client Singleton"""

    _instance: Optional['MongoDBClient'] = None
    _client: Optional[MongoClient] = None
    _db: Optional[Database] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._client is None:
            self._connect()

    def _connect(self):
        """Establish MongoDB connection to PanKB data"""
        try:
            if not Config.MONGODB_URI:
                raise ValueError("MONGODB_PANKB_CONN_STRING not configured")
            self._client = MongoClient(Config.MONGODB_URI)
            self._db = self._client[Config.MONGODB_NAME]
            self._client.server_info()  # Test connection
            logger.info(f"Connected to PanKB MongoDB: {Config.MONGODB_NAME}")
        except Exception as e:
            logger.error(f"Failed to connect to PanKB MongoDB: {e}")
            raise

    @property
    def db(self) -> Database:
        """Get database instance"""
        if self._db is None:
            self._connect()
        return self._db

    def get_collection(self, collection_name: str) -> Collection:
        """Get collection by name"""
        return self.db[collection_name]

    def close(self):
        """Close MongoDB connection"""
        if self._client:
            self._client.close()
            logger.info("MongoDB connection closed")


mongo_client = MongoDBClient()


class BlobClient:
    """Azure Blob Storage Client"""

    @staticmethod
    def _build_url(species: str, filename: str) -> str:
        return f"{Config.AZURE_BLOB_BASE_URL}species/{species}/{filename}"

    @classmethod
    def fetch_json(cls, species: str, filename: str, timeout: int = 30) -> dict:
        """Fetch JSON data from Azure Blob Storage"""
        url = cls._build_url(species, filename)
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()

    @classmethod
    def fetch_gzip_json(cls, species: str, filename: str, timeout: int = 60) -> dict:
        """Fetch gzipped JSON data from Azure Blob Storage"""
        url = cls._build_url(species, filename)
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        decompressed = gzip.decompress(response.content)
        return json.loads(decompressed.decode('utf-8'))

    @classmethod
    def fetch_text(cls, species: str, filename: str, timeout: int = 30) -> str:
        """Fetch text data from Azure Blob Storage"""
        url = cls._build_url(species, filename)
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.text


blob_client = BlobClient()


class RAGClient:
    """
    RAG Client - provides API clients for vector search.
    Lazy initialization: only connects when first accessed.
    Imports voyageai only when needed to avoid startup errors.
    """

    _instance: Optional['RAGClient'] = None
    _initialized: bool = False

    # API clients (typed as Any to avoid import at module level)
    _voyage_client = None
    _collection: Optional[Collection] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _ensure_initialized(self):
        """Initialize clients on first use (lazy loading)"""
        if self._initialized:
            return

        # Lazy import to avoid ModuleNotFoundError at server startup
        import voyageai

        if not Config.VECTOR_DB_URI:
            raise ValueError("MONGODB_VECTOR_CONN_STRING not configured")
        if not Config.VOYAGE_API_KEY:
            raise ValueError("VOYAGE_API_KEY not configured")

        self._voyage_client = voyageai.Client(api_key=Config.VOYAGE_API_KEY)

        client = MongoClient(Config.VECTOR_DB_URI)
        db = client[Config.VECTOR_DB_NAME]
        self._collection = db[Config.VECTOR_COLLECTION]

        self._initialized = True
        logger.info(f"RAG client initialized: {Config.VECTOR_DB_NAME}.{Config.VECTOR_COLLECTION}")

    @property
    def voyage(self):
        """VoyageAI client for embeddings and reranking"""
        self._ensure_initialized()
        return self._voyage_client

    @property
    def collection(self) -> Collection:
        """MongoDB collection for vector search"""
        self._ensure_initialized()
        return self._collection


def get_rag_client() -> RAGClient:
    """Get RAG client singleton (lazy initialization)"""
    return RAGClient()