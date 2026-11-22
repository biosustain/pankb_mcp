"""
PanKB MCP Server Configuration
"""
import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    # Environment
    ENV = os.getenv("ENV")  # development / production
    DEBUG = ENV == "development" # debug mode for development

    # Server
    SERVER_NAME = "pankb"
    SERVER_VERSION = "1.0.0"
    # LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG" if DEBUG else "INFO")

    # MongoDB - PanKB MongoDB data
    MONGODB_URI = os.getenv("MONGODB_PANKB_CONN_STRING")
    MONGODB_NAME = "pankb"
    MONGODB_COLLECTIONS = {
        "organisms": "pankb_organisms",
        "gene_annotations": "pankb_gene_annotations",
        "gene_info": "pankb_gene_info",
        "genome_info": "pankb_genome_info",
        "pathway_info": "pankb_pathway_info",
        "isolation_info": "pankb_isolation_info",
        "genome_phylons": "pankb_genome_phylons",
        "gene_phylons": "pankb_gene_phylons",
        "pankb_stats": "pankb_stats",
    }

    # MongoDB - Vector store for RAG
    VECTOR_DB_URI = os.getenv("MONGODB_VECTOR_CONN_STRING")
    VECTOR_DB_NAME = "pankb_llm"
    VECTOR_COLLECTION = "pankb_vector_store_v2" 

    # Azure Blob Storage
    AZURE_BLOB_BASE_URL = os.getenv("AZURE_BLOB_BASE_URL")

    # API Keys for RAG
    VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")

