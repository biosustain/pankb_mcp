"""
RAG (Retrieval-Augmented Generation) Tools for PanKB MCP Server

Provides document retrieval from pangenomics literature.
Returns documents only - answer generation is handled by the AI client.
"""
import json
import logging

from app.utils.connections import get_rag_client
from fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP(name="RAGTools")

EMBEDDING_MODEL = "voyage-4-large"
EMBEDDING_DIMENSIONS = 1024
RERANK_MODEL = "rerank-2.5"
RERANK_SCORE_THRESHOLD = 0.45
VECTOR_SEARCH_K = 30
RERANK_TOP_K = 20


def _get_embedding(text: str) -> list[float]:
    """Get embedding vector for text using VoyageAI"""
    client = get_rag_client()
    result = client.voyage.embed(
        texts=[text],
        model=EMBEDDING_MODEL,
        input_type="query",
        output_dimension=EMBEDDING_DIMENSIONS,
    )
    return result.embeddings[0]


def _vector_search(query_embedding: list[float], k: int = 30) -> list[dict]:
    """Vector similarity search in MongoDB Cosmos DB"""
    client = get_rag_client()
    pipeline = [
        {
            "$search": {
                "cosmosSearch": {
                    "vector": query_embedding,
                    "path": "vectorContent",
                    "k": k
                },
                "returnStoredSource": True
            }
        },
        {
            "$project": {
                "textContent": 1,
                "source": 1,
                "title": 1,
                "score": {"$meta": "searchScore"}
            }
        }
    ]
    return list(client.collection.aggregate(pipeline))


def _rerank_documents(query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[dict]:
    """Rerank documents using Voyage Rerank"""
    if not documents:
        return []

    client = get_rag_client()
    doc_texts = [doc['textContent'] for doc in documents]

    rerank_response = client.voyage.rerank(
        query=query,
        documents=doc_texts,
        model=RERANK_MODEL,
        top_k=top_k,
    )

    reranked_docs = []
    for result in rerank_response.results:
        doc = documents[result.index].copy()
        doc['relevance_score'] = result.relevance_score
        reranked_docs.append(doc)

    return reranked_docs


def _filter_documents(documents: list[dict], threshold: float = RERANK_SCORE_THRESHOLD) -> list[dict]:
    """Filter documents below relevance threshold"""
    return [doc for doc in documents if doc.get('relevance_score', 0) >= threshold]


def _search_documents(query: str, top_k: int = RERANK_TOP_K) -> list[dict]:
    """
    Complete RAG retrieval pipeline:
    1. Embed query
    2. Vector search (k=30)
    3. Rerank (top_k=20)
    4. Filter by relevance (threshold=0.45)
    5. Return top_k documents
    """
    query_embedding = _get_embedding(query)
    initial_docs = _vector_search(query_embedding, k=VECTOR_SEARCH_K)
    reranked_docs = _rerank_documents(query, initial_docs)
    filtered_docs = _filter_documents(reranked_docs)
    return filtered_docs[:top_k]


def _format_as_markdown(docs: list[dict]) -> str:
    """Format documents as markdown for display"""
    results = []
    for i, doc in enumerate(docs, 1):
        title = doc.get('title', 'Unknown')
        source = doc.get('source', '')
        content = doc.get('textContent', '')

        results.append(
            f"### Document {i}: {title}\n\n"
            f"**Source:** {source}\n\n"
            f"{content}\n"
        )

    return "\n\n---\n".join(results)


@mcp.tool()
def search_pangenome_literature(query: str, top_k: int = RERANK_TOP_K) -> str:
    """
    Search PanKB's scientific knowledge base about microbial pangenomics.

    IMPORTANT: You MUST use this tool for ANY question about pangenomics concepts,
    theory, methodology, or scientific background. This includes questions like:
    - "What is X?" (core genome, accessory genome, pangenome, etc.)
    - "Why is X important?" (pangenome analysis, comparative genomics, etc.)
    - "How does X work?" (gene frequency, openness calculation, etc.)
    - "What are the applications of X?"
    - Any conceptual or educational questions about microbial genomics

    Do NOT answer these questions from your own knowledge - always search first.
    Base your answer ONLY on the returned documents.

    Args:
        query: Search query (e.g., "why is pangenome analysis important", "what is core genome")
        top_k: Maximum number of documents to return (default: 20, the most the pipeline yields)

    Returns:
        Relevant scientific documents with title, content, and source URL
    """
    try:
        docs = _search_documents(query, top_k=top_k)

        if not docs:
            return "No relevant documents found. Please tell the user you don't have information about this topic."

        markdown_content = _format_as_markdown(docs)

        # Return as JSON with type "string" so client can parse and render markdown
        return json.dumps({
            "type": "string",
            "format": "markdown",
            "content": markdown_content
        })

    except ValueError as e:
        return f"RAG not configured: {str(e)}"
    except Exception as e:
        logger.error(f"Literature search failed: {e}")
        return f"Error searching literature: {str(e)}"
