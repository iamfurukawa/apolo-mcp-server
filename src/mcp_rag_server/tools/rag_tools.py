"""MCP tools for RAG operations with AI-optimized metadata.

This module provides tools for document management in a RAG system,
optimized for AI/LLM applications with comprehensive metadata tracking.
"""

from typing import Dict, Any, List, Optional
from ..rag.ingestion_pipeline import IngestionPipeline

# Global pipeline instance
_pipeline = None

def get_pipeline():
    """Get or create pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline()
    return _pipeline


async def create_document(content: str, metadata: Dict[str, Any]) -> str:
    """
    Create a new document in the RAG system.
    
    Args:
        content: The document content to be processed and stored
        metadata: Dictionary containing document required metadata. Supports:
            - title: Document title
            - category: Content category (technology, business, product)
            - source: Source system (confluence, jira, slack, google_drive, etc.)
    
    Returns:
        str: The unique document ID generated for tracking
    """
    pipeline = get_pipeline()
    return pipeline.process_document(content, metadata)


async def search_documents(
    query: str, 
    limit: int = 5,
    filter_conditions: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Search for documents in the RAG system using multi-vector similarity.
    
    Args:
        query: Search query string
        limit: Maximum number of results to return (default: 5)
        filter_conditions: Optional dictionary of filter conditions:
            - source: Filter by source system
            - category: Filter by content category
    
    Returns:
        List[Dict[str, Any]]: List of matching documents with:
            - id: Point ID in Qdrant
            - score: Similarity score (higher = more relevant)
            - payload: Complete document metadata including:
                * Content and chunk information
                * Search scores (when available)
                * Last accessed timestamps
    
    The search automatically:
    - Filters to active documents only
    - Updates last_accessed timestamp on all results
    - Uses multi-vector fusion (dense + sparse + ColBERT)
    - Orders results by relevance score
    """
    pipeline = get_pipeline()
    return pipeline.search_similar(query, limit, filter_conditions)


async def get_document_by_id(document_id: str) -> dict:
    """
    Retrieve a complete document by its unique identifier with all chunks grouped.
    
    Args:
        document_id: The unique identifier of the document to retrieve
    
    Returns:
        dict: Document object with all chunks and metadata
    """
    pipeline = get_pipeline()
    return pipeline.get_document_by_id(document_id)
