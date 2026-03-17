"""Embeddings module for document processing."""

from typing import List, Dict, Any
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding


class EmbeddingService:
    def __init__(self):
        self.dense_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
        self.sparse_model = SparseTextEmbedding("Qdrant/bm25")
        self.colbert_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
        self.dimension = 384
        self.colbert_dimension = 128

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create dense embeddings for a list of texts."""
        dense_embeddings = list(self.dense_model.passage_embed(texts))
        return [emb.tolist() for emb in dense_embeddings]

    def create_sparse_embeddings(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Create sparse embeddings using BM25."""
        sparse_embeddings = list(self.sparse_model.passage_embed(texts))
        return [emb.as_object() for emb in sparse_embeddings]

    def create_colbert_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create ColBERT embeddings for late interaction."""
        colbert_embeddings = list(self.colbert_model.passage_embed(texts))
        return [emb.tolist() for emb in colbert_embeddings]
