"""Main ingestion pipeline for MCP RAG Server."""

import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from .simple_chunker import SimpleChunker
from .embeddings import EmbeddingService

load_dotenv()


class IngestionPipeline:
    def __init__(
        self,
        collection_name: str = "apolo-docs",
        max_tokens: int = 300,
        qdrant_url: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.max_tokens = max_tokens
        
        # Initialize Qdrant client
        self.qdrant = QdrantClient(
            url=qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        )
        
        # Initialize components
        self.chunker = SimpleChunker(max_tokens=max_tokens)
        self.embeddings = EmbeddingService()
        
        # Set embedding dimension based on model
        self.vector_size = 384  # FastEmbed dimension

    def create_collection(self, recreate: bool = False) -> None:
        """Create Qdrant collection with multi-vector configuration."""
        if recreate:
            try:
                self.qdrant.delete_collection(self.collection_name)
                print(f"Deleted existing collection: {self.collection_name}")
            except Exception:
                pass

        # Multi-vector configuration (dense + sparse + colbert)
        vectors_config = {
            "dense": models.VectorParams(
                size=384,  # FastEmbed dense
                distance=models.Distance.COSINE
            ),
            "colbert": models.VectorParams(
                size=128,  # ColBERT dimension
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            ),
        }
        sparse_config = {"sparse": models.SparseVectorParams()}
        
        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_config
        )
        
        print(f"Created multi-vector collection: {self.collection_name}")
        
        # Create default indexes for common fields
        default_fields = ["source", "document_id", "last_accessed", "active"]
        self.create_indexes(default_fields)

    def create_indexes(self, fields: List[str]) -> None:
        """Create payload indexes for better filtering performance."""
        for field_name in fields:
            try:
                if field_name == "last_accessed":
                    # Create datetime index for last_accessed
                    self.qdrant.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema=models.PayloadSchemaType.DATETIME,
                    )
                else:
                    # Create keyword index for other fields
                    self.qdrant.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                print(f"Created index for field: {field_name}")
            except Exception as e:
                print(f"Failed to create index for {field_name}: {e}")

    def process_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        document_id: Optional[str] = None
    ) -> str:
        """Process a single document and upload to Qdrant."""
        if not document_id:
            document_id = str(uuid.uuid4())

        # Create chunks
        chunks = self.chunker.create_chunks(content)
        if not chunks:
            print(f"No chunks created for document {document_id}")
            return document_id

        # Create embeddings
        dense_embeddings = self.embeddings.create_embeddings(chunks)
        sparse_embeddings = self.embeddings.create_sparse_embeddings(chunks)
        colbert_embeddings = self.embeddings.create_colbert_embeddings(chunks)

        # Create points
        points = []
        for i, chunk in enumerate(chunks):
            point_id = str(uuid.uuid4())
            
            point_metadata = {
                # User metadata (preserve original)
                **metadata,
                # Standard metadata (don't override user fields)
                "document_id": document_id,
                "ingestion_date": datetime.now().isoformat(),
                "chunk_index": i,
                "chunk_count": len(chunks),
                "content": chunk,
                # Status metadata
                "active": metadata.get("active", True),  # Default to True if not specified
                # Technical metadata
                "embedding_model": "all-MiniLM-L6-v2",
                "vector_size": self.vector_size
            }

            point = models.PointStruct(
                id=point_id,
                vector={
                    "dense": dense_embeddings[i],
                    "sparse": sparse_embeddings[i],
                    "colbert": colbert_embeddings[i]
                },
                payload=point_metadata
            )
            
            points.append(point)

        # Upload to Qdrant
        self.qdrant.upload_points(
            collection_name=self.collection_name,
            points=points,
            batch_size=10
        )

        print(f"Uploaded {len(points)} multi-vector points for document {document_id}")
        return document_id

    def search_similar(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.7,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents using multi-vector."""
        # Multi-vector search (dense + sparse + colbert)
        query_dense = self.embeddings.create_embeddings([query])[0]
        query_sparse = self.embeddings.create_sparse_embeddings([query])[0]
        query_colbert = self.embeddings.create_colbert_embeddings([query])[0]
        
        # Build filter conditions - always include active=True by default
        must_conditions = [
            models.FieldCondition(
                key="active",
                match=models.MatchValue(value=True)
            )
        ]
        
        # Add user filter conditions if provided
        if filter_conditions:
            for key, value in filter_conditions.items():
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
        
        search_result = self.qdrant.query_points(
            collection_name=self.collection_name,
            prefetch=[
                {
                    "prefetch": [
                        {"query": query_dense, "using": "dense", "limit": 20},
                        {"query": query_sparse, "using": "sparse", "limit": 20},
                    ],
                    "query": models.FusionQuery(fusion=models.Fusion.RRF),
                    "limit": 15,
                }
            ],
            query=query_colbert,
            using="colbert",
            limit=limit,
            query_filter=models.Filter(must=must_conditions),
        ).points
        
        results = []
        for hit in search_result:
            results.append({
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            })
            
            # Update last_accessed timestamp (preserve existing payload)
            current_payload = hit.payload
            current_payload["last_accessed"] = datetime.now().isoformat()
            self.qdrant.overwrite_payload(
                collection_name=self.collection_name,
                payload=current_payload,
                points=[hit.id]
            )
        
        return results

