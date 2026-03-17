#!/usr/bin/env python3
"""Example usage of the ingestion pipeline."""

import os
from dotenv import load_dotenv

# Add the src directory to Python path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mcp_rag_server.ingestion.ingestion_pipeline import IngestionPipeline

load_dotenv()

def main():
    """Example ingestion pipeline usage."""
    
    # Initialize pipeline
    pipeline = IngestionPipeline(
        collection_name="apolo-docs",
        max_tokens=300
    )
    
    # Create collection
    pipeline.create_collection(recreate=True)
    
    # Create indexes for common metadata fields
    fields_to_index = ["source", "document_id", "file_name"]
    pipeline.create_indexes(fields_to_index)
    
    # Example 1: Process a text document
    sample_text = """
    # Artificial Intelligence Overview
    
    Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines 
    that can perform tasks that typically require human intelligence. These tasks include learning, 
    reasoning, problem-solving, perception, and language understanding.
    
    ## Machine Learning
    
    Machine Learning is a subset of AI that focuses on systems that can learn from and make decisions 
    based on data. The main types of machine learning are supervised learning, unsupervised learning, 
    and reinforcement learning.
    
    ## Deep Learning
    
    Deep Learning is a subset of machine learning that uses neural networks with multiple layers to 
    model and understand complex patterns in data. It has been particularly successful in areas 
    like computer vision, natural language processing, and speech recognition.
    """
    
    metadata = {
        "title": "AI Overview",
        "category": "technology",
        "author": "Example Author",
        "created_date": "2024-01-01"
    }
    
    doc_id = pipeline.process_document(sample_text, metadata)
    print(f"Processed document: {doc_id}")
    
    # Example 2: Process another document
    another_text = """
    # Natural Language Processing
    
    Natural Language Processing (NLP) is a field of AI that focuses on the interaction between 
    computers and human language. It involves teaching computers to understand, interpret, and 
    generate human language in a way that is valuable.
    
    ## Applications
    
    Common NLP applications include:
    - Machine translation
    - Sentiment analysis
    - Chatbots and virtual assistants
    - Text summarization
    - Information extraction
    """
    
    metadata_2 = {
        "title": "NLP Guide",
        "category": "technology",
        "author": "Example Author",
        "created_date": "2024-01-02"
    }
    
    doc_id_2 = pipeline.process_document(another_text, metadata_2)
    print(f"Processed document: {doc_id_2}")

    # Search examples
    print("\n=== Search Results ===")
    
    # Search 1: General query
    results = pipeline.search_similar("What is machine learning?", limit=3)
    print("Search for 'What is machine learning?' (multi-vector):")
    for result in results:
        print(f"  Score: {result['score']:.3f}")
        print(f"  Content: {result['payload']['content'][:100]}...")
        print()
    
    # Search 2: Specific query with filter
    results = pipeline.search_similar(
        "NLP", 
        limit=2,
        filter_conditions={"category": "technology"}
    )
    print("Search for 'NLP' with filter (multi-vector):")
    for result in results:
        print(f"  Score: {result['score']:.3f}")
        print(f"  Title: {result['payload'].get('title', 'N/A')}")
        print(f"  Content: {result['payload']['content'][:100]}...")
        print()

if __name__ == "__main__":
    main()
