"""Simple chunker for document processing."""

from typing import List


class SimpleChunker:
    def __init__(self, max_tokens: int = 300):
        self.max_tokens = max_tokens

    def create_chunks(self, text_content: str) -> List[str]:
        """Create simple chunks by paragraphs with token limit."""
        paragraphs = [
            p.strip() for p in text_content.split("\n") if len(p.strip().split()) > 10
        ]
        
        if not paragraphs:
            return []

        chunks = []
        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            # Simple token estimation (rough approximation)
            para_tokens = len(para.split()) * 1.3  # ~1.3 tokens per word

            if current_tokens + para_tokens > self.max_tokens and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks
