import os
import re
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class Chunk:
    def __init__(self, id: str, content: str, source: str):
        self.id = id
        self.content = content
        self.source = source

class Retriever:
    def __init__(self, docs_path: str = "docs"):
        self.chunks: List[Chunk] = []
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self.vectors = None
        self._load_documents(docs_path)
        
    def _load_documents(self, docs_path: str):
        """Load and chunk documents from the docs directory."""
        for filename in os.listdir(docs_path):
            if filename.endswith('.md'):
                filepath = os.path.join(docs_path, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Simple paragraph-based chunking
                paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                
                source_name = filename.replace('.md', '')
                for i, para in enumerate(paragraphs):
                    chunk_id = f"{source_name}::chunk{i}"
                    self.chunks.append(Chunk(chunk_id, para, source_name))
        
        # Build TF-IDF vectors
        if self.chunks:
            corpus = [chunk.content for chunk in self.chunks]
            self.vectors = self.vectorizer.fit_transform(corpus)
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for top-k relevant chunks."""
        if not self.chunks:
            return []
        
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.vectors)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append({
                'id': chunk.id,
                'content': chunk.content,
                'source': chunk.source,
                'score': float(similarities[idx])
            })
        
        return results
    
    def get_chunk_by_id(self, chunk_id: str) -> str:
        """Retrieve specific chunk content by ID."""
        for chunk in self.chunks:
            if chunk.id == chunk_id:
                return chunk.content
        return ""