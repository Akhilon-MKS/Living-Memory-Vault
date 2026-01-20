"""
backend/embeddings.py

Handles embedding generation using OpenAI's text-embedding-3-large and storage in ChromaDB.
"""

import os
import chromadb
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import time

class MemoryEmbeddings:
    def __init__(self, db_path: str = "./chroma_store"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name="memories")
        # Initialize sentence transformer model for local embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings for the given text using local SentenceTransformer."""
        return self.model.encode(text).tolist()

    def add_memories(self, memories: List[Dict]):
        """
        Add a list of memories to the ChromaDB collection.

        Each memory dict should have: 'content', 'filename', 'source_type', 'upload_time', 'year', 'file_path'.
        """
        if not memories:
            return  # No memories to add

        # Get existing count to generate unique IDs
        try:
            existing_count = len(self.collection.get(limit=10000)['ids'])  # Get all existing IDs
        except:
            existing_count = 0

        ids = []
        documents = []
        metadatas = []
        embeddings_list = []

        for i, memory in enumerate(memories):
            embedding = self.generate_embedding(memory['content'])
            embeddings_list.append(embedding)
            ids.append(f"memory_{existing_count + i}")
            documents.append(memory['content'])
            metadata = {
                'filename': memory['filename'],
                'source_type': memory['source_type'],
                'upload_time': memory['upload_time'],
                'year': memory.get('year', ''),
                'file_path': memory.get('file_path', '')
            }
            # Filter out None values as ChromaDB doesn't allow them
            metadata = {k: v for k, v in metadata.items() if v is not None}
            metadatas.append(metadata)

        self.collection.add(
            embeddings=embeddings_list,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def search_memories(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for the top n most relevant memories based on the query."""
        query_embedding = self.generate_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        memories = []
        for i in range(len(results['documents'][0])):
            memories.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })

        return memories
