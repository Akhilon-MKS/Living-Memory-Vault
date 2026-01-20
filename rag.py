"""
backend/rag.py

Handles Retrieval-Augmented Generation: retrieve relevant memories and generate responses using local models.
"""

import os
from typing import List, Dict
from transformers import pipeline
from .embeddings import MemoryEmbeddings

PERSONA_PROMPT = """
You are the Living Memory Vault — an AI archivist who answers based only on the user's memories.
Be warm, nostalgic, and factual. Cite the filename or year when referencing memories.
If something is imagined, clearly mark it as imagined.
"""

class RAGSystem:
    def __init__(self, embeddings: MemoryEmbeddings):
        self.embeddings = embeddings
        # Initialize local text generation pipeline
        self.generator = pipeline('text2text-generation', model='google/flan-t5-base')

    def retrieve_memories(self, query: str, n_results: int = 5) -> List[Dict]:
        """Retrieve the top n relevant memories for the query."""
        return self.embeddings.search_memories(query, n_results)

    def generate_response(self, query: str, retrieved_memories: List[Dict]) -> Dict:
        """Generate a response using local FLAN-T5 model based on the query and retrieved memories."""
        if not retrieved_memories:
            return {
                'response': "I don't have any memories to reference yet. Please upload some files first!",
                'images': [],
                'audio': []
            }

        # Separate different types of memories
        image_memories = [mem for mem in retrieved_memories if mem['metadata'].get('source_type') == 'image']
        audio_memories = [mem for mem in retrieved_memories if mem['metadata'].get('source_type') == 'audio']
        other_memories = [mem for mem in retrieved_memories if mem['metadata'].get('source_type') not in ['image', 'audio']]

        # Build context from non-media memories
        context = "\n".join([
            f"Memory from {mem['metadata']['filename']} ({mem['metadata']['source_type']}): {mem['content']}"
            for mem in other_memories
        ])

        # If asking about images or audio, don't include their descriptions in context
        if image_memories and any(word in query.lower() for word in ['photo', 'image', 'picture', 'show', 'display']):
            context = "The user is asking about images/photos in their memories."
        elif audio_memories and any(word in query.lower() for word in ['audio', 'sound', 'recording', 'play', 'listen', 'music', 'voice']):
            context = "The user is asking about audio recordings in their memories."

        # Create a simpler prompt for better FLAN-T5 performance
        prompt = f"Question: {query}\nContext: {context}\nAnswer:"

        # Generate response using local model with longer max_length for better responses
        response = self.generator(prompt, max_length=100, num_return_sequences=1)
        generated_text = response[0]['generated_text']

        # Clean up response and add persona wrapper
        if generated_text.startswith("Answer:"):
            generated_text = generated_text[7:].strip()

        # Collect media from retrieved memories
        images = []
        for mem in image_memories:
            if mem['metadata'].get('file_path'):
                images.append({
                    'path': mem['metadata']['file_path'],
                    'filename': mem['metadata']['filename']
                })

        audio = []
        for mem in audio_memories:
            if mem['metadata'].get('file_path'):
                audio.append({
                    'path': mem['metadata']['file_path'],
                    'filename': mem['metadata']['filename']
                })

        return {
            'response': f"As your memory archivist, I recall: {generated_text}",
            'images': images,
            'audio': audio
        }
