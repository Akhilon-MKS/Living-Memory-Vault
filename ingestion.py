"""
backend/ingestion.py

Handles the ingestion of uploaded files: text, audio, and images.
- Text: Read directly.
- Audio: Transcribe using OpenAI Whisper API.
- Images: Generate captions using GPT-4 Vision.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import pipeline
from PIL import Image
import io
from docx import Document

# Initialize local image captioning pipeline
captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

def process_text_file(file_path: Path) -> str:
    """Read and return the content of a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # If UTF-8 fails, try with latin-1 or other encoding
        with open(file_path, 'r', encoding='latin-1') as f:
            return f.read()

def process_word_file(file_path: Path) -> str:
    """Read and return the content of a Word document."""
    doc = Document(file_path)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return '\n'.join(full_text)

def process_audio_file(file_path: Path) -> str:
    """Transcribe an audio file using local Whisper model."""
    try:
        # Initialize Whisper pipeline for automatic speech recognition
        transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base")
        
        # Transcribe the audio
        result = transcriber(str(file_path))
        transcription = result["text"]
        
        return f"Audio transcription: {transcription}"
    except Exception as e:
        return f"Audio transcription failed: {str(e)}. This is a placeholder for audio content."

def process_image_file(file_path: Path) -> str:
    """Generate a caption for an image using local BLIP model."""
    # Load image
    image = Image.open(file_path)

    # Generate caption using local model
    caption = captioner(image)[0]['generated_text']
    return f"This image shows: {caption}. This appears to be a personal memory captured in a photograph."

def ingest_files(uploaded_files: List[Tuple[str, bytes, str, str]]) -> List[Dict]:
    """
    Process a list of uploaded files and return a list of memory dictionaries.

    Each memory dict contains: 'content', 'filename', 'source_type', 'upload_time', 'year' (optional), 'file_path' (for images).
    """
    memories = []
    from datetime import datetime
    import uuid

    # Ensure uploads directory exists
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)

    for filename, file_data, source_type, description in uploaded_files:
        file_path = None

        # For images and audio, save to uploads directory
        if source_type in ['image', 'audio']:
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = uploads_dir / unique_filename
            with open(file_path, 'wb') as f:
                f.write(file_data)

        # Save file temporarily to process
        temp_path = Path(f"temp_{filename}")
        with open(temp_path, 'wb') as f:
            f.write(file_data)

        try:
            if source_type == 'text':
                content = process_text_file(temp_path)
            elif source_type == 'word':
                content = process_word_file(temp_path)
            elif source_type == 'audio':
                content = process_audio_file(temp_path)
            elif source_type == 'image':
                content = process_image_file(temp_path)
            else:
                continue  # Skip unsupported types

            # Prepend user description if provided
            if description:
                content = f"Description: {description}\n\n{content}"

            memory = {
                'content': content,
                'filename': filename,
                'source_type': source_type,
                'upload_time': datetime.now().isoformat(),
                'year': None,  # Can be extracted later if needed
                'file_path': unique_filename if file_path else None
            }
            memories.append(memory)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            # Clean up saved image/audio file if processing failed
            if file_path and file_path.exists():
                file_path.unlink()
        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()

    return memories
