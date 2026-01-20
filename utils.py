"""
backend/utils.py

Utility functions for the Living Memory Vault.
"""

import re
from typing import Optional, Dict

def extract_year_from_content(content: str) -> Optional[int]:
    """Extract a year from the content using regex."""
    year_pattern = r'\b(19|20)\d{2}\b'
    match = re.search(year_pattern, content)
    if match:
        return int(match.group())
    return None

def format_memory_for_display(memory: Dict) -> str:
    """Format a memory dict for display in the UI."""
    return f"**{memory['metadata']['filename']}** ({memory['metadata']['source_type']}, {memory['metadata']['upload_time'][:10]}): {memory['content'][:200]}..."
