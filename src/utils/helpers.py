"""
Utility functions and helpers for the performance review system.

This module provides common functionality used across multiple components,
including date handling, text processing, and validation utilities.
"""
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple


def parse_review_cycle(review_cycle: str) -> Tuple[int, int]:
    """
    Parse a review cycle string to extract year and quarter.
    
    Supports multiple formats:
    - "2025-Q3" -> (2025, 3)
    - "annual-2025" -> (2025, 0)
    - "2025" -> (2025, 0)
    
    Args:
        review_cycle: Review cycle string
        
    Returns:
        Tuple of (year, quarter) where quarter is 0 for annual reviews
    """
    try:
        # Format: "2025-Q3"
        if "-Q" in review_cycle:
            year_str, quarter_str = review_cycle.split("-Q")
            return int(year_str), int(quarter_str)
        
        # Format: "annual-2025"
        elif "annual" in review_cycle.lower():
            year_str = review_cycle.split("-")[-1]
            return int(year_str), 0
        
        # Format: "2025"
        else:
            return int(review_cycle), 0
            
    except (ValueError, IndexError):
        # Default to current year if parsing fails
        return datetime.now().year, 0


def format_review_cycle(year: int, quarter: int = 0) -> str:
    """
    Format year and quarter into a review cycle string.
    
    Args:
        year: The year
        quarter: Quarter number (1-4), or 0 for annual
        
    Returns:
        Formatted review cycle string
    """
    if quarter == 0:
        return f"annual-{year}"
    else:
        return f"{year}-Q{quarter}"


def get_current_review_cycle(quarterly: bool = True) -> str:
    """
    Get the current review cycle based on today's date.
    
    Args:
        quarterly: If True, returns current quarter; if False, returns annual
        
    Returns:
        Current review cycle string
    """
    now = datetime.now()
    year = now.year
    
    if quarterly:
        quarter = (now.month - 1) // 3 + 1
        return f"{year}-Q{quarter}"
    else:
        return f"annual-{year}"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length, adding a suffix if truncated.
    
    This intelligently truncates at word boundaries when possible.
    
    Args:
        text: Text to truncate
        max_length: Maximum length including suffix
        suffix: String to append if text is truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    # Account for suffix length
    effective_length = max_length - len(suffix)
    
    # Try to truncate at a word boundary
    truncated = text[:effective_length]
    last_space = truncated.rfind(' ')
    
    if last_space > effective_length * 0.8:  # If space is reasonably close to end
        truncated = truncated[:last_space]
    
    return truncated + suffix


def extract_sentences(text: str, max_sentences: int = 3) -> str:
    """
    Extract the first N sentences from text.
    
    This is useful for creating summaries or previews.
    
    Args:
        text: Source text
        max_sentences: Maximum number of sentences to extract
        
    Returns:
        Text containing only the first N sentences
    """
    # Split on sentence boundaries (., !, ?)
    sentences = re.split(r'[.!?]+', text)
    
    # Take first N non-empty sentences
    selected = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            selected.append(sentence)
            if len(selected) >= max_sentences:
                break
    
    # Rejoin with periods
    return '. '.join(selected) + '.' if selected else ''


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    separator: str = "\n\n"
) -> List[str]:
    """
    Split text into overlapping chunks for processing.
    
    This is useful when text exceeds LLM context limits and needs to be
    processed in segments.
    
    Args:
        text: Text to chunk
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        separator: Preferred separator to split on (e.g., paragraphs)
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Calculate end position
        end = start + chunk_size
        
        # If this is not the last chunk, try to find a good break point
        if end < len(text):
            # Look for separator within the last 20% of the chunk
            search_start = end - int(chunk_size * 0.2)
            separator_pos = text.rfind(separator, search_start, end)
            
            if separator_pos > search_start:
                end = separator_pos + len(separator)
        
        # Extract chunk
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap
    
    return chunks


def validate_employee_id(employee_id: str) -> bool:
    """
    Validate employee ID format.
    
    This is a basic validation - customize for your organization's format.
    
    Args:
        employee_id: Employee ID to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Example: must be alphanumeric, 3-20 characters
    if not employee_id:
        return False
    
    if len(employee_id) < 3 or len(employee_id) > 20:
        return False
    
    # Allow letters, numbers, hyphens, and underscores
    if not re.match(r'^[a-zA-Z0-9_-]+$', employee_id):
        return False
    
    return True


def calculate_response_statistics(qa_pairs: List[Dict]) -> Dict:
    """
    Calculate statistics about interview responses.
    
    This provides metrics about the quality and depth of an interview session.
    
    Args:
        qa_pairs: List of question-answer pair dicts
        
    Returns:
        Dict with statistics
    """
    if not qa_pairs:
        return {
            "total_pairs": 0,
            "avg_answer_length": 0,
            "avg_depth_score": 0,
            "unique_topics": 0,
            "total_words": 0
        }
    
    total_length = 0
    total_depth = 0
    total_words = 0
    all_topics = set()
    
    for qa in qa_pairs:
        # Answer length
        answer = qa.get("answer", "")
        total_length += len(answer)
        total_words += len(answer.split())
        
        # Depth score
        depth = qa.get("depth_score", 0)
        total_depth += depth
        
        # Topics
        topics = qa.get("topics", [])
        all_topics.update(topics)
    
    n = len(qa_pairs)
    
    return {
        "total_pairs": n,
        "avg_answer_length": total_length // n if n > 0 else 0,
        "avg_depth_score": total_depth / n if n > 0 else 0,
        "unique_topics": len(all_topics),
        "total_words": total_words,
        "topics": list(all_topics)
    }


def format_datetime(dt: datetime, format_type: str = "full") -> str:
    """
    Format datetime objects consistently across the application.
    
    Args:
        dt: Datetime object to format
        format_type: Type of format - "full", "date", "time", or "short"
        
    Returns:
        Formatted datetime string
    """
    if format_type == "full":
        return dt.strftime("%B %d, %Y at %H:%M:%S")
    elif format_type == "date":
        return dt.strftime("%B %d, %Y")
    elif format_type == "time":
        return dt.strftime("%H:%M:%S")
    elif format_type == "short":
        return dt.strftime("%Y-%m-%d %H:%M")
    else:
        return dt.isoformat()


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing or replacing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename safe for all operating systems
    """
    # Remove or replace characters that are invalid in filenames
    invalid_chars = '<>:"/\\|?*'
    
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')
    
    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        max_name_len = 255 - len(ext) - 1
        filename = name[:max_name_len] + ('.' + ext if ext else '')
    
    return filename


def merge_metadata(base: Dict, override: Dict) -> Dict:
    """
    Merge two metadata dictionaries, with override taking precedence.
    
    This is useful when combining default metadata with session-specific values.
    
    Args:
        base: Base metadata dict
        override: Override metadata dict
        
    Returns:
        Merged metadata dict
    """
    merged = base.copy()
    merged.update({k: v for k, v in override.items() if v is not None})
    return merged


def extract_competency_mentions(text: str, competencies: List[str]) -> List[str]:
    """
    Extract which competencies are mentioned in the text.
    
    This uses case-insensitive matching to identify competency areas
    discussed in responses.
    
    Args:
        text: Text to analyze
        competencies: List of competency names to look for
        
    Returns:
        List of mentioned competencies
    """
    text_lower = text.lower()
    mentioned = []
    
    for competency in competencies:
        # Check for the competency name or common variations
        competency_lower = competency.lower()
        
        # Direct mention
        if competency_lower in text_lower:
            mentioned.append(competency)
            continue
        
        # Check for key words from multi-word competencies
        words = competency_lower.split()
        if len(words) > 1:
            # If at least half the words are present, consider it mentioned
            word_count = sum(1 for word in words if word in text_lower)
            if word_count >= len(words) / 2:
                mentioned.append(competency)
    
    return mentioned


def calculate_similarity_score(text1: str, text2: str) -> float:
    """
    Calculate a simple similarity score between two texts.
    
    This uses word overlap as a basic similarity metric. For production,
    you might want to use embeddings from the vector store instead.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    # Convert to lowercase and split into words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Calculate Jaccard similarity
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def format_list_as_text(items: List[str], conjunction: str = "and") -> str:
    """
    Format a list of items as grammatically correct text.
    
    Examples:
    - ["apple"] -> "apple"
    - ["apple", "banana"] -> "apple and banana"
    - ["apple", "banana", "cherry"] -> "apple, banana, and cherry"
    
    Args:
        items: List of items to format
        conjunction: Conjunction to use (usually "and" or "or")
        
    Returns:
        Formatted string
    """
    if not items:
        return ""
    
    if len(items) == 1:
        return items[0]
    
    if len(items) == 2:
        return f"{items[0]} {conjunction} {items[1]}"
    
    # Oxford comma for lists of 3+
    return ", ".join(items[:-1]) + f", {conjunction} {items[-1]}"