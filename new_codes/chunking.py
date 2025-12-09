import tiktoken
from typing import List, Dict, Optional
import uuid


def get_tokenizer(model: str = "gpt-4o-mini"):
    """
    Returns the correct tokenizer for the chosen OpenAI model.
    Works for GPT-3.5, GPT-4, GPT-4.1, GPT-5.1, and embedding models.
    
    Args:
        model: OpenAI model name (e.g., "gpt-4o-mini", "gpt-4", "text-embedding-3-small")
        
    Returns:
        tiktoken encoding object
    """
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base (used by GPT-3.5, GPT-4, and most OpenAI models)
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            raise ValueError(f"Could not load tokenizer for model {model}: {str(e)}")


def count_tokens(text: str, tokenizer=None, model: str = "gpt-4o-mini") -> int:
    """
    Returns number of tokens in a text string.
    
    Args:
        text: Input text string
        tokenizer: Optional tiktoken encoding object (if None, will create one)
        model: OpenAI model name (used if tokenizer is None)
        
    Returns:
        Number of tokens in the text
    """
    if tokenizer is None:
        tokenizer = get_tokenizer(model)
    return len(tokenizer.encode(text))


def find_word_boundaries(text: str, chunk_text: str, chunk_start_token: int, tokenizer, total_tokens: int) -> tuple:
    """
    Find approximate word boundaries for a chunk within the original text.
    
    Args:
        text: Original full text
        chunk_text: Decoded chunk text
        chunk_start_token: Starting token index of the chunk
        tokenizer: tiktoken encoding object
        total_tokens: Total number of tokens in the original text
        
    Returns:
        Tuple of (start_word_index, end_word_index)
    """
    # Split original text into words
    original_words = text.split()
    
    # Split chunk text into words
    chunk_words = chunk_text.split()
    
    if not chunk_words:
        return (0, 0)
    
    # Try to find the chunk's position in the original text
    # Use first few words of chunk to locate it
    search_text = " ".join(chunk_words[:min(5, len(chunk_words))])
    
    # Find position in original text
    chunk_start_char = text.find(search_text)
    
    if chunk_start_char == -1:
        # Fallback: estimate based on token ratio
        if total_tokens > 0:
            token_ratio = chunk_start_token / total_tokens
            estimated_char_pos = int(len(text) * token_ratio)
            chunk_start_char = max(0, estimated_char_pos - 100)  # Look back a bit
        else:
            chunk_start_char = 0
    
    # Count words that appear before chunk_start_char
    text_before_chunk = text[:chunk_start_char]
    words_before = len(text_before_chunk.split())
    
    start_word = words_before
    end_word = start_word + len(chunk_words)
    
    return (start_word, min(end_word, len(original_words)))


def chunk_text(
    text: str,
    model: str = "gpt-4o-mini",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    source: Optional[str] = None,
) -> List[Dict]:
    """
    Splits text into overlapping token-based chunks with metadata.
    
    Parameters:
        text: Input string to chunk
        model: Which OpenAI model/tokenizer to use (default: "gpt-4o-mini")
        chunk_size: Maximum tokens per chunk (default: 500)
        chunk_overlap: Number of tokens to overlap between chunks (default: 50)
        source: Source filename/path for metadata (optional)
        
    Returns:
        List of dictionaries, each containing:
        - id: Unique identifier for the chunk
        - text: Chunk text content
        - source: Source filename/path
        - chunk_id: Sequential chunk number (0-indexed)
        - start_token: Starting token index
        - end_token: Ending token index
        - start_word: Starting word index in original text
        - end_word: Ending word index in original text
        - token_count: Number of tokens in this chunk
    """
    if not text or not text.strip():
        return []
    
    # Validate parameters
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")
    
    # Get tokenizer
    tokenizer = get_tokenizer(model)
    
    # Encode text into tokens
    tokens = tokenizer.encode(text)
    
    if len(tokens) == 0:
        return []
    
    # If text fits in one chunk, return it
    if len(tokens) <= chunk_size:
        words = text.split()
        return [{
            "id": str(uuid.uuid4()),
            "text": text,
            "source": source or "unknown",
            "chunk_id": 0,
            "start_token": 0,
            "end_token": len(tokens),
            "start_word": 0,
            "end_word": len(words),
            "token_count": len(tokens)
        }]
    
    # Split text into words for word position tracking
    words = text.split()
    
    chunks = []
    start = 0
    chunk_id = 0
    
    while start < len(tokens):
        # Calculate end position
        end = min(start + chunk_size, len(tokens))
        
        # Extract token slice
        token_slice = tokens[start:end]
        
        # Decode token slice back to text
        chunk_text_decoded = tokenizer.decode(token_slice)
        
        # Find word boundaries for this chunk
        start_word, end_word = find_word_boundaries(text, chunk_text_decoded, start, tokenizer, len(tokens))
        
        # Create chunk dictionary
        chunk = {
            "id": str(uuid.uuid4()),
            "text": chunk_text_decoded,
            "source": source or "unknown",
            "chunk_id": chunk_id,
            "start_token": start,
            "end_token": end,
            "start_word": start_word,
            "end_word": min(end_word, len(words)),  # Ensure we don't exceed word count
            "token_count": len(token_slice)
        }
        
        chunks.append(chunk)
        
        # Move window with overlap
        start += (chunk_size - chunk_overlap)
        chunk_id += 1
        
        # Prevent infinite loop if overlap is too large relative to remaining tokens
        if start >= len(tokens):
            break
    
    return chunks


def chunk_documents(
    documents: List[Dict],
    model: str = "gpt-4o-mini",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Dict]:
    """
    Chunk multiple documents loaded from load_files.py.
    
    Parameters:
        documents: List of document dictionaries from load_files.py
                   Each dict should have 'text' and 'metadata' keys
        model: Which OpenAI model/tokenizer to use (default: "gpt-4o-mini")
        chunk_size: Maximum tokens per chunk (default: 500)
        chunk_overlap: Number of tokens to overlap between chunks (default: 50)
        
    Returns:
        List of chunk dictionaries with all metadata
    """
    all_chunks = []
    
    for doc in documents:
        text = doc.get("text", "")
        metadata = doc.get("metadata", {})
        source = metadata.get("source", metadata.get("file_path", "unknown"))
        
        chunks = chunk_text(
            text=text,
            model=model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            source=source
        )
        
        # Add document-level metadata to each chunk
        for chunk in chunks:
            chunk["document_metadata"] = metadata
        
        all_chunks.extend(chunks)
    
    return all_chunks


def chunk_text_simple(
    text: str,
    model: str = "gpt-4o-mini",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[str]:
    """
    Simple version that returns only chunk text strings (for backward compatibility).
    
    Parameters:
        text: Input string to chunk
        model: Which OpenAI model/tokenizer to use
        chunk_size: Maximum tokens per chunk
        chunk_overlap: Number of tokens to overlap between chunks
        
    Returns:
        List of chunk strings
    """
    chunks = chunk_text(text, model, chunk_size, chunk_overlap)
    return [chunk["text"] for chunk in chunks]

