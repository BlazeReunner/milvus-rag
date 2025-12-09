"""
Reranker Module
Uses cross-encoder model to rerank candidate chunks for better relevance ordering.

The reranker takes initial search results from Milvus and reorders them based on
cross-encoder scores, which provide more accurate relevance than pure vector similarity.
"""

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    raise ImportError(
        "sentence-transformers is not installed. "
        "Install it with: pip install sentence-transformers"
    )

from typing import List, Dict, Tuple, Optional
import warnings

# Initialize the reranker model (lazy loading)
_reranker = None
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # fast, small, good quality


def get_reranker() -> CrossEncoder:
    """
    Get or initialize the reranker model (singleton pattern).
    
    Returns:
        CrossEncoder instance
    """
    global _reranker
    if _reranker is None:
        print(f"Loading reranker model: {RERANKER_MODEL}")
        # Explicitly set device to avoid meta tensor issues
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _reranker = CrossEncoder(RERANKER_MODEL, device=device)
        print(f"✓ Reranker model loaded successfully on {device}")
    return _reranker


def rerank(
    query: str,
    candidates: List[Dict],
    top_k: Optional[int] = None
) -> List[Dict]:
    """
    Rerank candidate chunks based on query relevance using cross-encoder.
    
    Args:
        query: The search query string
        candidates: List of candidate dictionaries. Each dict should have:
            - 'text': The chunk text content (required)
            - Any other metadata fields (e.g., 'source', 'chunk_id', 'distance', etc.)
        top_k: Optional. If provided, return only top_k results after reranking.
               If None, return all reranked candidates.
    
    Returns:
        List of candidate dictionaries, sorted by relevance score (highest first).
        Each dict will have an additional 'rerank_score' field.
    
    Example:
        >>> candidates = [
        ...     {'text': 'Some chunk text', 'source': 'doc1.pdf', 'distance': 0.8},
        ...     {'text': 'Another chunk', 'source': 'doc2.pdf', 'distance': 0.7}
        ... ]
        >>> reranked = rerank("what is machine learning?", candidates)
        >>> print(reranked[0]['rerank_score'])  # Highest relevance score
    """
    if not candidates:
        return []
    
    if not query or not query.strip():
        warnings.warn("Empty query provided to reranker. Returning candidates unchanged.")
        return candidates
    
    # Get reranker model
    reranker = get_reranker()
    
    # Prepare query-candidate pairs for scoring
    # Format: [[query, candidate_text], ...]
    pairs = [[query, candidate.get('text', '')] for candidate in candidates]
    
    # Get relevance scores from cross-encoder
    # Higher score = more relevant
    scores = reranker.predict(pairs)
    
    # Add rerank_score to each candidate and sort by score
    for candidate, score in zip(candidates, scores):
        candidate['rerank_score'] = float(score)
    
    # Sort by rerank_score (descending - higher is better)
    reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
    
    # Return top_k if specified
    if top_k is not None:
        return reranked[:top_k]
    
    return reranked


def rerank_from_search_results(
    query: str,
    search_results: List[Tuple[str, float, Dict]],
    top_k: Optional[int] = None
) -> List[Dict]:
    """
    Convenience function to rerank results from vectorstore.search().
    
    This function converts the tuple format from vectorstore.search() into
    the dict format expected by rerank(), then reranks all candidates.
    
    Typical workflow:
        1. Search Milvus for top 200 chunks (vectorstore.search() with top_k=200)
        2. Rerank all 200 chunks using this function (top_k=None to get all reranked)
        3. Optionally filter to top N after reranking (top_k=N)
    
    Args:
        query: The search query string
        search_results: List of tuples from vectorstore.search():
            [(text, distance, metadata_dict), ...]
            Typically 200 results from initial vector search.
        top_k: Optional. If None, returns all reranked results.
               If provided, returns only top_k results after reranking.
    
    Returns:
        List of dictionaries with reranked results. Each dict contains:
            - 'text': The chunk text
            - 'distance': Original vector distance
            - 'rerank_score': Cross-encoder relevance score
            - All metadata fields from the original search results
        Results are sorted by rerank_score (highest first).
    
    Example:
        >>> from vectorstore import search
        >>> from embed import embed_text
        >>> 
        >>> query = "what is machine learning?"
        >>> query_vector = embed_text(query)
        >>> # Step 1: Get top 200 chunks from vector search (with embeddings for efficiency)
        >>> results = search("my_rag_collection", query_vector, top_k=200, include_embeddings=True)
        >>> # Step 2: Rerank all 200 chunks (embeddings preserved for downstream MMR)
        >>> reranked = rerank_from_search_results(query, results, top_k=None)
        >>> # Step 3: Optionally get top 5 after reranking
        >>> top_5 = reranked[:5]
    """
    # Convert tuple format to dict format
    candidates = []
    for text, distance, metadata in search_results:
        candidate = {
            'text': text,
            'distance': distance,
            **metadata  # Unpack all metadata fields (includes 'embedding' if available)
        }
        candidates.append(candidate)
    
    # Rerank and return (embeddings will be preserved in candidates)
    return rerank(query, candidates, top_k=top_k)


if __name__ == "__main__":
    # Example usage
    print("=" * 70)
    print("RERANKER MODULE TEST")
    print("=" * 70)
    
    # Sample query and candidates
    query = "What is machine learning?"
    candidates = [
        {
            'text': 'Machine learning is a subset of artificial intelligence that enables systems to learn from data.',
            'source': 'ai_basics.pdf',
            'chunk_id': 0,
            'distance': 0.85
        },
        {
            'text': 'Python is a popular programming language used for data science.',
            'source': 'python_guide.pdf',
            'chunk_id': 1,
            'distance': 0.75
        },
        {
            'text': 'Deep learning uses neural networks with multiple layers to learn complex patterns.',
            'source': 'deep_learning.pdf',
            'chunk_id': 2,
            'distance': 0.80
        }
    ]
    
    print(f"\nQuery: {query}")
    print(f"\nOriginal order (by distance):")
    for i, cand in enumerate(candidates, 1):
        print(f"  {i}. Distance: {cand['distance']:.3f} | {cand['text'][:60]}...")
    
    # Rerank
    print(f"\nReranking candidates...")
    reranked = rerank(query, candidates)
    
    print(f"\nReranked order (by relevance):")
    for i, cand in enumerate(reranked, 1):
        print(f"  {i}. Score: {cand['rerank_score']:.3f} | {cand['text'][:60]}...")
    
    print("\n✓ Reranker test completed successfully!")
    print("=" * 70)

