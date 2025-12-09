"""
MMR (Maximal Marginal Relevance) Module
Selects diverse chunks from reranked results to avoid redundancy.

MMR balances relevance and diversity:
- Relevance: How relevant is the chunk to the query (rerank_score)
- Diversity: How different is it from already selected chunks

Formula: MMR = λ * relevance - (1-λ) * max_similarity_to_selected
"""

from typing import List, Dict, Optional
import numpy as np
from embed import embed_text


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        Cosine similarity score between -1 and 1
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def mmr_select(
    reranked_chunks: List[Dict],
    top_n: int = 15,
    final_k: int = 6,
    lambda_param: float = 0.7,
    query: Optional[str] = None
) -> List[Dict]:
    """
    Select diverse chunks using Maximal Marginal Relevance (MMR).
    
    Process:
        1. Take top N chunks from reranked results (by rerank_score)
        2. Apply MMR to select final_k diverse chunks
    
    Args:
        reranked_chunks: List of reranked chunk dictionaries, each containing:
            - 'text': Chunk text content (required)
            - 'rerank_score': Relevance score from reranker (required)
            - Any other metadata fields
        top_n: Number of top chunks to consider for MMR (default: 15)
        final_k: Number of diverse chunks to select (default: 6)
        lambda_param: MMR lambda parameter (default: 0.7)
                     Higher λ = more weight on relevance, less on diversity
                     Lower λ = more weight on diversity, less on relevance
        query: Optional query string. If provided, uses query embedding for 
               better similarity calculation. If None, uses chunk-to-chunk similarity.
    
    Returns:
        List of final_k diverse chunk dictionaries, each with:
            - All original fields
            - 'mmr_score': MMR score used for selection
            - 'embedding': Cached embedding vector (for efficiency)
    
    Example:
        >>> from reranker import rerank_from_search_results
        >>> from vectorstore import search
        >>> from embed import embed_text
        >>> 
        >>> query = "what is machine learning?"
        >>> query_vector = embed_text(query)
        >>> results = search("my_rag_collection", query_vector, top_k=200)
        >>> reranked = rerank_from_search_results(query, results)
        >>> 
        >>> # Select 6 diverse chunks from top 15
        >>> diverse_chunks = mmr_select(reranked, top_n=15, final_k=6, lambda_param=0.7, query=query)
    """
    if not reranked_chunks:
        return []
    
    if final_k <= 0:
        return []
    
    # Step 1: Take top N chunks from reranked results
    top_n_chunks = reranked_chunks[:top_n]
    
    if len(top_n_chunks) <= final_k:
        # If we have fewer chunks than final_k, return all
        return top_n_chunks
    
    # Step 2: Pre-compute embeddings for all top_n chunks
    # Check if embeddings are already available (from vectorstore.search with include_embeddings=True)
    chunk_embeddings = {}
    embeddings_from_milvus = 0
    embeddings_computed = 0
    
    for i, chunk in enumerate(top_n_chunks):
        # Check if embedding already exists in chunk
        if 'embedding' in chunk and chunk['embedding'] is not None:
            chunk_embeddings[i] = chunk['embedding']
            embeddings_from_milvus += 1
        else:
            # Compute embedding if not available
            try:
                chunk_embeddings[i] = embed_text(chunk.get('text', ''))
                embeddings_computed += 1
            except Exception as e:
                print(f"Warning: Failed to embed chunk {i}: {str(e)}")
                # Use zero vector as fallback
                chunk_embeddings[i] = [0.0] * 1536  # Default embedding dimension
                embeddings_computed += 1
    
    if embeddings_from_milvus > 0:
        print(f"Using {embeddings_from_milvus} embeddings from Milvus, computed {embeddings_computed} new embeddings")
    else:
        print(f"Computing embeddings for {len(top_n_chunks)} chunks...")
    
    # Get query embedding if provided
    query_embedding = None
    if query:
        try:
            query_embedding = embed_text(query)
        except Exception as e:
            print(f"Warning: Failed to embed query: {str(e)}")
    
    # Step 3: Apply MMR algorithm
    selected_indices = []
    selected_chunks = []
    
    # Start with the most relevant chunk (highest rerank_score)
    selected_indices.append(0)
    selected_chunks.append(top_n_chunks[0].copy())
    selected_chunks[0]['mmr_score'] = top_n_chunks[0].get('rerank_score', 0.0)
    selected_chunks[0]['embedding'] = chunk_embeddings[0]
    
    print(f"Selecting {final_k} diverse chunks using MMR (λ={lambda_param})...")
    
    # Select remaining chunks using MMR
    while len(selected_indices) < final_k:
        best_mmr_score = float('-inf')
        best_idx = None
        
        for i, chunk in enumerate(top_n_chunks):
            if i in selected_indices:
                continue
            
            # Get relevance score (normalized rerank_score)
            rerank_score = chunk.get('rerank_score', 0.0)
            
            # Normalize rerank_score to [0, 1] range for better balance
            # Assuming rerank scores are typically in range [-10, 15]
            # We'll use a simple normalization
            normalized_relevance = max(0.0, min(1.0, (rerank_score + 10) / 25))
            
            # Calculate max similarity to already selected chunks
            max_similarity = 0.0
            chunk_emb = chunk_embeddings[i]
            
            for selected_idx in selected_indices:
                selected_emb = chunk_embeddings[selected_idx]
                similarity = cosine_similarity(chunk_emb, selected_emb)
                max_similarity = max(max_similarity, similarity)
            
            # Calculate MMR score
            # MMR = λ * relevance - (1-λ) * max_similarity
            mmr_score = (lambda_param * normalized_relevance) - ((1 - lambda_param) * max_similarity)
            
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = i
        
        if best_idx is not None:
            selected_indices.append(best_idx)
            selected_chunk = top_n_chunks[best_idx].copy()
            selected_chunk['mmr_score'] = best_mmr_score
            selected_chunk['embedding'] = chunk_embeddings[best_idx]
            selected_chunks.append(selected_chunk)
    
    print(f"✓ Selected {len(selected_chunks)} diverse chunks")
    
    return selected_chunks


def mmr_from_reranked(
    reranked_chunks: List[Dict],
    query: Optional[str] = None
) -> List[Dict]:
    """
    Convenience function with default parameters for the workflow.
    
    From 200 reranked chunks:
    - Takes top 15 chunks
    - Applies MMR with λ=0.7
    - Returns final_k=6 diverse chunks
    
    Args:
        reranked_chunks: List of reranked chunk dictionaries from reranker
        query: Optional query string for better similarity calculation
    
    Returns:
        List of 6 diverse chunk dictionaries
    """
    return mmr_select(
        reranked_chunks=reranked_chunks,
        top_n=15,
        final_k=6,
        lambda_param=0.7,
        query=query
    )


if __name__ == "__main__":
    # Example usage
    print("=" * 70)
    print("MMR MODULE TEST")
    print("=" * 70)
    
    # Simulate reranked chunks (would come from reranker in real usage)
    query = "What is machine learning?"
    reranked_chunks = [
        {
            'text': 'Machine learning is a subset of artificial intelligence that enables systems to learn from data.',
            'rerank_score': 11.063,
            'source': 'ai_basics.pdf',
            'chunk_id': 0
        },
        {
            'text': 'Machine learning algorithms can be supervised, unsupervised, or reinforcement learning.',
            'rerank_score': 10.5,
            'source': 'ai_basics.pdf',
            'chunk_id': 1
        },
        {
            'text': 'Deep learning uses neural networks with multiple layers to learn complex patterns.',
            'rerank_score': 9.2,
            'source': 'deep_learning.pdf',
            'chunk_id': 2
        },
        {
            'text': 'Neural networks are inspired by the structure of the human brain.',
            'rerank_score': 8.5,
            'source': 'deep_learning.pdf',
            'chunk_id': 3
        },
        {
            'text': 'Python is a popular programming language used for data science and machine learning.',
            'rerank_score': 7.8,
            'source': 'python_guide.pdf',
            'chunk_id': 4
        },
        {
            'text': 'TensorFlow and PyTorch are popular frameworks for building machine learning models.',
            'rerank_score': 7.2,
            'source': 'frameworks.pdf',
            'chunk_id': 5
        },
        {
            'text': 'Data preprocessing is an important step in machine learning pipelines.',
            'rerank_score': 6.9,
            'source': 'ml_pipeline.pdf',
            'chunk_id': 6
        },
        {
            'text': 'Feature engineering involves selecting and transforming input variables.',
            'rerank_score': 6.5,
            'source': 'ml_pipeline.pdf',
            'chunk_id': 7
        },
        {
            'text': 'Cross-validation helps evaluate model performance on unseen data.',
            'rerank_score': 6.1,
            'source': 'ml_pipeline.pdf',
            'chunk_id': 8
        },
        {
            'text': 'Overfitting occurs when a model learns training data too well.',
            'rerank_score': 5.8,
            'source': 'ml_pipeline.pdf',
            'chunk_id': 9
        },
        {
            'text': 'Regularization techniques help prevent overfitting in machine learning models.',
            'rerank_score': 5.5,
            'source': 'ml_pipeline.pdf',
            'chunk_id': 10
        },
        {
            'text': 'Gradient descent is an optimization algorithm used to train neural networks.',
            'rerank_score': 5.2,
            'source': 'optimization.pdf',
            'chunk_id': 11
        },
        {
            'text': 'Backpropagation is used to calculate gradients in neural networks.',
            'rerank_score': 5.0,
            'source': 'optimization.pdf',
            'chunk_id': 12
        },
        {
            'text': 'Convolutional neural networks are effective for image recognition tasks.',
            'rerank_score': 4.8,
            'source': 'cnn.pdf',
            'chunk_id': 13
        },
        {
            'text': 'Recurrent neural networks are designed for sequential data processing.',
            'rerank_score': 4.5,
            'source': 'rnn.pdf',
            'chunk_id': 14
        }
    ]
    
    print(f"\nQuery: {query}")
    print(f"\nInput: {len(reranked_chunks)} reranked chunks")
    print(f"Top 15 chunks by rerank_score:")
    for i, chunk in enumerate(reranked_chunks[:15], 1):
        print(f"  {i}. Score: {chunk['rerank_score']:.2f} | {chunk['text'][:60]}...")
    
    # Apply MMR
    print(f"\nApplying MMR (top_n=15, final_k=6, λ=0.7)...")
    diverse_chunks = mmr_from_reranked(reranked_chunks, query=query)
    
    print(f"\nSelected {len(diverse_chunks)} diverse chunks:")
    for i, chunk in enumerate(diverse_chunks, 1):
        print(f"  {i}. MMR: {chunk.get('mmr_score', 0):.3f} | "
              f"Rerank: {chunk['rerank_score']:.2f} | "
              f"{chunk['text'][:60]}...")
    
    print("\n✓ MMR test completed successfully!")
    print("=" * 70)

