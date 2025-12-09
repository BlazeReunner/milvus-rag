"""
Answer Generation Module
Takes diverse chunks from MMR and generates answers using GPT-4o.

This module formats chunks with provenance information and sends them to the LLM
to generate accurate, traceable answers.
"""

from typing import List, Dict, Optional
from openai import OpenAI

# Import config to load API key from .env
try:
    import config
except ImportError:
    from dotenv import load_dotenv
    import os
    load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Initialize OpenAI client (will use OPENAI_API_KEY from environment)
openai_client = OpenAI()

# LLM model configuration
LLM_MODEL = "gpt-4o"  # Using GPT-4o as specified

# System prompt to prevent hallucination and ensure citation
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based only on the provided context.

Important guidelines:
1. Use ONLY the information provided in the context below to answer the question.
2. If the answer is not present in the context, explicitly say "I don't know" or "The information is not available in the provided context."
3. Do NOT make up information, speculate, or use knowledge outside the provided context.
4. Always cite your sources using the format: (Source: filename.pdf, Chunk: X)
5. If multiple sources support your answer, cite all of them.
6. Be precise and accurate in your response."""


def build_prompt(query: str, chunks: List[Dict]) -> List[Dict[str, str]]:
    """
    Build a prompt for the LLM with formatted context chunks and query.
    
    Args:
        query: The user's question
        chunks: List of chunk dictionaries from MMR, each containing:
            - 'text': Chunk text content (required)
            - 'source': Source filename (required)
            - 'chunk_id': Chunk ID number (required)
            - Any other metadata fields
    
    Returns:
        List of message dictionaries in OpenAI chat format:
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": formatted_prompt}
        ]
    """
    # Format each chunk with provenance information
    formatted_chunks = []
    for i, chunk in enumerate(chunks, 1):
        text = chunk.get('text', '')
        
        # Try multiple ways to get source
        source = chunk.get('source', 'unknown')
        if source == 'unknown' and 'document_metadata' in chunk:
            doc_meta = chunk['document_metadata']
            source = doc_meta.get('source', doc_meta.get('file_path', 'unknown'))
            # Extract just filename if it's a full path
            if '/' in source:
                source = source.split('/')[-1]
        
        # Try multiple ways to get chunk_id
        chunk_id = chunk.get('chunk_id', 'N/A')
        if chunk_id == 'N/A':
            # Try alternative field names or use index
            chunk_id = chunk.get('chunk_id', chunk.get('id', i-1))
        
        # Format: [1] (Source: filename.pdf, Chunk: 5)
        # chunk text here...
        formatted_chunk = f"[{i}] (Source: {source}, Chunk: {chunk_id})\n{text}"
        formatted_chunks.append(formatted_chunk)
    
    # Combine all chunks with double newline separator
    context_text = "\n\n".join(formatted_chunks)
    
    # Build user message with context and question
    user_message = f"""<context>
{context_text}
</context>

<question>
{query}
</question>

Please answer the question using only the information from the context above. Cite your sources using the format (Source: filename.pdf, Chunk: X)."""
    
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]


def generate_answer(query: str, chunks: List[Dict], model: str = LLM_MODEL, temperature: float = 0.0) -> Dict:
    """
    Generate an answer from diverse chunks using GPT-4o.
    
    Args:
        query: The user's question
        chunks: List of diverse chunk dictionaries from MMR
        model: OpenAI model to use (default: "gpt-4o")
        temperature: Sampling temperature (default: 0.0 for deterministic responses)
    
    Returns:
        Dictionary containing:
            - 'answer': The generated answer text
            - 'sources': List of source files cited
            - 'chunks_used': Number of chunks used
            - 'model': Model used for generation
            - 'raw_response': Full response object from OpenAI
    
    Example:
        >>> from mmr import mmr_from_reranked
        >>> from reranker import rerank_from_search_results
        >>> from vectorstore import search
        >>> from embed import embed_text
        >>> 
        >>> query = "what is machine learning?"
        >>> query_vector = embed_text(query)
        >>> results = search("my_rag_collection", query_vector, top_k=200, include_embeddings=True)
        >>> reranked = rerank_from_search_results(query, results)
        >>> diverse_chunks = mmr_from_reranked(reranked, query=query)
        >>> 
        >>> answer_dict = generate_answer(query, diverse_chunks)
        >>> print(answer_dict['answer'])
    """
    if not chunks:
        return {
            'answer': "I don't know. No relevant context was found.",
            'sources': [],
            'chunks_used': 0,
            'model': model,
            'raw_response': None
        }
    
    # Build prompt
    messages = build_prompt(query, chunks)
    
    try:
        # Call OpenAI API
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        
        # Extract answer
        answer_text = response.choices[0].message.content
        
        # Extract sources from chunks
        sources = list(set(chunk.get('source', 'unknown') for chunk in chunks))
        
        return {
            'answer': answer_text,
            'sources': sources,
            'chunks_used': len(chunks),
            'model': model,
            'raw_response': response
        }
    
    except Exception as e:
        return {
            'answer': f"Error generating answer: {str(e)}",
            'sources': [],
            'chunks_used': len(chunks),
            'model': model,
            'raw_response': None,
            'error': str(e)
        }


def answer_query(query: str, chunks: List[Dict]) -> str:
    """
    Convenience function that returns just the answer string.
    
    Args:
        query: The user's question
        chunks: List of diverse chunk dictionaries from MMR
    
    Returns:
        Answer string
    """
    result = generate_answer(query, chunks)
    return result['answer']


if __name__ == "__main__":
    # Example usage
    print("=" * 70)
    print("ANSWER GENERATION MODULE TEST")
    print("=" * 70)
    
    # Simulate diverse chunks from MMR
    query = "What is machine learning?"
    test_chunks = [
        {
            'text': 'Machine learning is a subset of artificial intelligence that enables systems to learn from data without being explicitly programmed.',
            'source': 'ai_basics.pdf',
            'chunk_id': 0,
            'rerank_score': 11.063,
            'mmr_score': 0.85
        },
        {
            'text': 'Machine learning algorithms can be supervised, unsupervised, or reinforcement learning, each with different approaches to learning from data.',
            'source': 'ai_basics.pdf',
            'chunk_id': 1,
            'rerank_score': 10.5,
            'mmr_score': 0.72
        },
        {
            'text': 'Deep learning uses neural networks with multiple layers to learn complex patterns in data.',
            'source': 'deep_learning.pdf',
            'chunk_id': 2,
            'rerank_score': 9.2,
            'mmr_score': 0.65
        },
        {
            'text': 'Feature engineering involves selecting and transforming input variables to improve model performance.',
            'source': 'ml_pipeline.pdf',
            'chunk_id': 6,
            'rerank_score': 6.9,
            'mmr_score': 0.58
        },
        {
            'text': 'Cross-validation helps evaluate model performance on unseen data by splitting the dataset into training and testing sets.',
            'source': 'ml_pipeline.pdf',
            'chunk_id': 8,
            'rerank_score': 6.1,
            'mmr_score': 0.52
        },
        {
            'text': 'Overfitting occurs when a model learns training data too well and fails to generalize to new data.',
            'source': 'ml_pipeline.pdf',
            'chunk_id': 9,
            'rerank_score': 5.8,
            'mmr_score': 0.48
        }
    ]
    
    print(f"\nQuery: {query}")
    print(f"\nUsing {len(test_chunks)} diverse chunks:")
    for i, chunk in enumerate(test_chunks, 1):
        print(f"  {i}. Source: {chunk['source']}, Chunk: {chunk['chunk_id']}")
    
    print(f"\nGenerating answer using {LLM_MODEL}...")
    result = generate_answer(query, test_chunks)
    
    print(f"\n{'='*70}")
    print("ANSWER:")
    print(f"{'='*70}")
    print(result['answer'])
    print(f"\n{'='*70}")
    print(f"Sources: {', '.join(result['sources'])}")
    print(f"Chunks used: {result['chunks_used']}")
    print(f"Model: {result['model']}")
    print("=" * 70)

