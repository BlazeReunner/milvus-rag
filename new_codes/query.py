"""
RAG Query Interface
Complete search interface that connects all components of the RAG pipeline.

This is the main entry point for querying the RAG system:
1. Embed user query
2. Search Milvus (top 200 chunks)
3. Rerank chunks using cross-encoder
4. Apply MMR for diversity (top 15 → final 6)
5. Generate answer using GPT-4o
6. Return formatted answer with sources
"""

from typing import Dict, List, Optional
from embed import embed_text
from vectorstore import search
from reranker import rerank_from_search_results
from mmr import mmr_select
from answer import generate_answer
import time


def query_rag(
    query: str,
    collection_name: str = "my_rag_collection",
    top_k_search: int = 200,
    top_n_mmr: int = 15,
    final_k_mmr: int = 6,
    lambda_mmr: float = 0.7,
    include_embeddings: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Complete RAG query pipeline: Search → Rerank → MMR → Answer Generation.
    
    Args:
        query: User's question
        collection_name: Name of the Milvus collection
        top_k_search: Number of chunks to retrieve from Milvus (default: 200)
        top_n_mmr: Top N chunks to consider for MMR (default: 15)
        final_k_mmr: Final number of diverse chunks to select (default: 6)
        lambda_mmr: MMR lambda parameter (default: 0.7)
        include_embeddings: Whether to include embeddings in search results (default: True)
        verbose: Whether to print progress messages (default: True)
    
    Returns:
        Dictionary containing:
            - 'answer': Generated answer text
            - 'sources': List of source files used
            - 'chunks_used': Number of chunks used for answer
            - 'query': Original query
            - 'pipeline_stats': Statistics about each step
            - 'raw_answer': Full answer dictionary from answer.py
    
    Example:
        >>> result = query_rag("What is machine learning?")
        >>> print(result['answer'])
        >>> print(f"Sources: {result['sources']}")
    """
    start_time = time.time()
    pipeline_stats = {}
    
    if verbose:
        print("=" * 70)
        print("RAG QUERY PIPELINE")
        print("=" * 70)
        print(f"\nQuery: {query}\n")
    
    # Step 1: Embed query
    if verbose:
        print("[Step 1/5] Embedding query...")
    step_start = time.time()
    try:
        query_vector = embed_text(query)
        pipeline_stats['embedding_time'] = time.time() - step_start
        if verbose:
            print(f"✓ Query embedded ({len(query_vector)} dimensions)")
    except Exception as e:
        return {
            'answer': f"Error embedding query: {str(e)}",
            'sources': [],
            'chunks_used': 0,
            'query': query,
            'pipeline_stats': pipeline_stats,
            'error': str(e)
        }
    
    # Step 2: Search Milvus
    if verbose:
        print(f"\n[Step 2/5] Searching Milvus (top {top_k_search} chunks)...")
    step_start = time.time()
    try:
        search_results = search(
            collection_name=collection_name,
            query_vector=query_vector,
            top_k=top_k_search,
            include_embeddings=include_embeddings
        )
        pipeline_stats['search_time'] = time.time() - step_start
        pipeline_stats['chunks_found'] = len(search_results)
        if verbose:
            print(f"✓ Found {len(search_results)} chunks")
    except Exception as e:
        return {
            'answer': f"Error searching Milvus: {str(e)}",
            'sources': [],
            'chunks_used': 0,
            'query': query,
            'pipeline_stats': pipeline_stats,
            'error': str(e)
        }
    
    if not search_results:
        return {
            'answer': "I don't know. No relevant documents were found in the knowledge base.",
            'sources': [],
            'chunks_used': 0,
            'query': query,
            'pipeline_stats': pipeline_stats
        }
    
    # Step 3: Rerank chunks
    if verbose:
        print(f"\n[Step 3/5] Reranking {len(search_results)} chunks...")
    step_start = time.time()
    try:
        reranked_chunks = rerank_from_search_results(query, search_results, top_k=None)
        pipeline_stats['rerank_time'] = time.time() - step_start
        pipeline_stats['chunks_reranked'] = len(reranked_chunks)
        if verbose:
            print(f"✓ Reranked {len(reranked_chunks)} chunks")
    except Exception as e:
        return {
            'answer': f"Error reranking chunks: {str(e)}",
            'sources': [],
            'chunks_used': 0,
            'query': query,
            'pipeline_stats': pipeline_stats,
            'error': str(e)
        }
    
    # Step 4: Apply MMR for diversity
    if verbose:
        print(f"\n[Step 4/5] Applying MMR (top {top_n_mmr} → {final_k_mmr} diverse chunks)...")
    step_start = time.time()
    try:
        diverse_chunks = mmr_select(
            reranked_chunks=reranked_chunks,
            top_n=top_n_mmr,
            final_k=final_k_mmr,
            lambda_param=lambda_mmr,
            query=query
        )
        pipeline_stats['mmr_time'] = time.time() - step_start
        pipeline_stats['chunks_selected'] = len(diverse_chunks)
        if verbose:
            print(f"✓ Selected {len(diverse_chunks)} diverse chunks")
    except Exception as e:
        return {
            'answer': f"Error applying MMR: {str(e)}",
            'sources': [],
            'chunks_used': 0,
            'query': query,
            'pipeline_stats': pipeline_stats,
            'error': str(e)
        }
    
    # Step 5: Generate answer
    if verbose:
        print(f"\n[Step 5/5] Generating answer using GPT-4o...")
    step_start = time.time()
    try:
        answer_result = generate_answer(query, diverse_chunks)
        pipeline_stats['answer_time'] = time.time() - step_start
        if verbose:
            print(f"✓ Answer generated")
    except Exception as e:
        return {
            'answer': f"Error generating answer: {str(e)}",
            'sources': [],
            'chunks_used': len(diverse_chunks),
            'query': query,
            'pipeline_stats': pipeline_stats,
            'error': str(e)
        }
    
    # Calculate total time
    total_time = time.time() - start_time
    pipeline_stats['total_time'] = total_time
    
    if verbose:
        print("\n" + "=" * 70)
        print("PIPELINE STATISTICS")
        print("=" * 70)
        print(f"Embedding time: {pipeline_stats.get('embedding_time', 0):.2f}s")
        print(f"Search time: {pipeline_stats.get('search_time', 0):.2f}s")
        print(f"Rerank time: {pipeline_stats.get('rerank_time', 0):.2f}s")
        print(f"MMR time: {pipeline_stats.get('mmr_time', 0):.2f}s")
        print(f"Answer generation time: {pipeline_stats.get('answer_time', 0):.2f}s")
        print(f"Total time: {total_time:.2f}s")
        print("=" * 70)
    
    return {
        'answer': answer_result['answer'],
        'sources': answer_result['sources'],
        'chunks_used': answer_result['chunks_used'],
        'query': query,
        'pipeline_stats': pipeline_stats,
        'raw_answer': answer_result
    }


def format_answer(result: Dict) -> str:
    """
    Format the query result for display.
    
    Args:
        result: Result dictionary from query_rag()
    
    Returns:
        Formatted string with answer and sources
    """
    output = []
    output.append("\n" + "=" * 70)
    output.append("ANSWER")
    output.append("=" * 70)
    output.append(result['answer'])
    output.append("\n" + "=" * 70)
    output.append("SOURCES")
    output.append("=" * 70)
    
    if result['sources']:
        for i, source in enumerate(result['sources'], 1):
            output.append(f"{i}. {source}")
    else:
        output.append("No sources available")
    
    output.append("=" * 70)
    
    return "\n".join(output)


def interactive_query(collection_name: str = "my_rag_collection"):
    """
    Interactive CLI interface for querying the RAG system.
    
    Args:
        collection_name: Name of the Milvus collection
    
    Usage:
        >>> interactive_query()
        # Then type questions and get answers
    """
    print("=" * 70)
    print("RAG QUERY INTERFACE")
    print("=" * 70)
    print("\nType your questions below. Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            # Get user input
            query = input("\nQuestion: ").strip()
            
            # Check for exit commands
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not query:
                print("Please enter a question.")
                continue
            
            # Process query
            result = query_rag(query, collection_name=collection_name, verbose=True)
            
            # Display formatted answer
            print(format_answer(result))
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Query the RAG system")
    parser.add_argument("--query", type=str, help="Question to ask (if not provided, starts interactive mode)")
    parser.add_argument("--collection", type=str, default="my_rag_collection", help="Milvus collection name")
    parser.add_argument("--top-k", type=int, default=200, help="Number of chunks to retrieve from Milvus")
    parser.add_argument("--top-n", type=int, default=15, help="Top N chunks for MMR")
    parser.add_argument("--final-k", type=int, default=6, help="Final number of diverse chunks")
    parser.add_argument("--lambda", type=float, default=0.7, dest="lambda_mmr", help="MMR lambda parameter")
    parser.add_argument("--no-embeddings", action="store_true", help="Don't include embeddings in search")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages")
    
    args = parser.parse_args()
    
    if args.query:
        # Single query mode
        result = query_rag(
            query=args.query,
            collection_name=args.collection,
            top_k_search=args.top_k,
            top_n_mmr=args.top_n,
            final_k_mmr=args.final_k,
            lambda_mmr=args.lambda_mmr,
            include_embeddings=not args.no_embeddings,
            verbose=not args.quiet
        )
        print(format_answer(result))
    else:
        # Interactive mode
        interactive_query(collection_name=args.collection)

