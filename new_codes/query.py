"""
RAG Query Interface
Complete search interface that connects all components of the RAG pipeline.

This is the main entry point for querying the RAG system:
1. Decompose complex queries into sub-queries (if needed)
2. For each sub-query:
   a. Embed user query
   b. Search Milvus (top 200 chunks)
   c. Rerank chunks using cross-encoder
   d. Apply MMR for diversity (top 15 → final 6)
   e. Generate answer using GPT-4o
3. Combine answers from all sub-queries
4. Return formatted answer with sources
"""

from typing import Dict, List, Optional, Tuple, Tuple
from embed import embed_text, batch_embed
from vectorstore import search
from reranker import rerank_from_search_results
from mmr import mmr_select
from answer import generate_answer
from subquery import decompose_query, should_decompose
from paraphrase import generate_paraphrases, should_paraphrase
import time


def merge_and_deduplicate_search_results(all_search_results: List[List[Tuple[str, float, Dict]]]) -> List[Tuple[str, float, Dict]]:
    """
    Merge multiple search result lists and deduplicate by chunk ID.
    
    Args:
        all_search_results: List of search result lists, each from a different query variant
    
    Returns:
        Merged and deduplicated list of search results, sorted by distance (best first)
    """
    # Dictionary to store best result for each chunk_id
    chunk_dict = {}
    
    # Process all search results
    for search_results in all_search_results:
        for text, distance, metadata in search_results:
            chunk_id = metadata.get('chunk_id', metadata.get('id', None))
            
            # Use chunk_id as key, or fallback to text hash if no chunk_id
            if chunk_id is None or chunk_id == 'N/A':
                # Fallback: use source + first 50 chars of text as key
                source = metadata.get('source', 'unknown')
                text_key = f"{source}_{text[:50]}"
                chunk_id = text_key
            
            # Keep the result with the best (lowest) distance for each chunk
            if chunk_id not in chunk_dict:
                chunk_dict[chunk_id] = (text, distance, metadata)
            else:
                # Compare distances - keep the one with better (lower) distance
                existing_distance = chunk_dict[chunk_id][1]
                if distance < existing_distance:
                    chunk_dict[chunk_id] = (text, distance, metadata)
    
    # Convert back to list and sort by distance (best first)
    merged_results = list(chunk_dict.values())
    merged_results.sort(key=lambda x: x[1])  # Sort by distance
    
    return merged_results


def _process_single_query(
    query: str,
    collection_name: str,
    top_k_search: int,
    top_n_mmr: int,
    final_k_mmr: int,
    lambda_mmr: float,
    include_embeddings: bool,
    verbose: bool = False,
    enable_paraphrasing: bool = True
) -> Dict:
    """
    Process a single query through the RAG pipeline with optional paraphrasing.
    Internal function used by query_rag() for processing individual queries.
    
    Args:
        query: Single query string
        collection_name: Name of the Milvus collection
        top_k_search: Number of chunks to retrieve from Milvus per query variant
        top_n_mmr: Top N chunks to consider for MMR
        final_k_mmr: Final number of diverse chunks to select
        lambda_mmr: MMR lambda parameter
        include_embeddings: Whether to include embeddings in search results
        verbose: Whether to print progress messages
        enable_paraphrasing: Whether to generate and use query paraphrases (default: True)
    
    Returns:
        Dictionary with answer, sources, chunks_used, and pipeline_stats
    """
    pipeline_stats = {}
    original_query = query
    
    # Step 0: Generate paraphrases (if enabled)
    query_variants = [query]  # Start with original
    paraphrase_time = 0
    
    if enable_paraphrasing and should_paraphrase(query):
        if verbose:
            print(f"[Paraphrasing] Generating query paraphrases...")
        step_start = time.time()
        try:
            query_variants = generate_paraphrases(query, num_paraphrases=1)
            paraphrase_time = time.time() - step_start
            if verbose:
                print(f"✓ Generated {len(query_variants) - 1} paraphrase:")
                for i, variant in enumerate(query_variants, 1):
                    marker = "(original)" if i == 1 else "(paraphrase)"
                    print(f"  {i}. {variant} {marker}")
        except Exception as e:
            if verbose:
                print(f"⚠ Paraphrasing failed: {str(e)}. Using original query only.")
            query_variants = [query]
    
    # Step 1: Embed all query variants
    step_start = time.time()
    try:
        # Embed all variants in batch for efficiency
        query_vectors = batch_embed(query_variants)
        pipeline_stats['embedding_time'] = time.time() - step_start
        pipeline_stats['paraphrase_time'] = paraphrase_time
        pipeline_stats['num_query_variants'] = len(query_variants)
        if verbose:
            print(f"✓ Embedded {len(query_vectors)} query variants")
    except Exception as e:
        return {
            'answer': f"Error embedding queries: {str(e)}",
            'sources': [],
            'chunks_used': 0,
            'query': original_query,
            'pipeline_stats': pipeline_stats,
            'error': str(e)
        }
    
    # Step 2: Search Milvus for each query variant
    step_start = time.time()
    try:
        all_search_results = []
        for i, (query_variant, query_vector) in enumerate(zip(query_variants, query_vectors)):
            if verbose and len(query_variants) > 1:
                print(f"  Searching with variant {i+1}/{len(query_variants)}: {query_variant[:50]}...")
            
            variant_results = search(
                collection_name=collection_name,
                query_vector=query_vector,
                top_k=top_k_search,
                include_embeddings=include_embeddings
            )
            all_search_results.append(variant_results)
        
        # Merge and deduplicate results
        search_results = merge_and_deduplicate_search_results(all_search_results)
        pipeline_stats['search_time'] = time.time() - step_start
        pipeline_stats['chunks_found'] = len(search_results)
        pipeline_stats['chunks_before_dedup'] = sum(len(r) for r in all_search_results)
        
        if verbose:
            print(f"✓ Found {pipeline_stats['chunks_before_dedup']} total chunks, "
                  f"{len(search_results)} unique chunks after deduplication")
    except Exception as e:
        return {
            'answer': f"Error searching Milvus: {str(e)}",
            'sources': [],
            'chunks_used': 0,
            'query': original_query,
            'pipeline_stats': pipeline_stats,
            'error': str(e)
        }
    
    if not search_results:
        return {
            'answer': "I don't know. No relevant documents were found in the knowledge base.",
            'sources': [],
            'chunks_used': 0,
            'query': original_query,
            'pipeline_stats': pipeline_stats
        }
    
    # Step 3: Rerank chunks (use original query for reranking)
    step_start = time.time()
    try:
        reranked_chunks = rerank_from_search_results(original_query, search_results, top_k=None)
        pipeline_stats['rerank_time'] = time.time() - step_start
        pipeline_stats['chunks_reranked'] = len(reranked_chunks)
    except Exception as e:
        return {
            'answer': f"Error reranking chunks: {str(e)}",
            'sources': [],
            'chunks_used': 0,
            'query': query,
            'pipeline_stats': pipeline_stats,
            'error': str(e)
        }
    
    # Step 4: Apply MMR for diversity (use original query)
    step_start = time.time()
    try:
        diverse_chunks = mmr_select(
            reranked_chunks=reranked_chunks,
            top_n=top_n_mmr,
            final_k=final_k_mmr,
            lambda_param=lambda_mmr,
            query=original_query
        )
        pipeline_stats['mmr_time'] = time.time() - step_start
        pipeline_stats['chunks_selected'] = len(diverse_chunks)
    except Exception as e:
        return {
            'answer': f"Error applying MMR: {str(e)}",
            'sources': [],
            'chunks_used': 0,
            'query': query,
            'pipeline_stats': pipeline_stats,
            'error': str(e)
        }
    
    # Step 5: Generate answer (use original query)
    step_start = time.time()
    try:
        answer_result = generate_answer(original_query, diverse_chunks)
        pipeline_stats['answer_time'] = time.time() - step_start
        return {
            'answer': answer_result['answer'],
            'sources': answer_result['sources'],
            'chunks_used': answer_result['chunks_used'],
            'query': original_query,
            'pipeline_stats': pipeline_stats,
            'raw_answer': answer_result
        }
    except Exception as e:
        return {
            'answer': f"Error generating answer: {str(e)}",
            'sources': [],
            'chunks_used': len(diverse_chunks),
            'query': original_query,
            'pipeline_stats': pipeline_stats,
            'error': str(e)
        }


def combine_subquery_answers(subquery_results: List[Dict], original_query: str) -> Dict:
    """
    Combine answers from multiple sub-queries into a single comprehensive answer.
    
    For each sub-query, checks if there's enough evidence to answer it.
    If yes → includes the answer
    If no → explicitly states "Information not found for this sub-question"
    
    This prevents one missing answer from causing the entire answer to fail.
    
    Args:
        subquery_results: List of result dictionaries from processing each sub-query
        original_query: The original user query
    
    Returns:
        Combined result dictionary
    """
    if not subquery_results:
        return {
            'answer': "I don't know. No answers were generated.",
            'sources': [],
            'chunks_used': 0,
            'query': original_query,
            'pipeline_stats': {}
        }
    
    # If only one sub-query, return it as-is
    if len(subquery_results) == 1:
        result = subquery_results[0].copy()
        result['query'] = original_query
        return result
    
    # Combine all answers with explicit evidence checking
    combined_answers = []
    all_sources = set()
    total_chunks = 0
    
    for i, result in enumerate(subquery_results, 1):
        sub_query = result.get('query', f'Sub-question {i}')
        answer = result.get('answer', '')
        chunks_used = result.get('chunks_used', 0)
        sources = result.get('sources', [])
        has_error = result.get('error') is not None
        
        # Check if we have enough evidence to answer this sub-question
        has_enough_evidence = (
            chunks_used > 0 and  # Found some chunks
            answer and  # Got an answer
            not has_error and  # No errors
            not answer.startswith("Error") and  # Answer doesn't start with "Error"
            not answer.lower().startswith("i don't know") and  # Answer doesn't say "I don't know"
            not answer.lower().startswith("the information is not available") and  # Answer doesn't say info not available
            "not available" not in answer.lower() and  # Answer doesn't contain "not available"
            len(sources) > 0  # Has at least one source
        )
        
        if has_enough_evidence:
            # We have enough evidence - include the answer
            combined_answers.append(f"**Sub-question {i}:** {sub_query}\n{answer}")
        else:
            # Not enough evidence - explicitly state it
            if has_error or answer.startswith("Error"):
                combined_answers.append(
                    f"**Sub-question {i}:** {sub_query}\n"
                    f"❌ Error processing this sub-question: {answer if answer else 'Unknown error'}"
                )
            elif chunks_used == 0:
                combined_answers.append(
                    f"**Sub-question {i}:** {sub_query}\n"
                    f"⚠️ Information not found for this sub-question. No relevant documents were found."
                )
            else:
                combined_answers.append(
                    f"**Sub-question {i}:** {sub_query}\n"
                    f"⚠️ Information not found for this sub-question. The answer is not available in the provided documents."
                )
        
        # Collect sources (even if answer wasn't found, sources might still be useful)
        all_sources.update(sources)
        
        # Sum chunks used
        total_chunks += chunks_used
    
    # Combine into final answer
    if combined_answers:
        final_answer = "\n\n".join(combined_answers)
    else:
        # Fallback if somehow no answers were generated
        final_answer = "I don't know. I couldn't find relevant information to answer your question."
    
    # Aggregate pipeline stats
    aggregated_stats = {}
    for result in subquery_results:
        stats = result.get('pipeline_stats', {})
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                aggregated_stats[key] = aggregated_stats.get(key, 0) + value
            else:
                aggregated_stats[key] = value  # Take last value for non-numeric
    
    return {
        'answer': final_answer,
        'sources': list(all_sources),
        'chunks_used': total_chunks,
        'query': original_query,
        'pipeline_stats': aggregated_stats,
        'subquery_results': subquery_results,
        'num_subqueries': len(subquery_results)
    }


def query_rag(
    query: str,
    collection_name: str = "my_rag_collection",
    top_k_search: int = 200,
    top_n_mmr: int = 15,
    final_k_mmr: int = 6,
    lambda_mmr: float = 0.7,
    include_embeddings: bool = True,
    verbose: bool = True,
    enable_decomposition: bool = True,
    enable_paraphrasing: bool = True
) -> Dict:
    """
    Complete RAG query pipeline with subquery decomposition and query paraphrasing support.
    
    Features:
    1. Query Paraphrasing: Generates 1 paraphrase, searches with all variants, merges results
    2. Subquery Decomposition: If query is complex, decomposes into sub-queries
    3. Each sub-query goes through: Paraphrase → Search → Merge → Rerank → MMR → Answer
    
    Args:
        query: User's question
        collection_name: Name of the Milvus collection
        top_k_search: Number of chunks to retrieve from Milvus per query variant (default: 200, total will be ~400 after merging)
        top_n_mmr: Top N chunks to consider for MMR (default: 15)
        final_k_mmr: Final number of diverse chunks to select (default: 6)
        lambda_mmr: MMR lambda parameter (default: 0.7)
        include_embeddings: Whether to include embeddings in search results (default: True)
        verbose: Whether to print progress messages (default: True)
        enable_decomposition: Whether to enable subquery decomposition (default: True)
        enable_paraphrasing: Whether to enable query paraphrasing (default: True)
    
    Returns:
        Dictionary containing:
            - 'answer': Generated answer text (combined if multiple sub-queries)
            - 'sources': List of source files used
            - 'chunks_used': Total number of chunks used for answer
            - 'query': Original query
            - 'pipeline_stats': Statistics about each step
            - 'num_subqueries': Number of sub-queries (if decomposition was used)
            - 'subquery_results': List of individual sub-query results (if decomposition was used)
    
    Example:
        >>> result = query_rag("What is machine learning?")
        >>> print(result['answer'])
        
        >>> result = query_rag("How many shares are selling AND how many will remain after offering?")
        >>> # This will be decomposed into 2 sub-queries and answers combined
    """
    start_time = time.time()
    
    if verbose:
        print("=" * 70)
        print("RAG QUERY PIPELINE")
        print("=" * 70)
        print(f"\nQuery: {query}\n")
    
    # Step 0: Decompose query if needed
    sub_queries = [query]  # Default: single query
    decomposition_time = 0
    
    if enable_decomposition and should_decompose(query):
        if verbose:
            print("[Step 0/6] Decomposing complex query...")
        step_start = time.time()
        try:
            sub_queries = decompose_query(query)
            decomposition_time = time.time() - step_start
            
            if len(sub_queries) > 1:
                if verbose:
                    print(f"✓ Decomposed into {len(sub_queries)} sub-queries:")
                    for i, sq in enumerate(sub_queries, 1):
                        print(f"  {i}. {sq}")
            else:
                if verbose:
                    print("✓ Query is simple, no decomposition needed")
        except Exception as e:
            if verbose:
                print(f"⚠ Decomposition failed: {str(e)}. Using original query.")
            sub_queries = [query]
    elif verbose and enable_decomposition:
        print("[Step 0/6] Query appears simple, skipping decomposition")
    
    # Process each sub-query
    subquery_results = []
    total_embedding_time = 0
    total_search_time = 0
    total_rerank_time = 0
    total_mmr_time = 0
    total_answer_time = 0
    
    for i, sub_query in enumerate(sub_queries, 1):
        if verbose and len(sub_queries) > 1:
            print(f"\n{'='*70}")
            print(f"Processing sub-query {i}/{len(sub_queries)}: {sub_query}")
            print(f"{'='*70}\n")
        
        # Process single query (with paraphrasing)
        step_start = time.time()
        result = _process_single_query(
            query=sub_query,
            collection_name=collection_name,
            top_k_search=top_k_search,
            top_n_mmr=top_n_mmr,
            final_k_mmr=final_k_mmr,
            lambda_mmr=lambda_mmr,
            include_embeddings=include_embeddings,
            verbose=verbose and len(sub_queries) == 1,  # Only verbose for single query
            enable_paraphrasing=enable_paraphrasing
        )
        subquery_results.append(result)
        
        # Aggregate timing (if available in stats)
        stats = result.get('pipeline_stats', {})
        total_embedding_time += stats.get('embedding_time', 0)
        total_search_time += stats.get('search_time', 0)
        total_rerank_time += stats.get('rerank_time', 0)
        total_mmr_time += stats.get('mmr_time', 0)
        total_answer_time += stats.get('answer_time', 0)
    
    # Combine results if multiple sub-queries
    if len(sub_queries) > 1:
        if verbose:
            print(f"\n[Step 6/6] Combining answers from {len(sub_queries)} sub-queries...")
        final_result = combine_subquery_answers(subquery_results, query)
    else:
        final_result = subquery_results[0].copy()
        final_result['query'] = query
    
    # Update pipeline stats
    final_result['pipeline_stats'] = {
        'decomposition_time': decomposition_time,
        'embedding_time': total_embedding_time,
        'search_time': total_search_time,
        'rerank_time': total_rerank_time,
        'mmr_time': total_mmr_time,
        'answer_time': total_answer_time,
        'total_time': time.time() - start_time,
        'num_subqueries': len(sub_queries)
    }
    
    if verbose:
        print("\n" + "=" * 70)
        print("PIPELINE STATISTICS")
        print("=" * 70)
        if decomposition_time > 0:
            print(f"Decomposition time: {decomposition_time:.2f}s")
        print(f"Embedding time: {total_embedding_time:.2f}s")
        print(f"Search time: {total_search_time:.2f}s")
        print(f"Rerank time: {total_rerank_time:.2f}s")
        print(f"MMR time: {total_mmr_time:.2f}s")
        print(f"Answer generation time: {total_answer_time:.2f}s")
        print(f"Total time: {final_result['pipeline_stats']['total_time']:.2f}s")
        if len(sub_queries) > 1:
            print(f"Number of sub-queries: {len(sub_queries)}")
        print("=" * 70)
    
    return final_result


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

