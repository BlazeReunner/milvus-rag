"""
Subquery Decomposition Module
Splits complex user questions into smaller, simpler sub-questions.

This helps when a question contains multiple parts that might require
different retrieval strategies or appear in different parts of documents.
"""

from typing import List, Dict
from openai import OpenAI

# Import config to load API key
try:
    import config
except ImportError:
    from dotenv import load_dotenv
    import os
    load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Initialize OpenAI client
openai_client = OpenAI()

# LLM model for decomposition (using GPT-4o-mini for cost efficiency)
DECOMPOSITION_MODEL = "gpt-4o-mini"

# System prompt for query decomposition
DECOMPOSITION_PROMPT = """You are a query decomposition assistant. Your task is to break down complex questions into simpler sub-questions.

Rules:
1. If the question is simple and cannot be meaningfully split, return it as-is.
2. If the question contains multiple parts connected by words like "AND", "OR", "also", "and", "plus", etc., split it into separate sub-questions.
3. Each sub-question should be complete and answerable independently.
4. Remove connecting words like "AND", "OR", "also" from individual sub-questions.
5. Preserve the original meaning of each part.
6. Return ONLY the sub-questions, one per line, without numbering or bullets.

Examples:

Input: "How many shares are selling AND how many will remain after offering?"
Output:
How many shares are selling?
How many will remain after offering?

Input: "What is the revenue for Q1 and what are the main expenses?"
Output:
What is the revenue for Q1?
What are the main expenses?

Input: "What is machine learning?"
Output:
What is machine learning?

Input: "Explain neural networks and how they are trained"
Output:
Explain neural networks
How are neural networks trained?

Now decompose this question:"""


def decompose_query(query: str, model: str = DECOMPOSITION_MODEL) -> List[str]:
    """
    Decompose a complex query into simpler sub-queries.
    
    Args:
        query: The original user query
        model: LLM model to use for decomposition (default: gpt-4o-mini)
    
    Returns:
        List of sub-queries. If the query is simple, returns a list with just the original query.
    
    Example:
        >>> decompose_query("How many shares are selling AND how many will remain after offering?")
        ['How many shares are selling?', 'How many will remain after offering?']
        
        >>> decompose_query("What is machine learning?")
        ['What is machine learning?']
    """
    if not query or not query.strip():
        return [query]
    
    try:
        # Build the prompt
        messages = [
            {"role": "system", "content": DECOMPOSITION_PROMPT},
            {"role": "user", "content": query}
        ]
        
        # Call OpenAI API
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,  # Low temperature for consistent decomposition
            max_tokens=200
        )
        
        # Extract the response
        decomposed_text = response.choices[0].message.content.strip()
        
        # Split by newlines and clean up
        sub_queries = []
        for line in decomposed_text.split('\n'):
            line = line.strip()
            # Remove numbering/bullets if present (e.g., "1. ", "- ", etc.)
            if line:
                # Remove common prefixes
                for prefix in ['1.', '2.', '3.', '4.', '5.', '-', '*', '•']:
                    if line.startswith(prefix):
                        line = line[len(prefix):].strip()
                if line:
                    sub_queries.append(line)
        
        # If decomposition failed or returned empty, return original query
        if not sub_queries:
            return [query]
        
        # If only one sub-query and it's very similar to original, return original
        if len(sub_queries) == 1 and sub_queries[0].lower() == query.lower():
            return [query]
        
        return sub_queries
        
    except Exception as e:
        # If decomposition fails, return original query
        print(f"Warning: Query decomposition failed: {str(e)}. Using original query.")
        return [query]


def should_decompose(query: str) -> bool:
    """
    Quick heuristic to determine if a query might benefit from decomposition.
    
    Args:
        query: The user query
    
    Returns:
        True if query might benefit from decomposition, False otherwise
    """
    if not query:
        return False
    
    query_lower = query.lower()
    
    # Check for common connecting words that indicate multiple parts
    connecting_words = [' and ', ' or ', ' also ', ' plus ', ' as well as ', 
                       ' furthermore ', ' additionally ', ' moreover ',
                       ' both ', ' either ', ' neither ']
    
    # Check for multiple question marks (might indicate multiple questions)
    if query.count('?') > 1:
        return True
    
    # Check for connecting words
    for word in connecting_words:
        if word in query_lower:
            return True
    
    # Check for multiple clauses (rough heuristic)
    if query.count(',') >= 2:
        return True
    
    return False

