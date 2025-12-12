"""
Query Paraphrasing Module
Generates multiple paraphrases of user queries to improve retrieval diversity.

Paraphrasing helps capture different ways the same question might be asked,
leading to better retrieval coverage and more comprehensive answers.
"""

from typing import List
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

# LLM model for paraphrasing (using GPT-4o-mini for cost efficiency)
PARAPHRASE_MODEL = "gpt-4o-mini"

# System prompt for query paraphrasing
PARAPHRASE_PROMPT = """You are a query paraphrasing assistant. Your task is to generate a diverse paraphrase of a given question.

Rules:
1. Generate exactly 1 paraphrase of the input question.
2. The paraphrase should:
   - Maintain the same meaning and intent as the original
   - Use different wording, phrasing, or structure
   - Be a complete, standalone question
   - Be clear and grammatically correct
3. Vary the paraphrase to use different phrasing:
   - Use synonyms where appropriate
   - Change question structure (e.g., "What is X?" vs "Can you explain X?")
   - Vary formality level if appropriate
   - Use different question words if possible
4. Return ONLY the 1 paraphrase, without numbering, bullets, or explanations.

Examples:

Input: "What is machine learning?"
Output:
Can you explain machine learning?

Input: "How many shares are selling?"
Output:
What is the number of shares being sold?

Now paraphrase this question:"""


def generate_paraphrases(query: str, num_paraphrases: int = 1, model: str = PARAPHRASE_MODEL) -> List[str]:
    """
    Generate paraphrases of a query using OpenAI.
    
    Args:
        query: The original user query
        num_paraphrases: Number of paraphrases to generate (default: 1)
        model: LLM model to use for paraphrasing (default: gpt-4o-mini)
    
    Returns:
        List of paraphrased queries. Always includes the original query first, then paraphrases.
        Total length will be num_paraphrases + 1 (original + paraphrases).
    
    Example:
        >>> generate_paraphrases("What is machine learning?")
        ['What is machine learning?', 'Can you explain machine learning?']
    """
    if not query or not query.strip():
        return [query]
    
    try:
        # Build the prompt
        messages = [
            {"role": "system", "content": PARAPHRASE_PROMPT},
            {"role": "user", "content": query}
        ]
        
        # Call OpenAI API
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,  # Higher temperature for more diverse paraphrases
            max_tokens=200
        )
        
        # Extract the response
        paraphrased_text = response.choices[0].message.content.strip()
        
        # Parse paraphrases
        paraphrases = []
        for line in paraphrased_text.split('\n'):
            line = line.strip()
            # Remove numbering/bullets if present
            if line:
                for prefix in ['1.', '2.', '3.', '4.', '5.', '-', '*', '•', '• ']:
                    if line.startswith(prefix):
                        line = line[len(prefix):].strip()
                if line:
                    paraphrases.append(line)
        
        # Limit to requested number
        paraphrases = paraphrases[:num_paraphrases]
        
        # If paraphrasing failed or returned fewer than requested, return original
        if not paraphrases:
            return [query]
        
        # Return original query first, then paraphrases
        return [query] + paraphrases
        
    except Exception as e:
        # If paraphrasing fails, return original query only
        print(f"Warning: Query paraphrasing failed: {str(e)}. Using original query only.")
        return [query]


def should_paraphrase(query: str) -> bool:
    """
    Quick heuristic to determine if a query might benefit from paraphrasing.
    
    Args:
        query: The user query
    
    Returns:
        True if query might benefit from paraphrasing, False otherwise
    """
    if not query:
        return False
    
    # Paraphrase all queries by default (can be made more selective)
    # Simple queries might not need paraphrasing, but it generally helps
    return True

