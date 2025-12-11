# Query Paraphrasing Implementation Summary

## Overview

We've implemented **query paraphrasing** to improve retrieval diversity and coverage in our RAG system. This enhancement generates multiple query variants, performs parallel searches, and merges results for more comprehensive answers.

---

## What Changed

### New File: `paraphrase.py`
- **`generate_paraphrases()`**: Uses GPT-4o-mini to generate 3 paraphrases of user queries
- **`should_paraphrase()`**: Determines if a query should be paraphrased (currently returns True for all queries)

### Updated File: `query.py`
- **`merge_and_deduplicate_search_results()`**: Merges results from multiple searches and deduplicates by chunk ID
- **`_process_single_query()`**: Updated to:
  1. Generate paraphrases (if enabled)
  2. Embed all query variants
  3. Perform multiple vector searches
  4. Merge and deduplicate results
  5. Continue with reranking → MMR → Answer generation
- **`query_rag()`**: Added `enable_paraphrasing` parameter (default: True)

---

## How the New System Works

### Complete Pipeline Flow

```
User Query
    ↓
[Optional] Subquery Decomposition (if complex query)
    ↓
For each sub-query:
    ↓
    [Step 0] Generate 3 Paraphrases
        Original: "What is machine learning?"
        Paraphrase 1: "Can you explain machine learning?"
        Paraphrase 2: "How would you define machine learning?"
        Paraphrase 3: "Tell me about machine learning."
    ↓
    [Step 1] Embed All Variants (batch embedding)
        → 4 query vectors (original + 3 paraphrases)
    ↓
    [Step 2] Perform Multiple Vector Searches
        → Search 1: Original query → Top 200 chunks
        → Search 2: Paraphrase 1 → Top 200 chunks
        → Search 3: Paraphrase 2 → Top 200 chunks
        → Search 4: Paraphrase 3 → Top 200 chunks
    ↓
    [Step 3] Merge & Deduplicate Results
        → Combine all search results
        → Remove duplicates by chunk_id
        → Keep best distance score for each unique chunk
        → Result: ~400-800 unique chunks (depending on overlap)
    ↓
    [Step 4] Rerank (using original query)
        → Cross-encoder reranks all merged chunks
    ↓
    [Step 5] MMR Selection
        → Select top 15 → final 6 diverse chunks
    ↓
    [Step 6] Generate Answer
        → GPT-4o generates answer from 6 chunks
    ↓
[If multiple sub-queries] Combine Answers
    ↓
Final Answer with Sources
```

---

## Why This Change?

### Problem Solved

**Before Paraphrasing:**
- Single query → Single search → Limited retrieval coverage
- If the query phrasing doesn't match document phrasing, relevant chunks might be missed
- Example: Query "What is ML?" might miss chunks that say "machine learning" but not "ML"

**After Paraphrasing:**
- Multiple query variants → Multiple searches → Broader retrieval coverage
- Different phrasings capture different document sections
- Merged results ensure comprehensive coverage
- Deduplication prevents redundant processing

### Benefits

1. **Better Retrieval Coverage**
   - Different query phrasings retrieve different chunks
   - Merged results capture more relevant information
   - Reduces the chance of missing important content

2. **Improved Answer Quality**
   - More comprehensive context for answer generation
   - Better handling of synonym variations
   - Captures information phrased differently in documents

3. **Robustness**
   - Less sensitive to exact query wording
   - Works better with documents that use varied terminology
   - Handles different question styles (formal vs. casual)

4. **Deduplication**
   - Prevents processing the same chunk multiple times
   - Keeps best distance score for each unique chunk
   - Efficient merging without redundancy

---

## Technical Details

### Paraphrase Generation

- **Model**: GPT-4o-mini (cost-efficient)
- **Temperature**: 0.7 (for diversity)
- **Output**: Exactly 3 paraphrases per query
- **Fallback**: If paraphrasing fails, uses original query only

### Search Strategy

- **Per Variant**: Top 200 chunks per query variant
- **Total Before Dedup**: ~800 chunks (4 variants × 200)
- **After Dedup**: Typically 400-600 unique chunks
- **Deduplication Key**: `chunk_id` (or source + text hash as fallback)

### Merging Logic

```python
# For each chunk found:
1. Extract chunk_id (or generate key from source + text)
2. If chunk_id already seen:
   - Keep the result with better (lower) distance score
3. If chunk_id not seen:
   - Add to merged results
4. Sort merged results by distance (best first)
```

---

## Performance Considerations

### Additional Costs

- **Paraphrasing**: ~1 API call per query (GPT-4o-mini, ~$0.0001)
- **Embeddings**: 3 additional embeddings per query (~$0.0003)
- **Search**: 3 additional searches (no extra cost, just more results)

### Additional Time

- **Paraphrasing**: ~0.5-1 second
- **Extra Embeddings**: ~0.2 seconds (batch processing)
- **Extra Searches**: ~0.5-1 second (parallelizable)
- **Deduplication**: ~0.1 seconds

**Total Overhead**: ~1-2 seconds per query

### Benefits vs. Costs

- **Cost**: Minimal (~$0.0004 per query)
- **Time**: ~1-2 seconds overhead
- **Quality**: Significant improvement in retrieval coverage
- **ROI**: High - small cost for substantial quality improvement

---

## Usage

### Default Behavior (Paraphrasing Enabled)

```python
result = query_rag("What is machine learning?")
# Automatically generates paraphrases and uses them
```

### Disable Paraphrasing

```python
result = query_rag(
    "What is machine learning?",
    enable_paraphrasing=False
)
# Uses only original query
```

### Combined with Subquery Decomposition

```python
result = query_rag(
    "How many shares are selling AND how many will remain?",
    enable_paraphrasing=True,  # Each sub-query gets paraphrased
    enable_decomposition=True  # Query is decomposed first
)
```

---

## Example Output

### Query: "What is machine learning?"

**Paraphrases Generated:**
1. "What is machine learning?" (original)
2. "Can you explain machine learning?"
3. "How would you define machine learning?"
4. "Tell me about machine learning."

**Search Results:**
- Original query: 200 chunks found
- Paraphrase 1: 200 chunks found
- Paraphrase 2: 200 chunks found
- Paraphrase 3: 200 chunks found
- **After deduplication**: 487 unique chunks

**Pipeline Continues:**
- Rerank → Top 15
- MMR → Final 6 diverse chunks
- Answer generation → Comprehensive answer

---

## Statistics Tracked

The pipeline now tracks:
- `paraphrase_time`: Time to generate paraphrases
- `num_query_variants`: Number of query variants used (1-4)
- `chunks_before_dedup`: Total chunks before deduplication
- `chunks_found`: Unique chunks after deduplication

---

## Future Enhancements

Potential improvements:
1. **Selective Paraphrasing**: Only paraphrase complex queries
2. **Query Expansion**: Add related terms/concepts
3. **Hybrid Search**: Combine keyword and vector search
4. **Adaptive Paraphrasing**: Adjust number of paraphrases based on query complexity

---

## Summary

Query paraphrasing significantly improves retrieval coverage by:
- Generating multiple query variants
- Performing parallel searches
- Merging and deduplicating results
- Providing more comprehensive context for answer generation

This enhancement works seamlessly with existing features (subquery decomposition, reranking, MMR) and provides substantial quality improvements with minimal cost and time overhead.

