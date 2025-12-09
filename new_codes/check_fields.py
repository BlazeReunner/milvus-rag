"""Diagnostic script to check what fields are returned from Milvus"""
from vectorstore import search
from embed import embed_text

print("=" * 70)
print("MILVUS FIELD DIAGNOSTIC")
print("=" * 70)

try:
    print("\nEmbedding test query...")
    query_vector = embed_text("test")
    print("✓ Query embedded")
    
    print("\nSearching Milvus collection...")
    results = search("my_rag_collection", query_vector, top_k=3, include_embeddings=False)
    
    if results:
        print(f"\n✓ Found {len(results)} results")
        print("\n" + "=" * 70)
        print("FIELD ANALYSIS")
        print("=" * 70)
        
        for i, (text, distance, metadata) in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Text preview: {text[:80]}...")
            print(f"Distance: {distance}")
            print(f"\nMetadata keys: {list(metadata.keys())}")
            print(f"\nMetadata values:")
            for key, value in metadata.items():
                if key == 'embedding':
                    print(f"  {key}: [vector of length {len(value) if isinstance(value, list) else 'N/A'}]")
                else:
                    print(f"  {key}: {value}")
            
            # Check for source and chunk_id specifically
            print(f"\nSource field: {metadata.get('source', 'MISSING')}")
            print(f"Chunk ID field: {metadata.get('chunk_id', 'MISSING')}")
            
            if 'document_metadata' in metadata:
                print(f"Document metadata: {metadata['document_metadata']}")
    else:
        print("\n✗ No results found!")
        print("Make sure:")
        print("  1. Milvus is running")
        print("  2. Collection 'my_rag_collection' exists")
        print("  3. Data has been ingested (run ingest.py)")

except Exception as e:
    print(f"\n✗ Error: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)

