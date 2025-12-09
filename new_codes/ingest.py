"""
Ingest Orchestration Script
Orchestrates the complete RAG pipeline: load → chunk → embed → insert into Milvus

Run this once at the start and whenever you update documents.
"""

from load_files import load_all_docs, load_file
from chunking import chunk_documents, chunk_text
from embed import embed_chunks_with_progress, EMBEDDING_DIM
from vectorstore import create_collection, insert_embedded_chunks
from tqdm import tqdm
from pathlib import Path
import time


def ingest_documents(
    data_folder: str = "Data/",
    collection_name: str = "my_rag_collection",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    embedding_batch_size: int = 64,
    milvus_batch_size: int = 1000,
    create_index: bool = True,
    recursive: bool = False
):
    """
    Complete ingestion pipeline: Load → Chunk → Embed → Insert into Milvus
    
    Args:
        data_folder: Path to folder containing documents
        collection_name: Name of the Milvus collection
        chunk_size: Maximum tokens per chunk
        chunk_overlap: Number of tokens to overlap between chunks
        embedding_batch_size: Batch size for OpenAI embeddings (default: 64)
        milvus_batch_size: Batch size for Milvus insertion (default: 1000)
        create_index: Whether to create index and load collection after insertion
        recursive: Whether to search subdirectories for files
    """
    print("=" * 70)
    print("RAG INGESTION PIPELINE")
    print("=" * 70)
    
    # Step 1: Load documents
    print("\n[Step 1/4] Loading documents...")
    print("-" * 70)
    try:
        print(f"Loading documents from '{data_folder}'...")
        
        # Get list of files first
        folder_path = Path(data_folder)
        if not folder_path.exists():
            print(f"✗ Folder does not exist: {data_folder}")
            return
        
        supported_extensions = {".pdf", ".docx", ".txt", ".md"}
        if recursive:
            files = [f for f in folder_path.rglob("*") if f.is_file() and f.suffix.lower() in supported_extensions]
        else:
            files = [f for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() in supported_extensions]
        
        # Load files with progress bar
        docs = []
        for file_path in tqdm(files, desc="Loading files"):
            result = load_file(file_path)
            if result:
                docs.append(result)
        
        if not docs:
            print(f"✗ No documents found in {data_folder}")
            return
        
        print(f"✓ Successfully loaded {len(docs)} documents")
        
        # Show summary
        total_chars = sum(doc.get("metadata", {}).get("text_length", 0) for doc in docs)
        total_words = sum(doc.get("metadata", {}).get("word_count", 0) for doc in docs)
        print(f"  Total: {total_chars:,} characters, {total_words:,} words")
        
    except Exception as e:
        print(f"✗ Error loading documents: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Chunk documents
    print("\n[Step 2/4] Chunking documents...")
    print("-" * 70)
    try:
        print(f"Chunking with size={chunk_size}, overlap={chunk_overlap}...")
        
        chunks = []
        for doc in tqdm(docs, desc="Chunking documents"):
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            source = metadata.get("source", metadata.get("file_path", "unknown"))
            
            doc_chunks = chunk_text(
                text=text,
                model="gpt-4o-mini",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                source=source
            )
            
            # Add document-level metadata to each chunk
            for chunk in doc_chunks:
                chunk["document_metadata"] = metadata
            
            chunks.extend(doc_chunks)
        
        if not chunks:
            print("✗ No chunks created!")
            return
        
        print(f"✓ Created {len(chunks)} chunks")
        
        # Show statistics
        avg_tokens = sum(chunk.get("token_count", 0) for chunk in chunks) / len(chunks) if chunks else 0
        print(f"  Average tokens per chunk: {avg_tokens:.1f}")
        
    except Exception as e:
        print(f"✗ Error chunking documents: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Embed chunks (with rate limiting)
    print("\n[Step 3/4] Embedding chunks...")
    print("-" * 70)
    print(f"Using batch size: {embedding_batch_size}")
    print("Note: Respecting OpenAI rate limits with delays between batches")
    
    try:
        embedded_chunks = embed_chunks_with_progress(
            chunks,
            batch_size=embedding_batch_size
        )
        
        if not embedded_chunks:
            print("✗ No embedded chunks created!")
            return
        
        # Verify embeddings
        chunks_with_embeddings = sum(1 for chunk in embedded_chunks if "embedding" in chunk)
        
        if chunks_with_embeddings == 0:
            print("✗ Warning: No embeddings found in chunks!")
            return
        
        print(f"✓ Successfully embedded {chunks_with_embeddings}/{len(embedded_chunks)} chunks")
        
        if embedded_chunks and "embedding" in embedded_chunks[0]:
            emb_dim = len(embedded_chunks[0]["embedding"])
            print(f"  Embedding dimension: {emb_dim}")
        
    except Exception as e:
        print(f"✗ Error embedding chunks: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Insert into Milvus
    print("\n[Step 4/4] Inserting into Milvus...")
    print("-" * 70)
    try:
        # Create collection
        print(f"Creating collection '{collection_name}'...")
        create_collection(
            collection_name=collection_name,
            dim=EMBEDDING_DIM,
            metric="IP",
            consistency_level="Bounded"
        )
        
        # Insert embedded chunks
        print(f"Inserting {len(embedded_chunks)} chunks into Milvus...")
        insert_embedded_chunks(
            collection_name=collection_name,
            embedded_chunks=embedded_chunks,
            batch_size=milvus_batch_size,
            create_index=create_index
        )
        
        print(f"✓ Successfully inserted all chunks into Milvus collection '{collection_name}'")
        
    except Exception as e:
        print(f"✗ Error inserting into Milvus: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Final Summary
    print("\n" + "=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)
    print(f"✓ Documents loaded: {len(docs)}")
    print(f"✓ Chunks created: {len(chunks)}")
    print(f"✓ Chunks embedded: {len(embedded_chunks)}")
    print(f"✓ Chunks inserted: {len(embedded_chunks)}")
    print(f"✓ Collection: {collection_name}")
    print(f"✓ Embedding dimension: {EMBEDDING_DIM}")
    print("\n✓ Ingestion completed successfully!")
    print("=" * 70)
    
    return {
        "docs_loaded": len(docs),
        "chunks_created": len(chunks),
        "chunks_embedded": len(embedded_chunks),
        "collection_name": collection_name
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest documents into Milvus RAG system")
    parser.add_argument("--data-folder", type=str, default="Data/", help="Path to data folder")
    parser.add_argument("--collection", type=str, default="my_rag_collection", help="Milvus collection name")
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size in tokens")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Chunk overlap in tokens")
    parser.add_argument("--embed-batch-size", type=int, default=64, help="Batch size for embeddings")
    parser.add_argument("--milvus-batch-size", type=int, default=1000, help="Batch size for Milvus insertion")
    parser.add_argument("--no-index", action="store_true", help="Skip index creation")
    parser.add_argument("--recursive", action="store_true", help="Search subdirectories recursively")
    
    args = parser.parse_args()
    
    ingest_documents(
        data_folder=args.data_folder,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_batch_size=args.embed_batch_size,
        milvus_batch_size=args.milvus_batch_size,
        create_index=not args.no_index,
        recursive=args.recursive
    )

