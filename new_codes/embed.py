import os
from typing import List, Dict, Optional
from openai import OpenAI
import time

# Import config to load API key from .env
try:
    import config
except ImportError:
    # Fallback if config.py doesn't exist
    from dotenv import load_dotenv
    load_dotenv()

# Initialize OpenAI client (will use OPENAI_API_KEY from environment)
openai_client = OpenAI()

# Embedding model configuration
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  # Dimension for text-embedding-3-small


def embed_text(text: str) -> List[float]:
    """
    Create an embedding for a single text string.
    
    Args:
        text: Input text to embed
        
    Returns:
        Embedding vector as a list of floats
    """
    try:
        resp = openai_client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        return resp.data[0].embedding
    except Exception as e:
        raise Exception(f"Error creating embedding: {str(e)}")


def batch_embed(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    """
    Create embeddings for multiple texts in batches.
    
    Args:
        texts: List of text strings to embed
        batch_size: Number of texts to process in each batch (default: 64)
        
    Returns:
        List of embedding vectors (each is a list of floats)
    """
    if not texts:
        return []
    
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        try:
            # Call OpenAI batch API: pass list into input parameter
            resp = openai_client.embeddings.create(
                input=batch,
                model=EMBEDDING_MODEL
            )
            
            # Extract embeddings from response
            batch_embeddings = [item.embedding for item in resp.data]
            embeddings.extend(batch_embeddings)
            
            # Optional: Add small delay to respect rate limits
            if i + batch_size < len(texts):
                time.sleep(0.1)  # 100ms delay between batches
                
        except Exception as e:
            # If batch fails, try individual embeddings
            print(f"Warning: Batch embedding failed, trying individual embeddings. Error: {str(e)}")
            for text in batch:
                try:
                    embedding = embed_text(text)
                    embeddings.append(embedding)
                except Exception as e2:
                    print(f"Error embedding text: {str(e2)}")
                    # Append zero vector as fallback
                    embeddings.append([0.0] * EMBEDDING_DIM)
    
    return embeddings


def embed_chunks(chunks: List[Dict], batch_size: int = 64) -> List[Dict]:
    """
    Add embeddings to chunks created from chunking.py.
    
    This function takes chunks (dictionaries with 'text' key) and adds
    an 'embedding' key to each chunk containing the embedding vector.
    
    Args:
        chunks: List of chunk dictionaries from chunking.py
                Each chunk should have at least a 'text' key
        batch_size: Number of chunks to embed in each batch (default: 64)
        
    Returns:
        List of chunk dictionaries with 'embedding' key added to each
    """
    if not chunks:
        return []
    
    # Extract text from chunks
    texts = [chunk.get("text", "") for chunk in chunks]
    
    # Create embeddings in batches
    print(f"Creating embeddings for {len(chunks)} chunks in batches of {batch_size}...")
    embeddings = batch_embed(texts, batch_size=batch_size)
    
    # Add embeddings to chunks
    embedded_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_copy = chunk.copy()
        chunk_copy["embedding"] = embeddings[i]
        embedded_chunks.append(chunk_copy)
    
    print(f"✓ Successfully created embeddings for {len(embedded_chunks)} chunks")
    
    return embedded_chunks


def embed_chunks_with_progress(chunks: List[Dict], batch_size: int = 64) -> List[Dict]:
    """
    Add embeddings to chunks with progress tracking.
    
    Args:
        chunks: List of chunk dictionaries from chunking.py
        batch_size: Number of chunks to embed in each batch
        
    Returns:
        List of chunk dictionaries with 'embedding' key added
    """
    if not chunks:
        return []
    
    from tqdm import tqdm
    
    texts = [chunk.get("text", "") for chunk in chunks]
    embeddings = []
    
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    print(f"Creating embeddings for {len(chunks)} chunks...")
    
    with tqdm(total=len(texts), desc="Embedding chunks") as pbar:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                resp = openai_client.embeddings.create(
                    input=batch,
                    model=EMBEDDING_MODEL
                )
                batch_embeddings = [item.embedding for item in resp.data]
                embeddings.extend(batch_embeddings)
                pbar.update(len(batch))
                
                if i + batch_size < len(texts):
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"\nWarning: Batch failed, processing individually. Error: {str(e)}")
                for text in batch:
                    try:
                        embedding = embed_text(text)
                        embeddings.append(embedding)
                        pbar.update(1)
                    except Exception as e2:
                        print(f"Error embedding text: {str(e2)}")
                        embeddings.append([0.0] * EMBEDDING_DIM)
                        pbar.update(1)
    
    # Add embeddings to chunks
    embedded_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_copy = chunk.copy()
        chunk_copy["embedding"] = embeddings[i]
        embedded_chunks.append(chunk_copy)
    
    return embedded_chunks

