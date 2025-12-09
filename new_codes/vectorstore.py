"""
Milvus Vectorstore Helper Functions
Centralized Milvus create/insert/search logic and metadata mapping
"""

from pymilvus import MilvusClient
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm


# Initialize Milvus client (Zilliz Cloud)
milvus_client = MilvusClient(
    uri="https://in03-f90c9f3647d31d4.serverless.aws-eu-central-1.cloud.zilliz.com",
    token="2aeed5dcd363aeb5cee132e1719ac27d77e527b09db66417f463cc255c0fbf64011e31f6b69f43a3f707403a783b0f4ed4137d50"
)


def create_collection(collection_name: str, dim: int, metric: str = "IP", consistency_level: str = "Bounded"):
    """
    Create a new Milvus collection or replace existing one.
    
    Args:
        collection_name: Name of the collection
        dim: Dimension of the vectors
        metric: Distance metric ("IP", "L2", or "COSINE")
        consistency_level: Consistency level ("Strong", "Session", "Bounded", "Eventually")
    """
    if milvus_client.has_collection(collection_name):
        print(f"Collection '{collection_name}' already exists. Dropping it...")
        milvus_client.drop_collection(collection_name)
    
    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=dim,
        metric_type=metric,
        consistency_level=consistency_level
    )
    print(f"✓ Created collection '{collection_name}' with dimension {dim}, metric {metric}")


def insert_vectors(collection_name: str, items: List[Dict], batch_size: int = 1000, create_index: bool = True):
    """
    Insert vectors into Milvus collection.
    
    Args:
        collection_name: Name of the collection
        items: List of dictionaries, each containing:
            - id: Unique identifier (str or int)
            - vector: Embedding vector (list of floats)
            - text: Text content of the chunk
            - metadata: Optional dictionary with additional metadata
        batch_size: Number of records to insert per batch
        create_index: Whether to create index and load collection after insertion
    
    Example items format:
        [
            {
                "id": "uuid-123",
                "vector": [0.1, 0.2, ...],
                "text": "chunk text here",
                "metadata": {"source": "file.pdf", "chunk_id": 0}
            },
            ...
        ]
    """
    if not items:
        print("No items to insert.")
        return
    
    print(f"Inserting {len(items)} vectors into collection '{collection_name}'...")
    
    # Prepare records for Milvus
    # MilvusClient expects: id (int64), vector, and other fields as dynamic fields
    records = []
    
    for idx, item in enumerate(tqdm(items, desc="Preparing records")):
        # Convert ID to integer if it's a string (MilvusClient requires int64 for id field)
        item_id = item["id"]
        if isinstance(item_id, str):
            # Hash string ID to integer (using built-in hash, which is consistent within a session)
            item_id = abs(hash(item_id)) % (2**63)  # Ensure it fits in int64 range
        
        record = {
            "id": int(item_id),  # Ensure it's an integer
            "vector": item["vector"],
            "text": item.get("text", ""),
        }
        
        # Add metadata fields as separate fields (MilvusClient supports dynamic fields)
        metadata = item.get("metadata", {})
        if isinstance(metadata, dict):
            # Flatten metadata into the record
            for key, value in metadata.items():
                # Convert complex types to strings for storage
                if isinstance(value, (dict, list)):
                    record[key] = str(value)
                else:
                    record[key] = value
        
        # Also add common chunk fields if they exist
        if "source" in item:
            record["source"] = item["source"]
        if "chunk_id" in item:
            record["chunk_id"] = item["chunk_id"]
        if "start_token" in item:
            record["start_token"] = item["start_token"]
        if "end_token" in item:
            record["end_token"] = item["end_token"]
        if "token_count" in item:
            record["token_count"] = item["token_count"]
        
        records.append(record)
    
    # Insert in batches
    total_inserted = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            result = milvus_client.insert(collection_name=collection_name, data=batch)
            total_inserted += result.get("insert_count", len(batch))
        except Exception as e:
            print(f"Error inserting batch {i//batch_size + 1}: {str(e)}")
            raise
    
    print(f"✓ Successfully inserted {total_inserted} vectors")
    
    # Create index and load collection if requested
    if create_index:
        try:
            print("Creating index...")
            milvus_client.create_index(
                collection_name=collection_name,
                index={
                    "index_type": "HNSW",
                    "params": {"M": 16, "efConstruction": 200}
                }
            )
            print("Loading collection...")
            milvus_client.load_collection(collection_name)
            print("✓ Index created and collection loaded")
        except Exception as e:
            print(f"Warning: Could not create index/load collection: {str(e)}")
            print("Collection may still be usable, but search performance may be reduced")


def search(collection_name: str, query_vector: List[float], top_k: int = 200, output_fields: Optional[List[str]] = None, include_embeddings: bool = False) -> List[Tuple[str, float, Dict]]:
    """
    Search for similar vectors in Milvus collection.
    
    Args:
        collection_name: Name of the collection
        query_vector: Query embedding vector
        top_k: Number of results to return (default: 200 for reranking workflow)
        output_fields: Optional list of fields to return (default: ["text", "source", "chunk_id", "token_count"])
        include_embeddings: If True, includes embedding vectors in results (default: False)
                           This avoids recomputing embeddings in downstream steps like MMR.
    
    Returns:
        List of tuples: [(text, distance, metadata_dict), ...]
        Results are sorted by distance (lower is better for IP/L2, higher is better for COSINE)
        If include_embeddings=True, metadata_dict will contain 'embedding' field with the vector.
    """
    if output_fields is None:
        output_fields = ["text", "source", "chunk_id", "token_count"]
    
    # Add vector to output_fields if embeddings are requested
    search_output_fields = output_fields.copy()
    if include_embeddings and "vector" not in search_output_fields:
        search_output_fields.append("vector")
    
    try:
        results = milvus_client.search(
            collection_name=collection_name,
            data=[query_vector],
            limit=top_k,
            output_fields=search_output_fields
        )
        
        # Process results
        processed = []
        for hit in results[0]:
            # Extract text
            text = hit.entity.get("text", "")
            
            # Extract distance
            distance = float(hit.distance)
            
            # Extract metadata (all fields except vector and id)
            metadata = {}
            for field in output_fields:
                if field in hit.entity:
                    metadata[field] = hit.entity[field]
            
            # Fallback: Ensure source and chunk_id are always present
            # Try to get from entity directly if not already extracted
            if 'source' not in metadata:
                metadata['source'] = hit.entity.get('source', 'unknown')
            
            if 'chunk_id' not in metadata:
                metadata['chunk_id'] = hit.entity.get('chunk_id', 'N/A')
            
            # Add embedding vector if requested
            if include_embeddings and "vector" in hit.entity:
                embedding = hit.entity.get("vector")
                if embedding is not None:
                    # Convert to list if it's not already
                    if isinstance(embedding, (list, tuple)):
                        metadata["embedding"] = list(embedding)
                    else:
                        metadata["embedding"] = embedding
            
            # Add entity ID to metadata
            metadata["id"] = hit.id
            
            processed.append((text, distance, metadata))
        
        return processed
    
    except Exception as e:
        print(f"Error searching collection: {str(e)}")
        raise


def insert_embedded_chunks(collection_name: str, embedded_chunks: List[Dict], batch_size: int = 1000, create_index: bool = True):
    """
    Convenience function to insert embedded chunks from the workflow.
    
    This function adapts the chunk format from chunking.py + embed.py to Milvus format.
    
    Args:
        collection_name: Name of the collection
        embedded_chunks: List of chunks with embeddings (from embed_chunks function)
        batch_size: Number of records to insert per batch
        create_index: Whether to create index and load collection
    """
    # Transform embedded chunks to Milvus format
    items = []
    
    for idx, chunk in enumerate(embedded_chunks):
        # Use chunk ID or generate numeric ID
        chunk_id = chunk.get("id")
        if isinstance(chunk_id, str):
            # Convert UUID string to integer hash
            numeric_id = abs(hash(chunk_id)) % (2**63)
        else:
            # Use chunk_id or index as fallback
            numeric_id = chunk.get("chunk_id", idx)
        
        item = {
            "id": numeric_id,
            "vector": chunk.get("embedding", []),
            "text": chunk.get("text", ""),
            "source": chunk.get("source", "unknown"),
            "chunk_id": chunk.get("chunk_id", idx),
            "metadata": {
                "source": chunk.get("source", "unknown"),
                "chunk_id": chunk.get("chunk_id", idx),
                "start_token": chunk.get("start_token", 0),
                "end_token": chunk.get("end_token", 0),
                "start_word": chunk.get("start_word", 0),
                "end_word": chunk.get("end_word", 0),
                "token_count": chunk.get("token_count", 0),
            }
        }
        
        # Add document metadata if available
        if "document_metadata" in chunk:
            doc_meta = chunk["document_metadata"]
            item["metadata"].update({
                "file_path": doc_meta.get("file_path", ""),
                "file_size": doc_meta.get("file_size", 0),
                "file_extension": doc_meta.get("file_extension", ""),
            })
        
        items.append(item)
    
    # Insert using the main insert function
    insert_vectors(collection_name, items, batch_size=batch_size, create_index=create_index)


def get_collection_info(collection_name: str) -> Dict:
    """
    Get information about a collection.
    
    Args:
        collection_name: Name of the collection
    
    Returns:
        Dictionary with collection information
    """
    if not milvus_client.has_collection(collection_name):
        return {"exists": False}
    
    try:
        # Get collection stats
        stats = milvus_client.get_collection_stats(collection_name)
        return {
            "exists": True,
            "stats": stats
        }
    except Exception as e:
        return {
            "exists": True,
            "error": str(e)
        }

