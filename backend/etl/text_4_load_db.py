
import os
import json
import logging
import argparse

# --- Configuration ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR) # backend
INPUT_FILE = os.path.join(BACKEND_DIR, "data", "text_embeddings.json")
# Target the NEW text-specific database
QDRANT_PATH = os.path.join(BACKEND_DIR, "db", "text_db") 
COLLECTION_NAME = "hr_text_hybrid"

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SparseVectorParams

def main():
    print(f"--- Step 3: Upserting to Database ---")
    print(f"Reading from: {INPUT_FILE}")
    print(f"Database: {QDRANT_PATH}")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Run step2_embed_chunks.py first!")
        return

    # Load Embeddings
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        embedded_data = json.load(f)
    print(f"Found {len(embedded_data)} items to upsert.")
    
    # Initialize Qdrant
    # Force Remove Lock File if it exists (Fix for PermissionError)
    lock_file = os.path.join(QDRANT_PATH, ".lock")
    if os.path.exists(lock_file):
        try:
            os.remove(lock_file)
            print(f"Forcefully removed stale DB lock file: {lock_file}")
        except Exception as e:
            print(f"Warning: Could not remove lock file: {e}")

    client = QdrantClient(path=QDRANT_PATH)
    
    # 1. Create Collection
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        print(f"Cleaned existing collection: {COLLECTION_NAME}")
    
    vectors_config = VectorParams(size=1024, distance=Distance.COSINE)
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=vectors_config,
    )
    print(f"Created Collection: {COLLECTION_NAME}")
    
    # 2. Convert to Points
    points = []
    
    for i, item in enumerate(embedded_data):
        point_id = i # Simple integer ID
        
        # Payload
        payload = item["metadata"]
        payload["text"] = item["text"]
        payload["chunk_id"] = item["id"]
        
        # Dense Vector Only
        dense_vec = item["vector"]["dense"]
        
        points.append(PointStruct(
            id=point_id,
            vector=dense_vec,
            payload=payload
        ))
        
    print(f"Generated {len(points)} points.")
    
    # 3. Upsert Batch
    BATCH_SIZE = 100
    total_points = len(points)
    
    for i in range(0, total_points, BATCH_SIZE):
        batch = points[i : i + BATCH_SIZE]
        try:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch
            )
            print(f"Upserted items {i} to {min(i+BATCH_SIZE, total_points)}...", end='\r')
        except Exception as e:
            print(f"Error upserting batch {i}: {e}")
            
    print("\nUpsert Complete.")

if __name__ == "__main__":
    main()
