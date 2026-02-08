import os
import glob
import json
import re
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# --- Configuration ---
# Define paths relative to this script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# current dir is backend/etl
BACKEND_DIR = os.path.dirname(CURRENT_DIR) # backend
PROJECT_ROOT = os.path.dirname(BACKEND_DIR) # Demo

# Input Directories
CANONICAL_DIR = os.path.join(PROJECT_ROOT, "materials", "Canonicalization")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "materials", "tables_images")

# Output Directory (for JSON logs and Qdrant data)
OUTPUT_DIR = os.path.join(BACKEND_DIR, "data")
QDRANT_PATH = os.path.join(BACKEND_DIR, "db", "tables_db")
COLLECTION_NAME = "hr_tables"

def get_images_for_table(table_basename: str) -> List[str]:
    """
    Finds matching images for a table.
    Input: 'table11'
    Matches: 'Table11.png', 'Table11_1.png', 'Table11_2.png' in IMAGES_DIR
    """
    # Normalize to lowercase for matching, though filesystem might be strict
    # We'll rely on glob's case sensitivity depending on OS. 
    # To be safe, we list all pngs and filter.
    
    matches = []
    # Search for Table11*
    # Capital 'T' seems to be the convention in the images folder based on previous `ls`
    # e.g. Table11_1.png
    
    # Construct a case-insensitive-like search pattern manually or just search "Table"+ID
    # Extract the number part: table11 -> 11
    digits = re.search(r'\d+', table_basename)
    if not digits:
        return []
    
    number = digits.group()
    pattern = f"Table{number}*.png"  # Pattern: Table11*.png
    full_pattern = os.path.join(IMAGES_DIR, pattern)
    
    files = glob.glob(full_pattern)
    return [os.path.abspath(f) for f in files]

def extract_caption(text_content: str) -> str:
    """
    Heuristic: The first non-empty line is often the caption.
    Or look for lines starting with 'Table' or '**جدول'.
    """
    lines = [l.strip() for l in text_content.splitlines() if l.strip()]
    if not lines:
        return "No content"
    
    # Priority: Check for line starting with **جدول (Arabic for Table)
    for line in lines:
        if line.startswith("**جدول") or line.startswith("Table"):
            return line
            
    # Fallback: Return first line
    return lines[0]

def main():
    print(f"--- Starting Simplified Table Embedding Pipeline ---")
    print(f"Reading text from: {CANONICAL_DIR}")
    print(f"Matching images from: {IMAGES_DIR}")

    # 1. Initialize Embedding Model
    print("Loading BAAI/bge-m3 model...")
    embedding_model = SentenceTransformer("BAAI/bge-m3")

    # 2. Process Files
    text_files = glob.glob(os.path.join(CANONICAL_DIR, "*.txt"))
    print(f"Found {len(text_files)} text files.")

    embedding_results = []
    qdrant_points = []
    
    # We need unique integer IDs for Qdrant points. 
    # We'll use the table number if possible, or a hash.
    
    for file_path in text_files:
        filename = os.path.basename(file_path)
        basename, _ = os.path.splitext(filename) # e.g. table11
        
        # Extract number for ID: table11 -> 11
        id_match = re.search(r'\d+', basename)
        if not id_match:
            print(f"Skipping {filename}: No number found in filename")
            continue
            
        table_id_int = int(id_match.group())
        
        # Read Text
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Metadata Extraction
        caption = extract_caption(content)
        images = get_images_for_table(basename)
        
        print(f"Processing {basename}: Found {len(images)} images. Caption: {caption[:30]}...")

        # Generate Embedding
        # SentenceTransformers.encode returns a numpy array or list
        vector = embedding_model.encode(content, normalize_embeddings=True)

        # Prepare Record
        metadata = {
            "document_name": "sharjah_hr_law 8.pdf", # Contextual knowledge
            "page_number": "N/A", # Not available in simple mapping
            "content_type": "table",
            "table_caption": caption,
            "cropped_table_images": images,
            "original_page_image": "Refer to PDF", # Placeholder as requested by simplified logic
            "original_filename": filename
        }

        # Save to Embedding Log
        log_entry = metadata.copy()
        log_entry["vector_status"] = "generated"
        log_entry["vector_first_5"] = vector[:5].tolist() # Preview
        embedding_results.append(log_entry)

        # Create Qdrant Point
        point = PointStruct(
            id=table_id_int,
            vector=vector.tolist(),
            payload={
                **metadata,
                "full_text": content
            }
        )
        qdrant_points.append(point)

    # 3. Save Embedding JSON Output
    embed_out_file = os.path.join(OUTPUT_DIR, "embedding_output.json")
    with open(embed_out_file, 'w', encoding='utf-8') as f:
        json.dump(embedding_results, f, indent=4, ensure_ascii=False)
    print(f"Saved embedding logs to {embed_out_file}")

    # 4. Store in Qdrant
    print(f"Initializing Qdrant at {QDRANT_PATH}...")
    client = QdrantClient(path=QDRANT_PATH)

    # Re-create collection
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

    print(f"Upserting {len(qdrant_points)} points...")
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=qdrant_points
    )

    # 5. Save Qdrant JSON Output
    qdrant_out_file = os.path.join(OUTPUT_DIR, "qdrant_output.json")
    qdrant_log = {
        "status": "success",
        "storage_path": QDRANT_PATH,
        "collection_name": COLLECTION_NAME,
        "total_items": len(qdrant_points),
        "items": [
            {
                "id": p.id,
                "caption": p.payload.get("table_caption")
            }
            for p in qdrant_points
        ]
    }
    with open(qdrant_out_file, 'w', encoding='utf-8') as f:
        json.dump(qdrant_log, f, indent=4, ensure_ascii=False)
    print(f"Saved Qdrant logs to {qdrant_out_file}")

if __name__ == "__main__":
    main()
