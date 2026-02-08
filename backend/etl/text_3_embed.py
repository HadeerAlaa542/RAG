
import os
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Fix for Windows Symlink Error (OSError 1314)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Try importing FlagEmbedding for BGE-M3
try:
    from FlagEmbedding import BGEM3FlagModel
    USE_BGE_M3_HYBRID = True
except ImportError:
    print("Warning: FlagEmbedding not found. Falling back to Dense Only (sentence-transformers).")
    USE_BGE_M3_HYBRID = False
    from sentence_transformers import SentenceTransformer

# --- Configuration ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR) # backend
INPUT_FILE = os.path.join(BACKEND_DIR, "data", "text_chunks.json")
OUTPUT_FILE = os.path.join(BACKEND_DIR, "data", "text_embeddings.json")

def main():
    print(f"--- Step 2: Generating Embeddings ---")
    print(f"Reading from: {INPUT_FILE}")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Run step1_chunk_text.py first!")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
        
    print(f"Found {len(chunks_data)} chunks to embed.")
    
    # 1. Initialize Model
    print("Loading Embedding Model...")
    model = None
    is_hybrid_active = False

    if USE_BGE_M3_HYBRID:
        try:
            print("Attempting to load BGEM3FlagModel (Hybrid)...")
            model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False) 
            print("Successfully loaded Hybrid Model.")
            is_hybrid_active = True
        except Exception as e:
            print(f"Warning: Failed to load Hybrid Model: {e}")
            is_hybrid_active = False

    if not is_hybrid_active:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('BAAI/bge-m3')
        print("Loaded Dense-Only Model.")

    embedded_data = []
    
    # Extract Texts
    # FULL PROCESSING (All chunks)
    # chunks_data = chunks_data[:50] # Removed limit
    texts = [c["text"] for c in chunks_data]
    
    batch_size = 16 
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    print(f"Processing in {total_batches} batches...")
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_metas = chunks_data[i : i + batch_size]
        
        print(f"Embedding Batch {i//batch_size + 1}/{total_batches}...", end='\r')
        
        if is_hybrid_active:
            output = model.encode(batch_texts, return_dense=True, return_sparse=True, return_colbert_vecs=False)
            batch_dense = output['dense_vecs']
            batch_lexical = output['lexical_weights']
        else:
            batch_dense = model.encode(batch_texts, normalize_embeddings=True)
            batch_lexical = [None] * len(batch_texts)
        
        # Store Results
        for j, text in enumerate(batch_texts):
            meta = batch_metas[j]
            dense_vec = batch_dense[j].tolist()
            
            sparse_vec = None
            if is_hybrid_active and batch_lexical[j]:
                # Convert keys to int strings for JSON serialization
                # Or keep as dict {"token_id": weight} to be processed later
                sparse_vec = batch_lexical[j] 
            
            entry = {
                "id": meta["id"],
                "text": text,
                "metadata": meta["metadata"],
                "vector": {
                    "dense": dense_vec,
                    "sparse": sparse_vec # Will be None or dict
                }
            }
            embedded_data.append(entry)
            
    print("\nEncoding Complete.")
    
    # Save to JSON
    # Note: This file might be large (~3-5MB), but manageable.
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(embedded_data, f, cls=NumpyEncoder, ensure_ascii=False) # Skip indent to save space
        
    # Also save a 'Preview' file for easy reading (without massive vector arrays)
    preview_data = []
    for item in embedded_data[:5]: # First 5
        preview = item.copy()
        preview["vector"] = {"dense": "Array(1024)...", "sparse": "SparseDict..."}
        preview_data.append(preview)
            
    with open(OUTPUT_FILE.replace(".json", "_preview.json"), "w", encoding="utf-8") as f:
        json.dump(preview_data, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)

    print(f"Full embeddings saved to: {OUTPUT_FILE}")
    print(f"Preview saved to: {OUTPUT_FILE.replace('.json', '_preview.json')}")

if __name__ == "__main__":
    main()
