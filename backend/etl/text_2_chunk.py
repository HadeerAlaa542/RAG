
import os
import glob
import re
import json
import sys

# --- Configuration ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# current dir is backend/etl
BACKEND_DIR = os.path.dirname(CURRENT_DIR) # backend
PROJECT_ROOT = os.path.dirname(BACKEND_DIR) # Demo

INPUT_DIR = os.path.join(PROJECT_ROOT, "materials", "Normal_Text_Extraction")
OUTPUT_FILE = os.path.join(BACKEND_DIR, "data", "text_chunks.json")

def get_text_chunks_simple(text: str, chunk_size: int = 800, chunk_overlap: int = 150):
    chunks = []
    start = 0
    text_len = len(text)
    
    # Safety Check: Empty text
    if not text:
        return []

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        
        # Snap to newline/space logic
        if end < text_len: # Only adjust if not at the very end
            last_newline = chunk.rfind('\n')
            if last_newline != -1 and last_newline > chunk_size * 0.5:
                # Found a good newline
                end = start + last_newline + 1
            else:
                last_space = chunk.rfind(' ')
                if last_space != -1:
                    # Found a good space
                    end = start + last_space + 1
        
        final_chunk = text[start:end].strip()
        if final_chunk:
            chunks.append(final_chunk)
            
        # Move start forward
        # IMPORTANT: Ensure start ALWAYS increases to avoid infinite loop
        step = end - start
        new_start = end - chunk_overlap
        
        # If the step was smaller than overlap (e.g. very small chunk at end), just finish
        if step <= chunk_overlap and end == text_len:
            break
            
        # Ensure forward progress
        if new_start <= start:
            # Force move forward if overlap logic stalls
            new_start = start + max(1, step - chunk_overlap)
        
        start = new_start
        
        # Safety break for huge loops
        if len(chunks) > 10000:
            print("Warning: Too many chunks for one file! Breaking loop.")
            break

    return chunks

def main():
    print(f"--- Step 1: Chunking Text (Debug Mode) ---", flush=True)
    print(f"Reading from: {INPUT_DIR}", flush=True)
    
    input_files = glob.glob(os.path.join(INPUT_DIR, "*.txt"))
    # Sort
    try:
        input_files.sort(key=lambda f: int(re.search(r'page_(\d+)', f).group(1)))
    except:
        pass 
    
    print(f"Found {len(input_files)} text files.", flush=True)
    
    all_chunks_data = []
    
    debug_limit = 9999 # Set to low number (e.g. 5) if still hanging
    
    for i, file_path in enumerate(input_files):
        if i >= debug_limit:
            break
            
        filename = os.path.basename(file_path)
        print(f"[{i+1}/{len(input_files)}] Opening {filename}...", end=" ", flush=True)
        
        try:
            match = re.search(r'page_(\d+)', filename)
            page_num = int(match.group(1)) if match else 0
            
            with open(file_path, 'r', encoding='utf-8') as f:
                full_text = f.read()
                
            print(f"Read {len(full_text)} chars...", end=" ", flush=True)
                
            chunks = get_text_chunks_simple(full_text)
            print(f"Generated {len(chunks)} chunks.", flush=True)
            
            for j, chunk in enumerate(chunks):
                chunk_id = f"page_{page_num}_chunk_{j}"
                chunk_data = {
                    "id": chunk_id,
                    "text": chunk,
                    "metadata": {
                        "source": "sharjah_hr_law 8.pdf",
                        "page_number": page_num,
                        "content_type": "text",
                        "chunk_index": j
                    }
                }
                all_chunks_data.append(chunk_data)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)

    print(f"\nEncoding {len(all_chunks_data)} chunks to JSON...", flush=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks_data, f, indent=2, ensure_ascii=False)
        
    print(f"SUCCESS: Saved output to: {OUTPUT_FILE}", flush=True)

if __name__ == "__main__":
    main()
