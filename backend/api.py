from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import NamedVector
from openai import OpenAI
import os
import uvicorn
import re
import numpy as np

from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# Config
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

# Load environment variables from .env file
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

# Ensure OPENAI_API_KEY is set in your environment variables
if "OPENAI_API_KEY" not in os.environ:
    print("Warning: OPENAI_API_KEY not found in environment variables.")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "materials", "tables_images")
# Using clean names
QDRANT_PATH = os.path.join(BACKEND_DIR, "db", "tables_db")
QDRANT_PATH_TEXT = os.path.join(BACKEND_DIR, "db", "text_db")
COLLECTION_NAME = "hr_tables"
COLLECTION_NAME_TEXT = "hr_text_hybrid"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files for Images
if os.path.isdir(IMAGES_DIR):
    app.mount("/static", StaticFiles(directory=IMAGES_DIR), name="static")
else:
    print(f"Warning: Images directory {IMAGES_DIR} not found.")


# ... (rest of imports/mounts)

# Models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    context_caption: str
    image_paths: list[str]

# Global Service Instances
model = None
reranker = None  # New Reranker Model
qdrant_client_tables = None
qdrant_client_text = None
openai_client = None

def cleanup_lock(path):
    lock_file = os.path.join(path, ".lock")
    if os.path.exists(lock_file):
        try:
            os.remove(lock_file)
            print(f"Forcefully removed lock file: {lock_file}")
        except Exception as e:
            print(f"Warning: Could not remove lock {lock_file}: {e}")

@app.on_event("startup")
def startup_event():
    global model, reranker, qdrant_client_tables, qdrant_client_text, openai_client
    print("Initializing services...")
    
    # Force cleanup locks
    cleanup_lock(QDRANT_PATH)
    # Check if text path might exist (it's defined as v2 now)
    if QDRANT_PATH_TEXT: 
       cleanup_lock(QDRANT_PATH_TEXT)

    print("Loading Embedding Model...")
    model = SentenceTransformer("BAAI/bge-m3", device="cpu")
    
    print("Loading Reranker Model (BAAI/bge-reranker-v2-m3)...")
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512, device="cpu")
    
    qdrant_client_tables = QdrantClient(path=QDRANT_PATH)
    
    # Initialize Text Client (Fault tolerant)
    try:
        if os.path.exists(QDRANT_PATH_TEXT):
            qdrant_client_text = QdrantClient(path=QDRANT_PATH_TEXT)
            print("Text Qdrant Client Ready.")
        else:
            print("Warning: Text Qdrant Data not found yet.")
            qdrant_client_text = None
    except Exception as e:
        print(f"Error loading Text Qdrant: {e}")
        qdrant_client_text = None

    openai_client = OpenAI()
    print("Services ready.")

@app.post("/api/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    global model, reranker, qdrant_client_tables, qdrant_client_text, openai_client
    
    query = request.question
    print(f"Received query: {query}")
    
    # 1. Embed and Search (Initial Retrieval)
    query_vector = model.encode(query, normalize_embeddings=True).tolist()
    
    initial_hits_tables = []
    initial_hits_text = []
    
    # Fetch narrow pool for reranking (Top 5 is faster on CPU with Large Model)
    retrieval_limit = 5
    
    # Search Tables
    try:
        if hasattr(qdrant_client_tables, 'search'):
            initial_hits_tables = qdrant_client_tables.search(collection_name=COLLECTION_NAME, query_vector=query_vector, limit=retrieval_limit)
        else:
            initial_hits_tables = qdrant_client_tables.query_points(collection_name=COLLECTION_NAME, query=query_vector, limit=retrieval_limit).points
    except Exception as e:
        print(f"Error searching tables: {e}")

    # Search Text
    if qdrant_client_text:
        try:
            if hasattr(qdrant_client_text, 'search'):
                initial_hits_text = qdrant_client_text.search(
                    collection_name=COLLECTION_NAME_TEXT, 
                    query_vector=query_vector, 
                    limit=retrieval_limit
                )
            else:
                initial_hits_text = qdrant_client_text.query_points(
                    collection_name=COLLECTION_NAME_TEXT, 
                    query=query_vector, 
                    limit=retrieval_limit
                ).points
        except Exception as e:
            print(f"Error searching text: {e}")
        
    # --- RERANKING STEP ---
    # Combine Hits
    combined_candidates = []
    
    for hit in initial_hits_tables:
        text = hit.payload.get('full_text', '')
        combined_candidates.append({'hit': hit, 'text': text, 'type': 'table'})
        
    for hit in initial_hits_text:
        text = hit.payload.get('text', hit.payload.get('full_text', ''))
        combined_candidates.append({'hit': hit, 'text': text, 'type': 'text'})
    
    if not combined_candidates:
        return QueryResponse(answer="I could not find any relevant information in the HR documents.", context_caption="", image_paths=[])

    # Prepare pairs for CrossEncoder: [[query, doc_text], [query, doc_text], ...]
    rerank_pairs = [[query, c['text']] for c in combined_candidates]
    
    print(f"Reranking {len(rerank_pairs)} candidates...")
    scores = reranker.predict(rerank_pairs)
    
    # Attach scores
    for i, candidate in enumerate(combined_candidates):
        score = scores[i]
        # Handle different return types (scalar vs array)
        if hasattr(score, 'item'):
            score = score.item()
        elif isinstance(score, (list, tuple, np.ndarray)) and len(score) > 0:
            score = float(score[0])
        else:
            score = float(score)
            
        candidate['rerank_score'] = score
        # Update the hit object's score to reflect the new Reranker score (for consistency later)
        candidate['hit'].score = score

    # Sort by Reranker Score (Desc)
    combined_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
    
    # Take Top 3 Winners (Final Context)
    top_k_final = 3
    final_winners = combined_candidates[:top_k_final]
    
    # Separate back into 'hits_tables' and 'hits_text' for downstream logic compat
    hits_tables = [x['hit'] for x in final_winners if x['type'] == 'table']
    hits_text = [x['hit'] for x in final_winners if x['type'] == 'text']
    
    # Debug
    print("--- Top Reranked Results ---")
    for w in final_winners:
        print(f"Type: {w['type']}, Score: {w['rerank_score']:.4f}, Text Snippet: {w['text'][:50]}...")
    
    all_hits = hits_tables + hits_text # Used for empty check later

    context_text = ""
    
    # Score Comparison Logic
    best_table_score = hits_tables[0].score if hits_tables else -999.0
    best_text_score = hits_text[0].score if hits_text else -999.0
    
    print(f"Scores - Table: {best_table_score}, Text: {best_text_score}")
    
    # Heuristic: Prefer table if it's competitive (bias towards table for visualization)
    # If table score is within 0.15 of text score, prefer table (increased bias to catch Imams/Muazzins case)
    if hits_tables and (best_table_score + 0.15 >= best_text_score):
        use_table_context = True
    else:
        use_table_context = False
    
    images = []
    
    # SOFT FILTER: Only show images if the score is decent (e.g. > -2.0)
    # Stricter threshold to avoid showing random tables for chatty/irrelevant queries.
    show_images_threshold = -2.0
    is_image_relevant = best_table_score > show_images_threshold
    
    print(f"Image Relevance Check: Score {best_table_score} > {show_images_threshold}? {is_image_relevant}")

    # Only process images and captions if table context is used AND relevant
    if use_table_context and hits_tables and is_image_relevant:
        # Collect captions and images from ALL retrieved tables (Top-K)
        captions_list = []
        for hit in hits_tables:
            cap = hit.payload.get('table_caption', 'Unknown Table')
            captions_list.append(cap)
            
            # Append images from this hit
            raw_images = hit.payload.get('cropped_table_images', [])
            for img_path in raw_images:
                if img_path:
                    filename = os.path.basename(img_path)
                    # Avoid duplicates
                    img_url = f"http://localhost:8000/static/{filename}"
                    if img_url not in images:
                        images.append(img_url)
        
        caption = " | ".join(captions_list)
    elif hits_text:
        # If using text, try to get a better caption (e.g. Section Title or Page Number)
        top_text_hit = hits_text[0]
        page_num = top_text_hit.payload.get('page_number', '?')
        doc_name = top_text_hit.payload.get('document_name', 'HR Law')
        caption = f"{doc_name} - Page {page_num}"
    else:
        caption = "General HR Law Text"
    
    # Process Table Hits (Add to context)
    for hit in hits_tables:
        payload = hit.payload
        part_text = payload.get('full_text', '')
        context_text += f"\n--- Table Segment (Score: {hit.score}) ---\nCaption: {payload.get('table_caption')}\n{part_text}\n"

    # Process Text Hits
    for hit in hits_text:
        payload = hit.payload
        # Text collection stores content in 'text' field mostly, but let's check payload keys
        part_text = payload.get('text', payload.get('full_text', ''))
        page_num = payload.get('page_number', '?')
        context_text += f"\n--- Text Segment (Page {page_num}, Score: {hit.score}) ---\n{part_text}\n"

    # Language Detection (Python-side enforcement)
    # Check if query contains Arabic characters
    is_arabic = bool(re.search(r'[\u0600-\u06FF]', query))

    if is_arabic:
        print("Detected Language: Arabic")
        system_prompt = (
            "أنت مساعد موارد بشرية ذكي لحكومة الشارقة. "
            " لديك صلاحية الوصول إلى نوعين من المصادر: جداول (Tables) ونصوص قانونية (Text Articles).\n"
            "قاعدتك الأساسية هي: يجب أن تكون إجابتك باللغة العربية الفصحى حصراً.\n"
            "القواعد:\n"
            "1. **المحادثة العادية:** إذا بدأ المستخدم بالتحية (مثل 'مرحباً'، 'صباح الخير')، رد عليه بلطف وتحية مماثلة، وعرف بنفسك كمساعد للموارد البشرية.\n"
            "2. إذا كانت الإجابة مستمدة من جدول، **يجب** أن تذكر رقم الجدول بوضوح في إجابتك (مثلاً: 'حسب ما ورد في الجدول رقم 10...').\n"
            "3. إذا كانت الإجابة مستمدة من نص قانوني عادي، **لا تذكر أي جداول**، وبدلاً من ذلك اذكر اسم المادة أو رقم الصفحة إن وجد.\n"
            "4. انتبه جيداً للمسميات الوظيفية (مثل 'فني' و 'تقني') ولا تخلط بينها.\n"
            "5. الإجابة يجب أن تكون دقيقة ومبنية فقط على السياق المزود أدناه.\n"
            "6. ممنوع الإجابة باللغة الإنجليزية.\n"
        )
    else:
        print("Detected Language: English")
        system_prompt = (
            "You are an HR Assistant for the Government of Sharjah. "
            "You have access to two types of sources: Tables and Text Articles.\n"
            "Your PRIMARY RULE is to reply in ENGLISH.\n"
            "RULES:\n"
            "1. **Casual Chat:** If the user greets you (e.g., 'Hi', 'Good morning'), reply warmly and introduce yourself as an HR assistant.\n"
            "2. If the answer comes from a Table, you MUST explicitly mention the Table Number (e.g., 'According to Table 10...').\n"
            "3. If the answer comes from a Text Article, DO NOT mention any tables. Instead, cite the Article name or Page number if available.\n"
            "4. Pay extremely close attention to similar terms (e.g., 'Technician' vs 'Technologist').\n"
            "5. Answer strictly based on the provided context.\n"
            "6. Do not use Arabic in your response.\n"
        )

    user_message = f"Context:\n{context_text}\n\nQuestion: {query}"
    
    gpt_response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}],
        temperature=0.0
    )
    
    answer = gpt_response.choices[0].message.content
    
    return QueryResponse(
        answer=answer,
        context_caption=caption,
        image_paths=images
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
