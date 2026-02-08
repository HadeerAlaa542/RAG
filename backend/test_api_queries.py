
import requests
import json
import time

# Configuration
API_URL = "http://localhost:8000/api/ask"

# List of Test Questions
test_questions = [
    # 1. English - General Text Query
    "What are the types of leaves mentioned in the HR law?",
    
    # 2. English - Specific Table Query
    "What are the details of Table 11?",
    
    # 3. Arabic - General Query
    "ما هي العقوبات التأديبية؟",
    
    # 4. English - Table + Text logic
    "What is the salary for Grade 1?",
]

def run_tests():
    print(f"--- Starting API Tests against {API_URL} ---")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[Test {i}] Asking: '{question}'")
        
        payload = {"question": question}
        
        try:
            start_time = time.time()
            response = requests.post(API_URL, json=payload)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "No answer field found")
                caption = data.get("context_caption", "No caption")
                images = data.get("image_paths", [])
                
                print(f"Status: SUCCESS ({duration:.2f}s)")
                print(f"Caption: {caption}")
                print(f"Images: {images}")
                print("-" * 20)
                print(f"Answer Preview: {answer[:300]}...") # Show first 300 chars
                print("-" * 20)
            else:
                print(f"Status: FAILED (Code {response.status_code})")
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"Status: ERROR")
            print(f"Exception: {e}")
            
    print("\n--- All Tests Completed ---")

if __name__ == "__main__":
    run_tests()
