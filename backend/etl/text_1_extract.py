import pdfplumber
from bidi.algorithm import get_display
import re
import os

def repair_text(text):
    if not text: return ""
    # Generic Fix 1: English Text merged with Arabic 'أ' (Alif Hamza) acting as space
    # e.g., "Iأacknowledge" -> "I acknowledge"
    text = re.sub(r'([a-zA-Z])أ', r'\1 ', text)
    text = re.sub(r'أ([a-zA-Z])', r' \1', text)

    # Generic Fix 2: Arabic Non-Connectors followed by Alifs
    # Letters that CANNOT connect left: ا, د, ذ, ر, ز, و, ة, ؤ
    # If followed by Alif (start of new word), split them.
    text = re.sub(r'([اأإآدذرزوؤة])([اأإآ])', r'\1 \2', text)
    
    # Generic Fix 3: Ain/Ghain (ع/غ) followed by Alif Hamza (أ)
    # This addresses "رابعأمرة". While technically they can connect, 
    # 'Ain' followed immediately by 'Alif Hamza' is extremely rare inside a root word.
    text = re.sub(r'([عغ])(أ)', r'\1 \2', text)

    # Fix Brackets for RTL display
    # Swap ( and ) because of Bidi mirroring issues
    text = text.replace('(', 'TEMPOPEN').replace(')', '(').replace('TEMPOPEN', ')')
    
    return text

def extract_text_excluding_tables(pdf_path):
    # Create Output Directory
    etl_dir = os.path.dirname(os.path.abspath(__file__))
    # current dir is backend/etl
    backend_dir = os.path.dirname(etl_dir) # backend
    project_root = os.path.dirname(backend_dir) # Demo
    output_dir = os.path.join(project_root, "materials", "Normal_Text_Extraction")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    total_pages = 0
    
    with pdfplumber.open(pdf_path) as pdf:
        # We only want pages 1 to 89 (index 0 to 88)
        pages_to_process = pdf.pages[:89]
        total_pages = len(pages_to_process)
        
        for i, page in enumerate(pages_to_process):
            page_num = i + 1
            print(f"Processing Page {page_num}/{total_pages}...", end='\r')
            
            # 1. Find Tables
            tables = page.find_tables()
            
            # 2. Define a filter to ignore text inside tables AND footers (page numbers)
            def not_inside_tables_or_footer(obj):
                # Check if object (char) is inside any detected table bbox
                obj_x = (obj['x0'] + obj['x1']) / 2
                obj_y = (obj['top'] + obj['bottom']) / 2
                
                # Check footer area (approx bottom 5% of page)
                if obj['bottom'] > page.height * 0.95:
                    return False # Ignore bottom 5% (Footer/Page Number)
                
                for table in tables:
                    tx, ty, bx, by = table.bbox
                    if tx <= obj_x <= bx and ty <= obj_y <= by:
                        return False # It IS inside a table, so filter it OUT
                return True

            # 3. Create a filtered version of the page (Text Only)
            clean_page = page.filter(not_inside_tables_or_footer)
            
            # 4. Extract Text
            content = clean_page.extract_text()
            
            if not content:
                continue
                
            # 5. Fix Arabic/English
            cleaned_lines = []
            for line in content.split('\n'):
                line = repair_text(line)
                
                # Apply Bidi ONLY if line has Arabic
                if re.search(r'[\u0600-\u06FF]', line):
                    line = get_display(line)
                
                cleaned_lines.append(line)
            
            final_text = "\n".join(cleaned_lines)
            
            # Save to individual file
            output_filename = f"page_{page_num}.txt"
            output_path = os.path.join(output_dir, output_filename)
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_text)
        
    print(f"\nExtraction Complete.")
    print(f"Text files (excluding tables) saved to: '{output_dir}'")

if __name__ == "__main__":
    # Assuming the PDF is in the project root (Demo folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(os.path.dirname(script_dir), "sharjah_hr_law 8.pdf")
    
    if os.path.exists(pdf_path):
        extract_text_excluding_tables(pdf_path)
    else:
        print(f"Error: PDF not found at {pdf_path}")
