from PyPDF2 import PdfReader
import fitz  # PyMuPDF
import spacy
import collections
import re
def extract_specific_pages(pdf_path,pages=None):
    reader=PdfReader(pdf_path)
    if pages is None:
        pages=range(1,len(reader.pages)+1)
    max_page=len(reader.pages)
    valid_pages=[p for p in pages if p<=max_page]
    extracted_text=""
    for page_num in valid_pages:
        page=reader.pages[page_num-1]
        extracted_text+=page.extract_text()
    return extracted_text

def preprocess_text(text):
    text=text.replace("\n"," ")
    text=text.replace(".","\n")
    return text
def extract_and_structure_pymupdf(pdf_path,pages=None):
    doc = fitz.open(pdf_path)
    full_text_structured = []
    if pages is None:
        pages=range(1,len(doc)+1)  
    for page_num in pages:
        # page_num=14
        page = doc.load_page(page_num-1)
        # Extract blocks with detailed info: (x0, y0, x1, y1, "text", block_no, block_type)
        # block_type: 0=text, 1=image
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: b[1]) # Sort blocks vertically (by y0)

        page_content = []
        last_y1 = 0

        for x0, y0, x1, y1, text, block_no, block_type in blocks:
            if block_type == 0: # It's a text block
                text = text.strip()
                if not text:
                    continue

                # Check for paragraph break (significant vertical gap)
                if y0 > last_y1 + 10: # Threshold gap (adjust as needed)
                    page_content.append("\n\n") # Add paragraph break marker

                # Simple heuristic for headings: assume larger text might be centered or alone
                # More robust: analyze font size/style via page.get_text("dict")
                # For now, just add the text, cleaning internal newlines
                cleaned_text = re.sub(r'\s*\n\s*', ' ', text) # Join lines within block
                cleaned_text = re.sub(r'-\s+', '', cleaned_text) # Handle hyphenation
                page_content.append(cleaned_text)
                last_y1 = y1

        page_text = "".join(page_content)
        full_text_structured.append(page_text)
        # break

    doc.close()
    return "".join(full_text_structured) # Join pages with paragraph breaks