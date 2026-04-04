import os
from pathlib import Path
from docling.document_converter import DocumentConverter

'''
Convert PDF to Markdown
'''

def convert_pdfs():
    # Setup paths
    raw_dir = Path("../data/raw")
    processed_dir = Path("../data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    converter = DocumentConverter()
    
    # Process all PDFs in the raw folder
    for pdf_path in raw_dir.glob("*.pdf"):
        print(f"Converting: {pdf_path.name}...")
        
        # 1. Perform conversion
        result = converter.convert(pdf_path)
        
        # 2. Export to Markdown
        md_output = result.document.export_to_markdown()
        
        # 3. Save file
        output_file = processed_dir / f"{pdf_path.stem}.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(md_output)
            
    print("Done! Check your ../data/processed folder.")

if __name__ == "__main__":
    convert_pdfs()