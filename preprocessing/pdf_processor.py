import PyPDF2
import re
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
import json
from pathlib import Path
import os

class PDFProcessor:
    def __init__(self):
        self.cleaned_text = ""
        
    def read_pdf(self, pdf_path):
        """Read PDF file and extract text"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return None
    
    def clean_text(self, text):
        """Clean the extracted text while preserving important regulatory content"""
        # Remove extra whitespace while preserving paragraph structure
        text = re.sub(r'\s+', ' ', text)
        
        # Keep important punctuation and characters common in regulations
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\(\)\-\/]', '', text)
        
        # Split into sentences on periods while preserving numbering
        sentences = []
        # First split on obvious sentence boundaries
        raw_splits = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text)
        
        for split in raw_splits:
            # Further split on periods that are followed by spaces
            sub_sentences = re.split(r'(?<=\.)\s+', split)
            for sentence in sub_sentences:
                sentence = sentence.strip()
                if sentence:
                    # Don't split numbered items (e.g., "1.2" or "Art. 5")
                    if not re.match(r'^\d+\.$', sentence):
                        sentences.append(sentence)
        
        self.cleaned_text = '\n'.join(sentences)
        return self.cleaned_text

    def convert_to_json(self, output_path):
        """Save cleaned text in Label Studio compatible JSON format"""
        try:
            # Prepare data structure for Label Studio
            tasks = []
            
            # Split text into manageable chunks for labeling
            paragraphs = self.cleaned_text.split('\n')
            
            for idx, text in enumerate(paragraphs):
                if text.strip():
                    task = {
                        "id": idx + 1,
                        "text": text,
                        "annotations": [],
                        "meta": {
                            "source": "pollution_regulations",
                            "section": f"section_{idx + 1}"
                        }
                    }
                    tasks.append(task)

            # Save as JSON for Label Studio import
            json_path = str(Path(output_path).with_suffix('.json'))
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(tasks, f, indent=2, ensure_ascii=False)

            # Keep the original txt output
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(self.cleaned_text)

            print(f"Files saved for Label Studio:\n"
                  f"1. JSON format: {json_path}\n"
                  f"2. Text format: {output_path}")
        except Exception as e:
            print(f"Error saving Label Studio format: {e}")

    def process_directory(self, input_dir):
        """Process all PDF files in the input directory"""
        all_text = []
        input_path = Path(input_dir)
        
        for pdf_file in input_path.glob('*.pdf'):
            print(f"Processing: {pdf_file}")
            text = self.read_pdf(str(pdf_file))
            if text:
                cleaned = self.clean_text(text)
                all_text.append(cleaned)
        
        # Combine all cleaned text
        self.cleaned_text = '\n\n'.join(all_text)
        return self.cleaned_text

def process_pdfs(input_dir, output_path):
    processor = PDFProcessor()
    processor.process_directory(input_dir)
    processor.convert_to_json(output_path)
    return True

# Example usage
if __name__ == "__main__":
    input_dir = r"./data/input"
    output_path = r"./data/output/cleaned_text.txt"
    process_pdfs(input_dir, output_path)