import re
from pathlib import Path
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from pypdf import PdfReader

# Ensure you have the necessary NLTK data downloaded
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to clean text: remove stop words, special characters, and numbers
def clean_text(sentence, stop_words, stemmer, lemmatizer):
    # Remove special characters, numbers, and underscores
    sentence = re.sub(r'[^\w\s]|_', '', sentence)  # Remove special characters and underscores
    
    # Tokenize and process each word
    words = sentence.split()
    processed_words = []
    for word in words:
        if word.lower() not in stop_words:
            stemmed_word = stemmer.stem(word)  # Perform stemming
            lemmatized_word = lemmatizer.lemmatize(stemmed_word)  # Perform lemmatization
            processed_words.append(lemmatized_word)
    
    return ' '.join(processed_words)
num = 4
# Load PDF and extract text
pdf_path = f"{num}.pdf"  # Replace with your PDF file path
pdf_reader = PdfReader(pdf_path)

# Combine text from all pages
text = ""
for page in pdf_reader.pages:
    text += page.extract_text()

# Split text into sentences
sentences = sent_tokenize(text)

# Initialize NLTK tools
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Process sentences: clean and transform them
processed_sentences = [clean_text(sentence, stop_words, stemmer, lemmatizer) for sentence in sentences]

# Save processed sentences to a .txt file (one sentence per line)
output_path = f"{num}.txt"
with open(output_path, "w", encoding="utf-8") as output_file:
    output_file.write("\n".join(processed_sentences))

print(f"Processed text saved to {output_path}")
