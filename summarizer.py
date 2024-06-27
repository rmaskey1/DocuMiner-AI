import fitz  # PyMuPDF
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BartTokenizer, BartForConditionalGeneration
import torch
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def extract_text_from_pdf(pdf_path):
    logging.info("Extracting text from PDF")
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    logging.info("Text extraction complete")
    return text

def preprocess_text(text):
    logging.info("Preprocessing text")
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalpha() and token not in stop_words]
    return " ".join(tokens)

def classify_document(text):
    logging.info("Classifying document")
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("nlpaueb/legal-bert-base-uncased", num_labels=3)
    
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label = probs.argmax().item()
    logging.info(f"Document classified as type: {label}")
    return label  # Adjust to return the appropriate document type

def summarize_text(text):
    logging.info("Summarizing text")
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    logging.info("Summarization complete")
    return summary

def summarize_legal_document(pdf_path):
    logging.info(f"Summarizing legal document: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    preprocessed_text = preprocess_text(text)
    document_type = classify_document(preprocessed_text)
    
    # Customize the summarization approach based on the document type
    if document_type == 0:  # E.g., contract
        summary = summarize_text(preprocessed_text)
    elif document_type == 1:  # E.g., court ruling
        summary = summarize_text(preprocessed_text)
    elif document_type == 2:  # E.g., statute
        summary = summarize_text(preprocessed_text)
    else:
        summary = "Document type not recognized."
    
    return summary

# Test the function
if __name__ == "__main__":
    pdf_path = 'TestPDFfile.pdf'
    summary = summarize_legal_document(pdf_path)
    print("\n")
    print(summary)
