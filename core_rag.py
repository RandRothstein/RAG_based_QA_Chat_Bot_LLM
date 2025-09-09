# core_rag.py

import os
import json
import csv # <-- ADDED THIS IMPORT for CSV handling
from typing import List, Dict, Union
from io import BytesIO, StringIO # <-- ADDED StringIO for CSV text decoding

from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig, TrainingArguments, Trainer
# --- Configuration ---
DOCUMENTS_DIR = "documents"
FAISS_INDEX_PATH = "document_qa_index.faiss"
DOC_CHUNKS_PATH = "document_chunks.json"

# --- Model Loading ---
QA_PIPELINE = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base",torch_dtype=torch.bfloat16)
EMBEDDING_MODEL = AutoTokenizer.from_pretrained("google/flan-t5-base")
# --- Document Processing Functions ---

def process_file_content(file_name: str, file_content_bytes: bytes) -> Union[str, None]:
    """Processes file content (txt, pdf, or csv bytes) and returns its text."""
    file_extension = os.path.splitext(file_name)[1].lower()
    text_content = ""

    if file_extension == ".txt":
        try:
            text_content = file_content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text_content = file_content_bytes.decode("latin-1")
    elif file_extension == ".pdf":
        try:
            reader = PdfReader(BytesIO(file_content_bytes))
            for page in reader.pages:
                text_content += page.extract_text() or ""
        except Exception as e:
            print(f"Error reading PDF {file_name}: {e}")
            return None
    elif file_extension == ".csv": # <-- ADDED THIS BLOCK for CSV handling
        try:
            # Decode bytes to string, then use StringIO for csv.reader
            decoded_text = file_content_bytes.decode("utf-8")
            csv_file = StringIO(decoded_text)
            reader = csv.reader(csv_file)
            rows = []
            for row in reader:
                rows.append(", ".join(row)) # Join columns in a row with a comma and space
            text_content = "\n".join(rows) # Join rows with a newline
            print(f"Processed CSV file: {file_name}")
        except Exception as e:
            print(f"Error reading CSV {file_name}: {e}")
            return None
    else:
        print(f"Skipped unsupported file type: {file_name}. Only .txt, .pdf, and .csv are supported.")
        return None
    return text_content

def build_faiss_index(document_texts: List[str]):
    """
    Builds or rebuilds the FAISS index and saves chunks from a list of document texts.
    """
    if not document_texts:
        print("No documents provided to build the index. Clearing existing index if any.")
        if os.path.exists(FAISS_INDEX_PATH):
            os.remove(FAISS_INDEX_PATH)
        if os.path.exists(DOC_CHUNKS_PATH):
            os.remove(DOC_CHUNKS_PATH)
        return False # Indicate no index was built

    print(f"Chunking {len(document_texts)} documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_chunks = []
    for doc_text in document_texts:
        chunks = splitter.split_text(doc_text)
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No chunks generated from the provided documents. Index will not be built.")
        return False

    print(f"Encoding {len(all_chunks)} chunks...")
    vectors = EMBEDDING_MODEL.encode(all_chunks, convert_to_numpy=True)

    print("Creating FAISS index...")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"FAISS index saved to {FAISS_INDEX_PATH}")

    with open(DOC_CHUNKS_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"Chunks saved to {DOC_CHUNKS_PATH}")
    print("FAISS Index Building Complete!")
    return True # Indicate success

def get_answer_from_rag(query: str, top_k: int = 3) -> Dict:
    """
    Performs the RAG process: retrieves context and gets an answer from the QA pipeline.
    Returns a dictionary with 'answer', 'score', and 'context'.
    """
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(DOC_CHUNKS_PATH):
        return {"answer": "Error: Knowledge base not built. Please upload documents and build the index.",
                "score": 0.0,
                "context": ""}

    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(DOC_CHUNKS_PATH, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
    except Exception as e:
        print(f"Error loading index or chunks: {e}")
        return {"answer": "An error occurred while loading the knowledge base.",
                "score": 0.0,
                "context": ""}

    query_vec = EMBEDDING_MODEL.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vec, top_k)
    retrieved_chunks = [chunks[i] for i in I[0]]

    context_for_qa = "\n\n".join(retrieved_chunks)

    if not context_for_qa.strip():
        return {"answer": "No relevant context found in documents for your query. Please try a different query or upload more relevant documents.",
                "score": 0.0,
                "context": ""}

    try:
        qa_result = QA_PIPELINE(question=query, context=context_for_qa)
        return {"answer": qa_result['answer'],
                "score": qa_result['score'],
                "context": context_for_qa}
    except Exception as e:
        print(f"Error during question answering: {e}")
        return {"answer": "An error occurred during answer generation.",
                "score": 0.0,
                "context": ""}

# Ensure the documents directory exists when this module is imported
if not os.path.exists(DOCUMENTS_DIR):
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    print(f"Created '{DOCUMENTS_DIR}' directory.")

# Add a demo news.txt file if it doesn't exist for initial setup
news_filepath = os.path.join(DOCUMENTS_DIR, "news.txt")
if not os.path.exists(news_filepath):
    news_content = (
        "Artificial Intelligence (AI) is transforming the healthcare industry by enhancing diagnostics, personalizing treatment plans, and improving patient outcomes. "
        "AI models can analyze vast amounts of medical data to detect patterns that are not easily visible to human doctors. "
        "In recent years, AI-powered tools have been deployed to interpret medical imaging such as X-rays and MRIs. "
        "These tools can often match or exceed the accuracy of radiologists in identifying anomalies like tumors or fractures. "
        "Moreover, AI is being used to assist in drug discovery by predicting molecular behavior and accelerating research timelines. "
        "Personalized medicine is another growing area, where AI analyzes patient genetics and lifestyle to suggest the most effective treatments. "
        "Despite the potential, challenges remain in data privacy, bias in training data, and the need for regulatory oversight. "
        "Nevertheless, the integration of AI into healthcare continues to grow rapidly, offering promising advancements in patient care."
    )
    with open(news_filepath, "w", encoding="utf-8") as f:
        f.write(news_content)
    print(f"Created demo file: {news_filepath}")
