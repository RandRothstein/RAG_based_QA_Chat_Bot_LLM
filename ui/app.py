# ui/app.py

import streamlit as st
import os
import hashlib # For better hashing of content
from io import BytesIO

# Import functions and models from the core RAG logic
import core_rag

# --- Streamlit UI ---

st.set_page_config(page_title="Document Q&A with RAG", layout="wide")

st.title("📄 Document Q&A System with RAG")
st.markdown("Upload your `.txt` or `.pdf` documents, and ask questions about their content.")

# Initialize session state variables
if 'index_built' not in st.session_state:
    st.session_state['index_built'] = False
if 'document_count' not in st.session_state:
    st.session_state['document_count'] = 0
if 'current_documents_hash' not in st.session_state:
    st.session_state['current_documents_hash'] = None # To detect changes in uploaded files

# --- Helper to get current document texts ---
def get_current_document_texts_from_disk():
    """Reads all existing documents from the documents/ directory and returns their texts and a content hash."""
    doc_texts = []
    content_hash_builder = hashlib.md5() # Use MD5 for a content hash

    if os.path.exists(core_rag.DOCUMENTS_DIR):
        for filename in os.listdir(core_rag.DOCUMENTS_DIR):
            file_path = os.path.join(core_rag.DOCUMENTS_DIR, filename)
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'rb') as f:
                        file_content_bytes = f.read()
                    content_hash_builder.update(file_content_bytes) # Update hash with content
                    text = core_rag.process_file_content(filename, file_content_bytes)
                    if text:
                        doc_texts.append(text)
                except Exception as e:
                    st.warning(f"Could not read file {filename} from disk: {e}")
    return doc_texts, content_hash_builder.hexdigest()

# --- Initial/Reload Indexing ---
@st.cache_resource
def initial_index_build():
    """Performs initial index build on app startup using documents already on disk."""
    st.info("Performing initial/startup index check. Please wait...")
    doc_texts, current_hash = get_current_document_texts_from_disk()

    # Only build if there are documents and the hash doesn't match a previously built index
    # (This simple hash check is a basic cache key for `st.cache_resource`)
    if doc_texts:
        if core_rag.build_faiss_index(doc_texts):
            st.session_state['index_built'] = True
            st.session_state['document_count'] = len(doc_texts)
            st.session_state['current_documents_hash'] = current_hash # Store the hash of the indexed content
            st.success(f"Initial knowledge base built with {len(doc_texts)} documents from disk.")
        else:
            st.session_state['index_built'] = False
            st.session_state['document_count'] = 0
            st.error("Failed to build initial knowledge base from existing documents.")
    else:
        st.session_state['index_built'] = False
        st.session_state['document_count'] = 0
        st.info("No documents found on disk for initial index build. Please upload files.")


# Run initial build
initial_index_build()


# --- File Uploader Section ---
st.header("1. Upload New Documents (Optional)")
uploaded_files = st.file_uploader(
    "Upload .txt, .pdf, or .csv files to add to the knowledge base (Existing files will be replaced).",
    type=["txt", "pdf", "csv"], # <-- ADDED "csv" here
    accept_multiple_files=True,
    help="Uploading new files will rebuild the entire knowledge base with the new content."
)

if uploaded_files:
    if st.button("Process & Rebuild Knowledge Base"):
        new_doc_texts_from_upload = []
        
        with st.spinner("Clearing existing documents and saving new ones..."):
            # 1. Clear ALL existing files from the documents directory first
            if os.path.exists(core_rag.DOCUMENTS_DIR):
                for f in os.listdir(core_rag.DOCUMENTS_DIR):
                    file_path = os.path.join(core_rag.DOCUMENTS_DIR, f)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
            st.info(f"Cleared existing documents from '{core_rag.DOCUMENTS_DIR}'.")

            # 2. Save the NEWLY uploaded files to the now-empty documents directory
            for uploaded_file in uploaded_files:
                file_content_bytes = uploaded_file.getvalue()
                file_path_on_disk = os.path.join(core_rag.DOCUMENTS_DIR, uploaded_file.name)
                try:
                    with open(file_path_on_disk, "wb") as f:
                        f.write(file_content_bytes)
                    text = core_rag.process_file_content(uploaded_file.name, file_content_bytes)
                    if text:
                        new_doc_texts_from_upload.append(text)
                    st.success(f"Saved and processed: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error saving/processing {uploaded_file.name}: {e}")
        
        with st.spinner("Rebuilding knowledge base with new documents..."):
            if core_rag.build_faiss_index(new_doc_texts_from_upload):
                st.session_state['index_built'] = True
                st.session_state['document_count'] = len(new_doc_texts_from_upload)
                # Compute hash for the new set of documents to reflect changes
                current_hash_builder = hashlib.md5()
                for text in new_doc_texts_from_upload:
                    current_hash_builder.update(text.encode('utf-8')) # Hash the text content
                st.session_state['current_documents_hash'] = current_hash_builder.hexdigest()
                st.success(f"Knowledge base rebuilt successfully with {st.session_state['document_count']} documents!")
            else:
                st.session_state['index_built'] = False
                st.session_state['document_count'] = 0
                st.error("Failed to rebuild knowledge base. Check logs for errors during file processing or indexing.")
        st.rerun() # Rerun to update state/UI properly
else:
    # If no files are currently selected in the uploader, check the status from disk
    # This block ensures the UI reflects if an index is built from existing files
    # if `uploaded_files` is empty but files exist on disk from a previous run.
    if not st.session_state.get('index_built', False):
        # Only re-check if index isn't already built to avoid excessive re-indexing
        doc_texts_on_disk, disk_hash = get_current_document_texts_from_disk()
        if doc_texts_on_disk and disk_hash != st.session_state.get('current_documents_hash'):
            # Only rebuild if content on disk has changed and index is not built
            with st.spinner("Checking for existing documents and building index..."):
                if core_rag.build_faiss_index(doc_texts_on_disk):
                    st.session_state['index_built'] = True
                    st.session_state['document_count'] = len(doc_texts_on_disk)
                    st.session_state['current_documents_hash'] = disk_hash
                    st.success(f"Knowledge base loaded with {st.session_state['document_count']} documents from disk.")
                else:
                    st.session_state['index_built'] = False
                    st.session_state['document_count'] = 0
                    st.warning("No documents or failed to build from existing files on disk.")
        elif st.session_state.get('document_count', 0) == 0:
            st.info("No files selected and no documents found on disk. Please upload documents.")


# --- Q&A Section ---
st.header("2. Ask a Question")

if st.session_state.get('index_built', False):
    st.success(f"Knowledge base is ready with {st.session_state['document_count']} documents. Ask away!")
    user_query = st.text_area("Your Question:", placeholder="e.g., What are the main applications of AI in healthcare?", height=100)

    if st.button("Get Answer", key="get_answer_button"):
        if user_query:
            with st.spinner("Searching and generating answer..."):
                response = core_rag.get_answer_from_rag(user_query)

            st.subheader("Answer:")
            st.write(response['answer'])

            if response['context']:
                st.subheader("Retrieved Context:")
                st.markdown(f"```\n{response['context']}\n```")
            st.write(f"**Confidence Score:** {response['score']:.2f}")
        else:
            st.warning("Please enter a question.")
else:
    st.info("Upload documents and click 'Process & Rebuild Knowledge Base' to enable Q&A.")

st.markdown("---")
st.caption("Powered by Sentence Transformers, FAISS, and Hugging Face Transformers. UI by Streamlit.")