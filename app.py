import os
import nltk
import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

# Download NLTK data on first run
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set page title
st.set_page_config(page_title="BCCoE Chatbot")
st.title("Chat with your PDFs")

# Initialize session state variables
if "processed_data" not in st.session_state:
    st.session_state.processed_data = False
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "vectors" not in st.session_state:
    st.session_state.vectors = None

# PDF upload section
uploaded_files = st.file_uploader("Upload your PDFs", type="pdf", accept_multiple_files=True)

# Process PDFs when uploaded
if uploaded_files and not st.session_state.processed_data:
    with st.spinner("Processing PDFs..."):
        try:
            # Extract text from PDFs
            text = ""
            for pdf in uploaded_files:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            
            # Split text into chunks using NLTK
            sentences = nltk.sent_tokenize(text)
            chunks = []
            current_chunk = ""
            
            # Create chunks of approximately 1000 characters with overlap
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= 1000:
                    current_chunk += sentence + " "
                else:
                    chunks.append(current_chunk)
                    # Keep some overlap for context
                    current_chunk = sentence + " "
            
            # Add the last chunk if not empty
            if current_chunk:
                chunks.append(current_chunk)
            
            # Create TF-IDF vectors for similarity search
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform(chunks)
            
            st.session_state.chunks = chunks
            st.session_state.vectorizer = vectorizer
            st.session_state.vectors = vectors
            st.session_state.processed_data = True
            
            st.success(f"PDFs processed successfully! Split into {len(chunks)} chunks. You can now ask questions.")
        except Exception as e:
            st.error(f"Error processing PDFs: {str(e)}")
            st.info("Please try again or try with a different PDF.")

# Question answering section
if st.session_state.processed_data:
    query = st.text_input("Ask a question about your PDFs:")
    if query:
        with st.spinner("Thinking..."):
            try:
                # Transform query to vector and calculate similarity
                query_vector = st.session_state.vectorizer.transform([query])
                similarities = cosine_similarity(query_vector, st.session_state.vectors)[0]
                
                # Get top 3 most similar chunks
                top_indices = similarities.argsort()[-3:][::-1]
                context = "\n".join([st.session_state.chunks[i] for i in top_indices])
                
                # Use text generation to answer the question
                generator = pipeline('text2text-generation', model='google/flan-t5-small')
                prompt = f"Answer this question based on the context:\nContext: {context}\nQuestion: {query}\nAnswer:"
                response = generator(prompt, max_length=150)[0]['generated_text']
                
                st.write("### Answer")
                st.write(response)
                
                # Optionally show the relevant text
                with st.expander("Relevant text from documents"):
                    st.write(context)
                    
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
                st.error(str(e))
