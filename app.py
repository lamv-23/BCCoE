import os
import tempfile
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import ssl
import requests
import warnings

# SSL verification workaround
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''  # Disable certificate verification for curl
os.environ['REQUESTS_CA_BUNDLE'] = ''  # Disable certificate verification for requests
os.environ['SSL_CERT_FILE'] = ''  # Disable certificate verification globally

# Suppress warnings about insecure requests
warnings.filterwarnings('ignore', message='Unverified HTTPS request')
requests.packages.urllib3.disable_warnings()

# Set page title
st.set_page_config(page_title="BCCoE Chatbot")
st.title("Chat with your PDFs")

# Initialize session state variables
if "processed_data" not in st.session_state:
    st.session_state.processed_data = False
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

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
            
            # Split text into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            
            # Create embeddings and store in vector database
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2", 
                model_kwargs={"trust_remote_code": True},
                encode_kwargs={"normalize_embeddings": True}
            )
            st.session_state.vectorstore = FAISS.from_texts(chunks, embeddings)
            st.session_state.processed_data = True
            st.success("PDFs processed successfully! You can now ask questions.")
        except Exception as e:
            st.error(f"Error processing PDFs: {str(e)}")
            st.info("Please try again or try with a different PDF.")

# Question answering section
if st.session_state.processed_data:
    query = st.text_input("Ask a question about your PDFs:")
    if query:
        with st.spinner("Thinking..."):
            try:
                # Search for relevant content
                docs = st.session_state.vectorstore.similarity_search(query)
                
                # Use HuggingFace to generate an answer
                llm_pipeline = pipeline("text-generation", model="google/flan-t5-base", max_length=512)
                llm = HuggingFacePipeline(pipeline=llm_pipeline)
                chain = load_qa_chain(llm, chain_type="stuff")
                response = chain.run(input_documents=docs, question=query)
                
                st.write("### Answer")
                st.write(response)
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
