import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Set page title
st.set_page_config(page_title="BCCoE Chatbot")
st.title("Chat with your PDFs")

# Get OpenAI API key from secrets or environment
api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))

if not api_key:
    st.error("OpenAI API key is required.")
    st.info("Please add your OpenAI API key in Streamlit Cloud's secrets management or set it as an environment variable.")
    st.stop()

# Initialise session state
if "processed_data" not in st.session_state:
    st.session_state.processed_data = False
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Upload PDFs
uploaded_files = st.file_uploader("Upload your PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files and not st.session_state.processed_data:
    with st.spinner("Processing PDFs..."):
        try:
            text = ""
            for pdf in uploaded_files:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
            
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)

            embeddings = OpenAIEmbeddings(api_key=api_key)
            st.session_state.vectorstore = Chroma.from_texts(chunks, embedding=embeddings)
            st.session_state.processed_data = True
            st.success("PDFs processed successfully! You can now ask questions.")
        except Exception as e:
            st.error(f"Error processing PDFs: {str(e)}")

# Ask questions
if st.session_state.processed_data:
    query = st.text_input("Ask a question about your PDFs:")
    if query:
        with st.spinner("Thinking..."):
            try:
                docs = st.session_state.vectorstore.similarity_search(query)
                llm = ChatOpenAI(
                    model_name="gpt-3.5-turbo",
                    temperature=0.0,
                    api_key=api_key
                )
                chain = load_qa_chain(llm, chain_type="stuff")
                response = chain.run(input_documents=docs, question=query)

                st.write("### Answer")
                st.write(response)
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
