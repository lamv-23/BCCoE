import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Set page title
st.set_page_config(page_title="BCCoE Chatbot")
st.title("Chat with project PDFs")

# Load OpenAI API key
api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
if not api_key:
    st.error("OpenAI API key is required.")
    st.stop()

# Load all PDFs from a local folder
PDF_DIR = "data"  # your folder of PDFs in the repo
text = ""
for filename in os.listdir(PDF_DIR):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(PDF_DIR, filename)
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() or ""

# Split the text
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
chunks = text_splitter.split_text(text)

# Generate embeddings and store in vector DB
embeddings = OpenAIEmbeddings(api_key=api_key)
vectorstore = Chroma.from_texts(chunks, embedding=embeddings)

# Prompt template to control behaviour
CUSTOM_SYSTEM_PROMPT = '''You are a helpful assistant that answers questions based only on the content of the uploaded PDFs.
Do not guess or make up answers. If you cannot find the answer, say "I'm not sure based on the provided documents."'''

# Handle question input
query = st.text_input("Ask a question about the project:")
if query:
    with st.spinner("Thinking..."):
        try:
            docs = vectorstore.similarity_search(query)
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.0,
                api_key=api_key
            )

            chain = load_qa_chain(
                llm,
                chain_type="stuff",
                chain_type_kwargs={
                    "prompt": {
                        "input_variables": ["context", "question"],
                        "template": f"{CUSTOM_SYSTEM_PROMPT}\n\nContext:\n{{context}}\n\nQuestion:\n{{question}}"
                    }
                }
            )

            response = chain.run(input_documents=docs, question=query)
            st.write("### Answer")
            st.write(response)

        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
