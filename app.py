import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from chromadb.config import Settings as ChromaSettings

# Set Streamlit page settings
st.set_page_config(page_title="BCCoE Chatbot")
st.title("Chat with project PDFs")

# Get API key from environment or Streamlit Secrets
api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
if not api_key:
    st.error("OpenAI API key is required.")
    st.stop()

# Load and extract text from all PDFs in the data folder
PDF_DIR = "data"
text = ""
for filename in os.listdir(PDF_DIR):
    if filename.endswith(".pdf"):
        with open(os.path.join(PDF_DIR, filename), "rb") as f:
            pdf = PdfReader(f)
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"

# Split text into chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
chunks = text_splitter.split_text(text)

# Create embeddings and vector store using in-memory Chroma
embeddings = OpenAIEmbeddings(api_key=api_key)
chroma_settings = ChromaSettings(anonymized_telemetry=False, is_persistent=False)
vectorstore = Chroma.from_texts(chunks, embedding=embeddings, client_settings=chroma_settings)

# Custom system prompt to control chatbot behaviour
CUSTOM_SYSTEM_PROMPT = '''You are a helpful assistant that answers questions based only on the content of the provided PDFs.
Do not guess or make up answers. If you cannot find the answer, say "I'm not sure based on the provided documents."'''

# Ask user for a question
query = st.text_input("Ask a question about the project documents:")
if query:
    with st.spinner("Thinking..."):
        try:
            docs = vectorstore.similarity_search(query)
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0, api_key=api_key)

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
